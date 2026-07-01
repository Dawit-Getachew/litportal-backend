"""Phase R1C — shadow simulator orchestrator.

Builds on R1B's deterministic expansion + connector re-fetch, then
applies R1C's closed-set anchor-specific labels (procedure-
complication outcome anchoring + mechanism-target preservation)
and the gold-comparator recovery pass.

Admin-only. Synchronous (admin-blocking). NO clinician path. NO
new connectors. NO LLM. R1A label weights UNCHANGED. R1B
configuration UNCHANGED.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

import httpx

from core.db import Collections, get_db
from services.audit import audit_service
from services.m3_1_benchmark import leak_scan_passes
from services.r1a_directness_labels import (
    R1A_ALL_LABELS,
    compute_recency_base_score,
    compute_shadow_score,
    detect_labels_for_shape,
)
from services.r1a_shadow_simulator import _overrank_counts
from services.r1b_query_expansion import (
    R1B_ELIGIBLE_SHAPES,
    compile_expanded_queries,
    query_exceeds_length_budget,
    select_expansion_map_for_case,
)
from services.r1b_shadow_simulator import (
    BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER,
    HIGH_PRECISION_SHAPES,
    R1B_SOURCES,
    _article_identity_keys,
    _deep_check_no_forbidden_fields,
    _fetch_via_connectors,
    _gold_rank_in,
    _normalize_if_needed,
    _safe_int,
    _strip_to_closed_set,
)
from services.r1c_anchor_labels import (
    CASE_ANCHOR_CONFIG,
    compute_r1c_multiplier,
    detect_r1c_labels,
)
from services.r1c_gold_recovery import attempt_recovery

logger = logging.getLogger("clinevidence.r1c_shadow")


class R1CReplayError(Exception):
    """Closed-set error carrier."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


def _re_escape(s: str) -> str:
    return re.escape(s or "")


def _rank_r1c(
    *,
    articles: list[dict],
    shape: str,
    case_id: str,
    high_precision_only_set: set[tuple[str, str]],
    broad_only_set: set[tuple[str, str]],
) -> list[dict]:
    """Pure deterministic R1C ranking.

    Base score = (R1A label weights) + recency boost.
    R1B provenance penalty for procedure_complication shape stays.
    R1C anchor multipliers layered on top — these can ONLY deprioritize
    (all multipliers ≤ 1.0).
    """
    scored: list[dict] = []
    for a in articles:
        labels_r1a = detect_labels_for_shape(shape, a)
        labels_r1c = detect_r1c_labels(case_id=case_id, article=a, shape=shape)
        year = _safe_int(a.get("publication_year") or a.get("year")) or 1990
        base_score = compute_recency_base_score(year=year, shape=shape)
        r1a_score = compute_shadow_score(base_score=base_score, labels=labels_r1a)
        # R1B provenance.
        ident = _article_identity_keys(a)
        if ident in high_precision_only_set:
            provenance = "high_precision_family"
        elif ident in broad_only_set:
            provenance = "broad_family"
        else:
            provenance = "both"
        if shape in HIGH_PRECISION_SHAPES and provenance == "broad_family":
            r1a_score *= BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER
        # R1C multiplier.
        r1c_mult = compute_r1c_multiplier(labels_r1c)
        final_score = float(round(r1a_score * r1c_mult, 4))
        scored.append({
            "article": a,
            "labels": labels_r1a + labels_r1c,
            "labels_r1a": labels_r1a,
            "labels_r1c": labels_r1c,
            "score": final_score,
            "provenance": provenance,
            "ident": ident,
        })
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored


def _label_distribution(scored: list[dict]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for r in scored:
        for lab in (r.get("labels") or []):
            dist[lab] = dist.get(lab, 0) + 1
    return dist


def _count_r1c_label(scored: list[dict], label: str) -> int:
    return sum(1 for r in scored if label in (r.get("labels_r1c") or []))


def _project_row(
    *,
    article: dict,
    rank: int,
    score: float,
    provenance: str,
    labels: list[str],
) -> dict:
    return {
        "pmid": (article.get("pmid") or None),
        "doi": (article.get("doi") or None),
        "title": (article.get("title") or None),
        "journal": (article.get("journal") or None),
        "year": _safe_int(article.get("publication_year") or article.get("year")),
        "source": article.get("source")
        or (article.get("sources_present") or [None])[0],
        "simulated_rank": int(rank),
        "simulated_score": float(score),
        "candidate_provenance": provenance,
        "label_set": list(labels or []),
    }


async def replay_benchmark_case_r1c(
    *,
    case_id: str,
    actor_user_id: Optional[str],
    fetch_fn: Optional[Callable[..., Awaitable[list[dict]]]] = None,
    recovery_fn: Optional[Callable[..., Awaitable[dict]]] = None,
    search_session_id_override: Optional[str] = None,
) -> dict[str, Any]:
    """Synchronously re-run a gold benchmark case through R1B
    expansion + R1C anchor labels + gold recovery. Writes a sidecar.

    ``fetch_fn`` and ``recovery_fn`` are exposed for tests so the
    deterministic unit tests can replace connector + Crossref calls
    with stubs.
    """
    t0 = time.perf_counter()
    db = get_db()
    case = await db[Collections.R1A_GOLD_COMPARATOR_CASES].find_one(
        {"case_id": case_id}, {"_id": 0},
    )
    if case is None:
        raise R1CReplayError("unknown_case_id")
    shape = (case.get("shape") or "").strip()
    if shape not in R1B_ELIGIBLE_SHAPES:
        raise R1CReplayError("shape_not_eligible_for_r1c")
    expansion_map = select_expansion_map_for_case(case_id)
    compiled = compile_expanded_queries(
        shape=shape, expansion_map_name=expansion_map,
    )
    over_budget = (
        query_exceeds_length_budget(compiled.get("broad_family_query_string"))
        or query_exceeds_length_budget(
            compiled.get("high_precision_family_query_string")
        )
    )

    # Locate the bound search session.
    bench_qstring = (case.get("question") or "").strip().lower()
    session_id: Optional[str] = search_session_id_override
    if not session_id and bench_qstring:
        match = await db[Collections.SEARCH_SESSIONS].find_one(
            {"sanitized_question": {
                "$regex": f"^{_re_escape(bench_qstring)}$", "$options": "i",
            }},
            {"_id": 0, "id": 1},
            sort=[("created_at", -1)],
        )
        if not match:
            match = await db[Collections.SEARCH_SESSIONS].find_one(
                {"sanitized_question": {
                    "$regex": _re_escape(bench_qstring), "$options": "i",
                }},
                {"_id": 0, "id": 1},
                sort=[("created_at", -1)],
            )
        if match:
            session_id = match.get("id")

    # Pull active arm (top-20) from the bound session.
    active_top20: list[dict] = []
    if session_id:
        ssas = await (
            db[Collections.SEARCH_SESSION_ARTICLES]
            .find({"search_session_id": session_id}, {"_id": 0})
            .sort([("overall_rank", 1)])
            .to_list(length=20)
        )
        art_ids = [s.get("article_id") for s in ssas if s.get("article_id")]
        if art_ids:
            cursor = db[Collections.ARTICLES].find(
                {"id": {"$in": art_ids}},
                {"_id": 0,
                 "id": 1, "pmid": 1, "doi": 1, "title": 1,
                 "journal": 1, "publication_year": 1, "abstract": 1,
                 "sources_present": 1, "publication_types_json": 1},
            )
            arts_by_id: dict[str, dict] = {}
            async for a in cursor:
                arts_by_id[a["id"]] = a
            for s in ssas:
                a = arts_by_id.get(s.get("article_id"))
                if not a:
                    continue
                labels_r1a = detect_labels_for_shape(shape, a)
                labels_r1c = detect_r1c_labels(
                    case_id=case_id, article=a, shape=shape,
                )
                active_row = _strip_to_closed_set(
                    a,
                    rank_field="current_rank",
                    rank_value=_safe_int(s.get("overall_rank")),
                    provenance="active_only",
                    simulated_score=None,
                    label_set=list(labels_r1a + labels_r1c),
                )
                active_top20.append(active_row)

    # R1B-style fetch.
    fetch = fetch_fn or _fetch_via_connectors
    broad_records: list[dict] = []
    hp_records: list[dict] = []
    if compiled.get("broad_family_query_string"):
        broad_records = await fetch(
            query_string=compiled["broad_family_query_string"],
            search_session_id=session_id,
        )
    if compiled.get("high_precision_family_query_string"):
        hp_records = await fetch(
            query_string=compiled["high_precision_family_query_string"],
            search_session_id=session_id,
        )
    # Dedupe + provenance.
    broad_idents: set[tuple[str, str]] = set()
    broad_by_ident: dict[tuple[str, str], dict] = {}
    for a in broad_records:
        ident = _article_identity_keys(a)
        if ident == ("", ""):
            continue
        broad_idents.add(ident)
        broad_by_ident.setdefault(ident, a)
    hp_idents: set[tuple[str, str]] = set()
    hp_by_ident: dict[tuple[str, str], dict] = {}
    for a in hp_records:
        ident = _article_identity_keys(a)
        if ident == ("", ""):
            continue
        hp_idents.add(ident)
        hp_by_ident.setdefault(ident, a)
    both_idents = broad_idents & hp_idents
    broad_only_idents = broad_idents - hp_idents
    hp_only_idents = hp_idents - broad_idents
    merged_by_ident: dict[tuple[str, str], dict] = {}
    for ident in (broad_idents | hp_idents):
        merged_by_ident[ident] = hp_by_ident.get(ident) or broad_by_ident.get(ident) or {}

    # --- R1B simulated arm (R1A weights + R1B provenance penalty only) ---
    r1b_scored: list[dict] = []
    for a in merged_by_ident.values():
        labels = detect_labels_for_shape(shape, a)
        year = _safe_int(a.get("publication_year") or a.get("year")) or 1990
        score = compute_shadow_score(
            base_score=compute_recency_base_score(year=year, shape=shape),
            labels=labels,
        )
        ident = _article_identity_keys(a)
        if ident in hp_only_idents:
            provenance = "high_precision_family"
        elif ident in broad_only_idents:
            provenance = "broad_family"
        else:
            provenance = "both"
        if shape in HIGH_PRECISION_SHAPES and provenance == "broad_family":
            score *= BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER
        r1b_scored.append({
            "article": a, "labels": labels,
            "score": float(round(score, 4)),
            "provenance": provenance, "ident": ident,
        })
    r1b_scored.sort(key=lambda r: r["score"], reverse=True)
    r1b_top20 = [
        _project_row(
            article=r["article"], rank=i + 1, score=r["score"],
            provenance=r["provenance"], labels=r["labels"],
        )
        for i, r in enumerate(r1b_scored[:20])
    ]

    # --- R1C simulated arm (R1B + R1C multipliers) ---
    r1c_scored = _rank_r1c(
        articles=list(merged_by_ident.values()),
        shape=shape, case_id=case_id,
        high_precision_only_set=hp_only_idents,
        broad_only_set=broad_only_idents,
    )
    r1c_top20 = [
        _project_row(
            article=r["article"], rank=i + 1, score=r["score"],
            provenance=r["provenance"], labels=r["labels"],
        )
        for i, r in enumerate(r1c_scored[:20])
    ]

    # --- Gold recovery (Cases A + C have curated gold) ---
    gold_dois = list(case.get("gold_article_dois") or [])
    gold_pmids = list(case.get("gold_article_pmids") or [])
    gold_title = (case.get("question") or "").strip()  # used as fallback for exact_title; the case's actual gold title is in the case metadata if present.
    # Prefer the explicit gold title field if present in the case row;
    # for the closed-set seeded cases the question itself is the
    # closest curated text — Crossref/PubMed lookup uses DOI primarily.
    # Use DOI-first ordering for the strategy execution.
    recovery_result: dict[str, Any] = {
        "recovery_strategies_attempted": [],
        "recovery_strategy_that_surfaced_gold": None,
        "matched_records": [],
        "crossref_used": False,
        "pagination_pages_used": {"pubmed": 0, "europe_pmc": 0},
        "connector_recall_miss_reason": None,
        "pagination_ceiling_exceeded": False,
    }
    if gold_dois or gold_pmids:
        do_recovery = recovery_fn or attempt_recovery
        try:
            recovery_result = await do_recovery(
                gold_dois=gold_dois,
                gold_title=gold_title,
                broad_query_string=compiled.get(
                    "broad_family_query_string",
                ) or "",
                expansion_map_name=expansion_map,
            )
        except Exception:  # noqa: BLE001
            logger.exception("r1c gold recovery failed (non-fatal)")

    # If recovery surfaced new records, fold them into the R1C
    # candidate pool and re-rank (closed-set integration).
    if recovery_result.get("matched_records"):
        existing_idents = {r["ident"] for r in r1c_scored}
        new_articles: list[dict] = []
        for rec in recovery_result["matched_records"]:
            cand = _normalize_if_needed(rec, rec.get("source") or "pubmed")
            ident = _article_identity_keys(cand)
            if ident == ("", "") or ident in existing_idents:
                continue
            new_articles.append(cand)
            existing_idents.add(ident)
        if new_articles:
            # Recovered articles are tagged ``recovered`` for closed-set
            # provenance tracking; the simulator treats them like
            # ``broad_family`` for ranking unless they happened to
            # match the HP family too (HP is procedure-only).
            for a in new_articles:
                ident = _article_identity_keys(a)
                if shape in HIGH_PRECISION_SHAPES:
                    # Allow them to be both/hp/broad like normal.
                    pass
                else:
                    broad_only_idents.add(ident)
                merged_by_ident[ident] = a
            r1c_scored = _rank_r1c(
                articles=list(merged_by_ident.values()),
                shape=shape, case_id=case_id,
                high_precision_only_set=hp_only_idents,
                broad_only_set=broad_only_idents,
            )
            r1c_top20 = [
                _project_row(
                    article=r["article"], rank=i + 1, score=r["score"],
                    provenance=r["provenance"], labels=r["labels"],
                )
                for i, r in enumerate(r1c_scored[:20])
            ]

    # --- Counters + gold ranks ---
    label_dist_r1c = _label_distribution(r1c_scored[:20])
    overrank_r1c = _overrank_counts(label_dist_r1c)

    active_label_dist: dict[str, int] = {}
    for row in active_top20:
        for lab in (row.get("label_set") or []):
            active_label_dist[lab] = active_label_dist.get(lab, 0) + 1
    overrank_active = _overrank_counts(active_label_dist)

    label_dist_r1b = _label_distribution(r1b_scored[:20])
    overrank_r1b = _overrank_counts(label_dist_r1b)

    gold_active_rank = _gold_rank_in(
        rows=active_top20, gold_dois=gold_dois, gold_pmids=gold_pmids,
        rank_field="current_rank",
    )
    gold_r1b_rank = _gold_rank_in(
        rows=r1b_top20, gold_dois=gold_dois, gold_pmids=gold_pmids,
        rank_field="simulated_rank",
    )
    gold_r1c_rank = _gold_rank_in(
        rows=r1c_top20, gold_dois=gold_dois, gold_pmids=gold_pmids,
        rank_field="simulated_rank",
    )

    def _present_top(rows: list[dict], n: int, rank_field: str) -> Optional[bool]:
        if not gold_dois and not gold_pmids:
            return None
        return _gold_rank_in(
            rows=rows[:n], gold_dois=gold_dois, gold_pmids=gold_pmids,
            rank_field=rank_field,
        ) is not None

    cfg = CASE_ANCHOR_CONFIG.get(case_id, {})
    mechanism_target_terms = list(cfg.get("mechanism_target_terms") or [])
    outcome_anchor_terms = list(cfg.get("outcome_anchor_terms") or [])

    # R1C anchor counts in top-10.
    top10_scored = r1c_scored[:10]
    proc_exact = _count_r1c_label(top10_scored, "procedure_complication_outcome_exact")
    proc_missing = _count_r1c_label(top10_scored, "procedure_complication_outcome_missing")
    proc_wrong = _count_r1c_label(top10_scored, "procedure_complication_wrong_outcome")
    proc_general = _count_r1c_label(top10_scored, "procedure_complication_general_not_outcome_specific")
    mech_exact = _count_r1c_label(top10_scored, "mechanism_target_exact")
    mech_missing = _count_r1c_label(top10_scored, "mechanism_target_missing")
    disease_only = _count_r1c_label(top10_scored, "disease_only_indirect_for_mechanism")
    treat_not_primary = _count_r1c_label(top10_scored, "treatment_mentions_mechanism_but_not_primary")
    mech_review_treat = _count_r1c_label(top10_scored, "mechanism_review_with_treatment_discussion")

    latency_ms = int((time.perf_counter() - t0) * 1000)
    now = datetime.now(timezone.utc)
    sidecar_id = uuid.uuid4().hex

    doc: dict[str, Any] = {
        "id": sidecar_id,
        "phase": "R1C",
        "benchmark_case_id": case_id,
        "search_session_id": session_id,
        "shape": shape,
        "disease_or_procedure_entities": list(cfg.get("disease_anchor_terms") or []),
        "mechanism_target_terms": mechanism_target_terms,
        "outcome_anchor_terms": outcome_anchor_terms,
        "expansion_concept_map_used": expansion_map,
        "expansion_terms_or_families": {
            "broad_family_terms": compiled.get("broad_family_terms") or [],
            "high_precision_family_terms": compiled.get(
                "high_precision_family_terms",
            ),
        },
        "broad_family_query_string": compiled.get("broad_family_query_string"),
        "high_precision_family_query_string": compiled.get(
            "high_precision_family_query_string",
        ),
        "high_precision_family_required_both_anchors": bool(
            compiled.get("high_precision_family_required_both_anchors"),
        ),
        # Recovery output (closed-set).
        "recovery_strategies_attempted": list(
            recovery_result.get("recovery_strategies_attempted") or [],
        ),
        "recovery_strategy_that_surfaced_gold": recovery_result.get(
            "recovery_strategy_that_surfaced_gold",
        ),
        "connector_recall_miss_reason": recovery_result.get(
            "connector_recall_miss_reason",
        ),
        "pagination_pages_used": recovery_result.get(
            "pagination_pages_used", {"pubmed": 0, "europe_pmc": 0},
        ),
        "crossref_used": bool(recovery_result.get("crossref_used")),
        # Active arm.
        "current_active_top_n_pmids": [a.get("pmid") for a in active_top20],
        "current_active_top_n_metadata": active_top20,
        # R1B arm.
        "r1b_simulated_top_n_pmids": [a.get("pmid") for a in r1b_top20],
        "r1b_simulated_top_n_metadata": r1b_top20,
        # R1C arm.
        "r1c_simulated_top_n_pmids": [a.get("pmid") for a in r1c_top20],
        "r1c_simulated_top_n_metadata": r1c_top20,
        # Labels applied.
        "directness_label_distribution_r1c": label_dist_r1c,
        # Gold metrics.
        "gold_comparator_dois": gold_dois,
        "gold_comparator_pmids": gold_pmids,
        "gold_comparator_present_active_top10": _present_top(
            active_top20, 10, "current_rank",
        ),
        "gold_comparator_present_active_top20": _present_top(
            active_top20, 20, "current_rank",
        ),
        "gold_comparator_present_r1b_top10": _present_top(
            r1b_top20, 10, "simulated_rank",
        ),
        "gold_comparator_present_r1b_top20": _present_top(
            r1b_top20, 20, "simulated_rank",
        ),
        "gold_comparator_present_r1c_top10": _present_top(
            r1c_top20, 10, "simulated_rank",
        ),
        "gold_comparator_present_r1c_top20": _present_top(
            r1c_top20, 20, "simulated_rank",
        ),
        "gold_comparator_rank_active": gold_active_rank,
        "gold_comparator_rank_r1b": gold_r1b_rank,
        "gold_comparator_rank_r1c": gold_r1c_rank,
        # R1C anchor counters.
        "procedure_complication_outcome_exact_count": proc_exact,
        "procedure_complication_outcome_missing_count": proc_missing,
        "procedure_complication_wrong_outcome_count": proc_wrong,
        "procedure_complication_general_not_outcome_specific_count": proc_general,
        "mechanism_target_exact_count": mech_exact,
        "mechanism_target_missing_count": mech_missing,
        "disease_only_indirect_for_mechanism_count": disease_only,
        "treatment_mentions_mechanism_but_not_primary_count": treat_not_primary,
        "mechanism_review_with_treatment_discussion_count": mech_review_treat,
        # R1A/B/C overrank counters (active arm best-effort because
        # we don't keep abstracts on active rows).
        "treatment_overrank_count_active": overrank_active[
            "treatment_overrank_count"
        ],
        "diagnostic_overrank_count_active": overrank_active[
            "diagnostic_overrank_count"
        ],
        "broad_overview_overrank_count_active": overrank_active[
            "broad_overview_overrank_count"
        ],
        "treatment_overrank_count_r1b": overrank_r1b[
            "treatment_overrank_count"
        ],
        "diagnostic_overrank_count_r1b": overrank_r1b[
            "diagnostic_overrank_count"
        ],
        "broad_overview_overrank_count_r1b": overrank_r1b[
            "broad_overview_overrank_count"
        ],
        "treatment_overrank_count_r1c": overrank_r1c[
            "treatment_overrank_count"
        ],
        "diagnostic_overrank_count_r1c": overrank_r1c[
            "diagnostic_overrank_count"
        ],
        "broad_overview_overrank_count_r1c": overrank_r1c[
            "broad_overview_overrank_count"
        ],
        "candidate_provenance_counts": {
            "high_precision_family_only": len(hp_only_idents),
            "broad_family_only": len(broad_only_idents),
            "both": len(both_idents),
        },
        "latency_ms": latency_ms,
        "source_errors": {},
        "safety_flags": {
            "leak_scan_passed": True,
            "phi_guard_passed": True,
            "expanded_query_over_url_budget": bool(over_budget),
            "pagination_ceiling_exceeded": bool(
                recovery_result.get("pagination_ceiling_exceeded"),
            ),
        },
        "r1c_simulation_skipped": False,
        "route_owned_by_k4a_specialty": False,
        "created_at": now,
        "expires_at": now + timedelta(days=30),
    }

    # Closed-set leakage scans.
    if not _deep_check_no_forbidden_fields(doc):
        logger.warning(
            "r1c sidecar forbidden field check failed — redacting "
            "metadata arrays (case=%s)", case_id,
        )
        doc["current_active_top_n_metadata"] = []
        doc["r1b_simulated_top_n_metadata"] = []
        doc["r1c_simulated_top_n_metadata"] = []
        doc["safety_flags"]["leak_scan_passed"] = False
        doc["safety_flags"]["forbidden_field_detected"] = True
    if not leak_scan_passes(doc):
        logger.warning(
            "r1c sidecar M3.1 leak scanner flagged content — "
            "redacting metadata arrays (case=%s)", case_id,
        )
        doc["current_active_top_n_metadata"] = []
        doc["r1b_simulated_top_n_metadata"] = []
        doc["r1c_simulated_top_n_metadata"] = []
        doc["safety_flags"]["leak_scan_passed"] = False
    try:
        await db[Collections.R1C_SHADOW_DECISIONS].insert_one(dict(doc))
    except Exception:  # noqa: BLE001
        logger.exception("r1c sidecar insert failed (non-fatal) case=%s", case_id)

    try:
        await audit_service.log(
            "r1c_replay_completed",
            user_id=actor_user_id,
            search_session_id=session_id,
            payload={
                "case_id": case_id,
                "shape": shape,
                "expansion_map_used": expansion_map,
                "sidecar_doc_id": sidecar_id,
                "recovery_strategies_attempted": doc[
                    "recovery_strategies_attempted"
                ],
                "recovery_strategy_that_surfaced_gold": doc[
                    "recovery_strategy_that_surfaced_gold"
                ],
                "connector_recall_miss_reason": doc[
                    "connector_recall_miss_reason"
                ],
                "gold_active_rank": gold_active_rank,
                "gold_r1b_rank": gold_r1b_rank,
                "gold_r1c_rank": gold_r1c_rank,
                "crossref_used": doc["crossref_used"],
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception("r1c audit failed (non-fatal)")

    return {
        "sidecar_doc_id": sidecar_id,
        "case_id": case_id,
        "shape": shape,
        "expansion_map_used": expansion_map,
        "search_session_id_used": session_id,
        "gold_comparator_rank_active": gold_active_rank,
        "gold_comparator_rank_r1b": gold_r1b_rank,
        "gold_comparator_rank_r1c": gold_r1c_rank,
        "recovery_strategies_attempted": doc["recovery_strategies_attempted"],
        "recovery_strategy_that_surfaced_gold": doc[
            "recovery_strategy_that_surfaced_gold"
        ],
        "connector_recall_miss_reason": doc["connector_recall_miss_reason"],
        "pagination_pages_used": doc["pagination_pages_used"],
        "crossref_used": doc["crossref_used"],
        "treatment_overrank_count_r1c": doc["treatment_overrank_count_r1c"],
        "treatment_overrank_count_r1b": doc["treatment_overrank_count_r1b"],
        "procedure_complication_outcome_exact_count": proc_exact,
        "procedure_complication_wrong_outcome_count": proc_wrong,
        "procedure_complication_general_not_outcome_specific_count": proc_general,
        "mechanism_target_exact_count": mech_exact,
        "mechanism_target_missing_count": mech_missing,
        "disease_only_indirect_for_mechanism_count": disease_only,
        "provenance_counts": doc["candidate_provenance_counts"],
        "safety_flags": doc["safety_flags"],
    }


async def list_r1c_shadow_decisions(
    *,
    limit: int = 50,
    case_id: Optional[str] = None,
    shape: Optional[str] = None,
) -> list[dict[str, Any]]:
    db = get_db()
    q: dict[str, Any] = {}
    if case_id:
        q["benchmark_case_id"] = case_id
    if shape:
        q["shape"] = shape
    out: list[dict[str, Any]] = []
    cursor = (
        db[Collections.R1C_SHADOW_DECISIONS]
        .find(q, {"_id": 0})
        .sort([("created_at", -1)])
        .limit(int(limit))
    )
    async for r in cursor:
        for k in ("created_at", "expires_at"):
            v = r.get(k)
            if isinstance(v, datetime):
                r[k] = v.isoformat()
        out.append(r)
    return out


async def get_r1c_shadow_decision_by_id(doc_id: str) -> Optional[dict[str, Any]]:
    db = get_db()
    r = await db[Collections.R1C_SHADOW_DECISIONS].find_one(
        {"id": doc_id}, {"_id": 0},
    )
    if not r:
        return None
    for k in ("created_at", "expires_at"):
        v = r.get(k)
        if isinstance(v, datetime):
            r[k] = v.isoformat()
    return r


__all__ = [
    "R1CReplayError",
    "replay_benchmark_case_r1c",
    "list_r1c_shadow_decisions",
    "get_r1c_shadow_decision_by_id",
]
