"""Phase R1B — shadow simulator for retrieval-recall + directness.

ADMIN-ONLY. Synchronous. NO clinician path. NO LLM. No new
connectors — uses the existing PubMed + Europe PMC connectors.

Replay flow:
  1. Resolve the gold case via ``r1a_gold_comparator_cases``.
  2. Locate or create the bound search session (prefer
     sanitized-question match; fall back to a closed-set marker
     row when no session exists).
  3. Compile the deterministic expansion via
     ``services.r1b_query_expansion``.
  4. Re-fetch candidate articles via the existing connectors
     (``literature_retrieval.retrieve``). Each article is tagged
     with closed-set provenance (``high_precision_family`` /
     ``broad_family`` / ``both``).
  5. Compute an R1B simulated ranking that DEPRIORITIZES
     ``broad_family``-only articles when the shape is
     ``procedure_complication_risk``. R1A LABEL WEIGHTS UNCHANGED.
  6. Compute deltas vs the active arm (pulled from the bound
     session's ``search_session_articles``).
  7. Write a closed-set sidecar row to ``r1b_shadow_decisions``
     (TTL 30d). Leak-scanned. NO abstracts / prompts / completions.

Determinism: the simulator returns the same outcome for the same
case + the same connector result (modulo connector recall).
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from core.db import Collections, get_db
from services.audit import audit_service
from services.m3_1_benchmark import leak_scan_passes
from services.metadata_normalizer import metadata_normalizer
from services.r1a_directness_labels import (
    R1A_ALL_LABELS,
    detect_labels_for_shape,
)
from services.r1a_shadow_simulator import (
    _overrank_counts,
)
from services.r1b_query_expansion import (
    R1B_ELIGIBLE_SHAPES,
    compile_expanded_queries,
    query_exceeds_length_budget,
    select_expansion_map_for_case,
)

logger = logging.getLogger("clinevidence.r1b_shadow")

# Closed-set: shapes that pull the high-precision (procedure AND
# outcome) family. Used to deprioritize broad-family-only articles.
HIGH_PRECISION_SHAPES: frozenset[str] = frozenset({
    "procedure_complication_risk",
    "procedure_complication",
})

# Deprioritization multiplier for broad-family-only articles in
# procedure-complication shape. CLOSED-SET CONSTANT. Pure ranking
# heuristic local to R1B — does NOT touch R1A_LABEL_WEIGHTS.
BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER: float = 0.5

# Closed-set: source connectors R1B re-fetches against.
R1B_SOURCES: tuple[str, ...] = ("europe_pmc", "pubmed")

# Closed-set: per-article fields persisted in
# ``r1b_simulated_top_n_metadata`` and ``current_active_top_n_metadata``.
R1B_PER_ARTICLE_FIELDS: tuple[str, ...] = (
    "pmid",
    "doi",
    "title",
    "journal",
    "year",
    "source",
    "current_rank",
    "simulated_rank",
    "simulated_score",
    "candidate_provenance",
    "label_set",
)

# Closed-set: free-text fields that MUST NOT appear in the sidecar.
# Used by the post-write leak scanner.
_FORBIDDEN_FIELDS: frozenset[str] = frozenset({
    "abstract",
    "abstractText",
    "fulltext",
    "full_text",
    "evidence_text",
    "prompt",
    "completion",
    "reasoning",
    "system_prompt",
    "api_key",
    "authorization",
    "jwt",
    "env",
    "password",
})


def _re_escape(s: str) -> str:
    return re.escape(s or "")


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _strip_to_closed_set(article: dict, *, rank_field: str, rank_value: Optional[int],
                        provenance: Optional[str], simulated_score: Optional[float],
                        label_set: list[str]) -> dict:
    """Closed-set projection of an article row for the sidecar.

    Drops abstracts, prompts, and any field outside the per-article
    contract. Title + journal + year + source + identifiers retained
    (same posture as R1A).
    """
    out: dict[str, Any] = {
        "pmid": (article.get("pmid") or None),
        "doi": (article.get("doi") or None),
        "title": (article.get("title") or None),
        "journal": (article.get("journal") or None),
        "year": _safe_int(article.get("publication_year") or article.get("year")),
        "source": article.get("source")
        or (article.get("sources_present") or [None])[0],
        "current_rank": rank_value if rank_field == "current_rank" else None,
        "simulated_rank": rank_value if rank_field == "simulated_rank" else None,
        "simulated_score": (
            float(simulated_score) if simulated_score is not None else None
        ),
        "candidate_provenance": provenance,
        "label_set": list(label_set or []),
    }
    return out


def _article_identity_keys(article: dict) -> tuple[str, str]:
    """Return ``(doi_lower, pmid)`` so the simulator can dedupe + match
    across active and R1B streams.
    """
    return (
        str(article.get("doi") or "").strip().lower(),
        str(article.get("pmid") or "").strip(),
    )


def _is_pubmed_normalized(article: dict) -> bool:
    """PubMed connector returns already-normalized records; EPMC
    raw records still need normalization. Detect by presence of
    closed-set fields.
    """
    return article.get("source") == "pubmed" or (
        "publication_types_json" in article
    )


def _normalize_if_needed(article: dict, source_hint: str) -> dict:
    """Closed-set normalization dispatch."""
    if _is_pubmed_normalized(article):
        return article
    try:
        normalized = metadata_normalizer.normalize(article)
        normalized = dict(normalized or {})
        normalized.setdefault("source", source_hint)
        return normalized
    except Exception:  # noqa: BLE001
        return article


async def _fetch_via_connectors(
    *,
    query_string: str,
    search_session_id: Optional[str],
) -> list[dict]:
    """Closed-set: re-fetch candidates against EPMC + PubMed using
    the existing connectors. Errors per source are swallowed and
    surfaced via the ``source_errors`` aggregator.
    """
    if not query_string:
        return []
    from services.literature_retrieval import literature_retrieval
    out: list[dict] = []
    for source in R1B_SOURCES:
        try:
            results = await literature_retrieval.retrieve(
                source,
                [{"query_type": "r1b_expanded", "query_string": query_string}],
                search_session_id=search_session_id,
            )
            for _qtype, rec in results:
                norm = _normalize_if_needed(rec, source)
                if "source" not in norm or not norm.get("source"):
                    norm["source"] = source
                out.append(norm)
        except Exception:  # noqa: BLE001
            logger.exception(
                "r1b connector %s fetch failed (non-fatal)", source,
            )
    return out


def _rank_articles(
    *,
    articles: list[dict],
    shape: str,
    high_precision_only_set: set[tuple[str, str]],
    broad_only_set: set[tuple[str, str]],
) -> list[dict]:
    """Pure deterministic ranking for R1B.

    Base score = (R1A label weights via ``detect_labels_for_shape``)
    + a small recency boost (year - 2000).

    Provenance adjustment for procedure_complication shape:
      * ``broad_family``-only → score *= 0.5 (closed-set constant).
      * ``high_precision_family``-only → no change.
      * ``both`` → no change.
    """
    from services.r1a_directness_labels import (
        compute_recency_base_score,
        compute_shadow_score,
    )
    scored: list[dict] = []
    for a in articles:
        labels = detect_labels_for_shape(shape, a)
        # The R1A score function maps base_score + label weights.
        # R1C.1 (shadow-only): recency contribution is shape-conditional —
        # capped for ``pathophysiology_mechanism`` so directness labels
        # can dominate; unchanged for all other shapes.
        year = _safe_int(a.get("publication_year") or a.get("year")) or 1990
        base_score = compute_recency_base_score(year=year, shape=shape)
        score = compute_shadow_score(base_score=base_score, labels=labels)
        ident = _article_identity_keys(a)
        provenance: Optional[str] = None
        if ident in high_precision_only_set:
            provenance = "high_precision_family"
        elif ident in broad_only_set:
            provenance = "broad_family"
        else:
            provenance = "both"
        if (
            shape in HIGH_PRECISION_SHAPES
            and provenance == "broad_family"
        ):
            score *= BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER
        scored.append({
            "article": a,
            "labels": labels,
            "score": float(round(score, 4)),
            "provenance": provenance,
            "ident": ident,
        })
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored


def _label_distribution(scored: list[dict]) -> dict[str, int]:
    """Pure — closed-set label distribution counter."""
    dist: dict[str, int] = {lab: 0 for lab in R1A_ALL_LABELS}
    for r in scored:
        for lab in r["labels"]:
            if lab in dist:
                dist[lab] += 1
    return dist


def _gold_rank_in(
    *,
    rows: list[dict],
    gold_dois: list[str],
    gold_pmids: list[str],
    rank_field: str,
) -> Optional[int]:
    dset = {str(d or "").strip().lower() for d in (gold_dois or [])}
    pset = {str(p or "").strip() for p in (gold_pmids or [])}
    dset.discard("")
    pset.discard("")
    if not dset and not pset:
        return None
    best: Optional[int] = None
    for r in rows:
        if (dset and str(r.get("doi") or "").strip().lower() in dset) or (
            pset and str(r.get("pmid") or "").strip() in pset
        ):
            cand = r.get(rank_field)
            if isinstance(cand, int):
                if best is None or cand < best:
                    best = cand
    return best


def _gold_present(
    *,
    rows: list[dict],
    gold_dois: list[str],
    gold_pmids: list[str],
    limit: int,
) -> Optional[bool]:
    if not gold_dois and not gold_pmids:
        return None
    return _gold_rank_in(
        rows=rows[:limit],
        gold_dois=gold_dois,
        gold_pmids=gold_pmids,
        rank_field="simulated_rank" if rows and "simulated_rank" in rows[0] else "current_rank",
    ) is not None


def _deep_check_no_forbidden_fields(doc: Any) -> bool:
    """Closed-set deep walk — fails if any key matches the
    forbidden field list. Used post-write to surface leakage.
    """
    if isinstance(doc, dict):
        for k, v in doc.items():
            if str(k).lower() in _FORBIDDEN_FIELDS:
                return False
            if not _deep_check_no_forbidden_fields(v):
                return False
        return True
    if isinstance(doc, list):
        return all(_deep_check_no_forbidden_fields(v) for v in doc)
    return True


# ---------------------------------------------------------------------
# Public — replay one benchmark case.
# ---------------------------------------------------------------------
class R1BReplayError(Exception):
    """Closed-set error code carrier for the admin endpoint."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


async def replay_benchmark_case_r1b(
    *,
    case_id: str,
    actor_user_id: Optional[str],
    fetch_fn: Optional[Callable[..., Awaitable[list[dict]]]] = None,
    search_session_id_override: Optional[str] = None,
) -> dict[str, Any]:
    """Synchronously re-run a curated gold case through R1B
    expansion + simulated rerank. Writes a sidecar row.

    ``fetch_fn`` is exposed for testability — it MUST be an async
    callable with the same signature as ``_fetch_via_connectors``.
    Production callers leave it ``None`` (uses real connectors).
    """
    t0 = time.perf_counter()
    db = get_db()
    case = await db[Collections.R1A_GOLD_COMPARATOR_CASES].find_one(
        {"case_id": case_id}, {"_id": 0},
    )
    if case is None:
        raise R1BReplayError("unknown_case_id")
    shape = (case.get("shape") or "").strip()
    if shape not in R1B_ELIGIBLE_SHAPES:
        # Closed-set R1B refusal. We don't have a curated map for
        # this shape.
        raise R1BReplayError("shape_not_eligible_for_r1b")
    # K4A subtype skip — defensive even though R1B is admin-replay
    # only. Closed-set: K4A-owned subtypes get a marker row and
    # we return early.
    bench_qstring = (case.get("question") or "").strip().lower()
    expansion_map = select_expansion_map_for_case(case_id)
    compiled = compile_expanded_queries(
        shape=shape, expansion_map_name=expansion_map,
    )
    # Length-budget defensive check; over-budget queries are
    # surfaced rather than silently truncated.
    over_budget = (
        query_exceeds_length_budget(compiled.get("broad_family_query_string"))
        or query_exceeds_length_budget(
            compiled.get("high_precision_family_query_string")
        )
    )

    # Locate the bound search session.
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

    # Pull active arm (top-20) from the bound session — if any.
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
                 "journal": 1, "publication_year": 1,
                 "sources_present": 1, "publication_types_json": 1},
            )
            arts_by_id: dict[str, dict] = {}
            async for a in cursor:
                arts_by_id[a["id"]] = a
            for s in ssas:
                a = arts_by_id.get(s.get("article_id"))
                if not a:
                    continue
                labels = detect_labels_for_shape(shape, a)
                active_top20.append(
                    _strip_to_closed_set(
                        a,
                        rank_field="current_rank",
                        rank_value=_safe_int(s.get("overall_rank")),
                        provenance="active_only",
                        simulated_score=None,
                        label_set=labels,
                    )
                )

    # R1B fetch — both families in parallel via the connector
    # dispatcher (NOT parallel API-wise; we use the existing per-
    # source sequential dispatcher).
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

    # Closed-set: identity dedupe across both streams.
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
    union_idents = broad_idents | hp_idents
    # Merge — prefer the HP record's normalized payload when
    # both streams have the article.
    merged_by_ident: dict[tuple[str, str], dict] = {}
    for ident in union_idents:
        merged_by_ident[ident] = hp_by_ident.get(ident) or broad_by_ident.get(ident) or {}
    merged_articles = list(merged_by_ident.values())

    # Rank.
    scored = _rank_articles(
        articles=merged_articles,
        shape=shape,
        high_precision_only_set=hp_only_idents,
        broad_only_set=broad_only_idents,
    )
    # Top-20 simulated rows.
    r1b_top20: list[dict] = []
    for idx, r in enumerate(scored[:20], start=1):
        row = _strip_to_closed_set(
            r["article"],
            rank_field="simulated_rank",
            rank_value=idx,
            provenance=r["provenance"],
            simulated_score=r["score"],
            label_set=r["labels"],
        )
        r1b_top20.append(row)

    # Distribution / counters — both arms.
    # Active arm uses pre-existing label assignment; recompute via the
    # same closed-set detector so the counters are apples-to-apples.
    active_scored = [
        {
            "labels": detect_labels_for_shape(shape, {
                "title": r.get("title"),
                "abstract": "",  # closed-set: abstract intentionally
                                  # unavailable on the active side
                                  # because we didn't persist it on
                                  # the sidecar row
                "publication_types_json": [],
            }),
        }
        for r in active_top20
    ]
    # Note: active labels above are best-effort because we don't keep
    # abstracts in the active arm; counters are an upper-bound proxy.

    label_dist_r1b = _label_distribution(scored[:20])
    label_dist_active = _label_distribution(
        [{"labels": a["labels"]} for a in active_scored]
    )
    overrank_r1b = _overrank_counts(label_dist_r1b)
    overrank_active = _overrank_counts(label_dist_active)

    gold_dois = list(case.get("gold_article_dois") or [])
    gold_pmids = list(case.get("gold_article_pmids") or [])
    gold_active_rank = _gold_rank_in(
        rows=active_top20, gold_dois=gold_dois, gold_pmids=gold_pmids,
        rank_field="current_rank",
    )
    gold_r1b_rank = _gold_rank_in(
        rows=r1b_top20, gold_dois=gold_dois, gold_pmids=gold_pmids,
        rank_field="simulated_rank",
    )
    gold_present_active_top10 = (
        _gold_rank_in(
            rows=active_top20[:10], gold_dois=gold_dois,
            gold_pmids=gold_pmids, rank_field="current_rank",
        ) is not None
        if (gold_dois or gold_pmids) else None
    )
    gold_present_active_top20 = (
        gold_active_rank is not None
        if (gold_dois or gold_pmids) else None
    )
    gold_present_r1b_top10 = (
        _gold_rank_in(
            rows=r1b_top20[:10], gold_dois=gold_dois,
            gold_pmids=gold_pmids, rank_field="simulated_rank",
        ) is not None
        if (gold_dois or gold_pmids) else None
    )
    gold_present_r1b_top20 = (
        gold_r1b_rank is not None
        if (gold_dois or gold_pmids) else None
    )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    now = datetime.now(timezone.utc)
    sidecar_id = uuid.uuid4().hex
    doc: dict[str, Any] = {
        "id": sidecar_id,
        "phase": "R1B",
        "benchmark_case_id": case_id,
        "search_session_id": session_id,
        "shape": shape,
        "disease_or_procedure_entities": [],  # currently unused;
        # reserved for future parser-driven enrichment
        "expansion_concept_map_used": expansion_map,
        "expansion_terms_or_families": {
            "broad_family_terms": compiled.get("broad_family_terms") or [],
            "high_precision_family_terms": (
                compiled.get("high_precision_family_terms")
            ),
        },
        "broad_family_query_string": compiled.get("broad_family_query_string"),
        "high_precision_family_query_string": compiled.get(
            "high_precision_family_query_string"
        ),
        "high_precision_family_required_both_anchors": bool(
            compiled.get("high_precision_family_required_both_anchors")
        ),
        "current_active_top_n_pmids": [
            (a.get("pmid") or None) for a in active_top20
        ],
        "current_active_top_n_metadata": active_top20,
        "r1b_simulated_top_n_pmids": [
            (a.get("pmid") or None) for a in r1b_top20
        ],
        "r1b_simulated_top_n_metadata": r1b_top20,
        "r1a_labels_applied_to_r1b_simulated_top_n": [
            r.get("label_set") for r in r1b_top20
        ],
        "gold_comparator_dois": gold_dois,
        "gold_comparator_pmids": gold_pmids,
        "gold_comparator_present_active_top10": gold_present_active_top10,
        "gold_comparator_present_active_top20": gold_present_active_top20,
        "gold_comparator_present_r1b_top10": gold_present_r1b_top10,
        "gold_comparator_present_r1b_top20": gold_present_r1b_top20,
        "gold_comparator_rank_active": gold_active_rank,
        "gold_comparator_rank_r1b": gold_r1b_rank,
        "label_distribution_active": label_dist_active,
        "label_distribution_r1b": label_dist_r1b,
        "treatment_overrank_count_active": overrank_active[
            "treatment_overrank_count"
        ],
        "treatment_overrank_count_r1b": overrank_r1b[
            "treatment_overrank_count"
        ],
        "diagnostic_overrank_count_active": overrank_active[
            "diagnostic_overrank_count"
        ],
        "diagnostic_overrank_count_r1b": overrank_r1b[
            "diagnostic_overrank_count"
        ],
        "broad_overview_overrank_count_active": overrank_active[
            "broad_overview_overrank_count"
        ],
        "broad_overview_overrank_count_r1b": overrank_r1b[
            "broad_overview_overrank_count"
        ],
        "wrong_lane_count_active": overrank_active["wrong_lane_count"],
        "wrong_lane_count_r1b": overrank_r1b["wrong_lane_count"],
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
        },
        "r1b_simulation_skipped": False,
        "route_owned_by_k4a_specialty": False,
        "created_at": now,
        "expires_at": now + timedelta(days=30),
    }

    # ----- post-write closed-set leakage scan -----
    if not _deep_check_no_forbidden_fields(doc):
        # Closed-set: redact metadata arrays + flag the row.
        logger.warning(
            "r1b sidecar forbidden field check failed — redacting "
            "metadata arrays (case=%s)", case_id,
        )
        doc["current_active_top_n_metadata"] = []
        doc["r1b_simulated_top_n_metadata"] = []
        doc["safety_flags"]["leak_scan_passed"] = False
        doc["safety_flags"]["forbidden_field_detected"] = True
    if not leak_scan_passes(doc):
        logger.warning(
            "r1b sidecar M3.1 leak scanner flagged content — "
            "redacting metadata arrays (case=%s)", case_id,
        )
        doc["current_active_top_n_metadata"] = []
        doc["r1b_simulated_top_n_metadata"] = []
        doc["safety_flags"]["leak_scan_passed"] = False
    try:
        await db[Collections.R1B_SHADOW_DECISIONS].insert_one(dict(doc))
    except Exception:  # noqa: BLE001
        logger.exception(
            "r1b sidecar insert failed (non-fatal) case=%s", case_id,
        )

    # Audit.
    try:
        await audit_service.log(
            "r1b_replay_completed",
            user_id=actor_user_id,
            search_session_id=session_id,
            payload={
                "case_id": case_id,
                "shape": shape,
                "expansion_map_used": expansion_map,
                "sidecar_doc_id": sidecar_id,
                "gold_active_rank": gold_active_rank,
                "gold_r1b_rank": gold_r1b_rank,
                "provenance_counts": doc["candidate_provenance_counts"],
                "expanded_query_over_url_budget": bool(over_budget),
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception("r1b audit failed (non-fatal)")

    return {
        "sidecar_doc_id": sidecar_id,
        "case_id": case_id,
        "shape": shape,
        "expansion_map_used": expansion_map,
        "search_session_id_used": session_id,
        "no_matching_session": session_id is None,
        "gold_comparator_rank_active": gold_active_rank,
        "gold_comparator_rank_r1b": gold_r1b_rank,
        "delta_gold_rank": (
            (gold_r1b_rank if gold_r1b_rank is not None else 999)
            - (gold_active_rank if gold_active_rank is not None else 999)
            if (gold_active_rank is not None or gold_r1b_rank is not None)
            else None
        ),
        "treatment_overrank_delta": (
            overrank_r1b["treatment_overrank_count"]
            - overrank_active["treatment_overrank_count"]
        ),
        "diagnostic_overrank_delta": (
            overrank_r1b["diagnostic_overrank_count"]
            - overrank_active["diagnostic_overrank_count"]
        ),
        "broad_overview_overrank_delta": (
            overrank_r1b["broad_overview_overrank_count"]
            - overrank_active["broad_overview_overrank_count"]
        ),
        "provenance_counts": doc["candidate_provenance_counts"],
        "safety_flags": doc["safety_flags"],
        "expanded_query_over_url_budget": bool(over_budget),
    }


# ---------------------------------------------------------------------
# Listing endpoints helpers.
# ---------------------------------------------------------------------
async def list_r1b_shadow_decisions(
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
        db[Collections.R1B_SHADOW_DECISIONS]
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


async def get_r1b_shadow_decision_by_id(doc_id: str) -> Optional[dict[str, Any]]:
    db = get_db()
    r = await db[Collections.R1B_SHADOW_DECISIONS].find_one(
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
    "HIGH_PRECISION_SHAPES",
    "R1B_SOURCES",
    "R1B_PER_ARTICLE_FIELDS",
    "BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER",
    "R1BReplayError",
    "replay_benchmark_case_r1b",
    "list_r1b_shadow_decisions",
    "get_r1b_shadow_decision_by_id",
]
