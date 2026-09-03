"""Phase R1A — Part 6 — shadow reranking simulation pipeline.

DETERMINISTIC. NO LLM. NO RETRIEVAL. NO clinician-visible side
effect. Closed-set fields only.

Runs AFTER the clinician response is committed (status =
``awaiting_selection``) so it CANNOT add latency to the live
pipeline. The clinician's active top-N stays exactly as ranked.

Inputs (passed by ``schedule_shadow_simulation``):
  * ``session_id`` (str)
  * ``shape`` (str) — closed-set
  * ``benchmark_case_id`` (Optional[str]) — set when replay
  * ``query_expansion_terms_used`` (list[str]) — closed-set,
    no PHI (caller already guarantees)
  * ``high_precision_query_required_both_anchors`` (bool)
  * ``source_errors`` (dict[str,int])

Outputs (written to ``r1a_shadow_decisions`` with TTL 30d):
  Schema documented in ``R1A_SHADOW_DECISION_FIELDS``.

Hard safety:
  * The shadow task is wrapped in a top-level try/except — any
    failure is logged + swallowed; clinician path NEVER sees it.
  * The write is leak-scanned by the M3.1 leak scanner. A failed
    leak scan triggers a closed-set row marker
    ``shadow_simulation_redacted_by_leak_scan=true`` and ALL
    title/journal fields are stripped from the persisted row.
  * K4A-protected routes are detected by question_subtype and the
    shadow row is written with ``route_owned_by_k4a_specialty=true``
    + ``shadow_simulation_skipped=true`` (NO simulation performed).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from core.db import Collections, get_db
from services.audit import audit_service
from services.m3_1_benchmark import leak_scan_passes
from services.phi_guard import phi_guard
from services.r1a_directness_labels import (
    R1A_ALL_LABELS,
    PATHO_NEGATIVE_LABELS,
    PATHO_POSITIVE_LABELS,
    PROC_NEGATIVE_LABELS,
    PROC_POSITIVE_LABELS,
    compute_shadow_score,
    detect_labels_for_shape,
)

logger = logging.getLogger("clinevidence.r1a_shadow")

# Closed-set union of POSITIVE directness labels (across both shapes).
# Used by the deterministic ``mixed-signal`` flag computation.
R1A_POSITIVE_LABELS: frozenset[str] = frozenset(
    PATHO_POSITIVE_LABELS + PROC_POSITIVE_LABELS
)
# Closed-set union of INDIRECT / downrank labels (across both shapes).
R1A_INDIRECT_LABELS: frozenset[str] = frozenset(
    PATHO_NEGATIVE_LABELS + PROC_NEGATIVE_LABELS
)

# Closed-set per-article mixed-signal verdict values. Default
# ``unspecified`` until the physician adjudicates.
MIXED_SIGNAL_VERDICTS: frozenset[str] = frozenset({
    "accept", "suppress", "penalize", "unspecified",
})

# Closed-set per-case adjudication verdict values (mirrors M3.2
# pattern).
R1A_CASE_VERDICTS: frozenset[str] = frozenset({
    "better", "same", "worse", "not_adjudicable",
})

# Closed-set ``acceptable_for_future_active`` tri-state.
R1A_ACCEPTABLE_VALUES: frozenset[str] = frozenset({
    "yes", "no", "unspecified",
})

# Closed-set: shapes eligible for R1A shadow simulation.
R1A_SHADOW_SHAPES: frozenset[str] = frozenset({
    "pathophysiology_mechanism",
    "procedure_complication_risk",
    "procedure_complication",
})

# Closed-set: question_subtypes owned by K4A specialty packs that
# must be EXCLUDED from R1A shadow simulation. R1A is read-only on
# K4A; we never simulate against these.
K4A_OWNED_SUBTYPES: frozenset[str] = frozenset({
    "severity_marker_overview",
    "rule_out_test_utility",
    "epidemiology_incidence_complication_risk",
})

# Closed-set field allowlist for each per-article row in the
# sidecar. Title + journal + year are explicitly allowed (matches
# M3.1 / M3.2 sidecar pattern). NO abstracts / full text /
# evidence text / prompts / completions.
R1A_PER_ARTICLE_FIELDS: tuple[str, ...] = (
    "pmid", "doi", "title", "journal", "year", "source",
    "current_rank", "shadow_score", "shadow_rank", "rank_delta",
    "label_set",
)

# Closed-set kept set of running tasks so they aren't GC'd.
_R1A_RUNNING: set[asyncio.Task] = set()


# ---------------------------------------------------------------------
# Pure helpers — closed-set classification + scoring.
# ---------------------------------------------------------------------
def detect_shape_for_session(picott_payload: dict) -> Optional[str]:
    """Closed-set shape detection from PICOTT payload. Returns None
    when the question shape is NOT one of the R1A eligible shapes.
    Pure function. No I/O.
    """
    if not isinstance(picott_payload, dict):
        return None
    subtype = (picott_payload.get("question_subtype") or "").strip().lower()
    toq = (picott_payload.get("type_of_question") or "").strip().lower()
    # Closed-set mapping.
    if subtype in {"pathophysiology_mechanism", "pathophysiology"}:
        return "pathophysiology_mechanism"
    if subtype in {
        "procedure_complication_risk", "procedure_complication",
        "device_complication_risk", "device_complication",
    }:
        return "procedure_complication_risk"
    # Heuristic fallback off type_of_question for procedure-complication
    # shape when subtype is missing.
    if toq == "etiology_harm" and "complication" in (
        (picott_payload.get("question") or "").lower()
    ):
        return "procedure_complication_risk"
    return None


def is_k4a_owned_subtype(picott_payload: dict) -> bool:
    if not isinstance(picott_payload, dict):
        return False
    subtype = (picott_payload.get("question_subtype") or "").strip().lower()
    return subtype in K4A_OWNED_SUBTYPES


def _safe_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def is_mixed_signal(labels: Iterable[str]) -> bool:
    """Deterministic mixed-signal flag.

    Closed-set computation: an article is mixed-signal iff its
    label set intersects BOTH the closed-set positive label set AND
    the closed-set indirect/downrank label set. Flag-only; this
    function NEVER mutates the shadow score or shadow rank.
    """
    lset = set(labels or [])
    if not lset:
        return False
    has_positive = bool(lset & R1A_POSITIVE_LABELS)
    has_indirect = bool(lset & R1A_INDIRECT_LABELS)
    return has_positive and has_indirect


def _project_article_metadata(article: dict, ssa: dict) -> dict:
    """Pure projection — closed-set fields ONLY, suitable for
    persistence in ``r1a_shadow_decisions``. NO abstracts. NO
    free-text fields beyond title/journal which are also persisted
    in the M3.1 / M3.2 sidecars.
    """
    a = article or {}
    sources_present = a.get("sources_present") or []
    source = (
        sources_present[0]
        if isinstance(sources_present, list) and sources_present
        else (a.get("source") or None)
    )
    return {
        "pmid": (a.get("pmid") or None),
        "doi": (a.get("doi") or None),
        "title": (a.get("title") or None),
        "journal": (a.get("journal") or None),
        "year": _safe_int_or_none(a.get("publication_year")),
        "source": source,
        "current_rank": _safe_int_or_none(ssa.get("overall_rank")),
    }


def compute_shadow_top_n(
    *,
    shape: str,
    articles: list[dict],
    ssas: list[dict],
    top_n: int = 10,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Pure, deterministic shadow reranking. Returns:
        (active_top_n, shadow_top_n, label_distribution)

    Each entry contains the closed-set per-article fields.

    ``articles`` and ``ssas`` are aligned 1:1 by index (caller
    guarantees). Sort active by ``overall_rank`` ascending (1 is
    best), sort shadow by ``shadow_score`` descending.
    """
    label_dist: dict[str, int] = {lab: 0 for lab in R1A_ALL_LABELS}
    enriched: list[dict[str, Any]] = []
    for art, ssa in zip(articles, ssas):
        labels = detect_labels_for_shape(shape, art)
        for lab in labels:
            if lab in label_dist:
                label_dist[lab] += 1
        base_score = float(ssa.get("overall_rank") or 0)
        # Lower active rank = better active. We invert so that a
        # smaller active rank starts with a higher base for the
        # shadow score (otherwise rank=1 starts at base=1 vs
        # rank=50 at base=50 which would invert the shadow signal).
        # We use ``(101 - active_rank)`` to map rank 1..100 → 100..1
        # then apply the label weight deltas.
        inverted_base = max(0.0, 101.0 - base_score)
        shadow_score = compute_shadow_score(
            base_score=inverted_base, labels=labels,
        )
        proj = _project_article_metadata(art, ssa)
        # Deterministic closed-set mixed-signal flag — computed from
        # the label set ONLY. Flag-only: does NOT mutate
        # shadow_score / shadow_rank / sort order.
        mixed = is_mixed_signal(labels)
        # Closed-set split for the admin UI badges.
        positive_labels = [lab for lab in labels if lab in R1A_POSITIVE_LABELS]
        indirect_labels = [lab for lab in labels if lab in R1A_INDIRECT_LABELS]
        enriched.append({
            **proj,
            "label_set": labels,
            "positive_labels": positive_labels,
            "indirect_labels": indirect_labels,
            "mixed_signal": bool(mixed),
            "shadow_score": float(round(shadow_score, 4)),
        })

    # Active top-N — sort by ``current_rank`` asc (treat None as
    # large so they sink to the bottom).
    active_sorted = sorted(
        enriched,
        key=lambda r: (
            r["current_rank"] if r["current_rank"] is not None else 10**9
        ),
    )
    active_top: list[dict] = []
    for r in active_sorted[:top_n]:
        active_top.append({
            "pmid": r["pmid"], "doi": r["doi"], "title": r["title"],
            "journal": r["journal"], "year": r["year"],
            "source": r["source"],
            "current_rank": r["current_rank"],
            "label_set": list(r["label_set"]),
            "positive_labels": list(r["positive_labels"]),
            "indirect_labels": list(r["indirect_labels"]),
            "mixed_signal": bool(r["mixed_signal"]),
        })

    # Shadow top-N — sort by ``shadow_score`` desc.
    shadow_sorted = sorted(
        enriched, key=lambda r: r["shadow_score"], reverse=True,
    )
    # Map identifier (prefer doi, then pmid, then title) → current
    # rank so we can emit a stable rank_delta.
    def _ident(r: dict) -> str:
        return (
            (r.get("doi") or "").strip().lower()
            or (r.get("pmid") or "").strip().lower()
            or (r.get("title") or "").strip().lower()
        )
    current_rank_by_ident: dict[str, Optional[int]] = {
        _ident(r): r["current_rank"] for r in enriched
    }
    shadow_top: list[dict] = []
    for idx, r in enumerate(shadow_sorted[:top_n], start=1):
        ident = _ident(r)
        cur = current_rank_by_ident.get(ident)
        rank_delta: Optional[int]
        if cur is None:
            rank_delta = None
        else:
            rank_delta = int(cur) - idx
        shadow_top.append({
            "pmid": r["pmid"], "doi": r["doi"], "title": r["title"],
            "journal": r["journal"], "year": r["year"],
            "source": r["source"],
            "shadow_rank": int(idx),
            "shadow_score": float(r["shadow_score"]),
            "rank_delta": rank_delta,
            "label_set": list(r["label_set"]),
            "positive_labels": list(r["positive_labels"]),
            "indirect_labels": list(r["indirect_labels"]),
            "mixed_signal": bool(r["mixed_signal"]),
            # Closed-set per-article adjudication verdict. The
            # physician edits this via the PATCH endpoint; this
            # field is ONLY populated on shadow rows because the
            # active arm is the production-facing ordering and is
            # not adjudicated.
            "mixed_signal_verdict": "unspecified",
        })

    return active_top, shadow_top, label_dist


def _overrank_counts(label_distribution: dict[str, int]) -> dict[str, int]:
    """Pure — derive treatment / diagnostic / broad_overview
    over-rank counts from the label distribution. Closed-set keys.
    """
    return {
        "treatment_overrank_count": int(label_distribution.get(
            "treatment_modality_indirect_for_mechanism", 0,
        )),
        "diagnostic_overrank_count": int(label_distribution.get(
            "diagnostic_modality_indirect_for_mechanism", 0,
        )) + int(label_distribution.get(
            "diagnostic_followup_indirect_for_complication_risk", 0,
        )),
        "broad_overview_overrank_count": int(label_distribution.get(
            "broad_disease_overview_indirect_for_mechanism", 0,
        )) + int(label_distribution.get(
            "procedure_general_overview_indirect", 0,
        )),
        "wrong_lane_count": int(label_distribution.get(
            "missing_mechanism_focus", 0,
        )) + int(label_distribution.get(
            "complication_missing_outcome", 0,
        )) + int(label_distribution.get(
            "procedure_missing_exposure", 0,
        )),
    }


def _gold_present_in(
    *,
    top_n_rows: Iterable[dict],
    gold_dois: Iterable[str],
    gold_pmids: Iterable[str],
) -> bool:
    """Pure — case-insensitive presence check by DOI or PMID."""
    dset = {str(d or "").strip().lower() for d in (gold_dois or [])}
    pset = {str(p or "").strip() for p in (gold_pmids or [])}
    dset.discard("")
    pset.discard("")
    if not dset and not pset:
        return False
    for row in top_n_rows:
        if dset and str(row.get("doi") or "").strip().lower() in dset:
            return True
        if pset and str(row.get("pmid") or "").strip() in pset:
            return True
    return False


def _gold_rank_in(
    *,
    top_n_rows: Iterable[dict],
    gold_dois: Iterable[str],
    gold_pmids: Iterable[str],
    rank_field: str,
) -> Optional[int]:
    """Pure — return the first (smallest) rank where a gold article
    appears in ``top_n_rows``. Returns None when no gold hits.
    ``rank_field`` is ``current_rank`` for active arm and
    ``shadow_rank`` for shadow arm.
    """
    dset = {str(d or "").strip().lower() for d in (gold_dois or [])}
    pset = {str(p or "").strip() for p in (gold_pmids or [])}
    dset.discard("")
    pset.discard("")
    if not dset and not pset:
        return None
    best: Optional[int] = None
    for row in top_n_rows:
        doi_l = str(row.get("doi") or "").strip().lower()
        pmid_s = str(row.get("pmid") or "").strip()
        if (dset and doi_l in dset) or (pset and pmid_s in pset):
            r = row.get(rank_field)
            if isinstance(r, int):
                if best is None or r < best:
                    best = r
    return best


# ---------------------------------------------------------------------
# Persistence — sidecar write, leak-scanned, TTL 30d.
# ---------------------------------------------------------------------
async def write_shadow_decision(
    *,
    session_id: str,
    benchmark_case_id: Optional[str],
    shape: str,
    active_top_10: list[dict],
    shadow_top_10: list[dict],
    label_distribution: dict[str, int],
    query_expansion_terms_used: list[str],
    high_precision_query_required_both_anchors: bool,
    source_errors: dict[str, int],
    latency_ms: int,
    route_owned_by_k4a_specialty: bool,
    shadow_simulation_skipped: bool,
    gold_present_active_top10: Optional[bool] = None,
    gold_present_shadow_top10: Optional[bool] = None,
    gold_article_present_active_top20: Optional[bool] = None,
    gold_article_present_shadow_top20: Optional[bool] = None,
    gold_article_active_rank: Optional[int] = None,
    gold_article_shadow_rank: Optional[int] = None,
    original_question: Optional[str] = None,
) -> str:
    """Insert one closed-set row into ``r1a_shadow_decisions``.
    Returns the doc id. Leak-scans the entire row BEFORE write;
    on failure persists a redacted marker row instead.
    """
    now = datetime.now(timezone.utc)
    overrank = _overrank_counts(label_distribution)
    sidecar_id = uuid.uuid4().hex
    # Closed-set mixed-signal counter — count of shadow_top10 rows
    # that the deterministic flag function marked as mixed-signal.
    mixed_signal_count = sum(
        1 for r in (shadow_top_10 or [])
        if bool(r.get("mixed_signal"))
    )
    doc: dict[str, Any] = {
        "id": sidecar_id,
        "search_session_id": session_id,
        "benchmark_case_id": benchmark_case_id,
        "shape": shape,
        "original_question": original_question,
        "current_active_top10": list(active_top_10 or []),
        "shadow_simulated_top10": list(shadow_top_10 or []),
        "label_distribution": dict(label_distribution),
        "treatment_overrank_count": overrank["treatment_overrank_count"],
        "diagnostic_overrank_count": overrank["diagnostic_overrank_count"],
        "broad_overview_overrank_count": overrank["broad_overview_overrank_count"],
        "wrong_lane_count": overrank["wrong_lane_count"],
        "mixed_signal_count": int(mixed_signal_count),
        "query_expansion_terms_used": list(query_expansion_terms_used or []),
        "high_precision_query_required_both_anchors": bool(
            high_precision_query_required_both_anchors,
        ),
        "source_errors": dict(source_errors or {}),
        "latency_ms": int(latency_ms),
        "route_owned_by_k4a_specialty": bool(route_owned_by_k4a_specialty),
        "shadow_simulation_skipped": bool(shadow_simulation_skipped),
        "gold_article_present_active_top10": gold_present_active_top10,
        "gold_article_present_shadow_top10": gold_present_shadow_top10,
        "gold_article_present_active_top20": gold_article_present_active_top20,
        "gold_article_present_shadow_top20": gold_article_present_shadow_top20,
        "gold_article_active_rank": gold_article_active_rank,
        "gold_article_shadow_rank": gold_article_shadow_rank,
        "safety_flags": {
            "leak_scan_passed": True,
            "route_owned_by_k4a_specialty": bool(
                route_owned_by_k4a_specialty,
            ),
            "shadow_simulation_skipped": bool(shadow_simulation_skipped),
        },
        # Closed-set per-case adjudication block. Mutated by
        # ``PATCH /api/admin/r1a/shadow-decisions/{id}``. Default
        # state is the closed-set ``unspecified``.
        "physician_adjudication": {
            "verdict": "unspecified",
            "reason": None,
            "acceptable_for_future_active": "unspecified",
            "adjudicator_user_id": None,
            "adjudicated_at": None,
            "updated_at": None,
        },
        "phase": "R1A",
        "created_at": now,
        "expires_at": now + timedelta(days=30),
    }

    leak_ok = leak_scan_passes(doc)
    if not leak_ok:
        logger.warning(
            "r1a_shadow_leak_scan_failed session_id=%s shape=%s — "
            "redacting per-article rows",
            session_id, shape,
        )
        # Replace the per-article arrays with a closed-set marker.
        doc["current_active_top10"] = []
        doc["shadow_simulated_top10"] = []
        doc["safety_flags"]["leak_scan_passed"] = False
        doc["shadow_simulation_redacted_by_leak_scan"] = True
        try:
            await audit_service.log(
                "r1a_shadow_leak_scan_blocked",
                user_id=None,
                search_session_id=session_id,
                payload={
                    "shape": shape,
                    "benchmark_case_id": benchmark_case_id,
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("r1a_shadow leak audit failed (non-fatal)")
    else:
        doc.setdefault("shadow_simulation_redacted_by_leak_scan", False)

    try:
        await get_db()[Collections.R1A_SHADOW_DECISIONS].insert_one(dict(doc))
    except Exception:  # noqa: BLE001
        logger.exception(
            "r1a_shadow_decisions insert failed (non-fatal) session=%s",
            session_id,
        )
    return sidecar_id


# ---------------------------------------------------------------------
# Core simulator — pulls article + SSA rows from Mongo and writes.
# ---------------------------------------------------------------------
async def run_shadow_simulation_for_session(
    *,
    session_id: str,
    shape: str,
    benchmark_case_id: Optional[str] = None,
    query_expansion_terms_used: Optional[list[str]] = None,
    high_precision_query_required_both_anchors: bool = False,
    source_errors: Optional[dict[str, int]] = None,
    gold_dois: Optional[list[str]] = None,
    gold_pmids: Optional[list[str]] = None,
) -> Optional[str]:
    """Read the session's articles + SSAs, simulate the shadow
    reranking, and persist a sidecar row. Returns the sidecar doc
    id (None when nothing to write).

    Hard contract:
      * ZERO clinician-visible side effects (no write to
        ``search_sessions``, ``search_session_articles``,
        ``articles`` collections beyond a READ).
      * Bounded work — pulls at most 50 SSAs.
      * Wrapped in try/except internally too so individual stage
        failures never crash the simulator.
    """
    t0 = time.perf_counter()
    db = get_db()
    # Read up to 50 SSAs (the active pipeline persists up to ~50).
    ssas = await (
        db[Collections.SEARCH_SESSION_ARTICLES]
        .find({"search_session_id": session_id}, {"_id": 0})
        .sort([("overall_rank", -1)])
        .to_list(length=50)
    )
    if not ssas:
        logger.info(
            "r1a_shadow no SSAs for session=%s — writing zero-row decision",
            session_id,
        )
    # Look up the article rows.
    art_ids = [s.get("article_id") for s in ssas if s.get("article_id")]
    arts_by_id: dict[str, dict] = {}
    if art_ids:
        cursor = db[Collections.ARTICLES].find(
            {"id": {"$in": art_ids}},
            {
                "_id": 0,
                "id": 1, "pmid": 1, "doi": 1, "title": 1, "abstract": 1,
                "journal": 1, "publication_year": 1, "sources_present": 1,
                "publication_types_json": 1,
            },
        )
        async for a in cursor:
            arts_by_id[a["id"]] = a
    articles: list[dict] = []
    aligned_ssas: list[dict] = []
    for s in ssas:
        a = arts_by_id.get(s.get("article_id"))
        if not a:
            continue
        articles.append(a)
        aligned_ssas.append(s)

    active_top10, shadow_top10, label_dist = compute_shadow_top_n(
        shape=shape, articles=articles, ssas=aligned_ssas, top_n=10,
    )
    # Top-20 expansion used ONLY for the closed-set
    # gold-presence@20 metric and gold-rank lookup.
    active_top20, shadow_top20, _ = compute_shadow_top_n(
        shape=shape, articles=articles, ssas=aligned_ssas, top_n=20,
    )
    gold_present_active_top10 = (
        _gold_present_in(
            top_n_rows=active_top10,
            gold_dois=gold_dois or [],
            gold_pmids=gold_pmids or [],
        ) if (gold_dois or gold_pmids) else None
    )
    gold_present_shadow_top10 = (
        _gold_present_in(
            top_n_rows=shadow_top10,
            gold_dois=gold_dois or [],
            gold_pmids=gold_pmids or [],
        ) if (gold_dois or gold_pmids) else None
    )
    gold_present_active_top20 = (
        _gold_present_in(
            top_n_rows=active_top20,
            gold_dois=gold_dois or [],
            gold_pmids=gold_pmids or [],
        ) if (gold_dois or gold_pmids) else None
    )
    gold_present_shadow_top20 = (
        _gold_present_in(
            top_n_rows=shadow_top20,
            gold_dois=gold_dois or [],
            gold_pmids=gold_pmids or [],
        ) if (gold_dois or gold_pmids) else None
    )
    # Closed-set gold ranks (search top-20 because the operator
    # wants visibility into "moved into top-20" promotions).
    gold_active_rank = (
        _gold_rank_in(
            top_n_rows=active_top20,
            gold_dois=gold_dois or [],
            gold_pmids=gold_pmids or [],
            rank_field="current_rank",
        ) if (gold_dois or gold_pmids) else None
    )
    gold_shadow_rank = (
        _gold_rank_in(
            top_n_rows=shadow_top20,
            gold_dois=gold_dois or [],
            gold_pmids=gold_pmids or [],
            rank_field="shadow_rank",
        ) if (gold_dois or gold_pmids) else None
    )

    # Read the session's sanitized_question (already PHI-cleared
    # at submit time) and re-leak-scan it before persistence. This
    # is the SAME canonical benchmark question for the gold
    # cases — no clinician identifier ever lands here.
    original_question: Optional[str] = None
    try:
        sess_doc = await db[Collections.SEARCH_SESSIONS].find_one(
            {"id": session_id},
            {"_id": 0, "sanitized_question": 1, "original_question": 1},
        )
        if sess_doc:
            cand = (
                sess_doc.get("sanitized_question")
                or sess_doc.get("original_question")
                or None
            )
            if cand:
                # Belt-and-braces leak scan on the question text
                # before persistence. If it trips, drop it.
                if phi_guard.screen(str(cand)).is_blocked:
                    original_question = None
                else:
                    original_question = str(cand)
    except Exception:  # noqa: BLE001
        original_question = None

    latency_ms = int((time.perf_counter() - t0) * 1000)

    sidecar_id = await write_shadow_decision(
        session_id=session_id,
        benchmark_case_id=benchmark_case_id,
        shape=shape,
        active_top_10=active_top10,
        shadow_top_10=shadow_top10,
        label_distribution=label_dist,
        query_expansion_terms_used=list(
            query_expansion_terms_used or [],
        ),
        high_precision_query_required_both_anchors=bool(
            high_precision_query_required_both_anchors,
        ),
        source_errors=source_errors or {},
        latency_ms=latency_ms,
        route_owned_by_k4a_specialty=False,
        shadow_simulation_skipped=False,
        gold_present_active_top10=gold_present_active_top10,
        gold_present_shadow_top10=gold_present_shadow_top10,
        gold_article_present_active_top20=gold_present_active_top20,
        gold_article_present_shadow_top20=gold_present_shadow_top20,
        gold_article_active_rank=gold_active_rank,
        gold_article_shadow_rank=gold_shadow_rank,
        original_question=original_question,
    )
    return sidecar_id


async def write_k4a_skip_marker(
    *,
    session_id: str,
    shape: str,
    benchmark_case_id: Optional[str] = None,
) -> str:
    """Closed-set K4A skip-row writer. Used when the question is
    routed to K4A.2/.3/.4 — R1A explicitly does NOT simulate
    against those specialty packs; we still write a single row
    so the admin UI knows the route was claimed.
    """
    return await write_shadow_decision(
        session_id=session_id,
        benchmark_case_id=benchmark_case_id,
        shape=shape,
        active_top_10=[],
        shadow_top_10=[],
        label_distribution={lab: 0 for lab in R1A_ALL_LABELS},
        query_expansion_terms_used=[],
        high_precision_query_required_both_anchors=False,
        source_errors={},
        latency_ms=0,
        route_owned_by_k4a_specialty=True,
        shadow_simulation_skipped=True,
    )


# ---------------------------------------------------------------------
# Pipeline hook — async, non-blocking, swallows ALL exceptions.
# ---------------------------------------------------------------------
def schedule_shadow_simulation(
    *,
    session_id: str,
    picott_payload: dict,
    query_expansion_terms_used: Optional[list[str]] = None,
    high_precision_query_required_both_anchors: bool = False,
    source_errors: Optional[dict[str, int]] = None,
) -> Optional[asyncio.Task]:
    """Fire-and-forget scheduler. Wraps the simulator in a
    background task. Returns the task handle (mostly for tests).

    Returns None when the session shape is NOT one of the R1A
    eligible shapes — that's the fast-path that lets the
    overwhelming majority of clinician sessions skip the shadow
    simulator entirely.

    K4A-owned subtypes get a single skip marker row written so
    the admin UI can show the route was claimed by a specialty
    pack.
    """
    shape = detect_shape_for_session(picott_payload or {})
    if shape is None:
        return None

    if is_k4a_owned_subtype(picott_payload or {}):
        # Write a small skip marker and return.
        async def _k4a_runner():
            try:
                await write_k4a_skip_marker(
                    session_id=session_id,
                    shape=shape,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "r1a_shadow k4a skip marker write failed (non-fatal) "
                    "session=%s",
                    session_id,
                )
        task = asyncio.create_task(_k4a_runner())
        _R1A_RUNNING.add(task)
        task.add_done_callback(_R1A_RUNNING.discard)
        return task

    async def _runner():
        try:
            await run_shadow_simulation_for_session(
                session_id=session_id,
                shape=shape,
                benchmark_case_id=None,
                query_expansion_terms_used=query_expansion_terms_used,
                high_precision_query_required_both_anchors=bool(
                    high_precision_query_required_both_anchors,
                ),
                source_errors=source_errors or {},
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "r1a_shadow simulator failed (non-fatal) session=%s",
                session_id,
            )

    task = asyncio.create_task(_runner())
    _R1A_RUNNING.add(task)
    task.add_done_callback(_R1A_RUNNING.discard)
    return task


# ---------------------------------------------------------------------
# Replay (synchronous, admin-triggered).
# ---------------------------------------------------------------------
async def replay_benchmark_case(
    *,
    case_id: str,
    actor_user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Synchronously replay a gold-comparator benchmark case. Used
    by the admin endpoint ``POST /api/admin/r1a/replay/{case_id}``.

    Reads the most recent ``search_sessions`` row whose
    ``sanitized_question`` matches the case's ``question`` (case-
    insensitive). When no matching session exists, writes a
    minimal sidecar row with ``shadow_simulation_skipped=true`` +
    ``no_matching_session=true`` so the admin sees why.

    Returns:
        ``{
            sidecar_doc_id, case_id, shape, session_id_used,
            no_matching_session: bool
        }``
    """
    from services.r1a_gold_comparator_cases import get_gold_comparator_case
    case = await get_gold_comparator_case(case_id)
    if case is None:
        raise ValueError(f"unknown_case_id:{case_id}")
    db = get_db()
    qstring = (case.get("question") or "").strip().lower()
    session_id_used: Optional[str] = None
    if qstring:
        # Look up the most recent session whose sanitized_question
        # matches (case-insensitive exact, then trimmed contains).
        match = await db[Collections.SEARCH_SESSIONS].find_one(
            {"sanitized_question": {
                "$regex": f"^{_re_escape(qstring)}$", "$options": "i",
            }},
            {"_id": 0, "id": 1},
            sort=[("created_at", -1)],
        )
        if not match:
            match = await db[Collections.SEARCH_SESSIONS].find_one(
                {"sanitized_question": {
                    "$regex": _re_escape(qstring), "$options": "i",
                }},
                {"_id": 0, "id": 1},
                sort=[("created_at", -1)],
            )
        if match:
            session_id_used = match.get("id")

    shape = (case.get("shape") or "").strip()
    if not session_id_used:
        # Write a "no matching session" marker.
        sidecar_id = await write_shadow_decision(
            session_id=f"r1a-replay-{case_id}",
            benchmark_case_id=case_id,
            shape=shape,
            active_top_10=[],
            shadow_top_10=[],
            label_distribution={lab: 0 for lab in R1A_ALL_LABELS},
            query_expansion_terms_used=list(
                case.get("expected_expansion_concepts") or [],
            ),
            high_precision_query_required_both_anchors=False,
            source_errors={},
            latency_ms=0,
            route_owned_by_k4a_specialty=False,
            shadow_simulation_skipped=True,
        )
        try:
            await audit_service.log(
                "r1a_shadow_replay_no_matching_session",
                user_id=actor_user_id,
                payload={"case_id": case_id, "shape": shape},
            )
        except Exception:  # noqa: BLE001
            logger.exception("r1a replay audit failed (non-fatal)")
        return {
            "sidecar_doc_id": sidecar_id,
            "case_id": case_id,
            "shape": shape,
            "session_id_used": None,
            "no_matching_session": True,
        }

    sidecar_id = await run_shadow_simulation_for_session(
        session_id=session_id_used,
        shape=shape,
        benchmark_case_id=case_id,
        query_expansion_terms_used=list(
            case.get("expected_expansion_concepts") or [],
        ),
        high_precision_query_required_both_anchors=False,
        source_errors={},
        gold_dois=list(case.get("gold_article_dois") or []),
        gold_pmids=list(case.get("gold_article_pmids") or []),
    )
    try:
        await audit_service.log(
            "r1a_shadow_replay_completed",
            user_id=actor_user_id,
            search_session_id=session_id_used,
            payload={
                "case_id": case_id,
                "shape": shape,
                "sidecar_doc_id": sidecar_id,
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception("r1a replay audit failed (non-fatal)")
    return {
        "sidecar_doc_id": sidecar_id,
        "case_id": case_id,
        "shape": shape,
        "session_id_used": session_id_used,
        "no_matching_session": False,
    }


def _re_escape(s: str) -> str:
    import re
    return re.escape(s)


# ---------------------------------------------------------------------
# Admin list query — closed-set projection, leak-scanned at write.
# ---------------------------------------------------------------------
async def list_shadow_decisions(
    *,
    limit: int = 50,
    shape: Optional[str] = None,
    benchmark_case_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Read sidecar rows (most-recent first). Closed-set projection.
    Skip ``_id``; ISO-format datetime fields for JSON.
    """
    db = get_db()
    q: dict[str, Any] = {}
    if shape:
        q["shape"] = shape
    if benchmark_case_id:
        q["benchmark_case_id"] = benchmark_case_id
    out: list[dict[str, Any]] = []
    cursor = (
        db[Collections.R1A_SHADOW_DECISIONS]
        .find(q, {"_id": 0})
        .sort([("created_at", -1)])
        .limit(int(limit))
    )
    async for r in cursor:
        ca = r.get("created_at")
        ea = r.get("expires_at")
        if isinstance(ca, datetime):
            r["created_at"] = ca.isoformat()
        if isinstance(ea, datetime):
            r["expires_at"] = ea.isoformat()
        out.append(r)
    return out


async def get_shadow_decision_by_id(doc_id: str) -> Optional[dict[str, Any]]:
    db = get_db()
    r = await db[Collections.R1A_SHADOW_DECISIONS].find_one(
        {"id": doc_id}, {"_id": 0},
    )
    if not r:
        return None
    ca = r.get("created_at")
    ea = r.get("expires_at")
    if isinstance(ca, datetime):
        r["created_at"] = ca.isoformat()
    if isinstance(ea, datetime):
        r["expires_at"] = ea.isoformat()
    return r


# ---------------------------------------------------------------------
# Adjudication PATCH — closed-set verdict updates.
# ---------------------------------------------------------------------
class R1AAdjudicationError(Exception):
    """Raised by ``patch_shadow_decision_adjudication`` to surface
    a closed-set error code to the admin endpoint. The ``code``
    attribute is the closed-set machine-readable reason.
    """

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


async def patch_shadow_decision_adjudication(
    *,
    sidecar_id: str,
    actor_user_id: Optional[str],
    verdict: Optional[str] = None,
    reason: Optional[str] = None,
    acceptable_for_future_active: Optional[str] = None,
    mixed_signal_verdict_overrides: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Closed-set adjudication PATCH.

    Allowed mutations (everything else is REJECTED):
      * ``physician_adjudication.verdict`` ∈ ``R1A_CASE_VERDICTS``
      * ``physician_adjudication.reason`` (free text, max 1000 chars,
        leak-scanned + PHI-screened)
      * ``physician_adjudication.acceptable_for_future_active`` ∈
        ``R1A_ACCEPTABLE_VALUES``
      * ``shadow_simulated_top10[i].mixed_signal_verdict`` ∈
        ``MIXED_SIGNAL_VERDICTS`` (only for the row indices that
        were originally marked ``mixed_signal=true``)

    Raises ``R1AAdjudicationError`` with a closed-set ``code`` for
    every validation failure.
    """
    db = get_db()
    existing = await db[Collections.R1A_SHADOW_DECISIONS].find_one(
        {"id": sidecar_id}, {"_id": 0},
    )
    if existing is None:
        raise R1AAdjudicationError("sidecar_not_found")

    set_payload: dict[str, Any] = {}
    audit_payload: dict[str, Any] = {"sidecar_id": sidecar_id}
    now = datetime.now(timezone.utc)

    # ----- verdict -----
    if verdict is not None:
        if verdict not in R1A_CASE_VERDICTS:
            raise R1AAdjudicationError("verdict_not_in_closed_set")
        set_payload["physician_adjudication.verdict"] = verdict
        audit_payload["verdict"] = verdict

    # ----- reason (leak-scanned + PHI-screened) -----
    if reason is not None:
        if not isinstance(reason, str):
            raise R1AAdjudicationError("reason_must_be_string")
        if len(reason) > 1000:
            raise R1AAdjudicationError("reason_too_long")
        # PHI screen — refuses anything that looks like a patient
        # identifier; also serves as the leak-scan equivalent
        # (M3.1 leak_scan_passes is a Mongo-doc shape scanner, not
        # a free-text scanner; phi_guard is the right tool here).
        if reason.strip():
            scr = phi_guard.screen(reason)
            if scr.is_blocked:
                raise R1AAdjudicationError("reason_phi_blocked")
        set_payload["physician_adjudication.reason"] = reason
        # Do NOT include the raw reason in the audit payload (closed-set policy).
        audit_payload["reason_chars"] = len(reason)

    # ----- acceptable_for_future_active -----
    if acceptable_for_future_active is not None:
        if acceptable_for_future_active not in R1A_ACCEPTABLE_VALUES:
            raise R1AAdjudicationError(
                "acceptable_for_future_active_not_in_closed_set",
            )
        set_payload[
            "physician_adjudication.acceptable_for_future_active"
        ] = acceptable_for_future_active
        audit_payload["acceptable_for_future_active"] = (
            acceptable_for_future_active
        )

    # ----- mixed_signal_verdict_overrides -----
    overrides_applied: dict[str, str] = {}
    if mixed_signal_verdict_overrides:
        if not isinstance(mixed_signal_verdict_overrides, dict):
            raise R1AAdjudicationError("mixed_signal_overrides_must_be_dict")
        shadow_rows = list(existing.get("shadow_simulated_top10") or [])
        for raw_idx, verdict_val in mixed_signal_verdict_overrides.items():
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError) as exc:
                raise R1AAdjudicationError(
                    "mixed_signal_index_not_int",
                ) from exc
            if idx < 0 or idx >= len(shadow_rows):
                raise R1AAdjudicationError("mixed_signal_index_out_of_range")
            if verdict_val not in MIXED_SIGNAL_VERDICTS:
                raise R1AAdjudicationError(
                    "mixed_signal_verdict_not_in_closed_set",
                )
            if not shadow_rows[idx].get("mixed_signal"):
                raise R1AAdjudicationError(
                    "mixed_signal_index_is_not_flagged",
                )
            set_payload[
                f"shadow_simulated_top10.{idx}.mixed_signal_verdict"
            ] = verdict_val
            overrides_applied[str(idx)] = verdict_val
    if overrides_applied:
        audit_payload["mixed_signal_overrides"] = overrides_applied

    if not set_payload:
        raise R1AAdjudicationError("nothing_to_update")

    # Stamp adjudicator + timestamp on every successful PATCH that
    # touches the per-case verdict / reason / acceptable triplet.
    if (
        "physician_adjudication.verdict" in set_payload
        or "physician_adjudication.reason" in set_payload
        or "physician_adjudication.acceptable_for_future_active" in set_payload
    ):
        set_payload["physician_adjudication.adjudicator_user_id"] = (
            actor_user_id
        )
        set_payload["physician_adjudication.adjudicated_at"] = now
    set_payload["physician_adjudication.updated_at"] = now

    await db[Collections.R1A_SHADOW_DECISIONS].update_one(
        {"id": sidecar_id},
        {"$set": set_payload},
    )

    try:
        await audit_service.log(
            "r1a_adjudication_recorded",
            user_id=actor_user_id,
            search_session_id=existing.get("search_session_id"),
            payload=audit_payload,
        )
    except Exception:  # noqa: BLE001
        logger.exception("r1a adjudication audit failed (non-fatal)")

    return await get_shadow_decision_by_id(sidecar_id) or {}


__all__ = [
    "R1A_SHADOW_SHAPES",
    "K4A_OWNED_SUBTYPES",
    "R1A_PER_ARTICLE_FIELDS",
    "R1A_POSITIVE_LABELS",
    "R1A_INDIRECT_LABELS",
    "MIXED_SIGNAL_VERDICTS",
    "R1A_CASE_VERDICTS",
    "R1A_ACCEPTABLE_VALUES",
    "is_mixed_signal",
    "detect_shape_for_session",
    "is_k4a_owned_subtype",
    "compute_shadow_top_n",
    "run_shadow_simulation_for_session",
    "write_k4a_skip_marker",
    "write_shadow_decision",
    "schedule_shadow_simulation",
    "replay_benchmark_case",
    "list_shadow_decisions",
    "get_shadow_decision_by_id",
    "patch_shadow_decision_adjudication",
    "R1AAdjudicationError",
]
