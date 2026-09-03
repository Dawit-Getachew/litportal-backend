# DEPRECATED (Phase M4.2 — 2026-06-09): Legacy M3.1 A/B retrieval benchmark
# harness. The forward semantic path is now M4. This module is RETAINED because
# (a) compile_b_arm_queries / leak_scan_passes are reused by the M4 active compiler
# and (b) the admin M3.1 benchmark surface is read-only/legacy. NOT a forward
# clinician path. Do not extend; new benchmark work belongs to the M4/K-series.

"""Phase M3.1 — Preview/dev-only admin A/B retrieval benchmark.

DETERMINISTIC METRICS. NO LLM-AS-JUDGE.

This module is the orchestrator for the M3.1 admin-only benchmark
endpoint. It:

  1. Loads an existing search session by id and extracts the A-arm
     (current active CQM + Boolean queries + top-10 stored articles).
  2. Reuses the validator-accepted M-phase semantic-repair candidate
     from ``session_diagnostics_json.semantic_anchor_repair`` (no
     fresh LLM call — M2.1 already produced this).
  3. Compiles a minimal deterministic B-arm Boolean-query set from
     the candidate's ``required_anchors`` / ``slots`` /
     ``optional_or_ranking_only_concepts``.
  4. Issues those queries to PubMed E-utilities and Europe PMC ONLY
     (per scope decision) with per-source 5 s and total 12 s
     timeouts.
  5. Computes a closed-set of deterministic metrics.
  6. Persists the run to the ``m3_1_benchmark_runs`` Mongo
     collection with a 30-day TTL.

NEVER writes to ``search_session_articles``, evidence packets,
summaries, default-selection state, clinician recent-searches, or
any clinician-visible collection. The full-pipeline source-config
``last_success_at`` / ``last_error_at`` markers are also untouched
— this module uses direct HTTPX calls so benchmark traffic does
NOT pollute clinician-facing source-health metrics.

OpenAlex keyword search is intentionally NOT wired (enrichment-only
per existing app design; new keyword search would be a new
connector).

raw_question mode is intentionally NOT implemented in M3.1
(deferred per scope decision); the route handler returns 501 with
closed-set status ``m3_1_raw_question_deferred``.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from core.db import Collections, get_db

logger = logging.getLogger("clinevidence.m3_1_benchmark")


# ---------------------------------------------------------------------
# Closed-set diagnostic enums
# ---------------------------------------------------------------------

ELIGIBILITY_DECISIONS: frozenset[str] = frozenset({
    "eligible",
    "ineligible_phi_blocked",
    "ineligible_generic_only",
    "ineligible_schema_invalid",
    "ineligible_low_confidence",
    "ineligible_detector_not_triggered",
    "ineligible_no_repair_candidate",
    "ineligible_session_not_found",
    "ineligible_unsafe_or_refused",
    "ineligible_central_anchor_removed",
    "ineligible_unvalidated_mesh",
    "route_owned_by_k4a_specialty_K4A_2",
    "route_owned_by_k4a_specialty_K4A_3",
    "route_owned_by_k4a_specialty_K4A_4",
})

TIMEOUT_STATUSES: frozenset[str] = frozenset({
    "none",
    "m_repair_timeout",
    "b_retrieval_timeout",
    "per_source_timeout_pubmed",
    "per_source_timeout_epmc",
})

FALLBACK_STATUSES: frozenset[str] = frozenset({
    "none",
    "b_partial_results",
    "b_unavailable",
})


# K4A.* subtype → eligibility-decision mapping.
_K4A_DECISION_MAP: dict[str, str] = {
    "severity_marker_overview": "route_owned_by_k4a_specialty_K4A_2",
    "rule_out_test_utility": "route_owned_by_k4a_specialty_K4A_3",
    "epidemiology_incidence_complication_risk": "route_owned_by_k4a_specialty_K4A_4",
}

# Timeouts.
M_REPAIR_TIMEOUT_S = 8.0
B_RETRIEVAL_TOTAL_TIMEOUT_S = 12.0
PER_SOURCE_TIMEOUT_S = 5.0
TOP_N = 10


# ---------------------------------------------------------------------
# B-arm Boolean compiler — deterministic, minimal.
# ---------------------------------------------------------------------

def _q(s: str) -> str:
    """Quote a phrase for PubMed-style tiab / EPMC TITLE_ABS."""
    return s.replace('"', '').strip()


def compile_b_arm_queries(candidate: dict) -> dict[str, list[dict[str, str]]]:
    """Deterministic B-arm Boolean compiler.

    Returns a dict keyed by source name (``pubmed``, ``epmc``) whose
    values are lists of ``{"query_type": str, "query_string": str}``
    items. The query_type names use a closed set: ``b_required_AND``,
    ``b_required_AND_with_population``, ``b_required_AND_with_outcome``.

    No literal full-phrase anchors are emitted. The compiler operates
    only on the validator-accepted, closed-set candidate fields, so
    suspicious literal phrases (the M-phase detector's flag) can
    never reach the active-style queries.
    """
    required = [_q(a) for a in (candidate.get("required_anchors") or []) if a]
    slots = candidate.get("slots") or {}
    pop = [_q(s) for s in (slots.get("population") or []) if s]
    cond = [_q(s) for s in (slots.get("condition") or []) if s]
    exposure = [_q(s) for s in (slots.get("intervention_or_exposure") or []) if s]
    outcome = [_q(s) for s in (slots.get("outcome") or []) if s]
    index_test = [_q(s) for s in (slots.get("index_test") or []) if s]
    target_cond = [_q(s) for s in (slots.get("target_condition") or []) if s]
    procedure = [_q(s) for s in (slots.get("procedure_or_device") or []) if s]
    mechanism = [_q(s) for s in (slots.get("mechanism") or []) if s]

    # Pick the "central" anchors for the AND core: required_anchors
    # take precedence; otherwise fall back to condition or
    # target_condition or procedure_or_device.
    core = required[:] if required else []
    if not core:
        core = (cond or target_cond or procedure)[:1]
    if not core:
        return {"pubmed": [], "epmc": []}

    pubmed_qs: list[dict[str, str]] = []
    epmc_qs: list[dict[str, str]] = []

    # 1) Core AND
    pubmed_qs.append({
        "query_type": "b_required_AND",
        "query_string": (
            " AND ".join(f'"{x}"[tiab]' for x in core)
            + ' NOT "preprint"[Publication Type]'
        ),
    })
    epmc_qs.append({
        "query_type": "b_required_AND",
        "query_string": (
            " AND ".join(f'TITLE_ABS:"{x}"' for x in core)
            + ' NOT SRC:PPR'
        ),
    })

    # 2) Core AND with outcome (if outcome exists and is not in core)
    outcome_extra = [o for o in (outcome + mechanism) if o and o not in core]
    if outcome_extra:
        pubmed_qs.append({
            "query_type": "b_required_AND_with_outcome",
            "query_string": (
                "(" + " AND ".join(f'"{x}"[tiab]' for x in core) + ")"
                + " AND ("
                + " OR ".join(f'"{o}"[tiab]' for o in outcome_extra)
                + ')'
                + ' NOT "preprint"[Publication Type]'
            ),
        })
        epmc_qs.append({
            "query_type": "b_required_AND_with_outcome",
            "query_string": (
                "(" + " AND ".join(f'TITLE_ABS:"{x}"' for x in core) + ")"
                + " AND ("
                + " OR ".join(f'TITLE_ABS:"{o}"' for o in outcome_extra)
                + ") NOT SRC:PPR"
            ),
        })

    # 3) Core AND with population context (if exists and not in core)
    pop_extra = [p for p in pop if p and p not in core]
    if pop_extra:
        pubmed_qs.append({
            "query_type": "b_required_AND_with_population",
            "query_string": (
                "(" + " AND ".join(f'"{x}"[tiab]' for x in core) + ")"
                + " AND ("
                + " OR ".join(f'"{p}"[tiab]' for p in pop_extra)
                + ')'
                + ' NOT "preprint"[Publication Type]'
            ),
        })
        epmc_qs.append({
            "query_type": "b_required_AND_with_population",
            "query_string": (
                "(" + " AND ".join(f'TITLE_ABS:"{x}"' for x in core) + ")"
                + " AND ("
                + " OR ".join(f'TITLE_ABS:"{p}"' for p in pop_extra)
                + ") NOT SRC:PPR"
            ),
        })

    # 4) Core AND with exposure / index_test (if exists and not in core)
    extra_combine = [
        x for x in (exposure + index_test)
        if x and x not in core
    ]
    if extra_combine:
        pubmed_qs.append({
            "query_type": "b_required_AND_with_outcome",
            "query_string": (
                "(" + " AND ".join(f'"{x}"[tiab]' for x in core) + ")"
                + " AND ("
                + " OR ".join(f'"{c}"[tiab]' for c in extra_combine)
                + ')'
                + ' NOT "preprint"[Publication Type]'
            ),
        })
        epmc_qs.append({
            "query_type": "b_required_AND_with_outcome",
            "query_string": (
                "(" + " AND ".join(f'TITLE_ABS:"{x}"' for x in core) + ")"
                + " AND ("
                + " OR ".join(f'TITLE_ABS:"{c}"' for c in extra_combine)
                + ") NOT SRC:PPR"
            ),
        })

    return {"pubmed": pubmed_qs, "epmc": epmc_qs}


# ---------------------------------------------------------------------
# Minimal source clients (no source_configs side effects)
# ---------------------------------------------------------------------

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


async def _fetch_pubmed_top_n(
    client: httpx.AsyncClient,
    queries: list[dict[str, str]],
    *,
    top_n: int = TOP_N,
) -> tuple[list[dict[str, Any]], list[int], list[str]]:
    """Run a list of pubmed queries, return aggregated top-N article
    dicts (PMID/title/journal/year/source/query_type) deduped by
    PMID, plus per-query latency_ms and per-query error strings."""
    out: list[dict[str, Any]] = []
    seen_pmids: set[str] = set()
    latencies: list[int] = []
    errors: list[str] = []
    for q in queries:
        if len(out) >= top_n:
            break
        qs = q.get("query_string") or ""
        if not qs:
            continue
        t0 = time.perf_counter()
        try:
            r = await client.get(
                NCBI_ESEARCH,
                params={
                    "db": "pubmed",
                    "term": qs,
                    "retmode": "json",
                    "retmax": str(top_n * 2),
                    "tool": "ClinEvidenceSearch-M3_1",
                },
                timeout=PER_SOURCE_TIMEOUT_S,
            )
            r.raise_for_status()
            data = r.json()
            pmids = (data.get("esearchresult") or {}).get("idlist") or []
            if not pmids:
                latencies.append(int((time.perf_counter() - t0) * 1000))
                continue
            r2 = await client.get(
                NCBI_ESUMMARY,
                params={
                    "db": "pubmed",
                    "id": ",".join(pmids[: top_n * 2]),
                    "retmode": "json",
                    "tool": "ClinEvidenceSearch-M3_1",
                },
                timeout=PER_SOURCE_TIMEOUT_S,
            )
            r2.raise_for_status()
            summ = (r2.json() or {}).get("result") or {}
            for pmid in pmids:
                if pmid in seen_pmids or len(out) >= top_n:
                    continue
                doc = summ.get(pmid) or {}
                title = (doc.get("title") or "").strip()
                journal = (doc.get("fulljournalname") or doc.get("source") or "").strip()
                pubdate = (doc.get("pubdate") or "").strip()
                year = ""
                m = re.match(r"^(\d{4})", pubdate)
                if m:
                    year = m.group(1)
                seen_pmids.add(pmid)
                out.append({
                    "pmid": pmid,
                    "source_id": pmid,
                    "title": title[:300],
                    "journal": journal[:200],
                    "year": year,
                    "source": "pubmed",
                    "query_type": q.get("query_type", ""),
                })
            latencies.append(int((time.perf_counter() - t0) * 1000))
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc)[:200])
            latencies.append(int((time.perf_counter() - t0) * 1000))
    return out[:top_n], latencies, errors


async def _fetch_epmc_top_n(
    client: httpx.AsyncClient,
    queries: list[dict[str, str]],
    *,
    top_n: int = TOP_N,
) -> tuple[list[dict[str, Any]], list[int], list[str]]:
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    latencies: list[int] = []
    errors: list[str] = []
    for q in queries:
        if len(out) >= top_n:
            break
        qs = q.get("query_string") or ""
        if not qs:
            continue
        t0 = time.perf_counter()
        try:
            r = await client.get(
                EPMC_SEARCH,
                params={
                    "query": qs,
                    "format": "json",
                    "resultType": "lite",
                    "pageSize": str(top_n * 2),
                    "sort": "RELEVANCE",
                },
                timeout=PER_SOURCE_TIMEOUT_S,
            )
            r.raise_for_status()
            data = r.json()
            hits = ((data.get("resultList") or {}).get("result") or [])
            for h in hits:
                if len(out) >= top_n:
                    break
                pmid = (h.get("pmid") or "").strip()
                eid = (h.get("id") or "").strip()
                key = pmid or eid
                if not key or key in seen_ids:
                    continue
                seen_ids.add(key)
                out.append({
                    "pmid": pmid,
                    "source_id": eid,
                    "title": (h.get("title") or "").strip()[:300],
                    "journal": (h.get("journalTitle") or "").strip()[:200],
                    "year": (h.get("pubYear") or "").strip(),
                    "source": "epmc",
                    "query_type": q.get("query_type", ""),
                })
            latencies.append(int((time.perf_counter() - t0) * 1000))
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc)[:200])
            latencies.append(int((time.perf_counter() - t0) * 1000))
    return out[:top_n], latencies, errors


# Test hook — pytest can monkey-patch these.
async def _b_arm_retrieve(
    queries_per_source: dict[str, list[dict[str, str]]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[int]],
    dict[str, list[str]],
    str,
    str,
]:
    """Returns (top10, per_source_latencies, per_source_errors,
    timeout_status, fallback_status)."""
    pubmed_qs = queries_per_source.get("pubmed", []) or []
    epmc_qs = queries_per_source.get("epmc", []) or []
    if not pubmed_qs and not epmc_qs:
        return [], {"pubmed": [], "epmc": []}, {"pubmed": [], "epmc": []}, "none", "b_unavailable"
    timeout_status = "none"
    fallback_status = "none"
    try:
        async with httpx.AsyncClient(timeout=PER_SOURCE_TIMEOUT_S) as client:
            async def _pm():
                return await _fetch_pubmed_top_n(client, pubmed_qs)
            async def _ep():
                return await _fetch_epmc_top_n(client, epmc_qs)
            try:
                pm_res, ep_res = await asyncio.wait_for(
                    asyncio.gather(_pm(), _ep(), return_exceptions=True),
                    timeout=B_RETRIEVAL_TOTAL_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                return [], {"pubmed": [], "epmc": []}, {"pubmed": [], "epmc": []}, "b_retrieval_timeout", "b_unavailable"
    except Exception as exc:  # noqa: BLE001
        return [], {"pubmed": [], "epmc": []}, {"pubmed": [], "epmc": [str(exc)[:200]]}, "none", "b_unavailable"

    pm_articles: list[dict[str, Any]] = []
    pm_lat: list[int] = []
    pm_err: list[str] = []
    ep_articles: list[dict[str, Any]] = []
    ep_lat: list[int] = []
    ep_err: list[str] = []

    if isinstance(pm_res, Exception):
        pm_err = [str(pm_res)[:200]]
        if "Timeout" in type(pm_res).__name__:
            timeout_status = "per_source_timeout_pubmed"
    else:
        pm_articles, pm_lat, pm_err = pm_res  # type: ignore[misc]
    if isinstance(ep_res, Exception):
        ep_err = [str(ep_res)[:200]]
        if "Timeout" in type(ep_res).__name__:
            timeout_status = (
                "per_source_timeout_epmc"
                if timeout_status == "none" else timeout_status
            )
    else:
        ep_articles, ep_lat, ep_err = ep_res  # type: ignore[misc]

    # Interleave PubMed + EPMC for top-10 diversity, dedupe by PMID.
    seen_pmids: set[str] = set()
    interleaved: list[dict[str, Any]] = []
    for i in range(max(len(pm_articles), len(ep_articles))):
        if i < len(pm_articles):
            a = pm_articles[i]
            if a.get("pmid") and a["pmid"] not in seen_pmids:
                seen_pmids.add(a["pmid"])
                interleaved.append(a)
        if i < len(ep_articles):
            a = ep_articles[i]
            if a.get("pmid") and a["pmid"] in seen_pmids:
                continue
            if a.get("pmid"):
                seen_pmids.add(a["pmid"])
            interleaved.append(a)
        if len(interleaved) >= TOP_N:
            break
    if pm_err or ep_err:
        if interleaved:
            fallback_status = "b_partial_results"
        else:
            fallback_status = "b_unavailable"
    return (
        interleaved[:TOP_N],
        {"pubmed": pm_lat, "epmc": ep_lat},
        {"pubmed": pm_err, "epmc": ep_err},
        timeout_status,
        fallback_status,
    )


# ---------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------

def _norm(s: Any) -> str:
    return (str(s or "")).strip().lower()


def _query_set(qbf: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(qbf, list):
        for q in qbf:
            if isinstance(q, dict):
                s = q.get("query_string") or q.get("query") or ""
                if s:
                    out.add(s)
            elif isinstance(q, str) and q:
                out.add(q)
    elif isinstance(qbf, dict):
        for v in qbf.values():
            out |= _query_set(v)
    return out


# ---------------------------------------------------------------------
# Phase M3.1.1 — Shared A-arm serializer.
#
# Single source of truth for rendering the A-arm (current active
# retrieval) section of any M3.1 surface — admin GET payload, admin
# UI side-by-side panel, and the clinical adjudication packet.
#
# Reads ONLY from the canonical ``search_sessions`` schema fields:
#   * ``clinical_question_map_json``                — A-arm CQM
#   * ``clinical_question_map_json.branch_query_plan`` — branch + family queries
#   * ``picott_json``                                — PICOTT/PICOTS view
#   * ``primary_clinical_anchors_json.primary_anchors`` — primary anchors
#
# Does NOT read ``picott_payload`` or ``queries_by_family`` (those
# field names never existed on the real schema — they were the root
# of the M3.1.1 bug fixed by this commit).
#
# Pure function. No DB access. No LLM calls. No source calls. No
# side effects. Safe to call from any read path.
# ---------------------------------------------------------------------

_SHARED_A_ARM_SERIALIZER_NAME = "serialize_a_arm_from_session"


def _anchor_concept(a: dict) -> str:
    if not isinstance(a, dict):
        return ""
    return (a.get("canonical_concept") or a.get("text") or "").strip()


def serialize_a_arm_from_session(session_doc: Optional[dict]) -> dict[str, Any]:
    """Project a search-session document into the canonical A-arm
    display shape used by every M3.1 read path.

    Returns a dict with these top-level keys (always present, even if
    the session is missing the underlying fields — empty
    placeholders are returned instead of None to keep the UI
    rendering loop simple):

      * ``cqm``               (dict): primary_intent, question_subtype,
                                       frameworks, specialty, required_anchors
      * ``branch_query_plan`` (dict): schema_version, mode,
                                       active_for_retrieval, branches[*]
                                       each with query_families[*]
      * ``boolean_queries``   (list[str]): flat sorted unique query
                                       strings across all families —
                                       preserved for legacy callers /
                                       metrics-input parity.
      * ``picott_view``       (dict): type_of_question, question_subtype,
                                       specialty, must_have_concepts,
                                       ranking_concepts, negative_or_penalty_concepts,
                                       translated_question, interpretation_for_user
      * ``anchors``           (dict): primary_anchors[] list of
                                       {concept, entity_type, role}

    This serializer is deliberately schema-strict: it ignores
    ``picott_payload`` and ``queries_by_family`` entirely, even if
    present, because those are dead fields and reading them was the
    M3.1.1 bug.
    """
    if not isinstance(session_doc, dict):
        session_doc = {}

    cqm_doc = session_doc.get("clinical_question_map_json") or {}
    if not isinstance(cqm_doc, dict):
        cqm_doc = {}
    bqp_doc = cqm_doc.get("branch_query_plan") or {}
    if not isinstance(bqp_doc, dict):
        bqp_doc = {}
    picott_doc = session_doc.get("picott_json") or {}
    if not isinstance(picott_doc, dict):
        picott_doc = {}
    pca_doc = session_doc.get("primary_clinical_anchors_json") or {}
    if not isinstance(pca_doc, dict):
        pca_doc = {}

    # Primary anchors → normalized list of {concept, entity_type, role}.
    anchors_list: list[dict[str, Any]] = []
    for a in (pca_doc.get("primary_anchors") or []):
        if not isinstance(a, dict):
            continue
        c = _anchor_concept(a)
        if not c:
            continue
        anchors_list.append({
            "concept": c,
            "entity_type": a.get("entity_type"),
            "role": a.get("role"),
        })

    # A-arm CQM section.
    a_cqm = {
        "primary_intent": cqm_doc.get("primary_intent") or picott_doc.get("type_of_question"),
        "question_subtype": picott_doc.get("question_subtype"),
        "frameworks": list(cqm_doc.get("frameworks") or []),
        "specialty": picott_doc.get("specialty"),
        "required_anchors": [a["concept"] for a in anchors_list],
    }

    # Branch query plan — preserve the structured shape so the UI can
    # render per-family / per-source rows. Filter to deterministic
    # fields only (no prompts, no completions, no raw LLM
    # intermediates).
    branches_out: list[dict[str, Any]] = []
    for branch in (bqp_doc.get("branches") or []):
        if not isinstance(branch, dict):
            continue
        fams_out: list[dict[str, Any]] = []
        for fam in (branch.get("query_families") or []):
            if not isinstance(fam, dict):
                continue
            qs = fam.get("query_string") or ""
            if not isinstance(qs, str):
                qs = str(qs)
            fams_out.append({
                "family_name": fam.get("family_name"),
                "source": fam.get("source"),
                "query_string": qs,
                "qa_checks_passed": fam.get("qa_checks_passed"),
            })
        branches_out.append({
            "branch_id": branch.get("branch_id"),
            "intent": branch.get("intent"),
            "framework": branch.get("framework"),
            "branch_priority": branch.get("branch_priority"),
            "active_for_retrieval": branch.get("active_for_retrieval"),
            "query_families": fams_out,
        })
    branch_query_plan_out = {
        "schema_version": bqp_doc.get("schema_version"),
        "mode": bqp_doc.get("mode"),
        "active_for_retrieval": bqp_doc.get("active_for_retrieval"),
        "branches": branches_out,
    }

    # Flat sorted unique query-string list — kept for legacy callers
    # (e.g. ``compute_metrics`` a_queries input). NOTE: metric-input
    # plumbing in the orchestrator was historically the BROKEN code
    # path; switching that to use this flat list is an explicit
    # follow-up decision documented in the M3.1.1 closeout.
    flat_queries: list[str] = sorted(
        {
            fam["query_string"]
            for branch in branches_out
            for fam in branch.get("query_families") or []
            if fam.get("query_string")
        }
    )

    # PICOTT view — read-only projection of the deterministic
    # PICOTT/PICOTS payload. We only surface the salient fields a
    # physician/admin needs to read; the LLM-raw fields (e.g. rationale
    # text blobs) are deliberately omitted.
    picott_view = {
        "type_of_question": picott_doc.get("type_of_question"),
        "question_subtype": picott_doc.get("question_subtype"),
        "specialty": picott_doc.get("specialty"),
        "must_have_concepts": picott_doc.get("must_have_concepts") or [],
        "ranking_concepts": picott_doc.get("ranking_concepts") or [],
        "negative_or_penalty_concepts": picott_doc.get("negative_or_penalty_concepts") or [],
        "translated_question": picott_doc.get("translated_question"),
        "interpretation_for_user": picott_doc.get("interpretation_for_user"),
    }

    return {
        "cqm": a_cqm,
        "branch_query_plan": branch_query_plan_out,
        "boolean_queries": flat_queries,
        "picott_view": picott_view,
        "anchors": {"primary_anchors": anchors_list},
    }


def compute_metrics(
    *,
    a_top10: list[dict[str, Any]],
    b_top10: list[dict[str, Any]],
    a_queries: list[str],
    b_queries: list[str],
    candidate: dict,
    deterministic_primary_anchors: list[dict],
    original_question: str,
    per_source_latencies: dict[str, list[int]],
    per_source_errors: dict[str, list[str]],
) -> dict[str, Any]:
    a_set = set(a_queries)
    b_set = set(b_queries)
    query_diff = {
        "added": sorted(b_set - a_set),
        "removed": sorted(a_set - b_set),
        "common": sorted(a_set & b_set),
        "n_added": len(b_set - a_set),
        "n_removed": len(a_set - b_set),
    }
    # Required anchor coverage: fraction of candidate.required_anchors
    # that overlap (substring, case-insensitive) with deterministic
    # primary_anchors.
    cand_required = [_norm(a) for a in (candidate.get("required_anchors") or [])]
    det_terms = [
        _norm(a.get("canonical_concept") or a.get("text"))
        for a in (deterministic_primary_anchors or [])
        if isinstance(a, dict)
    ]
    if cand_required:
        covered = 0
        for c in cand_required:
            for d in det_terms:
                if not d:
                    continue
                if c in d or d in c:
                    covered += 1
                    break
        required_anchor_coverage = round(covered / max(1, len(cand_required)), 3)
    else:
        required_anchor_coverage = 0.0

    # Literal full-phrase anchor removed: was a long contiguous
    # n-gram from the original question used as a query anchor in
    # A but NOT in B? We strip leading wh-words first ("what are
    # the", "what is the", "does a", "how does", etc.) and look
    # for a 4+ word run from what remains.
    def _strip_wh_lead(q: str) -> str:
        q2 = q.strip(" ?.!\"'").lower()
        wh_leads = (
            "what are the ", "what is the ", "what are ", "what is ",
            "what's the ", "what's ", "does a ", "does the ", "does ",
            "how does the ", "how does ", "how do ", "how is the ",
            "how is ", "how are ", "why does ", "why is ", "why are ",
            "when does ", "when is ", "are there ", "is there ",
            "should ", "can a ", "can ", "in patients with ",
        )
        for lead in wh_leads:
            if q2.startswith(lead):
                q2 = q2[len(lead):]
                break
        return q2.strip()

    stem = _strip_wh_lead(original_question)
    # Take a 4+ word slice from the stem and check it's in A only.
    stem_tokens = stem.split()
    literal_in_a = False
    literal_in_b = False
    if len(stem_tokens) >= 4:
        # Build candidate n-grams of length 4..len.
        ngrams = []
        for n in range(min(len(stem_tokens), 8), 3, -1):
            for i in range(0, len(stem_tokens) - n + 1):
                ngrams.append(" ".join(stem_tokens[i:i + n]))
        a_low = [_norm(q) for q in a_set]
        b_low = [_norm(q) for q in b_set]
        for ng in ngrams:
            if any(ng in q for q in a_low):
                literal_in_a = True
                if any(ng in q for q in b_low):
                    literal_in_b = True
                    break
                else:
                    # found in A only — that's our signal
                    break
    literal_full_phrase_anchor_removed = bool(literal_in_a and not literal_in_b)

    # Generic-only-anchor absent: no purely generic required_anchor.
    GENERIC = {
        "patients", "adults", "young adults", "older adults", "elderly",
        "illness", "severity", "marker", "markers", "dose", "treatment",
        "causes", "risk", "risk factor", "risk factors", "infection",
        "thrombosis", "disease",
    }
    cand_required_set = {c for c in cand_required if c}
    generic_only_anchor_absent = bool(
        cand_required_set and not (cand_required_set <= GENERIC)
    )

    # overlap@10 — by PMID.
    a_pmids = {a.get("pmid") for a in a_top10 if a.get("pmid")}
    b_pmids = {b.get("pmid") for b in b_top10 if b.get("pmid")}
    overlap_at_10 = len(a_pmids & b_pmids)
    new_direct_hits_at_10 = len(b_pmids - a_pmids)

    # wrong_anchor_rate@10 — fraction of B top-10 whose title does NOT
    # contain ANY of the candidate's required_anchors (a deterministic
    # smell test for off-topic hits). Skipped when no required_anchor.
    if cand_required and b_top10:
        wrong = 0
        for art in b_top10:
            title = _norm(art.get("title"))
            if not any(c and c in title for c in cand_required):
                wrong += 1
        wrong_anchor_rate_at_10 = round(wrong / max(1, len(b_top10)), 3)
    else:
        wrong_anchor_rate_at_10 = 0.0

    # source_error_rate — fraction of per-source queries that errored.
    all_lat = []
    for src, lats in (per_source_latencies or {}).items():
        all_lat.extend(lats or [])
    total_attempts = sum(len(v or []) for v in per_source_latencies.values())
    total_errors = sum(len(v or []) for v in per_source_errors.values())
    source_error_rate = round(total_errors / max(1, total_attempts), 3)

    if all_lat:
        sorted_lat = sorted(all_lat)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p95 = sorted_lat[min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))]
        latency_max_ms = max(sorted_lat)
    else:
        p50 = p95 = latency_max_ms = 0

    return {
        "query_diff": query_diff,
        "required_anchor_coverage": required_anchor_coverage,
        "literal_full_phrase_anchor_removed": literal_full_phrase_anchor_removed,
        "generic_only_anchor_absent": generic_only_anchor_absent,
        "overlap_at_10": overlap_at_10,
        "new_direct_hits_at_10": new_direct_hits_at_10,
        "wrong_anchor_rate_at_10": wrong_anchor_rate_at_10,
        "source_error_rate": source_error_rate,
        "latency_p50_ms": int(p50),
        "latency_p95_ms": int(p95),
        "latency_max_ms": int(latency_max_ms),
    }


# ---------------------------------------------------------------------
# Leak scanner
# ---------------------------------------------------------------------

_PHI_PATTERNS = [
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\bMRN[:\s]*\d+\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"(?<![\w-])\+?\d{1,3}[\s\-.]?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}\b"),
]
_LEAK_SIGNALS = (
    "sk-", "Bearer ", "eyJ",  # api keys / JWT prefixes
    "You are the Bounded Semantic", "You are NOT answering",
    "system:", "assistant:",
)


def leak_scan_passes(doc: Any) -> bool:
    """Walk every string in ``doc`` and assert no PHI patterns and
    no LLM-prompt / credential boilerplate fragments are present.

    Known-identifier fields (``pmid``, ``source_id``, ``doi``,
    ``pmcid``, ``id``, ``_id``, ``triggered_by_admin_user_id``,
    ``search_session_id``, ``question_hash``, ``canonical_key``)
    are skipped: their values are structured PubMed/CrossRef IDs by
    construction (e.g. ``10.1103/1jsk-9q7w``,
    ``17843286.2026``) and the broad phone-number regex would
    otherwise false-positive on them. These fields are produced by
    the deterministic upstream pipeline (no LLM, no free text), so
    the deterministic-only safety contract is preserved.
    """
    _ID_FIELDS = frozenset({
        "pmid", "source_id", "doi", "pmcid", "id", "_id",
        "triggered_by_admin_user_id", "search_session_id",
        "question_hash", "canonical_key", "run_id",
    })

    def _walk(node: Any) -> bool:
        if isinstance(node, str):
            for pat in _PHI_PATTERNS:
                if pat.search(node):
                    return False
            for signal in _LEAK_SIGNALS:
                if signal in node:
                    return False
            return True
        if isinstance(node, dict):
            for k, v in node.items():
                if k in _ID_FIELDS:
                    # Known structured identifier field — skip scan.
                    continue
                if not _walk(v):
                    return False
            return True
        if isinstance(node, (list, tuple)):
            for v in node:
                if not _walk(v):
                    return False
            return True
        return True

    return _walk(doc)


# ---------------------------------------------------------------------
# Eligibility — adapts the M3-preflight eligibility decision to the
# M3.1 closed-set decision string.
# ---------------------------------------------------------------------

def _decide_eligibility(
    *,
    session_doc: Optional[dict],
    diagnostics: dict,
) -> tuple[str, dict[str, Any]]:
    """Returns (decision_str, eligibility_payload)."""
    # Session-not-found short-circuit.
    if not session_doc:
        return "ineligible_session_not_found", {
            "reason": "session_not_found",
            "k4a_route": None,
        }

    sar = (diagnostics or {}).get("session_diagnostics_json", {})
    if isinstance(sar, dict) and "semantic_anchor_repair" in sar:
        sar_block = sar.get("semantic_anchor_repair") or {}
    else:
        sar_block = (diagnostics or {}).get("semantic_anchor_repair") or {}

    triggered = bool(sar_block.get("semantic_repair_triggered"))
    trigger_reasons = list(sar_block.get("semantic_repair_trigger_reasons") or [])
    validation_status = sar_block.get("semantic_repair_validation_status")
    rejection_reasons = list(sar_block.get("semantic_repair_rejection_reasons") or [])
    candidate = sar_block.get("semantic_repair_candidate") if isinstance(sar_block, dict) else None
    latency_ms = int(sar_block.get("semantic_repair_latency_ms") or 0)

    # K4A.* short-circuit BEFORE candidate consideration.
    picott = diagnostics.get("picott") if isinstance(diagnostics, dict) else None
    question_subtype = (picott or {}).get("question_subtype") if isinstance(picott, dict) else None
    if not question_subtype:
        question_subtype = (session_doc or {}).get("question_subtype")
    k4a = _K4A_DECISION_MAP.get((question_subtype or "").strip().lower())
    if k4a:
        return k4a, {
            "k4a_route": question_subtype,
            "triggered": triggered,
            "candidate_validated": validation_status == "passed",
        }

    # PHI: a stored session that reached this code path is by
    # construction not PHI-blocked (PHI-blocked sessions never
    # produce a benchmark-able session). Still, defense in depth:
    if (session_doc or {}).get("status") == "blocked_phi":
        return "ineligible_phi_blocked", {"k4a_route": None}

    # Detector didn't fire.
    if not triggered or not trigger_reasons:
        return "ineligible_detector_not_triggered", {
            "triggered": triggered,
            "k4a_route": None,
        }

    # Validation result.
    if validation_status != "passed" or not isinstance(candidate, dict):
        if "generic_only_required_anchor" in rejection_reasons:
            return "ineligible_generic_only", {
                "candidate_validated": False,
                "rejection_reasons": rejection_reasons,
            }
        if "confidence_below_threshold" in rejection_reasons:
            return "ineligible_low_confidence", {
                "candidate_validated": False,
                "rejection_reasons": rejection_reasons,
            }
        if any(
            r in rejection_reasons for r in (
                "refused", "phi_pattern_detected",
                "article_reference_detected", "answer_text_detected",
                "unsafe_text_detected", "unapproved_source_referenced",
            )
        ):
            return "ineligible_unsafe_or_refused", {
                "rejection_reasons": rejection_reasons,
            }
        if "would_remove_central_anchor" in rejection_reasons:
            return "ineligible_central_anchor_removed", {
                "rejection_reasons": rejection_reasons,
            }
        if any(
            r in rejection_reasons for r in (
                "unvalidated_mesh_marked_validated", "unvalidated_mesh_promoted",
            )
        ):
            return "ineligible_unvalidated_mesh", {
                "rejection_reasons": rejection_reasons,
            }
        return "ineligible_schema_invalid", {
            "rejection_reasons": rejection_reasons,
        }

    # Candidate validated. Use the eligibility decision module for
    # full closed-set pass/block reasons but for the M3.1 endpoint
    # we collapse to the success path.
    return "eligible", {
        "candidate_validated": True,
        "triggered": True,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------
# Core orchestrator
# ---------------------------------------------------------------------

async def run_benchmark_for_session(
    *,
    session_id: str,
    admin_user_id: str,
) -> dict[str, Any]:
    """Main entrypoint. Loads the session, decides eligibility, runs
    B-arm comparison retrieval if eligible, persists the run, and
    returns the closed-set payload."""
    db = get_db()
    # 1) Load session + ensure existence.
    session_doc = await db[Collections.SEARCH_SESSIONS].find_one(
        {"id": session_id, "deleted_at": None}
    )
    if not session_doc:
        session_doc = await db[Collections.SEARCH_SESSIONS].find_one(
            {"id": session_id}
        )
    if not session_doc:
        return {
            "status": "ineligible_session_not_found",
            "eligibility_decision": "ineligible_session_not_found",
            "session_id": session_id,
        }

    # 2) Pull diagnostics-shaped payload from the session_doc.
    sdj = session_doc.get("session_diagnostics_json") or {}
    # Production sessions store the PICOTT payload under
    # ``picott_json``; the unit-test fixtures use ``picott_payload``.
    # Prefer the production key.
    picott_doc = session_doc.get("picott_json") or session_doc.get("picott_payload")
    decision, eligibility_payload = _decide_eligibility(
        session_doc=session_doc,
        diagnostics={"session_diagnostics_json": sdj,
                     "picott": picott_doc,
                     "anchors": {"primary_anchors": session_doc.get("primary_anchors") or session_doc.get("primary_clinical_anchors_json")}},
    )

    # 3) Always create a benchmark_run document so admin can see
    # the decision trace, even when ineligible.
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=30)
    benchmark_run_id = str(uuid.uuid4())
    original_question = (session_doc.get("sanitized_question")
                         or session_doc.get("original_question") or "")
    normalized_question = (session_doc.get("normalized_question") or original_question)
    question_hash = hashlib.sha256(
        normalized_question.encode("utf-8")
    ).hexdigest()

    sar_block = sdj.get("semantic_anchor_repair") or {}
    candidate = sar_block.get("semantic_repair_candidate") if isinstance(sar_block, dict) else None
    if not isinstance(candidate, dict):
        candidate = None

    # Phase M3.1.1 — A-arm DISPLAY shape now goes through the shared
    # serializer.
    # Phase M3.1.2 — the metric-input ``a_queries`` argument fed into
    # ``compute_metrics`` now also draws from the canonical serializer's
    # ``boolean_queries`` list. This makes ``query_diff`` a true
    # A-vs-B delta (previously one-sided because the legacy
    # ``queries_by_family`` field never existed on the real schema).
    # Article-based metrics (overlap@10, new_direct_hits@10, etc.) are
    # unaffected because they only consume A/B top-10 PMID sets.
    a_arm_display = serialize_a_arm_from_session(session_doc)
    a_arm_cqm = a_arm_display["cqm"]
    a_arm_branch_query_plan = a_arm_display["branch_query_plan"]
    a_arm_picott_view = a_arm_display["picott_view"]
    a_arm_queries = a_arm_display["boolean_queries"]

    # Pull A-arm top-10 from search_session_articles (admin read-only).
    a_top10: list[dict[str, Any]] = []
    try:
        cursor = (
            db[Collections.SEARCH_SESSION_ARTICLES]
            .find({"search_session_id": session_id})
            .sort([("overall_rank", 1), ("created_at", 1)])
            .limit(TOP_N)
        )
        async for ssa in cursor:
            art_id = ssa.get("article_id")
            if not art_id:
                continue
            art = await db[Collections.ARTICLES].find_one(
                {"id": art_id},
                {"pmid": 1, "title": 1, "journal": 1,
                 "publication_year": 1, "source_ids_json": 1,
                 "doi": 1},
            )
            if not art:
                continue
            a_top10.append({
                "pmid": (art.get("pmid") or "").strip(),
                "source_id": (art.get("doi") or art.get("pmid") or "").strip(),
                "title": (art.get("title") or "").strip()[:300],
                "journal": (art.get("journal") or "").strip()[:200],
                "year": str(art.get("publication_year") or "").strip(),
                "source": (
                    "pubmed" if art.get("pmid")
                    else (next(iter((art.get("source_ids_json") or {}).keys()), "") or "")
                ),
                "lane": ssa.get("lane"),
                "directness": ssa.get("directness"),
            })
    except Exception:  # noqa: BLE001
        pass

    # 4) Eligible path: build B-arm + comparison retrieval + metrics.
    b_arm_payload: dict[str, Any] = {
        "repair_candidate": None,
        "compiler_input": None,
        "boolean_queries": {"pubmed": [], "epmc": []},
        "top10": [],
        "per_source_latency_ms": {"pubmed": [], "epmc": []},
        "per_source_error": {"pubmed": [], "epmc": []},
    }
    metrics: dict[str, Any] = {}
    timeout_status = "none"
    fallback_status = "none"

    if decision == "eligible" and candidate is not None:
        b_queries = compile_b_arm_queries(candidate)
        b_arm_payload["repair_candidate"] = candidate
        b_arm_payload["compiler_input"] = {
            "primary_intent": candidate.get("primary_intent"),
            "frameworks": candidate.get("frameworks") or [],
            "required_anchors": candidate.get("required_anchors") or [],
            "slots": candidate.get("slots") or {},
            "semantic_repair_reason": candidate.get("semantic_repair_reason"),
        }
        b_arm_payload["boolean_queries"] = b_queries
        b_top10, per_lat, per_err, t_status, f_status = await _b_arm_retrieve(b_queries)
        b_arm_payload["top10"] = b_top10
        b_arm_payload["per_source_latency_ms"] = per_lat
        b_arm_payload["per_source_error"] = per_err
        timeout_status = t_status
        fallback_status = f_status

        b_query_strings = sorted(
            _query_set({"pubmed": b_queries["pubmed"], "epmc": b_queries["epmc"]})
        )
        metrics = compute_metrics(
            a_top10=a_top10,
            b_top10=b_top10,
            a_queries=a_arm_queries,
            b_queries=b_query_strings,
            candidate=candidate,
            deterministic_primary_anchors=session_doc.get("primary_anchors") or [],
            original_question=original_question,
            per_source_latencies=per_lat,
            per_source_errors=per_err,
        )

    # 5) Persist.
    route_owned = decision.startswith("route_owned_by_k4a_specialty_")
    doc = {
        "id": benchmark_run_id,
        "_id": benchmark_run_id,
        "created_at": now,
        "expires_at": expires_at,
        "triggered_by_admin_user_id": admin_user_id,
        "input_kind": "session_id",
        "search_session_id": session_id,
        "raw_question": None,
        "normalized_question": normalized_question[:2000],
        "question_hash": question_hash,
        "eligibility_decision": decision,
        "route_owned_by_k4a_specialty": bool(route_owned),
        "a_arm": {
            "cqm": a_arm_cqm,
            "boolean_queries": a_arm_queries,
            "branch_query_plan": a_arm_branch_query_plan,
            "picott_view": a_arm_picott_view,
            "top10": a_top10,
        },
        "b_arm": b_arm_payload,
        "metrics": metrics,
        "timeout_status": timeout_status,
        "fallback_status": fallback_status,
        "safety_flags": {"leak_scan_passed": True},  # validated below
    }
    # Run the leak scanner on the about-to-be-stored doc (excluding
    # safety_flags itself). If it fails, we still store but mark
    # the flag false — the admin can see the failure.
    leak_ok = leak_scan_passes({k: v for k, v in doc.items() if k != "safety_flags"})
    doc["safety_flags"]["leak_scan_passed"] = bool(leak_ok)
    try:
        await db[Collections.M3_1_BENCHMARK_RUNS].insert_one(dict(doc))
    except Exception:  # noqa: BLE001
        logger.exception("m3_1_benchmark_run insert failed")
    # Return the JSON-friendly view (strip Mongo _id) AND lazy-resolve
    # the A-arm from the live session so the POST response shape is
    # byte-identical to a subsequent GET on the same run.
    out = dict(doc)
    out.pop("_id", None)
    out["created_at"] = now.isoformat()
    out["expires_at"] = expires_at.isoformat()
    out = await _lazy_resolve_a_arm(out)
    return out


async def _lazy_resolve_a_arm(d: dict) -> dict:
    """Phase M3.1.1 — re-project the A-arm display from the live
    session document at GET time, so any field-mapping bug in older
    persisted ``a_arm`` blobs is corrected on read. Adds a closed-set
    ``a_arm_source`` marker so the admin knows whether the displayed
    A-arm was freshly resolved or fell back to the stored blob.

    Phase M3.1.2 — also lazy-recompute the query-dependent metric
    fields (``query_diff`` and ``literal_full_phrase_anchor_removed``)
    using the canonical A-arm boolean queries from the serializer.
    Article-based metrics (``overlap_at_10``, ``new_direct_hits_at_10``,
    ``wrong_anchor_rate_at_10``, ``source_error_rate``, latencies,
    ``required_anchor_coverage``, ``generic_only_anchor_absent``) are
    preserved verbatim from the stored doc because their inputs are
    frozen at create time. A closed-set ``metrics_source`` marker
    surfaces ``"live_recomputed"`` vs ``"stored"`` to mirror
    ``a_arm_source``.

    Read-only. No mutation of the stored Mongo document. No LLM
    calls. No source calls.
    """
    db = get_db()
    session_id = d.get("search_session_id")
    if not session_id:
        d["a_arm_source"] = "stored"
        d["metrics_source"] = "stored"
        return d
    try:
        sess = await db[Collections.SEARCH_SESSIONS].find_one(
            {"id": session_id},
            {
                "_id": 0,
                "clinical_question_map_json": 1,
                "picott_json": 1,
                "primary_clinical_anchors_json": 1,
                "original_question": 1,
                "sanitized_question": 1,
            },
        )
    except Exception:  # noqa: BLE001
        sess = None
    if not sess:
        d["a_arm_source"] = "stored"
        d["metrics_source"] = "stored"
        return d
    fresh = serialize_a_arm_from_session(sess)
    stored_a = d.get("a_arm") or {}
    # Re-project display fields. Preserve the stored top10 (we never
    # re-run retrieval at GET) and the legacy flat ``boolean_queries``
    # input that was fed to ``compute_metrics`` at create time
    # (storing it under a clearer key, ``metric_input_queries``, so
    # operator audit can correlate stored metrics with the queries
    # they were computed against).
    d["a_arm"] = {
        "cqm": fresh["cqm"],
        "boolean_queries": fresh["boolean_queries"],
        "branch_query_plan": fresh["branch_query_plan"],
        "picott_view": fresh["picott_view"],
        "anchors": fresh["anchors"],
        "top10": stored_a.get("top10") or [],
        "metric_input_queries": stored_a.get("boolean_queries") or [],
    }
    d["a_arm_source"] = "live_resolved"

    # Phase M3.1.2 — lazy recompute of query-dependent metrics.
    stored_metrics = dict(d.get("metrics") or {})
    a_queries_live = list(fresh["boolean_queries"])
    b_arm = d.get("b_arm") or {}
    b_bq = b_arm.get("boolean_queries") or {}
    # b_arm stores boolean_queries as a dict of {pubmed:[...], epmc:[...]}
    # where each entry is a dict with query_string. Flatten to a sorted
    # unique list of strings, mirroring how the orchestrator computes
    # b_queries for compute_metrics at create time.
    b_queries_live = sorted(_query_set(b_bq))
    if a_queries_live or b_queries_live or stored_metrics:
        a_set = set(a_queries_live)
        b_set = set(b_queries_live)
        new_query_diff = {
            "added": sorted(b_set - a_set),
            "removed": sorted(a_set - b_set),
            "common": sorted(a_set & b_set),
            "n_added": len(b_set - a_set),
            "n_removed": len(a_set - b_set),
        }
        # Recompute the literal-full-phrase signal too — it depends on
        # whether a 4+ word n-gram from the original question appears
        # in A queries only.
        original_question = sess.get("original_question") or sess.get("sanitized_question") or ""

        def _norm_local(s: Any) -> str:
            return (s or "").strip().lower()

        def _strip_wh_lead(q: str) -> str:
            q2 = q.strip(" ?.!\"'").lower()
            wh_leads = (
                "what are the ", "what is the ", "what are ", "what is ",
                "what's the ", "what's ", "does a ", "does the ", "does ",
                "how does the ", "how does ", "how do ", "how is the ",
                "how is ", "how are ", "why does ", "why is ", "why are ",
                "when does ", "when is ", "are there ", "is there ",
                "should ", "can a ", "can ", "in patients with ",
            )
            for lead in wh_leads:
                if q2.startswith(lead):
                    q2 = q2[len(lead):]
                    break
            return q2.strip()

        stem_tokens = _strip_wh_lead(original_question).split()
        literal_in_a = literal_in_b = False
        if len(stem_tokens) >= 4:
            ngrams = []
            for n in range(min(len(stem_tokens), 8), 3, -1):
                for i in range(0, len(stem_tokens) - n + 1):
                    ngrams.append(" ".join(stem_tokens[i:i + n]))
            a_low = [_norm_local(q) for q in a_set]
            b_low = [_norm_local(q) for q in b_set]
            for ng in ngrams:
                if any(ng in q for q in a_low):
                    literal_in_a = True
                    if any(ng in q for q in b_low):
                        literal_in_b = True
                        break
                    else:
                        break
        new_literal_removed = bool(literal_in_a and not literal_in_b)

        # Update only the query-dependent fields; preserve everything
        # else (overlap_at_10, new_direct_hits_at_10, source_error_rate,
        # latencies, required_anchor_coverage, generic_only_anchor_absent,
        # wrong_anchor_rate_at_10).
        new_metrics = dict(stored_metrics)
        new_metrics["query_diff"] = new_query_diff
        new_metrics["literal_full_phrase_anchor_removed"] = new_literal_removed
        d["metrics"] = new_metrics
        d["metrics_source"] = "live_recomputed"
    else:
        d["metrics_source"] = "stored"
    return d


async def get_recent_benchmark_runs(
    *,
    limit: int = 25,
) -> list[dict[str, Any]]:
    db = get_db()
    cursor = (
        db[Collections.M3_1_BENCHMARK_RUNS]
        .find({}, {"_id": 0})
        .sort([("created_at", -1)])
        .limit(int(limit))
    )
    out: list[dict[str, Any]] = []
    async for d in cursor:
        ca = d.get("created_at")
        ea = d.get("expires_at")
        if isinstance(ca, datetime):
            d["created_at"] = ca.isoformat()
        if isinstance(ea, datetime):
            d["expires_at"] = ea.isoformat()
        d = await _lazy_resolve_a_arm(d)
        out.append(d)
    return out


async def get_benchmark_run_by_id(run_id: str) -> Optional[dict[str, Any]]:
    db = get_db()
    d = await db[Collections.M3_1_BENCHMARK_RUNS].find_one(
        {"id": run_id}, {"_id": 0},
    )
    if not d:
        return None
    ca = d.get("created_at")
    ea = d.get("expires_at")
    if isinstance(ca, datetime):
        d["created_at"] = ca.isoformat()
    if isinstance(ea, datetime):
        d["expires_at"] = ea.isoformat()
    d = await _lazy_resolve_a_arm(d)
    return d


__all__ = [
    "ELIGIBILITY_DECISIONS",
    "TIMEOUT_STATUSES",
    "FALLBACK_STATUSES",
    "compile_b_arm_queries",
    "compute_metrics",
    "leak_scan_passes",
    "run_benchmark_for_session",
    "get_recent_benchmark_runs",
    "get_benchmark_run_by_id",
    "serialize_a_arm_from_session",
    "M_REPAIR_TIMEOUT_S",
    "B_RETRIEVAL_TOTAL_TIMEOUT_S",
    "PER_SOURCE_TIMEOUT_S",
    "TOP_N",
]
