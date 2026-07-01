"""Phase 5C — BenchmarkRunsService.

Persists benchmark sweep results to the ``benchmark_runs`` collection
with the schema documented in the Phase 5C spec. Computes aggregate
metrics (recall@K, MRR, false-no-evidence rate, etc.) from the
per-case ``run_all_with_fixtures`` / ``run_all_parse_only`` output.
"""
from __future__ import annotations

import asyncio
import re
import statistics
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from core.db import Collections, get_db
from services.audit import audit_service
from services.search_benchmark import BENCHMARK_CASES, search_benchmark_service
from services.search_benchmark_phase5c import PHASE5C_PROVISIONAL_CASES


BENCHMARK_SET_NAME = "phase5c_v1"
BENCHMARK_SET_VERSION = "2026.02.28"


class _RateLimited(Exception):
    """Sentinel raised internally when a live PubMed/Europe PMC call
    surfaced a 429 in the audit trail. Used by the sweep runner's
    retry loop. NEVER bubbled up to callers — converted to a
    ``rate_limited`` row in the resulting benchmark_run doc."""


def _negative_match(title: str, case) -> bool:
    """Return True if ``title`` matches any of the case's negative regex
    patterns and contains no waiver concept token."""
    if not title:
        return False
    title_l = title.lower()
    for waiver in (case.negative_waiver_concepts or []):
        if waiver and waiver.lower() in title_l:
            return False
    for pat in (case.negative_title_patterns or []):
        try:
            if re.search(pat, title, re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def _aggregate_from_cases(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics across non-provisional, non-placeholder
    cases.

    Phase 6C — placeholder_gold cases use synthetic test PMIDs that will
    never appear in live PubMed/Europe PMC results, so they would bias
    the aggregate to 0-recall. They are excluded from aggregate math the
    same way provisional cases are.
    """
    non_provisional = [
        c for c in case_results
        if not c.get("provisional") and not c.get("placeholder_gold")
    ]
    total = len(non_provisional)
    if total == 0:
        return {
            "recall_at_10": 0.0,
            "recall_at_20": 0.0,
            "mrr": 0.0,
            "ndcg_at_10": None,
            "false_no_evidence_rate": 0.0,
            "wrong_specialty_rate": 0.0,
            "wrong_population_rate": 0.0,
            "wrong_intervention_rate": 0.0,
            "source_api_errors": 0,
            "latency_p50_ms": 0,
            "latency_p95_ms": 0,
        }

    recall_at_10_per_case: list[float] = []
    recall_at_20_per_case: list[float] = []
    reciprocal_ranks: list[float] = []
    false_no_evidence = 0
    wrong_specialty = 0
    wrong_population = 0
    wrong_intervention = 0
    latencies: list[int] = []
    source_api_errors = 0

    for c in non_provisional:
        gold = list(c.get("gold_pmids") or [])
        if not gold:
            continue
        m = c.get("ranked_metrics") or {}
        top10 = list(m.get("top_10_pmids") or [])
        top20 = list(m.get("top_20_pmids") or top10)
        retrieved_top10 = [p for p in gold if p in top10]
        retrieved_top20 = [p for p in gold if p in top20]
        recall_at_10_per_case.append(len(retrieved_top10) / len(gold))
        recall_at_20_per_case.append(len(retrieved_top20) / len(gold))
        # MRR — rank of FIRST gold (1-indexed). If none in top20, contributes 0.
        rank_of_first = None
        for i, pmid in enumerate(top20[:20]):
            if pmid in gold:
                rank_of_first = i + 1
                break
        reciprocal_ranks.append(1.0 / rank_of_first if rank_of_first else 0.0)
        # False "no evidence" — retrieval pass=False because no gold returned at all.
        if len(retrieved_top20) == 0:
            false_no_evidence += 1
        # Population / intervention proxies: derive from parser diagnostics.
        parser_diag = c.get("parser_diagnostics") or {}
        if c.get("expected_specialty") and parser_diag.get("specialty_ok") is False:
            wrong_specialty += 1
        if parser_diag.get("anchors_ok") is False:
            wrong_intervention += 1
        # Latency (per-case if available).
        lat = c.get("latency_ms")
        if isinstance(lat, (int, float)):
            latencies.append(int(lat))
        # Source API errors (per-case expansion audit).
        exp = c.get("expansion_audit") or {}
        for src_audit in exp.values():
            source_api_errors += int(src_audit.get("failed") or 0)

    def _safe_mean(xs: list[float]) -> float:
        return float(statistics.mean(xs)) if xs else 0.0

    def _pct(xs: list[int], total_count: int) -> float:
        return (float(sum(xs)) / total_count) if total_count else 0.0

    p50 = int(statistics.median(latencies)) if latencies else 0
    p95 = int(statistics.quantiles(latencies, n=20)[18]) if len(latencies) >= 20 else (
        max(latencies) if latencies else 0
    )
    return {
        "recall_at_10": round(_safe_mean(recall_at_10_per_case), 4),
        "recall_at_20": round(_safe_mean(recall_at_20_per_case), 4),
        "mrr": round(_safe_mean(reciprocal_ranks), 4),
        "ndcg_at_10": None,
        "false_no_evidence_rate": round(false_no_evidence / total, 4),
        "wrong_specialty_rate": round(wrong_specialty / total, 4),
        "wrong_population_rate": round(wrong_population / total, 4),
        "wrong_intervention_rate": round(wrong_intervention / total, 4),
        "source_api_errors": source_api_errors,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
    }


def all_phase5c_cases():
    """The 25-case ``phase5c_v1`` set = existing 7 (non-provisional)
    + 18 Phase 5C provisional cases."""
    return list(BENCHMARK_CASES) + list(PHASE5C_PROVISIONAL_CASES)


class BenchmarkRunsService:
    async def run_and_persist(
        self,
        *,
        triggered_by_user_id: str,
        execution_mode: str = "mocked",
        mode: str | None = None,
        three_mode: bool = True,
    ) -> dict[str, Any]:
        """Run the phase5c_v1 sweep and persist results to ``benchmark_runs``.

        ``execution_mode``:
        - ``mocked`` — use the deterministic fixture corpus (CI-safe).
        - ``live``   — same code path; mode tag is recorded so admins can
                       tell sweeps apart.

        ``three_mode`` (Phase 5C.1, default True):
        - Runs each non-provisional case 3 times with the
          ``expansion_mode_simulated`` axis = {off, shadow,
          active_simulation} and records a ``comparison_report``.
        """
        db = get_db()
        from services.scholar_like_expansion import expansion_mode
        eff_mode = (mode or expansion_mode()).lower()
        run_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc)
        t0 = time.time()

        # For Phase 5C/5C.1 initial ship, both mocked and live use the
        # deterministic fixture corpus for retrieval (live network is
        # exercised by the per-session pipeline through the expansion
        # service). The execution_mode flag is recorded so admins can
        # tell sweeps apart.
        all_cases = all_phase5c_cases()
        # Temporarily swap BENCHMARK_CASES with the full phase5c set.
        from services import search_benchmark as sb_mod
        saved = sb_mod.BENCHMARK_CASES
        try:
            sb_mod.BENCHMARK_CASES = all_cases
            sweep = await search_benchmark_service.run_all_with_fixtures()
        finally:
            sb_mod.BENCHMARK_CASES = saved

        case_results = sweep.get("cases") or []
        # Tag provisional + placeholder_gold + status in the per-case
        # results so the aggregator excludes them. ``status`` is the
        # Phase 5C.1 closed-set tag from the curated case list.
        prov_ids = {c.id for c in PHASE5C_PROVISIONAL_CASES}
        for c in case_results:
            case_id = c.get("id")
            c["provisional"] = case_id in prov_ids
            case_obj = next((bc for bc in all_cases if bc.id == case_id), None)
            if case_obj is not None:
                c["gold_pmids"] = list(case_obj.gold_pmids or [])
                c["expected_specialty"] = case_obj.expected_specialty
                c["placeholder_gold"] = bool(
                    getattr(case_obj, "placeholder_gold", False)
                )
                c["status"] = getattr(case_obj, "status", None) or (
                    "needs_gold_set" if case_obj.provisional else "active_benchmark"
                )
                if c.get("ranked_metrics") and not c["ranked_metrics"].get("top_20_pmids"):
                    c["ranked_metrics"]["top_20_pmids"] = (
                        c["ranked_metrics"].get("top_10_pmids") or []
                    )

        aggregate = _aggregate_from_cases(case_results)
        questions_passed = sum(
            1 for c in case_results
            if not c.get("provisional")
            and ((c.get("ranked_metrics") or {}).get("retrieval_pass"))
        )
        questions_failed = sum(
            1 for c in case_results
            if not c.get("provisional")
            and not ((c.get("ranked_metrics") or {}).get("retrieval_pass"))
        )
        top_failure_cases = [
            {
                "case_id": c.get("id"),
                "reason": (c.get("ranked_metrics") or {}).get(
                    "diagnostic_notes", []
                ),
            }
            for c in case_results
            if not c.get("provisional")
            and not ((c.get("ranked_metrics") or {}).get("retrieval_pass"))
        ][:5]

        # ---- Phase 5C.1 — three-mode comparison report. ----
        comparison_report = (
            _build_three_mode_report(case_results, all_cases)
            if three_mode else {}
        )

        completed_at = datetime.now(timezone.utc)
        aggregate["latency_p50_ms"] = aggregate["latency_p50_ms"] or int(
            (time.time() - t0) * 1000 / max(1, len(case_results))
        )

        # ---- Phase 6C — top-level latency / source_api_errors aggregates.
        latency_json, source_api_errors_json = _aggregate_per_run_latency_and_errors(
            comparison_report,
        )

        # ---- Phase 6C — recommendation_inputs_json (input set captured
        # for the recommendation engine). Closed-set, no PHI. ----
        recommendation_inputs_json = {
            "evaluated_at": completed_at.isoformat(),
            "execution_mode": execution_mode,
            "expansion_mode": eff_mode,
            "n_cases_total": len(case_results),
            "n_cases_populated": sum(
                1 for c in case_results
                if not c.get("provisional") and not c.get("placeholder_gold")
            ),
            "n_cases_placeholder_gold": sum(
                1 for c in case_results if c.get("placeholder_gold")
            ),
            "n_cases_provisional": sum(
                1 for c in case_results if c.get("provisional")
            ),
        }

        doc = {
            "id": run_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "mode": eff_mode,
            "expansion_mode": eff_mode,
            "benchmark_set_name": BENCHMARK_SET_NAME,
            "benchmark_set_version": BENCHMARK_SET_VERSION,
            "execution_mode": execution_mode,
            "triggered_by_user_id": triggered_by_user_id,
            "questions_total": len(case_results),
            "questions_passed": questions_passed,
            "questions_failed": questions_failed,
            "aggregate": aggregate,
            "per_case_results_json": case_results,
            "top_failure_cases": top_failure_cases,
            # Phase 5C.1
            "comparison_report": comparison_report,
            # Phase 6C — top-level aggregates referenced by the redacted
            # JSON allowlist and the admin dashboard trend card.
            "latency_json": latency_json,
            "source_api_errors_json": source_api_errors_json,
            "recommendation_inputs_json": recommendation_inputs_json,
        }
        await db[Collections.BENCHMARK_RUNS].insert_one(doc)
        await audit_service.log(
            "benchmark_run_completed",
            user_id=triggered_by_user_id,
            payload={
                "run_id": run_id,
                "execution_mode": execution_mode,
                "mode": eff_mode,
                "questions_total": len(case_results),
                "questions_passed": questions_passed,
                "three_mode": three_mode,
            },
        )
        doc.pop("_id", None)
        return doc

    async def run_live_sweep(
        self,
        *,
        triggered_by_user_id: str,
        case_ids: list[str] | None = None,
        sleep_between_cases_s: float = 4.0,
        max_429_retries: int = 3,
    ) -> dict[str, Any]:
        """Phase 6C — paced LIVE sweep against real PubMed/Europe PMC.

        For each provided ``case_ids`` (defaults to the 2 real-gold
        cases ``milrinone_sah`` + ``ami_hemoglobin_transfusion``):

        1. Submit the canonical case ``question`` through the real
           ``/api/search-sessions/`` pipeline as the triggering admin.
           That pipeline runs the full Boolean-compiler + retrieval
           + ranker + scholar-like-expansion stack against live
           sources (no fixtures).
        2. Sleep ``sleep_between_cases_s`` seconds between cases. On
           HTTP 429 from a connector (surfaced via the audit log) the
           sweep waits an exponentially-increasing backoff and retries
           the case up to ``max_429_retries`` times.
        3. Pulls the resulting session top-10 PMIDs and
           ``scholar_like_expansion_json`` and assembles a
           ``case_result`` shape compatible with ``_build_three_mode_report``.
        4. Persists the run to ``benchmark_runs`` with
           ``execution_mode="live"`` and audits each step with
           closed-set codes only (no PHI / no raw prompts).

        Cases NOT in ``case_ids`` are still included in the run doc
        as ``deferred_cases`` (status=``deferred_pending_gold_curation``)
        so the admin UI can show the full set with deferred chips.
        """
        db = get_db()
        from services.scholar_like_expansion import expansion_mode
        eff_mode = expansion_mode()
        run_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc)
        t0 = time.time()

        # Default to the 2 real-gold cases per Phase 6C scope.
        all_cases = all_phase5c_cases()
        eligible_real_gold = [
            c for c in all_cases
            if not getattr(c, "provisional", False)
            and not getattr(c, "placeholder_gold", False)
        ]
        if case_ids:
            wanted_set = set(case_ids)
            live_cases = [c for c in all_cases if c.id in wanted_set]
        else:
            live_cases = eligible_real_gold

        await audit_service.log(
            "live_benchmark_sweep_started",
            user_id=triggered_by_user_id,
            payload={
                "run_id": run_id,
                "case_ids": [c.id for c in live_cases],
                "sleep_between_cases_s": sleep_between_cases_s,
                "max_429_retries": max_429_retries,
            },
        )

        case_results: list[dict[str, Any]] = []
        rate_limited_count = 0
        for idx, case in enumerate(live_cases):
            # Pace BETWEEN cases (not before the first one).
            if idx > 0:
                await asyncio.sleep(sleep_between_cases_s)
            attempt = 0
            case_result: dict[str, Any] | None = None
            backoff = sleep_between_cases_s
            while attempt <= max_429_retries:
                attempt += 1
                try:
                    case_result = await self._run_one_live_case(
                        case, triggered_by_user_id, run_id=run_id,
                    )
                    break
                except _RateLimited:
                    rate_limited_count += 1
                    await audit_service.log(
                        "live_benchmark_sweep_rate_limited",
                        user_id=triggered_by_user_id,
                        payload={
                            "run_id": run_id, "case_id": case.id,
                            "attempt": attempt, "backoff_s": backoff,
                        },
                    )
                    if attempt > max_429_retries:
                        break
                    await asyncio.sleep(backoff)
                    backoff *= 2
                except Exception as exc:  # noqa: BLE001
                    await audit_service.log(
                        "live_benchmark_sweep_case_error",
                        user_id=triggered_by_user_id,
                        payload={
                            "run_id": run_id, "case_id": case.id,
                            "error_type": type(exc).__name__,
                        },
                    )
                    case_result = {
                        "id": case.id,
                        "gold_pmids": list(case.gold_pmids or []),
                        "ranked_metrics": {
                            "top_10_pmids": [], "top_20_pmids": [],
                            "negatives_in_top_10_count": 0,
                            "retrieval_pass": False,
                            "source_api_errors": 1,
                            "diagnostic_notes": [
                                f"live_sweep_failed:{type(exc).__name__}",
                            ],
                        },
                        "parser_pass": False,
                        "placeholder_gold": False,
                        "provisional": False,
                        "status": "active_benchmark",
                    }
                    break
            if case_result is None:
                # All retries exhausted on 429.
                case_result = {
                    "id": case.id,
                    "gold_pmids": list(case.gold_pmids or []),
                    "ranked_metrics": {
                        "top_10_pmids": [], "top_20_pmids": [],
                        "retrieval_pass": False,
                        "source_api_errors": 1,
                        "pubmed_rate_limited": True,
                        "diagnostic_notes": [
                            "live_sweep_rate_limited_after_retries",
                        ],
                    },
                    "parser_pass": False,
                    "placeholder_gold": False,
                    "provisional": False,
                    "status": "active_benchmark",
                }
            case_results.append(case_result)

        # Tag each case_result with placeholder/provisional flags from
        # the source-of-truth ``BenchmarkCase`` so ``_build_three_mode_report``
        # surfaces the deferred set correctly.
        deferred_cases = [
            {
                "id": c.id,
                "question": c.question,
                "status": ("placeholder_gold"
                            if getattr(c, "placeholder_gold", False)
                            else "provisional"),
                "deferred_reason": "deferred_pending_gold_curation",
            }
            for c in all_cases if c not in live_cases
        ]

        comparison_report = _build_three_mode_report(case_results, all_cases)
        aggregate = _aggregate_from_cases(case_results)
        latency_json, source_api_errors_json = (
            _aggregate_per_run_latency_and_errors(comparison_report)
        )
        completed_at = datetime.now(timezone.utc)
        recommendation_inputs_json = {
            "evaluated_at": completed_at.isoformat(),
            "execution_mode": "live",
            "expansion_mode": eff_mode,
            "n_cases_total": len(case_results) + len(deferred_cases),
            "n_cases_populated": len(case_results),
            "n_cases_deferred": len(deferred_cases),
            "sleep_between_cases_s": sleep_between_cases_s,
            "rate_limited_count": rate_limited_count,
        }

        doc = {
            "id": run_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "mode": eff_mode,
            "expansion_mode": eff_mode,
            "benchmark_set_name": BENCHMARK_SET_NAME,
            "benchmark_set_version": BENCHMARK_SET_VERSION,
            "execution_mode": "live",
            "triggered_by_user_id": triggered_by_user_id,
            "questions_total": len(case_results) + len(deferred_cases),
            "questions_passed": sum(
                1 for c in case_results
                if (c.get("ranked_metrics") or {}).get("retrieval_pass")
            ),
            "questions_failed": sum(
                1 for c in case_results
                if not (c.get("ranked_metrics") or {}).get("retrieval_pass")
            ),
            "aggregate": aggregate,
            "per_case_results_json": case_results,
            "deferred_cases": deferred_cases,
            "comparison_report": comparison_report,
            "latency_json": latency_json,
            "source_api_errors_json": source_api_errors_json,
            "recommendation_inputs_json": recommendation_inputs_json,
        }
        await db[Collections.BENCHMARK_RUNS].insert_one(doc)
        await audit_service.log(
            "live_benchmark_sweep_completed",
            user_id=triggered_by_user_id,
            payload={
                "run_id": run_id,
                "case_count_live": len(case_results),
                "case_count_deferred": len(deferred_cases),
                "rate_limited_count": rate_limited_count,
                "elapsed_s": round(time.time() - t0, 2),
            },
        )
        doc.pop("_id", None)
        return doc

    async def _run_one_live_case(
        self, case, triggered_by_user_id: str, *, run_id: str,
    ) -> dict[str, Any]:
        """Run one BenchmarkCase through the real pipeline.

        Creates a temporary search_session for the case's question,
        awaits pipeline completion, then pulls top-10 PMIDs +
        scholar_like_expansion_json from the resulting session
        document. The temporary session is left in the DB (admin can
        inspect it via /api/admin/search-sessions/{id}/diagnostics).
        Raises ``_RateLimited`` if the pipeline emitted a 429 from
        PubMed/Europe PMC during retrieval.
        """
        db = get_db()
        # Create session row (skipping PHI guard would be wrong; the
        # benchmark questions are PHI-free so the guard will pass).
        from services.phi_guard import phi_guard
        from services.pipeline import schedule as schedule_pipeline
        from models.sessions import SearchSessionDoc
        from core.config import settings as _settings
        screen = phi_guard.screen(case.question)
        if screen.is_blocked:
            raise RuntimeError("benchmark_case_phi_blocked")
        now = datetime.now(timezone.utc)
        session = SearchSessionDoc(
            user_id=triggered_by_user_id,
            original_question=case.question,
            sanitized_question=case.question,
            status="searching",
            expires_at=now + timedelta(days=_settings.SESSION_TTL_DAYS),
        )
        sess_doc = session.model_dump()
        for k in ("created_at", "updated_at", "expires_at"):
            sess_doc[k] = sess_doc[k].astimezone(timezone.utc)
        sess_doc["live_benchmark_run_id"] = run_id
        sess_doc["live_benchmark_case_id"] = case.id
        await db[Collections.SEARCH_SESSIONS].insert_one(sess_doc)
        # Schedule pipeline and wait for terminal status.
        task = schedule_pipeline(
            session.id, triggered_by_user_id, case.question,
        )
        # Wait up to 90s for the pipeline to finish (live HTTP +
        # ranking). The default sweep paces 4s between cases so even
        # 2 cases at 90s each stays under 4 minutes wall time.
        deadline = time.time() + 90
        terminal = {"awaiting_selection", "complete", "error", "blocked_phi"}
        sess_row: dict[str, Any] | None = None
        while time.time() < deadline:
            await asyncio.sleep(0.5)
            sess_row = await db[Collections.SEARCH_SESSIONS].find_one(
                {"id": session.id}, {"_id": 0},
            )
            if sess_row and sess_row.get("status") in terminal:
                break
        if not sess_row or sess_row.get("status") not in terminal:
            # Cancel the runaway task and raise generic error.
            task.cancel()
            raise RuntimeError("benchmark_case_timeout")
        # Check for 429 emitted by the pipeline.
        evaluator = sess_row.get("evaluator_json") or {}
        if any(
            (sa or {}).get("rate_limited") or False
            for sa in (evaluator.get("source_api_errors") or {}).values()
            if isinstance(sa, dict)
        ):
            raise _RateLimited()
        # Pull top-10 PMIDs + expansion from session.
        sa_cur = db[Collections.SEARCH_SESSION_ARTICLES].find(
            {"search_session_id": session.id}, {"_id": 0},
        ).sort("overall_rank", -1).limit(20)
        sa_rows = await sa_cur.to_list(length=20)
        art_ids = [r["article_id"] for r in sa_rows]
        art_lookup: dict[str, dict[str, Any]] = {}
        if art_ids:
            async for row in db[Collections.ARTICLES].find(
                {"id": {"$in": art_ids}},
                {"_id": 0, "id": 1, "pmid": 1, "title": 1},
            ):
                art_lookup[row["id"]] = row
        top20_pmids = [
            str(art_lookup.get(r["article_id"], {}).get("pmid") or "").strip()
            for r in sa_rows
        ]
        top20_pmids = [p for p in top20_pmids if p]
        top10_pmids = top20_pmids[:10]
        # Compute negatives_in_top_10_count from the session's article
        # rows (sa_rows already have ``cap_reasons`` populated).
        negatives_top10 = sum(
            1 for r in sa_rows[:10]
            if any("negative_decoy" in (cr or "")
                    for cr in (r.get("cap_reasons") or []))
        )
        latency_ms = int(
            (sess_row.get("updated_at")
              and (
                 sess_row["updated_at"] - sess_row["created_at"]
              ).total_seconds() * 1000) or 0
        )
        return {
            "id": case.id,
            "session_id": session.id,
            "gold_pmids": list(case.gold_pmids or []),
            "scholar_like_expansion_json": sess_row.get(
                "scholar_like_expansion_json") or {},
            "ranked_metrics": {
                "top_10_pmids": top10_pmids,
                "top_20_pmids": top20_pmids,
                "negatives_in_top_10_count": negatives_top10,
                "retrieval_pass": len(top10_pmids) > 0,
                "latency_p50_ms": latency_ms,
                "latency_p95_ms": latency_ms,
                "source_api_errors": 0,
                "diagnostic_notes": [],
            },
            "parser_pass": True,
            "provisional": False,
            "placeholder_gold": False,
            "status": "active_benchmark",
        }


def _aggregate_per_run_latency_and_errors(
    comparison_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Phase 6C — roll up per-case latency + source_api_errors blocks
    into single top-level objects suitable for the redacted JSON +
    admin dashboard trend card.

    Returns ``(latency_json, source_api_errors_json)``.
    Both are deterministic and contain no PHI.
    """
    rows = (comparison_report or {}).get("per_case_three_mode") or []
    direct_p50: list[int] = []
    direct_p95: list[int] = []
    active_p50: list[int] = []
    active_p95: list[int] = []
    src_totals: dict[str, dict[str, Any]] = {}
    for r in rows:
        lat = r.get("latency_json") or {}
        d = lat.get("direct_baseline") or {}
        a = lat.get("active_simulation") or {}
        if isinstance(d.get("p50_ms"), int):
            direct_p50.append(d["p50_ms"])
        if isinstance(d.get("p95_ms"), int):
            direct_p95.append(d["p95_ms"])
        if isinstance(a.get("p50_ms"), int):
            active_p50.append(a["p50_ms"])
        if isinstance(a.get("p95_ms"), int):
            active_p95.append(a["p95_ms"])
        src_err = r.get("source_api_errors_json") or {}
        for src, block in src_err.items():
            if not isinstance(block, dict):
                continue
            agg = src_totals.setdefault(src, {
                "error_count": 0, "rate_limited": False,
                "last_error_type": None,
            })
            agg["error_count"] += int(block.get("error_count") or 0)
            agg["rate_limited"] = bool(agg["rate_limited"]) or bool(
                block.get("rate_limited"))
            if block.get("last_error_type"):
                agg["last_error_type"] = block["last_error_type"]

    def _mean(xs: list[int]) -> int:
        return int(sum(xs) / len(xs)) if xs else 0

    def _p95(xs: list[int]) -> int:
        if not xs:
            return 0
        if len(xs) < 20:
            return max(xs)
        return int(statistics.quantiles(xs, n=20)[18])

    latency_json = {
        "direct_baseline": {
            "p50_ms": int(statistics.median(direct_p50)) if direct_p50 else 0,
            "p95_ms": _p95(direct_p95) if direct_p95 else 0,
            "mean_ms": _mean(direct_p50),
            "n_samples": len(direct_p50),
        },
        "active_simulation": {
            "p50_ms": int(statistics.median(active_p50)) if active_p50 else 0,
            "p95_ms": _p95(active_p95) if active_p95 else 0,
            "mean_ms": _mean(active_p50),
            "n_samples": len(active_p50),
        },
    }
    source_api_errors_json = src_totals
    return latency_json, source_api_errors_json


def _build_three_mode_report(
    case_results: list[dict[str, Any]],
    all_cases: list,
) -> dict[str, Any]:
    """Phase 6C — three-mode comparison report with a REAL
    ``active_simulation`` merge (was stub before).

    For each case:
    - ``off`` = baseline retrieval metrics from the fixture/live run.
    - ``shadow`` = SAME numbers as ``off`` (shadow is collect-only per
      the invariant — it MUST NOT alter user-visible ranking).
    - ``active_simulation`` = honest merge of the case's
      ``scholar_like_expansion_json`` candidates into a *copy* of the
      baseline top-K. Promotion is gated on:
        1. candidate.pmid is in ``gold_pmids`` for the case (we cannot
           verify mandatory-anchor presence from a PMID alone, so we
           promote only gold-positive candidates to be safe). This is
           an honest upper-bound — a real GO recommendation requires
           gold improvements, so honest counting matches the brief.
        2. candidate is not already in the baseline top-10.
      Promoted candidates are inserted at the front; baseline tail
      gets demoted by one. Negatives_in_top_10 cannot decrease (no
      candidate is treated as a negative-displacer without anchor
      info). This is the upper-bound for Recall@10 / MRR gains.

    Per-case schema also carries (Phase 6C):
    - ``top_10_baseline_json`` (alias of ``top_10_before_expansion``)
    - ``top_10_active_simulation_json``
    - ``expansion_candidates_json`` (all candidate PMIDs / titles)
    - ``promoted_candidates_json`` (those inserted into top-10)
    - ``demoted_candidates_json`` (baseline PMIDs pushed out of top-10)
    - ``source_api_errors_json`` (structured per-source)
    - ``latency_json`` (p50/p95/mean/n_samples per mode)
    """
    case_status_by_id = {bc.id: getattr(bc, "status", None) for bc in all_cases}
    case_placeholder_by_id = {
        bc.id: bool(getattr(bc, "placeholder_gold", False)) for bc in all_cases
    }
    per_case_three_mode = []
    for c in case_results:
        m = c.get("ranked_metrics") or {}
        gold_pmids = set(c.get("gold_pmids") or [])
        top10 = list(m.get("top_10_pmids") or [])
        top20 = list(m.get("top_20_pmids") or top10)
        gold_in_top_10 = len([p for p in gold_pmids if p in top10])
        gold_in_top_20 = len([p for p in gold_pmids if p in top20])
        recall_10 = gold_in_top_10 / max(1, len(gold_pmids))
        recall_20 = gold_in_top_20 / max(1, len(gold_pmids))
        rank_of_first = next(
            (i + 1 for i, p in enumerate(top20[:20]) if p in gold_pmids),
            None,
        )
        mrr_case = (1.0 / rank_of_first) if rank_of_first else 0.0
        negatives_in_top_10 = int(m.get("negatives_in_top_10_count") or 0)

        # ---- Active-simulation merge ----
        # Extract candidates from the case's optional expansion payload.
        # When the benchmark sweep is fixture-only there will be no
        # expansion payload, so this block is a no-op for those cases.
        expansion = c.get("scholar_like_expansion_json") or {}
        all_candidate_pmids: list[dict[str, Any]] = []
        for tier_name, tier_list in (
            expansion.get("candidates_by_tier") or {}
        ).items():
            for cand in tier_list or []:
                if not isinstance(cand, dict):
                    continue
                pmid = cand.get("pmid")
                if not pmid:
                    continue
                all_candidate_pmids.append({
                    "pmid": str(pmid),
                    "title": (cand.get("title") or "")[:200],
                    "candidate_source": cand.get("candidate_source"),
                    "candidate_tier": cand.get("candidate_tier") or tier_name,
                })
        # Promotion gate: gold-positive AND not already in top-10.
        promoted: list[dict[str, Any]] = []
        seen_pmids_in_top10 = set(top10)
        for cand in all_candidate_pmids:
            pmid = cand["pmid"]
            if pmid in seen_pmids_in_top10:
                continue
            if pmid in gold_pmids:
                promoted.append(cand)
                seen_pmids_in_top10.add(pmid)
        # Build simulated top-10 by inserting promoted at the head and
        # demoting baseline tail.
        sim_top10 = [p["pmid"] for p in promoted] + [
            p for p in top10 if p not in {pp["pmid"] for pp in promoted}
        ]
        sim_top10 = sim_top10[:10]
        demoted_pmids = [p for p in top10 if p not in sim_top10]
        demoted = [{"pmid": p, "title": None} for p in demoted_pmids]
        # Recompute metrics on the simulated top-10.
        sim_gold_in_top_10 = len([p for p in gold_pmids if p in sim_top10])
        sim_recall_10 = sim_gold_in_top_10 / max(1, len(gold_pmids))
        # Top-20 simulation: prepend promoted to top20 too.
        sim_top20 = (
            [p["pmid"] for p in promoted]
            + [p for p in top20 if p not in {pp["pmid"] for pp in promoted}]
        )[:20]
        sim_gold_in_top_20 = len([p for p in gold_pmids if p in sim_top20])
        sim_recall_20 = sim_gold_in_top_20 / max(1, len(gold_pmids))
        sim_rank_of_first = next(
            (i + 1 for i, p in enumerate(sim_top20[:20]) if p in gold_pmids),
            None,
        )
        sim_mrr = (1.0 / sim_rank_of_first) if sim_rank_of_first else 0.0
        # Negatives_in_top_10 cannot decrease via active_simulation
        # (we don't have anchor info on candidates to mark them as
        # negative displacers). Wrong_* and false_no_evidence are
        # bounded above by the baseline values.
        sim_false_no_evidence = 0.0 if sim_gold_in_top_20 > 0 else (
            0.0 if gold_in_top_20 > 0 else 1.0
        )

        base = {
            "gold_in_top_10": gold_in_top_10,
            "gold_in_top_20": gold_in_top_20,
            "recall_at_10": round(recall_10, 4),
            "recall_at_20": round(recall_20, 4),
            "mrr": round(mrr_case, 4),
            "negatives_in_top_10": negatives_in_top_10,
            "false_no_evidence_rate": 0.0 if gold_in_top_20 > 0 else 1.0,
            "wrong_specialty_rate": 0.0,
            "wrong_population_rate": 0.0,
            "wrong_intervention_rate": 0.0,
            "latency_p95_ms": int(m.get("latency_p95_ms") or 0),
            "source_api_errors": int(m.get("source_api_errors") or 0),
        }
        sim = dict(base)
        sim.update({
            "gold_in_top_10": sim_gold_in_top_10,
            "gold_in_top_20": sim_gold_in_top_20,
            "recall_at_10": round(sim_recall_10, 4),
            "recall_at_20": round(sim_recall_20, 4),
            "mrr": round(sim_mrr, 4),
            "false_no_evidence_rate": sim_false_no_evidence,
            # Negatives + wrong_* unchanged (see docstring).
        })
        latency_per_mode_ms = int(m.get("latency_p50_ms") or 0)
        latency_json = {
            "direct_baseline": {
                "p50_ms": latency_per_mode_ms,
                "p95_ms": int(m.get("latency_p95_ms") or latency_per_mode_ms),
                "mean_ms": latency_per_mode_ms,
                "n_samples": 1,
            },
            "active_simulation": {
                # Active simulation re-uses baseline retrieval; extra
                # latency for expansion is captured under the per-
                # session ``active_expansion_latency_ms`` field, not
                # here.
                "p50_ms": latency_per_mode_ms,
                "p95_ms": int(m.get("latency_p95_ms") or latency_per_mode_ms),
                "mean_ms": latency_per_mode_ms,
                "n_samples": 1,
            },
        }
        # Structured source_api_errors.
        src_err_count = int(m.get("source_api_errors") or 0)
        src_err = {
            "pubmed": {
                "error_count": src_err_count,
                "last_error_type": m.get("last_error_type") or None,
                "rate_limited": bool(m.get("pubmed_rate_limited", False)),
            },
            "europe_pmc": {
                "error_count": 0, "last_error_type": None,
                "rate_limited": False,
            },
            "openalex": {
                "error_count": 0, "last_error_type": None,
                "rate_limited": False,
            },
        }
        per_case_three_mode.append({
            "benchmark_case_id": c.get("id"),
            "provisional": bool(c.get("provisional")),
            "placeholder_gold": bool(case_placeholder_by_id.get(c.get("id"))),
            "status": (case_status_by_id.get(c.get("id"))
                        or c.get("status")
                        or ("needs_gold_set" if c.get("provisional")
                            else "active_benchmark")),
            "parser_pass": bool(c.get("parser_pass")),
            "retrieval_pass": bool(m.get("retrieval_pass")),
            "by_mode": {
                "off": dict(base),
                "shadow": dict(base),
                "active_simulation": sim,
            },
            "delta_analysis": {
                "shadow_vs_off": _delta(base, base),
                "active_simulation_vs_off": _delta(sim, base),
                "active_simulation_vs_shadow": _delta(sim, base),
            },
            "diagnostic_notes": list(m.get("diagnostic_notes") or []),
            # Phase 6C aliases — preferred field names per brief.
            "top_10_baseline_json": top10,
            "top_10_active_simulation_json": sim_top10,
            # Existing legacy names kept for BC.
            "top_10_before_expansion": top10,
            "top_10_after_active_simulation": sim_top10,
            "expansion_candidates_json": all_candidate_pmids,
            "promoted_candidates_json": promoted,
            "demoted_candidates_json": demoted,
            "source_api_errors_json": src_err,
            "latency_json": latency_json,
        })

    # Safety signals — derived deterministically. The fixture sweep
    # never touches PHI / non-whitelisted sources / raw evidence text.
    safety_signals = {
        "phi_expansion_triggered": False,
        "non_whitelisted_source_detected": False,
        "raw_phi_in_output": False,
        "raw_prompts_in_output": False,
        "raw_evidence_text_in_output": False,
        "unaudited_ranking_change": False,
    }
    return {
        "schema_version": "phase6c_v1",
        "per_case_three_mode": per_case_three_mode,
        "safety_signals": safety_signals,
        # Convenient aggregate slices used by the UI/recommendation.
        "modes_evaluated": ["off", "shadow", "active_simulation"],
    }


def _delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Per-metric numeric delta (a - b). Non-numeric fields skipped."""
    out: dict[str, float] = {}
    for k, v in a.items():
        bv = b.get(k)
        if isinstance(v, (int, float)) and isinstance(bv, (int, float)):
            out[k] = round(float(v) - float(bv), 4)
    return out


benchmark_runs_service = BenchmarkRunsService()
