"""Phase 5C.1 — GO/NO-GO recommendation engine for Scholar-like expansion.

Computes a closed-set recommendation (``GO`` / ``CONTINUE_SHADOW`` /
``NO_GO``) from one or more ``benchmark_runs`` documents that contain a
``comparison_report`` with per-case three-mode metrics (off, shadow,
active_simulation).

Decision logic (applied in order — first match wins):

1. NO_GO   — any safety-invariant failure OR active simulation
              regresses negatives/wrong-population/latency past
              hard thresholds.
2. GO       — active_simulation outperforms ``off`` on the recall +
              false-no-evidence + negative-rate criteria (majority of
              populated cases).
3. CONTINUE_SHADOW — everything else (mixed results, incomplete data,
              <5 populated cases).

Output schema (see ``Recommendation.to_dict()``):
  - ``recommendation``: closed-set string.
  - ``rationale``: one-paragraph human-readable explanation.
  - ``criteria_evaluated``: per-criterion pass/fail + value + threshold.
  - ``failing_benchmarks``: case ids that regressed under
    active_simulation.
  - ``recommended_fixes``: closed-set tokens such as
    ``populate_more_gold_sets``, ``investigate_source_api_errors``, etc.
  - ``evaluated_at``: ISO timestamp.
  - ``run_ids_evaluated``: list of run UUIDs included.

Pure-deterministic; takes only what is already in ``benchmark_runs``
documents — no PHI, no raw prompts, no LLM, no external network.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Closed-set recommendation outputs.
RECOMMENDATION_VALUES = {"GO", "CONTINUE_SHADOW", "NO_GO"}

# Closed-set recommended-fix tokens.
RECOMMENDED_FIX_TOKENS = {
    "populate_more_gold_sets",
    "graduate_provisional_cases",
    "investigate_source_api_errors",
    "investigate_rate_limit_errors",
    "investigate_latency_regression",
    "investigate_safety_invariant_failure",
    "investigate_negative_promotion",
    "investigate_wrong_specialty_increase",
    "investigate_wrong_population_increase",
    "investigate_wrong_intervention_increase",
    "rerun_when_more_data_available",
}


def _summary_job_timeout_seconds() -> int:
    try:
        return int(os.environ.get("SUMMARY_JOB_TIMEOUT_SECONDS", "120"))
    except (TypeError, ValueError):
        return 120


@dataclass
class CriterionResult:
    criterion: str
    passed: bool
    value: Any
    threshold: str


@dataclass
class Recommendation:
    recommendation: str
    rationale: str
    criteria_evaluated: list[CriterionResult] = field(default_factory=list)
    failing_benchmarks: list[str] = field(default_factory=list)
    recommended_fixes: list[str] = field(default_factory=list)
    evaluated_at: str = ""
    run_ids_evaluated: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        # Phase 6C — keep the existing keys for BC AND add the brief's
        # preferred vocabulary (verdict / safety_failures / inputs_used /
        # delta_metrics / reasoning). Both name sets are equivalent.
        safety_failures: list[str] = [
            c.criterion for c in self.criteria_evaluated
            if not c.passed and c.criterion.startswith("safety.")
        ]
        delta_metrics = {
            c.criterion: c.value for c in self.criteria_evaluated
            if not c.criterion.startswith("safety.")
        }
        inputs_used = {
            c.criterion: {"value": c.value, "threshold": c.threshold,
                           "passed": c.passed}
            for c in self.criteria_evaluated
        }
        return {
            "recommendation": self.recommendation,
            "rationale": self.rationale,
            "criteria_evaluated": [
                {"criterion": c.criterion, "passed": c.passed,
                 "value": c.value, "threshold": c.threshold}
                for c in self.criteria_evaluated
            ],
            "failing_benchmarks": list(self.failing_benchmarks),
            "recommended_fixes": list(self.recommended_fixes),
            "evaluated_at": self.evaluated_at,
            "run_ids_evaluated": list(self.run_ids_evaluated),
            # Phase 6C aliases per brief vocabulary.
            "verdict": self.recommendation,
            "reasoning": self.rationale,
            "safety_failures": safety_failures,
            "inputs_used": inputs_used,
            "delta_metrics": delta_metrics,
        }


def _aggregate_modes(case_reports: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate per-mode metrics across populated cases.

    Returns ``{mode: {metric: value}}`` for mode in
    {"off", "shadow", "active_simulation"}.
    """
    agg: dict[str, dict[str, float]] = {
        m: {"gold_in_top_10": 0.0, "gold_in_top_20": 0.0,
            "recall_at_10": 0.0, "recall_at_20": 0.0,
            "mrr": 0.0, "false_no_evidence_rate": 0.0,
            "negatives_in_top_10": 0.0,
            "wrong_specialty_rate": 0.0,
            "wrong_population_rate": 0.0,
            "wrong_intervention_rate": 0.0,
            "latency_p95_ms": 0.0,
            "source_api_errors": 0.0,
            "n_cases": 0.0}
        for m in ("off", "shadow", "active_simulation")
    }
    for case in case_reports:
        modes = case.get("by_mode") or {}
        for mode_name in ("off", "shadow", "active_simulation"):
            m = modes.get(mode_name) or {}
            if not m:
                continue
            ag = agg[mode_name]
            ag["n_cases"] += 1
            ag["gold_in_top_10"] += float(m.get("gold_in_top_10") or 0)
            ag["gold_in_top_20"] += float(m.get("gold_in_top_20") or 0)
            ag["recall_at_10"] += float(m.get("recall_at_10") or 0)
            ag["recall_at_20"] += float(m.get("recall_at_20") or 0)
            ag["mrr"] += float(m.get("mrr") or 0)
            ag["negatives_in_top_10"] += float(m.get("negatives_in_top_10") or 0)
            ag["false_no_evidence_rate"] += float(m.get("false_no_evidence_rate") or 0)
            ag["wrong_specialty_rate"] += float(m.get("wrong_specialty_rate") or 0)
            ag["wrong_population_rate"] += float(m.get("wrong_population_rate") or 0)
            ag["wrong_intervention_rate"] += float(m.get("wrong_intervention_rate") or 0)
            ag["latency_p95_ms"] = max(ag["latency_p95_ms"],
                                       float(m.get("latency_p95_ms") or 0))
            ag["source_api_errors"] += float(m.get("source_api_errors") or 0)
    # Mean for the rate-like fields.
    for mode_name, ag in agg.items():
        n = ag["n_cases"] or 1
        for k in ("recall_at_10", "recall_at_20", "mrr",
                   "false_no_evidence_rate", "wrong_specialty_rate",
                   "wrong_population_rate", "wrong_intervention_rate"):
            ag[k] = round(ag[k] / n, 4)
        ag["negatives_in_top_10_mean"] = round(
            ag["negatives_in_top_10"] / n, 4,
        )
    return agg


def compute_recommendation(
    runs: list[dict[str, Any]] | None = None,
    *,
    pooled_cases: list[dict[str, Any]] | None = None,
    safety_signals: dict[str, Any] | None = None,
    run_ids: list[str] | None = None,
) -> Recommendation:
    """Compute the recommendation from one or more ``benchmark_runs``.

    Two call shapes supported:

    1. Legacy (Phase 5C.1) — ``compute_recommendation(runs)`` where
       each run MUST contain ``comparison_report.per_case_three_mode``.
       Caller is expected to filter to runs that have the three-mode
       comparison block.

    2. Phase 6C — ``compute_recommendation(pooled_cases=..., safety_signals=...,
       run_ids=...)``. The caller has already pooled per-case three-mode
       rows across one-or-many runs and is providing the union of
       safety_signals separately. This shape is what the live sweep
       runner + Phase 6C admin endpoints use.

    Older single-mode runs without ``comparison_report`` are ignored
    when using shape (1).
    """
    if pooled_cases is None and runs is None:
        runs = []
    if pooled_cases is not None:
        # Phase 6C shape.
        pooled = list(pooled_cases or [])
        safety_signals = safety_signals or {}
        run_ids_eff: list[str] = list(run_ids or [])
        legacy_run_docs: list[dict[str, Any]] = [
            {"comparison_report": {"safety_signals": safety_signals}},
        ]
    else:
        # Legacy shape.
        pooled = []
        for r in (runs or []):
            cr = (r.get("comparison_report") or {}).get(
                "per_case_three_mode") or []
            pooled.extend(cr)
        run_ids_eff = [r.get("id") for r in (runs or []) if r.get("id")]
        legacy_run_docs = list(runs or [])

    rec = Recommendation(
        recommendation="CONTINUE_SHADOW",
        rationale="No benchmark runs available to evaluate.",
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        run_ids_evaluated=run_ids_eff,
    )
    if not pooled and not legacy_run_docs:
        rec.recommended_fixes = ["rerun_when_more_data_available"]
        return rec
    if not pooled:
        rec.rationale = (
            "No three-mode comparison data found in evaluated runs. "
            "Phase 5C.1 three-mode execution may not have been run yet."
        )
        rec.recommended_fixes = ["rerun_when_more_data_available"]
        # Safety check still applies when pooled is empty but safety
        # signals were passed in directly (Phase 6C call shape).
        # Fall through to the safety block below.

    # Phase 6C — "populated" means: not provisional AND not placeholder_gold.
    # Placeholder-gold cases use synthetic test PMIDs that PubMed/Europe
    # PMC will never return under a live sweep — they're guaranteed
    # 0-recall and would bias the aggregate downward.
    populated = [
        c for c in pooled
        if not c.get("provisional") and not c.get("placeholder_gold")
    ]
    n_pop = len(populated)
    # Track which case ids regressed under active_simulation.
    failing: list[str] = []

    # ---- Safety invariant checks (NO_GO if any fails) ----
    safety_issues: list[str] = []
    for r in legacy_run_docs:
        sf = (r.get("comparison_report") or {}).get("safety_signals") or {}
        if sf.get("phi_expansion_triggered"):
            safety_issues.append("phi_expansion_triggered")
        if sf.get("non_whitelisted_source_detected"):
            safety_issues.append("non_whitelisted_source_detected")
        if sf.get("raw_phi_in_output"):
            safety_issues.append("raw_phi_in_output")
        if sf.get("raw_prompts_in_output"):
            safety_issues.append("raw_prompts_in_output")
        if sf.get("raw_evidence_text_in_output"):
            safety_issues.append("raw_evidence_text_in_output")
        if sf.get("unaudited_ranking_change"):
            safety_issues.append("unaudited_ranking_change")
    if safety_issues:
        # Phase 6C — prefix safety-criterion names with ``safety.`` so
        # ``Recommendation.to_dict().safety_failures`` correctly picks
        # them up.
        rec.recommendation = "NO_GO"
        rec.rationale = (
            "NO-GO: safety invariant failure(s) detected — "
            f"{', '.join(sorted(set(safety_issues)))}."
        )
        rec.failing_benchmarks = list({
            c.get("benchmark_case_id") for c in populated if c.get("benchmark_case_id")
        })
        rec.recommended_fixes = ["investigate_safety_invariant_failure"]
        rec.criteria_evaluated = [
            CriterionResult(
                f"safety.{name}", False, True, "must be false",
            ) for name in sorted(set(safety_issues))
        ]
        return rec

    if not pooled:
        # No three-mode data and no safety failures → CONTINUE_SHADOW.
        return rec

    # Aggregate per-mode metrics.
    by_mode = _aggregate_modes(populated)
    off = by_mode["off"]
    actv = by_mode["active_simulation"]
    summary_budget_ms = _summary_job_timeout_seconds() * 1000

    # ---- Hard NO_GO thresholds on active_simulation ----
    hard_checks: list[CriterionResult] = []
    # 1) Negatives in top-10: active_simulation must not increase by >= 1 on average
    neg_off = off.get("negatives_in_top_10_mean") or 0.0
    neg_actv = actv.get("negatives_in_top_10_mean") or 0.0
    hard_checks.append(CriterionResult(
        "negatives_in_top_10_active_simulation_not_worse_by_1",
        passed=(neg_actv - neg_off) < 1.0,
        value=f"off={neg_off:.2f} active={neg_actv:.2f}",
        threshold="active_simulation - off < 1 negative/case",
    ))
    # 2) Latency p95 budget
    hard_checks.append(CriterionResult(
        "latency_p95_within_summary_budget",
        passed=actv.get("latency_p95_ms", 0) <= summary_budget_ms,
        value=f"p95={int(actv.get('latency_p95_ms', 0))}ms",
        threshold=f"<= {summary_budget_ms}ms",
    ))
    # 3) Source API error rate < 10% per source — approximate
    src_err_actv = actv.get("source_api_errors") or 0
    src_err_rate = src_err_actv / max(1, n_pop)
    hard_checks.append(CriterionResult(
        "source_api_error_rate_under_10pct",
        passed=src_err_rate < 0.10,
        value=f"errors_per_case={src_err_rate:.3f}",
        threshold="< 0.10",
    ))
    # 4) Wrong-population / wrong-intervention not materially worse
    for metric, label in (
        ("wrong_population_rate", "wrong_population"),
        ("wrong_intervention_rate", "wrong_intervention"),
        ("wrong_specialty_rate", "wrong_specialty"),
    ):
        base = off.get(metric, 0.0)
        new = actv.get(metric, 0.0)
        passed = (new - base) <= 0.10  # 10pp materiality
        hard_checks.append(CriterionResult(
            f"{label}_rate_not_materially_worse",
            passed=passed,
            value=f"off={base:.2f} active={new:.2f}",
            threshold="active - off <= 0.10",
        ))

    if any(not c.passed for c in hard_checks):
        rec.recommendation = "NO_GO"
        failed = [c.criterion for c in hard_checks if not c.passed]
        rec.rationale = (
            "NO-GO: active_simulation regressed on one or more hard "
            f"thresholds — {', '.join(failed)}."
        )
        # Failing benchmarks: cases whose by_mode.active_simulation
        # underperformed by_mode.off on gold-in-top-10.
        for c in populated:
            modes = c.get("by_mode") or {}
            if (modes.get("active_simulation", {}).get("gold_in_top_10", 0)
                    < modes.get("off", {}).get("gold_in_top_10", 0)):
                cid = c.get("benchmark_case_id")
                if cid:
                    failing.append(cid)
        rec.failing_benchmarks = list(dict.fromkeys(failing))
        rec.criteria_evaluated = hard_checks
        rec.recommended_fixes = []
        for fname in failed:
            if "latency" in fname:
                rec.recommended_fixes.append("investigate_latency_regression")
            if "source_api_error" in fname:
                rec.recommended_fixes.append("investigate_source_api_errors")
            if "wrong_population" in fname:
                rec.recommended_fixes.append("investigate_wrong_population_increase")
            if "wrong_intervention" in fname:
                rec.recommended_fixes.append("investigate_wrong_intervention_increase")
            if "wrong_specialty" in fname:
                rec.recommended_fixes.append("investigate_wrong_specialty_increase")
            if "negatives" in fname:
                rec.recommended_fixes.append("investigate_negative_promotion")
        rec.recommended_fixes = list(dict.fromkeys(rec.recommended_fixes))
        return rec

    # ---- Sufficiency check — fewer than 5 populated cases → CONTINUE_SHADOW ----
    if n_pop < 5:
        rec.recommendation = "CONTINUE_SHADOW"
        rec.rationale = (
            f"CONTINUE_SHADOW: only {n_pop} populated case(s) evaluated; "
            "spec requires ≥5 for a GO decision. Graduate provisional "
            "cases by populating gold PMID sets, then re-run."
        )
        rec.recommended_fixes = [
            "populate_more_gold_sets", "graduate_provisional_cases",
        ]
        rec.criteria_evaluated = hard_checks + [
            CriterionResult(
                "minimum_populated_cases", False,
                n_pop, ">= 5",
            ),
        ]
        return rec

    # ---- GO criteria (all must hold across populated cases) ----
    # Count cases where active_simulation is ≥ off on gold_in_top_10 etc.
    gold10_majority = 0
    gold20_majority = 0
    # Sufficiency: count cases where active_simulation is STRICTLY better
    # than off on at least one core metric. If 0 cases improve, the run
    # carries no signal and we default to CONTINUE_SHADOW even if all
    # soft checks pass trivially (e.g., when fixtures make every mode
    # identical).
    strictly_better_cases = 0
    for c in populated:
        modes = c.get("by_mode") or {}
        a = modes.get("active_simulation", {}) or {}
        o = modes.get("off", {}) or {}
        if (a.get("gold_in_top_10", 0) or 0) >= (o.get("gold_in_top_10", 0) or 0):
            gold10_majority += 1
        if (a.get("gold_in_top_20", 0) or 0) >= (o.get("gold_in_top_20", 0) or 0):
            gold20_majority += 1
        # "Strictly better" = active_simulation beats off on recall@10
        # OR mrr OR gold_in_top_10 by any positive margin.
        if (
            (a.get("recall_at_10", 0) or 0) > (o.get("recall_at_10", 0) or 0)
            or (a.get("mrr", 0) or 0) > (o.get("mrr", 0) or 0)
            or (a.get("gold_in_top_10", 0) or 0) > (o.get("gold_in_top_10", 0) or 0)
        ):
            strictly_better_cases += 1
    gold10_pass = gold10_majority > n_pop / 2
    gold20_pass = gold20_majority > n_pop / 2
    fne_pass = (actv.get("false_no_evidence_rate") or 0) <= (
        off.get("false_no_evidence_rate") or 0
    )
    neg_pass = (actv.get("negatives_in_top_10_mean") or 0) <= (
        off.get("negatives_in_top_10_mean") or 0
    )
    # Sufficiency check: at least one case must show active_simulation
    # STRICTLY improves over off, otherwise there is no positive signal
    # to justify flipping (a "tie everywhere" trivially passes the
    # ≥-majority gates but provides no evidence of benefit).
    sufficiency_pass = strictly_better_cases >= 1
    soft_checks = [
        CriterionResult(
            "gold_in_top_10_active_ge_off_majority",
            gold10_pass,
            f"{gold10_majority}/{n_pop} cases",
            "majority (>n/2)",
        ),
        CriterionResult(
            "gold_in_top_20_active_ge_off_majority",
            gold20_pass,
            f"{gold20_majority}/{n_pop} cases",
            "majority (>n/2)",
        ),
        CriterionResult(
            "false_no_evidence_rate_active_le_off",
            fne_pass,
            f"off={off.get('false_no_evidence_rate'):.3f} "
            f"active={actv.get('false_no_evidence_rate'):.3f}",
            "active <= off",
        ),
        CriterionResult(
            "negatives_in_top_10_active_le_off",
            neg_pass,
            f"off={off.get('negatives_in_top_10_mean', 0):.3f} "
            f"active={actv.get('negatives_in_top_10_mean', 0):.3f}",
            "active <= off",
        ),
        CriterionResult(
            "active_simulation_strictly_better_on_at_least_one_case",
            sufficiency_pass,
            f"{strictly_better_cases}/{n_pop} cases",
            ">= 1 case strictly improves",
        ),
    ]
    rec.criteria_evaluated = hard_checks + soft_checks

    if all(c.passed for c in soft_checks):
        rec.recommendation = "GO"
        rec.rationale = (
            f"GO: active_simulation matched or beat ``off`` on all GO "
            f"criteria across {n_pop} populated cases — gold@10 majority "
            f"{gold10_majority}/{n_pop}, gold@20 majority "
            f"{gold20_majority}/{n_pop}, "
            f"{strictly_better_cases}/{n_pop} cases strictly improved, "
            "false-no-evidence rate did not regress, negatives in top-10 "
            "did not regress, and all hard safety/latency/source-error "
            "thresholds held."
        )
        rec.recommended_fixes = []
        return rec

    rec.recommendation = "CONTINUE_SHADOW"
    rec.rationale = (
        f"CONTINUE_SHADOW: mixed results across {n_pop} populated cases. "
        + " ".join(
            f"{c.criterion}={'PASS' if c.passed else 'FAIL'}."
            for c in soft_checks
        )
    )
    # Cases where active_simulation underperformed off in any metric.
    for c in populated:
        modes = c.get("by_mode") or {}
        a = modes.get("active_simulation", {})
        o = modes.get("off", {})
        if (a.get("gold_in_top_10", 0) < o.get("gold_in_top_10", 0)
                or a.get("negatives_in_top_10", 0) > o.get("negatives_in_top_10", 0)
                or a.get("false_no_evidence_rate", 0)
                    > o.get("false_no_evidence_rate", 0)):
            cid = c.get("benchmark_case_id")
            if cid:
                failing.append(cid)
    rec.failing_benchmarks = list(dict.fromkeys(failing))
    rec.recommended_fixes = ["populate_more_gold_sets",
                              "rerun_when_more_data_available"]
    return rec
