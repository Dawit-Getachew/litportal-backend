"""Phase ImpactRank-1 — iCite/OpenAlex Citation Impact Enrichment (SHADOW ONLY).

Deterministic. NO LLM. Article-level citation-impact enrichment + a
BOUNDED, SHADOW-ONLY simulated impact boost. NEVER changes clinician-visible
ranking, default selection, the evidence packet, the summary, or citation
verification.

Hard guarantees:
  * Clinical directness + safety caps + question-lane relevance DOMINATE
    citation impact. A boost is applied (in simulation only) ONLY after
    those checks pass.
  * No boost for policy-excluded (retracted / preprint / non-article),
    wrong-disease, wrong-lane, or wrong-outcome records — even when their
    citation counts are very high.
  * Missing metrics NEVER penalize an article (boost simply = 0).
  * Output carries only bibliographic metadata + closed-set tokens +
    numeric metrics. No abstracts / full text / PHI / prompts / credentials.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from services.rejected_article_rescue import is_policy_excluded

logger = logging.getLogger("clinevidence.impactrank")

SCHEMA_VERSION = "ImpactRank.1.1"

# Clinician-visible "included" floor. An article whose deterministic relevance
# score reaches this floor passed the clinical ranker and is treated as a
# positive directness signal (see ``_has_positive_directness_signal``).
INCLUDE_FLOOR = 60.0

# ---- Closed-set reasons for whether the simulated boost applied. ----
IMPACT_APPLIED = "impact_applied"
NOT_APPLIED_POLICY_EXCLUDED = "not_applied_policy_excluded"
NOT_APPLIED_WRONG_DISEASE = "not_applied_wrong_disease"
NOT_APPLIED_WRONG_LANE = "not_applied_wrong_lane"
NOT_APPLIED_WRONG_OUTCOME = "not_applied_wrong_outcome"
NOT_APPLIED_NON_ARTICLE = "not_applied_non_article"
NOT_APPLIED_NO_METRICS = "not_applied_no_metrics"
NOT_APPLIED_CAPPED = "not_applied_capped"
NOT_APPLIED_NOT_DIRECT = "not_applied_not_direct"

IMPACT_REASONS = frozenset({
    IMPACT_APPLIED, NOT_APPLIED_POLICY_EXCLUDED, NOT_APPLIED_WRONG_DISEASE,
    NOT_APPLIED_WRONG_LANE, NOT_APPLIED_WRONG_OUTCOME, NOT_APPLIED_NON_ARTICLE,
    NOT_APPLIED_NO_METRICS, NOT_APPLIED_CAPPED, NOT_APPLIED_NOT_DIRECT,
})

# R2A relevance_reason → closed-set impact-block reason.
_R2A_WRONG_REASON_MAP = {
    "wrong_disease": NOT_APPLIED_WRONG_DISEASE,
    "wrong_population": NOT_APPLIED_WRONG_LANE,
    "advanced_metastatic_wrong_stage": NOT_APPLIED_WRONG_OUTCOME,
    "treatment_not_prognosis": NOT_APPLIED_WRONG_LANE,
    "diagnostic_not_prognosis": NOT_APPLIED_WRONG_LANE,
    "screening_not_prognosis": NOT_APPLIED_WRONG_LANE,
    "mechanism_not_prognosis": NOT_APPLIED_WRONG_LANE,
    "non_article": NOT_APPLIED_NON_ARTICLE,
    "preprint_policy": NOT_APPLIED_POLICY_EXCLUDED,
    "retracted_policy": NOT_APPLIED_POLICY_EXCLUDED,
}
_R2A_CANDIDATE_LABELS = frozenset({
    "direct_relevance_candidate", "borderline_relevance_candidate",
    "indirect_relevance_candidate",
})

# Deterministic cap/flag token → impact-block reason (substring match).
# Phase ImpactRank-1.1: extended with the cap/rejection tokens emitted by the
# Phase 5A/6F ranker + clinical-question-map anchoring path (e.g.
# ``missing_condition_anchor`` / ``wrong_condition`` / ``missing_primary_anchor``)
# that were previously NOT recognized, so off-disease / un-anchored articles
# could slip through as "no disqualifier → direct".
_CAP_TOKEN_REASONS = (
    # --- disease / condition anchoring ---
    ("wrong_disease", NOT_APPLIED_WRONG_DISEASE),
    ("off_disease", NOT_APPLIED_WRONG_DISEASE),
    ("wrong_condition", NOT_APPLIED_WRONG_DISEASE),
    ("missing_condition_anchor", NOT_APPLIED_WRONG_DISEASE),
    ("missing_primary_anchor", NOT_APPLIED_WRONG_DISEASE),
    ("missing_disease_anchor", NOT_APPLIED_WRONG_DISEASE),
    # --- population / lane / outcome ---
    ("wrong_population", NOT_APPLIED_WRONG_LANE),
    ("wrong_lane", NOT_APPLIED_WRONG_LANE),
    ("wrong_framework_lane", NOT_APPLIED_WRONG_LANE),
    ("off_lane", NOT_APPLIED_WRONG_LANE),
    ("off_target", NOT_APPLIED_WRONG_LANE),
    ("screening", NOT_APPLIED_WRONG_LANE),
    ("diagnostic_only", NOT_APPLIED_WRONG_LANE),
    ("mechanism", NOT_APPLIED_WRONG_LANE),
    ("treatment_only", NOT_APPLIED_WRONG_LANE),
    ("wrong_outcome", NOT_APPLIED_WRONG_OUTCOME),
    ("wrong_stage", NOT_APPLIED_WRONG_OUTCOME),
    # --- policy / record-type ---
    ("non_article_record", NOT_APPLIED_NON_ARTICLE),
    ("non_article", NOT_APPLIED_NON_ARTICLE),
    ("policy_excluded", NOT_APPLIED_POLICY_EXCLUDED),
    ("retracted", NOT_APPLIED_POLICY_EXCLUDED),
    ("preprint", NOT_APPLIED_POLICY_EXCLUDED),
    ("expression_of_concern", NOT_APPLIED_POLICY_EXCLUDED),
    # --- dosing / safety caps ---
    ("off_label", NOT_APPLIED_CAPPED),
    ("contraindicat", NOT_APPLIED_CAPPED),
    ("dose", NOT_APPLIED_CAPPED),
    ("safety", NOT_APPLIED_CAPPED),
)

# Closed-set positive directness tokens (lane / answer-class / role labels).
# Their presence on a directness/lane/role field is a positive signal that the
# article directly addresses the clinical question (e.g. a prognosis lane).
_POSITIVE_DIRECTNESS_TOKENS = frozenset({
    "direct", "direct_answer", "directly_answers", "direct_evidence",
    "direct_relevance_candidate", "borderline_relevance_candidate",
    "prognosis_cohort", "survival_outcome", "recurrence_outcome",
    "cancer_specific_mortality", "stage_specific_prognosis",
    "localized_high_risk_context", "prognostic_model", "overall_survival",
    "disease_free_survival", "biochemical_recurrence",
})


def max_boost() -> float:
    """Bounded simulated boost ceiling (6–8 points)."""
    try:
        v = float(os.environ.get("IMPACTRANK_MAX_BOOST", "8"))
    except ValueError:
        v = 8.0
    return max(6.0, min(8.0, v))


def impactrank_shadow_enabled() -> bool:
    """Preview/dev only by default; OFF in production unless explicitly
    enabled (mirrors the R2A / M4 production-safety posture)."""
    is_prod = os.environ.get("APP_ENV", "development").strip().lower() == "production"
    if not is_prod:
        return True
    return os.environ.get("IMPACTRANK_SHADOW_PROD_ENABLED", "false").strip().lower() == "true"


def _num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def compute_impact_factor(metrics: dict) -> dict[str, Any]:
    """Deterministic article-level impact factor in [0,1] from iCite
    metrics. Returns the factor + its components + has_metrics flag."""
    metrics = metrics if isinstance(metrics, dict) else {}
    rcr = _num(metrics.get("relative_citation_ratio"))
    pct = _num(metrics.get("nih_percentile"))
    apt = _num(metrics.get("apt"))
    cbc = _num(metrics.get("cited_by_clin_count"))
    cnt = _num(metrics.get("citation_count"))

    comp_rcr = min(rcr / 2.0, 1.0) if rcr is not None else 0.0
    comp_pct = (pct / 100.0) if pct is not None else 0.0
    comp_apt = apt if (apt is not None and 0.0 <= apt <= 1.0) else 0.0
    comp_cbc = min(cbc / 5.0, 1.0) if cbc is not None else 0.0
    factor = 0.5 * comp_rcr + 0.25 * comp_pct + 0.15 * comp_apt + 0.10 * comp_cbc
    factor = max(0.0, min(1.0, factor))
    has_metrics = any(x is not None for x in (rcr, pct, cnt))
    return {
        "impact_factor": round(factor, 4),
        "has_metrics": has_metrics,
        "components": {
            "rcr": comp_rcr, "nih_percentile": comp_pct,
            "apt": comp_apt, "cited_by_clin": comp_cbc,
        },
    }


def _wrong_token_reason(cap_reasons, flags) -> Optional[str]:
    toks = " ".join(
        [str(x).lower() for x in (cap_reasons or [])]
        + [str(x).lower() for x in (flags or [])]
    )
    for token, reason in _CAP_TOKEN_REASONS:
        if token in toks:
            return reason
    return None


def _has_positive_directness_signal(
    *, article_doc: dict, sa_row: dict, r2a_label: Optional[dict],
) -> bool:
    """Deterministic POSITIVE directness evidence. A simulated boost is granted
    ONLY when at least one of these holds (and no blocking cap/rejection fired):

      * R2A ``direct_relevance_candidate`` / ``borderline_relevance_candidate``
        (wrong-reason borderline candidates are blocked earlier);
      * the article passed the clinical ranker into the clinician-visible
        included set (``included_by_default`` or relevance score ≥ INCLUDE_FLOOR);
      * a directness / answer-class / lane-role label matching the question
        type (closed-set ``_POSITIVE_DIRECTNESS_TOKENS``).
    """
    # 1) R2A directness candidate.
    if r2a_label:
        label = str(r2a_label.get("rescue_label") or "").strip().lower()
        if label in _R2A_CANDIDATE_LABELS:
            return True

    # 2) Passed the clinical ranker (clinician-visible included set).
    if sa_row.get("included_by_default") is True:
        return True
    score = _num(sa_row.get("overall_rank"))
    if score is not None and score >= INCLUDE_FLOOR:
        return True

    # 3) Directness / answer-class / lane-role label matches the question type.
    lane_bits = [
        str(sa_row.get("directness_label") or article_doc.get("directness_label") or ""),
        str(sa_row.get("article_answer_class") or article_doc.get("article_answer_class") or ""),
    ]
    for tok in (sa_row.get("query_types_matched") or []):
        lane_bits.append(str(tok))
    roles = sa_row.get("subquestion_roles_matched") or {}
    if isinstance(roles, dict):
        for v in roles.values():
            lane_bits.extend(str(x) for x in (v if isinstance(v, (list, tuple)) else [v]))
    for entry in (sa_row.get("role_match_breakdown_json") or []):
        if isinstance(entry, dict):
            lane_bits.append(str(entry.get("role") or entry.get("label") or ""))
    blob = " ".join(lane_bits).lower()
    return any(tok in blob for tok in _POSITIVE_DIRECTNESS_TOKENS)


def assess_boost_eligibility(
    *,
    article_doc: dict,
    sa_row: dict,
    r2a_label: Optional[dict],
) -> tuple[bool, str]:
    """Deterministic eligibility for a SIMULATED boost (Phase ImpactRank-1.1).

    Policy: **blocked unless a positive directness signal exists AND no blocking
    cap/rejection is present.** Clinical directness + safety caps + lane
    relevance dominate citation impact. Returns ``(eligible, reason)`` with
    ``reason`` in the closed set.
    """
    # ---- Blocking checks (dominant). ----
    # 1) Policy exclusion (retracted / preprint / non-article).
    if is_policy_excluded(article_doc, sa_row) is not None:
        return False, NOT_APPLIED_POLICY_EXCLUDED

    # 2) R2A directness label (authoritative for rejected articles).
    #    Explicit "not direct" verdicts (keep_rejected / indirect) override the
    #    structural score≥floor positive-signal proxy below.
    if r2a_label:
        reason = str(r2a_label.get("relevance_reason") or "").strip().lower()
        label = str(r2a_label.get("rescue_label") or "").strip().lower()
        if reason in _R2A_WRONG_REASON_MAP:
            return False, _R2A_WRONG_REASON_MAP[reason]
        if label in ("policy_excluded",):
            return False, NOT_APPLIED_POLICY_EXCLUDED
        if label in ("keep_rejected", "indirect_relevance_candidate"):
            return False, NOT_APPLIED_NOT_DIRECT

    # 3) Deterministic cap/flag block scan (wrong disease / lane / outcome /
    #    missing anchor / policy / non-article — extended in ImpactRank-1.1).
    cap_reason = _wrong_token_reason(
        sa_row.get("cap_reasons"), sa_row.get("flags"),
    )
    if cap_reason is not None:
        return False, cap_reason

    # ---- Positive directness REQUIRED (no longer "eligible unless blocked"). ----
    if _has_positive_directness_signal(
        article_doc=article_doc, sa_row=sa_row, r2a_label=r2a_label,
    ):
        return True, IMPACT_APPLIED
    return False, NOT_APPLIED_NOT_DIRECT


def compute_impact_for_article(
    *,
    article_doc: dict,
    sa_row: dict,
    icite_metrics: dict,
    r2a_label: Optional[dict] = None,
) -> dict[str, Any]:
    """Per-article SHADOW impact computation. Pure/deterministic."""
    factor_info = compute_impact_factor(icite_metrics)
    eligible, reason = assess_boost_eligibility(
        article_doc=article_doc, sa_row=sa_row, r2a_label=r2a_label,
    )
    if eligible and not factor_info["has_metrics"]:
        eligible, reason = False, NOT_APPLIED_NO_METRICS

    boost = round(max_boost() * factor_info["impact_factor"], 1) if eligible else 0.0
    openalex_cited = (
        article_doc.get("citation_count")
        or (article_doc.get("source_ids_json") or {}).get("openalex_cited_by_count")
    )
    return {
        "article_id": sa_row.get("article_id") or article_doc.get("id"),
        "pmid": article_doc.get("pmid"),
        "title": str(article_doc.get("title") or "")[:300],
        "journal": str(article_doc.get("journal") or "")[:160],
        "year": article_doc.get("publication_year") or article_doc.get("year"),
        "current_relevance_score": sa_row.get("overall_rank"),
        # ---- article-level impact metrics (shadow). ----
        "icite_rcr": icite_metrics.get("relative_citation_ratio"),
        "icite_nih_percentile": icite_metrics.get("nih_percentile"),
        "icite_apt": icite_metrics.get("apt"),
        "icite_cited_by_clin": icite_metrics.get("cited_by_clin_count"),
        "icite_citation_count": icite_metrics.get("citation_count"),
        "icite_is_clinical": icite_metrics.get("is_clinical"),
        "icite_provisional": icite_metrics.get("provisional"),
        "openalex_cited_by_count": openalex_cited,
        "impact_score_simulated": factor_info["impact_factor"],
        "impact_boost_simulated": boost,
        "impact_applied": eligible,
        "impact_reason": reason,
        "has_metrics": factor_info["has_metrics"],
        "r2a_rescue_label": (r2a_label or {}).get("rescue_label"),
    }


def _simulate_ranking(
    rows: list[dict], *, included_floor: float = 60.0,
) -> dict[str, Any]:
    """Shadow-only rank simulation. Adds the simulated boost to the
    current relevance score and re-orders the included set. Reports which
    clinically-direct high-impact papers WOULD move up and confirms
    wrong-lane high-citation papers stay capped. NO live change."""
    included = [r for r in rows if _num(r.get("current_relevance_score")) is not None
                and _num(r["current_relevance_score"]) >= included_floor]
    current = sorted(
        included, key=lambda r: _num(r["current_relevance_score"]) or 0.0, reverse=True,
    )
    cur_pos = {r["article_id"]: i + 1 for i, r in enumerate(current)}
    simulated = sorted(
        included,
        key=lambda r: (_num(r["current_relevance_score"]) or 0.0) + (r.get("impact_boost_simulated") or 0.0),
        reverse=True,
    )
    sim_pos = {r["article_id"]: i + 1 for i, r in enumerate(simulated)}

    moved_up = []
    for r in included:
        aid = r["article_id"]
        delta = cur_pos[aid] - sim_pos[aid]  # positive = moved up
        if delta > 0 and r.get("impact_applied"):
            moved_up.append({
                "article_id": aid, "pmid": r.get("pmid"),
                "title": r.get("title"),
                "current_rank_position": cur_pos[aid],
                "simulated_rank_position": sim_pos[aid],
                "positions_gained": delta,
                "impact_boost_simulated": r.get("impact_boost_simulated"),
                "impact_reason": r.get("impact_reason"),
            })
    moved_up.sort(key=lambda x: x["positions_gained"], reverse=True)

    # Wrong-lane high-citation papers that must remain capped (boost=0).
    capped_high_citation = []
    for r in rows:
        cited = _num(r.get("openalex_cited_by_count")) or _num(r.get("icite_citation_count")) or 0.0
        if not r.get("impact_applied") and cited >= 50 and r.get("impact_reason") in (
            NOT_APPLIED_WRONG_DISEASE, NOT_APPLIED_WRONG_LANE,
            NOT_APPLIED_WRONG_OUTCOME, NOT_APPLIED_NON_ARTICLE,
        ):
            capped_high_citation.append({
                "article_id": r["article_id"], "pmid": r.get("pmid"),
                "title": r.get("title"),
                "openalex_cited_by_count": r.get("openalex_cited_by_count"),
                "icite_citation_count": r.get("icite_citation_count"),
                "impact_boost_simulated": r.get("impact_boost_simulated"),
                "impact_reason": r.get("impact_reason"),
            })
    return {
        "note": "shadow_only_no_live_change",
        "included_floor_score": included_floor,
        "included_count": len(included),
        "clinically_direct_high_impact_moved_up": moved_up[:15],
        "wrong_lane_high_citation_remain_capped": capped_high_citation[:15],
    }


async def run_impact_enrichment_shadow(
    *,
    sa_rows: list[dict],
    articles_by_id: dict[str, dict],
    r2a_candidates_by_article_id: Optional[dict[str, dict]] = None,
    client: Any = None,
    db: Any = None,
    included_floor: float = 60.0,
) -> dict[str, Any]:
    """Orchestrate deterministic, shadow-only citation-impact enrichment
    over a session's article pool (active + rejected). Never raises."""
    from services.connectors.icite import icite_connector

    r2a_labels = r2a_candidates_by_article_id or {}
    rows = list(sa_rows or [])

    pmids = sorted({
        str((articles_by_id.get(r.get("article_id")) or {}).get("pmid") or "").strip()
        for r in rows
    } - {""})

    icite_result = await icite_connector.fetch_metrics(pmids, client=client, db=db)
    metrics_by_pmid = icite_result["metrics"]

    impact_rows: list[dict] = []
    for r in rows:
        art = articles_by_id.get(r.get("article_id"))
        if not art:
            continue
        pmid = str(art.get("pmid") or "").strip()
        m = metrics_by_pmid.get(pmid, {}) if pmid else {}
        impact_rows.append(
            compute_impact_for_article(
                article_doc=art, sa_row=r, icite_metrics=m,
                r2a_label=r2a_labels.get(r.get("article_id")),
            )
        )

    applied_rows = [r for r in impact_rows if r["impact_applied"]]
    metered_rows = [r for r in impact_rows if r["has_metrics"]]
    reason_counts: dict[str, int] = {}
    for r in impact_rows:
        reason_counts[r["impact_reason"]] = reason_counts.get(r["impact_reason"], 0) + 1

    # R2A rescue candidates enriched with impact (closed-set view).
    rescue_impact = [
        r for r in impact_rows
        if r.get("r2a_rescue_label") in _R2A_CANDIDATE_LABELS
    ]
    rescue_impact.sort(key=lambda r: r["impact_boost_simulated"], reverse=True)

    simulation = _simulate_ranking(impact_rows, included_floor=included_floor)

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "shadow",
        "no_clinician_visible_change": True,
        "max_impact_boost": max_boost(),
        "icite_stats": icite_result["stats"],
        "articles_enriched": len(metered_rows),
        "articles_total": len(impact_rows),
        "boost_eligible_count": len(applied_rows),
        "impact_reason_counts": reason_counts,
        "rescue_candidates_with_impact": rescue_impact[:20],
        "impact_enriched_articles": sorted(
            impact_rows, key=lambda r: r["impact_boost_simulated"], reverse=True,
        )[:30],
        "shadow_rank_simulation": simulation,
    }


__all__ = [
    "SCHEMA_VERSION",
    "IMPACT_REASONS",
    "max_boost",
    "impactrank_shadow_enabled",
    "compute_impact_factor",
    "assess_boost_eligibility",
    "compute_impact_for_article",
    "run_impact_enrichment_shadow",
]
