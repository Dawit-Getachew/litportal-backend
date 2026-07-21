"""Phase 5D.1 — default-selection module.

Pure function over ``(picott, ranked_articles, max_count)`` →
``{selected_article_ids, selection_rationale, excluded_count,
excluded_rationale_summary}``.

The default-selection function is the single source of truth for
``search_session_articles.included_by_default``. It replaces the
legacy ``evidence_ranker._apply_default_selection`` for sessions where
PICOTT exists.

Closed-set exclusion reasons (spec — must not extend without product
sign-off):
  cap_reason_missing_condition_anchor
  cap_reason_missing_intervention_anchor
  cap_reason_missing_index_test_anchor
  cap_reason_disease_only_match
  cap_reason_wrong_population
  cap_reason_wrong_intervention
  negative_concept_fired
  relevance_below_60
  excluded_retracted_or_preprint_or_unsafe

Tier order: guideline → SR/MA → RCT/primary → other directly relevant.
Each selected article must directly match at least one must-have anchor.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("clinevidence.default_selection")

DEFAULT_MAX_COUNT = 5

# Closed-set exclusion reasons. Frontend + tester rely on these literals.
EXCLUSION_REASONS = [
    "cap_reason_missing_condition_anchor",
    "cap_reason_missing_intervention_anchor",
    "cap_reason_missing_index_test_anchor",
    "cap_reason_disease_only_match",
    "cap_reason_wrong_population",
    "cap_reason_wrong_intervention",
    # Phase 6F — supplementary cap-reason codes (closed-set). These
    # are emitted by the picott_aware_ranker when an article fails the
    # primary clinical entity anchor check. They map 1:1 to existing
    # ``missing_intervention_anchor`` / ``missing_condition_anchor``
    # caps but use the more precise label for the UI + audit.
    "cap_reason_wrong_drug",
    "cap_reason_wrong_condition",
    "cap_reason_missing_primary_anchor",
    "cap_reason_wrong_primary_anchor",
    "cap_reason_drug_safety_no_adverse_effect",
    "cap_reason_treatment_sequencing_wrong_disease",
    # Phase 6G.4P — post-procedure management exclusion codes. Closed
    # set, additive — mirror the new ranker CAP_REASONS.
    "cap_reason_missing_procedure_anchor",
    "cap_reason_missing_management_anchor",
    "cap_reason_wrong_procedure",
    "cap_reason_related_but_distinct_procedure",
    "cap_reason_management_only_article",
    "cap_reason_procedure_only_article",
    "negative_concept_fired",
    "relevance_below_60",
    "excluded_retracted_or_preprint_or_unsafe",
]


def _has_negative_concept(r: dict) -> bool:
    """An article with a fired negative concept is excluded by default."""
    if r.get("penalty_concepts"):
        return True
    cap_reasons = r.get("cap_reasons") or []
    return "negative_concept_fired" in cap_reasons


def _is_safety_excluded(article: dict) -> bool:
    """Retracted, expression-of-concern, or unsafe-source article."""
    if not article:
        return False
    if article.get("is_retracted"):
        return True
    if article.get("has_expression_of_concern"):
        return True
    if article.get("is_preprint") and not article.get("is_preprint_safe_allowed"):
        # Per Phase 1 safety filter, preprints are excluded from
        # default-selection (clinician must explicitly include).
        return True
    return False


def _classify_cap_reasons(r: dict) -> list[str]:
    """Map the article's existing ``cap_reasons`` to the Phase 5D.1
    closed-set exclusion reasons. We do NOT invent new categories —
    only translate the Phase 5A vocabulary.

    Phase 6F — also recognizes the new closed-set supplementary codes
    (``wrong_drug``, ``wrong_condition``, ``missing_primary_anchor``,
    ``wrong_primary_anchor``, ``drug_safety_no_adverse_effect``,
    ``treatment_sequencing_wrong_disease``).
    """
    out: list[str] = []
    cap = r.get("cap_reasons") or []
    # cap_reasons in Phase 5A/6F use these labels:
    #   "missing_intervention_anchor", "missing_condition_anchor",
    #   "missing_index_test_anchor", "disease_only_match",
    #   "wrong_population", "wrong_intervention", "negative_concept_fired",
    #   "wrong_drug", "wrong_condition", "missing_primary_anchor",
    #   "wrong_primary_anchor", "drug_safety_no_adverse_effect",
    #   "treatment_sequencing_wrong_disease"
    for c in cap:
        if c in {
            "missing_condition_anchor", "missing_intervention_anchor",
            "missing_index_test_anchor", "disease_only_match",
            "wrong_population", "wrong_intervention",
            # Phase 6F additions.
            "wrong_drug", "wrong_condition",
            "missing_primary_anchor", "wrong_primary_anchor",
            "drug_safety_no_adverse_effect",
            "treatment_sequencing_wrong_disease",
            # Phase 6G.4P additions.
            "missing_procedure_anchor", "missing_management_anchor",
            "wrong_procedure", "related_but_distinct_procedure",
            "management_only_article", "procedure_only_article",
        }:
            out.append(f"cap_reason_{c}")
        elif c == "negative_concept_fired":
            out.append("negative_concept_fired")
    return out


def _role_match_directness(r: dict) -> int:
    """Phase 6B.2 — read the article's ``role_match_score`` (0..100)
    persisted by the picott_aware_ranker. Returns 0 if missing."""
    try:
        return int(r.get("role_match_score") or 0)
    except Exception:  # noqa: BLE001
        return 0


def _recalibrated_relevance(r: dict) -> float:
    """Phase 6B.2 — apply a role-match-derived recalibration multiplier
    to the article's relevance_score BEFORE the eligibility threshold
    check. The multiplier comes ONLY from role_match_score — NOT from
    query_family or planner_source. Cap-applied articles are NOT
    recalibrated; the caller short-circuits to the cap reason first."""
    base = float(r.get("relevance_score") or 0)
    rm = _role_match_directness(r)
    if rm >= 75:
        mult = 1.15
    elif rm >= 60:
        mult = 1.08
    elif rm >= 40:
        mult = 1.0
    else:
        mult = 0.95
    return min(100.0, base * mult)


def _exclusion_reason_for(r: dict) -> str | None:
    """Return the FIRST closed-set exclusion reason that applies, or
    None if the article is eligible for default selection.

    Phase 6B.2 — caps take precedence over recalibration. We check
    safety + caps FIRST, then re-evaluate the relevance threshold
    against the recalibrated score.
    """
    article = r.get("_article") or {}
    if _is_safety_excluded(article):
        return "excluded_retracted_or_preprint_or_unsafe"
    if _has_negative_concept(r):
        return "negative_concept_fired"
    closed_set = _classify_cap_reasons(r)
    if closed_set:
        # Capped articles can NEVER be default-selected, regardless of
        # role-match — mandatory-anchor caps are stronger than role
        # weights per the Phase 6B.2 guardrails.
        return closed_set[0]
    # Phase 6B.2 — recalibrated relevance check. Use the multiplier-
    # adjusted score; this is the ONLY threshold change in 6B.2 (the
    # global 60 threshold stays the same, but directly-relevant
    # articles with high role-match are lifted above it).
    rel = _recalibrated_relevance(r)
    if rel < 60:
        return "relevance_below_60"
    return None


def _matches_picott_must_haves(article: dict, picott: dict) -> bool:
    """True if the article's title+abstract mentions at least one
    must-have anchor (concept or synonym) per role group, restricted
    to the role groups that PICOTT declared.

    The check is intentionally permissive — the picott_ranker already
    fired the strict caps. We just verify the survivor mentions every
    PICOTT role at least once (condition AND intervention required;
    comparator optional).
    """
    if not article:
        return False
    if not picott or not picott.get("must_have_concepts"):
        # No PICOTT to enforce — fall back to "True" so all eligible
        # articles can be considered.
        return True

    haystack = (
        (article.get("title") or "") + " "
        + (article.get("abstract") or "")
    ).lower()

    by_role: dict[str, list[str]] = {}
    for mh in picott.get("must_have_concepts") or []:
        role = (mh.get("role") or "other").lower()
        terms = [mh.get("concept") or ""] + list(mh.get("synonyms") or [])
        by_role.setdefault(role, []).extend([t.lower() for t in terms if t])

    # Required role groups: condition and intervention (when present).
    required_role_groups = []
    for r in ("condition", "intervention", "exposure", "index_test"):
        if r in by_role:
            required_role_groups.append(r)

    for role in required_role_groups:
        terms = by_role[role]
        if not any(t in haystack for t in terms if t):
            return False
    return True


def _tier_rank(article: dict, picott: dict) -> int:
    """Lower tier number = higher priority for default selection.
    Tier 1 = guideline; 2 = SR/MA; 3 = RCT/primary-of-qtype;
    4 = other directly-relevant."""
    if not article:
        return 99
    if article.get("is_guideline"):
        return 1
    art_type = (article.get("article_type_classification") or "").lower()
    if art_type in {"systematic_review", "meta_analysis"}:
        return 2
    # Primary-of-qtype: use PICOTT type_of_question → article type.
    primary_by_qtype = {
        "therapy": "rct",
        "treatment": "rct",
        "prevention": "rct",
        "screening": "rct",
        "diagnosis": "diagnostic_accuracy_study",
        "harm": "cohort_study",
        "prognosis": "cohort_study",
        "etiology": "cohort_study",
        "pharmacology": "rct",
        "dosing_protocol": "rct",
        "epidemiology": "cohort_study",
        "guideline": "guideline",
    }
    toq = (picott.get("type_of_question") if picott else None) or "other"
    primary = primary_by_qtype.get(toq, "rct")
    if art_type == primary:
        return 3
    return 4


def select_default_articles(
    picott: dict | None,
    ranked_articles: list[dict],
    *,
    max_count: int = DEFAULT_MAX_COUNT,
) -> dict[str, Any]:
    """Pick default-selected article IDs per the Phase 5D.1 spec.

    Returns:
      {
        "selected_article_ids": [str, ...],            # at most max_count
        "selection_rationale": {article_id: reason_string},
        "excluded_count": int,
        "excluded_rationale_summary": {closed_set_reason: count},
      }
    """
    if not ranked_articles:
        return {
            "selected_article_ids": [],
            "selection_rationale": {},
            "excluded_count": 0,
            "excluded_rationale_summary": {},
        }

    eligible: list[dict] = []
    excluded_summary: dict[str, int] = {}

    for r in ranked_articles:
        article = r.get("_article") or {}
        article_id = article.get("id") or r.get("article_id")
        if not article_id:
            continue
        reason = _exclusion_reason_for(r)
        if reason is not None:
            excluded_summary[reason] = excluded_summary.get(reason, 0) + 1
            continue
        # Must directly match the must-have anchors per role.
        if not _matches_picott_must_haves(article, picott or {}):
            excluded_summary["cap_reason_missing_condition_anchor"] = (
                excluded_summary.get(
                    "cap_reason_missing_condition_anchor", 0,
                ) + 1
            )
            continue
        eligible.append({
            "article_id": article_id,
            "rank": r,
            "tier": _tier_rank(article, picott or {}),
            "score": r.get("overall_rank_score") or 0,
        })

    # Phase 6G.3 — disease_overview directness gating. When the
    # question subtype is ``disease_overview`` / ``definition``, we
    # default-select ONLY general/guideline/seminar/broad articles
    # if any are eligible; otherwise we fall back to the best
    # aspect-specific article with a closed-set rationale token so the
    # UI can surface the fallback warning.
    question_subtype = ((picott or {}).get("question_subtype") or "").lower()
    is_overview_route = question_subtype in {"disease_overview", "definition"}
    PRIMARY_OVERVIEW_DIRECTNESS = {
        "general_overview",
        "guideline_overview",
        "disease_seminar",
        "broad_clinical_review",
    }
    # Phase K4A.2 — severity-marker overview gating. When the question
    # subtype is ``severity_marker_overview``, default-select only the
    # closed-set "primary" directness labels (general overview /
    # guideline-severity / multi-marker SR / severity-score review).
    # If none are eligible, fall back to top aspect-specific articles
    # with a closed-set rationale token.
    is_severity_marker_route = question_subtype == "severity_marker_overview"
    PRIMARY_SEVERITY_MARKER_DIRECTNESS = {
        "general_severity_marker_overview",
        "guideline_severity_assessment",
        "multi_marker_systematic_review",
        "severity_score_review",
    }
    overview_fallback_reason: str | None = None
    if is_overview_route and eligible:
        primary = [
            e for e in eligible
            if (e["rank"].get("directness_label") or "")
            in PRIMARY_OVERVIEW_DIRECTNESS
        ]
        if primary:
            eligible = primary
        else:
            overview_fallback_reason = "no_general_overview_available"
            # Keep eligible as-is for fallback (top aspect/treatment
            # specific by score).
    elif is_severity_marker_route and eligible:
        primary = [
            e for e in eligible
            if (e["rank"].get("directness_label") or "")
            in PRIMARY_SEVERITY_MARKER_DIRECTNESS
        ]
        if primary:
            eligible = primary
        else:
            overview_fallback_reason = "no_severity_marker_overview_available"
    # Sort by (tier, -score). Lower tier first; higher score wins ties.
    eligible.sort(key=lambda e: (e["tier"], -float(e["score"])))

    selected_ids: list[str] = []
    rationale: dict[str, str] = {}
    selected_tiers: dict[int, int] = {}
    for e in eligible:
        if len(selected_ids) >= max_count:
            break
        # Prefer at least one of each tier 1..3 before doubling up.
        tier = e["tier"]
        if tier in (1, 2, 3) and selected_tiers.get(tier, 0) >= 1:
            # If we already have one of this tier and tier 4 hasn't been
            # picked yet, defer this one for the second pass.
            continue
        selected_ids.append(e["article_id"])
        selected_tiers[tier] = selected_tiers.get(tier, 0) + 1
        rationale[e["article_id"]] = _humanize_inclusion(e, picott)

    # Second pass: fill remaining slots from any unselected eligible.
    if len(selected_ids) < max_count:
        for e in eligible:
            if e["article_id"] in selected_ids:
                continue
            if len(selected_ids) >= max_count:
                break
            selected_ids.append(e["article_id"])
            rationale[e["article_id"]] = _humanize_inclusion(e, picott)

    excluded_count = sum(excluded_summary.values())
    # Phase K4A.2 — selected/excluded display invariant.
    # ``cleared_exclusion_for_ids`` is the closed-set list of article_ids
    # the caller MUST clear ``exclusion_reason`` on (so an article can
    # never appear as both default-selected AND excluded). The
    # selected_set is the source of truth — the pipeline persistence
    # layer reads this list and writes ``exclusion_reason=None`` for
    # each id before inserting the search_session_articles row. This
    # is defense-in-depth on top of the existing low_relevance clear
    # and the read-side ``default_selection_self_healed`` guard.
    cleared_exclusion_for_ids = list(selected_ids)
    # Phase 6B.2 — default_selection_summary block. Persisted on the
    # session document by the pipeline (search_sessions.default_selection_summary).
    # The lane key is always populated; for single-intent sessions it
    # is ``"primary"``.
    eligible_direct_count = len(eligible)
    excluded_by_cap_count = sum(
        v for k, v in excluded_summary.items()
        if k.startswith("cap_reason_") or k == "negative_concept_fired"
    )
    excluded_by_low_relevance_count = excluded_summary.get(
        "relevance_below_60", 0,
    )
    reason_each_selected = [
        {"article_id": aid, "reason": rationale.get(aid, "")}
        for aid in selected_ids
    ]
    summary = {
        "selected_count": len(selected_ids),
        "eligible_direct_count": eligible_direct_count,
        "excluded_by_cap_count": excluded_by_cap_count,
        "excluded_by_low_relevance_count": excluded_by_low_relevance_count,
        "selected_by_lane": {"primary": len(selected_ids)},
        "reason_each_selected": reason_each_selected,
        # Phase K4A.2 — surface the overview / severity-marker fallback
        # rationale (closed-set string) when the gating route had no
        # eligible primary-directness candidates and fell through to
        # aspect-specific articles. ``None`` on the happy path.
        "overview_fallback_reason": overview_fallback_reason,
    }
    return {
        "selected_article_ids": selected_ids,
        "selection_rationale": rationale,
        "excluded_count": excluded_count,
        "excluded_rationale_summary": excluded_summary,
        # Phase 6B.2 — persisted directly on the session.
        "default_selection_summary": summary,
        # Phase K4A.2 — selected/excluded display invariant guard.
        # The caller MUST clear ``exclusion_reason`` on every id in
        # this list. Always equal to ``selected_article_ids``; emitted
        # as a separate key so the contract is explicit at the call
        # site.
        "cleared_exclusion_for_ids": cleared_exclusion_for_ids,
    }


_TIER_LABEL = {
    1: "guideline",
    2: "systematic_review_or_meta_analysis",
    3: "primary_study_for_question_type",
    4: "other_directly_relevant",
}


def _humanize_inclusion(eligible_entry: dict, picott: dict | None) -> str:
    """Build a closed-set inclusion-rationale string. Always begins
    with the tier label so the frontend can group selections."""
    tier_label = _TIER_LABEL.get(eligible_entry["tier"], "unclassified")
    score = int(eligible_entry["score"] or 0)
    return (
        f"tier={tier_label} · rank_score={score} · "
        f"matched_must_have_anchors_for_question"
    )


def recount_summary_after_rescue(
    summary: dict[str, Any] | None,
    ranked: list[dict],
) -> dict[str, Any]:
    """Recompute the dynamic counts in a persisted ``default_selection_summary``
    after the LLM rescue re-add pass mutated ``ranked`` (rows carrying
    ``llm_rescued=True`` had their soft ``exclusion_reason`` cleared and
    ``included_by_default`` set).

    Pure — returns a NEW dict, never mutates the input. A no-op (returns the
    summary unchanged) when no row was rescued. The per-reason decrements use
    ``_exclusion_reason_for``, which still reflects the row's cap/relevance
    state after ``exclusion_reason`` was cleared (it reads cap_reasons /
    relevance_score, not the SSA exclusion_reason), so the closed-set keys line
    up with the ones ``select_default_articles`` originally emitted.
    """
    rescued = [r for r in (ranked or []) if r.get("llm_rescued")]
    if not rescued:
        return summary or {}

    out = dict(summary or {})
    excl = dict(out.get("excluded_rationale_summary") or {})
    low_rel = int(out.get("excluded_by_low_relevance_count") or 0)
    by_cap = int(out.get("excluded_by_cap_count") or 0)

    for r in rescued:
        reason = _exclusion_reason_for(r)
        if reason and excl.get(reason):
            excl[reason] -= 1
            if excl[reason] <= 0:
                excl.pop(reason, None)
        if reason == "relevance_below_60":
            low_rel = max(0, low_rel - 1)
        elif reason and (
            reason.startswith("cap_reason_") or reason == "negative_concept_fired"
        ):
            by_cap = max(0, by_cap - 1)

    selected_ids = [
        (r.get("_article") or {}).get("id")
        for r in ranked
        if r.get("included_by_default")
    ]
    selected_ids = [s for s in selected_ids if s]

    # Append a closed-set rationale for each rescued id not already listed.
    reason_each = list(out.get("reason_each_selected") or [])
    listed = {x.get("article_id") for x in reason_each}
    for r in rescued:
        aid = (r.get("_article") or {}).get("id")
        if aid and aid not in listed and r.get("included_by_default"):
            reason_each.append({"article_id": aid, "reason": "llm_rescued"})

    out["selected_article_ids"] = selected_ids
    out["selected_count"] = len(selected_ids)
    out["excluded_rationale_summary"] = excl
    out["excluded_count"] = sum(excl.values())
    out["excluded_by_low_relevance_count"] = low_rel
    out["excluded_by_cap_count"] = by_cap
    out["reason_each_selected"] = reason_each
    out["selected_by_lane"] = {"primary": len(selected_ids)}
    out["llm_rescued_count"] = len(rescued)
    return out


__all__ = [
    "DEFAULT_MAX_COUNT",
    "EXCLUSION_REASONS",
    "select_default_articles",
    "recount_summary_after_rescue",
]
