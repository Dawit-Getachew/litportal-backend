"""Phase 5A — PICOTT-aware ranker post-processor.

Runs AFTER ``evidence_ranker`` and applies hard-ceiling relevance caps
based on the PICOTT must-have/should-have/ranking/negative concepts. The
post-processor is additive — when ``picott_json`` is absent or has no
must-haves, behaviour is identical to legacy.

Cap precedence (lowest wins, per the user spec):
  - missing_must_have intervention (intervention/index_test/drug)   → 30
  - missing_must_have condition (target disease/condition)           → 50
  - missing abstract AND title has no must-have match                → 40
  - wrong question-type content (study type penalty)                 → 50
  - wrong population (matches drug but not population)               → 60
  - dosing_protocol tiered:
       drug missing                                                   → 30
       disease missing                                                → 60
       dosing context missing                                         → 80
  - negative concept fired (with no waiver)                           → 40
  - imaging_only_for_pharmacology (special-case decoy)                → 40

Each applied cap records a ``cap_reason`` token (closed set) that the UI
surfaces in the "Why ranked here" panel.

OA full-text signals: when an ``article_text_extracts`` row exists for the
article (per-user/per-session/per-article only — NEVER cross-session and
NEVER cross-user), the post-processor also scans the ingestible section
text for ranking concepts and adds a small bounded boost. The 30/50/60/80
caps are still hard ceilings — OA boosts cannot exceed them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from services.study_type_router import study_type_router
from services.cochrane_lane import (
    article_cochrane_provenance,
    cochrane_boost_enabled,
    cochrane_boost_points,
    cochrane_boost_tier,
)


# Closed set of cap reason codes — used by the UI "Why ranked here" panel.
CAP_REASONS = {
    "missing_intervention_anchor",
    "missing_condition_anchor",
    "missing_target_condition",
    "missing_abstract_no_title_match",
    "wrong_question_type_content",
    "wrong_population_correct_intervention",
    "dosing_drug_missing",
    "dosing_disease_missing",
    "dosing_context_missing",
    "negative_concept_fired",
    "imaging_only_for_pharmacology",
    "diagnostic_only_for_therapy",
    "treatment_only_for_diagnosis",
    "case_report_when_higher_evidence_exists",
    # Phase 5F — GLP-1 / LDL specific caps (closed-set, layered on top
    # of the existing generic caps; lowest-wins).
    "tirzepatide_only_for_glp1_question",
    "glp1_biology_no_lipid_outcome",
    "ldl_without_glp1_intervention",
    # Phase 6F — Primary Clinical Entity Anchor caps. These are
    # SUPPLEMENTARY codes appended to ``cap_reasons`` when the
    # picott_aware_ranker confirms that the article fails the primary
    # anchor check. The CAP value is already set by the existing
    # ``missing_relevance_cap`` machinery (now ≤ 20 for primary
    # entities). These codes give the UI + audit a more precise label
    # than the generic "missing_intervention_anchor".
    "wrong_drug",
    "wrong_condition",
    "missing_primary_anchor",
    "wrong_primary_anchor",
    "disease_only_match",
    "drug_safety_no_adverse_effect",
    "treatment_sequencing_wrong_disease",
    # Phase 6G.3 — disease_overview directness caps. Applied ONLY when
    # ``question_subtype ∈ {disease_overview, definition}``. The cap
    # ceilings preserve the Phase 6F invariant that primary anchor
    # caps (≤20) remain stronger than these aspect-specific caps.
    "multi_disease_dilution",
    "aspect_specific_for_overview",
    "treatment_specific_for_overview",
    "environmental_lifestyle_for_overview",
    "pediatric_only_unrequested",
    # Phase 6G.4P — post-procedure management caps. Mirror the Phase
    # 6F invariant: primary-anchor caps stay STRONGER than every
    # boost (including the +35 anchor_match_full boost and the
    # disease_overview directness boosts). Closed-set; lowest-wins
    # under the existing cap arithmetic.
    "missing_procedure_anchor",
    "missing_management_anchor",
    "wrong_procedure",
    "related_but_distinct_procedure",
    "management_only_article",
    "procedure_only_article",
    "noncardiac_surgery_for_cardiac_query",
    "af_only_without_valve_context",
    "pci_only_without_target_procedure_context",
}


# Phase 6G.3 — directness boost lookup. Additive on top of the
# Phase 6F.1 anchor_match_full / partial boost, capped together with
# (not against) the +10 misc-boost clamp. Caps still win.
_DIRECTNESS_BOOSTS: dict[str, int] = {
    "general_overview": 25,
    "guideline_overview": 20,
    "disease_seminar": 20,
    "broad_clinical_review": 15,
    "aspect_specific_review": 0,
    "treatment_specific_review": 0,
    "pathogenesis_specific_review": 0,
    "environmental_or_lifestyle_specific": 0,
    "multi_disease_review": 0,
    "wrong_disease": 0,
    "unrelated": 0,
}

# Phase 6G.3 — directness caps for the disease_overview subtype.
# These cap ceilings clamp the final ``relevance_score`` AND
# ``overall_rank_score`` so an aspect-specific review can never crowd
# out a general overview. None ⇒ no cap.
_DIRECTNESS_CAPS_OVERVIEW: dict[str, int | None] = {
    "general_overview": None,
    "guideline_overview": None,
    "disease_seminar": None,
    "broad_clinical_review": None,
    "aspect_specific_review": 65,
    "treatment_specific_review": 65,
    "pathogenesis_specific_review": 65,
    "environmental_or_lifestyle_specific": 55,
    "multi_disease_review": 70,
    "wrong_disease": None,  # caught by existing missing_primary_anchor
    "unrelated": None,
}

# Cap-reason token to surface for each directness class (when its cap
# applies). Maps to one of the CAP_REASONS set above.
_DIRECTNESS_CAP_REASONS: dict[str, str | None] = {
    "aspect_specific_review": "aspect_specific_for_overview",
    "treatment_specific_review": "treatment_specific_for_overview",
    "pathogenesis_specific_review": "aspect_specific_for_overview",
    "environmental_or_lifestyle_specific": "environmental_lifestyle_for_overview",
    "multi_disease_review": "multi_disease_dilution",
}


# Phase 5F — GLP-1 RA member terms (closed-set). Used by the
# tirzepatide_only_for_glp1_question / glp1_biology_no_lipid_outcome /
# ldl_without_glp1_intervention cap detectors. NEVER hallucinated; this
# list mirrors the curated cardiometabolic_lipids.json specialty pack.
_GLP1_RA_MEMBERS = (
    "semaglutide", "ozempic", "wegovy", "rybelsus",
    "liraglutide", "victoza", "saxenda",
    "dulaglutide", "trulicity",
    "exenatide", "byetta", "bydureon", "exendin-4",
    "lixisenatide", "adlyxin",
    "albiglutide", "tanzeum",
    "glp-1 receptor agonist", "glp-1 receptor agonists",
    "glp1 receptor agonist", "glp-1 ra", "glp1-ra",
    "glucagon-like peptide-1 receptor agonist",
)
_GLP1_FAMILY_TERMS = _GLP1_RA_MEMBERS + ("glp-1", "glp1")
_TIRZEPATIDE_TERMS = ("tirzepatide", "mounjaro", "zepbound",
                       "gip/glp-1", "dual gip and glp-1")
_LIPID_OUTCOMES_TERMS = (
    "ldl", "ldl-c", "ldl cholesterol", "low-density lipoprotein",
    "low density lipoprotein",
    "hdl", "hdl-c", "hdl cholesterol",
    "triglyceride", "triglycerides",
    "total cholesterol", "cholesterol",
    "apolipoprotein", "apob", "apo b",
    "lipoprotein(a)", "lp(a)",
    "non-hdl", "dyslipidemia", "dyslipidaemia",
    "hyperlipidemia", "hypercholesterolemia", "lipid profile", "lipids",
)


# Article types that count as "wrong question-type content" for therapy /
# prevention / pharmacology / dosing_protocol questions.
_DIAGNOSTIC_ONLY = {"diagnostic_accuracy_study", "validation_study"}
_THERAPY_FACING = {"rct", "systematic_review", "meta_analysis", "guideline"}


def _hits_text(haystack: str, needles: list[str]) -> int:
    if not haystack or not needles:
        return 0
    s = haystack.lower()
    n = 0
    for needle in needles:
        t = (needle or "").strip().lower()
        if not t:
            continue
        if re.search(rf"\b{re.escape(t)}\b", s):
            n += 1
    return n


def _matches_any(haystack: str, needles: list[str]) -> bool:
    return _hits_text(haystack, needles) > 0


@dataclass
class _ConceptCheck:
    matched_must_have: list[str]
    missing_must_have: list[str]
    matched_should_have: list[str]
    matched_ranking: list[str]
    fired_penalties: list[dict[str, Any]]
    cap: int
    cap_reasons: list[str]


def _check_concepts(
    title: str,
    abstract: str,
    ingest_full_text: str,
    picott: dict[str, Any],
) -> _ConceptCheck:
    """Apply all PICOTT concept gates to a single article and return the
    aggregated result. Hard-ceiling caps are reduced (i.e., min-collapsed)
    across multiple violations.
    """
    type_of_q = (picott.get("type_of_question") or "other").lower()
    must = picott.get("must_have_concepts") or []
    should = picott.get("should_have_concepts") or []
    ranking = picott.get("ranking_concepts") or []
    negatives = picott.get("negative_or_penalty_concepts") or []

    # We search title + abstract + (OA full text when available).
    body = " ".join([title or "", abstract or "", ingest_full_text or ""])
    title_only = title or ""

    matched_must: list[str] = []
    missing_must: list[str] = []
    cap = 100
    cap_reasons: list[str] = []

    # Per-role tracking so we can apply the global "missing target condition"
    # and "missing intervention" caps, and the dosing tiered rule.
    role_to_present: dict[str, bool] = {}
    role_to_concepts: dict[str, list[str]] = {}

    for mh in must:
        concept = mh.get("concept") or ""
        role = (mh.get("role") or "other").lower()
        terms = [concept] + (mh.get("synonyms") or [])
        anchor_cap = int(mh.get("missing_relevance_cap") or 30)
        is_present = _matches_any(body, terms)
        role_to_present.setdefault(role, False)
        role_to_concepts.setdefault(role, []).append(concept)
        if is_present:
            matched_must.append(concept)
            role_to_present[role] = True
        else:
            missing_must.append(concept)
            if anchor_cap < cap:
                cap = anchor_cap
            # Map role → cap_reason code.
            if role == "intervention":
                if "missing_intervention_anchor" not in cap_reasons:
                    cap_reasons.append("missing_intervention_anchor")
                # Phase 6F — also append the supplementary
                # ``wrong_drug`` code when the question is a
                # drug-safety / pharmacology question. The cap value is
                # already set by ``missing_relevance_cap`` (≤ 20 for
                # primary drug anchors per Phase 6F).
                if (
                    "wrong_drug" not in cap_reasons
                    and type_of_q in {
                        "pharmacology", "dosing_protocol", "therapy",
                    }
                ):
                    cap_reasons.append("wrong_drug")
                if "missing_primary_anchor" not in cap_reasons:
                    cap_reasons.append("missing_primary_anchor")
            elif role in {"condition", "population"}:
                # Phase 6F — both "condition" and "population" roles
                # are treated as disease/problem anchors. PICOTT
                # role_repair often rewrites condition → population
                # for therapy questions (e.g., "psoriasis" or
                # "subarachnoid hemorrhage" lands in the population
                # slot), but the cap semantics are identical: missing
                # the target disease/problem caps relevance ≤ 20.
                if "missing_condition_anchor" not in cap_reasons:
                    cap_reasons.append("missing_condition_anchor")
                if "wrong_condition" not in cap_reasons:
                    cap_reasons.append("wrong_condition")
                if "missing_primary_anchor" not in cap_reasons:
                    cap_reasons.append("missing_primary_anchor")
                # The "missing target condition" cap from the spec also
                # collapses to 50 — which is already the per-anchor cap.

    matched_should: list[str] = []
    for sh in should:
        concept = sh.get("concept") or ""
        terms = [concept] + (sh.get("synonyms") or [])
        if _matches_any(body, terms):
            matched_should.append(concept)

    matched_ranking: list[str] = []
    for rc in ranking:
        concept = rc.get("concept") or ""
        terms = [concept] + (rc.get("synonyms") or [])
        if _matches_any(body, terms):
            matched_ranking.append(concept)

    # Negative concepts → fire only when present AND none of the waiver
    # concepts are present.
    fired_penalties: list[dict[str, Any]] = []
    for neg in negatives:
        concept = neg.get("concept") or ""
        if not concept:
            continue
        unless = neg.get("unless_concepts_present") or []
        if not _matches_any(body, [concept]):
            continue
        # waiver: any of the must-present concepts present?
        if unless and _matches_any(body, unless):
            continue
        fired_penalties.append({
            "concept": concept,
            "penalty_reason": neg.get("penalty_reason") or "negative_concept_fired",
            "penalty_cap": int(neg.get("penalty_cap") or 50),
        })
        penalty_cap = int(neg.get("penalty_cap") or 50)
        if penalty_cap < cap:
            cap = penalty_cap
        # Imaging decoy → specific cap_reason for the UI.
        if any(t in concept.lower() for t in
                ("ct angiography", "cta", "dsa", "transcranial doppler",
                 "perfusion", "mra")):
            if "imaging_only_for_pharmacology" not in cap_reasons:
                cap_reasons.append("imaging_only_for_pharmacology")
        else:
            if "negative_concept_fired" not in cap_reasons:
                cap_reasons.append("negative_concept_fired")

    # Dosing tiered rule (additive on top of global caps; whichever is
    # stricter wins per the user spec).
    if type_of_q == "dosing_protocol":
        # Tier 1 — drug present? If not, cap at 30.
        drug_present = role_to_present.get("intervention", False)
        condition_present = role_to_present.get("condition", False)
        context_present = bool(matched_ranking and any(
            re.search(r"(dose|dosing|infusion|bolus|mcg/kg|mg/kg|"
                       r"microgram/kg|protocol|regimen)",
                       (r or "").lower())
            for r in matched_ranking
        ))
        if not drug_present:
            if 30 < cap:
                # Already covered by missing-intervention-anchor cap, but
                # ensure cap is at most 30.
                cap = min(cap, 30)
            if "dosing_drug_missing" not in cap_reasons:
                cap_reasons.append("dosing_drug_missing")
        elif not condition_present:
            cap = min(cap, 60)
            if "dosing_disease_missing" not in cap_reasons:
                cap_reasons.append("dosing_disease_missing")
        elif not context_present:
            cap = min(cap, 80)
            if "dosing_context_missing" not in cap_reasons:
                cap_reasons.append("dosing_context_missing")

    # Missing abstract AND no title match → 40.
    if not (abstract or "").strip():
        if not _hits_text(title_only, matched_must or [m.get("concept") for m in must]):
            cap = min(cap, 40)
            if "missing_abstract_no_title_match" not in cap_reasons:
                cap_reasons.append("missing_abstract_no_title_match")

    # Phase 5F — GLP-1 / LDL specific caps (closed-set; lowest-wins).
    # Detect "is this a GLP-1/LDL question?" by checking the picott
    # concept block contents (NOT the article).
    question_concept_names = " | ".join(
        list((picott.get("normalized_question") or "").lower().split())
        + [(m.get("concept") or "").lower() for m in must]
        + [(r.get("concept") or "").lower() for r in ranking]
        + [(s.get("concept") or "").lower() for s in should]
    )
    question_targets_glp1 = any(
        t in question_concept_names for t in _GLP1_FAMILY_TERMS
    )
    question_explicitly_targets_tirzepatide = any(
        t in question_concept_names for t in _TIRZEPATIDE_TERMS
    )
    question_targets_lipid = any(
        t in question_concept_names for t in _LIPID_OUTCOMES_TERMS
    )
    body_lower = body.lower()
    article_has_glp1_ra = any(
        re.search(rf"\b{re.escape(t)}\b", body_lower) for t in _GLP1_RA_MEMBERS
    )
    article_has_glp1_family = article_has_glp1_ra or any(
        re.search(rf"\b{re.escape(t)}\b", body_lower)
        for t in ("glp-1", "glp1")
    )
    article_has_tirzepatide = any(
        re.search(rf"\b{re.escape(t)}\b", body_lower) for t in _TIRZEPATIDE_TERMS
    )
    article_has_lipid_outcome = any(
        re.search(rf"\b{re.escape(t)}\b", body_lower) for t in _LIPID_OUTCOMES_TERMS
    )

    # 1) ``tirzepatide_only_for_glp1_question`` — cap=70. Article only
    #    mentions tirzepatide but the question targets GLP-1 RAs (and
    #    does NOT explicitly include tirzepatide).
    if (
        question_targets_glp1
        and not question_explicitly_targets_tirzepatide
        and article_has_tirzepatide
        and not article_has_glp1_ra
    ):
        cap = min(cap, 70)
        if "tirzepatide_only_for_glp1_question" not in cap_reasons:
            cap_reasons.append("tirzepatide_only_for_glp1_question")

    # 2) ``glp1_biology_no_lipid_outcome`` — cap=50. Article mentions
    #    GLP-1 / GLP-1 RA but contains NO lipid outcome term, when the
    #    question explicitly asks about a lipid outcome.
    if (
        question_targets_glp1
        and question_targets_lipid
        and article_has_glp1_family
        and not article_has_lipid_outcome
    ):
        cap = min(cap, 50)
        if "glp1_biology_no_lipid_outcome" not in cap_reasons:
            cap_reasons.append("glp1_biology_no_lipid_outcome")

    # 3) ``ldl_without_glp1_intervention`` — cap=30. Article mentions a
    #    lipid outcome but does NOT mention any GLP-1 RA / GLP-1 term,
    #    when the question requires a GLP-1 RA intervention.
    if (
        question_targets_glp1
        and question_targets_lipid
        and article_has_lipid_outcome
        and not article_has_glp1_family
    ):
        cap = min(cap, 30)
        if "ldl_without_glp1_intervention" not in cap_reasons:
            cap_reasons.append("ldl_without_glp1_intervention")

    # Phase RankPrecision-1 — central condition anchor precision gate.
    # For a single-condition CANCER question, an article whose TITLE is about a
    # DIFFERENT specific cancer (and does NOT name the target cancer in its
    # title) must not earn high relevance from an incidental abstract/full-text
    # mention. Cap wrong_condition. Deterministic; only fires for cancer-target
    # questions (no-op otherwise).
    try:
        from services.central_condition_anchor_gate import (
            PRECISION_CAP as _RP1_CAP,
            evaluate as _rp1_evaluate,
        )
        _rp1 = _rp1_evaluate(title_only, picott)
        if _rp1.get("fires"):
            cap = min(cap, _RP1_CAP)
            for _rc in ("wrong_condition", "missing_condition_anchor",
                        "missing_primary_anchor"):
                if _rc not in cap_reasons:
                    cap_reasons.append(_rc)
    except Exception:  # noqa: BLE001
        pass

    return _ConceptCheck(
        matched_must_have=matched_must,
        missing_must_have=missing_must,
        matched_should_have=matched_should,
        matched_ranking=matched_ranking,
        fired_penalties=fired_penalties,
        cap=cap,
        cap_reasons=cap_reasons,
    )


def _wrong_question_type_cap(art_type: str, type_of_q: str) -> tuple[int | None, str | None]:
    """Return (cap, cap_reason) when the article type is mismatched with the
    question type per the user spec, or (None, None) otherwise.
    """
    if type_of_q in {"therapy", "prevention", "pharmacology", "dosing_protocol"}:
        if art_type in _DIAGNOSTIC_ONLY:
            return 50, ("diagnostic_only_for_therapy"
                        if type_of_q in {"therapy", "prevention"}
                        else "wrong_question_type_content")
    if type_of_q in {"diagnosis", "clinical_exam"}:
        if art_type == "rct":
            # An RCT for a pure-diagnosis question is wrong type unless it
            # contains diagnostic accuracy content (we don't easily know
            # that here, so apply the cap).
            return 50, "treatment_only_for_diagnosis"
    return None, None


class PicottAwareRankerService:
    """Post-process ranker output with PICOTT anchors + caps + boosts."""

    def postprocess(
        self,
        ranked: list[dict[str, Any]],
        picott: dict[str, Any] | None,
        *,
        article_text_extracts_by_article: dict[str, str] | None = None,
        query_family_by_canonical: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Apply Phase 5A caps + boosts to the ranker output IN PLACE and
        return the same list (sorted descending by overall_rank_score)."""
        if not ranked or not picott:
            return ranked
        has_anchors = bool(picott.get("must_have_concepts"))
        if not has_anchors and not picott.get("negative_or_penalty_concepts"):
            return ranked

        type_of_q = (picott.get("type_of_question") or "other").lower()
        # Phase 6G.3 — disease_overview / definition subtype routes
        # into the directness classifier path below. We resolve the
        # closed-set subtype set once per call.
        question_subtype = (picott.get("question_subtype") or "").lower()
        is_overview_route = question_subtype in {
            "disease_overview", "definition",
        }
        # Phase K4A.2 — severity-marker overview subtype gates a
        # SEPARATE deterministic directness classifier (closed-set
        # labels distinct from the disease_overview classifier). The
        # two routes are mutually exclusive — only one runs per
        # session, so existing disease_overview behavior is unchanged.
        is_severity_marker_route = question_subtype == "severity_marker_overview"
        severity_anchor = ""
        if is_severity_marker_route:
            severity_anchor = (
                picott.get("severity_marker_condition_anchor") or ""
            ).strip().lower()
        # Phase K4A.3 — diagnostic rule-out / test-utility route gate.
        # Closed-set anchors come from the pipeline injection, NOT from
        # the LLM parser (which produces an empty PICOTT on this
        # question shape). Mutually exclusive with the severity-marker
        # route.
        is_rule_out_route = question_subtype == "rule_out_test_utility"
        rule_out_index_test_anchor = ""
        rule_out_target_condition_anchor = ""
        if is_rule_out_route:
            rule_out_index_test_anchor = (
                picott.get("rule_out_index_test_anchor") or ""
            ).strip()
            rule_out_target_condition_anchor = (
                picott.get("rule_out_target_condition_anchor") or ""
            ).strip()
        # Phase K4A.4 — incidence / risk epidemiology route gate.
        # Closed-set anchors come from the pipeline injection. Mutually
        # exclusive with the severity-marker AND rule-out routes — only
        # one directness route fires per session, so the other question
        # families are unaffected.
        is_incidence_risk_route = (
            question_subtype == "epidemiology_incidence_complication_risk"
        )
        incidence_outcome_anchor = ""
        incidence_risk_context_anchor = ""
        if is_incidence_risk_route:
            incidence_outcome_anchor = (
                picott.get("incidence_outcome_anchor") or ""
            ).strip()
            incidence_risk_context_anchor = (
                picott.get("incidence_risk_context_anchor") or ""
            ).strip()
        # Resolve the primary disease anchor (canonical + synonyms)
        # for the classifier. We pull the first ``condition``-roled
        # must_have, falling back to the first must_have entry.
        primary_anchor_canonical = ""
        primary_anchor_synonyms: list[str] = []
        if is_overview_route:
            for mh in (picott.get("must_have_concepts") or []):
                if not isinstance(mh, dict):
                    continue
                role = (mh.get("role") or "").lower()
                if role in {"condition", "population", "patient_group"}:
                    primary_anchor_canonical = mh.get("concept") or ""
                    primary_anchor_synonyms = list(mh.get("synonyms") or [])
                    break
            if not primary_anchor_canonical:
                first = next(iter(picott.get("must_have_concepts") or []), {})
                if isinstance(first, dict):
                    primary_anchor_canonical = first.get("concept") or ""
                    primary_anchor_synonyms = list(first.get("synonyms") or [])
        pathway = study_type_router.pathway(type_of_q)
        boosts_by_type = pathway.ranking_boosts or {}
        extracts = article_text_extracts_by_article or {}
        family_map = query_family_by_canonical or {}

        # Phase 6B.2 — role-weighted ranking. Compute the multiplier from
        # the closed-set role weights table for ``type_of_question``.
        # The multiplier is applied to ``overall_rank_score`` BEFORE the
        # mandatory-anchor cap clamp so the caps remain stronger than
        # role weights.
        from services.role_weighting import (  # noqa: WPS433
            ROLE_WEIGHTS, _canonical_role,
        )
        role_table = (
            ROLE_WEIGHTS.get(type_of_q)
            or ROLE_WEIGHTS.get("other")
            or {}
        )
        # Pre-compute per-role term lists so we can re-use them for the
        # role-match breakdown without re-walking must_have_concepts.
        must_haves = picott.get("must_have_concepts") or []
        terms_by_role: dict[str, list[str]] = {}
        for mh in must_haves:
            role = _canonical_role(mh.get("role") or "other")
            terms = [mh.get("concept") or ""] + list(
                mh.get("synonyms") or []
            )
            terms_by_role.setdefault(role, []).extend(
                [t for t in terms if t]
            )

        for r in ranked:
            art = r.get("_article") or {}
            ck = art.get("canonical_key") or ""
            title = art.get("title") or ""
            abstract = art.get("abstract") or ""
            ingest_text = extracts.get(ck, "")  # per-session/per-article ONLY

            checks = _check_concepts(title, abstract, ingest_text, picott)

            # ---- Phase 6B.2 — role-match breakdown ------------------
            body_for_roles = " ".join([title, abstract, ingest_text])
            role_match_breakdown: list[dict[str, Any]] = []
            role_match_total = 0
            for role, role_terms in terms_by_role.items():
                if not role_terms:
                    continue
                hit = _hits_text(body_for_roles, role_terms)
                weight = int(role_table.get(role, 5))
                # match_fraction: how many of the role's distinct terms
                # hit (0.0..1.0). One hit out of N is enough for credit
                # because synonyms are alternate forms.
                match_fraction = (
                    1.0 if hit > 0 else 0.0
                )
                contribution = int(round(weight * match_fraction))
                role_match_total += contribution
                role_match_breakdown.append({
                    "role": role,
                    "weight_pct": weight,
                    "matched": hit > 0,
                    "match_fraction": match_fraction,
                    "contribution_pct": contribution,
                })
            # Normalize total to a 0..100 role_match_score.
            denom = sum(role_table.get(b["role"], 5)
                        for b in role_match_breakdown) or 100
            role_match_score = int(round(100 * role_match_total / denom))
            role_match_score = max(0, min(100, role_match_score))
            r["role_match_score"] = role_match_score
            r["role_match_breakdown_json"] = role_match_breakdown
            r["role_weights_used"] = dict(role_table)

            # ---- Role-weighted multiplier --------------------------
            # Convert role_match_score (0..100) to a multiplier in
            # [0.85 .. 1.15] BEFORE caps. Capped articles still get the
            # multiplier but the caps clamp the final score below.
            if role_match_score >= 75:
                role_mult = 1.15
            elif role_match_score >= 50:
                role_mult = 1.07
            elif role_match_score >= 25:
                role_mult = 1.0
            else:
                role_mult = 0.92

            # Wrong-question-type cap.
            art_type = art.get("article_type_classification") or "other"
            wqt_cap, wqt_reason = _wrong_question_type_cap(art_type, type_of_q)
            cap = checks.cap
            cap_reasons = list(checks.cap_reasons)
            if wqt_cap is not None and wqt_cap < cap:
                cap = wqt_cap
                if wqt_reason and wqt_reason not in cap_reasons:
                    cap_reasons.append(wqt_reason)

            # Ranking-concept boost (small, bounded).
            boost = 0
            # Phase 5D Fix 5 — collect a structured list of closed-set
            # boost reason codes so the admin diagnostics and clinician
            # article-list endpoints can show *why* an article was
            # boosted. Each entry uses snake_case `code(+N)` form.
            boost_reasons: list[str] = []
            if checks.matched_ranking:
                ranking_boost = min(8, 2 * len(checks.matched_ranking))
                boost += ranking_boost
                boost_reasons.append(
                    f"matched_{len(checks.matched_ranking)}_ranking_concepts"
                    f"(+{ranking_boost})"
                )
            # Study-design pathway boost.
            if art_type in boosts_by_type:
                # Half-strength here because the base relevance score
                # already weights design. Cap-aware.
                design_boost = boosts_by_type[art_type] // 2
                if design_boost > 0:
                    boost += design_boost
                    boost_reasons.append(
                        f"study_design_{art_type}(+{design_boost})"
                    )

            # Query-family signal: high_precision + study_type_filtered
            # matches get an extra +5 (only if must-haves are satisfied,
            # per the spec — do not over-boost broad_recall-only articles).
            # Phase 5D — mesh_keyword / clinical_queries / trial_acronym
            # also receive the +5 boost when must-haves are satisfied.
            families = family_map.get(ck) or []
            cb_families = {
                f.replace("cb_", "") if isinstance(f, str) else f
                for f in families
            }
            high_signal_families = {
                "high_precision", "study_type_filtered",
                "mesh_keyword", "clinical_queries", "trial_acronym",
            }
            matched_high_signal = cb_families & high_signal_families
            if (
                not checks.missing_must_have
                and matched_high_signal
            ):
                boost += 5
                boost_reasons.append(
                    "must_haves_satisfied_"
                    f"{'+'.join(sorted(matched_high_signal))}(+5)"
                )
            # Articles retrieved ONLY by broad_recall: NO extra boost.

            # Phase 5D Fix 5 — per-article boost cap is +10. The reasons
            # list keeps EVERY reason that fired (transparency) but the
            # applied total is clamped. This clamp is computed BEFORE
            # the score arithmetic so the cap is respected by both
            # overall_rank_score and the surfaced ``ranking_boost_total``.
            #
            # Phase 6F.1 — the new ``anchor_match_*`` boost is computed
            # AFTER this +10 clamp (see below) and is INTENTIONALLY not
            # bound by it. Justification: the +10 cap was designed to
            # prevent ranking-concept + study-design + query-family
            # signals from over-boosting an article that doesn't
            # actually match the must-haves. The anchor-match boost
            # has a tighter precondition (ALL required must-haves
            # matched AND no wrong-anchor cap_reason) and a strictly
            # closed-set bound of +35 / +15 / +10, so it deserves its
            # own slot. The wrong-anchor cap STILL clamps the final
            # score so the anchor boost can never rescue a capped
            # article.
            boost_uncapped = boost
            boost = min(boost, 10)
            if boost_uncapped > 10:
                boost_reasons.append(
                    f"boost_total_capped_at_10(was_{boost_uncapped})"
                )

            # ----------------------------------------------------------
            # Phase 6F.1 — positive anchor-match boost.
            #
            # Diagnosis (Phase 6F.1 tester verdict): the matcher
            # correctly populates ``matched_must_have`` for on-target
            # articles, but the scorer never boosts ``relevance_score``
            # when ALL required must-haves match — it only clamps the
            # score DOWN via the wrong-anchor caps. Net effect:
            # legitimate on-target articles plateau below the 60
            # default-selection floor while wrong-anchor articles are
            # correctly capped at ≤20. This boost is the missing
            # symmetric positive component.
            #
            # Closed-set rules (NEVER weaken):
            #   1. The boost fires ONLY when ``cap_reasons`` is empty.
            #      An article that carries ANY wrong-anchor cap_reason
            #      is INELIGIBLE — the cap stays the authoritative
            #      verdict. This preserves the Phase 6F invariant that
            #      anchor caps win over all other boosts.
            #   2. Among the ``must_have_concepts`` flagged
            #      ``required_for_high_relevance=true``:
            #        - ALL required matched  → ``anchor_match_full``
            #          (+ANCHOR_MATCH_FULL_BOOST to BOTH
            #          relevance_score and overall_rank_score).
            #        - Primary anchor matched (intervention OR
            #          condition/population) but a SECONDARY required
            #          anchor missing (e.g., outcome) → partial boost.
            #        - No required match → no anchor boost (cap path
            #          handles this).
            #   3. The boost is APPLIED AFTER the +10 clamp and
            #      BEFORE the cap-clamp on overall_rank_score AND
            #      relevance_score below — so anchor caps still win.
            #   4. The boost is matched on TITLE + ABSTRACT scope only
            #      (no full text). An anchor that only appears in
            #      uploaded ingest text is too weak a signal for the
            #      full +35.
            #   5. Closed-set logging in ``ranking_boost_reasons`` so
            #      the admin diagnostics + clinician panel can show
            #      the provenance.
            anchor_match_boost = 0
            anchor_match_label: str | None = None
            if must_haves and not cap_reasons:
                required_concepts: list[dict[str, Any]] = [
                    mh for mh in must_haves
                    if bool(mh.get("required_for_high_relevance"))
                ]
                if not required_concepts:
                    # Spec fallback: when no must_have explicitly flags
                    # ``required_for_high_relevance``, treat ALL
                    # must_haves as required (every entry came through
                    # _build_must_have_for which already sets the flag).
                    required_concepts = list(must_haves)
                anchor_body = " ".join([title or "", abstract or ""])
                matched_required: list[dict[str, Any]] = []
                missed_required: list[dict[str, Any]] = []
                for mh in required_concepts:
                    terms = [mh.get("concept") or ""] + list(
                        mh.get("synonyms") or []
                    )
                    if _matches_any(anchor_body, terms):
                        matched_required.append(mh)
                    else:
                        missed_required.append(mh)
                missed_roles = {
                    (m.get("role") or "other").lower()
                    for m in missed_required
                }
                primary_roles = {
                    "intervention", "exposure", "condition", "population",
                    "index_test", "reference_standard",
                }
                outcome_roles = {"outcome", "target_diagnosis"}
                if matched_required and not missed_required:
                    anchor_match_boost = 35
                    anchor_match_label = "anchor_match_full"
                elif matched_required and missed_roles.issubset(outcome_roles):
                    anchor_match_boost = 15
                    anchor_match_label = "anchor_match_partial_outcome"
                elif matched_required and missed_roles.issubset(primary_roles):
                    anchor_match_boost = 10
                    anchor_match_label = "anchor_match_partial_intervention"
                else:
                    anchor_match_boost = 0
                    anchor_match_label = None
            if anchor_match_boost:
                boost_reasons.append(
                    f"{anchor_match_label}(+{anchor_match_boost})"
                )

            # Phase 5D Fix 5 — per-article boost cap is +10. The reasons
            # list keeps EVERY reason that fired (transparency) but the
            # applied total is clamped. This clamp is computed BEFORE
            # the score arithmetic so the cap is respected by both
            # overall_rank_score and the surfaced ``ranking_boost_total``.
            boost_uncapped = boost
            boost = min(boost, 10)
            if boost_uncapped > 10:
                boost_reasons.append(
                    f"boost_total_capped_at_10(was_{boost_uncapped})"
                )

            current = int(r.get("overall_rank_score") or 0)
            # Phase 6B.2 — apply the role-weighted multiplier BEFORE the
            # mandatory-anchor cap clamp. The clamp below will still cap
            # the final score so role weights CANNOT override caps.
            #
            # Phase 6F.1 — the ``anchor_match_boost`` is added OUTSIDE
            # the +10 boost cap (intentionally) but BEFORE the
            # wrong-anchor cap clamp, so anchor caps still win.
            new_score = int(round(current * role_mult)) + boost + anchor_match_boost

            # ----------------------------------------------------------
            # Phase 6G.3 — disease_overview directness boost + cap.
            # Runs ONLY for ``question_subtype ∈ {disease_overview,
            # definition}`` AND only when no wrong-anchor cap fired (so
            # primary-anchor caps still win, per the Phase 6F
            # invariant). Boosts add ON TOP of anchor_match_boost;
            # caps clamp the final overall_rank_score AND
            # relevance_score, just like all other Phase 6F.1 caps.
            directness_label: str | None = None
            directness_boost = 0
            directness_cap: int | None = None
            if is_overview_route and primary_anchor_canonical:
                try:
                    from services.overview_directness_classifier import (
                        classify_overview_directness,
                    )
                    directness_label = classify_overview_directness(
                        art,
                        primary_anchor_canonical,
                        primary_anchor_synonyms,
                    )
                except Exception:  # noqa: BLE001
                    directness_label = None
                if directness_label and not cap_reasons:
                    directness_boost = _DIRECTNESS_BOOSTS.get(
                        directness_label, 0,
                    )
                if directness_label:
                    directness_cap = _DIRECTNESS_CAPS_OVERVIEW.get(
                        directness_label,
                    )
                    cr = _DIRECTNESS_CAP_REASONS.get(directness_label)
                    if directness_cap is not None and cr and cr not in cap_reasons:
                        cap_reasons.append(cr)
            # Phase K4A.2 — severity-marker overview directness.
            # Runs ONLY for ``question_subtype == severity_marker_overview``.
            # Boost/cap tables live in services/severity_marker_overview.py.
            if is_severity_marker_route and severity_anchor:
                try:
                    from services.severity_marker_overview import (
                        classify_severity_marker_directness,
                        DIRECTNESS_BOOSTS as _SMO_BOOSTS,
                        DIRECTNESS_CAPS as _SMO_CAPS,
                        DIRECTNESS_CAP_REASONS as _SMO_CAP_REASONS,
                    )
                    directness_label = classify_severity_marker_directness(
                        art,
                        condition_anchor=severity_anchor,
                        condition_synonyms=(),
                    )
                except Exception:  # noqa: BLE001
                    directness_label = None
                if directness_label and not cap_reasons:
                    directness_boost = _SMO_BOOSTS.get(directness_label, 0)
                if directness_label:
                    smo_cap = _SMO_CAPS.get(directness_label)
                    if smo_cap is not None:
                        directness_cap = smo_cap
                    cr = _SMO_CAP_REASONS.get(directness_label)
                    if smo_cap is not None and cr and cr not in cap_reasons:
                        cap_reasons.append(cr)
            # Phase K4A.3 — diagnostic rule-out / test-utility directness.
            # Runs ONLY for ``question_subtype == rule_out_test_utility``.
            # Closed-set 10-label classifier in services/rule_out_test_utility.py.
            # Mutually exclusive with the severity-marker route — only
            # one fires per session, so other question families are
            # unaffected.
            if is_rule_out_route:
                try:
                    from services.rule_out_test_utility import (
                        classify_diagnostic_directness,
                        DIAGNOSTIC_DIRECTNESS_BOOSTS as _RO_BOOSTS,
                        DIAGNOSTIC_DIRECTNESS_CAPS as _RO_CAPS,
                        DIAGNOSTIC_DIRECTNESS_CAP_REASONS as _RO_CAP_REASONS,
                    )
                    directness_label = classify_diagnostic_directness(
                        art,
                        index_test_anchor=rule_out_index_test_anchor
                        or "MRSA nasal screening",
                        target_condition_anchor=rule_out_target_condition_anchor
                        or "MRSA pneumonia",
                    )
                except Exception:  # noqa: BLE001
                    directness_label = None
                if directness_label and not cap_reasons:
                    directness_boost = _RO_BOOSTS.get(directness_label, 0)
                if directness_label:
                    ro_cap = _RO_CAPS.get(directness_label)
                    if ro_cap is not None:
                        directness_cap = ro_cap
                    cr = _RO_CAP_REASONS.get(directness_label)
                    if ro_cap is not None and cr and cr not in cap_reasons:
                        cap_reasons.append(cr)
            # Phase K4A.4 — incidence / risk epidemiology directness.
            # Runs ONLY for ``question_subtype ==
            # epidemiology_incidence_complication_risk``. Closed-set
            # 10-label classifier in
            # services/incidence_risk_epidemiology.py. Mutually
            # exclusive with the severity-marker AND rule-out routes
            # — only one directness route fires per session.
            if is_incidence_risk_route:
                try:
                    from services.incidence_risk_epidemiology import (
                        classify_incidence_risk_directness,
                        INCIDENCE_RISK_DIRECTNESS_BOOSTS as _IR_BOOSTS,
                        INCIDENCE_RISK_DIRECTNESS_CAPS as _IR_CAPS,
                        INCIDENCE_RISK_DIRECTNESS_CAP_REASONS as _IR_CAP_REASONS,
                    )
                    directness_label = classify_incidence_risk_directness(
                        art,
                        outcome_anchor=incidence_outcome_anchor
                        or "aspiration pneumonia",
                        risk_context_anchor=incidence_risk_context_anchor
                        or "low Glasgow Coma Scale",
                    )
                except Exception:  # noqa: BLE001
                    directness_label = None
                if directness_label and not cap_reasons:
                    directness_boost = _IR_BOOSTS.get(directness_label, 0)
                if directness_label:
                    ir_cap = _IR_CAPS.get(directness_label)
                    if ir_cap is not None:
                        directness_cap = ir_cap
                    cr = _IR_CAP_REASONS.get(directness_label)
                    if ir_cap is not None and cr and cr not in cap_reasons:
                        cap_reasons.append(cr)
            if directness_boost:
                new_score += directness_boost
                boost_reasons.append(
                    f"directness_{directness_label}(+{directness_boost})"
                )
            # Fold the directness cap (if any) into the running cap.
            # The lowest-wins semantic preserves the Phase 6F invariant.
            if directness_cap is not None:
                cap = min(cap, directness_cap)

            # ----------------------------------------------------------
            # Phase K5.2C-0 — PubMed-indexed Cochrane review lane (PREVIEW).
            # Detection (labeling) is independent of the boost: a Cochrane
            # review is labeled even when capped, so the clinician badge
            # always shows. The BOUNDED, cap-limited, directness-gated,
            # lane-matched boost fires ONLY when:
            #   * the article is a PubMed-indexed Cochrane review,
            #   * NO relevance cap fired (cap_reasons empty), AND
            #   * the question has must-have anchors and ALL of them matched.
            # This guarantees Cochrane status can NEVER override wrong-disease
            # / wrong-outcome / wrong-framework / missing-primary-anchor /
            # policy / retraction / preprint caps — caps always dominate.
            cochrane_prov = article_cochrane_provenance(art)
            r["cochrane_review_lane"] = bool(cochrane_prov)
            r["cochrane_review_provenance"] = cochrane_prov
            # Compute the would-be boost (directly-relevant gate). It is
            # recorded as a SHADOW value regardless of environment, but only
            # APPLIED to the visible score when ``cochrane_boost_enabled()``
            # (preview/dev always; production only with
            # COCHRANE_PROD_BOOST_ALLOWED=true). This guarantees zero
            # production ranking change from K5.2C unless explicitly opted in.
            _coch_pts = 0
            if (
                cochrane_prov
                and must_haves
                and not cap_reasons
                and not checks.missing_must_have
            ):
                _coch_pts = cochrane_boost_points(picott)
            r["cochrane_review_shadow_boost"] = int(_coch_pts)
            if _coch_pts:
                if cochrane_boost_enabled():
                    new_score += _coch_pts
                    boost_reasons.append(
                        f"cochrane_review_{cochrane_boost_tier(picott)}"
                        f"(+{_coch_pts})"
                    )
                else:
                    # Production default: recorded but NOT applied.
                    boost_reasons.append(
                        f"cochrane_review_shadow_only"
                        f"(+0,would_be_{_coch_pts})"
                    )

            # Hard-ceiling caps applied last so boosts never escape the cap.
            # We clamp BOTH the user-facing ``relevance_score`` and the
            # composite ``overall_rank_score`` so the diagnostics view and
            # the ranked-list view agree.
            if cap < 100:
                new_score = min(new_score, cap)
            new_score = max(0, min(100, new_score))

            r["overall_rank_score"] = int(new_score)
            # Clamp relevance_score to the cap as well (lowest applicable
            # cap is the *ceiling*; lower scores are still allowed).
            #
            # Phase 6F.1 — apply the anchor_match_boost to the
            # user-facing relevance_score too. The boost only fires
            # when cap_reasons is empty (see guard above), so the
            # ``cap < 100`` clamp path will NOT be taken when the
            # boost applies; but we keep both branches defensive so
            # that even if a future change introduces a cap with a
            # non-zero anchor_match_boost, the cap still wins.
            raw_rel = float(r.get("relevance_score") or 0)
            boosted_rel = max(0.0, min(100.0, raw_rel + float(anchor_match_boost)))
            if cap < 100:
                boosted_rel = min(boosted_rel, float(cap))
            r["relevance_score"] = boosted_rel
            r["matched_must_have"] = checks.matched_must_have
            r["missing_must_have"] = checks.missing_must_have
            r["matched_should_have"] = checks.matched_should_have
            r["matched_ranking"] = checks.matched_ranking
            r["penalty_concepts"] = [p["concept"] for p in checks.fired_penalties]
            r["query_family_matches"] = families
            r["relevance_cap_applied"] = int(cap) if cap < 100 else None
            r["cap_reasons"] = cap_reasons
            # Phase 6G.3 — surface the directness classifier verdict
            # for the UI ``article-card-directness-badge``. None when
            # the question subtype is not a disease_overview route.
            r["directness_label"] = directness_label
            # Phase 5D Fix 5 — surface structured boost provenance.
            # ``ranking_boost_reasons`` is always a list (possibly empty);
            # ``ranking_boost_total`` is the integer points added before
            # the cap clamp (so admins can see if a boost was capped away).
            r["ranking_boost_reasons"] = boost_reasons
            r["ranking_boost_total"] = int(boost)

            # Extend rank_explanation with the cap reason if any.
            extra = []
            if cap < 100:
                extra.append(
                    f"Cap {cap} applied "
                    f"({', '.join(cap_reasons) or 'no_specific_reason'})."
                )
            if checks.missing_must_have:
                extra.append(
                    "Missing required concept(s): "
                    + ", ".join(checks.missing_must_have)
                )
            if extra:
                base = r.get("rank_explanation") or ""
                r["rank_explanation"] = (base + " " + " ".join(extra)).strip()

        # Re-sort after caps may have lowered scores.
        ranked.sort(key=lambda r: r.get("overall_rank_score", 0), reverse=True)
        return ranked


picott_aware_ranker = PicottAwareRankerService()
