"""Phase 5A — PICOTT extractor.

Takes a sanitized question and produces:
  - normalized question + spelling corrections
  - question scope (background/foreground/hybrid)
  - PICOTT structured shape (models/picott.py)

The legacy ``ParsedPICO`` (top-level fields like ``population``,
``intervention_or_exposure``, ``clinical_question_type``) is also returned
so existing UI/summarizer keep working with no change. ``type_of_question``
is mapped from the legacy ``clinical_question_type`` plus extra heuristics
(dosing_protocol, clinical_exam, etc.).
"""
from __future__ import annotations

import re
from typing import Any

from models.picott import PICOTT, PICOBlocks, SpellingCorrection
from services.clinical_concept_translation import clinical_concept_translation
from services.concept_expansion import (
    DRUG_GENERIC_TO_CLASS,
    IMAGING_DECOY_TERMS,
    applicable_specialty_packs,
    expand_concept,
    neurocrit_dosing_context_terms,
    neurocrit_ranking_terms,
)
from services.question_scope_classifier import question_scope_classifier
from services.spelling_normalizer import spelling_normalizer
from services.study_type_router import legacy_to_new, study_type_router


# ---------------------------------------------------------------------------
# Heuristics that promote legacy types to more specific new types.
# ---------------------------------------------------------------------------

_DOSING_WORDS = re.compile(
    r"\b(dose|doses|dosing|infusion|bolus|loading dose|maintenance dose|"
    r"protocol|regimen|mcg/kg/min|microgram/kg/min|mg/kg|ng/kg/min|"
    r"continuous infusion|titration)\b",
    re.IGNORECASE,
)
_PHYSICAL_EXAM_WORDS = re.compile(
    r"\b(physical exam|murmur|sign of|maneuver|maneuvers|"
    r"clinical sign|history\s+and\s+examination|bedside)\b",
    re.IGNORECASE,
)
_COST_WORDS = re.compile(
    r"\b(cost|cost-effective|cost effective|cost-effectiveness|"
    r"qaly|incremental cost|economic evaluation|health economics)\b",
    re.IGNORECASE,
)


_ETIOLOGY_HARM_WORDS = re.compile(
    r"\b(cause|causes|risk of|associated with|adverse effect|"
    r"harm|adverse event|side effect)\b", re.IGNORECASE,
)


# Phase 6F — drug-safety trigger phrases. When BOTH a drug anchor and
# any of these phrases are present, the question is a drug_safety
# pharmacology question (Q2 family — "side effects of Seroquel").
_DRUG_SAFETY_WORDS = re.compile(
    r"\b("
    r"side\s+effects?|"
    r"adverse\s+effects?|"
    r"adverse\s+reactions?|"
    r"adverse\s+events?|"
    r"toxicity|"
    r"tolerability|"
    r"drug\s+safety|"
    r"medication\s+safety|"
    r"safety\s+profile|"
    r"safety\s+of|"
    r"complications?\s+of|"
    r"long[-\s]term\s+effects?|"
    r"treatment[-\s]emergent\s+adverse\s+events?"
    r")\b",
    re.IGNORECASE,
)


# Phase 6F — treatment-sequencing trigger phrases. When BOTH a disease
# anchor and any of these phrases are present, the question is a
# treatment_sequencing therapy question (Q1 family — "first-line
# treatment and second-line treatment for psoriasis").
_TREATMENT_SEQUENCING_WORDS = re.compile(
    r"\b("
    r"first[-\s]line\s+(?:treatment|therapy)|"
    r"second[-\s]line\s+(?:treatment|therapy)|"
    r"third[-\s]line\s+(?:treatment|therapy)|"
    r"initial\s+(?:treatment|therapy)|"
    r"next\s+(?:treatment|therapy)|"
    r"treatment\s+algorithm|"
    r"management\s+algorithm|"
    r"recommended\s+treatment|"
    r"standard\s+(?:treatment|of\s+care)|"
    r"treatment\s+options|"
    r"therapy\s+sequence|"
    r"after\s+failure\s+of\s+first[-\s]line"
    r")\b",
    re.IGNORECASE,
)


def _promote_type_of_question(
    question: str, legacy_type: str, *,
    drug_anchors: list[str] | None = None,
    condition_anchors: list[str] | None = None,
) -> tuple[str, str | None]:
    """Return ``(type_of_question, question_subtype)`` using the legacy
    type plus phrase heuristics. Subtype is a short human label.

    Phase 5A: drug_anchors lets us promote even "other" → "dosing_protocol"
    when a drug + dosing words are both present (e.g., "What is the dose of
    milrinone for SAH?").

    Phase 6F: ``condition_anchors`` lets us promote "therapy" →
    "treatment_sequencing" when a disease + treatment-sequencing words
    are both present, and promote any base with a drug + drug-safety
    words into "pharmacology / drug_safety".
    """
    base_new = legacy_to_new(legacy_type)
    q = question or ""
    has_dosing = bool(_DOSING_WORDS.search(q))
    has_drug = bool(drug_anchors)
    has_condition = bool(condition_anchors)
    has_etiology = bool(_ETIOLOGY_HARM_WORDS.search(q))
    has_drug_safety = bool(_DRUG_SAFETY_WORDS.search(q))
    has_treatment_sequencing = bool(_TREATMENT_SEQUENCING_WORDS.search(q))

    # Phase 6F — drug + drug-safety phrase → pharmacology / drug_safety.
    # Must check BEFORE the generic dosing/etiology promotions because
    # "side effects of Seroquel" matches both _DRUG_SAFETY_WORDS and
    # _ETIOLOGY_HARM_WORDS (side effect).
    if has_drug and has_drug_safety:
        return "pharmacology", "drug_safety"

    # Strong promotion: any base + drug + dosing words → dosing_protocol.
    if has_dosing and has_drug:
        return "dosing_protocol", "dosing or protocol question"
    # Pharmacology + dosing words → dosing_protocol.
    if base_new == "pharmacology" and has_dosing:
        return "dosing_protocol", "dosing or protocol question"
    # Therapy + dosing words → dosing_protocol.
    if base_new == "therapy" and has_dosing:
        return "dosing_protocol", "dosing or protocol question"

    # Phase 6F — disease + treatment-sequencing phrase → therapy /
    # treatment_sequencing. Beats the plain "therapy" promotion below.
    if has_condition and has_treatment_sequencing:
        return "therapy", "treatment_sequencing"

    # Etiology/harm: "cause", "risk of", "associated with" → etiology_harm.
    if has_etiology and base_new in {"other", "pharmacology", "therapy"}:
        return "etiology_harm", "etiology / harm question"
    # Diagnosis with explicit physical-exam framing → clinical_exam.
    if base_new == "diagnosis" and _PHYSICAL_EXAM_WORDS.search(q):
        return "clinical_exam", "physical examination accuracy"
    if _COST_WORDS.search(q):
        return "cost", "economic / cost-effectiveness"
    # Drug present (no dosing words) — promote "other" to pharmacology so the
    # study-type router can pick the right pathway.
    if has_drug and base_new == "other":
        return "pharmacology", "drug question"
    return base_new, None


# ---------------------------------------------------------------------------
# Concept extraction
# ---------------------------------------------------------------------------

# Drugs we are willing to auto-mark as must-have intervention even when the
# legacy parser couldn't pin them. Conservative — only drugs whose name is
# unambiguous and not also a common English word.
_KNOWN_DRUGS_FOR_ANCHORING: set[str] = set(DRUG_GENERIC_TO_CLASS.keys()) | {
    # Drug-class anchors safe to detect by surface form.
    "NSAID", "NSAIDs",
    "DOAC", "DOACs",
    "SGLT2 inhibitor", "SGLT2 inhibitors",
    "GLP-1 agonist", "GLP-1 agonists",
    "statin", "statins",
    "ACE inhibitor", "ACE inhibitors",
    "beta blocker", "beta blockers",
    # Phase 6F — atypical antipsychotic drug anchors. Both generic
    # (quetiapine, quetiapine fumarate) and brand (Seroquel, Seroquel XR)
    # forms are detected. NEVER add a drug whose name is a common
    # English word.
    "quetiapine", "quetiapine fumarate",
    "Seroquel", "Seroquel XR",
    "olanzapine", "Zyprexa",
    "risperidone", "Risperdal",
    "aripiprazole", "Abilify",
    "clozapine", "Clozaril",
    "ziprasidone", "Geodon",
    "lurasidone", "Latuda",
    "atypical antipsychotic", "atypical antipsychotics",
    "second-generation antipsychotic", "second-generation antipsychotics",
    # Phase 6F — psoriasis-related drugs commonly named in derm questions.
    # The class anchor is also added so the ranker can credit
    # class-named papers.
    "methotrexate",
    "cyclosporine", "cyclosporin", "ciclosporin",
    "acitretin",
    "apremilast", "Otezla",
    "adalimumab", "Humira",
    "etanercept", "Enbrel",
    "infliximab", "Remicade",
    "secukinumab", "Cosentyx",
    "ixekizumab", "Taltz",
    "ustekinumab", "Stelara",
    "guselkumab", "Tremfya",
    "risankizumab", "Skyrizi",
    "tildrakizumab",
    "brodalumab",
    "biologic agent", "biologic agents", "biologic therapy", "biologics",
}

# Disease/condition phrases that — if mentioned — are anchor candidates.
_CONDITION_ANCHOR_PATTERNS = [
    (re.compile(r"\bsubarachnoid hemorrhage\b", re.IGNORECASE), "subarachnoid hemorrhage"),
    (re.compile(r"\bSAH\b"), "subarachnoid hemorrhage"),
    (re.compile(r"\baSAH\b"), "aneurysmal subarachnoid hemorrhage"),
    (re.compile(r"\b(?:cerebral\s+)?vasospasm\b", re.IGNORECASE), "cerebral vasospasm"),
    (re.compile(r"\bnonvalvular atrial fibrillation\b", re.IGNORECASE), "nonvalvular atrial fibrillation"),
    (re.compile(r"\batrial fibrillation\b", re.IGNORECASE), "atrial fibrillation"),
    (re.compile(r"\bheart failure with preserved ejection fraction\b", re.IGNORECASE), "HFpEF"),
    (re.compile(r"\bHFpEF\b"), "HFpEF"),
    (re.compile(r"\bHFrEF\b"), "HFrEF"),
    (re.compile(r"\bacute kidney injury\b", re.IGNORECASE), "acute kidney injury"),
    (re.compile(r"\bchronic kidney disease\b", re.IGNORECASE), "chronic kidney disease"),
    (re.compile(r"\bCKD\b"), "chronic kidney disease"),
    (re.compile(r"\bAKI\b"), "acute kidney injury"),
    (re.compile(r"\bpulmonary embolism\b", re.IGNORECASE), "pulmonary embolism"),
    (re.compile(r"\bPE\b"), "pulmonary embolism"),
    (re.compile(r"\bdeep vein thrombosis\b", re.IGNORECASE), "deep vein thrombosis"),
    (re.compile(r"\bcolorectal cancer\b", re.IGNORECASE), "colorectal cancer"),
    (re.compile(r"\bseptic shock\b", re.IGNORECASE), "septic shock"),
    # Phase 6F — dermatology / psoriasis (Q1 reproducer).
    # ``psoriatic arthritis`` is detected separately so it does NOT
    # dominate plain psoriasis questions; the ranker uses the
    # related_cap_reason on the specialty-pack entry to cap mismatched
    # papers.
    (re.compile(r"\bplaque psoriasis\b", re.IGNORECASE), "plaque psoriasis"),
    (re.compile(r"\bchronic plaque psoriasis\b", re.IGNORECASE), "plaque psoriasis"),
    (re.compile(r"\bmoderate[-\s]to[-\s]severe psoriasis\b", re.IGNORECASE), "moderate-to-severe psoriasis"),
    (re.compile(r"\bsevere psoriasis\b", re.IGNORECASE), "psoriasis"),
    (re.compile(r"\bpsoriasis\b", re.IGNORECASE), "psoriasis"),
]


def _detect_drug_anchors(q: str) -> list[str]:
    s = q.lower()
    out: list[str] = []
    seen: set[str] = set()
    for drug in _KNOWN_DRUGS_FOR_ANCHORING:
        key = drug.lower()
        if key in seen:
            continue
        if re.search(rf"\b{re.escape(key)}\b", s):
            seen.add(key)
            # Preserve original casing (e.g. ``NSAID`` not ``nsaid``).
            out.append(drug)
    return out


def _detect_condition_anchors(q: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for patt, canonical in _CONDITION_ANCHOR_PATTERNS:
        if patt.search(q):
            if canonical not in seen:
                seen.add(canonical)
                out.append(canonical)
    return out


def _build_must_have_for(role: str, concept: str, *, cap: int) -> dict[str, Any]:
    syns = [s for s in expand_concept(concept) if s.lower() != concept.lower()]
    return {
        "concept": concept,
        "role": role,
        "synonyms": syns,
        "required_for_high_relevance": True,
        "missing_relevance_cap": cap,
    }


# ---------------------------------------------------------------------------
# Phase 5B retest — post-translation scope refinement.
#
# The deterministic scope classifier runs BEFORE concept translation and only
# sees the raw question text. For queries like
#     "What is the difference between high X target and low X target in
#      patients with Y?"
# the classifier fires both a background pattern ("What is the ...") and a
# foreground pattern ("in patients") and returns ``hybrid``. After translation
# we know X and Y are intervention/comparator/population concepts plus a
# comparison-pattern signal is present — that is a foreground question, not
# hybrid. ``hybrid`` is reserved for questions that legitimately mix a
# definitional "what is X" with a decision frame about X.
# ---------------------------------------------------------------------------

_DEFINITIONAL_BG_RE = re.compile(
    # Definitional "what is/are X" — bare concept follows, NOT "the …".
    # • "What is septic shock?"           → MATCH (definitional)
    # • "What are NSAIDs?"                → MATCH (definitional)
    # • "Define heart failure"            → MATCH (definitional)
    # • "Definition of stroke"            → MATCH (definitional)
    # • "What is the dose of milrinone?"  → NO MATCH (asks about a property)
    # • "What is the difference between A and B?" → NO MATCH (decision frame)
    # • "What is the evidence for X?"     → NO MATCH (decision frame)
    # • "What is the role of Y in Z?"     → NO MATCH (decision frame)
    r"\b(?:what\s+(?:is|are)\s+(?!the\b)|define|definition\s+of)\b",
    re.IGNORECASE,
)


def _refine_scope_after_translation(
    scope, translation: dict[str, Any], must_haves: list[dict[str, Any]],
    normalized_q: str, *, type_of_q: str = "",
):
    """Promote ``hybrid`` → ``foreground`` when translation output proves
    the question has explicit roles + (a comparison-pattern signal OR a
    foreground type-of-question).

    Conversely, do NOT demote anything: a question that the classifier
    judged ``background`` stays background even if translation added
    must-haves (background packs do include must-have anchors).
    """
    if scope is None or getattr(scope, "scope", None) != "hybrid":
        return scope

    # Comparison-pattern signal: either the translator flagged a
    # comparison_pattern_matched warning, or normalized_q contains an
    # explicit comparison/decision frame ("vs", "compared with",
    # "difference between", "compared to").
    warnings = translation.get("warnings") or []
    has_comparison_pattern_signal = any(
        isinstance(w, str) and w.startswith("comparison_pattern_matched:")
        for w in warnings
    )
    if not has_comparison_pattern_signal:
        if re.search(
            r"\b(?:vs\.?|versus|compared (?:with|to)|"
            r"difference between|comparison of|comparing)\b",
            normalized_q, re.IGNORECASE,
        ):
            has_comparison_pattern_signal = True

    # Role coverage from the merged must_haves (post-translation).
    roles = {(m.get("role") or "").lower() for m in must_haves}
    has_population_anchor = bool({"population", "condition", "context"} & roles)
    has_intervention_or_comparator = bool({"intervention", "comparator"} & roles)
    # Or a comparator concept proven by translation output (e.g. liberal
    # vs restrictive — both intervention + comparator concepts are added).
    if not has_intervention_or_comparator:
        added_roles = {
            (m.get("role") or "").lower()
            for m in (translation.get("must_have_concepts_added") or [])
        }
        if {"intervention", "comparator"} & added_roles:
            has_intervention_or_comparator = True

    # A foreground type-of-question is also strong evidence that this is
    # NOT a definitional question even if it starts with "What is …".
    # ``background`` and ``other`` are excluded from the foreground set.
    is_foreground_type = type_of_q in {
        "therapy", "diagnosis", "prognosis", "prevention", "pharmacology",
        "dosing_protocol", "clinical_exam", "etiology_harm", "guideline",
        "cost", "epidemiology",
    }

    # Genuinely-hybrid guard: the question must NOT contain a
    # definitional "what is X" frame where X is a concept (not "the
    # difference"/"the role"/etc.). If a definitional frame IS present
    # AND a decision frame is also present, the question is hybrid.
    has_definitional_frame = bool(_DEFINITIONAL_BG_RE.search(normalized_q))

    # Promotion rule:
    #   foreground iff
    #     population anchor present AND
    #     (intervention/comparator anchor present OR foreground type) AND
    #     (comparison-pattern signal present OR foreground type) AND
    #     no definitional "what is <concept>?" frame remains.
    decision_frame = (
        has_comparison_pattern_signal or is_foreground_type
    )
    anchor_frame = has_intervention_or_comparator or is_foreground_type
    if (
        has_population_anchor
        and anchor_frame
        and decision_frame
        and not has_definitional_frame
    ):
        scope.scope = "foreground"
        prev_rationale = scope.rationale or ""
        reason_bits = []
        if has_comparison_pattern_signal:
            reason_bits.append("comparison-pattern signal")
        if is_foreground_type:
            reason_bits.append(f"foreground type-of-question ({type_of_q})")
        if has_intervention_or_comparator:
            reason_bits.append("intervention/comparator anchor")
        scope.rationale = (
            "Foreground (refined after translation): population anchor + "
            + (" + ".join(reason_bits) or "decision frame")
            + " all present; classifier originally said hybrid because "
            "of a non-definitional 'what is the …' frame. "
            f"(Original: {prev_rationale})"
        )
    return scope


# ---------------------------------------------------------------------------
# PICOTT extractor service
# ---------------------------------------------------------------------------

class PicottExtractorService:
    """Build the PICOTT structured payload given the legacy parser output
    and the original (pre-normalization) sanitized question.
    """

    def build(
        self,
        sanitized_question: str,
        legacy_parsed: dict[str, Any],
    ) -> dict[str, Any]:
        # Step 1 — spelling normalize. ``sanitized_question`` is post-PHI.
        norm = spelling_normalizer.normalize(sanitized_question)
        normalized_q = norm.normalized
        spelling_corrections = [
            SpellingCorrection(**c).model_dump() for c in norm.corrections
        ]

        # Step 2 — question scope classification.
        scope = question_scope_classifier.classify(normalized_q)

        # Step 3a — detect anchors BEFORE type promotion so dosing+drug
        # questions like "What is the dose of milrinone for SAH?" get the
        # correct ``dosing_protocol`` type even when the legacy parser
        # returns ``other``.
        drug_anchors = _detect_drug_anchors(normalized_q)
        condition_anchors = _detect_condition_anchors(normalized_q)

        # Step 3b — type-of-question promotion + subtype.
        legacy_type = (legacy_parsed.get("clinical_question_type") or "other").lower()
        type_of_q, subtype = _promote_type_of_question(
            normalized_q, legacy_type,
            drug_anchors=drug_anchors,
            condition_anchors=condition_anchors,
        )
        if scope.scope == "background":
            # Background asks force ``background`` type-of-question.
            type_of_q = "background"
            subtype = None

        # Step 4 — PICO concept blocks (carry over from legacy + dedupe).
        intervention = legacy_parsed.get("intervention_or_exposure") or ""
        comparator = legacy_parsed.get("comparator") or ""
        population = legacy_parsed.get("population") or ""
        outcomes = list(legacy_parsed.get("outcomes") or [])
        setting = legacy_parsed.get("setting") or ""
        time_horizon = legacy_parsed.get("time_horizon") or ""

        # Specialty inference enrichment — promote ``primary_care`` to
        # ``neurology`` / ``critical_care`` when neurocrit-pack triggers fire.
        specialty = legacy_parsed.get("specialty") or ""
        if applicable_specialty_packs(normalized_q) == ["neurocritical_care"]:
            if specialty in {"primary_care", "", None}:
                specialty = "neurology"

        pico_blocks = PICOBlocks(
            population=[population] if population else [],
            intervention_or_exposure_or_index_test=[s for s in [intervention] + drug_anchors if s],
            comparator_or_reference_standard=[comparator] if comparator else [],
            outcomes_or_target_diagnosis=list(outcomes) + condition_anchors,
            time_horizon=[time_horizon] if time_horizon else [],
            setting=[setting] if setting else [],
        )

        # Step 5 — concept role assignments.
        must_haves: list[dict[str, Any]] = []
        should_haves: list[dict[str, Any]] = []
        ranking: list[dict[str, Any]] = []
        negatives: list[dict[str, Any]] = []

        # Drug anchors become MUST-HAVE intervention concepts.
        # Phase 6F — cap is 20 (was 30). Drug is a primary clinical
        # entity anchor; missing this anchor must cap relevance ≤ 20.
        for drug in drug_anchors:
            must_haves.append(_build_must_have_for("intervention", drug, cap=20))
        # If legacy intervention was extracted but no drug-anchor matched,
        # add it as must-have too (could be a procedure/test).
        if intervention and not any(
            (mh["concept"].lower() == intervention.lower()) for mh in must_haves
        ):
            must_haves.append(_build_must_have_for("intervention", intervention, cap=30))

        # Condition anchors become MUST-HAVE conditions.
        # Phase 6F — cap is 20 (was 50). Disease/problem is a primary
        # clinical entity anchor; missing this anchor must cap
        # relevance ≤ 20 per the Phase 6F directive.
        for cond in condition_anchors:
            must_haves.append(_build_must_have_for("condition", cond, cap=20))

        # Phase 6F — drug_safety subtype: add an OUTCOME must-have so the
        # ranker requires articles to also mention an adverse-effect term.
        # missing_relevance_cap=50 is intentionally looser than the drug
        # anchor (cap=30) — broad safety articles without the target
        # drug still get capped via the drug anchor, but an article
        # that mentions the drug and adverse effects deserves higher
        # placement than one that only mentions the drug.
        if subtype == "drug_safety" and drug_anchors:
            ae_terms = [
                "side effects", "adverse effects", "adverse reactions",
                "adverse events", "toxicity", "tolerability",
                "drug safety", "safety profile",
                "treatment-emergent adverse events",
            ]
            must_haves.append({
                "concept": "adverse effects",
                "role": "outcome",
                "synonyms": ae_terms,
                "required_for_high_relevance": True,
                "missing_relevance_cap": 50,
            })

        # Phase 6F — treatment_sequencing subtype: add a RANKING concept
        # for first-line / second-line / treatment-algorithm phrases so
        # disease-anchored articles that ALSO discuss sequencing rank
        # higher. NOT a must-have (sequencing can be implicit in a
        # guideline article).
        if subtype == "treatment_sequencing":
            for term in (
                "first-line treatment", "first line treatment",
                "second-line treatment", "second line treatment",
                "initial treatment",
                "treatment algorithm", "management algorithm",
                "treatment guideline", "clinical practice guideline",
                "treatment options",
            ):
                ranking.append({
                    "concept": term,
                    "role": "context",
                    "synonyms": [],
                    "boost_weight": 3,
                })

        # Comparator → SHOULD-HAVE (only matters for therapy comparisons).
        if comparator:
            should_haves.append({
                "concept": comparator,
                "role": "comparator",
                "synonyms": [s for s in expand_concept(comparator)
                              if s.lower() != comparator.lower()],
                "boost_weight": 4,
            })

        # Outcome terms become ranking concepts.
        for o in outcomes:
            if not o:
                continue
            ranking.append({
                "concept": o,
                "role": "outcome",
                "synonyms": [],
                "boost_weight": 4,
            })

        # Specialty pack contributions.
        packs = applicable_specialty_packs(normalized_q)
        if "neurocritical_care" in packs:
            for term in neurocrit_ranking_terms():
                if any(r["concept"].lower() == term.lower() for r in ranking):
                    continue
                # Dosing terms are ``context`` role; condition terms are
                # ``condition``; drug name is intervention.
                role = "context"
                if term.lower() in {"milrinone", "nimodipine"}:
                    role = "intervention"
                elif term.lower() in {"subarachnoid hemorrhage", "sah", "asah",
                                       "aneurysmal subarachnoid hemorrhage"}:
                    role = "condition"
                ranking.append({
                    "concept": term,
                    "role": role,
                    "synonyms": [],
                    "boost_weight": 3,
                })

        # Dosing/protocol type-of-question: add dosing context tokens as
        # ranking concepts (role="context"). These DO NOT cap, but they
        # cause the article to clear the >80 tier.
        if type_of_q == "dosing_protocol":
            for term in neurocrit_dosing_context_terms():
                if any(r["concept"].lower() == term.lower() for r in ranking):
                    continue
                ranking.append({
                    "concept": term,
                    "role": "context",
                    "synonyms": [],
                    "boost_weight": 3,
                })

        # Pharmacology + dosing_protocol questions: imaging-only papers are
        # decoys unless the drug name is present.
        if type_of_q in {"pharmacology", "dosing_protocol"} and drug_anchors:
            for decoy in IMAGING_DECOY_TERMS:
                negatives.append({
                    "concept": decoy,
                    "penalty_reason": (
                        "imaging-only article for a pharmacology/dosing "
                        "question (no drug mention)"
                    ),
                    "unless_concepts_present": list(drug_anchors),
                    "penalty_cap": 40,
                })

        # Step 5d — Phase 5B: Clinical concept translation.
        # Adds canonical concept must-haves/ranking/negative concepts for
        # idioms like "high hemoglobin target" → "liberal transfusion
        # strategy". Pure-deterministic. Refuses on PHI-blocked sessions
        # (the caller never reaches us in that case but we accept the
        # flag defensively).
        translation = clinical_concept_translation.translate(
            normalized_q, picott=None, is_phi_blocked=False,
        )

        # Merge translated must-haves / ranking / negatives into the
        # existing lists (dedupe by canonical name).
        existing_mh_names = {(m.get("concept") or "").lower() for m in must_haves}
        for added_mh in (translation.get("must_have_concepts_added") or []):
            name = (added_mh.get("concept") or "").lower()
            if name and name not in existing_mh_names:
                must_haves.append(added_mh)
                existing_mh_names.add(name)
        existing_rk_names = {(r.get("concept") or "").lower() for r in ranking}
        for added_rc in (translation.get("ranking_concepts_added") or []):
            name = (added_rc.get("concept") or "").lower()
            if name and name not in existing_rk_names:
                ranking.append(added_rc)
                existing_rk_names.add(name)
        for added_neg in (translation.get("negative_concepts_added") or []):
            negatives.append(added_neg)

        # Phase 5B: if a transfusion-threshold comparison was detected,
        # promote type_of_question to "therapy" (not a new enum value)
        # and stash the subtype string for the UI.
        translation_subtype = translation.get("question_subtype")
        if translation_subtype == "transfusion_threshold_comparison":
            type_of_q = "therapy"
            subtype = "transfusion_threshold_comparison"
            # Specialty inference: cardiology if AMI tokens present.
            specialty = "cardiology"

        # Phase 5B retest — refine question scope using translation
        # output. The classifier alone can't always distinguish a
        # genuinely hybrid question ("What is X, and how does Y compare
        # with Z?") from a foreground comparison-frame question that
        # happens to start with "What is the difference between X and Y
        # in patients ...". After translation we know the roles of
        # every concept; if the question has BOTH an intervention/
        # comparator AND a population anchor AND a comparison-pattern
        # signal, it's foreground.
        scope = _refine_scope_after_translation(
            scope, translation, must_haves, normalized_q,
            type_of_q=type_of_q,
        )

        # Step 6 — ideal study types via the router.
        pathway = study_type_router.pathway(type_of_q)
        ideal = list(pathway.preferred_study_types) + list(pathway.acceptable_study_types)

        # Step 7 — interpretation_for_user (single short sentence).
        interp_pieces: list[str] = []
        if drug_anchors:
            interp_pieces.append(
                f"about {drug_anchors[0]}"
                + (f" (and {len(drug_anchors)-1} more drug)" if len(drug_anchors) > 1 else "")
            )
        if condition_anchors:
            interp_pieces.append(f"in {condition_anchors[0]}")
        if type_of_q in {"dosing_protocol", "pharmacology"}:
            interp_pieces.append("dosing/protocol question")
        elif type_of_q == "therapy":
            interp_pieces.append("therapy question")
        elif type_of_q == "diagnosis":
            interp_pieces.append("diagnosis question")
        elif type_of_q == "background":
            interp_pieces.append("background question")
        elif type_of_q == "guideline":
            interp_pieces.append("guideline question")
        interpretation = ", ".join(interp_pieces) or (
            "general clinical question (no anchor concepts detected)"
        )

        # Step 8 — missing elements report.
        missing_elements: dict[str, str] = {}
        if not pico_blocks.population:
            missing_elements["population"] = "missing"
        if not pico_blocks.comparator_or_reference_standard:
            missing_elements["comparator_or_reference_standard"] = "missing"
        if not pico_blocks.outcomes_or_target_diagnosis:
            missing_elements["outcomes_or_target_diagnosis"] = "missing"

        # Step 9 — confidence.
        confidence = 0.4
        if drug_anchors and condition_anchors:
            confidence = 0.85
        elif drug_anchors or condition_anchors:
            confidence = 0.65
        if type_of_q == "background":
            confidence = max(confidence, 0.7)

        rationale_bits = [f"scope={scope.scope}", f"type_of_question={type_of_q}"]
        if subtype:
            rationale_bits.append(f"subtype={subtype}")
        if drug_anchors:
            rationale_bits.append("drug_anchors=" + ",".join(drug_anchors))
        if condition_anchors:
            rationale_bits.append("condition_anchors=" + ",".join(condition_anchors))
        if spelling_corrections:
            rationale_bits.append(
                "spelling_corrected="
                + ",".join(c["original"] for c in spelling_corrections)
            )
        if packs:
            rationale_bits.append("specialty_packs=" + ",".join(packs))

        picott = PICOTT(
            question_scope=scope.scope,
            normalized_question=normalized_q,
            original_question=sanitized_question or "",
            spelling_corrections=[SpellingCorrection(**c) for c in spelling_corrections],
            pico=pico_blocks,
            type_of_question=type_of_q,
            question_subtype=subtype,
            specialty=specialty,
            ideal_study_types=ideal,
            must_have_concepts=must_haves,
            should_have_concepts=should_haves,
            ranking_concepts=ranking,
            negative_or_penalty_concepts=negatives,
            missing_elements=missing_elements,
            interpretation_for_user=interpretation,
            rationale=" | ".join(rationale_bits),
            confidence=confidence,
            translated_question=translation.get("translated_question") or "",
            concept_translations=translation.get("concept_translations") or [],
            latent_concepts=translation.get("latent_concepts") or [],
            translation_warnings=translation.get("warnings") or [],
        )
        return picott.model_dump()


picott_extractor = PicottExtractorService()
