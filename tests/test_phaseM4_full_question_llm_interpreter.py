"""Phase M4-shadow — unit tests for the Full-Question LLM
ClinicalQuestionMap Interpreter (deterministic surfaces only — the
bounded LLM call is exercised by the live preview smoke).

Covers: mode resolver (prod hard-off), trigger detector for the
spec test cases A–N, schema-locked validator accept/reject paths,
PHI / article-reference / unapproved-source scrubs, stage separation,
cleaner-than-baseline, central-anchor preservation, and canonicalization.
"""
from __future__ import annotations


import services.m4_full_question_interpreter as m4


# ---- canonical clean candidate (mirrors SYSTEM_PROMPT Example A) ----
def _clean_prognosis_candidate() -> dict:
    return {
        "primary_intent": "prognosis_natural_history",
        "secondary_intents": ["prognostic_factor"],
        "frameworks": ["PFO", "prognosis"],
        "clinical_problem": ["colon cancer"],
        "stage": ["stage I"],
        "biomarker": [],
        "population": ["patients with stage I colon cancer"],
        "exposure": [],
        "intervention": [],
        "procedure": [],
        "medication": [],
        "index_test": [],
        "outcome": ["overall survival", "recurrence", "mortality"],
        "setting": [],
        "timeframe": [],
        "mechanism": [],
        "comparator": [],
        "required_anchors": ["colon cancer"],
        "optional_or_ranking_only_terms": ["prognosis", "survival", "stage I"],
        "candidate_synonyms_unvalidated": ["colon carcinoma", "colorectal cancer"],
        "candidate_mesh_terms_unvalidated": ["Colonic Neoplasms", "Survival Rate"],
        "clarification_required": False,
        "confidence": 0.9,
        "repair_reason": "long_literal_phrase_anchor",
        "provenance": "m4_full_question_interpreter_shadow",
    }


def _baseline_literal(text: str):
    return [{"canonical_concept": text, "required_for_search": True}]


# --------------------------- mode resolver ---------------------------
def test_mode_disabled_in_production():
    assert m4.resolve_m4_mode(is_production=True, env={"M4_FULL_QUESTION_INTERPRETER_MODE": "shadow"}) == "disabled"


def test_mode_shadow_in_preview_default():
    assert m4.resolve_m4_mode(is_production=False, env={}) == "shadow"


def test_mode_active_collapses_to_shadow_never_active():
    assert m4.resolve_m4_mode(is_production=False, env={"M4_FULL_QUESTION_INTERPRETER_MODE": "active"}) == "shadow"


def test_mode_disabled_respected_in_preview():
    assert m4.resolve_m4_mode(is_production=False, env={"M4_FULL_QUESTION_INTERPRETER_MODE": "disabled"}) == "disabled"


# --------------------------- trigger detector ------------------------
def test_trigger_case_A_prognosis_stage_literal_anchor():
    t = m4.detect_m4_triggers(
        normalized_question="What is the prognosis of stage 1 colon cancer?",
        baseline_primary_intent="prognosis_natural_history",
        baseline_subtype="prognosis_natural_history",
        primary_anchors=_baseline_literal("prognosis of stage 1 colon cancer"),
        generic_modifiers=[],
        anchor_failure_reason=None,
    )
    assert "long_literal_phrase_anchor" in t
    assert "intent_word_fused_in_anchor" in t
    assert "stage_biomarker_population_fused" in t
    assert "stop_grammar_required_anchor" in t


def test_trigger_case_B_survival_rate_stage():
    t = m4.detect_m4_triggers(
        normalized_question="What is the survival rate for stage II colon cancer?",
        baseline_primary_intent="prognosis_natural_history",
        baseline_subtype="prognosis_natural_history",
        primary_anchors=_baseline_literal("survival rate for stage II colon cancer"),
        generic_modifiers=[],
        anchor_failure_reason=None,
    )
    assert "long_literal_phrase_anchor" in t
    assert "stage_biomarker_population_fused" in t


def test_trigger_case_D_her2_biomarker_fused():
    t = m4.detect_m4_triggers(
        normalized_question="What is the prognosis for HER2 positive stage 2 breast cancer?",
        baseline_primary_intent="prognosis_natural_history",
        baseline_subtype="prognosis_natural_history",
        primary_anchors=_baseline_literal("prognosis for HER2 positive stage 2 breast cancer"),
        generic_modifiers=[],
        anchor_failure_reason=None,
    )
    assert "stage_biomarker_population_fused" in t
    assert "intent_word_fused_in_anchor" in t


def test_trigger_case_F_causes_what_is_overview_forced():
    # "what is" not present; "what are the causes" uses what_are lead-in.
    t = m4.detect_m4_triggers(
        normalized_question="What are the causes of schizophrenia in young adults?",
        baseline_primary_intent="disease_overview",
        baseline_subtype="disease_overview",
        primary_anchors=_baseline_literal("causes of schizophrenia in young adults"),
        generic_modifiers=[],
        anchor_failure_reason=None,
    )
    assert "what_is_forced_disease_overview" in t
    assert "long_literal_phrase_anchor" in t


def test_trigger_case_E_ipf_intent_word():
    t = m4.detect_m4_triggers(
        normalized_question="What is the prognosis of idiopathic pulmonary fibrosis?",
        baseline_primary_intent="disease_overview",
        baseline_subtype="disease_overview",
        primary_anchors=_baseline_literal("prognosis of idiopathic pulmonary fibrosis"),
        generic_modifiers=[],
        anchor_failure_reason=None,
    )
    assert "what_is_forced_disease_overview" in t
    assert "long_literal_phrase_anchor" in t


def test_no_trigger_case_J_pure_disease_overview():
    # "What is colon cancer?" — no intent word, short anchor → no triggers.
    t = m4.detect_m4_triggers(
        normalized_question="What is colon cancer?",
        baseline_primary_intent="disease_overview",
        baseline_subtype="disease_overview",
        primary_anchors=[{"canonical_concept": "colon cancer", "required_for_search": True}],
        generic_modifiers=[],
        anchor_failure_reason=None,
    )
    assert t == []


def test_trigger_case_K_generic_dose_no_entity_meaningful():
    t = m4.detect_m4_triggers(
        normalized_question="What is the dose?",
        baseline_primary_intent=None,
        baseline_subtype=None,
        primary_anchors=[],
        generic_modifiers=["dose"],
        anchor_failure_reason="no_primary_clinical_entity",
    )
    # "what is the dose" = 4 words + intent word 'dose' → may trigger.
    assert "no_primary_entity_but_clinically_meaningful" in t


def test_no_trigger_when_k4a_route_owns_session():
    # K4A.* fired (severity_marker_overview) → M4 must NOT trigger.
    t = m4.detect_m4_triggers(
        normalized_question="What are the markers of severity in acute pancreatitis?",
        baseline_primary_intent="severity_marker_overview",
        baseline_subtype="severity_marker_overview",
        primary_anchors=_baseline_literal("markers of severity in acute pancreatitis"),
        generic_modifiers=[],
        anchor_failure_reason=None,
    )
    assert t == []


def test_no_trigger_case_N_guideline_screening_short_anchor():
    # K5 guideline lane regression — clean short anchor, no intent fuse.
    t = m4.detect_m4_triggers(
        normalized_question="What are the guidelines for colorectal cancer screening in average-risk adults?",
        baseline_primary_intent="guideline_recommendation",
        baseline_subtype="guideline_recommendation",
        primary_anchors=[{"canonical_concept": "colorectal cancer", "required_for_search": True}],
        generic_modifiers=[],
        anchor_failure_reason=None,
    )
    # 'screening' is an intent word but it is NOT fused in the (clean)
    # anchor, and the baseline is not an overview → no what_is trigger,
    # no long-literal trigger.
    assert "long_literal_phrase_anchor" not in t
    assert "what_is_forced_disease_overview" not in t


# --------------------------- validator: accept -----------------------
def test_validator_accepts_clean_prognosis_candidate():
    cand, rej = m4.validate_m4_candidate(
        _clean_prognosis_candidate(),
        deterministic_primary_anchors=_baseline_literal("prognosis of stage 1 colon cancer"),
        baseline_required_anchors=["prognosis of stage 1 colon cancer"],
    )
    assert rej == []
    assert cand is not None
    assert cand.primary_intent == "prognosis_natural_history"
    assert cand.required_anchors == ["colon cancer"]


# --------------------------- validator: reject -----------------------
def test_validator_rejects_fused_literal_required_anchor():
    bad = _clean_prognosis_candidate()
    bad["required_anchors"] = ["prognosis of stage 1 colon cancer"]
    cand, rej = m4.validate_m4_candidate(
        bad,
        deterministic_primary_anchors=[],
        baseline_required_anchors=[],
    )
    assert cand is None
    assert "literal_full_phrase_required_anchor" in rej


def test_validator_rejects_stage_in_required_anchor():
    bad = _clean_prognosis_candidate()
    bad["required_anchors"] = ["stage I colon cancer"]
    cand, rej = m4.validate_m4_candidate(
        bad,
        deterministic_primary_anchors=[],
        baseline_required_anchors=[],
    )
    assert cand is None
    assert "stage_not_separated_from_disease" in rej


def test_validator_rejects_generic_only_anchor():
    bad = _clean_prognosis_candidate()
    bad["required_anchors"] = ["patients"]
    cand, rej = m4.validate_m4_candidate(
        bad,
        deterministic_primary_anchors=[],
        baseline_required_anchors=[],
    )
    assert cand is None
    assert ("generic_only_required_anchor" in rej) or ("no_disease_problem_anchor" in rej)


def test_validator_rejects_low_confidence():
    bad = _clean_prognosis_candidate()
    bad["confidence"] = 0.4
    cand, rej = m4.validate_m4_candidate(
        bad, deterministic_primary_anchors=[], baseline_required_anchors=[],
    )
    assert cand is None
    assert "confidence_below_threshold" in rej


def test_validator_rejects_phi_in_output():
    bad = _clean_prognosis_candidate()
    bad["population"] = ["john.doe@example.com"]
    cand, rej = m4.validate_m4_candidate(
        bad, deterministic_primary_anchors=[], baseline_required_anchors=[],
    )
    assert cand is None
    assert "phi_pattern_detected" in rej


def test_validator_rejects_article_reference():
    bad = _clean_prognosis_candidate()
    bad["candidate_synonyms_unvalidated"] = ["see PMID 12345678"]
    cand, rej = m4.validate_m4_candidate(
        bad, deterministic_primary_anchors=[], baseline_required_anchors=[],
    )
    assert cand is None
    assert "article_reference_detected" in rej


def test_validator_rejects_unapproved_source():
    bad = _clean_prognosis_candidate()
    bad["candidate_synonyms_unvalidated"] = ["google scholar"]
    cand, rej = m4.validate_m4_candidate(
        bad, deterministic_primary_anchors=[], baseline_required_anchors=[],
    )
    assert cand is None
    assert "unapproved_source_referenced" in rej


def test_validator_accepts_cleaner_than_baseline():
    # Baseline had a fused literal anchor; a clean candidate that drops
    # it and keeps the canonical disease anchor must be ACCEPTED.
    cand, rej = m4.validate_m4_candidate(
        _clean_prognosis_candidate(),
        deterministic_primary_anchors=_baseline_literal("prognosis of stage 1 colon cancer"),
        baseline_required_anchors=["prognosis of stage 1 colon cancer"],
    )
    assert rej == []
    assert cand is not None
    assert "prognosis of stage 1 colon cancer" not in cand.required_anchors


def test_validator_rejects_invalid_provenance():
    bad = _clean_prognosis_candidate()
    bad["provenance"] = "something_else"
    cand, rej = m4.validate_m4_candidate(
        bad, deterministic_primary_anchors=[], baseline_required_anchors=[],
    )
    assert cand is None
    assert ("provenance_mismatch" in rej) or ("invalid_enum" in rej)


def test_validator_rejects_extra_field():
    bad = _clean_prognosis_candidate()
    bad["hallucinated_field"] = "x"
    cand, rej = m4.validate_m4_candidate(
        bad, deterministic_primary_anchors=[], baseline_required_anchors=[],
    )
    assert cand is None
    assert "extra_field" in rej


def test_validator_rejects_overbroad_synonyms():
    bad = _clean_prognosis_candidate()
    bad["candidate_synonyms_unvalidated"] = [f"syn{i}" for i in range(20)]
    cand, rej = m4.validate_m4_candidate(
        bad, deterministic_primary_anchors=[], baseline_required_anchors=[],
    )
    assert cand is None
    assert "overbroad_synonym_expansion" in rej


def test_validator_rejects_would_remove_central_anchor():
    bad = _clean_prognosis_candidate()
    bad["required_anchors"] = ["lung cancer"]
    bad["clinical_problem"] = ["lung cancer"]
    bad["optional_or_ranking_only_terms"] = []
    cand, rej = m4.validate_m4_candidate(
        bad,
        deterministic_primary_anchors=[{"canonical_concept": "colon cancer", "required_for_search": True}],
        baseline_required_anchors=[],
    )
    assert cand is None
    assert "would_remove_central_anchor" in rej


def test_validator_refusal_envelope():
    cand, rej = m4.validate_m4_candidate(
        {"refused": True, "refusal_reason": "off_topic"},
        deterministic_primary_anchors=[], baseline_required_anchors=[],
    )
    assert cand is None
    assert rej == ["refused"]


def test_validator_empty_output():
    cand, rej = m4.validate_m4_candidate(
        None, deterministic_primary_anchors=[], baseline_required_anchors=[],
    )
    assert cand is None
    assert rej == ["empty_output"]


# --------------------------- canonicalization ------------------------
def test_canonicalize_intent_and_framework_aliases():
    raw = {"primary_intent": "natural_history", "frameworks": ["peco"]}
    out = m4.canonicalize_m4_candidate(raw)
    assert out["primary_intent"] == "prognosis_natural_history"
    assert out["frameworks"] == ["CoCoPop"]
    # never mutates input
    assert raw["primary_intent"] == "natural_history"


# --------------------------- system prompt contract ------------------
def test_system_prompt_has_hard_not_contract():
    p = m4.SYSTEM_PROMPT.lower()
    for phrase in ("not allowed", "search literature", "rank", "pmids", "google scholar", "strict json"):
        assert phrase in p
    assert "m4_full_question_interpreter_shadow" in m4.SYSTEM_PROMPT
