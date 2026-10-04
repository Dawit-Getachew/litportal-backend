"""W1.4 — interpretation, verification of model output, conflicts, chips (pure)."""
from __future__ import annotations

import pytest

from clinical_engine.contracts.question import ClinicalQuestionMap, QuestionElement, material_hash
from clinical_engine.gateway import ModelCallResult
from clinical_engine.interpretation import rules
from clinical_engine.interpretation.consistency import check
from clinical_engine.interpretation.model_reading import quote_in_question, validate_model_output
from clinical_engine.interpretation.present import (
    ChipEdit,
    EngineInputError,
    apply_edits,
    build_summary,
    chips,
)
from clinical_engine.interpretation.service import interpret, phi_preflight


def roles(reading) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for e in reading.elements:
        out.setdefault(e.role, []).append(e.value.lower())
    return out


class FakeGateway:
    def __init__(self, data=None, error=None):
        self.data, self.error, self.calls = data, error, 0

    def is_configured(self):
        return True

    async def complete_json(self, system, user, *, max_tokens, timeout_s):
        self.calls += 1
        return ModelCallResult(self.data, "fake-model", 120, 80, 5, self.error)


class UnconfiguredGateway(FakeGateway):
    def is_configured(self):
        return False


# ── deterministic reading: the three required example types ─────────────────


def test_treatment_example_steroid_question():
    r = rules.read("Impact of corticosteroids on mortality in septic shock")
    assert r.question_type == "treatment"  # legacy parser read this as prognosis
    got = roles(r)
    assert got["intervention"] == ["corticosteroids"]
    assert got["outcome"] == ["mortality"]
    assert got["condition"] == ["septic shock"]
    assert "outcome" not in got or "septic shock" not in got["outcome"]  # no condition-as-outcome


def test_exposure_example():
    r = rules.read("Is exposure to air pollution associated with asthma in children?")
    assert r.question_type == "etiology_risk"
    got = roles(r)
    assert got["exposure"] == ["air pollution"]
    assert got["outcome"] == ["asthma"]
    assert got["age_group"] == ["child"]


def test_diagnostic_example():
    r = rules.read("What is the accuracy of procalcitonin for diagnosing sepsis in the emergency department?")
    assert r.question_type == "diagnosis"
    got = roles(r)
    assert got["index_test"] == ["procalcitonin"]
    assert got["condition"] == ["sepsis"]
    assert got["setting"] == ["emergency department"]


@pytest.mark.parametrize("question", [
    "Impact of corticosteroids on mortality in septic shock",
    "Does early mobilization prevent delirium in the ICU?",
    "What is the accuracy of procalcitonin for diagnosing sepsis?",
])
def test_no_age_group_is_ever_inferred_from_rules(question):
    assert not [e for e in rules.read(question).elements if e.role == "age_group"]


@pytest.mark.parametrize("question,value", [
    ("Excluding children, does ketamine reduce agitation?", "child"),
    ("Effect of early vasopressin on mortality in non-elderly patients with septic shock", "older_adult"),
])
def test_negated_age_groups(question, value):
    ages = [e for e in rules.read(question).elements if e.role == "age_group"]
    assert ages and ages[0].value == value and ages[0].negated


def test_what_is_the_impact_is_not_background():
    assert rules.read("What is the impact of dapagliflozin on mortality in heart failure?").question_type == "treatment"
    assert rules.read("What is septic shock?").question_type == "background"


# ── model output verification ───────────────────────────────────────────────

Q = "Does hydrocortisone reduce 28-day mortality in septic shock?"


def test_model_quotes_must_come_from_the_question():
    out = validate_model_output(Q, {"question_type": "treatment", "elements": [
        {"role": "intervention", "value": "hydrocortisone", "stated": True, "quote": "hydrocortisone"},
        {"role": "comparator", "value": "placebo", "stated": True, "quote": "versus placebo"},
    ]})
    by_role = {e.role: e for e in out.elements}
    assert by_role["intervention"].active and by_role["intervention"].source == "user_text"
    # Claimed as stated, but the quote is not in the question → inactive suggestion.
    assert not by_role["comparator"].active and by_role["comparator"].source == "ai_inferred"
    assert "unsupported_quote:comparator" in out.rejected


def test_model_cannot_state_an_unmentioned_age_group():
    out = validate_model_output(Q, {"question_type": "treatment", "elements": [
        {"role": "age_group", "value": "adults", "stated": True, "quote": "septic shock"},
    ]})
    age = out.elements[0]
    assert age.value == "adult" and not age.active and age.source == "ai_inferred"


def test_model_output_rejects_bad_types_roles_and_citations():
    out = validate_model_output(Q, {"question_type": "magic", "elements": [
        {"role": "hallucination", "value": "x", "stated": False},
        {"role": "outcome", "value": "see PMID 12345678", "stated": False},
        {"role": "age_group", "value": "martians", "stated": False},
    ]})
    assert out.question_type is None and out.elements == []
    assert {"question_type:magic", "role:hallucination", "citation_like:outcome", "age_value:martians"} <= set(out.rejected)


def test_generic_effect_words_are_not_outcomes():
    out = validate_model_output(Q, {"question_type": "treatment", "elements": [
        {"role": "outcome", "value": "impact", "stated": False},
        {"role": "outcome", "value": "Clinical outcomes", "stated": False},
        {"role": "population", "value": "adults", "stated": False},
        {"role": "outcome", "value": "28-day mortality", "stated": True, "quote": "28-day mortality"},
    ]})
    assert [(e.role, e.value) for e in out.elements] == [("outcome", "28-day mortality")]


def test_role_name_placeholders_are_dropped():
    out = validate_model_output(Q, {"question_type": "harm", "elements": [
        {"role": "intervention", "value": "intervention", "stated": False},
        {"role": "exposure", "value": "the exposure", "stated": False},
    ]})
    assert out.elements == []


async def test_prevention_outcome_is_not_also_a_condition():
    gw = FakeGateway({"question_type": "prevention", "elements": [
        {"role": "intervention", "value": "early mobilization", "stated": True, "quote": "early mobilization"},
        {"role": "outcome", "value": "delirium", "stated": True, "quote": "delirium"},
        {"role": "condition", "value": "delirium", "stated": True, "quote": "delirium"},
    ]})
    result = await interpret("Does early mobilization prevent delirium in critically ill older adults?", gateway=gw)
    assert [e.value for e in result.map.active("condition")] == []
    assert [e.value for e in result.map.active("outcome")] == ["delirium"]
    assert result.map.summary.startswith("Prevention of delirium with early mobilization")


def test_a_clause_about_the_patients_is_not_part_of_the_condition():
    q = ("In adults with septic shock who continue to require vasopressors after initial resuscitation, "
         "what is the evidence that corticosteroids improve outcomes?")
    out = validate_model_output(q, {"question_type": "treatment", "elements": [
        {"role": "condition", "value": "septic shock who continue to require vasopressors after initial resuscitation",
         "stated": True, "quote": "septic shock who continue to require vasopressors after initial resuscitation"},
        {"role": "intervention", "value": "corticosteroids", "stated": True, "quote": "corticosteroids"},
    ]})
    by_role = {e.role: e for e in out.elements}
    assert by_role["condition"].value == "septic shock" and by_role["condition"].span_text == "septic shock"
    pop = by_role["population"]
    assert pop.value == "continue to require vasopressors after initial resuscitation" and pop.active
    # The deterministic reader (used when the model files the condition elsewhere) splits it too.
    r = {(e.role, e.value) for e in rules.read(q).elements}
    assert ("condition", "septic shock") in r
    assert ("population", "continue to require vasopressors after initial resuscitation") in r
    # A head that is only "patients" is not split.
    out = validate_model_output("Do patients receiving dialysis benefit?", {"question_type": "treatment", "elements": [
        {"role": "condition", "value": "patients receiving dialysis", "stated": True,
         "quote": "patients receiving dialysis"}]})
    assert [e.value for e in out.elements if e.role == "condition"] == ["patients receiving dialysis"]


def test_schema_invalid_model_output():
    assert validate_model_output(Q, {"elements": "nope"}).rejected == ["schema_invalid"]


def test_quote_matching_ignores_case_and_punctuation():
    assert quote_in_question("28-Day Mortality", Q)
    assert not quote_in_question("mortal", Q)


# ── service: merge + fallbacks ──────────────────────────────────────────────


async def test_interpret_with_model_merges_and_keeps_suggestions_inactive():
    gw = FakeGateway({"question_type": "treatment", "elements": [
        {"role": "intervention", "value": "hydrocortisone", "stated": True, "quote": "hydrocortisone"},
        {"role": "outcome", "value": "28-day mortality", "stated": True, "quote": "28-day mortality",
         "importance": "critical"},
        {"role": "condition", "value": "septic shock", "stated": True, "quote": "septic shock"},
        {"role": "comparator", "value": "placebo", "stated": False},
        {"role": "age_group", "value": "adult", "stated": False},
    ]})
    result = await interpret(Q, gateway=gw)
    assert result.meta.method == "model" and result.meta.tokens_in == 120
    cqm = result.map
    assert cqm.question_type == "treatment" and cqm.framework == "PICO"
    assert cqm.age_group() == "not_stated"  # the model's adult guess is only a suggestion
    assert {e.value for e in cqm.suggestions()} == {"placebo", "adult"}
    assert "adults" not in cqm.summary.lower()
    assert "patients with septic shock" in cqm.summary
    assert not result.conflicts
    assert any("No age group stated" in w for w in result.warnings)


@pytest.mark.parametrize("gateway,reason", [
    (UnconfiguredGateway(), "model_not_configured"),
    (FakeGateway(error="timeout"), "model_timeout"),
    (FakeGateway(error="http_429"), "model_http_429"),
    (FakeGateway({"question_type": "zzz", "elements": []}), "model_output_rejected"),
])
async def test_interpret_falls_back_to_rules(gateway, reason):
    result = await interpret("Impact of corticosteroids on mortality in septic shock", gateway=gateway)
    assert result.meta.method == "rules_only" and result.meta.fallback_reason == reason
    assert result.map.question_type == "treatment"
    assert [e.value for e in result.map.active("intervention")] == ["corticosteroids"]


async def test_budget_exhausted_skips_the_model():
    gw = FakeGateway({"question_type": "treatment", "elements": []})
    result = await interpret(Q, gateway=gw, allow_model=False)
    assert gw.calls == 0 and result.meta.fallback_reason == "model_budget_exhausted"


async def test_empty_and_long_questions_rejected():
    with pytest.raises(EngineInputError):
        await interpret("   ", gateway=UnconfiguredGateway())
    with pytest.raises(EngineInputError):
        await interpret("x" * 5000, gateway=UnconfiguredGateway())


@pytest.mark.parametrize("text,code", [
    ("Treat Mr. John Smith for sepsis?", "name_title_detected"),
    ("Steroids for a 94-year-old with septic shock", "age_over_89_detected"),
    ("Patient MRN: 12345678 with sepsis", "mrn_pattern_detected"),
])
def test_phi_preflight(text, code):
    res = phi_preflight(text)
    assert res.blocked and code in res.reason_codes


def test_phi_preflight_passes_clinical_question():
    assert not phi_preflight("Does hydrocortisone reduce mortality in adults aged 65 with septic shock?").blocked


# ── conflicts fail safely ───────────────────────────────────────────────────


def _cqm(qtype="treatment", *elements):
    return ClinicalQuestionMap(question_text="q", question_type=qtype, framework="PICO", elements=list(elements))


def el(role, value, **kw):
    return QuestionElement(role=role, value=value, source=kw.pop("source", "user_text"), **kw)


def codes(cqm):
    return {c.code for c in check(cqm)[0] if c.severity == "blocking"}


def test_condition_as_outcome_is_blocking_for_treatment_only():
    tx = _cqm("treatment", el("intervention", "steroids"), el("condition", "septic shock"),
              el("outcome", "Septic shock"))
    assert "condition_as_outcome" in codes(tx)
    prev = _cqm("prevention", el("intervention", "early mobilization"), el("condition", "delirium"),
                el("outcome", "delirium"))
    assert "condition_as_outcome" not in codes(prev)


def test_intervention_equals_comparator():
    cqm = _cqm("treatment", el("intervention", "apixaban"), el("comparator", "Apixaban"), el("condition", "af"))
    assert "intervention_equals_comparator" in codes(cqm)


def test_missing_core_element_and_contradictory_ages():
    assert "missing_core_element" in codes(_cqm("treatment", el("condition", "sepsis")))
    ages = _cqm("treatment", el("intervention", "x"), el("condition", "y"), el("age_group", "neonate"),
                el("age_group", "older_adult"))
    assert "contradictory_age_groups" in codes(ages)


def test_inactive_suggestions_do_not_create_conflicts():
    cqm = _cqm("treatment", el("intervention", "apixaban"), el("condition", "af"),
               el("comparator", "apixaban", source="ai_inferred", active=False))
    assert not codes(cqm)


# ── chips, summary, edits ───────────────────────────────────────────────────


def test_chips_show_age_placeholder_when_not_stated():
    cqm = _cqm("treatment", el("intervention", "hydrocortisone"), el("condition", "septic shock"))
    cs = chips(cqm)
    assert cs[0].id == "question_type"
    placeholder = [c for c in cs if c.id == "add:age_group"]
    assert placeholder and placeholder[0].display == "Not specified" and placeholder[0].state == "placeholder"


def test_edits_accept_add_remove_and_set_type():
    cqm = _cqm("treatment", el("intervention", "hydrocortisone"), el("condition", "septic shock"),
               el("comparator", "placebo", source="ai_inferred", active=False))
    comp = cqm.elements[2]
    edited = apply_edits(cqm, [
        ChipEdit(op="accept", element_id=comp.id),
        ChipEdit(op="add", role="age_group", value="adult"),
        ChipEdit(op="add", role="outcome", value="28-day mortality"),
    ])
    assert edited.age_group() == "adult"
    assert "compared with placebo" in edited.summary and "adults with septic shock" in edited.summary
    assert material_hash(edited) != material_hash(cqm)
    assert all(e.source == "user_edited" for e in edited.elements if e.value in ("placebo", "adult", "28-day mortality"))
    retyped = apply_edits(edited, [ChipEdit(op="set_type", value="prevention")])
    assert retyped.framework == "PICO" and retyped.question_type_source == "user_edited"
    removed = apply_edits(edited, [ChipEdit(op="remove", element_id=comp.id)])
    assert "placebo" not in removed.summary


def test_edit_validation():
    cqm = _cqm("treatment", el("intervention", "x"))
    with pytest.raises(EngineInputError):
        apply_edits(cqm, [ChipEdit(op="add", role="age_group", value="toddlers")])
    with pytest.raises(EngineInputError):
        apply_edits(cqm, [ChipEdit(op="remove", element_id="missing")])
    with pytest.raises(EngineInputError):
        apply_edits(cqm, [])


@pytest.mark.parametrize("qtype,elements,expected", [
    ("diagnosis", [el("index_test", "procalcitonin"), el("condition", "sepsis")],
     "Accuracy of procalcitonin for diagnosing sepsis in patients."),
    ("etiology_risk", [el("exposure", "air pollution"), el("outcome", "asthma"), el("age_group", "child")],
     "Association between air pollution and asthma in children."),
    ("background", [el("condition", "septic shock")], "Background on septic shock."),
])
def test_summary_sentences(qtype, elements, expected):
    assert build_summary(_cqm(qtype, *elements)) == expected


# ── readings from the live question trial (model outputs recorded 2026-09-25) ──


def test_age_codes_and_ages_are_not_populations():
    q = "Apixaban versus warfarin for stroke prevention in atrial fibrillation patients over 80"
    out = validate_model_output(q, {"question_type": "treatment", "elements": [
        {"role": "population", "value": "older_adult", "stated": True, "quote": "patients over 80"},
        {"role": "population", "value": "patients over 80", "stated": True, "quote": "patients over 80"},
        {"role": "population", "value": "aged 65 and older", "stated": False},
        {"role": "population", "value": "patients with atrial fibrillation", "stated": False},
    ]})
    assert [e.value for e in out.elements] == ["patients with atrial fibrillation"]


def test_generic_treatment_wording_is_not_a_treatment():
    q = "Treatment options for stage 4 prostate cancer"
    out = validate_model_output(q, {"question_type": "clinical_management", "elements": [
        {"role": "intervention", "value": "treatment options", "stated": True, "quote": "Treatment options"},
        {"role": "intervention", "value": "best therapy", "stated": False},
        {"role": "comparator", "value": "standard therapy", "stated": False},
        {"role": "condition", "value": "stage 4 prostate cancer", "stated": True, "quote": "stage 4 prostate cancer"},
    ]})
    assert [(e.role, e.value) for e in out.elements] == [
        ("comparator", "standard therapy"), ("condition", "stage 4 prostate cancer")]


@pytest.mark.parametrize("value,question,span", [
    ("early antibiotics", "Early versus delayed antibiotics for sepsis", "Early versus delayed antibiotics"),
    ("oral antibiotics", "Oral vs. intravenous antibiotics in children", "Oral vs. intravenous antibiotics"),
    ("high-dose aspirin", "high or low dose aspirin after stroke", "high or low dose aspirin"),
    ("early antibiotics", "Early antibiotics compared with delayed antibiotics", "Early antibiotics"),
])
def test_elided_comparison_arm_counts_as_stated(value, question, span):
    from clinical_engine.interpretation.model_reading import elliptical_span
    got = elliptical_span(value, question) if not quote_in_question(value, question) else value
    assert got.lower() == span.lower()


@pytest.mark.parametrize("value,question", [
    ("early mobilization", "Early versus delayed antibiotics for sepsis"),  # not in the question
    ("early antibiotics", "Early antibiotics"),                             # verbatim, not elided
    ("early sepsis", "Early versus delayed antibiotics for sepsis"),        # tail too far away
    ("antibiotics", "Early versus delayed antibiotics for sepsis"),         # single word
])
def test_elided_span_rejects_anything_else(value, question):
    from clinical_engine.interpretation.model_reading import elliptical_span
    assert elliptical_span(value, question) is None


async def test_both_arms_listed_as_interventions_keeps_the_comparator():
    q = "Apixaban versus warfarin for stroke prevention in atrial fibrillation patients over 80"
    gw = FakeGateway({"question_type": "treatment", "elements": [
        {"role": "intervention", "value": "apixaban", "stated": True, "quote": "Apixaban"},
        {"role": "intervention", "value": "warfarin", "stated": True, "quote": "warfarin"},
        {"role": "condition", "value": "atrial fibrillation", "stated": True, "quote": "atrial fibrillation"},
        {"role": "population", "value": "older_adult", "stated": True, "quote": "patients over 80"},
        {"role": "outcome", "value": "stroke prevention", "stated": True, "quote": "stroke prevention"},
    ]})
    result = await interpret(q, gateway=gw)
    cqm = result.map
    assert [e.value for e in cqm.active("intervention")] == ["apixaban"]
    assert [e.value.lower() for e in cqm.active("comparator")] == ["warfarin"]
    assert not [c for c in result.conflicts if c.severity == "blocking"]
    assert cqm.summary == ("Effect of apixaban compared with warfarin on stroke prevention "
                           "in older adults with atrial fibrillation.")


async def test_early_versus_delayed_reads_as_two_arms():
    q = "Early versus delayed antibiotics for emergency department patients with sepsis"
    gw = FakeGateway({"question_type": "treatment", "elements": [
        {"role": "condition", "value": "sepsis", "stated": True, "quote": "patients with sepsis"},
        {"role": "intervention", "value": "early antibiotics", "stated": True, "quote": "Early antibiotics"},
        {"role": "intervention", "value": "delayed antibiotics", "stated": True, "quote": "delayed antibiotics"},
    ]})
    result = await interpret(q, gateway=gw)
    cqm = result.map
    assert [e.value for e in cqm.active("intervention")] == ["early antibiotics"]
    assert [e.value for e in cqm.active("comparator")] == ["delayed antibiotics"]
    assert not cqm.suggestions() and not [c for c in result.conflicts if c.severity == "blocking"]
    assert cqm.summary == ("Effect of early antibiotics compared with delayed antibiotics on clinical outcomes "
                           "in patients with sepsis in the emergency department.")


def test_rules_fill_the_elided_noun_only_between_modifiers():
    r = roles(rules.read("Early versus delayed antibiotics for emergency department patients with sepsis"))
    assert r["intervention"] == ["early antibiotics"] and r["comparator"] == ["delayed antibiotics"]
    r = roles(rules.read("Surgery versus medical therapy for patients with angina"))
    assert r["intervention"] == ["surgery"] and r["comparator"] == ["medical therapy"]


async def test_incidence_after_a_procedure_keeps_the_patient_group():
    q = "What is the incidence of acute kidney injury after cardiac surgery?"
    r = roles(rules.read(q))
    assert r["condition"] == ["acute kidney injury"] and r["population"] == ["cardiac surgery"]
    gw = FakeGateway({"question_type": "epidemiology", "elements": [
        {"role": "condition", "value": "acute kidney injury", "stated": True, "quote": "acute kidney injury"},
        {"role": "population", "value": "cardiac surgery patients", "stated": False, "quote": None},
        {"role": "outcome", "value": "incidence", "stated": True, "quote": "incidence"},
    ]})
    cqm = (await interpret(q, gateway=gw)).map
    assert cqm.summary == "How common acute kidney injury is in cardiac surgery patients."
    assert not cqm.suggestions()  # the model's restatement of the stated group is not offered again


def test_treatment_with_an_outcome_but_no_condition_is_confirmable_with_a_warning():
    cqm = _cqm("treatment", el("intervention", "ketamine"), el("outcome", "agitation"),
               el("setting", "emergency department"))
    conflicts, warnings = check(cqm)
    assert not [c for c in conflicts if c.severity == "blocking"]
    assert any("No condition or patient group" in w for w in warnings)
    assert "missing_core_element" in codes(_cqm("treatment", el("intervention", "ketamine")))


@pytest.mark.parametrize("qtype,elements,expected", [
    ("treatment", [el("intervention", "dexamethasone"), el("outcome", "survival"),
                   el("population", "hospitalized adults"), el("condition", "severe COVID-19"),
                   el("age_group", "adult"), el("setting", "inpatient")],
     "Effect of dexamethasone on survival in hospitalized adults with severe COVID-19."),
    ("treatment", [el("intervention", "heparin"), el("condition", "pneumonia"), el("setting", "inpatient")],
     "Effect of heparin on clinical outcomes in patients with pneumonia in hospital."),
    ("prevention", [el("intervention", "chlorhexidine bathing"), el("outcome", "bloodstream infections"),
                    el("population", "ICU patients"), el("setting", "intensive care unit")],
     "Prevention of bloodstream infections with chlorhexidine bathing in ICU patients."),
    ("screening", [el("population", "adults with type 2 diabetes"),
                   el("index_test", "screening for diabetic retinopathy"),
                   el("outcome", "diabetic retinopathy"), el("age_group", "adult")],
     "Whether screening adults with type 2 diabetes for diabetic retinopathy improves outcomes."),
    ("screening", [el("index_test", "mammography"), el("outcome", "breast cancer mortality"),
                   el("population", "women")],
     "Whether screening women with mammography improves breast cancer mortality."),
    ("clinical_management", [el("condition", "stage 4 prostate cancer")],
     "How to manage stage 4 prostate cancer."),
])
def test_summary_wording_from_the_trial(qtype, elements, expected):
    assert build_summary(_cqm(qtype, *elements)) == expected


async def test_a_setting_must_be_a_place_of_care():
    q = "Timing and dose of corticosteroids in ARDS"
    gw = FakeGateway({"question_type": "clinical_management", "elements": [
        {"role": "intervention", "value": "corticosteroids", "stated": True, "quote": "corticosteroids"},
        {"role": "condition", "value": "acute respiratory distress syndrome", "stated": True, "quote": "ARDS"},
        {"role": "setting", "value": "timing and dose", "stated": True, "quote": "Timing and dose"},
    ]})
    result = await interpret(q, gateway=gw)
    assert not result.map.active("setting")
    assert [e.value for e in result.map.suggestions()] == ["timing and dose"]  # still offered, not applied
    assert result.map.summary == "How to manage acute respiratory distress syndrome with corticosteroids."
    out = validate_model_output("Steroids for sepsis in low-income countries", {"question_type": "treatment", "elements": [
        {"role": "setting", "value": "low-income countries", "stated": True, "quote": "low-income countries"}]})
    assert out.elements[0].active


def test_one_malformed_element_does_not_discard_the_reading():
    q = "How accurate is procalcitonin for diagnosing sepsis?"
    out = validate_model_output(q, {"question_type": "diagnosis", "elements": [
        {"role": "index_test", "value": "procalcitonin", "stated": True, "quote": "procalcitonin"},
        {"role": "condition", "value": "sepsis", "stated": True, "quote": "sepsis"},
        {"role": "reference_standard", "value": None, "stated": False},  # model did not know it
    ]})
    assert out.question_type == "diagnosis"
    assert [(e.role, e.value) for e in out.elements] == [("index_test", "procalcitonin"), ("condition", "sepsis")]
    assert out.rejected == ["element_invalid"]
    assert validate_model_output(q, ["not", "a", "dict"]).rejected == ["schema_invalid"]
