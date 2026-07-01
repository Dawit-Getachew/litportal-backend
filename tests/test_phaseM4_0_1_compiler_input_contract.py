"""Phase M4.0.1-shadow — compiler-input contract projection tests.

Deterministic (no LLM/network). Validates slot mapping, conservative slot
policy, dry-run query families, projection validation flags, and the
preserved/dropped report against the spec acceptance cases 1-8.
"""
from __future__ import annotations

from services.m4_full_question_interpreter import M4Candidate
import services.m4_compiler_input_projection as proj


def _cand(**over) -> M4Candidate:
    base = dict(
        primary_intent="prognosis_natural_history",
        secondary_intents=[],
        frameworks=["PFO", "prognosis"],
        clinical_problem=["colon cancer"],
        stage=["stage I"],
        biomarker=[],
        population=["patients with stage I colon cancer"],
        exposure=[],
        intervention=[],
        procedure=[],
        medication=[],
        index_test=[],
        outcome=["overall survival", "disease-free survival", "recurrence", "mortality"],
        setting=[],
        timeframe=[],
        mechanism=[],
        comparator=[],
        required_anchors=["colon cancer"],
        optional_or_ranking_only_terms=["prognosis", "stage I"],
        candidate_synonyms_unvalidated=["colon carcinoma"],
        candidate_mesh_terms_unvalidated=["Colonic Neoplasms", "Survival Rate"],
        clarification_required=False,
        confidence=0.9,
        repair_reason="long_literal_phrase_anchor",
        provenance="m4_full_question_interpreter_shadow",
    )
    base.update(over)
    return M4Candidate.model_validate(base)


def _families_for(c):
    ci = proj.project_compiler_input(c)
    return ci, proj.project_query_families(c, ci)


# ----------------------- slot mapping -----------------------
def test_slot_mapping_preserves_stage_outcome():
    ci = proj.project_compiler_input(_cand())
    assert ci["required_anchors"] == ["colon cancer"]
    assert ci["stage_terms"] == ["stage I"]
    assert "overall survival" in ci["outcome_terms"]
    # stage is a precision + ranking term, never a broad required anchor.
    assert "stage I" in ci["high_precision_terms"]
    assert "stage I" in ci["ranking_only_terms"]
    assert "stage I" not in ci["required_anchors"]
    # unvalidated mesh stays in its slot.
    assert "Colonic Neoplasms" in ci["mesh_terms_unvalidated"]


def test_query_policy_marks_mesh_not_applied():
    ci = proj.project_compiler_input(_cand())
    assert ci["query_policy"]["mesh_terms_unvalidated"] == "not_applied_pending_validation"
    assert ci["query_policy"]["stage_terms"].startswith("precision_and_ranking")


# ----------------------- query families -----------------------
def test_broad_recall_has_only_anchor_in_must():
    ci, fams = _families_for(_cand())
    broad = next(f for f in fams if f["family"] == "broad_recall")
    assert broad["must"] == ["colon cancer"]
    # stage must NOT be a hard filter on the recall family.
    assert "stage I" not in broad["must"]


def test_high_precision_family_keeps_stage():
    ci, fams = _families_for(_cand())
    hp = next(f for f in fams if f["family"] == "high_precision")
    assert "stage I" in hp["precision"]
    assert hp["must"] == ["colon cancer"]


def test_prognosis_family_present_for_prognosis_intent():
    ci, fams = _families_for(_cand())
    pf = next((f for f in fams if f["family"] == "prognosis_secondary_evidence"), None)
    assert pf is not None
    assert any("survival" in s.lower() for s in pf["should"])


def test_mechanism_family_for_mechanism_candidate():
    c = _cand(
        primary_intent="mechanism", frameworks=["mechanism"],
        clinical_problem=["cholestatic liver disease"], stage=[], outcome=[],
        mechanism=["pruritus"], required_anchors=["cholestatic liver disease"],
        optional_or_ranking_only_terms=["pruritus", "itch"],
        candidate_mesh_terms_unvalidated=["Cholestasis"], repair_reason="intent_word_fused_in_anchor",
    )
    ci, fams = _families_for(c)
    mf = next((f for f in fams if f["family"] == "mechanism_translational"), None)
    assert mf is not None
    assert "pruritus" in mf["precision"]


# ----------------------- validation -----------------------
def test_validation_clean_for_good_candidate():
    c = _cand()
    ci, fams = _families_for(c)
    assert proj.validate_projection(c, ci, fams) == []


def test_validation_flags_stage_dropped():
    c = _cand()
    ci, fams = _families_for(c)
    ci["stage_terms"] = []  # simulate a projector bug
    flags = proj.validate_projection(c, ci, fams)
    assert "stage_dropped" in flags


def test_validation_flags_stage_specific_broadened():
    c = _cand()
    ci = proj.project_compiler_input(c)
    # families that drop stage entirely (only broad recall, no stage anywhere).
    fams = [{"family": "broad_recall", "must": ["colon cancer"], "precision": [], "should": []}]
    flags = proj.validate_projection(c, ci, fams)
    assert "stage_specific_question_broadened" in flags


def test_validation_flags_literal_phrase_anchor():
    c = _cand(required_anchors=["colon cancer"])
    ci = proj.project_compiler_input(c)
    ci["required_anchors"] = ["prognosis of stage 1 colon cancer"]
    fams = proj.project_query_families(c, ci)
    flags = proj.validate_projection(c, ci, fams)
    assert "literal_phrase_in_required_anchor" in flags


def test_validation_flags_unvalidated_mesh_applied():
    c = _cand()
    ci = proj.project_compiler_input(c)
    fams = proj.project_query_families(c, ci)
    fams[0]["should"].append("Colonic Neoplasms")  # leak a mesh term into a family
    flags = proj.validate_projection(c, ci, fams)
    assert "unvalidated_mesh_applied" in flags


def test_validation_flags_generic_anchor():
    c = _cand(required_anchors=["patients"], clinical_problem=["patients"])
    ci = proj.project_compiler_input(c)
    fams = proj.project_query_families(c, ci)
    flags = proj.validate_projection(c, ci, fams)
    assert "disease_anchor_generic" in flags


# ----------------------- preserved / dropped -----------------------
def test_preserved_dropped_report():
    c = _cand(biomarker=["HER2-positive"], clinical_problem=["breast cancer"],
              required_anchors=["breast cancer"], stage=["stage II"])
    ci = proj.project_compiler_input(c)
    preserved, dropped = proj.compute_preserved_dropped(c, ci)
    assert "required_anchors" in preserved
    assert "stage_terms" in preserved
    assert "biomarker_terms" in preserved
    assert "outcome_terms" in preserved
    assert dropped == []


# ----------------------- spec acceptance cases 1-8 -----------------------
def test_case1_prognosis_stage1_colon():
    c = _cand()
    blk = proj.build_m4_0_1_projection(c, baseline_primary_intent="background",
                                       baseline_required_anchors=["prognosis of stage 1 colon cancer"])
    ci = blk["projected_compiler_input"]
    assert "colon cancer" in ci["required_anchors"]
    assert any("stage" in s.lower() for s in ci["stage_terms"])
    assert any("survival" in s.lower() for s in ci["outcome_terms"])
    assert all(not proj._is_long_literal(a) for a in ci["required_anchors"])
    hp = next(f for f in blk["dry_run_query_families"] if f["family"] == "high_precision")
    assert any("stage" in s.lower() for s in hp["precision"])
    assert blk["projection_validation_status"] == "passed"
    assert blk["active_retrieval_changed"] is False
    assert blk["dispatched_any_source"] is False


def test_case4_her2_stage2_breast():
    c = _cand(
        clinical_problem=["breast cancer"], stage=["stage II"],
        biomarker=["HER2-positive"], required_anchors=["breast cancer"],
        population=["patients with HER2-positive stage II breast cancer"],
        outcome=["overall survival", "recurrence", "mortality"],
        optional_or_ranking_only_terms=["prognosis", "HER2-positive", "stage II"],
        candidate_mesh_terms_unvalidated=["Breast Neoplasms"],
        repair_reason="stage_biomarker_population_fused",
    )
    blk = proj.build_m4_0_1_projection(c, baseline_primary_intent="prognosis_natural_history",
                                       baseline_required_anchors=["HER2 positive stage 2 breast cancer"])
    ci = blk["projected_compiler_input"]
    assert ci["required_anchors"] == ["breast cancer"]
    assert "stage II" in ci["stage_terms"]
    assert "HER2-positive" in ci["biomarker_terms"]
    assert blk["projection_validation_status"] == "passed"
    assert "biomarker_terms" in blk["preserved_slots"]


def test_case7_procedure_complication_family():
    c = _cand(
        primary_intent="complication_after_procedure", frameworks=["complication", "PFO"],
        clinical_problem=[], stage=[], outcome=["pneumothorax"],
        procedure=["central venous catheterization"],
        required_anchors=["central venous catheterization"],
        optional_or_ranking_only_terms=["pneumothorax", "central line"],
        candidate_mesh_terms_unvalidated=["Pneumothorax"],
        repair_reason="intent_word_fused_in_anchor",
    )
    blk = proj.build_m4_0_1_projection(c)
    fams = blk["dry_run_query_families"]
    pc = next((f for f in fams if f["family"] == "procedure_complication"), None)
    assert pc is not None
    assert "pneumothorax" in [s.lower() for s in pc["precision"]]
    assert blk["projected_compiler_input"]["outcome_terms"] == ["pneumothorax"]
    assert blk["projection_validation_status"] == "passed"


def test_case8_medication_safety_family():
    c = _cand(
        primary_intent="drug_safety", frameworks=["narrative_overview"],
        clinical_problem=[], stage=[], outcome=[],
        medication=["dexmedetomidine"], required_anchors=["dexmedetomidine"],
        optional_or_ranking_only_terms=["agitation"],
        candidate_mesh_terms_unvalidated=["Dexmedetomidine"],
        repair_reason="intent_word_fused_in_anchor",
    )
    blk = proj.build_m4_0_1_projection(c)
    fams = blk["dry_run_query_families"]
    ms = next((f for f in fams if f["family"] == "medication_safety"), None)
    assert ms is not None
    assert ms["must"] == ["dexmedetomidine"]
    assert blk["projected_compiler_input"]["drug_terms"] == ["dexmedetomidine"]


def test_build_returns_empty_when_no_candidate():
    assert proj.build_m4_0_1_projection(None) == {}
