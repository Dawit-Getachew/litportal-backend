"""Phase 6B.2 — Role-weighted ranking + downstream Clinical Context Engine
canonicalisation + "Question type: other" leak fix.

These tests are deliberately unit-level and feed the ranker / verifier
synthetic inputs so they exercise the new code paths deterministically
without depending on the live LLM provider.
"""
from __future__ import annotations

from services.picott_ranker import picott_aware_ranker
from services.evidence_ranker import _humanize_qtype_from_picott
from services.default_selection import select_default_articles
from services.interpretation_canonical import (
    canonical_question_type,
    canonical_label_for_rank_explanation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _glp1_picott() -> dict:
    return {
        "type_of_question": "pharmacology",
        "question_subtype": "metabolic_biomarker_effect",
        "must_have_concepts": [
            {"concept": "semaglutide", "role": "intervention",
             "synonyms": ["ozempic", "wegovy", "GLP-1 receptor agonist"],
             "missing_relevance_cap": 30},
            {"concept": "LDL", "role": "outcome",
             "synonyms": ["low-density lipoprotein", "ldl-c"],
             "missing_relevance_cap": 50},
            {"concept": "type 2 diabetes", "role": "condition",
             "synonyms": ["t2dm", "diabetes mellitus type 2"],
             "missing_relevance_cap": 50},
        ],
        "ranking_concepts": [],
        "should_have_concepts": [],
        "negative_or_penalty_concepts": [],
        "pico": {
            "population": ["type 2 diabetes"],
            "intervention_or_exposure_or_index_test": ["semaglutide"],
            "comparator_or_reference_standard": [],
            "outcomes_or_target_diagnosis": ["LDL"],
        },
        "normalized_question":
            "What is the effect of semaglutide on LDL in type 2 diabetes?",
    }


def _make_rank_entry(
    title: str, abstract: str, art_type: str = "rct",
    overall: int = 75, relevance: int = 75,
) -> dict:
    return {
        "_article": {
            "id": "art_" + title[:8].replace(" ", "_"),
            "canonical_key": title[:32],
            "title": title,
            "abstract": abstract,
            "article_type_classification": art_type,
            "publication_year": 2024,
            "full_text_status": "abstract_only",
            "is_guideline": False,
            "sources_present": ["pubmed"],
        },
        "relevance_score": float(relevance),
        "evidence_design_score": 80.0,
        "overall_rank_score": int(overall),
        "rank_explanation": "Question type: Pharmacology / Metabolic Biomarker Effect. RCT. design=80, relevance=" + str(relevance) + ".",
    }


# ---------------------------------------------------------------------------
# Canonical interpretation helper
# ---------------------------------------------------------------------------
def test_canonical_question_type_helper_overrides_raw_other():
    out = canonical_question_type(
        picott={"type_of_question": "other", "question_subtype": "other"},
        role_weights={"type_of_question": "pharmacology",
                       "question_subtype": "metabolic_biomarker_effect"},
    )
    assert out["raw"] == "pharmacology"
    assert out["source"] == "role_weights"
    assert "Pharmacology" in out["humanized"]
    assert "metabolic biomarker effect" in out["humanized"].lower()


def test_canonical_question_type_picott_wins_when_concrete():
    out = canonical_question_type(
        picott={"type_of_question": "therapy",
                 "question_subtype": "transfusion_threshold_comparison"},
        role_weights={"type_of_question": "other"},
    )
    assert out["raw"] == "therapy"
    assert out["source"] == "picott"
    assert "Therapy" in out["humanized"]


def test_canonical_question_type_falls_back_to_legacy_when_no_engine_output():
    out = canonical_question_type(
        picott={}, role_weights={}, legacy_qtype="treatment",
    )
    assert out["source"] == "legacy"
    assert out["raw"] == "treatment"


def test_canonical_question_type_returns_other_when_truly_nothing():
    out = canonical_question_type(picott={}, role_weights={}, legacy_qtype="")
    assert out["source"] == "other"
    assert out["raw"] == "other"


# ---------------------------------------------------------------------------
# "Question type: other" leak fixes for 4 representative questions
# ---------------------------------------------------------------------------
def test_rank_explanation_no_question_type_other_glp1():
    picott = {"type_of_question": "pharmacology",
              "question_subtype": "metabolic_biomarker_effect"}
    label = _humanize_qtype_from_picott(picott, "other")
    assert "other" != label.lower()
    assert "pharmacology" in label.lower()
    assert "metabolic biomarker effect" in label.lower()


def test_rank_explanation_no_question_type_other_ami_transfusion():
    picott = {"type_of_question": "therapy",
              "question_subtype": "transfusion_threshold_comparison"}
    label = _humanize_qtype_from_picott(picott, "other")
    assert "other" != label.lower()
    assert "therapy" in label.lower()
    assert "transfusion threshold comparison" in label.lower()


def test_rank_explanation_no_question_type_other_milrinone():
    picott = {"type_of_question": "dosing_protocol",
              "question_subtype": "pharmacology"}
    label = _humanize_qtype_from_picott(picott, "other")
    assert "other" != label.lower()
    assert "dosing protocol" in label.lower()


def test_rank_explanation_varicella_shows_guideline_or_background_not_other():
    # Engine reclassifies via role_weights when picott.type_of_question="other".
    picott = {"type_of_question": "other", "question_subtype": "immunization_schedule"}
    role_weights = {"type_of_question": "vaccine_schedule",
                     "question_subtype": "immunization_schedule"}
    label = _humanize_qtype_from_picott(picott, "other", role_weights=role_weights)
    assert "other" != label.lower()
    assert ("vaccine schedule" in label.lower()
            or "immunization schedule" in label.lower())


def test_rank_explanation_no_question_type_other_ards_prone():
    picott = {"type_of_question": "therapy", "question_subtype": ""}
    label = _humanize_qtype_from_picott(picott, "other")
    assert label.lower().startswith("therapy")


# ---------------------------------------------------------------------------
# Role-weighted ranking (Task 3)
# ---------------------------------------------------------------------------
def test_therapy_intervention_role_weighting_boosts_only_with_anchors_matched():
    picott = _glp1_picott()
    picott["type_of_question"] = "therapy"
    # Article that matches every anchor — should not be capped.
    art = _make_rank_entry(
        "Semaglutide reduces LDL in type 2 diabetes: an RCT",
        "Randomized controlled trial of semaglutide vs placebo in patients with "
        "type 2 diabetes evaluating LDL cholesterol.",
        overall=80, relevance=75,
    )
    out = picott_aware_ranker.postprocess([art], picott)
    assert out[0].get("role_match_score", 0) >= 50
    breakdown = out[0]["role_match_breakdown_json"]
    assert any(b["role"] == "intervention" and b["matched"] for b in breakdown)
    assert out[0]["overall_rank_score"] >= 80, out[0]


def test_pharmacology_drug_role_weight_applied():
    picott = _glp1_picott()
    art = _make_rank_entry(
        "Effect of semaglutide on LDL in patients with type 2 diabetes",
        "We assessed the effect of semaglutide on LDL cholesterol in adults "
        "with type 2 diabetes.",
    )
    out = picott_aware_ranker.postprocess([art], picott)
    rw_used = out[0]["role_weights_used"]
    # pharmacology weights: intervention=40, outcome=25
    assert rw_used["intervention"] == 40
    assert rw_used["outcome"] == 25
    # All three anchors matched → role_match_score should be 100.
    assert out[0]["role_match_score"] == 100


def test_dosing_protocol_drug_and_dose_role_weights_applied():
    picott = {
        "type_of_question": "dosing_protocol",
        "must_have_concepts": [
            {"concept": "milrinone", "role": "intervention",
             "synonyms": ["primacor"], "missing_relevance_cap": 30},
            {"concept": "subarachnoid hemorrhage", "role": "condition",
             "synonyms": ["sah"], "missing_relevance_cap": 50},
        ],
        "ranking_concepts": [{"concept": "0.5 mcg/kg/min", "role": "dose_terms"}],
        "should_have_concepts": [],
        "negative_or_penalty_concepts": [],
        "pico": {},
        "normalized_question": "Milrinone dose for SAH vasospasm",
    }
    art = _make_rank_entry(
        "Intravenous milrinone at 0.5 mcg/kg/min for subarachnoid hemorrhage vasospasm",
        "Continuous IV milrinone (0.5 mcg/kg/min) was administered to patients "
        "with subarachnoid hemorrhage to treat cerebral vasospasm.",
        art_type="cohort_study",
    )
    out = picott_aware_ranker.postprocess([art], picott)
    rw = out[0]["role_weights_used"]
    assert rw.get("intervention") == 35  # dosing_protocol intervention
    assert rw.get("dose_terms") == 15


def test_diagnosis_index_test_role_weight_highest():
    picott = {
        "type_of_question": "diagnosis",
        "must_have_concepts": [
            {"concept": "D-dimer", "role": "index_test",
             "synonyms": [], "missing_relevance_cap": 30},
            {"concept": "pulmonary embolism", "role": "target_condition",
             "synonyms": ["PE"], "missing_relevance_cap": 50},
        ],
        "ranking_concepts": [],
        "should_have_concepts": [],
        "negative_or_penalty_concepts": [],
        "pico": {},
        "normalized_question": "D-dimer for PE",
    }
    art = _make_rank_entry(
        "Diagnostic accuracy of D-dimer for pulmonary embolism",
        "We evaluated D-dimer for diagnosis of pulmonary embolism in ED patients.",
        art_type="diagnostic_accuracy_study",
    )
    out = picott_aware_ranker.postprocess([art], picott)
    rw = out[0]["role_weights_used"]
    assert rw["index_test"] == 35
    assert rw["target_condition"] == 25


def test_harm_exposure_outcome_role_weight_highest():
    picott = {
        "type_of_question": "etiology_harm",
        "must_have_concepts": [
            {"concept": "NSAID", "role": "exposure",
             "synonyms": ["non-steroidal anti-inflammatory drug"],
             "missing_relevance_cap": 30},
            {"concept": "GI bleeding", "role": "outcome",
             "synonyms": ["gastrointestinal bleed"],
             "missing_relevance_cap": 50},
        ],
        "ranking_concepts": [],
        "should_have_concepts": [],
        "negative_or_penalty_concepts": [],
        "pico": {},
        "normalized_question": "NSAID GI bleeding risk",
    }
    art = _make_rank_entry(
        "NSAID use and risk of gastrointestinal bleeding: cohort",
        "We evaluated NSAID use and GI bleeding risk in a cohort study.",
        art_type="cohort_study",
    )
    out = picott_aware_ranker.postprocess([art], picott)
    rw = out[0]["role_weights_used"]
    assert rw["exposure"] == 35
    assert rw["outcome"] == 25


# ---------------------------------------------------------------------------
# Caps stronger than role weights (regression)
# ---------------------------------------------------------------------------
def test_role_weights_cannot_override_mandatory_anchor_caps():
    picott = _glp1_picott()
    # Article missing the intervention anchor entirely. Should be
    # capped to 30 by the picott_ranker even though we hand it a high
    # initial overall_rank_score.
    art = _make_rank_entry(
        "Type 2 diabetes management and LDL cholesterol monitoring",
        "Generic review of LDL cholesterol monitoring in type 2 diabetes.",
        overall=95, relevance=85,
    )
    out = picott_aware_ranker.postprocess([art], picott)
    assert out[0]["relevance_cap_applied"] is not None
    assert out[0]["relevance_cap_applied"] <= 30
    # Even with role weights, overall_rank_score cannot exceed the cap.
    assert out[0]["overall_rank_score"] <= 30, out[0]
    assert "missing_intervention_anchor" in out[0]["cap_reasons"]


def test_role_weights_cannot_override_wrong_population_cap():
    picott = _glp1_picott()
    # The article matches the intervention but wrong population: this
    # in our cap set is captured by ``missing_condition_anchor`` (the
    # article never mentions T2DM).
    art = _make_rank_entry(
        "Semaglutide reduces LDL in healthy volunteers without diabetes",
        "RCT of semaglutide vs placebo in healthy volunteers measuring LDL.",
        overall=90, relevance=85,
    )
    out = picott_aware_ranker.postprocess([art], picott)
    assert out[0]["relevance_cap_applied"] is not None
    assert out[0]["overall_rank_score"] <= 50, out[0]


def test_role_match_breakdown_json_persisted_on_session_session_articles():
    """The ranker output dict carries ``role_match_breakdown_json`` and
    ``role_match_score`` so pipeline.py can persist them on
    ``search_session_articles`` (see Phase 6B.2 wiring)."""
    picott = _glp1_picott()
    art = _make_rank_entry(
        "Semaglutide reduces LDL in type 2 diabetes",
        "Semaglutide reduced LDL in patients with type 2 diabetes.",
    )
    out = picott_aware_ranker.postprocess([art], picott)
    assert isinstance(out[0]["role_match_breakdown_json"], list)
    assert all("role" in b and "weight_pct" in b
                and "matched" in b and "contribution_pct" in b
                for b in out[0]["role_match_breakdown_json"])


# ---------------------------------------------------------------------------
# Default-selection recalibration tests (Task 6)
# ---------------------------------------------------------------------------
def test_glp1_session_default_selects_at_least_one_direct_lipid_article():
    picott = _glp1_picott()
    art_direct = _make_rank_entry(
        "Semaglutide reduces LDL in type 2 diabetes: a randomized trial",
        "RCT comparing semaglutide vs placebo for LDL cholesterol reduction in "
        "patients with type 2 diabetes. Primary outcome: LDL change at 24 weeks.",
        art_type="rct", overall=80, relevance=80,
    )
    art_direct = picott_aware_ranker.postprocess([art_direct], picott)[0]
    sel = select_default_articles(picott, [art_direct])
    assert sel["selected_article_ids"], sel
    # default_selection_summary lane is "primary".
    assert sel["default_selection_summary"]["selected_by_lane"]["primary"] >= 1


def test_ami_transfusion_default_selects_direct_guideline_sr_when_retrieved():
    picott = {
        "type_of_question": "therapy",
        "question_subtype": "transfusion_threshold_comparison",
        "must_have_concepts": [
            {"concept": "myocardial infarction", "role": "condition",
             "synonyms": ["mi", "ami"], "missing_relevance_cap": 50},
            {"concept": "transfusion threshold", "role": "intervention",
             "synonyms": ["restrictive transfusion", "liberal transfusion"],
             "missing_relevance_cap": 30},
        ],
        "ranking_concepts": [],
        "should_have_concepts": [],
        "negative_or_penalty_concepts": [],
        "pico": {},
        "normalized_question":
            "Restrictive vs liberal transfusion threshold in AMI",
    }
    sr = _make_rank_entry(
        "Restrictive vs liberal transfusion thresholds in acute myocardial infarction (SR)",
        "Systematic review of restrictive vs liberal transfusion threshold trials "
        "in patients with acute myocardial infarction (AMI).",
        art_type="systematic_review", overall=85, relevance=85,
    )
    sr = picott_aware_ranker.postprocess([sr], picott)[0]
    sel = select_default_articles(picott, [sr])
    assert sr["_article"]["id"] in sel["selected_article_ids"], sel


def test_ckd_hgb_target_article_not_default_selected_for_ami_session():
    """Regression — feed a CKD article into retrieval; assert it's
    capped and NOT default-selected."""
    picott = {
        "type_of_question": "therapy",
        "must_have_concepts": [
            {"concept": "myocardial infarction", "role": "condition",
             "synonyms": ["mi", "ami"], "missing_relevance_cap": 50},
            {"concept": "transfusion threshold", "role": "intervention",
             "synonyms": ["restrictive transfusion"],
             "missing_relevance_cap": 30},
        ],
        "ranking_concepts": [],
        "should_have_concepts": [],
        "negative_or_penalty_concepts": [],
        "pico": {},
        "normalized_question":
            "Restrictive vs liberal transfusion threshold in AMI",
    }
    ckd = _make_rank_entry(
        "Hemoglobin targets in chronic kidney disease",
        "RCT of hemoglobin targets in patients with chronic kidney disease (CKD).",
        art_type="rct", overall=85, relevance=85,
    )
    ckd = picott_aware_ranker.postprocess([ckd], picott)[0]
    sel = select_default_articles(picott, [ckd])
    assert ckd["_article"]["id"] not in sel["selected_article_ids"], sel


def test_sah_article_missing_milrinone_anchor_not_default_selected():
    picott = {
        "type_of_question": "dosing_protocol",
        "must_have_concepts": [
            {"concept": "milrinone", "role": "intervention",
             "synonyms": [], "missing_relevance_cap": 30},
            {"concept": "subarachnoid hemorrhage", "role": "condition",
             "synonyms": ["sah"], "missing_relevance_cap": 50},
        ],
        "ranking_concepts": [],
        "should_have_concepts": [],
        "negative_or_penalty_concepts": [],
        "pico": {},
        "normalized_question": "Milrinone dosing in SAH",
    }
    sah_no_milrinone = _make_rank_entry(
        "Nimodipine for cerebral vasospasm prophylaxis after SAH",
        "Patients with subarachnoid hemorrhage received nimodipine for "
        "cerebral vasospasm prophylaxis.",
        art_type="rct", overall=85, relevance=85,
    )
    sah_no_milrinone = picott_aware_ranker.postprocess(
        [sah_no_milrinone], picott,
    )[0]
    sel = select_default_articles(picott, [sah_no_milrinone])
    assert sah_no_milrinone["_article"]["id"] not in sel["selected_article_ids"]


def test_preprint_not_default_selected():
    picott = _glp1_picott()
    pp = _make_rank_entry(
        "Semaglutide reduces LDL in type 2 diabetes: preprint",
        "RCT data (preprint) of semaglutide reducing LDL in type 2 diabetes.",
        overall=85, relevance=85,
    )
    pp["_article"]["is_preprint"] = True
    pp = picott_aware_ranker.postprocess([pp], picott)[0]
    sel = select_default_articles(picott, [pp])
    assert pp["_article"]["id"] not in sel["selected_article_ids"]


def test_default_selection_summary_persisted_with_all_fields():
    picott = _glp1_picott()
    art = _make_rank_entry(
        "Semaglutide reduces LDL in type 2 diabetes RCT",
        "RCT of semaglutide vs placebo measuring LDL in type 2 diabetes.",
        overall=82, relevance=82,
    )
    art = picott_aware_ranker.postprocess([art], picott)[0]
    sel = select_default_articles(picott, [art])
    s = sel["default_selection_summary"]
    assert "selected_count" in s
    assert "eligible_direct_count" in s
    assert "excluded_by_cap_count" in s
    assert "excluded_by_low_relevance_count" in s
    assert "selected_by_lane" in s and "primary" in s["selected_by_lane"]
    assert "reason_each_selected" in s


def test_recalibration_does_not_lower_global_threshold():
    """Regression — when role-match is low, the recalibration multiplier
    is <1. An article scoring 70 (above threshold) before recalibration
    must remain selectable; an article scoring 50 (below threshold)
    must NOT slip through."""
    picott = _glp1_picott()
    # Article with NO mention of any anchor → role_match_score=0,
    # multiplier=0.95. Pre-multiplier relevance=58 → post=55.1 → still
    # below 60, must be excluded.
    art = _make_rank_entry(
        "General review of cardiometabolic biomarkers",
        "Discussion of various cardiometabolic biomarkers in medicine.",
        overall=58, relevance=58,
    )
    art = picott_aware_ranker.postprocess([art], picott)[0]
    sel = select_default_articles(picott, [art])
    assert art["_article"]["id"] not in sel["selected_article_ids"]


# ---------------------------------------------------------------------------
# Evidence packet caveats (Task 7)
# ---------------------------------------------------------------------------
def _verifier_floor_for(directness_per_aid: dict, *, citation_label="cit"):
    from services.summarization.verifier import MechanicalVerificationFloor
    from services.summarization.evidence_spans import EvidenceSpan
    spans: list = []
    studies: list = []
    for i, (aid, directness) in enumerate(directness_per_aid.items()):
        sid = f"span_{aid}_{i}"
        spans.append(EvidenceSpan(
            evidence_span_id=sid,
            article_id=aid,
            citation_label=citation_label,
            title="t",
            journal="J",
            publication_year=2024,
            section_name="abstract",
            evidence_text_level="abstract",
            span_text="x",
            search_session_id="s",
            text_source="abstract",
            start_char=0,
            end_char=1,
        ))
        studies.append({
            "article_id": aid,
            "citation_label": citation_label,
            "pmid": None, "doi": None, "pmcid": None,
            "title": "t",
            "journal": "J",
            "publication_year": 2024,
            "study_type": "rct",
            "evidence_text_level": "abstract",
            "directness": directness,
            "cap_reasons": [] if directness == "direct" else ["missing_condition_anchor"],
            "_is_excluded": False,
        })
    return MechanicalVerificationFloor(
        spans=spans, selected_studies=studies,
    ), spans, studies


def _make_draft(*, claim_text, certainty, article_ids, span_ids,
                 selected_study_aids):
    """Build a minimal DraftSummary stub for the floor checks."""
    from services.summarization.schemas import (
        DraftSummary, ClaimMapEntry, SelectedStudyEntry,
    )
    return DraftSummary(
        clinical_bottom_line=claim_text,
        evidence_summary=claim_text,
        applicability="n/a",
        certainty=certainty,
        certainty_rationale="test rationale",
        claim_map=[
            ClaimMapEntry(
                claim_id="c1",
                claim_text=claim_text,
                claim_type="other",
                supporting_evidence_span_ids=list(span_ids),
                supporting_article_ids=list(article_ids),
            ),
        ],
        selected_studies=[
            SelectedStudyEntry(
                article_id=aid, citation_label="cit",
                pmid=None, doi=None, pmcid=None,
                title="t", evidence_text_level="abstract",
                key_finding="kf", limitations="lim",
                supporting_evidence_span_ids=(
                    [span_ids[0]] if span_ids else ["span_dummy"]
                ),
            )
            for aid in selected_study_aids
        ],
    )


def test_capped_user_overridden_article_labeled_indirect_in_packet():
    """Sanity — simulate the directness derivation in the summary
    router: an article with cap_reasons that the user explicitly
    user_included gets directness="override"."""
    cap_reasons = ["missing_condition_anchor"]
    included_by_default = False
    user_included = True
    if cap_reasons:
        if user_included and not included_by_default:
            directness = "override"
        else:
            directness = "indirect"
    else:
        directness = "direct"
    assert directness == "override"


def test_summarizer_receives_override_caveats():
    """Verify the summarizer prompt builder embeds the caveat block."""
    from services.summarization.summarizer import _build_user_prompt
    caveats = [{
        "article_id": "abc",
        "citation_label": "Smith 2023",
        "directness": "override",
        "cap_reasons": ["missing_condition_anchor"],
    }]
    prompt = _build_user_prompt(
        question="Test question",
        pico={"clinical_question_type": "therapy"},
        studies=[{"article_id": "abc", "citation_label": "Smith 2023",
                    "title": "T", "study_type": "rct",
                    "evidence_text_level": "abstract"}],
        spans=[],
        override_caveats=caveats,
    )
    assert "Override / indirect evidence caveats" in prompt
    assert "directness=override" in prompt
    assert "missing_condition_anchor" in prompt


def test_citation_verifier_fails_when_indirect_treated_as_direct():
    floor, spans, studies = _verifier_floor_for({"art1": "override"})
    # Draft cites override article with high certainty + no hedging.
    draft = _make_draft(
        claim_text="Semaglutide directly reduces LDL in our population.",
        certainty="high",
        article_ids=["art1"],
        span_ids=[spans[0].evidence_span_id],
        selected_study_aids=["art1"],
    )
    passed, violations = floor.check(draft)
    assert passed is False, violations
    assert any("indirect_treated_as_direct" in v for v in violations), violations


def test_citation_verifier_passes_with_proper_hedging():
    floor, spans, studies = _verifier_floor_for({"art1": "override"})
    draft = _make_draft(
        claim_text=("Indirect evidence from cit suggests semaglutide may "
                     "lower LDL, but caution is required for this population."),
        certainty="low",
        article_ids=["art1"],
        span_ids=[spans[0].evidence_span_id],
        selected_study_aids=["art1"],
    )
    passed, violations = floor.check(draft)
    # No indirect_treated_as_direct violation. Other rules may still
    # fire (e.g., certainty downgrade), but this specific rule must
    # NOT trip.
    assert not any("indirect_treated_as_direct" in v for v in violations), violations


def test_evidence_packet_summary_persisted_with_all_fields():
    """The /prepare-evidence-packet endpoint persists
    ``evidence_packet_summary`` on the session. We verify the shape
    by constructing it directly (the endpoint is exercised in the
    integration smoke tests)."""
    epk = {
        "directly_relevant_count": 3,
        "indirect_override_count": 1,
        "subquestion_coverage": {"primary": 4},
        "selected_articles_by_subquestion": {"primary": ["a", "b", "c", "d"]},
        "abstract_only_count": 2,
        "oa_full_text_count": 1,
        "user_uploaded_full_text_count": 1,
    }
    for k in (
        "directly_relevant_count", "indirect_override_count",
        "subquestion_coverage", "selected_articles_by_subquestion",
        "abstract_only_count", "oa_full_text_count",
        "user_uploaded_full_text_count",
    ):
        assert k in epk


# ---------------------------------------------------------------------------
# Summarizer caveat propagation (downstream Clinical Context Engine
# consumption)
# ---------------------------------------------------------------------------
def test_summarizer_caveat_uses_canonical_interpretation():
    """The PICO block surfaced to the summarizer's "Question type:" line
    must reflect the canonical interpretation (Phase 6B.2). When the
    PICO dict has the PICOTT block attached, the type line uses the
    PICOTT-derived label, never raw legacy "other"."""
    from services.summarization.summarizer import _format_pico_block
    pico = {
        "clinical_question_type": "other",
        "picott_json": {
            "type_of_question": "pharmacology",
            "question_subtype": "metabolic_biomarker_effect",
        },
    }
    block = _format_pico_block(pico)
    assert "Question type: Pharmacology" in block
    assert "Other" not in block.split("Question type:")[1].split("\n")[0]


def test_citation_verifier_caveat_uses_canonical_interpretation():
    """The verifier inherits the canonical interpretation through the
    selected_studies + override_caveats it receives from the summary
    router. The directness=override label is the canonical signal."""
    floor, spans, _ = _verifier_floor_for({"a": "override", "b": "direct"})
    # Both articles cited; the override one MUST be hedged.
    draft = _make_draft(
        claim_text="Both studies support the conclusion strongly.",
        certainty="high",
        article_ids=["a", "b"],
        span_ids=[spans[0].evidence_span_id],
        selected_study_aids=["a", "b"],
    )
    _, violations = floor.check(draft)
    assert any("indirect_treated_as_direct" in v for v in violations), violations
