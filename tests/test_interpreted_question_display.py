"""Phase 5D.1 — display normalizer (interpreted_question_display_json).

Test Groups A + B + BC fallback:
  A. AMI/transfusion session displays therapy / cardiology / liberal / restrictive,
     no "primary_care", no "difference between" fragments.
  B. Milrinone/SAH session displays pharmacology / dosing_protocol / neurology,
     milrinone intervention, SAH/aSAH population, vasospasm/DCI ranking.
  Fallback. Sessions with only legacy pico_json gracefully fall back.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.interpreted_question_display import (  # noqa: E402
    NOT_SPECIFIED,
    build_interpreted_question_display,
)


# =============================================================================
# Fixtures — full session_doc shape
# =============================================================================

def _ami_session_doc() -> dict:
    """Realistic AMI/transfusion-threshold session document shape with
    PICOTT populated by the Phase 5A parser. Legacy pico_json + legacy
    clinical_question_type carry the OLD parser's mis-extractions
    ("primary_care" specialty, "other" question type, "difference between"
    fragment) — the display normalizer MUST NOT let any of those leak."""
    return {
        "id": "ami-sess",
        "original_question": (
            "What is the optimal red-cell transfusion threshold in adults "
            "with acute myocardial infarction and anemia?"
        ),
        # Legacy parser noise — MUST NOT override PICOTT.
        "clinical_question_type": "other",
        "pico_json": {
            "population": "patients with MI",
            "intervention_or_exposure": "difference between liberal and restrictive transfusion",
            "comparator": "",
            "outcomes": ["mortality"],
            "key_concepts": [
                "difference between",
                "between high",
                "transfusion",
            ],
            "specialty": "primary_care",
        },
        # PICOTT — authoritative.
        "picott_json": {
            "question_scope": "foreground",
            "type_of_question": "therapy",
            "question_subtype": "transfusion_threshold_comparison",
            "specialty": "cardiology",
            "normalized_question": (
                "what is the optimal red-cell transfusion threshold in adults "
                "with acute myocardial infarction and anemia"
            ),
            "pico": {
                "population": [
                    "acute myocardial infarction",
                    "acute coronary syndrome",
                    "anemia",
                ],
                "intervention_or_exposure_or_index_test": [
                    "liberal red blood cell transfusion strategy",
                    "higher hemoglobin threshold",
                ],
                "comparator_or_reference_standard": [
                    "restrictive red blood cell transfusion strategy",
                    "lower hemoglobin threshold",
                ],
                "outcomes_or_target_diagnosis": [
                    "death",
                    "recurrent myocardial infarction",
                    "MACE",
                    "heart failure",
                    "stroke",
                ],
                "time_horizon": ["30 days", "1 year"],
                "setting": ["hospital", "ICU"],
            },
            "must_have_concepts": [
                {"concept": "acute myocardial infarction",
                 "role": "condition",
                 "synonyms": ["AMI", "acute coronary syndrome", "ACS"]},
                {"concept": "anemia",
                 "role": "condition",
                 "synonyms": ["low hemoglobin", "low hematocrit"]},
                {"concept": "transfusion threshold",
                 "role": "intervention",
                 "synonyms": ["restrictive transfusion strategy",
                              "liberal transfusion strategy"]},
            ],
            "ranking_concepts": [
                {"concept": "MINT trial", "role": "context"},
                {"concept": "REALITY trial", "role": "context"},
            ],
            "ideal_study_types": ["rct", "systematic_review", "meta_analysis"],
            "concept_translations": [
                {"original": "transfusion threshold",
                 "canonical": "Erythrocyte Transfusion",
                 "source": "MeSH"},
            ],
            "spelling_corrections": [],
            "interpretation_for_user": (
                "Comparing liberal vs restrictive transfusion strategies in "
                "AMI with anemia, focusing on mortality, recurrent MI, and "
                "MACE outcomes."
            ),
            "rationale": (
                "Question matches the therapy / transfusion threshold "
                "comparison subtype. Specialty-routed to cardiology."
            ),
            "confidence": 0.92,
        },
    }


def _milrinone_session_doc() -> dict:
    """Milrinone/SAH dosing-protocol session — legacy fields wrong,
    PICOTT correct."""
    return {
        "id": "mil-sess",
        "original_question": (
            "What is the dose of milrinone in subarachnoid hemorrhage with "
            "cerebral vasospasm?"
        ),
        "clinical_question_type": "other",
        "pico_json": {
            "population": "patients with SAH",
            "intervention_or_exposure": "what is the dose of milrinone",
            "comparator": "",
            "outcomes": ["vasospasm"],
            "specialty": "primary_care",
            "key_concepts": [
                "what is the dose of",
                "having acute",
                "milrinone",
            ],
        },
        "picott_json": {
            "question_scope": "foreground",
            "type_of_question": "dosing_protocol",
            "question_subtype": "intra-arterial_milrinone_dose",
            "specialty": "neurology",
            "normalized_question": (
                "what is the dose of milrinone in subarachnoid hemorrhage "
                "with cerebral vasospasm"
            ),
            "pico": {
                "population": [
                    "subarachnoid hemorrhage",
                    "aneurysmal subarachnoid hemorrhage",
                    "aSAH",
                ],
                "intervention_or_exposure_or_index_test": ["milrinone"],
                "comparator_or_reference_standard": [],
                "outcomes_or_target_diagnosis": [
                    "cerebral vasospasm",
                    "delayed cerebral ischemia",
                    "DCI",
                ],
                "time_horizon": [],
                "setting": ["neurocritical care", "ICU"],
            },
            "must_have_concepts": [
                {"concept": "milrinone", "role": "intervention"},
                {"concept": "subarachnoid hemorrhage", "role": "condition",
                 "synonyms": ["SAH", "aSAH"]},
            ],
            "ranking_concepts": [
                {"concept": "cerebral vasospasm", "role": "outcome"},
                {"concept": "delayed cerebral ischemia", "role": "outcome"},
                {"concept": "DCI", "role": "outcome"},
            ],
            "ideal_study_types": ["rct", "protocol_paper", "cohort_study"],
            "concept_translations": [],
            "spelling_corrections": [],
            "interpretation_for_user": (
                "Milrinone dosing protocols for cerebral vasospasm and "
                "delayed cerebral ischemia following aneurysmal "
                "subarachnoid hemorrhage."
            ),
            "rationale": (
                "Question matches dosing_protocol type with intra-arterial "
                "milrinone subtype. Specialty-routed to neurology."
            ),
            "confidence": 0.85,
        },
    }


# =============================================================================
# A. AMI / transfusion-threshold display mapping
# =============================================================================

class TestGroupA_AMIDisplay:
    """Tester's reported AMI bug fixed: PICOTT card MUST show therapy /
    cardiology / liberal vs restrictive, never primary_care, never
    'difference between'."""

    def test_picott_is_authoritative_over_legacy_qtype(self):
        out = build_interpreted_question_display(_ami_session_doc())
        # type_of_question MUST be the humanized PICOTT value, not "Other".
        assert out["type_of_question_raw"] == "therapy"
        assert "Therapy" in out["type_of_question"]
        assert "transfusion threshold comparison" in out["type_of_question"]

    def test_specialty_is_cardiology_not_primary_care(self):
        out = build_interpreted_question_display(_ami_session_doc())
        assert out["specialty"] == "Cardiology"
        assert "primary_care" not in out["specialty"].lower()
        assert "primary care" not in out["specialty"].lower()

    def test_intervention_block_shows_liberal_strategy(self):
        out = build_interpreted_question_display(_ami_session_doc())
        chips = out["intervention_or_exposure_or_index_test"]
        assert any("liberal" in c.lower() for c in chips)
        assert any("transfusion" in c.lower() for c in chips)

    def test_comparator_block_shows_restrictive_strategy(self):
        out = build_interpreted_question_display(_ami_session_doc())
        chips = out["comparator_or_reference_standard"]
        assert any("restrictive" in c.lower() for c in chips)

    def test_population_block_shows_ami_and_anemia(self):
        out = build_interpreted_question_display(_ami_session_doc())
        chips = out["population"]
        joined = " | ".join(chips).lower()
        assert "myocardial infarction" in joined or "ami" in joined
        assert "anemia" in joined

    def test_outcomes_include_mace_death_recurrent_mi(self):
        out = build_interpreted_question_display(_ami_session_doc())
        chips = out["outcomes_or_target_diagnosis"]
        joined = " | ".join(chips).lower()
        for needle in ("death", "mace"):
            assert needle in joined, f"missing {needle!r} in outcomes {chips!r}"

    def test_required_concepts_lists_anchors(self):
        out = build_interpreted_question_display(_ami_session_doc())
        mh = out["must_have_concepts"]
        concepts_lower = " | ".join(c["concept"] for c in mh).lower()
        for needle in ("myocardial infarction", "anemia", "transfusion threshold"):
            assert needle in concepts_lower

    def test_no_legacy_fragment_terms_leak_into_display(self):
        """NEVER display 'difference between' or 'between high' fragments
        in any of the MAIN PICOTT-card fields when PICOTT is authoritative.
        (The ``legacy_pico_json`` blob is exempted — it is intentionally
        preserved for the admin/developer 'Legacy parser fallback' toggle.)
        """
        out = build_interpreted_question_display(_ami_session_doc())
        # Build a copy that excludes the legacy debug payloads, then scan.
        import json
        main_card = {k: v for k, v in out.items()
                     if k not in ("legacy_pico_json",
                                   "legacy_clinical_question_type")}
        rendered = json.dumps(main_card).lower()
        for noise in ("difference between", "between high"):
            assert noise not in rendered, (
                f"legacy fragment {noise!r} leaked into PICOTT main card"
            )

    def test_question_scope_is_foreground(self):
        out = build_interpreted_question_display(_ami_session_doc())
        assert out["question_scope"] == "Foreground"

    def test_source_field_marked_picott(self):
        out = build_interpreted_question_display(_ami_session_doc())
        assert out["source"] == "picott"
        assert out["has_picott"] is True

    def test_legacy_pico_demoted_into_advanced_block(self):
        out = build_interpreted_question_display(_ami_session_doc())
        # Legacy fields are PRESENT (for the advanced toggle) but the
        # main card values do not include them.
        assert "legacy_pico_json" in out
        assert out["legacy_pico_json"].get("specialty") == "primary_care"
        # The main specialty value is NOT primary_care.
        assert out["specialty"] != "Primary Care"

    def test_ideal_study_types_populated(self):
        out = build_interpreted_question_display(_ami_session_doc())
        assert "rct" in out["ideal_study_types"]
        assert "systematic_review" in out["ideal_study_types"]


# =============================================================================
# B. Milrinone / SAH dosing display mapping
# =============================================================================

class TestGroupB_MilrinoneDisplay:

    def test_type_is_pharmacology_or_dosing_not_other(self):
        out = build_interpreted_question_display(_milrinone_session_doc())
        assert out["type_of_question_raw"] in ("pharmacology", "dosing_protocol")
        # Humanized label includes "Dosing" or "Pharmacology"
        label = out["type_of_question"].lower()
        assert "dosing" in label or "pharmacology" in label

    def test_specialty_is_neurology(self):
        out = build_interpreted_question_display(_milrinone_session_doc())
        assert out["specialty"] == "Neurology"

    def test_intervention_is_milrinone(self):
        out = build_interpreted_question_display(_milrinone_session_doc())
        chips = out["intervention_or_exposure_or_index_test"]
        assert any("milrinone" in c.lower() for c in chips)

    def test_population_shows_sah_aSAH(self):
        out = build_interpreted_question_display(_milrinone_session_doc())
        joined = " | ".join(out["population"]).lower()
        assert "subarachnoid hemorrhage" in joined

    def test_ranking_concepts_include_vasospasm_DCI(self):
        out = build_interpreted_question_display(_milrinone_session_doc())
        ranking = " | ".join(c["concept"] for c in out["ranking_concepts"]).lower()
        assert "vasospasm" in ranking
        assert "delayed cerebral ischemia" in ranking or "dci" in ranking

    def test_comparator_is_not_specified_explicit_string(self):
        """When PICOTT comparator block is empty, the literal string
        ``"Not specified"`` MUST appear — never an empty list or dash."""
        out = build_interpreted_question_display(_milrinone_session_doc())
        assert out["comparator_or_reference_standard"] == [NOT_SPECIFIED]

    def test_time_horizon_not_specified_when_empty(self):
        out = build_interpreted_question_display(_milrinone_session_doc())
        assert out["time_horizon"] == [NOT_SPECIFIED]

    def test_no_legacy_fragment_terms_leak_into_display(self):
        out = build_interpreted_question_display(_milrinone_session_doc())
        import json
        main_card = {k: v for k, v in out.items()
                     if k not in ("legacy_pico_json",
                                   "legacy_clinical_question_type")}
        rendered = json.dumps(main_card).lower()
        for noise in ("what is the dose of", "having acute"):
            assert noise not in rendered, (
                f"legacy fragment {noise!r} leaked into main card"
            )

    def test_subtype_carries_through(self):
        out = build_interpreted_question_display(_milrinone_session_doc())
        # subtype is humanized in type_of_question label too.
        label = out["type_of_question"].lower()
        assert "milrinone" in label or "intra-arterial" in label \
            or out["question_subtype"] != NOT_SPECIFIED


# =============================================================================
# Legacy-only BC fallback
# =============================================================================

class TestLegacyFallback:
    """Sessions without PICOTT must fall through to legacy pico_json
    display gracefully — no crash, no missing fields."""

    def test_legacy_only_session_falls_through_cleanly(self):
        legacy_only = {
            "id": "old-sess",
            "original_question": "What is GERD?",
            "clinical_question_type": "background",
            "pico_json": {
                "population": "adults",
                "intervention_or_exposure": "PPI",
                "comparator": "",
                "outcomes": ["symptom relief"],
                "specialty": "primary_care",
            },
            "picott_json": {},  # empty — legacy is sole source
        }
        out = build_interpreted_question_display(legacy_only)
        assert out["source"] == "legacy_pico_json"
        assert out["has_picott"] is False
        # All fields must be present, with "Not specified" for absent ones.
        assert out["specialty"] == NOT_SPECIFIED
        assert out["question_scope"] == NOT_SPECIFIED
        # PICOTT-specific lists are empty (no Phase 5A data exists).
        assert out["must_have_concepts"] == []
        assert out["ranking_concepts"] == []
        # Legacy intervention surfaces.
        chips = out["intervention_or_exposure_or_index_test"]
        assert any("ppi" in c.lower() for c in chips)

    def test_session_with_no_picott_no_legacy_still_returns_shape(self):
        """Defensive: a session with NO picott AND empty pico_json
        still returns the full schema with 'Not specified' everywhere."""
        empty = {"id": "empty", "picott_json": {}, "pico_json": {}}
        out = build_interpreted_question_display(empty)
        # Every chip field is at least [NOT_SPECIFIED].
        for k in (
            "population", "intervention_or_exposure_or_index_test",
            "comparator_or_reference_standard", "outcomes_or_target_diagnosis",
            "time_horizon", "setting", "ideal_study_types",
        ):
            assert out[k] == [NOT_SPECIFIED], f"{k} should be [Not specified] when empty"
        for k in ("question_scope", "specialty", "question_subtype",
                   "translated_question", "interpretation_for_user", "rationale"):
            assert out[k] == NOT_SPECIFIED

    def test_picott_missing_must_haves_falls_through_to_legacy(self):
        """If picott_json is non-empty but has NO must_have_concepts AND
        no pico blocks, ``has_picott`` returns False and legacy is used."""
        empty_picott = {
            "id": "weak",
            "picott_json": {
                "type_of_question": "other",
                "must_have_concepts": [],
                "pico": {},
            },
            "pico_json": {"population": "adults", "intervention_or_exposure": "ASA"},
            "clinical_question_type": "treatment",
        }
        out = build_interpreted_question_display(empty_picott)
        assert out["source"] == "legacy_pico_json"
        # Legacy intervention rendered.
        chips = out["intervention_or_exposure_or_index_test"]
        assert any("asa" in c.lower() for c in chips)


# =============================================================================
# Schema contract — frontend can rely on these keys
# =============================================================================

class TestDisplaySchemaContract:
    """Closed-set contract: every key the frontend reads is present in
    every response (or explicitly absent + documented)."""

    _required_keys = [
        "source", "has_picott",
        "question_scope", "type_of_question", "type_of_question_raw",
        "question_subtype", "specialty",
        "population", "intervention_or_exposure_or_index_test",
        "comparator_or_reference_standard", "outcomes_or_target_diagnosis",
        "time_horizon", "setting", "ideal_study_types",
        "must_have_concepts", "should_have_concepts", "ranking_concepts",
        "concept_translations", "translated_question", "latent_concepts",
        "translation_warnings", "spelling_corrections",
        "interpretation_for_user", "rationale", "confidence",
        "legacy_clinical_question_type", "legacy_pico_json",
    ]

    def test_picott_response_has_all_keys(self):
        out = build_interpreted_question_display(_ami_session_doc())
        for k in self._required_keys:
            assert k in out, f"PICOTT response missing key: {k}"

    def test_legacy_response_has_all_keys(self):
        empty = {"id": "empty", "picott_json": {}, "pico_json": {}}
        out = build_interpreted_question_display(empty)
        for k in self._required_keys:
            assert k in out, f"legacy response missing key: {k}"



# =============================================================================
# Phase 5D.1 — P0 regression tests
#   Fix 1 — derive PICO blocks from must_have_concepts[].role when the
#           ``picott_json.pico.<field>`` arrays are empty (the parser
#           sometimes only populates must_have_concepts).
#   Fix 2 — ``rank_explanation`` strings must use the PICOTT
#           ``type_of_question`` (humanized) instead of the legacy
#           ``clinical_question_type="other"``.
# =============================================================================

def _ami_session_doc_pico_blocks_empty() -> dict:
    """AMI/transfusion session shape — same as ``_ami_session_doc`` but
    the structured ``pico`` sub-dict is INTENTIONALLY empty. The
    must_have_concepts and ranking_concepts carry the role-tagged terms,
    so the display normalizer must derive the PICO arrays from the role
    fallback (Fix 1)."""
    return {
        "id": "ami-pico-empty-sess",
        "original_question": (
            "What is the optimal red-cell transfusion threshold in adults "
            "with acute myocardial infarction and anemia?"
        ),
        "clinical_question_type": "other",
        "pico_json": {
            "population": "patients with MI",
            "specialty": "primary_care",
        },
        "picott_json": {
            "question_scope": "foreground",
            "type_of_question": "therapy",
            "question_subtype": "transfusion_threshold_comparison",
            "specialty": "cardiology",
            # CRITICAL: structured pico block left empty.
            "pico": {
                "population": [],
                "intervention_or_exposure_or_index_test": [],
                "comparator_or_reference_standard": [],
                "outcomes_or_target_diagnosis": [],
                "time_horizon": [],
                "setting": [],
            },
            # Role-tagged anchors — the only place P/I/C/O can be derived.
            "must_have_concepts": [
                {"concept": "acute myocardial infarction",
                 "role": "condition",
                 "synonyms": ["AMI", "acute coronary syndrome"]},
                {"concept": "anemia",
                 "role": "condition",
                 "synonyms": ["low hemoglobin"]},
                {"concept": "liberal transfusion strategy",
                 "role": "intervention",
                 "synonyms": ["higher hemoglobin threshold"]},
                {"concept": "restrictive transfusion strategy",
                 "role": "comparator",
                 "synonyms": ["lower hemoglobin threshold"]},
            ],
            "ranking_concepts": [
                {"concept": "all-cause mortality", "role": "outcome"},
                {"concept": "MACE", "role": "outcome"},
                {"concept": "recurrent myocardial infarction", "role": "outcome"},
            ],
            "ideal_study_types": ["rct", "systematic_review"],
            "concept_translations": [],
            "spelling_corrections": [],
            "interpretation_for_user": "Comparing transfusion strategies in AMI.",
            "rationale": "PICOTT therapy / transfusion threshold comparison.",
            "confidence": 0.90,
        },
    }


def _milrinone_session_doc_pico_blocks_empty() -> dict:
    """Milrinone/SAH session — pico sub-dict empty, role-tagged anchors
    on must_have_concepts only. (No translations, no ranking outcomes.)"""
    return {
        "id": "mil-pico-empty-sess",
        "original_question": (
            "What is the dose of milrinone in subarachnoid hemorrhage with "
            "cerebral vasospasm?"
        ),
        "clinical_question_type": "other",
        "pico_json": {"specialty": "primary_care"},
        "picott_json": {
            "question_scope": "foreground",
            "type_of_question": "dosing_protocol",
            "question_subtype": "intra-arterial_milrinone_dose",
            "specialty": "neurology",
            "pico": {
                "population": [],
                "intervention_or_exposure_or_index_test": [],
                "comparator_or_reference_standard": [],
                "outcomes_or_target_diagnosis": [],
            },
            "must_have_concepts": [
                {"concept": "subarachnoid hemorrhage",
                 "role": "condition",
                 "synonyms": ["SAH", "aSAH"]},
                {"concept": "milrinone", "role": "intervention"},
            ],
            "ranking_concepts": [],
            "ideal_study_types": ["rct", "protocol_paper"],
            "concept_translations": [],
            "interpretation_for_user": "Milrinone dosing for SAH.",
            "rationale": "PICOTT dosing protocol / neurology.",
            "confidence": 0.85,
        },
    }


class TestFix1_DeriveFromMustHaves:
    """Fix 1 — when ``picott_json.pico.<field>`` is empty but
    ``must_have_concepts`` carries role-tagged anchors, the display
    normalizer derives the P/I/C/O arrays from the role tags."""

    def test_ami_session_pico_arrays_populated_from_must_haves(self):
        """AMI session with empty pico blocks: population, intervention,
        comparator, AND outcomes must be derived from must-haves +
        ranking concepts (outcomes-only fallback). Specialty stays
        Cardiology, type_of_question stays Therapy."""
        out = build_interpreted_question_display(
            _ami_session_doc_pico_blocks_empty()
        )

        # has_picott still True because must_have_concepts has content.
        assert out["has_picott"] is True
        assert out["source"] == "picott"

        # Population — derived from role="condition" must-haves.
        pop_lower = " | ".join(out["population"]).lower()
        assert "myocardial infarction" in pop_lower, out["population"]
        assert "anemia" in pop_lower, out["population"]
        assert out["population"] != [NOT_SPECIFIED]

        # Intervention — derived from role="intervention" must-haves.
        iv_lower = " | ".join(out["intervention_or_exposure_or_index_test"]).lower()
        assert "liberal" in iv_lower, out["intervention_or_exposure_or_index_test"]
        assert out["intervention_or_exposure_or_index_test"] != [NOT_SPECIFIED]

        # Comparator — derived from role="comparator" must-haves.
        cmp_lower = " | ".join(out["comparator_or_reference_standard"]).lower()
        assert "restrictive" in cmp_lower, out["comparator_or_reference_standard"]
        assert out["comparator_or_reference_standard"] != [NOT_SPECIFIED]

        # Outcomes — derived from role="outcome" ranking_concepts (the
        # outcomes-only fallback path), since outcomes don't appear on
        # must-haves and pico.outcomes_or_target_diagnosis is empty.
        out_lower = " | ".join(out["outcomes_or_target_diagnosis"]).lower()
        assert ("mortality" in out_lower or "mace" in out_lower
                or "myocardial infarction" in out_lower), \
            out["outcomes_or_target_diagnosis"]
        assert out["outcomes_or_target_diagnosis"] != [NOT_SPECIFIED]

        # PICOTT type / specialty unchanged.
        assert out["type_of_question_raw"] == "therapy"
        assert out["specialty"] == "Cardiology"

    def test_milrinone_session_population_populated_from_must_haves(self):
        """Milrinone/SAH session with empty pico blocks: population is
        populated from role="condition" must-haves; intervention from
        role="intervention" must-haves; comparator and outcomes fall to
        ``"Not specified"`` since no role-tagged source carries them."""
        out = build_interpreted_question_display(
            _milrinone_session_doc_pico_blocks_empty()
        )

        # Population — must include subarachnoid hemorrhage.
        pop_lower = " | ".join(out["population"]).lower()
        assert "subarachnoid hemorrhage" in pop_lower, out["population"]
        assert out["population"] != [NOT_SPECIFIED]

        # Intervention — must include milrinone.
        iv_lower = " | ".join(out["intervention_or_exposure_or_index_test"]).lower()
        assert "milrinone" in iv_lower, out["intervention_or_exposure_or_index_test"]

        # Comparator and outcomes — no role-tagged source → "Not specified".
        assert out["comparator_or_reference_standard"] == [NOT_SPECIFIED]
        assert out["outcomes_or_target_diagnosis"] == [NOT_SPECIFIED]

        # PICOTT type / specialty unchanged.
        assert out["specialty"] == "Neurology"
        assert out["type_of_question_raw"] in ("dosing_protocol", "pharmacology")


# =============================================================================
# Fix 2 — rank_explanation derived from PICOTT type_of_question
# =============================================================================

class TestFix2_RankExplanationFromPicott:
    """Fix 2 — ``rank_explanation`` strings must read 'Question type:
    Therapy / …' when PICOTT carries a non-other type, NEVER 'Question
    type: other.' even when legacy ``clinical_question_type='other'``."""

    def _ranker_inputs(self, picott: dict | None, legacy_qtype: str = "other"):
        """Minimal inputs to invoke ``EvidenceRankerService.rank``."""
        from services.evidence_ranker import EvidenceRankerService  # noqa: WPS433

        parsed = {
            "clinical_question_type": legacy_qtype,
            "key_concepts": ["empagliflozin", "hfpef"],
            "mesh_terms": [],
            "synonyms": [],
            "drug_names_and_classes": ["empagliflozin"],
            "intervention_or_exposure": "empagliflozin",
            "comparator": "placebo",
            "picott_json": picott or {},
        }
        article = {
            "canonical_key": "doi:10.1056/nejmoa2107038",
            "title": "Empagliflozin in HFpEF",
            "abstract": "RCT of empagliflozin vs placebo in HFpEF.",
            "publication_year": 2021,
            "article_type_classification": "rct",
            "is_guideline": False,
            "full_text_status": "abstract_only",
            "publication_types_json": ["Randomized Controlled Trial"],
        }
        qtypes = {article["canonical_key"]: ["high_recall", "evidence_filtered"]}
        return EvidenceRankerService(), parsed, [article], qtypes

    def test_rank_explanation_uses_picott_type_not_legacy_other(self):
        """PICOTT type_of_question='therapy' + legacy
        clinical_question_type='other' → header reads 'Question type:
        Therapy …', NEVER 'Question type: other.'."""
        ranker, parsed, articles, qtypes = self._ranker_inputs(
            picott={
                "type_of_question": "therapy",
                "question_subtype": "transfusion_threshold_comparison",
            },
            legacy_qtype="other",
        )
        ranked = ranker.rank(parsed, articles, qtypes)
        assert ranked, "ranker returned no rows"
        explanation = ranked[0]["rank_explanation"]
        assert explanation.startswith("Question type: Therapy"), explanation
        # Subtype rendered.
        assert "transfusion threshold comparison" in explanation, explanation
        # NEVER the legacy "other" header.
        assert not explanation.startswith("Question type: other"), explanation

    def test_rank_explanation_falls_back_to_legacy_when_picott_absent(self):
        """No PICOTT dict (pre-Phase-5A session) + legacy
        clinical_question_type='treatment' → header reads 'Question
        type: Treatment.'"""
        # Empty PICOTT, legacy treatment.
        ranker, parsed, articles, qtypes = self._ranker_inputs(
            picott={},
            legacy_qtype="treatment",
        )
        ranked = ranker.rank(parsed, articles, qtypes)
        assert ranked
        explanation = ranked[0]["rank_explanation"]
        assert explanation.startswith("Question type: Treatment"), explanation

        # Also: PICOTT carries type_of_question="other" + no subtype +
        # legacy is also "other" → historical "other" string preserved
        # (matches the docstring of ``_humanize_qtype_from_picott``).
        ranker2, parsed2, articles2, qtypes2 = self._ranker_inputs(
            picott={"type_of_question": "other", "question_subtype": ""},
            legacy_qtype="other",
        )
        ranked2 = ranker2.rank(parsed2, articles2, qtypes2)
        assert ranked2[0]["rank_explanation"].startswith("Question type: other"), \
            ranked2[0]["rank_explanation"]
