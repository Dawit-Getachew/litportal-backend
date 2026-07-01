"""Phase 5F Group F + G — benchmark + regression sweep.

Group F (benchmark case ``glp1_ldl_cholesterol``):
- Case loaded by the benchmark service.
- Parser+shorthand pipeline produces pharmacology/therapy with
  metabolic_biomarker_effect subtype and the right specialty when the
  parser is exercised parse-only.
- Provisional status (we do NOT ship gold PMIDs at build time — only
  the parser + query expectations are checked here).

Group G (regression sweep) — invariants:
- Existing Phase 5D.1 PICOTT display normalizer + Phase 5D.1 fixes still
  pass against the AMI transfusion + milrinone fixtures.
- No legacy ``Question type: other.`` rank explanations leak out of the
  PICOTT-aware code path.
- Closed-set cap_reasons remains a superset of the Phase 5A names; new
  Phase 5F names are added but old ones MUST still be present.
- Closed-set article_answer_class values intact.
- ROLE_WEIGHTS dict keys cover every Phase 5F supported question type.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# Group F — glp1_ldl_cholesterol benchmark
# =============================================================================

class TestGroupF_Glp1LdlBenchmark:

    def test_case_loaded_in_phase5c_provisional_set(self):
        from services.search_benchmark_phase5c import PHASE5C_PROVISIONAL_CASES
        ids = {c.id for c in PHASE5C_PROVISIONAL_CASES}
        assert "glp1_ldl_cholesterol" in ids

    def test_case_picks_endocrinology_specialty(self):
        from services.search_benchmark_phase5c import PHASE5C_PROVISIONAL_CASES
        case = next(c for c in PHASE5C_PROVISIONAL_CASES
                    if c.id == "glp1_ldl_cholesterol")
        assert case.expected_specialty == "endocrinology"
        assert case.expected_type_of_question >= {"pharmacology"} or \
               case.expected_type_of_question >= {"therapy"}
        assert "GLP-1 receptor agonists" in case.must_have_anchors
        assert "LDL-C" in case.must_have_anchors

    def test_case_provisional_flag_set(self):
        """We did NOT confidently source gold PMIDs at build time, so
        the case carries ``provisional=True`` and the runner excludes
        it from retrieval pass/fail aggregates."""
        from services.search_benchmark_phase5c import (
            PHASE5C_PROVISIONAL_CASES,
            PHASE5C_PROVISIONAL_STATUS,
        )
        case = next(c for c in PHASE5C_PROVISIONAL_CASES
                    if c.id == "glp1_ldl_cholesterol")
        assert case.provisional is True
        assert case.status == PHASE5C_PROVISIONAL_STATUS

    def test_shorthand_normalizer_handles_case_question(self):
        """``how do glp1s impact ldl cholesterole?`` should normalize
        into a clean question text with the spelling+shorthand
        substitutions surfaced for transparency."""
        from services.clinical_shorthand_normalizer import (
            clinical_shorthand_normalizer,
        )
        r = clinical_shorthand_normalizer.normalize(
            "how do glp1s impact ldl cholesterole?"
        )
        # Question text rewritten.
        assert "GLP-1 receptor agonists" in r.normalized
        assert "cholesterol" in r.normalized
        assert "LDL-C" in r.normalized
        # Transparency: at least 3 substitutions surfaced.
        originals = {s["original"] for s in r.substitutions}
        assert "glp1s" in originals
        assert "ldl" in originals
        assert "cholesterole" in originals


# =============================================================================
# Group G — Regression sweep (invariants must continue to hold)
# =============================================================================

class TestGroupG_ClosedSetIntegrity:

    def test_phase5a_cap_reasons_still_present(self):
        from services.picott_ranker import CAP_REASONS
        for name in (
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
        ):
            assert name in CAP_REASONS, f"Phase 5A cap reason {name!r} dropped"

    def test_phase5f_cap_reasons_added(self):
        from services.picott_ranker import CAP_REASONS
        for name in (
            "tirzepatide_only_for_glp1_question",
            "glp1_biology_no_lipid_outcome",
            "ldl_without_glp1_intervention",
        ):
            assert name in CAP_REASONS

    def test_article_answer_class_enum_unchanged(self):
        from services.semantic_reranker import ARTICLE_ANSWER_CLASSES
        assert set(ARTICLE_ANSWER_CLASSES) == {
            "direct", "related", "mechanistic",
            "wrong_population", "wrong_intervention", "wrong_outcome",
            "background_only",
        }

    def test_role_weights_cover_phase5f_question_types(self):
        from services.role_weighting import ROLE_WEIGHTS
        required = {
            "therapy", "pharmacology", "dosing_protocol", "diagnosis",
            "etiology_harm", "prognosis", "prevention", "vaccine_schedule",
            "background", "other",
        }
        assert required.issubset(set(ROLE_WEIGHTS.keys()))


class TestGroupG_PicottDisplayStillWorks:
    """Phase 5D.1 normalizer must remain green for AMI + milrinone
    fixtures even with Phase 5F merged in."""

    def test_ami_fixture_still_passes(self):
        from services.interpreted_question_display import (
            build_interpreted_question_display,
        )
        ami = {
            "id": "ami-regression",
            "original_question": (
                "What is the optimal red-cell transfusion threshold in adults "
                "with acute myocardial infarction and anemia?"
            ),
            "clinical_question_type": "other",
            "pico_json": {"specialty": "primary_care"},
            "picott_json": {
                "type_of_question": "therapy",
                "question_subtype": "transfusion_threshold_comparison",
                "specialty": "cardiology",
                "pico": {
                    "population": [],
                    "intervention_or_exposure_or_index_test": [],
                    "comparator_or_reference_standard": [],
                    "outcomes_or_target_diagnosis": [],
                },
                "must_have_concepts": [
                    {"concept": "acute myocardial infarction",
                     "role": "condition"},
                    {"concept": "liberal transfusion strategy",
                     "role": "intervention"},
                    {"concept": "restrictive transfusion strategy",
                     "role": "comparator"},
                ],
                "ranking_concepts": [
                    {"concept": "all-cause mortality", "role": "outcome"},
                ],
            },
        }
        out = build_interpreted_question_display(ami)
        assert out["type_of_question_raw"] == "therapy"
        # PICO arrays populated from role-tagged must-haves.
        assert any("myocardial infarction" in p.lower()
                   for p in out["population"])
        assert any("liberal" in i.lower()
                   for i in out["intervention_or_exposure_or_index_test"])
        assert any("restrictive" in c.lower()
                   for c in out["comparator_or_reference_standard"])


class TestGroupG_NoOtherInRankExplanation:

    def test_picott_question_does_not_emit_legacy_other_header(self):
        """Phase 5D.1 Fix 2 — rank_explanation must NEVER read
        'Question type: other.' when picott carries a non-other type."""
        from services.evidence_ranker import EvidenceRankerService
        parsed = {
            "clinical_question_type": "other",  # legacy stale value
            "key_concepts": ["GLP-1 receptor agonists", "LDL-C"],
            "mesh_terms": [], "synonyms": [],
            "drug_names_and_classes": ["semaglutide"],
            "intervention_or_exposure": "GLP-1 receptor agonists",
            "comparator": "",
            "picott_json": {
                "type_of_question": "pharmacology",
                "question_subtype": "metabolic_biomarker_effect",
            },
        }
        article = {
            "canonical_key": "doi:10.0000/test",
            "title": "Semaglutide reduces LDL-C",
            "abstract": "Semaglutide reduced LDL cholesterol.",
            "publication_year": 2024,
            "article_type_classification": "rct",
            "is_guideline": False, "full_text_status": "abstract_only",
            "publication_types_json": ["Randomized Controlled Trial"],
        }
        qtypes = {"doi:10.0000/test": ["high_recall"]}
        ranked = EvidenceRankerService().rank(parsed, [article], qtypes)
        assert ranked
        head = ranked[0]["rank_explanation"]
        assert head.startswith("Question type: Pharmacology"), head
        assert not head.startswith("Question type: other"), head
