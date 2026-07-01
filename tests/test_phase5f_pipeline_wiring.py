"""Phase 5F integration tests — Clinical Context Engine wiring.

Covers the four refinements from the green-light:
- pipeline cascade order (shorthand → ontology → role-repair → role-weighting)
- safety_filter: editorials/letters/comments kept (with flag),
  patents/grants/errata/retraction-notices excluded
- multi-intent detection on the varicella query
- known ARDS-corticosteroid review PMID survives the new safety_filter

These tests use mocked retrieval — they validate the orchestration logic
without hitting PubMed / Europe PMC. The live retrieval validation lives
in the deliverable report (sandbox-limited; captured queries + repaired
PICOTT for each of the three target questions).
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# Group D1 — safety_filter editorial/letter/comment kept; non-article
#            records still excluded.
# =============================================================================

class TestSafetyFilter_EditorialKept:

    def _make(self, pub_type: str, **extra) -> dict:
        return {
            "canonical_key": f"doi:10.test/{pub_type.lower().replace(' ', '_')}",
            "title": f"Test {pub_type}",
            "abstract": "An ARDS corticosteroid review.",
            "publication_types_json": [pub_type],
            "is_preprint": False,
            "is_retracted": False,
            "is_retraction_notice": False,
            **extra,
        }

    def test_editorial_kept_with_flag(self):
        from services.safety_filter import safety_filter
        kept, excluded = safety_filter.pre_rank_filter([self._make("Editorial")])
        assert len(kept) == 1 and not excluded
        assert "editorial_or_commentary" in (kept[0].get("flags") or [])

    def test_letter_kept_with_flag(self):
        from services.safety_filter import safety_filter
        kept, _ = safety_filter.pre_rank_filter([self._make("Letter")])
        assert "editorial_or_commentary" in (kept[0].get("flags") or [])

    def test_comment_kept_with_flag(self):
        from services.safety_filter import safety_filter
        kept, _ = safety_filter.pre_rank_filter([self._make("Comment")])
        assert "editorial_or_commentary" in (kept[0].get("flags") or [])

    def test_patent_still_excluded(self):
        from services.safety_filter import safety_filter
        kept, excluded = safety_filter.pre_rank_filter([self._make("Patent")])
        assert not kept and len(excluded) == 1
        assert excluded[0]["exclusion_reason"] == "non_article_record"

    def test_grant_still_excluded(self):
        from services.safety_filter import safety_filter
        _, excluded = safety_filter.pre_rank_filter([self._make("Grant")])
        assert excluded[0]["exclusion_reason"] == "non_article_record"

    def test_published_erratum_excluded(self):
        from services.safety_filter import safety_filter
        _, excluded = safety_filter.pre_rank_filter([self._make("Published Erratum")])
        assert excluded[0]["exclusion_reason"] == "non_article_record"

    def test_retraction_notice_pubtype_excluded(self):
        from services.safety_filter import safety_filter
        _, excluded = safety_filter.pre_rank_filter([self._make("Retraction-Notice")])
        assert excluded[0]["exclusion_reason"] == "non_article_record"

    def test_dataset_excluded(self):
        from services.safety_filter import safety_filter
        _, excluded = safety_filter.pre_rank_filter([self._make("Dataset")])
        assert excluded[0]["exclusion_reason"] == "non_article_record"

    def test_corrected_and_republished_article_excluded(self):
        from services.safety_filter import safety_filter
        _, excluded = safety_filter.pre_rank_filter([
            self._make("Corrected and Republished Article"),
        ])
        assert excluded[0]["exclusion_reason"] == "non_article_record"

    def test_known_ards_corticosteroid_review_pmid_survives(self):
        """Regression: a real-world ARDS-corticosteroid review (PMID
        32043986 — Villar et al., Lancet Respir Med 2020, "Dexamethasone
        treatment for the acute respiratory distress syndrome: a
        multicentre, randomised controlled trial") with a "Journal
        Article" + "Randomized Controlled Trial" pubtype mix MUST
        survive the safety_filter. A clinical review pubtype must NOT
        be excluded by the closed-set non_article_record filter."""
        from services.safety_filter import safety_filter
        rct = {
            "canonical_key": "doi:10.1016/S2213-2600(20)30045-3",
            "pmid": "32043986",
            "title": "Dexamethasone treatment for the acute respiratory distress syndrome",
            "abstract": "Background: ARDS has high mortality...",
            "publication_types_json": ["Journal Article", "Randomized Controlled Trial",
                                         "Multicenter Study"],
        }
        review = {
            "canonical_key": "doi:10.test/ards-steroid-review",
            "pmid": "30872119",  # corticosteroids in ARDS review
            "title": "Corticosteroids for adults with ARDS: a review",
            "abstract": "Systemic corticosteroids in ARDS — narrative review.",
            "publication_types_json": ["Review", "Journal Article"],
        }
        commentary = {
            "canonical_key": "doi:10.test/ards-steroid-commentary",
            "pmid": "33000000",
            "title": "Steroids in ARDS — Point/Counterpoint",
            "abstract": "Editorial discussion of steroid use in ARDS.",
            "publication_types_json": ["Editorial", "Journal Article"],
        }
        kept, excluded = safety_filter.pre_rank_filter([rct, review, commentary])
        # Zero exclusions: the closed-set filter shouldn't fire for any
        # of these substantive clinical records.
        assert not excluded, [a["exclusion_reason"] for a in excluded]
        assert len(kept) == 3
        # Commentary carries the editorial_or_commentary flag.
        comm = next(k for k in kept if "Editorial" in k["publication_types_json"])
        assert "editorial_or_commentary" in (comm.get("flags") or [])


# =============================================================================
# Group D2 — Clinical Context Engine orchestration
# =============================================================================

class TestContextEngine_GLP1Cascade:

    def _parsed_glp1(self) -> dict:
        return {
            "normalized_question": "how do GLP-1 receptor agonists impact LDL-C cholesterol?",
            "shorthand_substitutions": [
                {"original": "glp1s", "normalized": "GLP-1 receptor agonists",
                 "confidence": 1.0, "reason": "drug_class_shorthand"},
                {"original": "ldl", "normalized": "LDL-C / low-density lipoprotein cholesterol",
                 "confidence": 1.0, "reason": "lab_shorthand"},
                {"original": "cholesterole", "normalized": "cholesterol",
                 "confidence": 1.0, "reason": "clinical_spelling_correction"},
            ],
            "picott_json": {
                "normalized_question": "how do GLP-1 receptor agonists impact LDL-C?",
                "type_of_question": "pharmacology",
                "question_subtype": "metabolic_biomarker_effect",
                "specialty": "endocrinology",
                "pico": {
                    "population": [],
                    "intervention_or_exposure_or_index_test": [],
                    "comparator_or_reference_standard": [],
                    "outcomes_or_target_diagnosis": [],
                },
                "must_have_concepts": [
                    {"concept": "GLP-1 receptor agonists", "role": "context"},
                    {"concept": "LDL-C", "role": "context"},
                ],
                "ranking_concepts": [],
            },
        }

    def test_context_engine_produces_full_output(self):
        from services.clinical_context_engine import build_clinical_context
        ctx = build_clinical_context(self._parsed_glp1())
        # Closed-set keys present.
        for k in ("shorthand_substitutions", "ontology_mappings",
                   "role_repairs", "role_weights", "multi_intent",
                   "repaired_picott"):
            assert k in ctx

    def test_role_repair_fires_for_glp1_intervention(self):
        from services.clinical_context_engine import build_clinical_context
        ctx = build_clinical_context(self._parsed_glp1())
        repaired = ctx["repaired_picott"]
        glp1 = next(m for m in repaired["must_have_concepts"]
                    if m["concept"] == "GLP-1 receptor agonists")
        assert glp1["role"] == "intervention", glp1
        ldl = next(m for m in repaired["must_have_concepts"]
                   if m["concept"] == "LDL-C")
        assert ldl["role"] == "outcome", ldl

    def test_role_weights_pharmacology_intervention_40(self):
        from services.clinical_context_engine import build_clinical_context
        ctx = build_clinical_context(self._parsed_glp1())
        weights = ctx["role_weights"]["weights"]
        # pharmacology → intervention 40, outcome 25.
        assert weights["intervention"] == 40

    def test_ontology_synonyms_enriched_on_glp1_intervention(self):
        from services.clinical_context_engine import build_clinical_context
        ctx = build_clinical_context(self._parsed_glp1())
        repaired = ctx["repaired_picott"]
        glp1 = next(m for m in repaired["must_have_concepts"]
                    if m["concept"] == "GLP-1 receptor agonists")
        syn_lower = " | ".join(s.lower() for s in (glp1.get("synonyms") or []))
        # At least one canonical synonym surfaces (semaglutide / liraglutide
        # are members of the GLP-1 RA class).
        assert "semaglutide" in syn_lower or "liraglutide" in syn_lower
        assert glp1.get("ontology_enrichment_applied") is True

    def test_no_multi_intent_for_glp1_single_question(self):
        from services.clinical_context_engine import build_clinical_context
        ctx = build_clinical_context(self._parsed_glp1())
        assert ctx["multi_intent"]["is_multi_intent"] is False


class TestContextEngine_VaricellaMultiIntent:

    def _parsed_varicella(self) -> dict:
        return {
            "normalized_question": (
                "What is the recommended age to get varicella vaccination? "
                "What is its benefit?"
            ),
            "shorthand_substitutions": [],
            "picott_json": {
                "normalized_question": (
                    "What is the recommended age to get varicella vaccination? "
                    "What is its benefit?"
                ),
                "type_of_question": "prevention",
                "question_subtype": None,
                "specialty": "preventive_medicine",
                "must_have_concepts": [
                    {"concept": "varicella vaccine", "role": "intervention"},
                ],
                "ranking_concepts": [],
            },
        }

    def test_multi_intent_fires_on_varicella_question(self):
        from services.clinical_context_engine import build_clinical_context
        ctx = build_clinical_context(self._parsed_varicella())
        mi = ctx["multi_intent"]
        assert mi["is_multi_intent"] is True, mi
        assert len(mi["sub_questions"]) >= 2
        subtypes = {s["subtype"] for s in mi["sub_questions"]}
        # Schedule + benefit subtypes both present.
        assert "immunization_schedule" in subtypes
        assert "vaccine_effectiveness_benefit" in subtypes

    def test_each_sub_question_has_its_own_role_weights(self):
        from services.clinical_context_engine import build_clinical_context
        ctx = build_clinical_context(self._parsed_varicella())
        for spec in ctx["multi_intent"]["sub_questions"]:
            assert isinstance(spec.get("role_weights"), dict)
            assert spec["role_weights"]  # not empty


class TestContextEngine_ARDSSteroid:

    def _parsed_ards(self) -> dict:
        return {
            "normalized_question": (
                "What is the right timing and dose to administer steroids "
                "for patients with ARDS?"
            ),
            "shorthand_substitutions": [
                {"original": "ards", "normalized": "acute respiratory distress syndrome",
                 "confidence": 1.0, "reason": "procedure_acronym"},
            ],
            "picott_json": {
                "normalized_question": (
                    "What is the right timing and dose to administer "
                    "corticosteroids for patients with acute respiratory "
                    "distress syndrome?"
                ),
                "type_of_question": "dosing_protocol",
                "question_subtype": "timing_dose_protocol",
                "specialty": "critical_care",
                "must_have_concepts": [
                    {"concept": "acute respiratory distress syndrome", "role": "context"},
                    {"concept": "corticosteroids", "role": "context"},
                ],
                "ranking_concepts": [],
            },
        }

    def test_ards_steroid_role_repair_fires(self):
        from services.clinical_context_engine import build_clinical_context
        ctx = build_clinical_context(self._parsed_ards())
        repaired = ctx["repaired_picott"]
        steroid = next(m for m in repaired["must_have_concepts"]
                        if m["concept"] == "corticosteroids")
        assert steroid["role"] == "intervention", steroid
        ards = next(m for m in repaired["must_have_concepts"]
                     if m["concept"] == "acute respiratory distress syndrome")
        assert ards["role"] == "population", ards

    def test_dosing_protocol_role_weights(self):
        from services.clinical_context_engine import build_clinical_context
        ctx = build_clinical_context(self._parsed_ards())
        weights = ctx["role_weights"]["weights"]
        # dosing_protocol → intervention 35, dose_terms 15.
        assert weights.get("intervention") == 35
        assert weights.get("dose_terms") == 15
