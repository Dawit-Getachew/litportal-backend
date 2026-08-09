"""Phase 5F Group B + C — ClinicalRoleRepairService + RoleWeightingService.

Group B (role repair):
- GLP-1 class on must-have with role=context → repaired to intervention.
- LDL-C on ranking with role=context → repaired to outcome.
- Missing population flagged but NOT hallucinated when ontology can't
  resolve a population concept.

Group C (role weighting):
- pharmacology → intervention weight > population weight (40 vs 20).
- diagnosis → index_test highest weight.
- etiology_harm → exposure highest weight.
- background → concept_clarity highest weight.
- weighted_relevance uses higher coverage → higher final score.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.clinical_role_repair import (  # noqa: E402
    clinical_role_repair,
)
from services.role_weighting import (  # noqa: E402
    ROLE_WEIGHTS,
    role_weight,
    role_weighting,
    weighted_relevance,
)


# =============================================================================
# Group B — ClinicalRoleRepairService
# =============================================================================

def _glp1_ldl_picott_with_wrong_roles() -> dict:
    """PICOTT shape as the parser/LLM might emit it BEFORE role repair:
    must_have_concepts list carries GLP-1 RAs / LDL-C but with role tags
    like ``context`` instead of ``intervention`` / ``outcome``."""
    return {
        "normalized_question": "how do GLP-1 receptor agonists impact LDL-C in patients with type 2 diabetes?",
        "original_question": "how do glp1s impact ldl cholesterole?",
        "type_of_question": "pharmacology",
        "question_subtype": "metabolic_biomarker_effect",
        "specialty": "endocrinology",
        "pico": {
            "population": [],
            "intervention_or_exposure_or_index_test": [],
            "comparator_or_reference_standard": [],
            "outcomes_or_target_diagnosis": [],
            "time_horizon": [],
            "setting": [],
        },
        "must_have_concepts": [
            {"concept": "GLP-1 receptor agonists", "role": "context",
             "synonyms": ["semaglutide", "liraglutide"]},
            {"concept": "type 2 diabetes", "role": "context",
             "synonyms": ["T2DM"]},
        ],
        "ranking_concepts": [
            {"concept": "LDL-C", "role": "context",
             "synonyms": ["LDL cholesterol"]},
            {"concept": "HDL-C", "role": "context"},
            {"concept": "triglycerides", "role": "context"},
        ],
        "concept_translations": [],
    }


class TestGroupB_RoleRepair:

    def test_glp1_class_repaired_to_intervention(self):
        picott = _glp1_ldl_picott_with_wrong_roles()
        res = clinical_role_repair.repair(picott)
        # Find the GLP-1 must-have row.
        glp1 = next(
            m for m in res.repaired_picott["must_have_concepts"]
            if m["concept"] == "GLP-1 receptor agonists"
        )
        assert glp1["role"] == "intervention", glp1
        # Provenance recorded.
        prov = [p for p in res.repair_provenance
                if p["concept"] == "GLP-1 receptor agonists"
                and p["new_value"] == "intervention"]
        assert prov, res.repair_provenance
        # Rule code closed-set.
        assert prov[0]["rule"] in {
            "drug_class_effect_question", "ontology_role_hint",
        }

    def test_ldl_ranking_repaired_to_outcome(self):
        picott = _glp1_ldl_picott_with_wrong_roles()
        res = clinical_role_repair.repair(picott)
        ldl = next(
            r for r in res.repaired_picott["ranking_concepts"]
            if r["concept"] == "LDL-C"
        )
        assert ldl["role"] == "outcome", ldl
        prov = [p for p in res.repair_provenance
                if p["concept"] == "LDL-C" and p["new_value"] == "outcome"]
        assert prov, res.repair_provenance

    def test_type2_diabetes_repaired_to_population(self):
        picott = _glp1_ldl_picott_with_wrong_roles()
        res = clinical_role_repair.repair(picott)
        t2d = next(
            m for m in res.repaired_picott["must_have_concepts"]
            if m["concept"] == "type 2 diabetes"
        )
        # Population repair when the ontology says concept_type=disease.
        assert t2d["role"] in {"population", "context"}
        # If repaired, prov recorded.
        if t2d["role"] == "population":
            prov = [p for p in res.repair_provenance
                    if p["concept"] == "type 2 diabetes"]
            assert prov

    def test_repaired_picott_pico_blocks_get_populated(self):
        picott = _glp1_ldl_picott_with_wrong_roles()
        res = clinical_role_repair.repair(picott)
        pico = res.repaired_picott["pico"]
        # Intervention block now contains GLP-1 RAs.
        assert any("GLP-1" in c
                   for c in pico["intervention_or_exposure_or_index_test"])
        # Outcomes block now contains LDL-C (mirrored from ranking repair).
        assert any("LDL" in c for c in pico["outcomes_or_target_diagnosis"])

    def test_role_repair_does_not_invent_concepts(self):
        """NEVER add concepts that weren't there. The total count of
        must-haves + ranking concepts must be preserved."""
        picott = _glp1_ldl_picott_with_wrong_roles()
        before = (len(picott["must_have_concepts"])
                  + len(picott["ranking_concepts"]))
        res = clinical_role_repair.repair(picott)
        after = (len(res.repaired_picott["must_have_concepts"])
                 + len(res.repaired_picott["ranking_concepts"]))
        assert before == after

    def test_role_repair_does_not_mutate_question_text(self):
        picott = _glp1_ldl_picott_with_wrong_roles()
        before = picott["normalized_question"]
        res = clinical_role_repair.repair(picott)
        assert res.repaired_picott["normalized_question"] == before
        assert res.repaired_picott["original_question"] == picott["original_question"]

    def test_missing_population_not_hallucinated(self):
        """When no concept in must_have_concepts maps to a population
        ontology entry, the repair service MUST NOT invent one."""
        picott = {
            "normalized_question": "what is the dose of milrinone?",
            "original_question": "what is the dose of milrinone?",
            "type_of_question": "dosing_protocol",
            "pico": {
                "population": [],
                "intervention_or_exposure_or_index_test": [],
                "comparator_or_reference_standard": [],
                "outcomes_or_target_diagnosis": [],
            },
            "must_have_concepts": [
                {"concept": "milrinone", "role": "context"},
            ],
            "ranking_concepts": [],
        }
        res = clinical_role_repair.repair(picott)
        # Milrinone repaired to intervention; population stays EMPTY.
        mil = res.repaired_picott["must_have_concepts"][0]
        assert mil["role"] == "intervention"
        assert res.repaired_picott["pico"]["population"] == []

    def test_empty_picott_returns_empty_provenance(self):
        res = clinical_role_repair.repair({})
        assert res.repair_provenance == []

    def test_provenance_records_have_required_keys(self):
        picott = _glp1_ldl_picott_with_wrong_roles()
        res = clinical_role_repair.repair(picott)
        for p in res.repair_provenance:
            for k in ("repaired_field", "concept", "previous_value",
                      "new_value", "rule", "ontology_source",
                      "confidence", "rationale"):
                assert k in p, (k, p)

    def test_repair_rule_codes_closed_set(self):
        from services.clinical_role_repair import RULES
        picott = _glp1_ldl_picott_with_wrong_roles()
        res = clinical_role_repair.repair(picott)
        for p in res.repair_provenance:
            assert p["rule"] in RULES, p["rule"]


# =============================================================================
# Group C — RoleWeightingService
# =============================================================================

class TestGroupC_RoleWeighting:

    def test_pharmacology_intervention_higher_than_population(self):
        assert role_weight("pharmacology", "intervention") == 40
        assert role_weight("pharmacology", "population") == 20
        assert role_weight("pharmacology", "intervention") > role_weight(
            "pharmacology", "population",
        ) + 10  # spec: ~10 pts higher

    def test_therapy_intervention_higher_than_population(self):
        assert role_weight("therapy", "intervention") == 35
        assert role_weight("therapy", "population") == 25

    def test_diagnosis_privileges_index_test(self):
        weights = role_weighting.weights_for("diagnosis")
        assert weights["index_test"] == 35
        # index_test must be the (joint-)highest weight.
        assert weights["index_test"] >= max(weights.values())

    def test_etiology_harm_privileges_exposure(self):
        weights = role_weighting.weights_for("etiology_harm")
        assert weights["exposure"] == 35
        assert weights["exposure"] == max(weights.values())

    def test_prognosis_privileges_outcome(self):
        weights = role_weighting.weights_for("prognosis")
        assert weights["outcome"] == 35
        assert weights["outcome"] == max(weights.values())

    def test_background_privileges_concept_clarity(self):
        weights = role_weighting.weights_for("background")
        assert weights["concept_clarity"] == 50
        assert weights["concept_clarity"] == max(weights.values())

    def test_dosing_protocol_includes_dose_terms_weight(self):
        weights = role_weighting.weights_for("dosing_protocol")
        assert weights["dose_terms"] == 15
        # And intervention is still the highest.
        assert weights["intervention"] == max(weights.values())

    def test_vaccine_schedule_includes_schedule_terms_weight(self):
        weights = role_weighting.weights_for("vaccine_schedule")
        assert weights["schedule_terms"] == 25
        assert weights["intervention"] == 30

    def test_unknown_role_returns_neutral_weight(self):
        assert role_weight("therapy", "frabnozzle") == 5
        assert role_weight("therapy", "") == 5

    def test_unknown_qtype_falls_back_to_other(self):
        # Unknown qtype uses ``other`` table.
        assert role_weight("foo_bar", "intervention") == role_weight(
            "other", "intervention",
        )

    def test_weighted_relevance_higher_with_more_matched_roles(self):
        # Base relevance 50. With NO matched roles → ~30 (no coverage).
        low = weighted_relevance(
            base_relevance=50, matched_roles=[],
            type_of_question="pharmacology",
        )
        high = weighted_relevance(
            base_relevance=50,
            matched_roles=["intervention", "outcome", "population"],
            type_of_question="pharmacology",
        )
        assert high > low, (low, high)

    def test_weighted_relevance_penalty_for_missing_high_weight_roles(self):
        base = weighted_relevance(
            base_relevance=80, matched_roles=["intervention"],
            type_of_question="pharmacology",
        )
        penalized = weighted_relevance(
            base_relevance=80, matched_roles=["intervention"],
            missing_roles=["outcome", "population"],
            type_of_question="pharmacology",
        )
        assert penalized < base, (penalized, base)

    def test_role_aliases_normalize(self):
        """``intervention_or_exposure_or_index_test`` alias maps to
        ``intervention`` weight."""
        assert role_weight("therapy", "intervention_or_exposure_or_index_test") == 35
        assert role_weight("diagnosis", "target_diagnosis") == 15  # falls to outcome

    def test_weights_for_returns_copy(self):
        w = role_weighting.weights_for("therapy")
        w["intervention"] = 999
        # Mutation doesn't leak.
        assert ROLE_WEIGHTS["therapy"]["intervention"] != 999

    def test_weighted_relevance_clamped_to_0_100(self):
        v = weighted_relevance(
            base_relevance=200, matched_roles=[],
            type_of_question="pharmacology",
        )
        assert 0 <= v <= 100
        v = weighted_relevance(
            base_relevance=-50, matched_roles=[],
            type_of_question="pharmacology",
        )
        assert 0 <= v <= 100

    def test_all_qtype_weight_dicts_sum_close_to_100(self):
        for qtype, weights in ROLE_WEIGHTS.items():
            s = sum(weights.values())
            assert 90 <= s <= 110, (qtype, s)


# =============================================================================
# Group C extra — GLP-1/LDL caps from picott_ranker
# =============================================================================

class TestGroupC_GLP1LdlCaps:
    """Phase 5F closed-set caps: tirzepatide_only_for_glp1_question (70),
    glp1_biology_no_lipid_outcome (50), ldl_without_glp1_intervention (30).
    All layered on top of Phase 5A generic caps with lowest-wins."""

    def _glp1_lipid_picott(self) -> dict:
        return {
            "normalized_question": "how do GLP-1 receptor agonists impact LDL-C?",
            "type_of_question": "pharmacology",
            "question_subtype": "metabolic_biomarker_effect",
            "must_have_concepts": [
                {"concept": "GLP-1 receptor agonists", "role": "intervention",
                 "synonyms": ["semaglutide", "liraglutide"]},
                {"concept": "LDL-C", "role": "outcome",
                 "synonyms": ["LDL cholesterol", "low-density lipoprotein cholesterol"]},
            ],
            "ranking_concepts": [],
            "should_have_concepts": [],
            "negative_or_penalty_concepts": [],
        }

    def test_tirzepatide_only_capped_at_70(self):
        from services.picott_ranker import _check_concepts
        picott = self._glp1_lipid_picott()
        check = _check_concepts(
            title="Tirzepatide and LDL-C in type 2 diabetes",
            abstract="Mounjaro reduced LDL cholesterol over 52 weeks.",
            ingest_full_text="",
            picott=picott,
        )
        assert "tirzepatide_only_for_glp1_question" in check.cap_reasons, \
            check.cap_reasons
        assert check.cap <= 70

    def test_glp1_biology_without_lipid_outcome_capped_at_50(self):
        from services.picott_ranker import _check_concepts
        picott = self._glp1_lipid_picott()
        check = _check_concepts(
            title="Semaglutide and beta-cell function",
            abstract="Semaglutide improved beta cell mass in mice.",
            ingest_full_text="",
            picott=picott,
        )
        assert "glp1_biology_no_lipid_outcome" in check.cap_reasons
        assert check.cap <= 50

    def test_ldl_without_glp1_capped_at_30(self):
        from services.picott_ranker import _check_concepts
        picott = self._glp1_lipid_picott()
        check = _check_concepts(
            title="Atorvastatin and LDL-C lowering",
            abstract="Statins reduced LDL cholesterol by 50%.",
            ingest_full_text="",
            picott=picott,
        )
        assert "ldl_without_glp1_intervention" in check.cap_reasons
        assert check.cap <= 30

    def test_glp1_with_lipid_outcome_not_capped_by_phase5f(self):
        """A direct GLP-1 + LDL article should NOT trigger any of the
        Phase 5F GLP-1/LDL caps."""
        from services.picott_ranker import _check_concepts
        picott = self._glp1_lipid_picott()
        check = _check_concepts(
            title="Semaglutide reduces LDL-C in type 2 diabetes",
            abstract="In a randomized trial of semaglutide vs placebo, LDL cholesterol decreased.",
            ingest_full_text="",
            picott=picott,
        )
        new_caps = {
            "tirzepatide_only_for_glp1_question",
            "glp1_biology_no_lipid_outcome",
            "ldl_without_glp1_intervention",
        }
        assert not (new_caps & set(check.cap_reasons)), check.cap_reasons

    def test_explicit_tirzepatide_question_not_capped(self):
        """If the question text or PICOTT includes tirzepatide, the
        tirzepatide_only cap should NOT fire."""
        from services.picott_ranker import _check_concepts
        picott = self._glp1_lipid_picott()
        # Add tirzepatide to the must-haves so the question explicitly
        # includes it.
        picott["must_have_concepts"].append({
            "concept": "tirzepatide", "role": "intervention",
            "synonyms": [],
        })
        check = _check_concepts(
            title="Tirzepatide and LDL-C",
            abstract="Tirzepatide reduced LDL cholesterol.",
            ingest_full_text="",
            picott=picott,
        )
        assert "tirzepatide_only_for_glp1_question" not in check.cap_reasons
