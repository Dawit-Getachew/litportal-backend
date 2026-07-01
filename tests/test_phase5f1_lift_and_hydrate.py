"""Phase 5F.1 — Ontology lift + display serializer hydration regression.

8 tests covering the FIX 1 + FIX 2 requirements from the user's brief.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# FIX 1 — Ontology lift into PICOTT/must_have_concepts
# =============================================================================

def _empty_picott_glp1() -> dict:
    """Phase 5F.1 scenario — LLM returned a near-empty PICOTT (which is
    exactly what the user reported)."""
    return {
        "normalized_question": "how do GLP-1 receptor agonists impact LDL-C?",
        "type_of_question": "other",
        "question_subtype": None,
        "specialty": "primary_care",
        "pico": {
            "population": [],
            "intervention_or_exposure_or_index_test": [],
            "comparator_or_reference_standard": [],
            "outcomes_or_target_diagnosis": [],
        },
        "must_have_concepts": [],
        "ranking_concepts": [],
    }


def _ontology_glp1_hits() -> list[dict]:
    """Ontology mappings the mapper would emit for the GLP-1/LDL question."""
    return [
        {
            "original_term": "GLP-1 receptor agonists",
            "normalized_term": "GLP-1 receptor agonists",
            "canonical_concept": "GLP-1 receptor agonists",
            "concept_type": "drug_class",
            "role_hint": "intervention",
            "synonyms": ["semaglutide", "liraglutide"],
            "acronyms": ["GLP-1 RA"],
            "mesh_terms": ["Glucagon-Like Peptide-1 Receptor"],
            "rxnorm_terms": ["semaglutide", "liraglutide"],
            "umls_cui": None,
            "confidence": 0.95,
            "source": "specialty_pack",
            "rationale": "specialty pack 'cardiometabolic_lipids' / group 'glp1_receptor_agonists'.",
        },
        {
            "original_term": "LDL-C",
            "normalized_term": "LDL-C",
            "canonical_concept": "LDL-C",
            "concept_type": "biomarker",
            "role_hint": "outcome",
            "synonyms": ["LDL cholesterol", "low-density lipoprotein cholesterol"],
            "acronyms": [],
            "mesh_terms": ["Cholesterol, LDL"],
            "rxnorm_terms": [],
            "umls_cui": "C0023823",
            "confidence": 0.99,
            "source": "specialty_pack",
            "rationale": "specialty pack 'cardiometabolic_lipids' / group 'lipid_outcomes'.",
        },
    ]


class TestFix1_OntologyLift:

    def test_ontology_lift_creates_missing_intervention(self):
        from services.clinical_context_engine import _lift_ontology_into_picott
        from services.role_weighting import ROLE_WEIGHTS
        picott = _empty_picott_glp1()
        lifts = _lift_ontology_into_picott(
            picott, _ontology_glp1_hits(),
            role_weights_dict=dict(ROLE_WEIGHTS["pharmacology"]),
        )
        # Intervention concept created.
        mhs = picott["must_have_concepts"]
        glp1 = next(m for m in mhs if m["concept"] == "GLP-1 receptor agonists")
        assert glp1["role"] == "intervention"
        assert glp1["source"] == "ontology_lift"
        assert glp1["provenance"]["ontology_source"] in {"specialty_pack", "curated_mesh", "curated_rxnorm"}
        assert glp1["required_for_high_relevance"] is True  # pharmacology intervention=40 → required
        # Mirrored into pico block.
        assert "GLP-1 receptor agonists" in picott["pico"]["intervention_or_exposure_or_index_test"]
        # ontology_lifts entry recorded.
        glp1_lift = next(l for l in lifts if l["canonical_term"] == "GLP-1 receptor agonists")
        assert glp1_lift["role"] == "intervention"
        assert glp1_lift["reason"] == "LLM omitted concept; lifted from ontology hit"
        # Outcome also lifted.
        ldl = next(m for m in mhs if m["concept"] == "LDL-C")
        assert ldl["role"] == "outcome"
        assert "LDL-C" in picott["pico"]["outcomes_or_target_diagnosis"]

    def test_ontology_lift_does_not_override_llm_concepts(self):
        """If LLM provided an intervention with role=intervention, the
        ontology lift must NOT add the same concept twice (case-
        insensitive on canonical or synonym), but it MAY add a different
        intervention concept."""
        from services.clinical_context_engine import _lift_ontology_into_picott
        from services.role_weighting import ROLE_WEIGHTS
        picott = _empty_picott_glp1()
        # LLM provided semaglutide explicitly (which is a synonym of the
        # GLP-1 RA class entry).
        picott["must_have_concepts"] = [
            {"concept": "semaglutide", "role": "intervention", "synonyms": []},
        ]
        before = len(picott["must_have_concepts"])
        lifts = _lift_ontology_into_picott(
            picott, _ontology_glp1_hits(),
            role_weights_dict=dict(ROLE_WEIGHTS["pharmacology"]),
        )
        # Semaglutide already exists as a synonym of GLP-1 RAs → NOT
        # lifted again. LDL-C is unrelated → IS lifted.
        glp1_lifts = [l for l in lifts if l["canonical_term"] == "GLP-1 receptor agonists"]
        assert glp1_lifts == [], glp1_lifts
        ldl_lifts = [l for l in lifts if l["canonical_term"] == "LDL-C"]
        assert ldl_lifts, "LDL-C should still be lifted"
        # LLM concept preserved (not overridden).
        sem = next(m for m in picott["must_have_concepts"]
                    if m["concept"] == "semaglutide")
        assert sem["role"] == "intervention"
        # 1 LLM concept + 1 lifted (LDL-C) = 2.
        assert len(picott["must_have_concepts"]) == before + 1

    def test_ontology_lift_respects_confidence_threshold(self):
        from services.clinical_context_engine import _lift_ontology_into_picott
        from services.role_weighting import ROLE_WEIGHTS
        picott = _empty_picott_glp1()
        low_conf_hits = [
            dict(_ontology_glp1_hits()[0], confidence=0.7),  # below 0.85
        ]
        lifts = _lift_ontology_into_picott(
            picott, low_conf_hits,
            role_weights_dict=dict(ROLE_WEIGHTS["pharmacology"]),
        )
        assert lifts == []
        assert picott["must_have_concepts"] == []

    def test_ontology_lift_skips_non_pico_roles(self):
        from services.clinical_context_engine import _lift_ontology_into_picott
        from services.role_weighting import ROLE_WEIGHTS
        picott = _empty_picott_glp1()
        # role_hint=schedule_terms / context / dose_terms / modifier
        # must NOT be lifted into PICOTT.
        non_pico_hits = [
            {**_ontology_glp1_hits()[0], "role_hint": "context",
             "canonical_concept": "diabetes screening context"},
            {**_ontology_glp1_hits()[1], "role_hint": "schedule_terms",
             "canonical_concept": "recommended age"},
            {**_ontology_glp1_hits()[0], "role_hint": "dose_terms",
             "canonical_concept": "high dose"},
        ]
        lifts = _lift_ontology_into_picott(
            picott, non_pico_hits,
            role_weights_dict=dict(ROLE_WEIGHTS["pharmacology"]),
        )
        assert lifts == []
        assert picott["must_have_concepts"] == []


# =============================================================================
# FIX 2 — Display serializer hydration
# =============================================================================

class TestFix2_DisplayHydration:

    def test_display_serializer_hydrates_subtype_from_role_weights(self):
        from services.interpreted_question_display import (
            build_interpreted_question_display,
        )
        session_doc = {
            "id": "test",
            "original_question": "how do glp1s impact ldl?",
            "picott_json": {
                "type_of_question": "other",
                "question_subtype": None,
                "specialty": "primary_care",
                "must_have_concepts": [
                    {"concept": "GLP-1 receptor agonists", "role": "intervention"},
                ],
                "pico": {
                    "population": [],
                    "intervention_or_exposure_or_index_test": ["GLP-1 receptor agonists"],
                    "comparator_or_reference_standard": [],
                    "outcomes_or_target_diagnosis": [],
                },
                "ranking_concepts": [],
            },
            "role_weights_json": {
                "type_of_question": "other",
                "question_subtype": "metabolic_biomarker_effect",
                "weights": {"intervention": 40, "outcome": 25},
            },
        }
        out = build_interpreted_question_display(session_doc)
        assert out["question_subtype"] == "metabolic_biomarker_effect", out
        # ``type_of_question`` should include the subtype (with or
        # without underscores).
        toq_lower = out["type_of_question"].lower()
        assert ("metabolic biomarker effect" in toq_lower
                or "metabolic_biomarker_effect" in toq_lower), out

    def test_display_serializer_hydrates_specialty_from_ontology(self):
        from services.interpreted_question_display import (
            build_interpreted_question_display,
        )
        session_doc = {
            "id": "test",
            "picott_json": {
                "type_of_question": "other",
                "specialty": "primary_care",  # blocklisted
                "must_have_concepts": [
                    {"concept": "LDL-C", "role": "outcome"},
                ],
                "pico": {"outcomes_or_target_diagnosis": ["LDL-C"]},
                "ranking_concepts": [],
            },
            "ontology_mappings_json": [
                {
                    "canonical_concept": "LDL-C", "confidence": 0.99,
                    "role_hint": "outcome",
                    "rationale": "specialty pack 'cardiometabolic_lipids' / group 'lipid_outcomes'.",
                },
                {
                    "canonical_concept": "GLP-1 receptor agonists", "confidence": 0.95,
                    "role_hint": "intervention",
                    "rationale": "specialty pack 'cardiometabolic_lipids' / group 'glp1_receptor_agonists'.",
                },
            ],
        }
        out = build_interpreted_question_display(session_doc)
        # Specialty derived from cardiometabolic_lipids majority.
        assert "Cardiometabolic" in out["specialty"], out["specialty"]

    def test_display_serializer_question_scope_foreground(self):
        from services.interpreted_question_display import (
            build_interpreted_question_display,
        )
        session_doc = {
            "id": "test",
            "picott_json": {
                "type_of_question": "therapy",
                "specialty": "cardiology",
                "question_scope": "unclear",  # raw is unclear
                "must_have_concepts": [
                    {"concept": "GLP-1 receptor agonists", "role": "intervention"},
                    {"concept": "LDL-C", "role": "outcome"},
                ],
                "pico": {},
                "ranking_concepts": [],
            },
        }
        out = build_interpreted_question_display(session_doc)
        assert out["question_scope"].lower() == "foreground", out["question_scope"]

    def test_display_serializer_pico_arrays_filled_from_lifts(self):
        """Phase 5F.1 AMI baseline check — when must_have_concepts has
        RBC transfusion (intervention) and pico.intervention_array is
        empty, display must still show RBC transfusion under
        intervention_or_exposure_or_index_test, not 'Not specified'."""
        from services.interpreted_question_display import (
            build_interpreted_question_display,
        )
        session_doc = {
            "id": "test-ami",
            "picott_json": {
                "type_of_question": "therapy",
                "question_subtype": "transfusion_threshold_comparison",
                "specialty": "cardiology",
                "must_have_concepts": [
                    {"concept": "red blood cell transfusion",
                     "role": "intervention", "synonyms": ["RBC transfusion"]},
                    {"concept": "acute myocardial infarction",
                     "role": "population"},
                ],
                "pico": {
                    "population": [],
                    "intervention_or_exposure_or_index_test": [],
                    "comparator_or_reference_standard": [],
                    "outcomes_or_target_diagnosis": [],
                },
                "ranking_concepts": [],
            },
        }
        out = build_interpreted_question_display(session_doc)
        intervention = out["intervention_or_exposure_or_index_test"]
        assert any("red blood cell transfusion" in i.lower() or "RBC transfusion" in i
                    for i in intervention), intervention


# =============================================================================
# FIX 1 + 2 — End-to-end GLP-1/LDL smoke test
# =============================================================================

class TestFix1And2_E2E:

    def test_glp1_ldl_end_to_end_picott_filled(self):
        """Run the full context engine over a near-empty PICOTT and
        confirm the resulting interpreted_question_display_json has all
        the headline fields populated (no 'Not specified' for the
        intervention / outcome / question_subtype / specialty / scope)."""
        from services.clinical_context_engine import build_clinical_context
        from services.interpreted_question_display import (
            build_interpreted_question_display,
        )
        parsed = {
            "normalized_question": (
                "how do GLP-1 receptor agonists impact LDL-C?"
            ),
            "shorthand_substitutions": [
                {"original": "glp1s", "normalized": "GLP-1 receptor agonists",
                 "confidence": 1.0, "reason": "drug_class_shorthand"},
                {"original": "ldl", "normalized": "LDL-C / low-density lipoprotein cholesterol",
                 "confidence": 1.0, "reason": "lab_shorthand"},
            ],
            "picott_json": _empty_picott_glp1(),
        }
        ctx = build_clinical_context(parsed)
        # Build the session doc shape the pipeline would persist.
        session_doc = {
            "id": "e2e-test",
            "original_question": "how do glp1s impact ldl cholesterole?",
            "picott_json": ctx["repaired_picott"],
            "role_weights_json": ctx["role_weights"],
            "ontology_mappings_json": ctx["ontology_mappings"],
            "ontology_lifts_json": ctx["ontology_lifts"],
            "multi_intent_json": ctx["multi_intent"],
            "shorthand_substitutions_json": ctx["shorthand_substitutions"],
            "role_repairs_json": ctx["role_repairs"],
        }
        out = build_interpreted_question_display(session_doc)
        # Headline assertions.
        assert out["question_subtype"] != "Not specified", out
        assert out["question_subtype"] == "metabolic_biomarker_effect", out
        # Intervention block populated by the lift.
        interv = out["intervention_or_exposure_or_index_test"]
        assert any("GLP-1" in i for i in interv), interv
        outcome = out["outcomes_or_target_diagnosis"]
        assert any("LDL" in o for o in outcome), outcome
        # Specialty derived from cardiometabolic ontology hits.
        assert "Cardiometabolic" in out["specialty"], out["specialty"]
        # Foreground scope.
        assert out["question_scope"].lower() == "foreground", out["question_scope"]
        # Lift audit trail exposed on the display.
        assert isinstance(out.get("ontology_lifts"), list)
        assert len(out["ontology_lifts"]) >= 2
