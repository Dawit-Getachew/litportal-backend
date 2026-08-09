"""Phase 6F — Primary Clinical Entity Anchoring + PICOTT Extraction Hardening.

Test groups:
  A) PrimaryClinicalEntityAnchorService unit tests
  B) PICOTT extraction for Q1 (psoriasis) and Q2 (Seroquel)
  C) Generic-modifier detection
  D) Boolean query compiler includes the anchor on every query
  E) Ranking caps for wrong-disease / wrong-drug
  F) Default-selection excludes wrong-anchor articles
  G) Closed-set EXCLUSION_REASONS extension

These are pure unit tests — they exercise the deterministic pipeline
(parser → context engine → primary_clinical_anchor_service) without
hitting the network or the LLM provider (LLM is left unconfigured for
the test runner).
"""
from __future__ import annotations

import asyncio


from services.clinical_context_engine import build_clinical_context
from services.clinical_question_parser import clinical_question_parser
from services.concept_block_query_builder import concept_block_query_builder
from services.default_selection import (
    EXCLUSION_REASONS,
    select_default_articles,
)
from services.picott_ranker import CAP_REASONS, picott_aware_ranker
from services.primary_clinical_anchor import (
    CLARIFICATION_MESSAGE,
    GENERIC_MODIFIERS,
    primary_clinical_anchor_service,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(question: str) -> dict:
    """Run the deterministic pipeline (no LLM) and return the parsed +
    context-engine + anchor-service output.
    """
    parsed = asyncio.get_event_loop().run_until_complete(
        clinical_question_parser.parse(question)
    )
    context = build_clinical_context(parsed)
    picott = context.get("repaired_picott") or parsed.get("picott_json") or {}
    parsed["picott_json"] = picott
    anchors = primary_clinical_anchor_service.derive(
        picott=picott,
        ontology_mappings=list(context.get("ontology_mappings") or []),
        normalized_question=parsed.get("normalized_question") or question,
    )
    return {"parsed": parsed, "context": context, "picott": picott, "anchors": anchors}


# Q1 / Q2 reproducer questions from the Phase 6F directive.
Q1 = "What is the first line treatment and second line treatment for psoriasis?"
Q2 = "What are the side effects of Seroquel?"


# ---------------------------------------------------------------------------
# A) PrimaryClinicalEntityAnchorService unit tests
# ---------------------------------------------------------------------------

class TestPrimaryClinicalEntityAnchorService:
    def test_q1_psoriasis_yields_disease_anchor(self):
        out = _parse(Q1)
        anchors = out["anchors"]["primary_anchors"]
        assert anchors, "Q1 should produce at least one primary anchor"
        diseases = [
            a for a in anchors if a["entity_type"] == "disease_or_problem"
        ]
        assert diseases, "Q1 should have a disease_or_problem anchor"
        names = [a["canonical_concept"].lower() for a in diseases]
        assert any("psoriasis" in n for n in names)
        # All disease anchors are required + capped ≤ 20.
        for a in diseases:
            assert a["required_for_search"] is True
            assert a["missing_relevance_cap"] <= 20

    def test_q2_seroquel_yields_drug_anchor(self):
        out = _parse(Q2)
        anchors = out["anchors"]["primary_anchors"]
        assert anchors, "Q2 should produce at least one primary anchor"
        drugs = [
            a for a in anchors if a["entity_type"] == "drug_or_drug_class"
        ]
        assert drugs, "Q2 should have a drug_or_drug_class anchor"
        names = [
            a["canonical_concept"].lower() for a in drugs
        ] + [s.lower() for a in drugs for s in a["synonyms"]]
        assert any(n in names for n in (
            "seroquel", "quetiapine", "quetiapine fumarate",
        ))
        # Drug anchors are required + capped ≤ 20.
        for a in drugs:
            assert a["required_for_search"] is True
            assert a["missing_relevance_cap"] <= 20

    def test_no_anchor_returns_failure_reason(self):
        # Pure-generic question — no clinical entity at all.
        out = _parse("What are the side effects?")
        anchors = out["anchors"]
        primary_eligible = [
            a for a in anchors["primary_anchors"]
            if a["entity_type"] in {
                "disease_or_problem", "drug_or_drug_class", "vaccine",
                "diagnostic_test", "procedure", "exposure",
            }
        ]
        # Either repair produced a hit OR the failure_reason is set —
        # this is a closed-set contract.
        if not primary_eligible:
            assert anchors["anchor_failure_reason"] == "no_primary_clinical_entity"

    def test_generic_modifiers_detected_q1(self):
        out = _parse(Q1)
        gens = set(out["anchors"]["generic_modifiers"])
        assert "first line treatment" in gens or "first-line treatment" in gens
        assert "second line treatment" in gens or "second-line treatment" in gens

    def test_generic_modifiers_detected_q2(self):
        out = _parse(Q2)
        gens = set(out["anchors"]["generic_modifiers"])
        assert "side effects" in gens

    def test_clarification_message_is_closed_set(self):
        assert CLARIFICATION_MESSAGE
        assert "Please specify" in CLARIFICATION_MESSAGE

    def test_generic_modifiers_constant_covers_spec_phrases(self):
        for phrase in (
            "side effects", "adverse effects", "first-line treatment",
            "second-line treatment", "benefit", "recommended age",
            "treatment", "therapy", "management", "guideline",
            "diagnosis", "accuracy", "dose", "dosing", "timing",
        ):
            assert phrase in GENERIC_MODIFIERS, phrase


# ---------------------------------------------------------------------------
# B) PICOTT extraction for Q1 and Q2
# ---------------------------------------------------------------------------

class TestPICOTTExtractionPhase6F:
    def test_q1_picott_not_blank(self):
        out = _parse(Q1)
        picott = out["picott"]
        # Q1 expectations from the Phase 6F directive.
        assert picott.get("type_of_question") in {"therapy", "guideline"}
        assert picott.get("question_subtype") == "treatment_sequencing"
        # Specialty MUST NOT be primary_care for a disease-anchored Q.
        assert picott.get("specialty", "").lower() in {
            "dermatology", "rheumatology",
        }
        # The picott must carry a disease anchor in must_have_concepts.
        # role can be "condition" OR "population" (role_repair commonly
        # rewrites condition → population for therapy questions).
        mh = picott.get("must_have_concepts") or []
        assert any(
            (m.get("role") or "").lower() in {"condition", "population"}
            and "psoriasis" in (m.get("concept") or "").lower()
            for m in mh
        )

    def test_q2_picott_not_blank(self):
        out = _parse(Q2)
        picott = out["picott"]
        assert picott.get("type_of_question") == "pharmacology"
        assert picott.get("question_subtype") == "drug_safety"
        assert "psych" in (picott.get("specialty") or "").lower() \
            or "pharm" in (picott.get("specialty") or "").lower()
        # The picott must carry a drug intervention anchor.
        mh = picott.get("must_have_concepts") or []
        drug_concepts = [
            m for m in mh
            if (m.get("role") or "").lower() == "intervention"
        ]
        assert drug_concepts
        all_terms = " ".join(
            [m.get("concept") or "" for m in drug_concepts]
            + [s for m in drug_concepts for s in (m.get("synonyms") or [])]
        ).lower()
        assert any(
            t in all_terms
            for t in ("seroquel", "quetiapine", "quetiapine fumarate")
        )
        # Adverse-effect outcome must-have should also be present
        # (drug_safety subtype).
        assert any(
            (m.get("role") or "").lower() == "outcome"
            and ("adverse" in (m.get("concept") or "").lower()
                 or "side effect" in (m.get("concept") or "").lower())
            for m in mh
        )

    def test_q1_no_blank_background_classification(self):
        out = _parse(Q1)
        picott = out["picott"]
        assert picott.get("type_of_question") != "background"
        assert picott.get("type_of_question") != "other"

    def test_q2_no_blank_background_classification(self):
        out = _parse(Q2)
        picott = out["picott"]
        assert picott.get("type_of_question") != "background"
        assert picott.get("type_of_question") != "other"


# ---------------------------------------------------------------------------
# C) Generic-modifier detection / clarification behaviour
# ---------------------------------------------------------------------------

class TestGenericModifierDetection:
    def test_treatment_sequencing_triggers_subtype(self):
        for q in (
            "What is the first line treatment for psoriasis?",
            "What is the second-line treatment for psoriasis?",
            "What is the treatment algorithm for psoriasis?",
            "What are the treatment options for psoriasis?",
        ):
            out = _parse(q)
            assert out["picott"].get("question_subtype") == "treatment_sequencing", q

    def test_drug_safety_triggers_subtype(self):
        for q in (
            "What are the side effects of Seroquel?",
            "What are the adverse effects of quetiapine?",
            "What is the safety profile of olanzapine?",
            "What is the tolerability of risperidone?",
        ):
            out = _parse(q)
            assert out["picott"].get("question_subtype") == "drug_safety", q

    def test_psoriatic_arthritis_does_not_dominate_plain_psoriasis(self):
        out = _parse(Q1)
        anchors = out["anchors"]["primary_anchors"]
        # Plain psoriasis Q must carry a psoriasis anchor — psoriatic
        # arthritis must NOT replace it.
        names = " ".join([a["canonical_concept"].lower() for a in anchors]) \
                + " " + " ".join([s.lower() for a in anchors for s in a["synonyms"]])
        assert "psoriasis" in names


# ---------------------------------------------------------------------------
# D) Boolean query compiler includes the anchor on every query
# ---------------------------------------------------------------------------

class TestBooleanQueryAnchorEnforcement:
    def test_psoriasis_in_every_query(self):
        out = _parse(Q1)
        picott = out["picott"]
        # Emit queries for both Europe PMC and PubMed.
        queries = list(concept_block_query_builder.build(
            picott, sources=("europe_pmc", "pubmed"),
        ))
        assert queries, "compiler should emit at least one query for Q1"
        for cbq in queries:
            rec = cbq.to_query_record()
            text = rec["query_string"].lower()
            assert "psoriasis" in text, (
                f"Q1 query family={rec['query_family']} source={rec['source']} "
                f"does not include the psoriasis anchor: {rec['query_string']}"
            )

    def test_seroquel_or_quetiapine_in_every_query(self):
        out = _parse(Q2)
        picott = out["picott"]
        queries = list(concept_block_query_builder.build(
            picott, sources=("europe_pmc", "pubmed"),
        ))
        assert queries, "compiler should emit at least one query for Q2"
        for cbq in queries:
            rec = cbq.to_query_record()
            text = rec["query_string"].lower()
            assert any(t in text for t in (
                "seroquel", "quetiapine",
            )), (
                f"Q2 query family={rec['query_family']} source={rec['source']} "
                f"is missing the Seroquel/quetiapine anchor: {rec['query_string']}"
            )

    def test_no_side_effects_only_query(self):
        out = _parse(Q2)
        picott = out["picott"]
        queries = list(concept_block_query_builder.build(
            picott, sources=("pubmed",),
        ))
        # No emitted query may be ONLY "side effects" without the drug.
        for cbq in queries:
            text = cbq.to_query_record()["query_string"].lower()
            if "side effect" in text or "adverse effect" in text:
                assert any(t in text for t in (
                    "seroquel", "quetiapine",
                )), f"Side-effects-only query emitted: {text}"


# ---------------------------------------------------------------------------
# E) Ranking caps — wrong disease / wrong drug
# ---------------------------------------------------------------------------

class TestRankingCapsPhase6F:
    def test_new_cap_codes_in_closed_set(self):
        for c in (
            "wrong_drug", "wrong_condition", "missing_primary_anchor",
            "wrong_primary_anchor", "disease_only_match",
            "drug_safety_no_adverse_effect",
            "treatment_sequencing_wrong_disease",
        ):
            assert c in CAP_REASONS, c

    def test_wrong_disease_article_for_psoriasis_question_capped(self):
        out = _parse(Q1)
        picott = out["picott"]
        ranked = [{
            "article_id": "a1",
            "_article": {
                "id": "a1",
                "title": "Second-line treatment for biliary tract cancer",
                "abstract": "Outcomes after first-line gemcitabine in cholangiocarcinoma.",
                "article_type_classification": "guideline",
            },
            "relevance_score": 80.0,
            "overall_rank_score": 80,
            "rank_explanation": "",
        }]
        picott_aware_ranker.postprocess(ranked, picott)
        r = ranked[0]
        assert r.get("relevance_cap_applied") is not None
        assert r["relevance_cap_applied"] <= 20
        # The new wrong_condition / missing_condition_anchor codes must
        # appear in cap_reasons.
        assert "missing_condition_anchor" in (r.get("cap_reasons") or [])
        assert "wrong_condition" in (r.get("cap_reasons") or [])

    def test_wrong_drug_article_for_seroquel_question_capped(self):
        out = _parse(Q2)
        picott = out["picott"]
        ranked = [{
            "article_id": "b1",
            "_article": {
                "id": "b1",
                "title": "Side effects of caffeine in young adults",
                "abstract": "Caffeine adverse effects including insomnia and palpitations.",
                "article_type_classification": "rct",
            },
            "relevance_score": 70.0,
            "overall_rank_score": 70,
            "rank_explanation": "",
        }]
        picott_aware_ranker.postprocess(ranked, picott)
        r = ranked[0]
        assert r["relevance_cap_applied"] is not None
        assert r["relevance_cap_applied"] <= 20
        assert "missing_intervention_anchor" in (r.get("cap_reasons") or [])
        assert "wrong_drug" in (r.get("cap_reasons") or [])

    def test_correct_anchor_article_not_capped(self):
        out = _parse(Q2)
        picott = out["picott"]
        ranked = [{
            "article_id": "c1",
            "_article": {
                "id": "c1",
                "title": "Adverse effects of quetiapine in long-term use",
                "abstract": (
                    "Treatment-emergent adverse events with quetiapine "
                    "including weight gain, somnolence, and metabolic "
                    "effects in adults with bipolar disorder."
                ),
                "article_type_classification": "systematic_review",
            },
            "relevance_score": 75.0,
            "overall_rank_score": 75,
            "rank_explanation": "",
        }]
        picott_aware_ranker.postprocess(ranked, picott)
        r = ranked[0]
        # Cap should NOT be ≤ 20 — the correct article has both
        # intervention and outcome anchors.
        if r.get("relevance_cap_applied") is not None:
            assert r["relevance_cap_applied"] > 20


# ---------------------------------------------------------------------------
# F) Default-selection — wrong-anchor articles never selected
# ---------------------------------------------------------------------------

class TestDefaultSelectionPhase6F:
    def test_wrong_disease_article_excluded(self):
        out = _parse(Q1)
        picott = out["picott"]
        ranked = [{
            "article_id": "x1",
            "_article": {
                "id": "x1",
                "title": "First-line treatment for melanoma",
                "abstract": "Targeted therapy for advanced melanoma.",
                "article_type_classification": "guideline",
            },
            "relevance_score": 80.0,
            "overall_rank_score": 80,
            "rank_explanation": "",
        }]
        picott_aware_ranker.postprocess(ranked, picott)
        sel = select_default_articles(picott, ranked)
        assert "x1" not in sel["selected_article_ids"]
        # Closed-set exclusion reason must be one of the new codes.
        reasons = sel["excluded_rationale_summary"]
        keys = set(reasons.keys())
        assert keys & {
            "cap_reason_wrong_condition",
            "cap_reason_missing_condition_anchor",
            "cap_reason_missing_primary_anchor",
        }

    def test_wrong_drug_article_excluded(self):
        out = _parse(Q2)
        picott = out["picott"]
        ranked = [{
            "article_id": "y1",
            "_article": {
                "id": "y1",
                "title": "Adverse effects of corticosteroids in Duchenne MD",
                "abstract": "Long-term corticosteroid adverse events.",
                "article_type_classification": "rct",
            },
            "relevance_score": 75.0,
            "overall_rank_score": 75,
            "rank_explanation": "",
        }]
        picott_aware_ranker.postprocess(ranked, picott)
        sel = select_default_articles(picott, ranked)
        assert "y1" not in sel["selected_article_ids"]
        reasons = sel["excluded_rationale_summary"]
        keys = set(reasons.keys())
        assert keys & {
            "cap_reason_wrong_drug",
            "cap_reason_missing_intervention_anchor",
            "cap_reason_missing_primary_anchor",
        }


# ---------------------------------------------------------------------------
# G) Closed-set EXCLUSION_REASONS extension
# ---------------------------------------------------------------------------

class TestExclusionReasonsClosedSet:
    def test_new_phase6f_codes_in_exclusion_reasons(self):
        for c in (
            "cap_reason_wrong_drug",
            "cap_reason_wrong_condition",
            "cap_reason_missing_primary_anchor",
            "cap_reason_wrong_primary_anchor",
            "cap_reason_drug_safety_no_adverse_effect",
            "cap_reason_treatment_sequencing_wrong_disease",
        ):
            assert c in EXCLUSION_REASONS, c

    def test_existing_codes_still_present(self):
        # Regression — existing exclusion reasons must still be in the
        # closed set.
        for c in (
            "cap_reason_missing_condition_anchor",
            "cap_reason_missing_intervention_anchor",
            "cap_reason_disease_only_match",
            "negative_concept_fired",
            "relevance_below_60",
            "excluded_retracted_or_preprint_or_unsafe",
        ):
            assert c in EXCLUSION_REASONS, c


# ---------------------------------------------------------------------------
# H) Regression — prior benchmark questions still produce non-blank PICOTT.
# ---------------------------------------------------------------------------

class TestRegressionPriorBenchmarks:
    def test_glp1_ldl_still_works(self):
        out = _parse("How do GLP-1 receptor agonists impact LDL cholesterol?")
        picott = out["picott"]
        assert picott.get("type_of_question") != "background"
        anchors = out["anchors"]["primary_anchors"]
        assert anchors, "GLP-1 / LDL question must produce anchors"

    def test_milrinone_sah_still_works(self):
        out = _parse("What is the dose of milrinone for subarachnoid hemorrhage?")
        picott = out["picott"]
        assert picott.get("type_of_question") in {"dosing_protocol", "pharmacology"}
        anchors = out["anchors"]["primary_anchors"]
        assert any(
            a["entity_type"] == "drug_or_drug_class"
            for a in anchors
        )
        assert any(
            a["entity_type"] == "disease_or_problem"
            for a in anchors
        )

    def test_ami_hemoglobin_transfusion_still_works(self):
        out = _parse(
            "Should AMI patients be transfused at higher hemoglobin targets?"
        )
        picott = out["picott"]
        assert picott.get("type_of_question") != "background"
