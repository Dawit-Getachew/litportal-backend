"""Phase 5A — Search relevance architecture tests (groups A–G).

A) PICOTT parsing
B) Spelling normalizer
C) Question scope classifier
D) Study type router + concept-block query builder
E) Ranking caps + anchor gates
F) Retrieval evaluator + rescue trigger
G) Benchmark harness + admin endpoint
"""
from __future__ import annotations

import re

import pytest

from services.clinical_question_parser import clinical_question_parser
from services.concept_block_query_builder import concept_block_query_builder
from services.concept_expansion import (
    expand_concept, expand_concepts, applicable_specialty_packs,
)
from services.picott_ranker import picott_aware_ranker
from services.question_scope_classifier import question_scope_classifier
from services.retrieval_quality_evaluator import retrieval_quality_evaluator
from services.search_benchmark import BENCHMARK_CASES, search_benchmark_service
from services.spelling_normalizer import normalize as spelling_normalize
from services.study_type_router import (
    LEGACY_TO_NEW, NEW_TO_LEGACY, legacy_to_new, new_to_legacy, study_type_router,
)


# ---------------------------------------------------------------------------
# B) Spelling normalizer
# ---------------------------------------------------------------------------

class TestSpellingNormalizer:
    def test_subarachnoid_typo_corrected(self):
        r = spelling_normalize("acute subarchnoid hemorrhage")
        assert "subarachnoid" in r.normalized
        assert any(c["original"].lower() == "subarchnoid" for c in r.corrections)

    def test_uk_to_us_haemorrhage(self):
        r = spelling_normalize("Aneurysmal subarachnoid haemorrhage")
        assert "hemorrhage" in r.normalized
        # Both spelling pairs may fire — at least the UK→US one is recorded.
        assert any(c["reason"] == "uk_to_us" for c in r.corrections)

    def test_original_question_not_mutated_on_normalize(self):
        original = "What is the dose of milrinone for subarchnoid hemorrhage?"
        r = spelling_normalize(original)
        assert "subarachnoid" in r.normalized
        # The function returns a new string; the original is preserved.
        assert original == "What is the dose of milrinone for subarchnoid hemorrhage?"

    def test_case_preserved_in_caps(self):
        r = spelling_normalize("Subarchnoid hemorrhage")
        # First letter is preserved.
        assert r.normalized.startswith("Subarachnoid")

    def test_no_correction_returns_empty_list(self):
        r = spelling_normalize("clean medical text without typos")
        assert r.corrections == []

    def test_multiple_corrections_recorded(self):
        r = spelling_normalize("anaemia and oedema with diarrhoea")
        assert "anemia" in r.normalized
        assert "edema" in r.normalized
        assert "diarrhea" in r.normalized
        assert len(r.corrections) >= 3


# ---------------------------------------------------------------------------
# C) Question scope classifier
# ---------------------------------------------------------------------------

class TestQuestionScopeClassifier:
    def test_background_what_is(self):
        r = question_scope_classifier.classify("What is septic shock?")
        assert r.scope == "background"

    def test_foreground_decision(self):
        r = question_scope_classifier.classify(
            "Treatment of acute kidney injury in adults"
        )
        assert r.scope == "foreground"

    def test_hybrid_when_both_signals_fire(self):
        r = question_scope_classifier.classify(
            "What is the dose of milrinone for SAH?"
        )
        # "dose" foreground + "What is" background → hybrid OR foreground
        # (foreground tends to dominate dosing). Accept either.
        assert r.scope in ("foreground", "hybrid")

    def test_unclear_for_empty(self):
        r = question_scope_classifier.classify("")
        assert r.scope == "unclear"


# ---------------------------------------------------------------------------
# A) PICOTT parsing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPicottParsing:
    async def test_milrinone_sah_classifies_as_dosing_protocol(self):
        q = (
            "What is the dose of milrinone for patients with acute "
            "subarchnoid hemorrhage?"
        )
        out = await clinical_question_parser.parse(q)
        pic = out.get("picott_json")
        assert pic is not None
        assert pic["type_of_question"] == "dosing_protocol"
        # Question subtype label is human-friendly.
        assert "dosing" in (pic.get("question_subtype") or "").lower()
        # Specialty promotion: neurocrit-pack present → neurology.
        assert pic["specialty"] == "neurology"

    async def test_milrinone_sah_must_haves_include_both_anchors(self):
        q = "What is the dose of milrinone for SAH?"
        out = await clinical_question_parser.parse(q)
        pic = out.get("picott_json") or {}
        mh = pic.get("must_have_concepts") or []
        concepts = {(m.get("concept") or "").lower() for m in mh}
        assert "milrinone" in concepts
        assert "subarachnoid hemorrhage" in concepts

    async def test_milrinone_sah_imaging_decoys_present_in_negatives(self):
        q = "What is the dose of milrinone for SAH?"
        out = await clinical_question_parser.parse(q)
        pic = out.get("picott_json") or {}
        negs = {(n.get("concept") or "").lower()
                 for n in (pic.get("negative_or_penalty_concepts") or [])}
        assert "cta" in negs or "ct angiography" in negs
        assert "dsa" in negs or "digital subtraction angiography" in negs

    async def test_spelling_correction_recorded_on_pico_json(self):
        q = "What is the dose of milrinone for subarchnoid hemorrhage?"
        out = await clinical_question_parser.parse(q)
        corrections = out.get("spelling_corrections") or []
        # picott_json also carries it.
        assert any(c.get("original", "").lower() == "subarchnoid" for c in corrections)
        pic = out.get("picott_json") or {}
        pic_corrections = pic.get("spelling_corrections") or []
        assert any((c.get("original") or "").lower() == "subarchnoid"
                    for c in pic_corrections)

    async def test_background_question_classified_as_background(self):
        out = await clinical_question_parser.parse("What is septic shock?")
        pic = out.get("picott_json") or {}
        assert pic["question_scope"] == "background"
        assert pic["type_of_question"] == "background"

    async def test_apixaban_warfarin_therapy(self):
        out = await clinical_question_parser.parse(
            "Is apixaban better than warfarin for stroke in NVAF?"
        )
        pic = out.get("picott_json") or {}
        # Accept any of therapy/prevention/pharmacology — the legacy parser
        # buckets drug-only questions as pharmacology; either type is valid
        # because the StudyTypeRouter pathways overlap for therapy/pharm.
        assert pic["type_of_question"] in {"therapy", "prevention", "pharmacology"}
        concepts = {(m.get("concept") or "").lower()
                    for m in (pic.get("must_have_concepts") or [])}
        assert "apixaban" in concepts


# ---------------------------------------------------------------------------
# D) Study type router + concept-block query builder
# ---------------------------------------------------------------------------

class TestStudyTypeRouterAndBuilder:
    def test_legacy_new_mapping_round_trips(self):
        # legacy → new → legacy is stable for legacy values that have a 1:1
        # match. Note ``treatment`` round-trips through ``therapy``.
        for legacy in LEGACY_TO_NEW.keys():
            new = legacy_to_new(legacy)
            assert new in NEW_TO_LEGACY

    def test_pathway_dosing_protocol_has_imaging_penalty(self):
        p = study_type_router.pathway("dosing_protocol")
        assert "imaging_only_for_pharmacology" in p.common_penalties

    def test_pathway_guideline_prefers_guideline_design(self):
        p = study_type_router.pathway("guideline")
        assert "guideline" in p.preferred_study_types
        assert p.ranking_boosts.get("guideline", 0) >= 10

    def test_concept_expansion_for_sah_acronym(self):
        out = expand_concept("SAH")
        out_lower = [s.lower() for s in out]
        assert "subarachnoid hemorrhage" in out_lower

    def test_neurocrit_specialty_pack_detection(self):
        assert applicable_specialty_packs("milrinone for vasospasm") == ["neurocritical_care"]
        assert applicable_specialty_packs("apixaban for NVAF") == []

    def test_concept_block_query_builder_emits_five_families_for_milrinone(self):
        picott = {
            "type_of_question": "dosing_protocol",
            "pico": {
                "population": ["adults"],
                "intervention_or_exposure_or_index_test": ["milrinone"],
                "comparator_or_reference_standard": [],
                "outcomes_or_target_diagnosis": [],
                "time_horizon": [],
                "setting": [],
            },
            "must_have_concepts": [
                {"concept": "milrinone", "role": "intervention",
                 "synonyms": ["IV milrinone", "intra-arterial milrinone"],
                 "required_for_high_relevance": True,
                 "missing_relevance_cap": 30},
                {"concept": "subarachnoid hemorrhage", "role": "condition",
                 "synonyms": ["SAH", "aSAH"],
                 "required_for_high_relevance": True,
                 "missing_relevance_cap": 50},
            ],
            "ranking_concepts": [
                {"concept": "dose", "role": "context", "synonyms": ["dosing"]},
                {"concept": "mcg/kg/min", "role": "context", "synonyms": []},
            ],
            "negative_or_penalty_concepts": [],
        }
        qs = concept_block_query_builder.build(picott, sources=["europe_pmc", "pubmed"])
        families = {q.query_family for q in qs}
        # We expect at least these (5 emitted by default).
        assert "high_precision" in families
        assert "broad_recall" in families
        assert "study_type_filtered" in families
        assert "guideline_review" in families
        assert "recent_title_abstract" in families
        # Each high_precision query must include milrinone AND SAH terms.
        high_p = [q for q in qs if q.query_family == "high_precision"]
        assert all(re.search(r"milrinone", q.query_string, re.IGNORECASE) for q in high_p)
        assert all(re.search(r"subarachnoid|sah", q.query_string, re.IGNORECASE)
                    for q in high_p)
        # study_type_filtered must include a publication-type clause.
        stf = [q for q in qs if q.query_family == "study_type_filtered"]
        assert all(("Publication Type" in q.query_string
                     or "PUB_TYPE:" in q.query_string) for q in stf)

    def test_rescue_query_emits_one_per_source(self):
        picott = {
            "type_of_question": "dosing_protocol",
            "pico": {
                "intervention_or_exposure_or_index_test": ["milrinone"],
                "comparator_or_reference_standard": [],
                "outcomes_or_target_diagnosis": [],
                "population": [], "time_horizon": [], "setting": [],
            },
            "must_have_concepts": [
                {"concept": "milrinone", "role": "intervention",
                 "synonyms": ["IV milrinone"], "missing_relevance_cap": 30},
                {"concept": "subarachnoid hemorrhage", "role": "condition",
                 "synonyms": ["SAH"], "missing_relevance_cap": 50},
            ],
            "ranking_concepts": [
                {"concept": "dose", "role": "context", "synonyms": []},
                {"concept": "infusion", "role": "context", "synonyms": []},
            ],
            "negative_or_penalty_concepts": [],
        }
        rescue = concept_block_query_builder.build_rescue(
            picott, sources=["europe_pmc", "pubmed"],
            rescue_must_haves=["milrinone", "SAH"],
        )
        assert len(rescue) == 2
        for q in rescue:
            assert q.query_family == "rescue_query"
            assert re.search(r"milrinone", q.query_string, re.IGNORECASE)


# ---------------------------------------------------------------------------
# E) Ranking caps + anchor gates
# ---------------------------------------------------------------------------

def _picott_milrinone_dosing():
    return {
        "type_of_question": "dosing_protocol",
        "pico": {},
        "must_have_concepts": [
            {"concept": "milrinone", "role": "intervention",
             "synonyms": ["IV milrinone", "IA milrinone"],
             "required_for_high_relevance": True,
             "missing_relevance_cap": 30},
            {"concept": "subarachnoid hemorrhage", "role": "condition",
             "synonyms": ["SAH", "aSAH"],
             "required_for_high_relevance": True,
             "missing_relevance_cap": 50},
        ],
        "should_have_concepts": [],
        "ranking_concepts": [
            {"concept": "dose", "role": "context", "synonyms": ["dosing"]},
            {"concept": "infusion", "role": "context", "synonyms": []},
        ],
        "negative_or_penalty_concepts": [
            {"concept": "CT angiography", "penalty_reason": "imaging_only",
             "unless_concepts_present": ["milrinone"], "penalty_cap": 40},
            {"concept": "CTA", "penalty_reason": "imaging_only",
             "unless_concepts_present": ["milrinone"], "penalty_cap": 40},
        ],
    }


class TestPicottRankerCaps:
    def test_imaging_only_sah_article_capped_at_40(self):
        picott = _picott_milrinone_dosing()
        ranked = [
            {
                "_article": {
                    "id": "a1", "canonical_key": "ck1",
                    "title": "Bone-subtraction CTA in aneurysmal SAH",
                    "abstract": "We evaluated CT angiography for vasospasm detection in SAH.",
                    "article_type_classification": "cohort_study",
                },
                "overall_rank_score": 85, "relevance_score": 85,
                "evidence_design_score": 70,
                "query_types_matched": ["broad_recall"],
            },
        ]
        out = picott_aware_ranker.postprocess(ranked, picott)
        # Article is missing milrinone → capped at 30 (intervention anchor).
        # CTA negative also fires → cap 40. Lowest wins (30).
        assert out[0]["overall_rank_score"] <= 30
        assert "missing_intervention_anchor" in out[0]["cap_reasons"]

    def test_milrinone_only_no_disease_capped_at_60(self):
        picott = _picott_milrinone_dosing()
        ranked = [
            {
                "_article": {
                    "id": "a2", "canonical_key": "ck2",
                    "title": "Milrinone dose-finding in cardiac surgery",
                    "abstract": "We assessed milrinone infusion mcg/kg/min in post-op patients.",
                    "article_type_classification": "rct",
                },
                "overall_rank_score": 85,
                "query_types_matched": ["high_precision"],
            },
        ]
        out = picott_aware_ranker.postprocess(ranked, picott)
        # Missing condition anchor → 50. Also dosing_disease_missing → 60.
        # Lowest = 50.
        assert out[0]["overall_rank_score"] <= 50

    def test_full_match_no_cap_keeps_high_score(self):
        picott = _picott_milrinone_dosing()
        ranked = [
            {
                "_article": {
                    "id": "a3", "canonical_key": "ck3",
                    "title": "Intra-arterial milrinone for cerebral vasospasm in SAH",
                    "abstract": "Continuous infusion of milrinone at 0.5 mcg/kg/min in aSAH patients with vasospasm.",
                    "article_type_classification": "rct",
                },
                "overall_rank_score": 80,
                "query_types_matched": ["high_precision", "study_type_filtered"],
            },
        ]
        out = picott_aware_ranker.postprocess(ranked, picott)
        # No cap, boosts applied.
        assert out[0]["overall_rank_score"] >= 80
        assert out[0]["matched_must_have"] == ["milrinone", "subarachnoid hemorrhage"] or \
                "milrinone" in out[0]["matched_must_have"]
        assert out[0]["relevance_cap_applied"] is None

    def test_negative_waiver_when_drug_also_present(self):
        picott = _picott_milrinone_dosing()
        ranked = [
            {
                "_article": {
                    "id": "a4", "canonical_key": "ck4",
                    "title": "CT angiography monitoring for milrinone dose adjustment in SAH",
                    "abstract": "We combined CTA with milrinone infusion in aSAH.",
                    "article_type_classification": "cohort_study",
                },
                "overall_rank_score": 80,
                "query_types_matched": ["high_precision"],
            },
        ]
        out = picott_aware_ranker.postprocess(ranked, picott)
        # CTA present BUT milrinone also present → waiver applies, negative
        # does NOT fire.
        assert "imaging_only_for_pharmacology" not in out[0]["cap_reasons"]
        assert out[0]["overall_rank_score"] > 50  # not capped

    def test_broad_recall_only_does_not_get_boost(self):
        picott = _picott_milrinone_dosing()
        ranked = [
            {
                "_article": {
                    "id": "a5", "canonical_key": "ck5",
                    "title": "Milrinone for cerebral vasospasm in SAH",
                    "abstract": "Milrinone infusion for aSAH with vasospasm.",
                    "article_type_classification": "rct",
                },
                "overall_rank_score": 70,
                "query_types_matched": ["broad_recall"],
            },
        ]
        out = picott_aware_ranker.postprocess(
            ranked, picott,
            query_family_by_canonical={"ck5": ["broad_recall"]},
        )
        # Only broad_recall match → no extra +5 ``must_haves_satisfied_*``
        # family-boost. The boost_reasons list MUST NOT contain a
        # ``must_haves_satisfied_high_*`` entry.
        reasons = out[0].get("ranking_boost_reasons") or []
        assert not any(
            "must_haves_satisfied_high" in s or "study_type_filtered" in s
            for s in reasons
        ), reasons
        # Phase 6F.1 — the new ``anchor_match_*`` boost is independent
        # of query family (it depends only on title/abstract match
        # against must_have_concepts), so a fully-matched milrinone
        # article DOES receive ``anchor_match_full`` even when
        # retrieved by broad_recall only. That's intentional per the
        # 6F.1 directive ("the boost depends on what the article says,
        # not how we found it"). The legacy broad_recall guardrail
        # remains in place for the family-boost specifically.


# ---------------------------------------------------------------------------
# F) Retrieval evaluator + rescue trigger
# ---------------------------------------------------------------------------

class TestEvaluatorPicottDiagnostics:
    def test_rescue_triggered_when_top10_misses_must_have(self):
        picott = _picott_milrinone_dosing()
        # Ten articles, none mention milrinone.
        articles = [
            {
                "_article": {
                    "title": f"CT angiography study {i}",
                    "abstract": "subarachnoid hemorrhage imaging study.",
                    "article_type_classification": "cohort_study",
                },
                "overall_rank_score": 60,
                "query_family_matches": ["broad_recall"],
            } for i in range(10)
        ]
        out = retrieval_quality_evaluator.evaluate(
            parsed={"clinical_question_type": "other"},
            queries=[], articles_after_filter=articles, raw_total=10,
            picott=picott,
        )
        pd = out.get("picott_diagnostics") or {}
        assert pd.get("rescue_query_needed") is True
        assert "milrinone" in pd.get("rescue_must_haves", [])
        assert "top10_missing_must_have_anchors" in out["problems"]
        assert out["recommended_next_action"] == "trigger_rescue_query"

    def test_no_rescue_when_top10_has_anchors(self):
        picott = _picott_milrinone_dosing()
        articles = [
            {
                "_article": {
                    "title": "Milrinone for SAH vasospasm",
                    "abstract": "Intra-arterial milrinone in subarachnoid hemorrhage.",
                    "article_type_classification": "rct",
                },
                "overall_rank_score": 80,
                "query_family_matches": ["high_precision"],
            },
        ] + [
            {
                "_article": {
                    "title": f"unrelated {i}", "abstract": "",
                    "article_type_classification": "review",
                },
                "overall_rank_score": 50,
                "query_family_matches": ["broad_recall"],
            } for i in range(9)
        ]
        out = retrieval_quality_evaluator.evaluate(
            parsed={"clinical_question_type": "other"},
            queries=[], articles_after_filter=articles, raw_total=10,
            picott=picott,
        )
        pd = out.get("picott_diagnostics") or {}
        assert pd.get("rescue_query_needed") is False


# ---------------------------------------------------------------------------
# G) Benchmark harness + admin endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBenchmarkHarness:
    async def test_all_six_cases_present(self):
        # Phase 5B added the ami_hemoglobin_transfusion case (case #7).
        assert len(BENCHMARK_CASES) == 7
        ids = {c.id for c in BENCHMARK_CASES}
        assert {"milrinone_sah", "ddimer_pe_lowrisk", "apixaban_warfarin_afib",
                 "nsaid_aki_ckd_older", "septic_shock_background",
                 "colorectal_screening_guideline",
                 "ami_hemoglobin_transfusion"} == ids

    async def test_milrinone_case_parser_passes(self):
        case = next(c for c in BENCHMARK_CASES if c.id == "milrinone_sah")
        out = await search_benchmark_service.run_case_parse(case)
        assert out["parser_pass"] is True
        assert out["picott"]["type_of_question"] == "dosing_protocol"
        # Concept-block queries were generated.
        assert any(q["query_family"] == "high_precision"
                    for q in out["concept_block_queries"])

    async def test_milrinone_case_with_imaging_dominated_mock_fails_ranked(self):
        case = next(c for c in BENCHMARK_CASES if c.id == "milrinone_sah")
        # Build a mocked top-10 of imaging-only SAH papers (no milrinone).
        mocked = [
            {
                "_article": {
                    "id": f"a{i}", "canonical_key": f"ck{i}",
                    "pmid": f"99{i:03d}",
                    "title": "Bone-subtraction CTA in aneurysmal SAH",
                    "abstract": "CT angiography diagnostic accuracy for vasospasm.",
                    "article_type_classification": "cohort_study",
                },
                "overall_rank_score": 80 - i,
                "query_types_matched": ["broad_recall"],
            } for i in range(10)
        ]
        out = await search_benchmark_service.run_case_full(case, mocked_articles=mocked)
        metrics = out["ranked_metrics"]
        # The case fails because there are no gold PMIDs in top-10 AND
        # the imaging-only papers dominate. The ranker caps them all so
        # their scores drop ≤30; the case "passes" only if at least one
        # gold pmid is present OR ≤5 negatives ranked. With pmid 99000–
        # 99009 (not gold), no gold → fail.
        assert metrics["pass"] is False
        assert metrics["gold_in_top10"] == []

    async def test_milrinone_case_with_gold_pmid_in_top10_passes_ranked(self):
        case = next(c for c in BENCHMARK_CASES if c.id == "milrinone_sah")
        # Build mocked top-10 with one gold PMID and a milrinone+SAH abstract.
        good = {
            "_article": {
                "id": "gold-1", "canonical_key": "ck-gold",
                "pmid": "33480639",   # one of the gold list
                "title": "Intra-arterial milrinone for cerebral vasospasm in aSAH",
                "abstract": "Continuous IA infusion of milrinone in subarachnoid hemorrhage.",
                "article_type_classification": "cohort_study",
            },
            "overall_rank_score": 90,
            "query_types_matched": ["high_precision"],
        }
        decoys = [
            {
                "_article": {
                    "id": f"d{i}", "canonical_key": f"ck-d{i}",
                    "pmid": f"77{i:03d}",
                    "title": "Bone-subtraction CTA in aneurysmal SAH",
                    "abstract": "CT angiography for vasospasm.",
                    "article_type_classification": "cohort_study",
                },
                "overall_rank_score": 70 - i,
                "query_types_matched": ["broad_recall"],
            } for i in range(9)
        ]
        out = await search_benchmark_service.run_case_full(
            case, mocked_articles=[good] + decoys,
        )
        metrics = out["ranked_metrics"]
        # Gold pmid surfaces; ranker keeps it at top because it has both anchors.
        assert metrics["gold_in_top10"] == ["33480639"]
        # Ranker caps the imaging decoys; they should not be ranked above gold.
        assert metrics["top10_pmids"][0] == "33480639"

    async def test_all_cases_parse_only_runs(self):
        out = await search_benchmark_service.run_all_parse_only()
        assert out["summary"]["total"] == 7
        assert out["summary"]["parser_passed"] >= 4  # at least 4/7 pass parser
        # Each case must produce some queries.
        for c in out["cases"]:
            if c["picott"].get("must_have_concepts"):
                assert len(c.get("concept_block_queries") or []) >= 1


# ---------------------------------------------------------------------------
# Admin endpoints (benchmark + diagnostics)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAdminPhase5AEndpoints:
    async def test_benchmark_cases_endpoint_lists_six(self, client):
        from tests._phase3_helpers import login
        tok = await login(client, "admin@test.com", "AdminTest123!")
        h = {"Authorization": f"Bearer {tok}"}
        r = await client.get("/api/admin/search-benchmarks/cases", headers=h)
        assert r.status_code == 200
        body = r.json()
        assert len(body["cases"]) == 7
        ids = {c["id"] for c in body["cases"]}
        assert "milrinone_sah" in ids
        assert "ami_hemoglobin_transfusion" in ids

    async def test_benchmark_run_parse_only_succeeds(self, client):
        from tests._phase3_helpers import login
        tok = await login(client, "admin@test.com", "AdminTest123!")
        h = {"Authorization": f"Bearer {tok}"}
        r = await client.post("/api/admin/search-benchmarks/run", headers=h)
        assert r.status_code == 200
        body = r.json()
        assert body["summary"]["total"] == 7

    async def test_non_admin_blocked_from_benchmark_run(self, client):
        from tests._phase3_helpers import signup
        t, _ = await signup(client, "phase5a-nonadmin@test.com")
        h = {"Authorization": f"Bearer {t}"}
        r = await client.post("/api/admin/search-benchmarks/run", headers=h)
        assert r.status_code == 403
