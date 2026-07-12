"""Unit tests for the deterministic clinical-subject salvage.

Guards the fix for questions that previously dead-ended at
``awaiting_clinical_subject`` with zero articles because the LLM parser failed
to extract a usable clinical entity (e.g. "prognosis of a patient after
cerebellar hemorrhage" was reduced to "bleeding"). The salvage pulls the real
clinical noun phrase out of the raw question for ANY phrasing, while genuinely
vague questions still return None so the pipeline clarifies.

Pure / deterministic — no Mongo, no LLM, no network.
"""
import pytest

from services.topic_subject_rescue import build_topic_anchor, detect_topic_subject


class TestDetectTopicSubject:
    # (question, expected_topic)
    @pytest.mark.parametrize("q,expected", [
        # --- the reported failure: prognosis / "X after Y" ---
        ("what is the prognosis of a patient after cerebellar hemorrhage?", "cerebellar hemorrhage"),
        # --- benefit / effect / role of an intervention or supplement ---
        ("What is the benefite of creatine supplementation in adult males?", "creatine supplementation"),
        ("What is the benefit of exercise in the elderly?", "exercise"),
        ("What are the effects of vitamin D on bone health?", "vitamin D"),
        ("efficacy of omega-3 supplementation for depression", "omega-3 supplementation"),
        ("Does creatine improve muscle strength in athletes?", "creatine"),
        ("What is the value of probiotics for IBS?", "probiotics"),
        # --- general clinical-subject salvage across question types ---
        ("differential diagnosis of chest pain", "chest pain"),
        ("what causes acute kidney injury?", "acute kidney injury"),
        ("epidemiology of tuberculosis", "tuberculosis"),
        ("what is the treatment of sepsis?", "sepsis"),
        ("first-line treatment of type 2 diabetes", "type 2 diabetes"),
        ("complications after total knee replacement", "total knee replacement"),
        ("management of atrial fibrillation in older adults", "atrial fibrillation"),
        ("what is the pathophysiology of heart failure?", "heart failure"),
    ])
    def test_extracts_clinical_subject(self, q, expected):
        assert detect_topic_subject(q) == expected

    @pytest.mark.parametrize("q", [
        "What is the benefit of first-line treatment?",
        "What is the management?",
        "What are the recommendations?",
        "what is the best treatment?",
        "",
        "   ",
    ])
    def test_vague_questions_still_clarify(self, q):
        assert detect_topic_subject(q) is None


class TestBuildTopicAnchor:
    def test_disease_phrase_is_disease_problem(self):
        a = build_topic_anchor("cerebellar hemorrhage")
        assert a["entity_type"] == "disease_or_problem"
        assert a["role"] == "population"
        assert a["required_for_search"] is True

    def test_supplement_is_drug_class_intervention(self):
        a = build_topic_anchor("creatine supplementation")
        assert a["entity_type"] == "drug_or_drug_class"
        assert a["role"] == "intervention"
        assert "creatine" in a["synonyms"]

    def test_other_topic_is_exposure(self):
        a = build_topic_anchor("exercise")
        assert a["entity_type"] == "exposure"

    def test_shape_matches_primary_anchor_to_dict(self):
        a = build_topic_anchor("cerebellar hemorrhage")
        assert set(a) == {
            "text", "canonical_concept", "entity_type", "role", "synonyms",
            "source", "confidence", "required_for_search", "missing_relevance_cap",
        }
