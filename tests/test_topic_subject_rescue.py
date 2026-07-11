"""Unit tests for the deterministic topic subject rescue.

Guards the fix for supplement / nutrient / "benefit-of-<topic>" questions that
previously dead-ended at ``awaiting_clinical_subject`` with zero articles
(e.g. "What is the benefit of creatine supplementation in adult males?").
Pure / deterministic — no Mongo, no LLM, no network.
"""
from services.topic_subject_rescue import build_topic_anchor, detect_topic_subject


class TestDetectTopicSubject:
    def test_reported_creatine_question(self):
        q = "What is the benefite of creatine supplementation in adult males?"
        assert detect_topic_subject(q) == "creatine supplementation"

    def test_benefit_of_generic_intervention(self):
        assert detect_topic_subject("What is the benefit of exercise in the elderly?") == "exercise"

    def test_effect_of_supplement_strips_outcome_tail(self):
        assert detect_topic_subject("What are the effects of vitamin D on bone health?") == "vitamin D"

    def test_efficacy_framing_strips_population(self):
        assert (
            detect_topic_subject("efficacy of omega-3 supplementation for depression")
            == "omega-3 supplementation"
        )

    def test_bare_supplement_mention(self):
        assert detect_topic_subject("Does creatine improve muscle strength in athletes?") == "creatine"

    def test_value_of_probiotics(self):
        assert detect_topic_subject("What is the value of probiotics for IBS?") == "probiotics"

    # --- genuinely vague questions must still clarify (return None) ---
    def test_generic_treatment_still_clarifies(self):
        assert detect_topic_subject("What is the benefit of first-line treatment?") is None

    def test_bare_management_still_clarifies(self):
        assert detect_topic_subject("What is the management?") is None

    def test_recommendations_still_clarifies(self):
        assert detect_topic_subject("What are the recommendations?") is None

    def test_empty(self):
        assert detect_topic_subject("") is None
        assert detect_topic_subject("   ") is None


class TestBuildTopicAnchor:
    def test_supplement_is_drug_class_intervention(self):
        a = build_topic_anchor("creatine supplementation")
        assert a["entity_type"] == "drug_or_drug_class"
        assert a["role"] == "intervention"
        assert a["required_for_search"] is True
        # bare base term is also searched
        assert "creatine" in a["synonyms"]
        # shape matches PrimaryAnchor.to_dict()
        assert set(a) == {
            "text", "canonical_concept", "entity_type", "role", "synonyms",
            "source", "confidence", "required_for_search", "missing_relevance_cap",
        }

    def test_non_drug_topic_is_exposure(self):
        a = build_topic_anchor("exercise")
        assert a["entity_type"] == "exposure"
        assert a["role"] == "exposure"
