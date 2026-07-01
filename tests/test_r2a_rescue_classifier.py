"""Phase R2A — Rejected-Article Rescue Classifier (SHADOW) tests.

Covers the deterministic layer: closed-set schema, validation gate
(confidence threshold, policy-exclusion override, central-disease
mismatch, no-identifier guard), policy-exclusion detection, central
disease tokens, the production shadow gate, and the orchestrator with a
stubbed LLM (no network).
"""
from __future__ import annotations

import asyncio

import services.rejected_article_rescue as r2a


# --- Closed-set model --------------------------------------------------

def test_model_accepts_closed_set():
    c = r2a.RescueClassification(
        rescue_label="direct_relevance_candidate",
        relevance_reason="stage_specific_prognosis",
        confidence=0.9, short_rationale="stage III survival",
    )
    assert c.rescue_label == "direct_relevance_candidate"


def test_model_rejects_unknown_label_and_extra_fields():
    import pytest
    with pytest.raises(Exception):
        r2a.RescueClassification(
            rescue_label="bogus", relevance_reason="survival_outcome",
            confidence=0.5,
        )
    with pytest.raises(Exception):
        r2a.RescueClassification(
            rescue_label="keep_rejected", relevance_reason="wrong_disease",
            confidence=0.5, pmid="123",  # extra forbidden
        )


def test_rationale_length_capped():
    import pytest
    with pytest.raises(Exception):
        r2a.RescueClassification(
            rescue_label="keep_rejected", relevance_reason="wrong_disease",
            confidence=0.5, short_rationale="x" * 241,
        )


# --- Validation gate ---------------------------------------------------

def test_validate_valid_rescue():
    c, rej = r2a.validate_classification(
        {"rescue_label": "direct_relevance_candidate",
         "relevance_reason": "survival_outcome", "confidence": 0.9,
         "short_rationale": "ok"},
        central_disease_present=True, policy_excluded_reason=None,
    )
    assert c.rescue_label == "direct_relevance_candidate"
    assert rej == []


def test_validate_central_disease_mismatch_downgrades():
    c, rej = r2a.validate_classification(
        {"rescue_label": "direct_relevance_candidate",
         "relevance_reason": "survival_outcome", "confidence": 0.9},
        central_disease_present=False, policy_excluded_reason=None,
    )
    assert c.rescue_label == "keep_rejected"
    assert c.relevance_reason == "wrong_disease"
    assert "central_disease_mismatch" in rej


def test_validate_policy_exclusion_override():
    c, rej = r2a.validate_classification(
        {"rescue_label": "direct_relevance_candidate",
         "relevance_reason": "survival_outcome", "confidence": 0.95},
        central_disease_present=True, policy_excluded_reason="retracted_policy",
    )
    assert c.rescue_label == "policy_excluded"
    assert c.relevance_reason == "retracted_policy"


def test_validate_below_confidence():
    c, rej = r2a.validate_classification(
        {"rescue_label": "borderline_relevance_candidate",
         "relevance_reason": "survival_outcome", "confidence": 0.3},
        central_disease_present=True, policy_excluded_reason=None,
    )
    assert c.rescue_label == "keep_rejected"
    assert "below_confidence_threshold" in rej


def test_validate_no_identifier_guard():
    c, rej = r2a.validate_classification(
        {"rescue_label": "direct_relevance_candidate",
         "relevance_reason": "survival_outcome", "confidence": 0.9,
         "pmid": "12345"},
        central_disease_present=True, policy_excluded_reason=None,
    )
    assert c is None
    assert "llm_emitted_identifier_or_score" in rej


def test_validate_schema_invalid_and_empty():
    c, rej = r2a.validate_classification(
        {"rescue_label": "nope", "relevance_reason": "x", "confidence": 0.9},
        central_disease_present=True, policy_excluded_reason=None,
    )
    assert c is None and "schema_invalid" in rej
    c2, rej2 = r2a.validate_classification(
        None, central_disease_present=True, policy_excluded_reason=None,
    )
    assert c2 is None and "empty_output" in rej2


# --- Policy-exclusion detection ---------------------------------------

def test_is_policy_excluded_retracted():
    assert r2a.is_policy_excluded({"is_retracted": True}, {}) == "retracted_policy"
    assert r2a.is_policy_excluded({}, {"cap_reasons": ["retracted"]}) == "retracted_policy"


def test_is_policy_excluded_preprint():
    assert r2a.is_policy_excluded({"is_preprint": True}, {}) == "preprint_policy"


def test_is_policy_excluded_none_for_normal_article():
    art = {"publication_types": ["Journal Article"], "is_retracted": False,
           "is_preprint": False}
    assert r2a.is_policy_excluded(art, {"cap_reasons": ["low_relevance"]}) is None


# --- Central disease tokens -------------------------------------------

def test_central_disease_tokens():
    toks = r2a.central_disease_tokens(
        {"clinical_problem": ["prostate cancer"],
         "required_anchors": ["prostate cancer"]})
    assert "prostate" in toks and "prostate cancer" in toks


def test_central_disease_present_check():
    toks = ["prostate cancer", "prostate"]
    assert r2a._central_disease_present(
        {"title": "Survival in localized prostate cancer"}, toks) is True
    assert r2a._central_disease_present(
        {"title": "Breast cancer screening"}, toks) is False
    # no tokens -> cannot block
    assert r2a._central_disease_present({"title": "x"}, []) is True


# --- Production shadow gate -------------------------------------------

def test_shadow_enabled_dev(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    assert r2a.r2a_shadow_enabled() is True


def test_shadow_disabled_in_prod_by_default(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("R2A_RESCUE_PROD_ENABLED", raising=False)
    assert r2a.r2a_shadow_enabled() is False


def test_shadow_enabled_in_prod_when_flagged(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("R2A_RESCUE_PROD_ENABLED", "true")
    assert r2a.r2a_shadow_enabled() is True


# --- Orchestrator with stubbed LLM (no network) -----------------------

def test_run_rescue_shadow_with_stub(monkeypatch):
    async def fake_complete_json(*, system, user, temperature=0.0, max_tokens=320, **k):
        # Rescue prostate-cancer survival articles; reject the breast one.
        if "prostate" in user.lower() and "survival" in user.lower():
            return {"rescue_label": "direct_relevance_candidate",
                    "relevance_reason": "survival_outcome", "confidence": 0.85,
                    "short_rationale": "stage-specific survival"}
        return {"rescue_label": "keep_rejected", "relevance_reason": "wrong_disease",
                "confidence": 0.8, "short_rationale": "off topic"}

    monkeypatch.setattr(r2a.llm_provider, "complete_json", fake_complete_json)
    monkeypatch.setattr(r2a.llm_provider, "is_configured", lambda: True)

    articles = {
        "a1": {"id": "a1", "pmid": "1", "title": "Overall survival in stage III prostate cancer",
               "abstract": "prostate cancer survival outcomes", "publication_year": 2023,
               "publication_types": ["Journal Article"]},
        "a2": {"id": "a2", "pmid": "2", "title": "Breast cancer screening trial",
               "abstract": "mammography", "publication_year": 2022,
               "publication_types": ["Journal Article"]},
        "a3": {"id": "a3", "pmid": "3", "title": "Retracted prostate survival study",
               "abstract": "prostate survival", "is_retracted": True,
               "publication_year": 2020, "publication_types": ["Journal Article"]},
    }
    rejected = [
        {"article_id": "a1", "overall_rank": 45, "exclusion_reason": None,
         "cap_reasons": ["low_relevance"]},
        {"article_id": "a2", "overall_rank": 30, "exclusion_reason": "low_relevance",
         "cap_reasons": []},
        {"article_id": "a3", "overall_rank": 20, "exclusion_reason": None,
         "cap_reasons": ["retracted"]},
    ]
    cqm = {"primary_intent": "prognosis_natural_history",
           "clinical_problem": ["prostate cancer"],
           "required_anchors": ["prostate cancer"]}

    out = asyncio.run(r2a.run_rescue_shadow(
        question="What is the prognosis of stage 3 prostate cancer?",
        cqm=cqm, rejected_rows=rejected, articles_by_id=articles,
    ))
    assert out["mode"] == "shadow"
    assert out["no_clinician_visible_change"] is True
    assert out["total_rejected_reviewed"] == 3
    assert out["direct_rescue_candidates"] == 1   # a1
    assert out["policy_excluded"] == 1            # a3 retracted (deterministic)
    assert out["keep_rejected"] == 1              # a2 breast
    # a1 is the top rescue candidate
    assert out["top_rescue_candidates"][0]["article_id"] == "a1"
    # output carries NO abstract / evidence text
    blob = str(out)
    assert "mammography" not in blob
    assert "survival outcomes" not in blob  # abstract text not leaked


def test_orchestrator_never_raises_on_bad_data():
    out = asyncio.run(r2a.run_rescue_shadow(
        question="", cqm={}, rejected_rows=[{"article_id": "missing"}],
        articles_by_id={},
    ))
    assert out["total_rejected_reviewed"] == 0
