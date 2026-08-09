"""Phase 6B.3 (collapsed final) — ontology morphology + AMI pack +
multi-intent retrieval + transparency cleanup contract tests.
"""
from __future__ import annotations

import asyncio

import pytest

from services.clinical_ontology_mapper import clinical_ontology_mapper
from services.interpretation_canonical import (
    canonical_label_for_rank_explanation,
)


# ---------------------------------------------------------------------------
# Helpers (reuse the pattern from earlier 6B tests).
# ---------------------------------------------------------------------------
async def _signup(client, email: str) -> dict:
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": "Pass1234567!",
              "name": email.split("@")[0]},
    )
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def _wait_ready(client, sid: str, headers: dict, max_iter: int = 200):
    last = None
    for _ in range(max_iter):
        await asyncio.sleep(0.2)
        sr = await client.get(
            f"/api/search-sessions/{sid}/status", headers=headers,
        )
        last = sr.json()
        if last.get("articles_ready") or last.get("status") == "error":
            return last
    return last


async def _disable_pubmed():
    from core.db import Collections, get_db
    await get_db()[Collections.SOURCE_CONFIGS].update_one(
        {"source_name": "pubmed"}, {"$set": {"enabled": False}},
    )


async def _restore_pubmed():
    from core.db import Collections, get_db
    await get_db()[Collections.SOURCE_CONFIGS].update_one(
        {"source_name": "pubmed"}, {"$set": {"enabled": True}},
    )


import pytest_asyncio  # noqa: E402


@pytest_asyncio.fixture(autouse=True)
async def _pubmed_state_guard():
    yield
    await _restore_pubmed()


# ---------------------------------------------------------------------------
# Task 1 — Ontology morphology
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "term,expected_substr,expected_role",
    [
        ("varicella vaccination", "varicella vaccine", "intervention"),
        ("varicella immunization", "varicella vaccine", "intervention"),
        ("glp1s", "GLP-1", "intervention"),
        ("ldl cholesterole", "LDL", "outcome"),
        ("steroids", "corticosteroid", "intervention"),
        ("varicella", "varicella", "population"),
    ],
)
def test_morphology_resolves_curated_variants(term, expected_substr, expected_role):
    m = clinical_ontology_mapper.map_term(term)
    assert m.canonical_concept, f"no canonical for {term!r}"
    assert expected_substr.lower() in (m.canonical_concept or "").lower(), m.to_dict()
    assert m.role_hint == expected_role, m.to_dict()
    assert m.confidence >= 0.7


def test_morphology_does_not_false_positive_random_text():
    m = clinical_ontology_mapper.map_term("foobar zxcvbn")
    assert m.confidence == 0.0
    assert m.canonical_concept == ""


def test_morphology_does_not_broaden_to_unlisted_drug_names():
    """Curated stem rules only — no broad fuzzy matching for drug names.
    ``ibupron`` (a non-existent typo of ibuprofen) must NOT silently
    map to ibuprofen."""
    m = clinical_ontology_mapper.map_term("ibupron 800")
    assert m.confidence < 0.85 or m.canonical_concept == "", m.to_dict()


def test_morphology_index_built_at_load():
    """Smoke: the pack index carries variants we know we added."""
    idx = clinical_ontology_mapper._pack_index
    # ``glp-1 receptor agonists`` (or one of its morphology variants)
    # MUST be in the index.
    assert any(
        ("glp" in k and "receptor agonist" in k) or k == "glp1s"
        for k in idx
    ), f"glp-1 morphology variant missing — sample keys: {list(idx)[:20]}"


# ---------------------------------------------------------------------------
# Task 2 — AMI pack + question_type_hint repair
# ---------------------------------------------------------------------------
def test_transfusion_threshold_question_type_hint_present():
    m = clinical_ontology_mapper.map_term("transfusion threshold")
    assert m.canonical_concept.lower() == "transfusion threshold"
    assert m.question_type_hint == "therapy"
    assert m.question_subtype_hint == "transfusion_threshold_comparison"
    assert m.role_hint == "intervention"


def test_canonical_label_picks_up_repaired_type_for_ami_session():
    picott = {"type_of_question": "therapy",
              "question_subtype": "transfusion_threshold_comparison"}
    label = canonical_label_for_rank_explanation(picott=picott)
    assert "Therapy" in label
    assert "transfusion threshold comparison" in label.lower()
    assert "other" != label.lower()


def test_morphology_resolves_high_low_hemoglobin_target_phrasing():
    """The AMI question phrasing "high hemoglobin target" / "low
    hemoglobin target" must hit the transfusion_threshold pack so the
    ontology_question_type_hint repair fires downstream."""
    for phrase in (
        "high hemoglobin target", "low hemoglobin target",
        "hemoglobin target", "transfusion threshold comparison",
    ):
        m = clinical_ontology_mapper.map_term(phrase)
        assert m.canonical_concept, f"no canonical for {phrase!r}"
        assert m.question_type_hint == "therapy", (phrase, m.to_dict())


# ---------------------------------------------------------------------------
# Task 3 — Multi-intent retrieval (mocked retrieval)
# ---------------------------------------------------------------------------
_EPMC_SCHEDULE = {
    "id": "AAAAAAAA",
    "source": "MED",
    "title": "Recommended varicella vaccination schedule in children",
    "abstractText": (
        "Two-dose varicella vaccine schedule is recommended at 12-15 months "
        "and 4-6 years. Immunization timing affects effectiveness."
    ),
    "journalInfo": {"journal": {"title": "Pediatrics"},
                     "yearOfPublication": "2024"},
    "pubYear": "2024",
    "firstPublicationDate": "2024-01-12",
    "doi": "10.1001/peds.2024.001",
    "pmid": "AAAAAAAA",
    "pmcid": "PMC0000001",
    "pubTypeList": {"pubType": ["Practice Guideline"]},
    "authorList": {"author": [{"fullName": "Doe J", "lastName": "Doe"}]},
    "fullTextUrlList": {"fullTextUrl": []},
    "inEPMC": "Y",
}
_EPMC_BENEFIT = {
    **_EPMC_SCHEDULE,
    "id": "BBBBBBBB",
    "title": "Benefit and effectiveness of varicella vaccine: long-term immunity",
    "abstractText": (
        "We assessed varicella vaccine effectiveness and long-term benefit "
        "with respect to chickenpox incidence in vaccinated children."
    ),
    "doi": "10.1001/peds.2024.002",
    "pmid": "BBBBBBBB",
    "pubTypeList": {"pubType": ["Systematic Review"]},
}


@pytest.mark.asyncio
async def test_multi_intent_session_tags_articles_with_subquestion_badges(
    client, monkeypatch,
):
    """Varicella schedule + benefit → 2 sub-questions. Articles must
    carry ``subquestion_ids_matched`` / ``article_subquestion_badge``."""
    await _disable_pubmed()
    headers = await _signup(client, "p6b3a@test.com")

    async def fake_retrieve(source, queries, **kwargs):
        if not queries:
            return []
        # Return BOTH the schedule + benefit records so the lane-balance
        # logic has candidates in each lane.
        return [
            ("cb_high_precision", _EPMC_SCHEDULE),
            ("cb_high_precision", _EPMC_BENEFIT),
        ]

    monkeypatch.setattr(
        "services.pipeline.literature_retrieval.retrieve", fake_retrieve,
    )
    r = await client.post(
        "/api/search-sessions/",
        json={"question": "What is the recommended age to get varicella "
                          "vaccination? What is its benefit?"},
        headers=headers,
    )
    sid = r.json()["id"]
    await _wait_ready(client, sid, headers)
    sess = (await client.get(
        f"/api/search-sessions/{sid}", headers=headers,
    )).json()
    mi = sess.get("multi_intent_json") or {}
    assert mi.get("is_multi_intent") is True, mi
    arts = sess.get("articles") or []
    assert arts, sess
    # At least one article carries a non-Related badge.
    badges = [a.get("article_subquestion_badge") for a in arts]
    assert any(b and b != "Related" for b in badges), badges
    # The matched ids list is populated.
    assert any(a.get("subquestion_ids_matched") for a in arts)


@pytest.mark.asyncio
async def test_single_intent_session_has_no_subquestion_tagging(
    client, monkeypatch,
):
    """Single-intent session: articles MUST NOT carry sub-question
    ids — the badge defaults to empty / unset and the matched-ids list
    is empty. Regression for the "single-intent unchanged" guarantee."""
    await _disable_pubmed()
    headers = await _signup(client, "p6b3b@test.com")

    async def fake_retrieve(source, queries, **kwargs):
        if not queries:
            return []
        return [("cb_high_precision", _EPMC_BENEFIT)]

    monkeypatch.setattr(
        "services.pipeline.literature_retrieval.retrieve", fake_retrieve,
    )
    r = await client.post(
        "/api/search-sessions/",
        json={"question": "What is the evidence for SGLT2 inhibitors in HFpEF?"},
        headers=headers,
    )
    sid = r.json()["id"]
    await _wait_ready(client, sid, headers)
    sess = (await client.get(
        f"/api/search-sessions/{sid}", headers=headers,
    )).json()
    mi = sess.get("multi_intent_json") or {}
    assert mi.get("is_multi_intent") is False or not mi, mi
    arts = sess.get("articles") or []
    # No article carries a sub-question id list with content.
    for a in arts:
        assert not (a.get("subquestion_ids_matched") or []), a
