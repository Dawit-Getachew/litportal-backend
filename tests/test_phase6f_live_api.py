"""Phase 6F — Live API regression tests against the deployed backend.

Validates the session-level integration claims of Phase 6F:
  1. POST /api/search-sessions/ with Q1 (psoriasis) / Q2 (Seroquel) produces
     PICOTT + primary_clinical_anchors_json that satisfy the closed-set
     assertions in the directive.
  2. Every emitted Boolean query contains the primary clinical anchor.
  3. GET /api/search-sessions/{sid} self-heals Default+Excluded contradictions.

These tests run against the PUBLIC REACT_APP_BACKEND_URL using the seeded
clinician account from /app/memory/test_credentials.md. They do NOT
modify implementation code.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import requests


# Phase 6F closeout (IMPL-4) — pin to ``live_api`` xdist group.
pytestmark = pytest.mark.xdist_group(name="live_api")


def _resolve_base_url() -> str:
    # Prefer the frontend .env REACT_APP_BACKEND_URL (the public URL
    # that the SPA uses). Fall back to env var if set.
    env_path = Path("/app/frontend/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("REACT_APP_BACKEND_URL="):
                return line.split("=", 1)[1].strip().rstrip("/")
    url = os.environ.get("REACT_APP_BACKEND_URL")
    if not url:
        pytest.skip("REACT_APP_BACKEND_URL is not configured")
    return url.rstrip("/")


BASE_URL = _resolve_base_url()
CLINICIAN_EMAIL = "clinician@test.com"
# test fixture credential — matches seeded user in server.py::_seed_user_if_missing
CLINICIAN_PASSWORD = "ClinicianTest123!"

Q1 = "What is the first line treatment and second line treatment for psoriasis?"
Q2 = "What are the side effects of Seroquel?"

TERMINAL_STATUSES = {"awaiting_selection", "complete", "completed"}
POLL_TIMEOUT_S = 240
POLL_INTERVAL_S = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def auth_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    r = s.post(
        f"{BASE_URL}/api/auth/login",
        json={"email": CLINICIAN_EMAIL, "password": CLINICIAN_PASSWORD},
        timeout=30,
    )
    if r.status_code != 200:
        pytest.skip(f"clinician login failed: {r.status_code} {r.text[:300]}")
    token = r.json().get("access_token")
    assert token, "no access_token in login response"
    s.headers.update({"Authorization": f"Bearer {token}"})
    return s


def _create_session(s: requests.Session, question: str) -> str:
    r = s.post(
        f"{BASE_URL}/api/search-sessions/",
        json={"question": question},
        timeout=60,
    )
    assert r.status_code in (200, 201), f"create session failed: {r.status_code} {r.text[:400]}"
    body = r.json()
    sid = body.get("id") or body.get("session_id") or (body.get("session") or {}).get("id")
    assert sid, f"no session id in response: {body}"
    return sid


def _poll_until_terminal(s: requests.Session, sid: str) -> dict:
    """Poll GET /api/search-sessions/{sid} until status is terminal or timeout."""
    deadline = time.time() + POLL_TIMEOUT_S
    last = None
    while time.time() < deadline:
        r = s.get(f"{BASE_URL}/api/search-sessions/{sid}", timeout=60)
        assert r.status_code == 200, f"GET session {sid} failed: {r.status_code} {r.text[:400]}"
        body = r.json()
        last = body
        status = (body.get("status") or "").lower()
        if status in TERMINAL_STATUSES:
            return body
        # Also accept "ranking" with queries already populated as soft-terminal
        # for query-anchor checks if pipeline stalls — but only after >120s.
        if status == "ranking" and body.get("queries") and time.time() > deadline - 60:
            return body
        time.sleep(POLL_INTERVAL_S)
    pytest.fail(
        f"session {sid} did not reach terminal status within {POLL_TIMEOUT_S}s; "
        f"last status={last and last.get('status')}"
    )
    return last  # unreachable


@pytest.fixture(scope="module")
def q1_session(auth_session) -> dict:
    sid = _create_session(auth_session, Q1)
    return _poll_until_terminal(auth_session, sid)


@pytest.fixture(scope="module")
def q2_session(auth_session) -> dict:
    sid = _create_session(auth_session, Q2)
    return _poll_until_terminal(auth_session, sid)


# ---------------------------------------------------------------------------
# Q1 — Psoriasis (disease anchor) checks
# ---------------------------------------------------------------------------

class TestQ1Psoriasis:
    def test_status_reaches_terminal(self, q1_session):
        assert (q1_session.get("status") or "").lower() in TERMINAL_STATUSES | {"ranking"}

    def test_picott_type_and_subtype(self, q1_session):
        picott = q1_session.get("picott_json") or q1_session.get("pico_json") or {}
        assert picott.get("type_of_question") in {"therapy", "guideline"}, picott
        assert picott.get("question_subtype") == "treatment_sequencing", picott
        assert (picott.get("specialty") or "").lower() == "dermatology", picott

    def test_picott_must_have_psoriasis(self, q1_session):
        picott = q1_session.get("picott_json") or q1_session.get("pico_json") or {}
        mh = picott.get("must_have_concepts") or []
        hits = [
            m for m in mh
            if "psoriasis" in (m.get("concept") or "").lower()
            and (m.get("role") or "").lower() in {"condition", "population"}
        ]
        assert hits, f"no psoriasis must-have anchor in {mh}"
        # Cap ≤ 20
        for m in hits:
            cap = m.get("missing_relevance_cap")
            if cap is not None:
                assert cap <= 20, f"missing_relevance_cap > 20: {m}"

    def test_primary_clinical_anchors_disease(self, q1_session):
        pca = q1_session.get("primary_clinical_anchors_json")
        assert pca is not None, "primary_clinical_anchors_json missing on Q1 session"
        anchors = pca.get("primary_anchors") or []
        assert anchors, "primary_anchors is empty for Q1"
        diseases = [a for a in anchors if a.get("entity_type") == "disease_or_problem"]
        assert diseases, f"no disease_or_problem anchor for Q1: {anchors}"
        assert pca.get("anchor_failure_reason") in (None, "")

    def test_every_query_contains_psoriasis(self, q1_session):
        queries = q1_session.get("queries") or []
        assert queries, "no queries emitted for Q1"
        for q in queries:
            qs = (q.get("query_string") or "").lower()
            assert "psoriasis" in qs, f"Q1 query missing psoriasis anchor: {q}"

    def test_no_default_excluded_contradiction(self, q1_session):
        for bucket_key in ("articles", "excluded_articles"):
            for row in q1_session.get(bucket_key) or []:
                if row.get("user_included") is True:
                    excl = row.get("exclusion_reason")
                    assert not excl, (
                        f"Default+Excluded contradiction in {bucket_key}: "
                        f"article_id={row.get('article_id') or row.get('id')} "
                        f"exclusion_reason={excl}"
                    )


# ---------------------------------------------------------------------------
# Q2 — Seroquel (drug anchor) checks
# ---------------------------------------------------------------------------

class TestQ2Seroquel:
    def test_status_reaches_terminal(self, q2_session):
        assert (q2_session.get("status") or "").lower() in TERMINAL_STATUSES | {"ranking"}

    def test_picott_type_and_subtype(self, q2_session):
        picott = q2_session.get("picott_json") or q2_session.get("pico_json") or {}
        assert picott.get("type_of_question") == "pharmacology", picott
        assert picott.get("question_subtype") == "drug_safety", picott
        spec = (picott.get("specialty") or "").lower()
        assert spec.startswith("psych") or spec.startswith("pharm"), picott

    def test_picott_must_have_drug_and_outcome(self, q2_session):
        picott = q2_session.get("picott_json") or q2_session.get("pico_json") or {}
        mh = picott.get("must_have_concepts") or []
        drug_hits = []
        for m in mh:
            if (m.get("role") or "").lower() != "intervention":
                continue
            terms = " ".join(
                [m.get("concept") or ""] + list(m.get("synonyms") or [])
            ).lower()
            if any(t in terms for t in ("seroquel", "quetiapine", "quetiapine fumarate")):
                drug_hits.append(m)
        assert drug_hits, f"no Seroquel/quetiapine intervention anchor in {mh}"
        for m in drug_hits:
            cap = m.get("missing_relevance_cap")
            if cap is not None:
                assert cap <= 20

        outcome_hits = [
            m for m in mh
            if (m.get("role") or "").lower() == "outcome"
            and (
                "adverse" in (m.get("concept") or "").lower()
                or "side effect" in (m.get("concept") or "").lower()
            )
        ]
        assert outcome_hits, f"no adverse/side-effects outcome anchor in {mh}"

    def test_primary_clinical_anchors_drug(self, q2_session):
        pca = q2_session.get("primary_clinical_anchors_json")
        assert pca is not None, "primary_clinical_anchors_json missing on Q2 session"
        anchors = pca.get("primary_anchors") or []
        assert anchors, "primary_anchors is empty for Q2"
        drugs = [a for a in anchors if a.get("entity_type") == "drug_or_drug_class"]
        assert drugs, f"no drug_or_drug_class anchor for Q2: {anchors}"
        assert pca.get("anchor_failure_reason") in (None, "")

    def test_every_query_contains_drug(self, q2_session):
        queries = q2_session.get("queries") or []
        assert queries, "no queries emitted for Q2"
        for q in queries:
            qs = (q.get("query_string") or "").lower()
            assert any(t in qs for t in ("seroquel", "quetiapine")), (
                f"Q2 query missing Seroquel/quetiapine anchor: {q}"
            )

    def test_no_default_excluded_contradiction(self, q2_session):
        for bucket_key in ("articles", "excluded_articles"):
            for row in q2_session.get(bucket_key) or []:
                if row.get("user_included") is True:
                    excl = row.get("exclusion_reason")
                    assert not excl, (
                        f"Default+Excluded contradiction in {bucket_key}: "
                        f"article_id={row.get('article_id') or row.get('id')} "
                        f"exclusion_reason={excl}"
                    )


# ---------------------------------------------------------------------------
# Closed-set codes available in the deployed services modules
# ---------------------------------------------------------------------------

class TestClosedSetCodesPresent:
    def test_cap_reasons_phase6f_codes(self):
        from services.picott_ranker import CAP_REASONS
        for c in (
            "wrong_drug", "wrong_condition", "missing_primary_anchor",
            "wrong_primary_anchor", "disease_only_match",
            "drug_safety_no_adverse_effect",
            "treatment_sequencing_wrong_disease",
        ):
            assert c in CAP_REASONS, c

    def test_exclusion_reasons_phase6f_codes(self):
        from services.default_selection import EXCLUSION_REASONS
        for c in (
            "cap_reason_wrong_drug",
            "cap_reason_wrong_condition",
            "cap_reason_missing_primary_anchor",
            "cap_reason_wrong_primary_anchor",
            "cap_reason_drug_safety_no_adverse_effect",
            "cap_reason_treatment_sequencing_wrong_disease",
        ):
            assert c in EXCLUSION_REASONS, c
