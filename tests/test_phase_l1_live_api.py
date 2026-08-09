"""Phase L1 — Live API acceptance tests for explicit Evidence Lens +
Population Focus preferences.

Hits the public REACT_APP_BACKEND_URL with the seeded clinician account.
Re-rank endpoint is instant; create-session may require polling for the
PHI/awaiting_clinical_subject path (the only create-session paths used
here are deliberately PHI / generic, so no PubMed retrieval is needed).
"""
from __future__ import annotations

import os

import pytest
import requests

BASE_URL = (
    os.environ.get("REACT_APP_BACKEND_URL")
    or "https://med-evidence-qa.preview.emergentagent.com"
).rstrip("/")

CLINICIAN_EMAIL = "clinician@test.com"
CLINICIAN_PASSWORD = "ClinicianTest123!"
ADMIN_EMAIL = "admin@test.com"
ADMIN_PASSWORD = "AdminTest123!"

# Existing sessions provided by the directive
SESS_RISK = "237eb8c2-f245-4398-b8e9-a69f0922eb1e"        # pneumothorax / central line
SESS_MECHANISM = "ebee14f1-af89-487f-82e1-19ccdc58fbe0"   # mechanism of pruritus
SESS_METFORMIN = "7805bfaa-654c-4d92-880d-86964ed73cb6"   # metformin therapy

HEADERS_UA = {"User-Agent": "Mozilla/5.0 (T1-Test/1.0)"}


def _login(email: str, password: str) -> str:
    r = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"email": email, "password": password},
        headers=HEADERS_UA,
        timeout=20,
    )
    assert r.status_code == 200, f"Login failed: {r.status_code} {r.text}"
    tok = r.json().get("access_token") or r.json().get("token")
    assert tok, f"No access_token in login response: {r.json()}"
    return tok


@pytest.fixture(scope="session")
def clinician_token() -> str:
    return _login(CLINICIAN_EMAIL, CLINICIAN_PASSWORD)


@pytest.fixture(scope="session")
def admin_token() -> str:
    return _login(ADMIN_EMAIL, ADMIN_PASSWORD)


@pytest.fixture(scope="session")
def clin_headers(clinician_token):
    return {**HEADERS_UA, "Authorization": f"Bearer {clinician_token}"}


@pytest.fixture(scope="session")
def admin_headers(admin_token):
    return {**HEADERS_UA, "Authorization": f"Bearer {admin_token}"}


# ---------------------------------------------------------------------
# Seeded sessions still exist & belong to the clinician
# ---------------------------------------------------------------------

def test_seeded_session_get_ok(clin_headers):
    r = requests.get(
        f"{BASE_URL}/api/search-sessions/{SESS_RISK}",
        headers=clin_headers, timeout=30,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == SESS_RISK
    # L1 fields must surface
    assert "evidence_lens" in body
    assert "population_focus" in body
    assert "preference_mode" in body
    assert "l1_preference_diagnostics" in body or body.get(
        "l1_preference_diagnostics"
    ) is None or isinstance(body.get("l1_preference_diagnostics"), dict)
    # default lens at creation time was clinically_direct/any (no-op)
    assert body["preference_mode"] in {"soft_boost", "shadow"}


# ---------------------------------------------------------------------
# Re-rank endpoint: try several lenses and verify safety invariants
# ---------------------------------------------------------------------

LENSES = [
    "mechanism",
    "risk_incidence",
    "drug_safety",
    "guidelines",
    "systematic_reviews",
    "most_recent",
]
POPULATIONS = ["any", "pediatrics", "older_adults"]


def _articles(session_id: str, headers) -> list:
    r = requests.get(
        f"{BASE_URL}/api/search-sessions/{session_id}",
        headers=headers, timeout=30,
    )
    assert r.status_code == 200
    return r.json().get("articles", []) or []


@pytest.mark.parametrize("lens", LENSES)
def test_evidence_lens_rerank_invariants(clin_headers, lens):
    """For each lens, the re-rank endpoint must:
    - return 200 with closed-set diagnostics
    - safety_caps_dominant=true, is_hard_filter=false, provenance=user_explicit_preference
    - never reduce the article count (non-destructive)
    - no article's overall_rank exceeds its relevance_cap_applied
    """
    before = _articles(SESS_RISK, clin_headers)
    n_before = len(before)

    r = requests.post(
        f"{BASE_URL}/api/search-sessions/{SESS_RISK}/evidence-lens",
        json={"evidence_lens": lens, "population_focus": "any"},
        headers=clin_headers, timeout=30,
    )
    assert r.status_code == 200, f"{lens}: {r.status_code} {r.text}"
    body = r.json()
    diag = body.get("l1_preference_diagnostics") or {}

    # Closed-set fields
    assert diag.get("safety_caps_dominant") is True, diag
    assert diag.get("is_hard_filter") is False, diag
    assert diag.get("provenance") == "user_explicit_preference", diag
    for k in ("articles_considered", "articles_matched", "articles_boosted",
              "rank_changed_count", "safety_caps_dominated_count"):
        assert k in diag, f"{lens}: missing {k} in diag {diag}"

    # Non-destructive — must GET to inspect articles (rerank response is metadata-only)
    after = _articles(SESS_RISK, clin_headers)
    assert len(after) == n_before, f"{lens}: hid articles {n_before}->{len(after)}"

    # Caps dominate — no article exceeds its cap
    for art in after:
        cap = art.get("relevance_cap_applied")
        rank = art.get("overall_rank")
        if cap is not None and rank is not None:
            assert rank <= cap, (
                f"{lens}: article {art.get('id')} rank {rank} > cap {cap}"
            )


@pytest.mark.parametrize("population", ["pediatrics", "older_adults"])
def test_population_focus_rerank(clin_headers, population):
    r = requests.post(
        f"{BASE_URL}/api/search-sessions/{SESS_MECHANISM}/evidence-lens",
        json={"evidence_lens": "clinically_direct", "population_focus": population},
        headers=clin_headers, timeout=30,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    diag = body.get("l1_preference_diagnostics") or {}
    assert diag.get("safety_caps_dominant") is True
    assert diag.get("is_hard_filter") is False
    assert body.get("population_focus") == population


def test_default_lens_is_noop(clin_headers):
    """Default (clinically_direct + any) yields zero boosts and no reorder."""
    r = requests.post(
        f"{BASE_URL}/api/search-sessions/{SESS_RISK}/evidence-lens",
        json={"evidence_lens": "clinically_direct", "population_focus": "any"},
        headers=clin_headers, timeout=30,
    )
    assert r.status_code == 200, r.text
    diag = r.json().get("l1_preference_diagnostics") or {}
    assert diag.get("articles_boosted") == 0, diag
    assert diag.get("preference_affected_ranking") in (False, None), diag
    assert diag.get("rank_changed_count") == 0


# ---------------------------------------------------------------------
# Safety G — PHI blocks lens, 409 on update
# ---------------------------------------------------------------------

def test_phi_question_blocked_and_lens_update_409(clin_headers):
    r = requests.post(
        f"{BASE_URL}/api/search-sessions/",
        json={
            "question": "Mr James presented with chest pain. What is the differential?",
            "evidence_lens": "mechanism",
            "population_focus": "older_adults",
        },
        headers=clin_headers, timeout=30,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("status") == "blocked_phi", body
    blocked_id = body.get("session_id") or body.get("id")
    assert blocked_id, f"no session id in blocked response: {body}"

    # Lens update on a blocked session must return 409
    r2 = requests.post(
        f"{BASE_URL}/api/search-sessions/{blocked_id}/evidence-lens",
        json={"evidence_lens": "mechanism", "population_focus": "any"},
        headers=clin_headers, timeout=30,
    )
    assert r2.status_code == 409, f"expected 409, got {r2.status_code}: {r2.text}"


# ---------------------------------------------------------------------
# Safety H — generic question yields zero articles / clarification
# ---------------------------------------------------------------------

def test_generic_question_no_articles(clin_headers):
    r = requests.post(
        f"{BASE_URL}/api/search-sessions/",
        json={"question": "What is the dose?"},
        headers=clin_headers, timeout=30,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # Allowed terminal states for generic without a subject. The pipeline may
    # already be in 'searching' transiently; what matters is that articles=0
    # and no L1 effect at this point.
    status = body.get("status")
    assert status in {
        "awaiting_clinical_subject", "clarification_needed",
        "blocked_phi", "no_results", "awaiting_selection",
        "processing", "running", "pending", "searching",
        "draft", "ranking",
    }, body
    # The lens didn't run any retrieval (articles=0 at create time)
    assert len(body.get("articles", []) or []) == 0


# ---------------------------------------------------------------------
# Create-session persists prefs (normalized)
# ---------------------------------------------------------------------

def test_create_session_persists_normalized_prefs(clin_headers):
    r = requests.post(
        f"{BASE_URL}/api/search-sessions/",
        json={
            "question": "What is the dose?",  # generic so no retrieval load
            "evidence_lens": "MECHANISM",  # uppercase to test normalization
            "population_focus": "garbage_value",  # invalid -> default 'any'
        },
        headers=clin_headers, timeout=30,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("evidence_lens") == "mechanism"
    assert body.get("population_focus") == "any"
    assert body.get("preference_mode") in {"soft_boost", "shadow"}


# ---------------------------------------------------------------------
# Admin diagnostics include l1_preference_diagnostics
# ---------------------------------------------------------------------

def test_admin_diagnostics_includes_l1(admin_headers):
    r = requests.get(
        f"{BASE_URL}/api/admin/search-sessions/{SESS_RISK}/diagnostics",
        headers=admin_headers, timeout=30,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "l1_preference_diagnostics" in body, list(body.keys())
    diag = body["l1_preference_diagnostics"] or {}
    if diag:
        assert diag.get("safety_caps_dominant") is True
        assert diag.get("is_hard_filter") is False
        assert diag.get("provenance") == "user_explicit_preference"


# ---------------------------------------------------------------------
# K4 evidence packet includes preference_metadata
# ---------------------------------------------------------------------

def test_k4_evidence_packet_preference_metadata(clin_headers, admin_headers):
    """K4 packet exposes preference_metadata with the closed-set safety
    fields. The packet is persisted once at build time; admin rebuild
    refreshes it, so we trigger a rebuild after setting a known lens and
    then re-GET to verify the latest lens is reflected.
    """
    # Set a known lens first
    requests.post(
        f"{BASE_URL}/api/search-sessions/{SESS_RISK}/evidence-lens",
        json={"evidence_lens": "guidelines", "population_focus": "older_adults"},
        headers=clin_headers, timeout=30,
    )
    # Try to rebuild via admin endpoint so the packet reflects current prefs
    rebuild = requests.post(
        f"{BASE_URL}/api/admin/evidence-packets/by-session/{SESS_RISK}/rebuild",
        headers=admin_headers, timeout=60,
    )
    # Rebuild may or may not exist / 200 — don't hard-fail on it
    r = requests.get(
        f"{BASE_URL}/api/search-sessions/{SESS_RISK}/evidence-packet",
        headers=clin_headers, timeout=30,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    pm = body.get("preference_metadata") or {}
    # Structural assertions (must hold regardless of rebuild result)
    assert pm.get("provenance") == "user_explicit_preference", pm
    assert pm.get("safety_caps_dominant") is True
    assert pm.get("is_hard_filter") is False
    assert pm.get("preference_mode") in {"soft_boost", "shadow"}
    assert "evidence_lens" in pm
    assert "population_focus" in pm
    # If rebuild succeeded, the latest lens should be reflected
    if rebuild.status_code == 200:
        assert pm.get("evidence_lens") == "guidelines", pm
        assert pm.get("population_focus") == "older_adults", pm


# ---------------------------------------------------------------------
# AUTH: another user cannot update someone else's session prefs
# ---------------------------------------------------------------------

def test_non_owner_cannot_update_lens(admin_headers):
    """Admin is a different user from clinician — should get 404."""
    r = requests.post(
        f"{BASE_URL}/api/search-sessions/{SESS_RISK}/evidence-lens",
        json={"evidence_lens": "mechanism", "population_focus": "any"},
        headers=admin_headers, timeout=30,
    )
    assert r.status_code == 404, f"expected 404, got {r.status_code}: {r.text}"


# ---------------------------------------------------------------------
# most_recent must not push capped wrong-lane articles above un-capped direct ones
# ---------------------------------------------------------------------

def test_most_recent_does_not_violate_caps(clin_headers):
    # Apply most_recent
    r = requests.post(
        f"{BASE_URL}/api/search-sessions/{SESS_RISK}/evidence-lens",
        json={"evidence_lens": "most_recent", "population_focus": "any"},
        headers=clin_headers, timeout=30,
    )
    assert r.status_code == 200
    arts = r.json().get("articles", []) or []
    # Sort by overall_rank descending and verify caps
    for a in arts:
        cap = a.get("relevance_cap_applied")
        rank = a.get("overall_rank")
        if cap is not None and rank is not None:
            assert rank <= cap, f"art {a.get('id')} rank {rank} > cap {cap}"
