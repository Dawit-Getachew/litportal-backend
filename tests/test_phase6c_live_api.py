"""Phase 6C live-integration tests against deployed REACT_APP_BACKEND_URL.

Validates Phase 6C contract end-to-end:
- POST /api/admin/benchmark-runs/run?execution_mode=mocked produces
  Phase 6C schema doc (latency_json, source_api_errors_json,
  recommendation_inputs_json, expansion_mode at top-level + per-case rows).
- GET /api/admin/benchmark-runs/{id}?redact=true enforces closed-set
  allowlist + _redaction_schema='phase6c_v1'.
- GET /api/admin/benchmark-runs?limit=5 (admin=200, clinician=403).
- GET /api/admin/benchmark-runs/recommendation returns
  GO/CONTINUE_SHADOW/NO_GO.
- POST /api/search-sessions/ with PHI-laden question is blocked AND
  session has active_expansion_applied=false +
  active_expansion_reason='phi_blocked'.
- POST /api/admin/benchmark-runs/run?execution_mode=live with
  case_ids=nonexistent must NOT crash; should still return a doc with
  deferred_cases (or empty ranked_cases + no error).
"""
from __future__ import annotations

import os
import time
import uuid
import requests
import pytest


# Phase 6F closeout (IMPL-4) — pin to ``live_api`` xdist group.
pytestmark = pytest.mark.xdist_group(name="live_api")

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "").rstrip("/")
if not BASE_URL:
    # Fallback: read frontend .env (testing-agent setup).
    try:
        with open("/app/frontend/.env") as fh:
            for line in fh:
                if line.startswith("REACT_APP_BACKEND_URL="):
                    BASE_URL = line.split("=", 1)[1].strip().rstrip("/")
    except Exception:
        pass
if not BASE_URL:
    # No deployed backend configured (e.g. this CI runner) -- skip the whole
    # module before any fixture makes a real HTTP call, instead of every
    # admin_token/clinician_token-dependent test erroring on a bad/unreachable
    # URL. Matches the guard already used in test_phase2b_live_api.py /
    # test_phase6f_live_api.py.
    pytest.skip(
        "REACT_APP_BACKEND_URL is not configured -- live API tests need a deployed backend",
        allow_module_level=True,
    )

ADMIN_EMAIL = "admin@test.com"
ADMIN_PW = "AdminTest123!"
CLINICIAN_EMAIL = "clinician@test.com"
CLINICIAN_PW = "ClinicianTest123!"

# Phase 6C top-level allowlist (mirrors benchmark_redaction._TOP_LEVEL_ALLOWLIST).
PHASE6C_TOP_ALLOWLIST = {
    "id", "schema_version", "created_at", "execution_mode", "expansion_mode",
    "mode", "three_mode", "comparison_report", "aggregate_metrics",
    "summary_metrics", "latency_json", "source_api_errors_json",
    "recommendation", "recommendation_inputs_json", "ranked_cases_count",
    "evaluated_at", "_redacted", "_redaction_schema",
}

# Forbidden raw text keys per Phase 6C closed-set allowlist.
FORBIDDEN_PER_CASE_KEYS = {
    "raw_abstract", "raw_phi", "raw_evidence_text", "uploaded_user_text",
    "secrets", "evidence_packet", "summary_text", "prompt",
}


# ----- fixtures ------------------------------------------------------------
@pytest.fixture(scope="module")
def admin_token():
    r = requests.post(f"{BASE_URL}/api/auth/login",
                      json={"email": ADMIN_EMAIL, "password": ADMIN_PW},
                      timeout=20)
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.fixture(scope="module")
def clinician_token():
    r = requests.post(f"{BASE_URL}/api/auth/login",
                      json={"email": CLINICIAN_EMAIL, "password": CLINICIAN_PW},
                      timeout=20)
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _h(tok):
    return {"Authorization": f"Bearer {tok}"}


# ----- A. Mocked sweep -> Phase 6C schema ---------------------------------
class TestMockedSweepSchema:
    """Verify POST /admin/benchmark-runs/run?execution_mode=mocked produces
    a Phase 6C v1 document."""

    @pytest.fixture(scope="class")
    def mocked_run(self, admin_token):
        # POST returns thin summary; fetch persisted doc by run_id.
        r = requests.post(
            f"{BASE_URL}/api/admin/benchmark-runs/run",
            params={"execution_mode": "mocked"},
            headers=_h(admin_token), timeout=180,
        )
        assert r.status_code == 200, r.text
        summary = r.json()
        run_id = summary.get("run_id") or summary.get("id")
        assert run_id, summary
        rg = requests.get(
            f"{BASE_URL}/api/admin/benchmark-runs/{run_id}",
            headers=_h(admin_token), timeout=30,
        )
        assert rg.status_code == 200, rg.text
        return rg.json()

    def test_top_level_phase6c_fields_present(self, mocked_run):
        for k in ("latency_json", "source_api_errors_json",
                  "recommendation_inputs_json", "expansion_mode"):
            assert k in mocked_run, f"missing top-level {k}; keys={list(mocked_run.keys())}"

    def test_comparison_report_schema_version(self, mocked_run):
        cr = mocked_run.get("comparison_report") or mocked_run.get("three_mode")
        assert cr is not None
        assert cr.get("schema_version") == "phase6c_v1", cr.get("schema_version")

    def test_per_case_rows_carry_phase6c_fields(self, mocked_run):
        cr = mocked_run.get("comparison_report") or mocked_run.get("three_mode")
        rows = cr.get("per_case_three_mode") or cr.get("per_case") or []
        assert len(rows) > 0
        row = rows[0]
        for k in ("top_10_baseline_json", "top_10_active_simulation_json",
                  "expansion_candidates_json", "promoted_candidates_json",
                  "demoted_candidates_json", "source_api_errors_json",
                  "latency_json"):
            assert k in row, f"missing per-case key {k} in row {row.keys()}"


# ----- B. Redacted JSON ----------------------------------------------------
class TestRedactedJson:
    @pytest.fixture(scope="class")
    def latest_run_id(self, admin_token):
        r = requests.get(f"{BASE_URL}/api/admin/benchmark-runs",
                         params={"limit": 5}, headers=_h(admin_token), timeout=20)
        assert r.status_code == 200, r.text
        payload = r.json()
        runs = payload.get("runs") if isinstance(payload, dict) else payload
        assert isinstance(runs, list) and len(runs) > 0
        return runs[0]["id"]

    def test_redacted_returns_schema_marker(self, admin_token, latest_run_id):
        r = requests.get(
            f"{BASE_URL}/api/admin/benchmark-runs/{latest_run_id}",
            params={"redact": "true"}, headers=_h(admin_token), timeout=20,
        )
        assert r.status_code == 200, r.text
        doc = r.json()
        assert doc.get("_redacted") is True
        assert doc.get("_redaction_schema") == "phase6c_v1"

    def test_redacted_top_level_keys_subset_of_allowlist(
            self, admin_token, latest_run_id):
        r = requests.get(
            f"{BASE_URL}/api/admin/benchmark-runs/{latest_run_id}",
            params={"redact": "true"}, headers=_h(admin_token), timeout=20,
        )
        doc = r.json()
        extra = set(doc.keys()) - PHASE6C_TOP_ALLOWLIST
        assert not extra, f"unexpected top-level keys leaked: {extra}"

    def test_redacted_per_case_no_raw_text_or_secrets(
            self, admin_token, latest_run_id):
        r = requests.get(
            f"{BASE_URL}/api/admin/benchmark-runs/{latest_run_id}",
            params={"redact": "true"}, headers=_h(admin_token), timeout=20,
        )
        doc = r.json()
        cr = doc.get("comparison_report") or doc.get("three_mode") or {}
        rows = cr.get("per_case_three_mode") or cr.get("per_case") or []
        for row in rows:
            leaked = set(row.keys()) & FORBIDDEN_PER_CASE_KEYS
            assert not leaked, f"forbidden keys leaked per-case: {leaked}"


# ----- C. Authz on list ---------------------------------------------------
class TestListAuthz:
    def test_admin_list_200(self, admin_token):
        r = requests.get(f"{BASE_URL}/api/admin/benchmark-runs",
                         params={"limit": 5}, headers=_h(admin_token), timeout=20)
        assert r.status_code == 200
        payload = r.json()
        runs = payload.get("runs") if isinstance(payload, dict) else payload
        assert isinstance(runs, list)

    def test_clinician_list_forbidden(self, clinician_token):
        r = requests.get(f"{BASE_URL}/api/admin/benchmark-runs",
                         params={"limit": 5}, headers=_h(clinician_token), timeout=20)
        assert r.status_code == 403


# ----- D. Recommendation endpoint -----------------------------------------
class TestRecommendation:
    def test_recommendation_returns_known_verdict(self, admin_token):
        r = requests.get(f"{BASE_URL}/api/admin/benchmark-runs/recommendation",
                         params={"limit": 5}, headers=_h(admin_token), timeout=30)
        assert r.status_code == 200, r.text
        data = r.json()
        verdict = data.get("verdict") or data.get("recommendation") or \
            (data.get("recommendation") or {}).get("verdict")
        assert verdict in ("GO", "CONTINUE_SHADOW", "NO_GO"), data

    def test_recommendation_not_forced_go_with_only_mocked(self, admin_token):
        """With only placeholder/mocked runs, must NOT be GO."""
        r = requests.get(f"{BASE_URL}/api/admin/benchmark-runs/recommendation",
                         params={"limit": 5}, headers=_h(admin_token), timeout=30)
        data = r.json()
        verdict = data.get("verdict") or data.get("recommendation") or \
            (data.get("recommendation") or {}).get("verdict")
        # CONTINUE_SHADOW or NO_GO are both acceptable, but never GO under mocked.
        assert verdict != "GO", f"Forced GO under mocked sweeps: {data}"


# ----- E. PHI-blocked session does NOT invoke expansion --------------------
class TestPhiBlocksExpansion:
    def test_phi_question_blocks_and_sets_expansion_metadata(self, clinician_token):
        # MRN+DOB+email PHI -> guard should block.
        body = {
            "question": (
                "Patient John Smith MRN 12345678 DOB 01/05/1972 "
                "email john.smith@hospital.org has SAH; "
                "does milrinone improve outcomes?"
            ),
        }
        r = requests.post(f"{BASE_URL}/api/search-sessions/", json=body,
                          headers=_h(clinician_token), timeout=30)
        assert r.status_code in (200, 201), r.text
        data = r.json()
        sid = data.get("id") or data.get("session_id")
        assert sid
        # Allow brief settle.
        time.sleep(1.5)
        rg = requests.get(f"{BASE_URL}/api/search-sessions/{sid}",
                          headers=_h(clinician_token), timeout=20)
        assert rg.status_code == 200, rg.text
        sess = rg.json()
        assert sess.get("status") == "blocked_phi", sess.get("status")
        assert sess.get("active_expansion_applied") is False
        assert sess.get("active_expansion_candidate_count") == 0
        assert sess.get("active_expansion_latency_ms") == 0
        assert sess.get("active_expansion_reason") == "phi_blocked"
        assert sess.get("scholar_like_expansion_json") in (None, {}, [])


# ----- F. Live sweep endpoint accepts params + no crash -------------------
class TestLiveSweepEndpoint:
    """Hit the live endpoint with a nonexistent case_id so we don't burn
    real PubMed quota; the endpoint must still respond gracefully."""

    def test_live_sweep_accepts_params_and_handles_unknown_case(
            self, admin_token):
        r = requests.post(
            f"{BASE_URL}/api/admin/benchmark-runs/run",
            params={
                "execution_mode": "live",
                "sleep_between_cases_s": 4,
                "case_ids": f"nonexistent_{uuid.uuid4().hex[:8]}",
            },
            headers=_h(admin_token), timeout=60,
        )
        # 200 (graceful empty/deferred result) or 400 (validation) acceptable;
        # MUST NOT 500.
        assert r.status_code in (200, 400, 404), \
            f"live endpoint crashed: {r.status_code} {r.text[:300]}"
        if r.status_code == 200:
            doc = r.json()
            # Per Phase 6C contract: doc may include deferred_cases.
            # If ranked_cases empty + no crash, that's the pass condition.
            assert "schema_version" in doc or "comparison_report" in doc or \
                "deferred_cases" in doc or "ranked_cases_count" in doc, doc
