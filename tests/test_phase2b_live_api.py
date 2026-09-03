"""Live API integration tests for Phase 2B multi-source retrieval.

Runs against the deployed backend at REACT_APP_BACKEND_URL using the public
`/api` prefix. Validates admin endpoints, source-config toggling, PHI guard
across all sources, multi-source search planning, and transparency output.
"""
from __future__ import annotations

import os
import time
import uuid

import pytest
import requests


# Phase 5D Fix 3 — when the dev env enables DEV_FORCE_ENABLE_PUBMED the
# admin PATCH endpoint blocks disabling PubMed (HTTP 409). Tests that
# explicitly disable PubMed pre-date that guard, so we skip them when
# the env var is set. The guard itself has its own coverage in
# ``test_phase5d_round3_fixes.py``.
_DEV_FORCE_ENABLE_PUBMED = (
    os.environ.get("DEV_FORCE_ENABLE_PUBMED", "false").lower() == "true"
)
skip_if_pubmed_locked = pytest.mark.skipif(
    _DEV_FORCE_ENABLE_PUBMED,
    reason=(
        "DEV_FORCE_ENABLE_PUBMED=true — admin PATCH endpoint refuses to "
        "disable PubMed in this env (see Phase 5D Fix 3 guard). "
        "Toggle behavior is covered separately."
    ),
)


def _resolve_base_url() -> str:
    url = os.environ.get("REACT_APP_BACKEND_URL")
    if url:
        return url.rstrip("/")
    # Fall back to reading frontend/.env
    env_path = "/app/frontend/.env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("REACT_APP_BACKEND_URL"):
                    return line.split("=", 1)[1].strip().strip('"').rstrip("/")
    pytest.skip(
        "REACT_APP_BACKEND_URL not configured — live API tests need a "
        "deployed backend",
        allow_module_level=True,
    )


BASE_URL = _resolve_base_url()
API = f"{BASE_URL}/api"

ADMIN_EMAIL = "admin@test.com"
ADMIN_PASS = "AdminTest123!"
CLIN_EMAIL = "clinician@test.com"
CLIN_PASS = "ClinicianTest123!"


# ---------- helpers / fixtures ----------

def _login(email: str, password: str) -> str:
    r = requests.post(
        f"{API}/auth/login",
        json={"email": email, "password": password},
        timeout=15,
    )
    assert r.status_code == 200, f"login failed: {r.status_code} {r.text}"
    data = r.json()
    assert "access_token" in data or "token" in data, data
    return data.get("access_token") or data.get("token")


@pytest.fixture(scope="module")
def admin_token() -> str:
    return _login(ADMIN_EMAIL, ADMIN_PASS)


@pytest.fixture(scope="module")
def clinician_token() -> str:
    return _login(CLIN_EMAIL, CLIN_PASS)


@pytest.fixture(scope="module")
def admin_headers(admin_token) -> dict:
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="module")
def clin_headers(clinician_token) -> dict:
    return {"Authorization": f"Bearer {clinician_token}"}


def _wait_for_queries(headers: dict, sid: str, timeout_s: float = 30.0) -> dict:
    """Poll transparency until at least one query is recorded or timeout."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        r = requests.get(f"{API}/search-sessions/{sid}/transparency", headers=headers, timeout=15)
        if r.status_code == 200:
            last = r.json()
            if last.get("queries"):
                return last
        time.sleep(1.0)
    return last or {}


# ---------- Admin auth ----------

class TestAdminAuth:
    def test_admin_login_returns_token(self, admin_token):
        assert isinstance(admin_token, str) and len(admin_token) > 20

    def test_clinician_login_returns_token(self, clinician_token):
        assert isinstance(clinician_token, str) and len(clinician_token) > 20


# ---------- Admin dashboard ----------

class TestAdminDashboard:
    def test_dashboard_shape(self, admin_headers):
        r = requests.get(f"{API}/admin/dashboard", headers=admin_headers, timeout=15)
        assert r.status_code == 200, r.text
        d = r.json()
        # Required Phase 2B fields
        assert "source_health" in d and isinstance(d["source_health"], list)
        assert "source_configs" in d and isinstance(d["source_configs"], dict)
        for k in ("total", "enabled", "disabled"):
            assert k in d["source_configs"]
        assert "phi_blocked_submissions" in d
        assert "source_query_errors" in d
        assert "recent_audit_events" in d and isinstance(d["recent_audit_events"], list)

    def test_dashboard_forbidden_for_clinician(self, clin_headers):
        r = requests.get(f"{API}/admin/dashboard", headers=clin_headers, timeout=15)
        assert r.status_code == 403, r.text


# ---------- Admin source-configs list ----------

class TestSourceConfigsList:
    def test_returns_sources_no_api_keys_leaked(self, admin_headers):
        r = requests.get(f"{API}/admin/source-configs", headers=admin_headers, timeout=15)
        assert r.status_code == 200, r.text
        data = r.json()
        assert isinstance(data, list)
        # Spec says 16 sources scaffolded
        assert len(data) >= 16, f"expected >=16 source configs, got {len(data)}"
        for row in data:
            # Must never leak API key material — but `requires_api_key` is a
            # boolean metadata flag, not a secret. Block only secret-bearing
            # fields like `api_key`, `secret`, `auth_token`.
            for k, v in row.items():
                low = k.lower()
                if low in {"requires_api_key", "has_api_key"}:
                    continue
                if "api_key" in low or "secret" in low or low.endswith("_token"):
                    assert v in (None, "", False), f"source-configs leaked secret-bearing key {k}={v!r}"
            # Must have stable shape
            assert "id" in row
            assert "source_name" in row
            assert "enabled" in row

    def test_clinician_cannot_list_source_configs(self, clin_headers):
        r = requests.get(f"{API}/admin/source-configs", headers=clin_headers, timeout=15)
        assert r.status_code == 403


# ---------- PATCH source-configs (admin only) ----------

class TestPatchSourceConfig:
    def test_patch_rejects_non_admin(self, admin_headers, clin_headers):
        # Find pubmed config id with admin
        r = requests.get(f"{API}/admin/source-configs", headers=admin_headers, timeout=15)
        cfgs = r.json()
        pubmed = next((c for c in cfgs if c["source_name"] == "pubmed"), None)
        assert pubmed is not None, "pubmed source config missing"
        # Clinician PATCH attempt
        r2 = requests.patch(
            f"{API}/admin/source-configs/{pubmed['id']}",
            headers=clin_headers,
            json={"enabled": True},
            timeout=15,
        )
        assert r2.status_code == 403, r2.text

    @skip_if_pubmed_locked
    def test_patch_toggles_pubmed_and_logs_audit(self, admin_headers):
        r = requests.get(f"{API}/admin/source-configs", headers=admin_headers, timeout=15)
        cfgs = r.json()
        pubmed = next(c for c in cfgs if c["source_name"] == "pubmed")
        original = bool(pubmed["enabled"])
        target = not original

        r2 = requests.patch(
            f"{API}/admin/source-configs/{pubmed['id']}",
            headers=admin_headers,
            json={"enabled": target},
            timeout=15,
        )
        assert r2.status_code == 200, r2.text
        updated = r2.json()
        assert updated["enabled"] is target
        assert updated["source_name"] == "pubmed"
        # Should NOT leak api_key secrets (requires_api_key flag is OK)
        for k, v in updated.items():
            low = k.lower()
            if low in {"requires_api_key", "has_api_key"}:
                continue
            if "api_key" in low or "secret" in low or low.endswith("_token"):
                assert v in (None, "", False), f"PATCH response leaked {k}={v!r}"

        # Audit event recorded
        ar = requests.get(
            f"{API}/admin/audit-events?event_type=source_config_updated&limit=10",
            headers=admin_headers,
            timeout=15,
        )
        assert ar.status_code == 200
        events = ar.json()
        assert any(
            (e.get("event_payload_json") or e.get("payload") or {}).get("source_config_id") == pubmed["id"]
            for e in events
        ), f"source_config_updated audit not found in {len(events)} events: {events[:2]}"

        # Restore for downstream tests: leave it set to whatever we toggled to
        # (tests below assume we explicitly enable). We restore at teardown.
        # Save state on the test class for later restoration.
        TestPatchSourceConfig._pubmed_id = pubmed["id"]
        TestPatchSourceConfig._pubmed_original = original


# ---------- Admin cleanup ----------

class TestAdminCleanup:
    def test_cleanup_returns_deleted_count_and_audits(self, admin_headers):
        r = requests.post(f"{API}/admin/cleanup", headers=admin_headers, timeout=30)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "deleted_count" in data or "expired_sessions_deleted" in data, data
        # Verify audit event
        ar = requests.get(
            f"{API}/admin/audit-events?event_type=admin_cleanup_triggered&limit=5",
            headers=admin_headers,
            timeout=15,
        )
        assert ar.status_code == 200
        events = ar.json()
        assert len(events) >= 1, "admin_cleanup_triggered audit not found"


# ---------- PHI guard blocks all sources ----------

class TestPhiGuardBlocksAllSources:
    def test_phi_question_blocked_no_retrieval(self, clin_headers, admin_headers):
        # Make sure pubmed is enabled so that "all sources" path is exercised
        cfgs = requests.get(f"{API}/admin/source-configs", headers=admin_headers, timeout=15).json()
        pubmed = next(c for c in cfgs if c["source_name"] == "pubmed")
        if not pubmed["enabled"]:
            requests.patch(
                f"{API}/admin/source-configs/{pubmed['id']}",
                headers=admin_headers,
                json={"enabled": True},
                timeout=15,
            )

        phi_q = "Mr. John Doe DOB 03/14/1979 MRN 12345678 best HTN treatment"
        r = requests.post(
            f"{API}/search-sessions/",
            headers=clin_headers,
            json={"question": phi_q},
            timeout=15,
        )
        assert r.status_code == 200, r.text
        d = r.json()
        assert d.get("status") == "blocked_phi", d
        sid = d.get("session_id") or d.get("id")
        assert sid

        # No retrieval queries should be recorded
        time.sleep(1.0)
        tr = requests.get(f"{API}/search-sessions/{sid}/transparency", headers=clin_headers, timeout=15)
        # Blocked sessions may 404 from /transparency if ownership/state requires
        # — but session belongs to clinician so it should return 200.
        if tr.status_code == 200:
            t = tr.json()
            assert not t.get("queries"), f"PHI-blocked session must have no queries, got {t.get('queries')}"

        # phi_blocked audit recorded
        ar = requests.get(
            f"{API}/admin/audit-events?event_type=phi_blocked&limit=10",
            headers=admin_headers,
            timeout=15,
        )
        assert ar.status_code == 200
        events = ar.json()
        assert any(e.get("search_session_id") == sid for e in events), \
            "phi_blocked audit event not found for this session"


# ---------- Search session: europe_pmc only default ----------

class TestSearchEuropePmcOnly:
    @skip_if_pubmed_locked
    def test_only_europe_pmc_when_pubmed_disabled(self, clin_headers, admin_headers):
        # Force pubmed disabled
        cfgs = requests.get(f"{API}/admin/source-configs", headers=admin_headers, timeout=15).json()
        pubmed = next(c for c in cfgs if c["source_name"] == "pubmed")
        if pubmed["enabled"]:
            requests.patch(
                f"{API}/admin/source-configs/{pubmed['id']}",
                headers=admin_headers,
                json={"enabled": False},
                timeout=15,
            )

        question = f"sglt2 inhibitors in heart failure with preserved ejection fraction {uuid.uuid4().hex[:6]}"
        r = requests.post(
            f"{API}/search-sessions/",
            headers=clin_headers,
            json={"question": question},
            timeout=15,
        )
        assert r.status_code == 200, r.text
        d = r.json()
        assert d.get("status") == "searching", d
        sid = d["id"]

        tr = _wait_for_queries(clin_headers, sid, timeout_s=25)
        queries = tr.get("queries") or []
        sources_searched = tr.get("sources_searched") or []
        assert sources_searched, f"no sources searched: {tr}"
        assert "europe_pmc" in sources_searched
        assert "pubmed" not in sources_searched, f"pubmed leaked while disabled: {sources_searched}"
        # Expect 4 europe_pmc legacy plans. (Phase 5A may add concept-block
        # queries on top; we only assert the legacy 4 are present.)
        legacy_families = {"high_recall", "high_precision", "evidence_filtered", "guideline_focused"}
        epmc_qs = [q for q in queries if (q.get("source") or "europe_pmc") == "europe_pmc"
                    and q.get("query_type") in legacy_families]
        assert len(epmc_qs) == 4, f"expected 4 europe_pmc legacy queries, got {len(epmc_qs)}: {epmc_qs}"


# ---------- Search session: both sources enabled ----------

class TestSearchMultiSource:
    def test_both_sources_plan_eight_queries(self, clin_headers, admin_headers):
        # Enable pubmed
        cfgs = requests.get(f"{API}/admin/source-configs", headers=admin_headers, timeout=15).json()
        pubmed = next(c for c in cfgs if c["source_name"] == "pubmed")
        if not pubmed["enabled"]:
            r = requests.patch(
                f"{API}/admin/source-configs/{pubmed['id']}",
                headers=admin_headers,
                json={"enabled": True},
                timeout=15,
            )
            assert r.status_code == 200

        question = f"apixaban vs warfarin for atrial fibrillation stroke prevention {uuid.uuid4().hex[:6]}"
        r = requests.post(
            f"{API}/search-sessions/",
            headers=clin_headers,
            json={"question": question},
            timeout=15,
        )
        assert r.status_code == 200, r.text
        d = r.json()
        sid = d["id"]
        assert d.get("status") == "searching"

        tr = _wait_for_queries(clin_headers, sid, timeout_s=40)
        queries = tr.get("queries") or []
        sources_searched = tr.get("sources_searched") or []
        # Pipeline should plan from both
        assert "europe_pmc" in sources_searched, f"europe_pmc missing: {sources_searched}"
        assert "pubmed" in sources_searched, f"pubmed missing: {sources_searched}"
        # Phase 6B.1 — BooleanQueryCompiler is the PRIMARY planner. Legacy
        # ``search_planning.plan()`` is fallback-only (fires per-source when
        # the compiler emits zero queries OR the high-precision compiler
        # queries return zero hits). Assert primary compiler queries are
        # present for both sources; legacy is allowed to be 0 (compiler
        # sufficient) OR 4 (fallback fired for that source).
        epmc_primary = [q for q in queries
                         if (q.get("source") or "europe_pmc") == "europe_pmc"
                         and q.get("planner_source") == "boolean_compiler"]
        pmed_primary = [q for q in queries
                         if q.get("source") == "pubmed"
                         and q.get("planner_source") == "boolean_compiler"]
        assert len(epmc_primary) >= 1, f"epmc primary count: {len(epmc_primary)}"
        assert len(pmed_primary) >= 1, f"pubmed primary count: {len(pmed_primary)}"
        legacy_families = {"high_recall", "high_precision",
                            "evidence_filtered", "guideline_focused"}
        epmc_legacy = [q for q in queries
                        if (q.get("source") or "europe_pmc") == "europe_pmc"
                        and (q.get("query_type") in legacy_families)
                        and q.get("planner_source") == "legacy_planner"]
        pmed_legacy = [q for q in queries
                        if q.get("source") == "pubmed"
                        and (q.get("query_type") in legacy_families)
                        and q.get("planner_source") == "legacy_planner"]
        assert len(epmc_legacy) in (0, 4), \
            f"epmc legacy count: {len(epmc_legacy)} (must be 0 or 4)"
        assert len(pmed_legacy) in (0, 4), \
            f"pubmed legacy count: {len(pmed_legacy)} (must be 0 or 4)"

    def test_session_articles_have_sources_present_and_evaluator_coverage(self, clin_headers, admin_headers):
        # Reuse a session from prior step; create another fresh one and wait until ready
        question = f"metformin add-on therapy type 2 diabetes hba1c {uuid.uuid4().hex[:6]}"
        r = requests.post(
            f"{API}/search-sessions/",
            headers=clin_headers,
            json={"question": question},
            timeout=15,
        )
        assert r.status_code == 200, r.text
        sid = r.json()["id"]

        # Poll until awaiting_selection/complete or timeout
        ready = False
        deadline = time.time() + 90
        last_status = None
        while time.time() < deadline:
            s = requests.get(f"{API}/search-sessions/{sid}/status", headers=clin_headers, timeout=15)
            if s.status_code == 200:
                last_status = s.json()
                if last_status.get("articles_ready"):
                    ready = True
                    break
                if last_status.get("status") == "error":
                    pytest.skip(f"pipeline errored — external API may be unavailable: {last_status}")
            time.sleep(2.0)
        if not ready:
            pytest.skip(f"pipeline did not finish in time (status={last_status}); external API likely slow")

        full = requests.get(f"{API}/search-sessions/{sid}", headers=clin_headers, timeout=15).json()
        articles = full.get("articles") or []
        excluded = full.get("excluded_articles") or []
        rows = articles + excluded
        if not rows:
            pytest.skip("No articles returned by external APIs; cannot assert sources_present")
        # Every article row should have sources_present (list)
        for row in rows[:20]:
            assert "sources_present" in row, f"row missing sources_present: {row}"
            assert isinstance(row["sources_present"], list)

        evaluator = full.get("evaluator_json") or {}
        sc = evaluator.get("source_coverage") or {}
        assert "per_source_raw_counts" in sc, f"source_coverage missing per_source_raw_counts: {sc}"
        assert "cross_source_confirmed_articles" in sc, f"source_coverage missing cross_source_confirmed_articles: {sc}"


# ---------- Teardown: restore pubmed to default (disabled) ----------

class TestZRestoreDefaults:
    @skip_if_pubmed_locked
    def test_restore_pubmed_disabled(self, admin_headers):
        cfgs = requests.get(f"{API}/admin/source-configs", headers=admin_headers, timeout=15).json()
        pubmed = next(c for c in cfgs if c["source_name"] == "pubmed")
        if pubmed["enabled"]:
            r = requests.patch(
                f"{API}/admin/source-configs/{pubmed['id']}",
                headers=admin_headers,
                json={"enabled": False},
                timeout=15,
            )
            assert r.status_code == 200
            assert r.json()["enabled"] is False
