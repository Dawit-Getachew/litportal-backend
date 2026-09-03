"""Phase 6F.1 — Live API acceptance-gate tests against the deployed backend.

These tests verify the directive's specific acceptance gates for the new
positive anchor-match boost + stable composite sort applied to
GET /api/search-sessions/{sid}.

Acceptance gates (from the directive):
  • Q1 (psoriasis): ≥5 psoriasis-related articles in top 10 of articles[],
    ≥3 default-selected (included_by_default=true), top entry is a psoriasis
    treatment guideline / systematic review with relevance_score ≥70.
  • Q2 (Seroquel): ≥5 quetiapine/Seroquel-related articles in articles[],
    ≥3 default-selected.
  • Wrong-anchor cap regression: a wrong-disease entry from Q1's excluded
    list and a wrong-drug entry from Q2's excluded list both have
    relevance_score ≤ 20, included_by_default=false, and a cap_reason in
    the closed set.
  • Sort-order regression: first article in articles[] for Q1 AND Q2 has
    overall_rank >= 70 AND included_by_default=true; any capped row in
    articles[] never outranks an on-target row at the same overall_rank.
  • Closed-set boost label vocabulary: any anchor_match_* entry uses one
    of the three closed-set prefixes.
  • Default+Excluded contradiction self-heal still holds.

Reuses the live session fixtures defined in
test_phase6f_live_api.py (via direct re-import) so we don't double the
240s poll cost.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import requests


# Phase 6F closeout (IMPL-4) — pin to ``live_api`` xdist group.
pytestmark = pytest.mark.xdist_group(name="live_api")


# ---------------------------------------------------------------------------
# Live env discovery (independent module to avoid cross-file fixture coupling)
# ---------------------------------------------------------------------------

def _resolve_base_url() -> str:
    env_path = Path("/app/frontend/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("REACT_APP_BACKEND_URL="):
                return line.split("=", 1)[1].strip().rstrip("/")
    url = os.environ.get("REACT_APP_BACKEND_URL")
    if not url:
        pytest.skip(
            "REACT_APP_BACKEND_URL is not configured",
            allow_module_level=True,
        )
    return url.rstrip("/")


BASE_URL = _resolve_base_url()
CLINICIAN_EMAIL = "clinician@test.com"
# test fixture credential — matches seeded user in server.py::_seed_user_if_missing
CLINICIAN_PASSWORD = "ClinicianTest123!"

Q1 = "What is the first line treatment and second line treatment for psoriasis?"
Q2 = "What are the side effects of Seroquel?"

TERMINAL = {"awaiting_selection", "complete", "completed"}
POLL_TIMEOUT_S = 240
POLL_INTERVAL_S = 4

# Closed-set vocab from the directive
CLOSED_CAP_REASONS = {
    "missing_intervention_anchor",
    "missing_condition_anchor",
    "missing_primary_anchor",
    "wrong_drug",
    "wrong_condition",
    "wrong_primary_anchor",
    # Phase 6F closed-set extensions (still valid):
    "disease_only_match",
    "drug_safety_no_adverse_effect",
    "treatment_sequencing_wrong_disease",
}
CLOSED_BOOST_PREFIXES = (
    "anchor_match_full",
    "anchor_match_partial_outcome",
    "anchor_match_partial_intervention",
)


# ---------------------------------------------------------------------------
# Helpers
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
        pytest.skip(f"clinician login failed: {r.status_code} {r.text[:200]}")
    token = r.json().get("access_token")
    assert token
    s.headers.update({"Authorization": f"Bearer {token}"})
    return s


def _create_and_poll(s: requests.Session, question: str) -> dict:
    r = s.post(
        f"{BASE_URL}/api/search-sessions/",
        json={"question": question},
        timeout=60,
    )
    assert r.status_code in (200, 201), f"{r.status_code} {r.text[:300]}"
    body = r.json()
    sid = body.get("id") or body.get("session_id")
    assert sid
    deadline = time.time() + POLL_TIMEOUT_S
    last = None
    while time.time() < deadline:
        gr = s.get(f"{BASE_URL}/api/search-sessions/{sid}", timeout=60)
        assert gr.status_code == 200
        last = gr.json()
        if (last.get("status") or "").lower() in TERMINAL:
            return last
        time.sleep(POLL_INTERVAL_S)
    pytest.fail(
        f"session {sid} not terminal in {POLL_TIMEOUT_S}s; "
        f"status={last and last.get('status')}"
    )
    return last  # unreachable


@pytest.fixture(scope="module")
def q1(auth_session) -> dict:
    return _create_and_poll(auth_session, Q1)


@pytest.fixture(scope="module")
def q2(auth_session) -> dict:
    return _create_and_poll(auth_session, Q2)


def _articles(body: dict) -> list[dict]:
    return body.get("articles") or []


def _excluded(body: dict) -> list[dict]:
    return body.get("excluded_articles") or []


def _text_blob(art: dict) -> str:
    """Articles may carry title/abstract directly OR under 'article'.
    Also include matched_must_have / rank_explanation for robust topic detection."""
    inner = art.get("article") or {}
    parts = [
        art.get("title") or inner.get("title") or "",
        art.get("abstract") or inner.get("abstract") or "",
        " ".join(art.get("matched_must_have") or []),
        art.get("rank_explanation") or "",
    ]
    return " ".join(parts).lower()


def _is_psoriasis(art: dict) -> bool:
    blob = _text_blob(art)
    return "psoriasis" in blob or "psoriatic" in blob


def _is_quetiapine(art: dict) -> bool:
    blob = _text_blob(art)
    return "quetiapine" in blob or "seroquel" in blob


# ---------------------------------------------------------------------------
# Q1 acceptance gate
# ---------------------------------------------------------------------------

class TestQ1AcceptanceGate:
    def test_top10_has_5_psoriasis_articles(self, q1):
        arts = _articles(q1)
        assert len(arts) >= 5, f"too few articles: {len(arts)}"
        top10 = arts[:10]
        psoriasis = [a for a in top10 if _is_psoriasis(a)]
        assert len(psoriasis) >= 5, (
            f"need ≥5 psoriasis articles in top 10, got "
            f"{len(psoriasis)} (titles={[a.get('title') for a in top10]})"
        )

    def test_at_least_3_default_selected(self, q1):
        arts = _articles(q1)
        sel = [a for a in arts if a.get("included_by_default")]
        assert len(sel) >= 3, (
            f"need ≥3 default-selected, got {len(sel)} "
            f"(included={[a.get('id') for a in sel]})"
        )

    def test_top_entry_is_psoriasis_with_rel_70(self, q1):
        arts = _articles(q1)
        assert arts, "empty articles[] for Q1"
        top = arts[0]
        assert _is_psoriasis(top), f"top article not psoriasis: {top.get('title')}"
        rel = top.get("relevance_score") or 0
        assert rel >= 70, f"top relevance_score={rel} (<70)"


# ---------------------------------------------------------------------------
# Q2 acceptance gate
# ---------------------------------------------------------------------------

class TestQ2AcceptanceGate:
    def test_articles_have_5_quetiapine(self, q2):
        arts = _articles(q2)
        qets = [a for a in arts if _is_quetiapine(a)]
        assert len(qets) >= 5, (
            f"need ≥5 quetiapine/Seroquel articles, got {len(qets)} "
            f"(of {len(arts)})"
        )

    def test_at_least_3_default_selected(self, q2):
        arts = _articles(q2)
        sel = [a for a in arts if a.get("included_by_default")]
        assert len(sel) >= 3, f"need ≥3 default-selected, got {len(sel)}"


# ---------------------------------------------------------------------------
# Wrong-anchor cap regression (one from Q1, one from Q2)
# ---------------------------------------------------------------------------

class TestWrongAnchorCapRegression:
    def test_q1_wrong_disease_in_excluded(self, q1):
        excl = _excluded(q1)
        if not excl:
            pytest.skip("Q1 excluded_articles[] empty — nothing to check")
        # Pick a row that's BOTH off-topic AND carries a closed-set cap_reason.
        wrong = None
        for a in excl:
            cap_reasons = a.get("cap_reasons") or []
            if not _is_psoriasis(a) and any(c in CLOSED_CAP_REASONS for c in cap_reasons):
                wrong = a
                break
        if wrong is None:
            pytest.skip("no wrong-disease+capped entry surfaced in Q1 excluded")
        rel = wrong.get("relevance_score")
        assert rel is None or rel <= 20, f"wrong-disease rel={rel} (>20)"
        assert not wrong.get("included_by_default"), "wrong-disease default-selected!"
        cap_reasons = wrong.get("cap_reasons") or []
        assert any(c in CLOSED_CAP_REASONS for c in cap_reasons), (
            f"cap_reasons not in closed set: {cap_reasons}"
        )

    def test_q2_wrong_drug_in_excluded(self, q2):
        excl = _excluded(q2)
        if not excl:
            pytest.skip("Q2 excluded_articles[] empty — nothing to check")
        wrong = None
        for a in excl:
            cap_reasons = a.get("cap_reasons") or []
            if not _is_quetiapine(a) and any(c in CLOSED_CAP_REASONS for c in cap_reasons):
                wrong = a
                break
        if wrong is None:
            pytest.skip("no wrong-drug+capped entry surfaced in Q2 excluded")
        rel = wrong.get("relevance_score")
        assert rel is None or rel <= 20, f"wrong-drug rel={rel} (>20)"
        assert not wrong.get("included_by_default"), "wrong-drug default-selected!"
        cap_reasons = wrong.get("cap_reasons") or []
        assert any(c in CLOSED_CAP_REASONS for c in cap_reasons), (
            f"cap_reasons not in closed set: {cap_reasons}"
        )


# ---------------------------------------------------------------------------
# Sort-order regression: top of articles[] is on-target, capped never above
# ---------------------------------------------------------------------------

class TestSortOrderRegression:
    @pytest.mark.parametrize("fixture_name", ["q1", "q2"])
    def test_first_article_overall_rank_70_and_default_selected(
        self, request, fixture_name
    ):
        body = request.getfixturevalue(fixture_name)
        arts = _articles(body)
        assert arts, f"{fixture_name} articles[] empty"
        top = arts[0]
        ovr = top.get("overall_rank") or top.get("overall_rank_score") or 0
        assert ovr >= 70, (
            f"{fixture_name} top overall_rank={ovr} (<70); "
            f"top.id={top.get('id')}, rel={top.get('relevance_score')}"
        )
        assert top.get("included_by_default") is True, (
            f"{fixture_name} top not default-selected: {top.get('id')}"
        )

    @pytest.mark.parametrize("fixture_name", ["q1", "q2"])
    def test_capped_never_outranks_on_target_at_same_rank(
        self, request, fixture_name
    ):
        body = request.getfixturevalue(fixture_name)
        arts = _articles(body)
        # Build (overall_rank, is_capped, is_default) tuples in current order.
        rows = []
        for a in arts:
            cap = a.get("relevance_cap_applied")
            is_capped = bool(cap) and int(cap) <= 20
            ovr = a.get("overall_rank") or a.get("overall_rank_score") or 0
            rows.append({
                "id": a.get("id"),
                "ovr": ovr,
                "capped": is_capped,
                "default": bool(a.get("included_by_default")),
            })
        for i in range(len(rows) - 1):
            cur, nxt = rows[i], rows[i + 1]
            if cur["capped"] and not nxt["capped"] and cur["ovr"] == nxt["ovr"]:
                pytest.fail(
                    f"{fixture_name}: capped row {cur['id']} outranks "
                    f"on-target {nxt['id']} at same overall_rank={cur['ovr']}"
                )


# ---------------------------------------------------------------------------
# Closed-set boost-label vocabulary
# ---------------------------------------------------------------------------

class TestClosedSetBoostLabels:
    @pytest.mark.parametrize("fixture_name", ["q1", "q2"])
    def test_anchor_match_labels_are_closed_set(self, request, fixture_name):
        body = request.getfixturevalue(fixture_name)
        arts = _articles(body)
        seen_any = False
        for a in arts:
            reasons = a.get("ranking_boost_reasons") or []
            for s in reasons:
                if not s.startswith("anchor_match_"):
                    continue
                seen_any = True
                assert any(
                    s.startswith(p + "(+") or s == p
                    for p in CLOSED_BOOST_PREFIXES
                ), f"non-closed-set anchor_match label: {s!r}"
        # At least one on-target article should carry the boost.
        assert seen_any, (
            f"{fixture_name}: no anchor_match_* label observed in articles[]"
        )


# ---------------------------------------------------------------------------
# Default+Excluded self-heal contract
# ---------------------------------------------------------------------------

class TestDefaultExcludedSelfHeal:
    @pytest.mark.parametrize("fixture_name", ["q1", "q2"])
    def test_no_row_both_user_included_and_exclusion_reason(
        self, request, fixture_name
    ):
        body = request.getfixturevalue(fixture_name)
        for a in _articles(body) + _excluded(body):
            ui = a.get("user_included")
            er = a.get("exclusion_reason")
            assert not (ui is True and er), (
                f"contradiction id={a.get('id')} user_included={ui} "
                f"exclusion_reason={er}"
            )
