"""Phase K3A.3 — Tests for the article-join hygiene fix.

Covers:

- ``compute_article_join_status`` returns each closed-set status code
  for the matching synthetic input combination.
- ``should_enforce_cap_for_article`` returns ``False`` for every
  combination while ``CAP_ENFORCEMENT_MODE != "active"``.
- The guard returns ``False`` on an unstable join EVEN when
  ``CAP_ENFORCEMENT_MODE`` is explicitly mocked to ``"active"`` —
  defense-in-depth: orphan / unstable joins are never enforced
  regardless of mode.
- The guard never raises on missing / ``None`` / malformed inputs.
- Clinician projection surface bumps to ``trimmed_v1.1`` and exposes
  the closed-set ``article_join_status`` field.
- Admin projection surface bumps to ``full_v1.1`` and exposes the
  closed-set ``article_join_status`` field.
- The soak collector buckets orphan / join-missing candidates
  separately from ``true_positive`` and ``false_positive`` rows.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.cap_enforcement_guard import (  # noqa: E402
    compute_article_join_status,
    should_enforce_cap_for_article,
)


# ---------------------------------------------------------------------
# compute_article_join_status — closed-set return values.
# ---------------------------------------------------------------------


def test_join_status_stable():
    article = {"id": "a1", "canonical_key": "ck-1"}
    ssa = {"id": "ssa-1", "article_id": "a1"}
    assert compute_article_join_status(
        article_doc=article, search_session_article_doc=ssa,
        classification_canonical_key="ck-1",
    ) == "stable"


def test_join_status_article_doc_missing():
    ssa = {"id": "ssa-1", "article_id": "a1"}
    assert compute_article_join_status(
        article_doc=None, search_session_article_doc=ssa,
        classification_canonical_key="ck-1",
    ) == "article_doc_missing"


def test_join_status_search_session_article_missing():
    article = {"id": "a1", "canonical_key": "ck-1"}
    assert compute_article_join_status(
        article_doc=article, search_session_article_doc=None,
        classification_canonical_key="ck-1",
    ) == "search_session_article_missing"


def test_join_status_canonical_key_changed():
    article = {"id": "a1", "canonical_key": "ck-NEW"}
    ssa = {"id": "ssa-1", "article_id": "a1"}
    assert compute_article_join_status(
        article_doc=article, search_session_article_doc=ssa,
        classification_canonical_key="ck-OLD",
    ) == "canonical_key_missing_or_changed"


def test_join_status_canonical_key_missing_on_live_doc():
    article = {"id": "a1", "canonical_key": None}
    ssa = {"id": "ssa-1", "article_id": "a1"}
    assert compute_article_join_status(
        article_doc=article, search_session_article_doc=ssa,
        classification_canonical_key=None,
    ) == "canonical_key_missing_or_changed"


def test_join_status_article_removed_between_classification_and_review():
    assert compute_article_join_status(
        article_doc=None, search_session_article_doc=None,
        classification_canonical_key="ck-1",
    ) == "article_removed_between_classification_and_review"


# ---------------------------------------------------------------------
# should_enforce_cap_for_article — guard semantics.
# ---------------------------------------------------------------------


def test_guard_returns_false_when_enforcement_disabled():
    # Default mode is "disabled" via env.
    assert should_enforce_cap_for_article(
        article_join_status="stable",
        cap_reason_codes=["retracted"],
        enforcement_mode="disabled",
    ) is False


def test_guard_returns_false_for_every_unstable_join_even_when_active():
    """Even with a hypothetical ``"active"`` mode mocked in, the guard
    NEVER returns True for an unstable join. Defense-in-depth.
    """
    for unstable in (
        "article_doc_missing",
        "search_session_article_missing",
        "canonical_key_missing_or_changed",
        "article_removed_between_classification_and_review",
        "",
        None,
        "garbage_unknown_status",
    ):
        assert should_enforce_cap_for_article(
            article_join_status=unstable,
            cap_reason_codes=["retracted"],
            enforcement_mode="active",
        ) is False, f"guard wrongly True for status={unstable!r}"


def test_guard_returns_true_only_for_stable_join_under_active():
    """The only combination that returns True is ``stable`` +
    ``active`` + ``"retracted"`` in cap_reason_codes. K3B Wave 1
    extended the guard to also require the reason-code membership.
    """
    assert should_enforce_cap_for_article(
        article_join_status="stable",
        cap_reason_codes=["retracted"],
        enforcement_mode="active",
    ) is True


def test_guard_returns_true_under_shadow_with_retracted():
    """K3B Wave 1 Step 1 — ``shadow`` is now an allowed enforcement
    mode and the guard returns True under ``shadow`` + stable + retracted.
    """
    assert should_enforce_cap_for_article(
        article_join_status="stable",
        cap_reason_codes=["retracted"],
        enforcement_mode="shadow",
    ) is True


def test_guard_returns_false_when_retracted_missing_from_reason_codes():
    """K3B Wave 1 Step 1 — without ``"retracted"`` in
    ``cap_reason_codes``, the guard never returns True even on
    stable+active.
    """
    for codes in (
        [], None, ["expression_of_concern"],
        ["missing_primary_anchor"], ["preprint_unverified"],
    ):
        assert should_enforce_cap_for_article(
            article_join_status="stable",
            cap_reason_codes=codes,
            enforcement_mode="active",
        ) is False, f"guard wrongly True for codes={codes!r}"


def test_guard_never_raises_on_missing_inputs(monkeypatch):
    """The guard NEVER raises on missing / ``None`` inputs. Pin
    enforcement_mode to ``disabled`` for the stable+retracted case so
    the test doesn't depend on the running env (env is ``shadow`` on
    dev/staging since K3B Wave 1 Step 1).
    """
    monkeypatch.setenv(
        "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE", "disabled",
    )
    assert should_enforce_cap_for_article(
        article_join_status=None,
        cap_reason_codes=["retracted"],
        enforcement_mode=None,
    ) is False
    assert should_enforce_cap_for_article(
        article_join_status=None,
        cap_reason_codes=None,
        enforcement_mode="disabled",
    ) is False
    assert should_enforce_cap_for_article(
        article_join_status="stable",
        cap_reason_codes=["retracted"],
        enforcement_mode=None,
    ) is False


def test_guard_reads_env_when_mode_not_passed(monkeypatch):
    """When ``enforcement_mode`` is not passed, the guard reads
    ``CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE`` from the environment.
    """
    monkeypatch.setenv(
        "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE", "active",
    )
    assert should_enforce_cap_for_article(
        article_join_status="stable",
        cap_reason_codes=["retracted"],
    ) is True
    monkeypatch.setenv(
        "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE", "disabled",
    )
    assert should_enforce_cap_for_article(
        article_join_status="stable",
        cap_reason_codes=["retracted"],
    ) is False


# ---------------------------------------------------------------------
# Projection contract upgrades — trimmed_v1.1 + full_v1.1.
# ---------------------------------------------------------------------


def _ald_blob(article_id: str, join_status: str = "stable") -> dict:
    return {
        "schema_version": "K3A.1.0",
        "lane_label_mode": "shadow",
        "cap_detection_mode": "shadow",
        "projection_contract": "full_v1",
        "diagnostics": [
            {
                "article_id": article_id,
                "lane_label": "therapy_rct",
                "lane_confidence": 0.9,
                "branch_alignment": "aligned",
                "matched_branch_ids": ["primary_0"],
                "article_intent_fit": "full",
                "directness_label": "direct",
                "cap_candidates": [],
                "cap_reason_codes": [],
                "cap_severity": "low",
                "cap_confidence": 0.0,
                "cap_mode": "shadow",
                "would_cap": False,
                "would_exclude": False,
                "explanation": ["rule:one"],
                "evidence_role_candidate": "primary_evidence",
                "qa_warnings": [],
                "article_join_status": join_status,
            },
        ],
    }


async def _seed_session_with_ald(client_, ald: dict) -> tuple[str, str]:
    from core.db import get_db
    r = await client_.post(
        "/api/auth/login",
        json={
            "email": "clinician@test.com", "password": "ClinicianTest123!",
        },
    )
    assert r.status_code == 200
    token = r.json()["access_token"]
    uid = r.json()["user"]["id"]
    db = get_db()
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    await db.search_sessions.insert_one({
        "id": sid, "user_id": uid,
        "original_question": "q", "sanitized_question": "q",
        "topic_label": "test", "clinical_question_type": "therapy",
        "pico_json": {}, "picott_json": {},
        "status": "awaiting_selection",
        "created_at": now, "updated_at": now, "expires_at": now,
        "deleted_at": None,
        "session_diagnostics_json": {
            "article_lane_diagnostics": ald,
        },
    })
    return sid, token


@pytest.mark.asyncio
async def test_clinician_projection_is_trimmed_v1_1_with_join_status(client):
    ald = _ald_blob("art-1", join_status="article_doc_missing")
    sid, token = await _seed_session_with_ald(client, ald)
    r = await client.get(
        f"/api/search-sessions/{sid}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    block = (body.get("session_diagnostics_json") or {}).get(
        "article_lane_diagnostics"
    )
    assert isinstance(block, dict)
    assert block.get("projection_contract") == "trimmed_v1.3"
    diags = block.get("diagnostics") or []
    assert len(diags) == 1
    d = diags[0]
    assert d["article_join_status"] == "article_doc_missing"
    # Still no forbidden raw fields on clinician.
    for forbidden in (
        "cap_candidates", "branch_alignment", "matched_branch_ids",
        "lane_confidence", "cap_severity", "cap_confidence",
        "article_intent_fit", "evidence_role_candidate",
    ):
        assert forbidden not in d


@pytest.mark.asyncio
async def test_admin_projection_is_full_v1_1_with_join_status(client):
    from core.db import get_db
    ald = _ald_blob("art-2", join_status="canonical_key_missing_or_changed")
    # Use the admin login for the admin endpoint.
    rr = await client.post(
        "/api/auth/login",
        json={"email": "admin@test.com", "password": "AdminTest123!"},
    )
    atoken = rr.json()["access_token"]
    # Seed session under any user (admin endpoint is unscoped on owner).
    db = get_db()
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    await db.search_sessions.insert_one({
        "id": sid, "user_id": "seed-user",
        "original_question": "q", "sanitized_question": "q",
        "topic_label": "test", "clinical_question_type": "therapy",
        "pico_json": {}, "picott_json": {},
        "status": "awaiting_selection",
        "created_at": now, "updated_at": now, "expires_at": now,
        "deleted_at": None,
        "session_diagnostics_json": {
            "article_lane_diagnostics": ald,
        },
    })
    r = await client.get(
        f"/api/admin/search-sessions/{sid}/diagnostics",
        headers={"Authorization": f"Bearer {atoken}"},
    )
    assert r.status_code == 200, r.text
    block = r.json().get("article_lane_diagnostics")
    assert isinstance(block, dict)
    assert block.get("projection_contract") == "full_v1.3"
    diags = block.get("diagnostics") or []
    assert len(diags) == 1
    d = diags[0]
    assert d["article_join_status"] == "canonical_key_missing_or_changed"
    # The full-trace fields are still present on admin.
    for k in (
        "cap_candidates", "branch_alignment", "matched_branch_ids",
        "lane_confidence", "cap_severity",
    ):
        assert k in d


# ---------------------------------------------------------------------
# Soak collector — separate bucket for orphan / unstable joins.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soak_collector_buckets_orphan_join_separately(client):
    """Phase K3A.3 — the K3A.3 collector reports a third bucket for
    ``article_join_status != "stable"`` candidates so the K3B Wave 1
    review cannot conflate orphan/unstable joins with true positives
    or false positives.
    """
    from core.db import get_db
    from scripts.k3a3_soak_collector import collect

    db = get_db()
    # Login as clinician (we need a valid user id).
    r = await client.post(
        "/api/auth/login",
        json={
            "email": "clinician@test.com", "password": "ClinicianTest123!",
        },
    )
    uid = r.json()["user"]["id"]
    sid_ok = str(uuid.uuid4())
    sid_orphan = str(uuid.uuid4())
    art_ok = str(uuid.uuid4())
    art_orphan = str(uuid.uuid4())
    # OK article — populated and joinable.
    await db.articles.insert_one({
        "id": art_ok, "pmid": "k3a3-test-ok",
        "doi": "10.999/k3a3-ok",
        "title": "A retracted publication test fixture (K3A.3)",
        "journal": "J Test", "publication_year": 2024,
        "publication_date": "2024-01-01",
        "publication_types_json": [
            "Journal Article", "Retracted Publication",
        ],
        "is_retracted": True, "is_retraction_notice": False,
        "has_expression_of_concern": False,
        "source_ids_json": {}, "source_urls_json": [],
        "canonical_key": "ck-k3a3-test-ok",
    })
    now = datetime.now(timezone.utc)
    # Session 1 — diagnostic with article_join_status=stable.
    await db.search_sessions.insert_one({
        "id": sid_ok, "user_id": uid,
        "original_question": "q", "sanitized_question": "q",
        "topic_label": "t", "clinical_question_type": "therapy",
        "pico_json": {}, "picott_json": {},
        "status": "awaiting_selection",
        "created_at": now, "updated_at": now, "expires_at": now,
        "deleted_at": None,
        "session_diagnostics_json": {
            "article_lane_diagnostics": {
                "schema_version": "K3A.1.0",
                "lane_label_mode": "shadow",
                "cap_detection_mode": "shadow",
                "projection_contract": "full_v1.1",
                "diagnostics": [{
                    "article_id": art_ok,
                    "lane_label": "retracted_or_expression_of_concern",
                    "directness_label": "unsafe",
                    "cap_reason_codes": ["retracted"],
                    "cap_candidates": [{
                        "reason_code": "retracted", "severity": "high",
                        "explanation_id": "safety:retracted",
                    }],
                    "cap_severity": "high", "cap_confidence": 0.99,
                    "cap_mode": "shadow", "would_cap": False,
                    "would_exclude": False,
                    "explanation": ["safety:retracted_or_eoc"],
                    "qa_warnings": [],
                    "branch_alignment": "aligned",
                    "matched_branch_ids": ["primary_0"],
                    "article_intent_fit": "full",
                    "evidence_role_candidate": "to_exclude_candidate",
                    "lane_confidence": 0.99,
                    "article_join_status": "stable",
                }],
            },
        },
    })
    # Session 2 — diagnostic that's orphaned at soak time
    # (article_join_status is one of the unstable codes, AND the
    # article doc indeed is not in articles).
    await db.search_sessions.insert_one({
        "id": sid_orphan, "user_id": uid,
        "original_question": "q", "sanitized_question": "q",
        "topic_label": "t", "clinical_question_type": "therapy",
        "pico_json": {}, "picott_json": {},
        "status": "awaiting_selection",
        "created_at": now, "updated_at": now, "expires_at": now,
        "deleted_at": None,
        "session_diagnostics_json": {
            "article_lane_diagnostics": {
                "schema_version": "K3A.1.0",
                "lane_label_mode": "shadow",
                "cap_detection_mode": "shadow",
                "projection_contract": "full_v1.1",
                "diagnostics": [{
                    "article_id": art_orphan,
                    "lane_label": "retracted_or_expression_of_concern",
                    "directness_label": "unsafe",
                    "cap_reason_codes": ["retracted"],
                    "cap_candidates": [{
                        "reason_code": "retracted", "severity": "high",
                        "explanation_id": "safety:retracted",
                    }],
                    "cap_severity": "high", "cap_confidence": 0.85,
                    "cap_mode": "shadow", "would_cap": False,
                    "would_exclude": False,
                    "explanation": ["safety:retracted_or_eoc"],
                    "qa_warnings": [],
                    "branch_alignment": "aligned",
                    "matched_branch_ids": ["primary_0"],
                    "article_intent_fit": "full",
                    "evidence_role_candidate": "to_exclude_candidate",
                    "lane_confidence": 0.99,
                    "article_join_status":
                        "article_removed_between_classification_and_review",
                }],
            },
        },
    })
    out = Path(tempfile.mkdtemp()) / "k3a3_soak.jsonl"
    summary = await collect(
        mongo_url=os.environ["MONGO_URL"],
        db_name=os.environ["DB_NAME"],
        window_days=1, out_path=out, provenance="test",
    )
    # Two retracted rows scanned, but the second goes into the
    # orphan bucket.
    assert summary["candidates_total"] >= 2
    assert summary["orphan_or_join_unstable_total"] >= 1
    rows = [json.loads(L) for L in out.read_text().splitlines() if L.strip()]
    stables = [r for r in rows if r.get("article_join_status") == "stable"]
    orphans = [
        r for r in rows
        if r.get("article_join_status") != "stable"
    ]
    assert len(stables) >= 1
    assert len(orphans) >= 1
    # And the JSONL never leaks the forbidden text fields.
    blob = json.dumps(rows)
    for tok in (
        "FORBIDDEN_RAW", "original_question", "sanitized_question",
        "system:", "SYSTEM_PROMPT", "sk-", "@gmail.com",
    ):
        assert tok not in blob, (
            f"forbidden token {tok!r} leaked to K3A.3 JSONL"
        )
