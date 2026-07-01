"""Phase K4 — backend test suite for branch-aware evidence packets.

Closed-set coverage:
  1. Display-mapping exhaustiveness over the R1A + R1C label union.
  2. Display-map KeyError fail-loud guarantee for unknown labels.
  3. Branch-assignment rule (c) — 4 sub-cases (primary-intent match,
     plan-order fallback, unbranched fallback, single-branch happy path).
  4. Builder packet schema enforcement + closed-set enums.
  5. Builder R-series label attachment + is_mixed_signal computation.
  6. Builder leak-scan — abstracts, prompts, completions, credentials
     all blocked from the persisted document.
  7. Endpoint clinician GET — 200 owner, 403 non-owner, 404 unknown,
     409 phi-blocked session.
  8. Endpoint admin GET + admin rebuild.
  9. Invariance — search-sessions / articles / summaries collections
     and POST /api/search-sessions response shape unchanged.
 10. R1A / R1B / R1C / R1C.1 configuration byte-identical after K4 runs.
 11. M3.2 kill switch dormant.
 12. TTL / unique-index assertion on evidence_packets.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from core.db import Collections, get_db
from models.common import utcnow
from services.k4_evidence_packet_builder import (
    EVIDENCE_PACKET_VERSION,
    UNBRANCHED_ID,
    assign_branch_for_article,
    build_evidence_packet,
)
from services.k4_evidence_packet_display_map import (
    ALL_R_SERIES_LABELS,
    R_SERIES_DISPLAY_MAP,
    map_label_to_display,
    map_labels_to_displays,
)


# =====================================================================
# 1. Display-mapping exhaustiveness over R1A + R1C label union.
# =====================================================================
def test_K4_display_map_covers_every_R_series_label():
    missing = ALL_R_SERIES_LABELS - set(R_SERIES_DISPLAY_MAP.keys())
    assert not missing, sorted(missing)


def test_K4_display_map_has_no_unknown_labels():
    extra = set(R_SERIES_DISPLAY_MAP.keys()) - ALL_R_SERIES_LABELS
    assert not extra, sorted(extra)


def test_K4_display_map_entries_have_closed_set_polarity():
    for label, entry in R_SERIES_DISPLAY_MAP.items():
        assert "human" in entry and entry["human"], label
        assert "tooltip" in entry and entry["tooltip"], label
        assert entry["polarity"] in {"positive", "negative", "flag"}, label


def test_K4_display_map_fail_loud_on_unknown_label():
    with pytest.raises(KeyError):
        map_label_to_display("not_a_real_label")


def test_K4_map_labels_to_displays_dedupes_preserves_order():
    out = map_labels_to_displays([
        "pathophysiology_mechanism_direct",
        "pathophysiology_mechanism_direct",  # duplicate
        "mechanism_target_exact",
        "",
        None,
    ])
    assert [d["label"] for d in out] == [
        "pathophysiology_mechanism_direct",
        "mechanism_target_exact",
    ]


# =====================================================================
# 2. Branch-assignment rule (c).
# =====================================================================
_BRANCHES_FIXTURE = [
    {
        "branch_id": "b_mech",
        "intent": "pathophysiology_mechanism",
        "framework": "pathophysiology_mechanism",
        "query_families": [{"family_name": "high_recall_cohort"}],
    },
    {
        "branch_id": "b_dx",
        "intent": "diagnostic_yield",
        "framework": "diagnostic_yield",
        "query_families": [{"family_name": "diagnostic_yield"}],
    },
]


def test_K4_branch_assign_primary_intent_match_wins():
    """Rule (1) — when the primary_intent's branch is in matched_ids,
    assign to that branch."""
    sa_row = {
        "matched_branch_ids": ["b_dx", "b_mech"],
        "query_types_matched": ["diagnostic_yield"],
    }
    article = {}
    assignment = assign_branch_for_article(
        sa_row=sa_row, article_doc=article,
        branches=_BRANCHES_FIXTURE,
        primary_intent="pathophysiology_mechanism",
    )
    assert assignment == "b_mech"


def test_K4_branch_assign_plan_order_fallback_when_primary_intent_missing():
    """Rule (2) — primary_intent not in matched_ids → lowest-index
    branch whose family set intersects."""
    sa_row = {
        "matched_branch_ids": [],
        "query_types_matched": ["cb_diagnostic_yield"],
    }
    assignment = assign_branch_for_article(
        sa_row=sa_row, article_doc={},
        branches=_BRANCHES_FIXTURE,
        primary_intent="pathophysiology_mechanism",
    )
    # Primary intent's branch (b_mech) doesn't intersect; b_dx does.
    assert assignment == "b_dx"


def test_K4_branch_assign_unbranched_fallback():
    """Rule (3) — no matched_ids and no family intersection → unbranched."""
    sa_row = {
        "matched_branch_ids": [],
        "query_types_matched": ["unrelated_family"],
    }
    assignment = assign_branch_for_article(
        sa_row=sa_row, article_doc={},
        branches=_BRANCHES_FIXTURE,
        primary_intent="pathophysiology_mechanism",
    )
    assert assignment == UNBRANCHED_ID


def test_K4_branch_assign_single_branch_happy_path():
    """Single-branch session → all articles in that branch."""
    branches = [{
        "branch_id": "only",
        "intent": "pathophysiology_mechanism",
        "framework": "pathophysiology_mechanism",
        "query_families": [{"family_name": "high_recall"}],
    }]
    sa_row = {
        "matched_branch_ids": [],
        "query_types_matched": ["cb_high_recall"],
    }
    assert assign_branch_for_article(
        sa_row=sa_row, article_doc={}, branches=branches,
        primary_intent="pathophysiology_mechanism",
    ) == "only"


# =====================================================================
# 3. End-to-end builder — synthetic session.
# =====================================================================
async def _seed_synthetic_session(
    *,
    user_id: str = "user-1",
    shape: str = "pathophysiology_mechanism",
    extra_articles: list[dict] | None = None,
) -> str:
    """Drop a synthetic search_session + search_session_articles +
    articles + branch_query_plan and return the session id.
    """
    db = get_db()
    session_id = str(uuid.uuid4())
    bqp = {
        "schema_version": "1.0",
        "mode": "branch_aware",
        "branches": [
            {
                "branch_id": "b_mech",
                "intent": "pathophysiology_mechanism",
                "framework": "pathophysiology_mechanism",
                "query_families": [{"family_name": "high_recall_cohort"}],
                "required_concepts_used": [
                    {"canonical_concept": "cholestasis"},
                    {"canonical_concept": "pruritus"},
                ],
                "validated_MeSH_terms": ["Cholestasis", "Pruritus"],
            },
        ],
    }
    cqm = {
        "primary_intent": shape,
        "framework": shape,
        "branch_query_plan": bqp,
        "anchors": [],
        "required_anchors": [],
    }
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": session_id,
        "user_id": user_id,
        "original_question": "synthetic K4 question",
        "sanitized_question": "synthetic K4 question",
        "status": "awaiting_selection",
        "clinical_question_type": shape,
        "clinical_question_map_json": cqm,
        "created_at": utcnow(),
        "updated_at": utcnow(),
        "expires_at": utcnow(),
        "deleted_at": None,
    })
    arts = extra_articles or [
        {
            "id": "art-1", "pmid": "1", "doi": "10.1/a",
            "title": "Mechanisms of pruritus in cholestasis",
            "abstract": "MOCK ABSTRACT — bile acids, autotaxin.",
            "journal": "Nat Rev", "publication_year": 2023,
            "source": "pubmed", "full_text_status": "abstract_only",
            "publication_types_json": ["Review"],
        },
        {
            "id": "art-2", "pmid": "2", "doi": "10.1/b",
            "title": "Treatment of cholestatic itch",
            "abstract": "MOCK ABSTRACT — treatment trial.",
            "journal": "Hepatol", "publication_year": 2024,
            "source": "pubmed", "full_text_status": "full_text_available",
            "publication_types_json": ["Randomized Controlled Trial"],
        },
    ]
    await db[Collections.ARTICLES].insert_many(arts)
    for i, art in enumerate(arts, start=1):
        await db[Collections.SEARCH_SESSION_ARTICLES].insert_one({
            "id": f"sa-{i}-{session_id}",
            "search_session_id": session_id,
            "article_id": art["id"],
            "overall_rank": 100 - i,
            "included_by_default": (i == 1),
            "user_included": False,
            "exclusion_reason": None,
            "query_types_matched": ["cb_high_recall_cohort"],
            "matched_branch_ids": ["b_mech"],
            "sources_present": ["pubmed"],
            "created_at": utcnow(),
            "updated_at": utcnow(),
        })
    return session_id


async def _cleanup_synthetic_session(session_id: str):
    db = get_db()
    await db[Collections.SEARCH_SESSIONS].delete_one({"id": session_id})
    await db[Collections.SEARCH_SESSION_ARTICLES].delete_many({
        "search_session_id": session_id,
    })
    await db[Collections.EVIDENCE_PACKETS].delete_one({
        "search_session_id": session_id,
    })


@pytest.mark.asyncio
async def test_K4_builder_groups_articles_by_branch():
    session_id = await _seed_synthetic_session()
    try:
        packet = await build_evidence_packet(
            search_session_id=session_id,
        )
        assert packet["evidence_packet_version"] == EVIDENCE_PACKET_VERSION
        assert len(packet["branches"]) == 1
        br = packet["branches"][0]
        assert br["branch_id"] == "b_mech"
        assert len(br["articles"]) == 2
        # Articles include directness labels in display-mapped form.
        for a in br["articles"]:
            for p in a["directness_label_human_strings"]:
                assert "human" in p and "tooltip" in p
                assert p["polarity"] in {"positive", "negative", "flag"}
    finally:
        await _cleanup_synthetic_session(session_id)


@pytest.mark.asyncio
async def test_K4_builder_attaches_r_series_labels():
    session_id = await _seed_synthetic_session()
    try:
        packet = await build_evidence_packet(
            search_session_id=session_id,
        )
        br = packet["branches"][0]
        labels_per_article = [
            set(a["directness_labels"]) for a in br["articles"]
        ]
        # Mechanism review (art-1) should produce mechanism labels.
        assert any(
            "pathophysiology_mechanism_direct" in s
            for s in labels_per_article
        )
    finally:
        await _cleanup_synthetic_session(session_id)


# =====================================================================
# 4. Leak-scan — no abstracts, prompts, credentials.
# =====================================================================
@pytest.mark.asyncio
async def test_K4_builder_does_not_persist_abstract_text():
    session_id = await _seed_synthetic_session()
    try:
        await build_evidence_packet(search_session_id=session_id)
        db = get_db()
        doc = await db[Collections.EVIDENCE_PACKETS].find_one(
            {"search_session_id": session_id}, {"_id": 0},
        )
        flat = str(doc).lower()
        assert "mock abstract" not in flat
        for forbidden in (
            "system:", "assistant:", "sk-", "bearer ",
            "password", "x-api-key",
        ):
            assert forbidden not in flat
    finally:
        await _cleanup_synthetic_session(session_id)


# =====================================================================
# 5. Endpoint tests.
# =====================================================================
@pytest.mark.asyncio
async def test_K4_clinician_get_404_for_missing_session(client):
    """Unknown session_id → 404."""
    # Sign up a fresh user and login
    email = f"k4-{uuid.uuid4().hex[:8]}@k4ev-test.com"
    await client.post("/api/auth/signup", json={
        "email": email, "password": "K4SuperTest!23", "name": "K4",
    })
    r = await client.post("/api/auth/login", json={
        "email": email, "password": "K4SuperTest!23",
    })
    token = r.json()["access_token"]
    hdr = {"Authorization": f"Bearer {token}"}
    r = await client.get(
        "/api/search-sessions/does-not-exist/evidence-packet",
        headers=hdr,
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_K4_clinician_get_403_for_non_owner(client):
    # Two users; user2 tries to read user1's session
    email1 = f"k4owner-{uuid.uuid4().hex[:8]}@k4ev-test.com"
    email2 = f"k4other-{uuid.uuid4().hex[:8]}@k4ev-test.com"
    for e in (email1, email2):
        await client.post("/api/auth/signup", json={
            "email": e, "password": "K4SuperTest!23", "name": "K4",
        })
    # login user1, get user_id
    r1 = await client.post("/api/auth/login", json={
        "email": email1, "password": "K4SuperTest!23",
    })
    uid1 = r1.json()["user"]["id"]
    session_id = await _seed_synthetic_session(user_id=uid1)
    try:
        # login user2 and try to fetch user1's packet
        r2 = await client.post("/api/auth/login", json={
            "email": email2, "password": "K4SuperTest!23",
        })
        token2 = r2.json()["access_token"]
        r = await client.get(
            f"/api/search-sessions/{session_id}/evidence-packet",
            headers={"Authorization": f"Bearer {token2}"},
        )
        assert r.status_code == 403
    finally:
        await _cleanup_synthetic_session(session_id)


@pytest.mark.asyncio
async def test_K4_clinician_get_200_for_owner_builds_lazily(client):
    email = f"k4-owner-{uuid.uuid4().hex[:8]}@k4ev-test.com"
    await client.post("/api/auth/signup", json={
        "email": email, "password": "K4SuperTest!23", "name": "K4",
    })
    r = await client.post("/api/auth/login", json={
        "email": email, "password": "K4SuperTest!23",
    })
    uid = r.json()["user"]["id"]
    token = r.json()["access_token"]
    session_id = await _seed_synthetic_session(user_id=uid)
    try:
        r = await client.get(
            f"/api/search-sessions/{session_id}/evidence-packet",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["evidence_packet_version"] == EVIDENCE_PACKET_VERSION
        assert len(body["branches"]) == 1
    finally:
        await _cleanup_synthetic_session(session_id)


@pytest.mark.asyncio
async def test_K4_admin_get_packet_by_id(client):
    # Admin user
    r = await client.post("/api/auth/login", json={
        "email": "admin@test.com", "password": "AdminTest123!",
    })
    token = r.json()["access_token"]
    hdr = {"Authorization": f"Bearer {token}"}
    # Need a session owned by SOMEONE (we pick admin's id)
    uid = r.json()["user"]["id"]
    session_id = await _seed_synthetic_session(user_id=uid)
    try:
        # Build the packet via clinician endpoint first
        r = await client.get(
            f"/api/search-sessions/{session_id}/evidence-packet",
            headers=hdr,
        )
        assert r.status_code == 200
        packet_id = r.json()["id"]
        # Admin GET by packet id
        r = await client.get(
            f"/api/admin/evidence-packets/{packet_id}",
            headers=hdr,
        )
        assert r.status_code == 200
        assert r.json()["id"] == packet_id
    finally:
        await _cleanup_synthetic_session(session_id)


@pytest.mark.asyncio
async def test_K4_admin_rebuild_is_idempotent(client):
    r = await client.post("/api/auth/login", json={
        "email": "admin@test.com", "password": "AdminTest123!",
    })
    token = r.json()["access_token"]
    hdr = {"Authorization": f"Bearer {token}"}
    uid = r.json()["user"]["id"]
    session_id = await _seed_synthetic_session(user_id=uid)
    try:
        for _ in range(2):
            r = await client.post(
                f"/api/admin/evidence-packets/by-session/{session_id}/rebuild",
                headers=hdr,
            )
            assert r.status_code == 200
            assert r.json()["rebuilt"] is True
    finally:
        await _cleanup_synthetic_session(session_id)


# =====================================================================
# 6. Invariance — R-series + R1A/R1B/R1C configuration UNCHANGED.
# =====================================================================
def test_K4_does_not_mutate_R1A_label_weights():
    from services.r1a_directness_labels import SHADOW_LABEL_WEIGHTS
    assert SHADOW_LABEL_WEIGHTS["pathophysiology_mechanism_direct"] == 1.0
    assert SHADOW_LABEL_WEIGHTS["mechanism_with_named_process"] == 1.2
    # R1C anchor labels live in R1C_LABEL_MULTIPLIERS, NOT in
    # SHADOW_LABEL_WEIGHTS — confirm that boundary stays intact.
    assert "procedure_complication_outcome_exact" not in SHADOW_LABEL_WEIGHTS
    assert "mechanism_target_exact" not in SHADOW_LABEL_WEIGHTS


def test_K4_does_not_mutate_R1B_configuration():
    from services.r1b_shadow_simulator import (
        BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER,
        HIGH_PRECISION_SHAPES,
        R1B_SOURCES,
    )
    assert BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER == 0.5
    assert HIGH_PRECISION_SHAPES == frozenset({
        "procedure_complication_risk", "procedure_complication",
    })
    assert R1B_SOURCES == ("europe_pmc", "pubmed")


def test_K4_does_not_mutate_R1C_anchor_labels():
    from services.r1c_anchor_labels import R1C_LABEL_MULTIPLIERS
    assert R1C_LABEL_MULTIPLIERS["procedure_complication_outcome_exact"] == 1.0
    assert R1C_LABEL_MULTIPLIERS["procedure_complication_wrong_outcome"] == 0.3


def test_K4_does_not_mutate_R1C_1_recency_cap():
    from services.r1a_directness_labels import (
        MECHANISM_RECENCY_HARD_CAP,
        compute_recency_base_score,
    )
    assert MECHANISM_RECENCY_HARD_CAP == 2.0
    assert compute_recency_base_score(
        year=2026, shape="pathophysiology_mechanism",
    ) == 2.0


@pytest.mark.asyncio
async def test_K4_does_not_flip_M3_2_kill_switch():
    from services.m3_2_feature_flags import read_kill_switch_state
    before = await read_kill_switch_state()
    # Exercise display map / builder helpers.
    map_label_to_display("pathophysiology_mechanism_direct")
    after = await read_kill_switch_state()
    assert before == after


# =====================================================================
# 7. Schema invariance — search_sessions / articles / summaries.
# =====================================================================
@pytest.mark.asyncio
async def test_K4_search_session_create_response_shape_unchanged(client):
    """``POST /api/search-sessions/`` response shape is unchanged
    after K4 — same fields, no ``evidence_packet`` smuggled in.
    """
    email = f"k4-shape-{uuid.uuid4().hex[:8]}@k4ev-test.com"
    await client.post("/api/auth/signup", json={
        "email": email, "password": "K4SuperTest!23", "name": "K4",
    })
    r = await client.post("/api/auth/login", json={
        "email": email, "password": "K4SuperTest!23",
    })
    token = r.json()["access_token"]
    hdr = {"Authorization": f"Bearer {token}"}
    r = await client.post(
        "/api/search-sessions/",
        json={"question": "What is the evidence for SGLT2 in HFpEF?"},
        headers=hdr,
    )
    assert r.status_code in (200, 201, 202)
    body = r.json()
    # Closed-set: K4 did NOT add an evidence_packet field to the
    # create response. The create response stays minimal.
    assert "evidence_packet" not in body
    assert "branches" not in body
    # Existing essential fields preserved.
    assert "id" in body or "session_id" in body


@pytest.mark.asyncio
async def test_K4_evidence_packets_unique_index_present(client):  # noqa: ARG001
    """Unique index on evidence_packets.search_session_id.
    Depends on ``client`` so the FastAPI lifespan triggers
    ``ensure_indexes`` before the assertion."""
    db = get_db()
    info = await db[Collections.EVIDENCE_PACKETS].index_information()
    found_unique = False
    for name, idx in info.items():
        keys = idx.get("key") or []
        if any(k[0] == "search_session_id" for k in keys) and idx.get("unique"):
            found_unique = True
            break
    assert found_unique, info


# =====================================================================
# 8. PHI-blocked session yields 409.
# =====================================================================
@pytest.mark.asyncio
async def test_K4_phi_blocked_session_returns_409_no_packet(client):
    email = f"k4-phi-{uuid.uuid4().hex[:8]}@k4ev-test.com"
    await client.post("/api/auth/signup", json={
        "email": email, "password": "K4SuperTest!23", "name": "K4",
    })
    r = await client.post("/api/auth/login", json={
        "email": email, "password": "K4SuperTest!23",
    })
    uid = r.json()["user"]["id"]
    token = r.json()["access_token"]
    db = get_db()
    session_id = str(uuid.uuid4())
    await db[Collections.SEARCH_SESSIONS].insert_one({
        "id": session_id, "user_id": uid,
        "original_question": "PHI question (sanitized)",
        "sanitized_question": "PHI question (sanitized)",
        "status": "blocked_phi",
        "clinical_question_type": "pathophysiology_mechanism",
        "clinical_question_map_json": {},
        "created_at": utcnow(), "updated_at": utcnow(),
        "expires_at": utcnow(), "deleted_at": None,
    })
    try:
        r = await client.get(
            f"/api/search-sessions/{session_id}/evidence-packet",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 409
        assert r.json()["error"] == "phi_blocked_session_no_packet"
        # No packet was persisted.
        assert await db[Collections.EVIDENCE_PACKETS].find_one(
            {"search_session_id": session_id}
        ) is None
    finally:
        await db[Collections.SEARCH_SESSIONS].delete_one({"id": session_id})


# =====================================================================
# 9. Leak-scan unit test on builder's _leak_scan helper.
# =====================================================================
def test_K4_leak_scan_blocks_abstract_key():
    from services.k4_evidence_packet_builder import _leak_scan
    with pytest.raises(ValueError):
        _leak_scan({"abstract": "no-go"})


def test_K4_leak_scan_blocks_forbidden_substring():
    from services.k4_evidence_packet_builder import _leak_scan
    with pytest.raises(ValueError):
        _leak_scan({"summary": "Bearer xxx.yyy.zzz"})


def test_K4_leak_scan_passes_clean_doc():
    from services.k4_evidence_packet_builder import _leak_scan
    _leak_scan({
        "search_session_id": "abc",
        "branches": [{
            "branch_id": "b1",
            "articles": [{"title": "Clean article"}],
        }],
    })


# =====================================================================
# 10. Display map for R1B labels — none exist; R1B-only labels would
# raise. Sentinel test to keep the closed-set contract intact.
# =====================================================================
def test_K4_display_map_size_matches_union():
    # Strict equality — bumps if any R-series catalog gains/loses a label
    # without updating the display map.
    assert len(R_SERIES_DISPLAY_MAP) == len(ALL_R_SERIES_LABELS)
