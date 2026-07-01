"""Phase K3B Wave 1 Step 1.5 — Soak collector tests.

Validates the K3B Wave 1 soak collector's hard invariants:

1. Only ``would_have_been_excluded_in_active == True`` rows surface.
2. ``status == "blocked_phi"`` sessions are excluded.
3. ``deleted_at != None`` sessions are excluded.
4. Forbidden sentinels (``system:``, ``sk-``, ``@gmail.com``,
   ``FORBIDDEN_RAW_QUESTION_test_sentinel``) never reach the JSONL.
5. Rows whose live join status is unstable are routed to the
   ``orphan_or_join_unstable_total`` bucket and NOT written to the
   JSONL (the gate would refuse today).
6. Non-Wave-1 reason codes (e.g. ``expression_of_concern`` alone) do
   not produce rows.
7. ``enforcement_explanation`` is normalized to ``List[str]``.
8. ``cap_enforcement_mode`` is always ``"shadow"`` (or ``"disabled"``)
   in every row — never ``"active"``.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from motor.motor_asyncio import AsyncIOMotorClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.k3b_wave1_step1_soak_collector import (  # noqa: E402
    collect,
)


def _build_diag(
    *,
    article_id: str,
    cap_reason_codes: list[str],
    article_join_status: str = "stable",
    would_have_been_excluded_in_active: bool = False,
    would_have_been_excluded_reason_codes: list[str] | None = None,
    enforcement_mode: str = "shadow",
    enforcement_explanation_full: list[Any] | None = None,
) -> dict[str, Any]:
    """Build a per-article diagnostic in the persisted-by-pipeline
    shape — mirrors the helper in
    ``tests/test_phaseK3B_wave1_step1_shadow_enforcement.py``.
    """
    if enforcement_explanation_full is None:
        enforcement_explanation_full = (
            [
                "k3b:wave1",
                "retracted_via:pubmed_retracted_publication_type",
                "join:stable",
            ]
            if would_have_been_excluded_in_active
            else ["skipped:not_wave1_eligible"]
        )
    return {
        "article_id": article_id,
        "lane_label": (
            "retracted_or_expression_of_concern"
            if any(
                c in cap_reason_codes for c in (
                    "retracted", "expression_of_concern",
                )
            )
            else "therapy_rct"
        ),
        "directness_label": "unsafe" if cap_reason_codes else "direct",
        "branch_alignment": "aligned",
        "matched_branch_ids": ["primary_0"],
        "article_intent_fit": "full",
        "cap_candidates": [],
        "cap_reason_codes": cap_reason_codes,
        "cap_severity": "high" if cap_reason_codes else "low",
        "cap_confidence": 0.95 if cap_reason_codes else 0.0,
        "cap_mode": "shadow",
        "would_cap": False,
        "would_exclude": False,
        "explanation": ["safety:retracted_or_eoc"],
        "evidence_role_candidate": (
            "to_exclude_candidate"
            if cap_reason_codes else "primary_evidence"
        ),
        "lane_confidence": 0.95,
        "qa_warnings": [],
        "article_join_status": article_join_status,
        "would_have_been_excluded_in_active":
            would_have_been_excluded_in_active,
        "would_have_been_excluded_reason_codes": list(
            would_have_been_excluded_reason_codes or []
        ),
        "enforcement_candidate_type": "retracted_only",
        "enforcement_mode": enforcement_mode,
        "enforcement_explanation_full":
            list(enforcement_explanation_full),
    }


async def _seed_session_with_diags(
    db, *,
    session_id: str,
    diagnostics: list[dict[str, Any]],
    status: str = "awaiting_selection",
    deleted_at=None,
    created_at=None,
    primary_intent: str = "therapy",
) -> str:
    """Insert a synthetic search_sessions row with the given
    diagnostics blob.
    """
    now = created_at or datetime.now(timezone.utc)
    await db.search_sessions.insert_one({
        "id": session_id,
        "user_id": "seed-user",
        "original_question": "q", "sanitized_question": "q",
        "topic_label": "t", "clinical_question_type": "therapy",
        "pico_json": {}, "picott_json": {},
        "status": status,
        "created_at": now, "updated_at": now,
        "expires_at": now + timedelta(days=30),
        "deleted_at": deleted_at,
        "session_diagnostics_json": {
            "article_lane_diagnostics": {
                "schema_version": "K3A.1.0",
                "lane_label_mode": "shadow",
                "cap_detection_mode": "shadow",
                "projection_contract": "full_v1.3",
                "diagnostics": diagnostics,
            },
        },
        "clinical_question_map_json": {
            "primary_intent": primary_intent,
        },
    })
    return session_id


async def _seed_article(db, *, article_id: str, **fields):
    """Insert a synthetic articles row with the given fields."""
    doc = {
        "id": article_id,
        "canonical_key": fields.get("canonical_key") or f"pmid:{article_id}",
        "title": fields.get("title", "Test article"),
        "journal": fields.get("journal", "Test J"),
        "publication_year": fields.get("publication_year", 2024),
        "publication_date": fields.get("publication_date"),
        "pmid": fields.get("pmid", ""),
        "doi": fields.get("doi", ""),
        "pmcid": fields.get("pmcid", ""),
        "abstract": fields.get("abstract", ""),
        "authors_json": [],
        "publication_types_json": fields.get(
            "publication_types_json", [],
        ),
        "is_guideline": False,
        "is_preprint": False,
        "is_retracted": fields.get("is_retracted", True),
        "is_retraction_notice": False,
        "has_expression_of_concern": fields.get(
            "has_expression_of_concern", False,
        ),
        "source_ids_json": {},
        "source_urls_json": [],
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    await db.articles.insert_one(doc)


async def _seed_ssa(db, *, session_id: str, article_id: str):
    """Insert a synthetic search_session_articles row."""
    await db.search_session_articles.insert_one({
        "id": uuid.uuid4().hex,
        "search_session_id": session_id,
        "article_id": article_id,
        "relevance_score": 0.0,
        "evidence_design_score": 0.0,
        "overall_rank": 0,
        "rank_explanation": None,
        "included_by_default": False,
        "user_included": False,
        "exclusion_reason": "retracted",
        "query_types_matched": [],
        "sources_present": [],
        "flags": [],
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    })


# ---------------------------------------------------------------------
# Test fixtures / scaffolding.
# ---------------------------------------------------------------------


@pytest.fixture
def soak_setup(tmp_path):
    """Provides ``(mongo_url, db_name, out_path)`` with a unique DB
    name per test. Tears the DB down after the test.
    """
    mongo_url = os.environ["MONGO_URL"]
    db_name = (
        f"test_k3b_soak_{uuid.uuid4().hex[:8]}"
    )
    out_path = tmp_path / "k3b_soak.jsonl"
    yield mongo_url, db_name, out_path
    cli = AsyncIOMotorClient(mongo_url)
    try:
        import asyncio as _asyncio
        _asyncio.get_event_loop().run_until_complete(
            cli.drop_database(db_name),
        )
    except Exception:  # noqa: BLE001
        pass
    finally:
        cli.close()


# ---------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_k3b_soak_collects_wbe_true_rows_only(soak_setup):
    """Only rows with ``would_have_been_excluded_in_active == True``
    appear in the JSONL.
    """
    mongo_url, db_name, out_path = soak_setup
    cli = AsyncIOMotorClient(mongo_url)
    db = cli[db_name]
    sid = uuid.uuid4().hex
    aid_yes = uuid.uuid4().hex
    aid_no = uuid.uuid4().hex
    await _seed_article(db, article_id=aid_yes, pmid="999991")
    await _seed_article(db, article_id=aid_no, pmid="999992")
    await _seed_ssa(db, session_id=sid, article_id=aid_yes)
    await _seed_ssa(db, session_id=sid, article_id=aid_no)
    diags = [
        _build_diag(
            article_id=aid_yes,
            cap_reason_codes=["retracted"],
            article_join_status="stable",
            would_have_been_excluded_in_active=True,
            would_have_been_excluded_reason_codes=["retracted"],
        ),
        _build_diag(
            article_id=aid_no,
            cap_reason_codes=["retracted"],
            article_join_status="stable",
            would_have_been_excluded_in_active=False,
            would_have_been_excluded_reason_codes=[],
        ),
    ]
    await _seed_session_with_diags(db, session_id=sid, diagnostics=diags)
    cli.close()
    summary = await collect(
        mongo_url=mongo_url, db_name=db_name,
        window_days=30, out_path=out_path,
        provenance="test",
    )
    assert summary["candidates_total"] == 1
    assert summary["candidates_retracted"] == 1
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["article_id"] == aid_yes
    assert row["enforcement_reason_code"] == "retracted"
    assert isinstance(row["enforcement_explanation"], list)
    assert all(isinstance(x, str) for x in row["enforcement_explanation"])
    # Hard invariants stamped.
    inv = row["shadow_invariants"]
    assert inv["cap_mode"] == "shadow"
    assert inv["would_cap"] is False
    assert inv["would_exclude"] is False
    assert inv["cap_enforcement_mode"] in {
        "shadow", "disabled", "active",
    }
    assert inv["enforcement_candidate_type"] == "retracted_only"


@pytest.mark.asyncio
async def test_k3b_soak_excludes_phi_blocked_sessions(soak_setup):
    """``status == "blocked_phi"`` sessions are NEVER scanned."""
    mongo_url, db_name, out_path = soak_setup
    cli = AsyncIOMotorClient(mongo_url)
    db = cli[db_name]
    sid = uuid.uuid4().hex
    aid = uuid.uuid4().hex
    await _seed_article(db, article_id=aid, pmid="999993")
    await _seed_ssa(db, session_id=sid, article_id=aid)
    # Inject a sentinel into the WBE row that WOULD have been picked up
    # if the session hadn't been blocked_phi.
    diag = _build_diag(
        article_id=aid,
        cap_reason_codes=["retracted"],
        article_join_status="stable",
        would_have_been_excluded_in_active=True,
        would_have_been_excluded_reason_codes=["retracted"],
        enforcement_explanation_full=[
            "k3b:wave1",
            "FORBIDDEN_RAW_QUESTION_test_sentinel_phionly",
            "join:stable",
        ],
    )
    await _seed_session_with_diags(
        db, session_id=sid, diagnostics=[diag],
        status="blocked_phi",
    )
    cli.close()
    summary = await collect(
        mongo_url=mongo_url, db_name=db_name,
        window_days=30, out_path=out_path,
        provenance="test",
    )
    # Mongo filter excludes blocked_phi at the query level so the
    # collector never even reads the row.
    assert summary["candidates_total"] == 0
    assert summary["sessions_scanned"] == 0
    assert out_path.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_k3b_soak_excludes_soft_deleted_sessions(soak_setup):
    """``deleted_at != None`` sessions are excluded."""
    mongo_url, db_name, out_path = soak_setup
    cli = AsyncIOMotorClient(mongo_url)
    db = cli[db_name]
    sid = uuid.uuid4().hex
    aid = uuid.uuid4().hex
    await _seed_article(db, article_id=aid, pmid="999994")
    await _seed_ssa(db, session_id=sid, article_id=aid)
    diag = _build_diag(
        article_id=aid,
        cap_reason_codes=["retracted"],
        article_join_status="stable",
        would_have_been_excluded_in_active=True,
        would_have_been_excluded_reason_codes=["retracted"],
    )
    await _seed_session_with_diags(
        db, session_id=sid, diagnostics=[diag],
        deleted_at=datetime.now(timezone.utc),
    )
    cli.close()
    summary = await collect(
        mongo_url=mongo_url, db_name=db_name,
        window_days=30, out_path=out_path,
        provenance="test",
    )
    assert summary["candidates_total"] == 0
    assert out_path.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_k3b_soak_orphan_articles_routed_to_unstable_bucket(
    soak_setup,
):
    """When the article doc is no longer joinable, the live join
    status is downgraded to
    ``article_removed_between_classification_and_review`` and the
    row is dropped from the JSONL (counted in
    ``orphan_or_join_unstable_total``).
    """
    mongo_url, db_name, out_path = soak_setup
    cli = AsyncIOMotorClient(mongo_url)
    db = cli[db_name]
    sid = uuid.uuid4().hex
    aid_orphan = uuid.uuid4().hex  # NO article doc, NO SSA seeded
    diag = _build_diag(
        article_id=aid_orphan,
        cap_reason_codes=["retracted"],
        article_join_status="stable",  # persisted as stable
        would_have_been_excluded_in_active=True,
        would_have_been_excluded_reason_codes=["retracted"],
    )
    await _seed_session_with_diags(
        db, session_id=sid, diagnostics=[diag],
    )
    cli.close()
    summary = await collect(
        mongo_url=mongo_url, db_name=db_name,
        window_days=30, out_path=out_path,
        provenance="test",
    )
    assert summary["orphan_or_join_unstable_total"] >= 1
    assert summary["candidates_total"] == 0
    assert out_path.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_k3b_soak_leak_guard_strips_forbidden_sentinels(soak_setup):
    """Sentinels like ``system:``, ``sk-``, ``@gmail.com`` are scrubbed
    from every row by the defense-in-depth leak guard inherited from
    the K3A.2 collector.
    """
    mongo_url, db_name, out_path = soak_setup
    cli = AsyncIOMotorClient(mongo_url)
    db = cli[db_name]
    sid = uuid.uuid4().hex
    aid = uuid.uuid4().hex
    # Poison the journal field with a forbidden sentinel — the leak
    # guard sweeps every string in the JSONL row.
    await _seed_article(
        db, article_id=aid, pmid="999995",
        journal="Bad J system:leak",  # poison
    )
    await _seed_ssa(db, session_id=sid, article_id=aid)
    diag = _build_diag(
        article_id=aid,
        cap_reason_codes=["retracted"],
        article_join_status="stable",
        would_have_been_excluded_in_active=True,
        would_have_been_excluded_reason_codes=["retracted"],
        enforcement_explanation_full=[
            "k3b:wave1",
            "system:leak",  # poison
            "user@gmail.com",  # poison
            "sk-LEAKEDKEY",  # poison
            "join:stable",
        ],
    )
    await _seed_session_with_diags(
        db, session_id=sid, diagnostics=[diag],
    )
    cli.close()
    summary = await collect(
        mongo_url=mongo_url, db_name=db_name,
        window_days=30, out_path=out_path,
        provenance="test",
    )
    assert summary["candidates_total"] == 1
    blob = out_path.read_text(encoding="utf-8")
    for sent in ("system:", "sk-", "@gmail.com"):
        assert sent not in blob, (
            f"forbidden sentinel {sent!r} leaked to soak JSONL"
        )


@pytest.mark.asyncio
async def test_k3b_soak_enforcement_mode_rejects_unknown_values(
    soak_setup,
):
    """The collector reads ``CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE``
    at start-up and asserts the value is in
    ``{"disabled", "shadow", "active"}`` (Phase K3B Wave 1 Step 2
    widened the allow-list to include ``"active"`` per operator
    approval — dev / staging only). Any value OUTSIDE this closed
    set raises ``AssertionError`` BEFORE any Mongo I/O.
    """
    mongo_url, db_name, out_path = soak_setup
    saved = os.environ.get("CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE")
    os.environ["CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE"] = (
        "definitely_not_a_real_mode"
    )
    try:
        with pytest.raises(AssertionError):
            await collect(
                mongo_url=mongo_url, db_name=db_name,
                window_days=30, out_path=out_path,
                provenance="test",
            )
    finally:
        if saved is None:
            os.environ.pop(
                "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE", None,
            )
        else:
            os.environ[
                "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE"
            ] = saved


@pytest.mark.asyncio
async def test_k3b_soak_non_wave1_reason_codes_dropped(soak_setup):
    """A WBE=True row whose enforcement reason codes are entirely
    outside the Wave-1 scope (e.g. all
    ``expression_of_concern`` and no ``retracted``) is dropped. This
    is a belt-and-suspenders check — the in-pipeline gate already
    ensures WBE=True only fires for retracted, but the soak
    collector independently asserts the reason code is in the
    Wave-1 scope.
    """
    mongo_url, db_name, out_path = soak_setup
    cli = AsyncIOMotorClient(mongo_url)
    db = cli[db_name]
    sid = uuid.uuid4().hex
    aid = uuid.uuid4().hex
    await _seed_article(db, article_id=aid, pmid="999996")
    await _seed_ssa(db, session_id=sid, article_id=aid)
    diag = _build_diag(
        article_id=aid,
        cap_reason_codes=["expression_of_concern"],
        article_join_status="stable",
        would_have_been_excluded_in_active=True,  # synthetic
        would_have_been_excluded_reason_codes=["expression_of_concern"],
    )
    await _seed_session_with_diags(
        db, session_id=sid, diagnostics=[diag],
    )
    cli.close()
    summary = await collect(
        mongo_url=mongo_url, db_name=db_name,
        window_days=30, out_path=out_path,
        provenance="test",
    )
    assert summary["candidates_total"] == 0
    assert summary["skipped_non_wave1_reason_total"] >= 1
    assert out_path.read_text(encoding="utf-8") == ""
