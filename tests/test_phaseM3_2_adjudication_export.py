"""Phase M3.2 — Adjudication Export + PATCH endpoint tests."""
from __future__ import annotations

import pytest

from core.db import Collections, get_db
from services.semantic_repair_active_pipeline import (
    M3_2_ACCEPTABLE_VALUES,
    M3_2_PHYSICIAN_VERDICTS,
    build_adjudication_export_rows,
    run_m3_2_active_decision,
    set_physician_adjudication,
)
from services.semantic_repair_active_config import (
    SemanticRepairActiveConfig,
)


def _allowed_cfg(**overrides):
    fields = dict(
        enabled=True,
        env_allowlist=("dev", "preview"),
        shapes_allowlist=(
            "etiology_risk",
            "pathophysiology_mechanism",
            "procedure_complication",
        ),
        max_latency_ms=3500,
        kill_switch=False,
        prod_allow_active=False,
        current_env="preview",
    )
    fields.update(overrides)
    return SemanticRepairActiveConfig(**fields)


def _candidate(shape: str = "etiology_risk") -> dict:
    reason = {
        "etiology_risk": "missing_risk_etiology_framework",
        "pathophysiology_mechanism": "missing_mechanism_pathophysiology_framework",
        "procedure_complication": "missing_complication_after_procedure_split",
    }[shape]
    return {
        "primary_intent": "risk_etiology",
        "frameworks": ["risk_etiology"],
        "required_anchors": ["schizophrenia"],
        "slots": {
            "population": ["young adults"],
            "condition": ["schizophrenia"],
        },
        "semantic_repair_reason": reason,
        "confidence": 0.85,
        "branches": [{"branch_id": "b1"}],
    }


async def _seed_one(session_id: str, shape: str = "etiology_risk") -> None:
    db = get_db()
    # Seed a session row so the export can join on it. Use a
    # plain clinical question that does NOT contain ``sk-`` / ``Bearer``
    # / ``eyJ`` / ``system:`` / ``assistant:`` substrings (these are
    # closed-set leak signals in the upstream scanner and would
    # otherwise fail the leak scan as false positives).
    shape_question = {
        "etiology_risk": "What are the etiology and risk factors for schizophrenia?",
        "pathophysiology_mechanism": "What is the pathophysiology of cardiogenic shock?",
        "procedure_complication": "What complications follow appendectomy?",
    }.get(shape, "What is the evidence for this clinical question?")
    await db[Collections.SEARCH_SESSIONS].update_one(
        {"id": session_id},
        {"$set": {
            "id": session_id,
            "user_id": "u-export-test",
            "status": "complete",
            "sanitized_question": shape_question,
        }},
        upsert=True,
    )
    # Seed a couple of articles.
    for rank in (1, 2):
        await db[Collections.SEARCH_SESSION_ARTICLES].insert_one({
            "id": f"ssa-{session_id}-{rank}",
            "search_session_id": session_id,
            "pmid": f"3000000{rank}",
            "doi": f"10.1234/{rank}",
            "title": f"Risk factor study {rank}",
            "journal": "BMJ",
            "publication_year": 2024,
            "source": "pubmed",
            "lane": "primary",
            "overall_rank": rank,
        })
    await db[Collections.SEARCH_QUERIES].insert_one({
        "id": f"q-{session_id}",
        "search_session_id": session_id,
        "source": "pubmed",
        "fallback_reason": None,
    })
    # Run the M3.2 decision so a sidecar row exists.
    await run_m3_2_active_decision(
        session_id=session_id,
        user_id="u-export-test",
        semantic_repair_block={
            "semantic_repair_triggered": True,
            "semantic_repair_trigger_reasons": [
                _candidate(shape)["semantic_repair_reason"],
            ],
            "semantic_repair_candidate": _candidate(shape),
            "semantic_repair_validation_status": "passed",
            "semantic_repair_rejection_reasons": [],
            "semantic_repair_diff": {},
            "semantic_repair_latency_ms": 50,
        },
        picott_payload={
            "question_subtype": f"{shape}_overview",
            "type_of_question": "background",
            "framework_list_json": ["background"],
        },
        deterministic_primary_anchors=[
            {"canonical_concept": "schizophrenia",
             "required_for_search": True},
        ],
        sources=["pubmed", "europe_pmc"],
        a_arm_cqm={"primary_intent": "background"},
        a_arm_boolean_queries=["schizophrenia"],
        config=_allowed_cfg(),
        mongo_kill_engaged=False,
    )


# =====================================================================
# build_adjudication_export_rows — basic shape + closed-set fields
# =====================================================================
@pytest.mark.asyncio
async def test_M3_2_export_returns_closed_set_fields_with_no_phi() -> None:
    await _seed_one("sess-export-001")
    rows, count, leaked = await build_adjudication_export_rows(limit=10)
    assert count >= 1
    assert leaked == 0
    r = next(r for r in rows if r.get("search_session_id") == "sess-export-001")
    assert "applied" in r
    assert "shape" in r
    assert "a_arm_cqm" in r
    assert "m_repaired_cqm" in r
    assert "a_arm_boolean_queries" in r
    assert "m_repaired_boolean_queries" in r
    assert "top_10_articles" in r
    assert "source_errors" in r
    assert "kill_switch_state" in r
    assert "physician_adjudication" in r
    assert "clinician_chip_shown" in r
    # No forbidden fields ever surface.
    forbidden = {
        "abstract", "abstract_text", "full_text", "evidence_text",
        "prompt", "completion", "api_key", "bearer", "jwt",
        "openai_api_key", "fal_api_key",
    }
    for k in forbidden:
        assert k not in r, f"row exposes forbidden field {k}"
    # Top articles are abstract-free.
    for art in r["top_10_articles"]:
        assert "abstract" not in art
        assert "full_text" not in art


@pytest.mark.asyncio
async def test_M3_2_export_filters_by_shape() -> None:
    await _seed_one("sess-export-shape-eti", shape="etiology_risk")
    await _seed_one("sess-export-shape-path", shape="pathophysiology_mechanism")
    rows_eti, _, _ = await build_adjudication_export_rows(
        limit=50, shape_filter="etiology_risk",
    )
    rows_path, _, _ = await build_adjudication_export_rows(
        limit=50, shape_filter="pathophysiology_mechanism",
    )
    eti_sessions = {r.get("search_session_id") for r in rows_eti}
    path_sessions = {r.get("search_session_id") for r in rows_path}
    assert "sess-export-shape-eti" in eti_sessions
    assert "sess-export-shape-path" in path_sessions
    # Each row's shape matches the filter.
    for r in rows_eti:
        if not r.get("row_redacted"):
            assert r["shape"] == "etiology_risk"
    for r in rows_path:
        if not r.get("row_redacted"):
            assert r["shape"] == "pathophysiology_mechanism"


@pytest.mark.asyncio
async def test_M3_2_export_filters_by_applied() -> None:
    await _seed_one("sess-export-applied-true")
    rows, _, _ = await build_adjudication_export_rows(
        limit=50, applied_filter=True,
    )
    for r in rows:
        if not r.get("row_redacted"):
            assert r["applied"] is True
    rows_false, _, _ = await build_adjudication_export_rows(
        limit=50, applied_filter=False,
    )
    for r in rows_false:
        if not r.get("row_redacted"):
            assert r["applied"] is False


# =====================================================================
# PATCH — closed-set validation + immutability of other fields
# =====================================================================
@pytest.mark.asyncio
async def test_M3_2_patch_physician_adjudication_round_trip() -> None:
    sid = "sess-export-patch-001"
    await _seed_one(sid)
    # Initial state — all None.
    rows, _, _ = await build_adjudication_export_rows(limit=10)
    row = next(r for r in rows if r.get("search_session_id") == sid)
    assert row["physician_adjudication"]["verdict"] is None
    assert row["physician_adjudication"]["acceptable_yes_no"] is None
    # PATCH all three closed-set fields.
    result = await set_physician_adjudication(
        search_session_id=sid,
        actor_user_id="admin-test",
        verdict="better",
        reason="Cleaner Boolean queries, fewer false-positive disease matches.",
        acceptable_yes_no="yes",
    )
    assert result is not None
    block = result["physician_adjudication"]
    assert block["verdict"] == "better"
    assert "Cleaner Boolean" in block["reason"]
    assert block["acceptable_yes_no"] == "yes"
    assert block["updated_by_user_id"] == "admin-test"
    # Re-export and verify the row reflects the patch.
    rows2, _, _ = await build_adjudication_export_rows(limit=10)
    row2 = next(r for r in rows2 if r.get("search_session_id") == sid)
    assert row2["physician_adjudication"]["verdict"] == "better"
    assert row2["physician_adjudication"]["acceptable_yes_no"] == "yes"


@pytest.mark.asyncio
async def test_M3_2_patch_rejects_out_of_set_verdict() -> None:
    sid = "sess-export-patch-002"
    await _seed_one(sid)
    with pytest.raises(ValueError) as exc:
        await set_physician_adjudication(
            search_session_id=sid,
            actor_user_id="admin-test",
            verdict="great_job",
        )
    assert "verdict" in str(exc.value)


@pytest.mark.asyncio
async def test_M3_2_patch_rejects_out_of_set_acceptable() -> None:
    sid = "sess-export-patch-003"
    await _seed_one(sid)
    with pytest.raises(ValueError) as exc:
        await set_physician_adjudication(
            search_session_id=sid,
            actor_user_id="admin-test",
            acceptable_yes_no="maybe",
        )
    assert "acceptable_yes_no" in str(exc.value)


@pytest.mark.asyncio
async def test_M3_2_patch_truncates_long_reason() -> None:
    sid = "sess-export-patch-004"
    await _seed_one(sid)
    long_reason = "x" * 5000
    result = await set_physician_adjudication(
        search_session_id=sid,
        actor_user_id="admin-test",
        reason=long_reason,
    )
    # 1200 char hard cap.
    assert len(result["physician_adjudication"]["reason"]) <= 1200


@pytest.mark.asyncio
async def test_M3_2_patch_returns_none_when_no_sidecar() -> None:
    result = await set_physician_adjudication(
        search_session_id="sess-does-not-exist",
        actor_user_id="admin-test",
        verdict="same",
    )
    assert result is None


@pytest.mark.asyncio
async def test_M3_2_patch_does_NOT_mutate_deterministic_fields() -> None:
    """The PATCH must NEVER mutate a_arm_cqm, m_repaired_cqm,
    a_arm_boolean_queries, m_repaired_boolean_queries,
    fallback_status, latency_ms, kill_switch_state, or
    config_snapshot. Verified by snapshot-comparing before/after."""
    sid = "sess-export-patch-immut"
    await _seed_one(sid)
    db = get_db()
    before = await db[Collections.M3_2_SESSION_DECISIONS].find_one(
        {"search_session_id": sid},
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    await set_physician_adjudication(
        search_session_id=sid,
        actor_user_id="admin-test",
        verdict="worse",
        reason="Worse retrieval — added irrelevant terms.",
        acceptable_yes_no="no",
    )
    after = await db[Collections.M3_2_SESSION_DECISIONS].find_one(
        {"search_session_id": sid},
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    # Immutable fields — defense-in-depth.
    for field in (
        "a_arm_cqm",
        "m_repaired_cqm",
        "a_arm_boolean_queries",
        "m_repaired_boolean_queries",
        "m_repaired_compiler_input",
        "fallback_status",
        "timeout_status",
        "latency_ms",
        "kill_switch_state",
        "config_snapshot",
        "eligibility_decision",
        "shape",
        "semantic_repair_active_applied",
        "route_owned_by_k4a_specialty",
    ):
        assert before.get(field) == after.get(field), (
            f"Immutable field `{field}` was mutated by PATCH"
        )


# =====================================================================
# Closed-set enum membership tests
# =====================================================================
def test_M3_2_physician_verdict_closed_set() -> None:
    assert M3_2_PHYSICIAN_VERDICTS == frozenset({
        "better", "same", "worse", "not_adjudicable",
    })


def test_M3_2_acceptable_closed_set() -> None:
    assert M3_2_ACCEPTABLE_VALUES == frozenset({"yes", "no"})
