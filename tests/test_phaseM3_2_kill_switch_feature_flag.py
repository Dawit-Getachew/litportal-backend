"""Phase M3.2 — Mongo-backed kill switch feature-flag tests.

Asserts:
  * fail-closed default (missing row → engaged True)
  * upsert writes ``feature_flags`` row + audit event
  * env source OR mongo source engaged = combined engaged
"""
from __future__ import annotations

import pytest

from core.db import Collections, get_db
from services.m3_2_feature_flags import (
    M3_2_KILL_SWITCH_KEY,
    is_mongo_kill_engaged,
    read_kill_switch_state,
    set_mongo_kill_switch_state,
)
from services.semantic_repair_active_config import SemanticRepairActiveConfig


def _cfg(env_engaged: bool) -> SemanticRepairActiveConfig:
    return SemanticRepairActiveConfig(
        enabled=True,
        kill_switch=env_engaged,
        env_allowlist=("dev", "preview"),
        shapes_allowlist=("etiology_risk",),
        max_latency_ms=3500,
        prod_allow_active=False,
        current_env="preview",
    )


@pytest.mark.asyncio
async def test_M3_2_kill_switch_default_engaged_when_missing() -> None:
    # Clear the row first (idempotent).
    db = get_db()
    await db[Collections.FEATURE_FLAGS].delete_one(
        {"key": M3_2_KILL_SWITCH_KEY},
    )
    val = await is_mongo_kill_engaged()
    assert val is True  # fail-closed default


@pytest.mark.asyncio
async def test_M3_2_kill_switch_set_disengaged_then_engaged() -> None:
    # Set disengaged.
    set1 = await set_mongo_kill_switch_state(
        engaged=False, actor_user_id="admin-1",
    )
    assert set1["engaged"] is False
    assert set1["prev_state"] in {True, False}
    assert set1["version"] >= 1
    val = await is_mongo_kill_engaged()
    assert val is False

    # Set engaged.
    set2 = await set_mongo_kill_switch_state(
        engaged=True, actor_user_id="admin-1",
    )
    assert set2["engaged"] is True
    assert set2["prev_state"] is False
    assert set2["version"] == set1["version"] + 1
    val2 = await is_mongo_kill_engaged()
    assert val2 is True


@pytest.mark.asyncio
async def test_M3_2_kill_switch_combined_state_either_engaged() -> None:
    # Set mongo to disengaged.
    await set_mongo_kill_switch_state(engaged=False, actor_user_id="x")
    # Env engaged → combined engaged.
    state = await read_kill_switch_state(_cfg(env_engaged=True))
    assert state["env_kill_switch_engaged"] is True
    assert state["mongo_kill_switch_engaged"] is False
    assert state["combined_engaged"] is True
    # Env disengaged + mongo disengaged → combined disengaged.
    state2 = await read_kill_switch_state(_cfg(env_engaged=False))
    assert state2["combined_engaged"] is False
    # Env disengaged + mongo engaged → combined engaged.
    await set_mongo_kill_switch_state(engaged=True, actor_user_id="x")
    state3 = await read_kill_switch_state(_cfg(env_engaged=False))
    assert state3["combined_engaged"] is True


@pytest.mark.asyncio
async def test_M3_2_kill_switch_toggle_audit_event_emitted() -> None:
    """The Mongo flag toggle writes a closed-set ``m3_2_kill_switch_toggle``
    audit event with prev_state / new_state / actor."""
    db = get_db()
    # Capture before-count.
    before = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "m3_2_kill_switch_toggle"},
    )
    await set_mongo_kill_switch_state(engaged=False, actor_user_id="audit-actor")
    after = await db[Collections.AUDIT_EVENTS].count_documents(
        {"event_type": "m3_2_kill_switch_toggle"},
    )
    assert after >= before + 1
    # Inspect the most recent event.
    recent = await db[Collections.AUDIT_EVENTS].find_one(
        {"event_type": "m3_2_kill_switch_toggle"},
        sort=[("created_at", -1)],
    )
    assert recent is not None
    payload = recent.get("event_payload_json") or {}
    assert payload.get("flag_key") == M3_2_KILL_SWITCH_KEY
    assert payload.get("new_state") is False
    assert payload.get("phase") == "M3.2"
