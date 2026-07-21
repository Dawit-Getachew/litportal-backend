"""Unit tests for the Agent-3b rescued-article RE-ADD path (maybe_rescue_readd).

These lock the production-safety invariants:
  * flag OFF => byte-identical (no mutation of ``ranked``)
  * only high-confidence 'direct_candidate' rows are re-added
  * hard/safety exclusions are NEVER re-added (blocked in _collect_candidates)
  * the wrong-disease (central-anchor) guard blocks promotion
  * MAX_ADDS / MIN_CONFIDENCE honored
  * production requires the second opt-in (AI_RESCUE_READD_PROD_ALLOWED)
  * AUTOCHECK toggles included_by_default; SKIP_IF_INCLUDED_ATLEAST skips the call

All LLM / DB / source-policy dependencies are mocked; no Mongo required.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("DB_NAME", "test_dummy")
os.environ.setdefault("APP_ENV", "development")

import agents.evidence_curator as ec  # noqa: E402
import services.downstream_eligibility as de  # noqa: E402
from agents.schemas import EvidenceCurationResult, SurfacedCandidate  # noqa: E402
from services.default_selection import recount_summary_after_rescue  # noqa: E402

PICOTT = {"must_have_concepts": [{"role": "condition", "concept": "sepsis", "synonyms": []}]}


def _settings(**over):
    base = dict(
        AI_RESCUE_READD_ENABLED=True,
        AI_RESCUE_READD_PROD_ALLOWED=False,
        AI_RESCUE_READD_MIN_CONFIDENCE=0.75,
        AI_RESCUE_READD_MAX_CANDIDATES=30,
        AI_RESCUE_READD_MAX_ADDS=15,
        AI_RESCUE_READD_AUTOCHECK=True,
        AI_RESCUE_READD_SKIP_IF_INCLUDED_ATLEAST=0,
        APP_ENV="development",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _row(aid, excl="low_relevance", title="sepsis trial", abstract="sepsis outcomes"):
    return {
        "_article": {"id": aid, "title": title, "abstract": abstract},
        "exclusion_reason": excl,
        "included_by_default": False,
    }


def _shown(aid):
    return {"_article": {"id": aid, "title": "t", "abstract": "a"}, "exclusion_reason": None}


def _sc(index, label="direct_candidate", conf=0.9):
    return SurfacedCandidate(index=index, advisory_label=label, confidence=conf, rationale="r")


@pytest.fixture(autouse=True)
def _patch_common(monkeypatch):
    monkeypatch.setattr(de, "can_use_article_for_tool", lambda art, tool: {"allowed": True})

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(ec.audit_service, "log", _noop)


def _mock_llm(monkeypatch, surfaced, counter=None):
    result = EvidenceCurationResult(surfaced=surfaced)

    async def _run(**kwargs):
        if counter is not None:
            counter["n"] += 1
        return result, None

    monkeypatch.setattr(ec, "run_json_agent", _run)


async def test_flag_off_no_mutation(monkeypatch):
    monkeypatch.setattr(ec, "get_settings", lambda: _settings(AI_RESCUE_READD_ENABLED=False))
    counter = {"n": 0}
    _mock_llm(monkeypatch, [_sc(0)], counter)
    ranked = [_row("a1")]
    added = await ec.maybe_rescue_readd("s", "u", "sepsis?", ranked, PICOTT)
    assert added == 0
    assert ranked[0]["exclusion_reason"] == "low_relevance"
    assert ranked[0].get("llm_rescued") is None
    assert counter["n"] == 0  # LLM never called when disabled


async def test_direct_candidate_readded(monkeypatch):
    monkeypatch.setattr(ec, "get_settings", lambda: _settings())
    _mock_llm(monkeypatch, [_sc(0, conf=0.8)])
    ranked = [_row("a1")]
    added = await ec.maybe_rescue_readd("s", "u", "sepsis?", ranked, PICOTT)
    assert added == 1
    assert ranked[0]["exclusion_reason"] is None
    assert ranked[0]["included_by_default"] is True
    assert ranked[0]["llm_rescued"] is True
    assert ranked[0]["default_selection_rationale"] == "llm_rescued"


async def test_autocheck_off_surfaces_unchecked(monkeypatch):
    monkeypatch.setattr(ec, "get_settings", lambda: _settings(AI_RESCUE_READD_AUTOCHECK=False))
    _mock_llm(monkeypatch, [_sc(0, conf=0.8)])
    ranked = [_row("a1")]
    added = await ec.maybe_rescue_readd("s", "u", "q", ranked, PICOTT)
    assert added == 1
    assert ranked[0]["exclusion_reason"] is None
    assert ranked[0]["included_by_default"] is False  # surfaced but not pre-checked


async def test_below_confidence_not_readded(monkeypatch):
    monkeypatch.setattr(ec, "get_settings", lambda: _settings())
    _mock_llm(monkeypatch, [_sc(0, conf=0.70)])  # below 0.75 floor
    ranked = [_row("a1")]
    added = await ec.maybe_rescue_readd("s", "u", "q", ranked, PICOTT)
    assert added == 0
    assert ranked[0]["exclusion_reason"] == "low_relevance"


async def test_only_direct_label_readded(monkeypatch):
    monkeypatch.setattr(ec, "get_settings", lambda: _settings())
    _mock_llm(monkeypatch, [_sc(0, label="borderline_candidate", conf=0.99)])
    ranked = [_row("a1")]
    added = await ec.maybe_rescue_readd("s", "u", "q", ranked, PICOTT)
    assert added == 0  # borderline/indirect are not re-added, only direct


async def test_hard_exclusion_never_readded(monkeypatch):
    monkeypatch.setattr(ec, "get_settings", lambda: _settings())
    _mock_llm(monkeypatch, [_sc(0, conf=0.99)])
    ranked = [_row("a1", excl="excluded_retracted_or_preprint_or_unsafe")]
    added = await ec.maybe_rescue_readd("s", "u", "q", ranked, PICOTT)
    assert added == 0
    assert ranked[0]["exclusion_reason"] == "excluded_retracted_or_preprint_or_unsafe"


async def test_wrong_disease_guard(monkeypatch):
    monkeypatch.setattr(ec, "get_settings", lambda: _settings())
    _mock_llm(monkeypatch, [_sc(0, conf=0.99)])
    ranked = [_row("a1", title="diabetes review", abstract="glucose control")]
    added = await ec.maybe_rescue_readd("s", "u", "sepsis?", ranked, PICOTT)
    assert added == 0  # lacks the central 'sepsis' anchor


async def test_max_adds_cap(monkeypatch):
    monkeypatch.setattr(ec, "get_settings", lambda: _settings(AI_RESCUE_READD_MAX_ADDS=2))
    ranked = [_row(f"a{i}") for i in range(5)]
    _mock_llm(monkeypatch, [_sc(i, conf=0.9) for i in range(5)])
    added = await ec.maybe_rescue_readd("s", "u", "q", ranked, PICOTT)
    assert added == 2
    assert sum(1 for r in ranked if r["exclusion_reason"] is None) == 2


async def test_prod_requires_opt_in(monkeypatch):
    monkeypatch.setattr(
        ec, "get_settings",
        lambda: _settings(APP_ENV="production", AI_RESCUE_READD_PROD_ALLOWED=False),
    )
    _mock_llm(monkeypatch, [_sc(0, conf=0.9)])
    ranked = [_row("a1")]
    added = await ec.maybe_rescue_readd("s", "u", "q", ranked, PICOTT)
    assert added == 0


async def test_prod_opt_in_allows(monkeypatch):
    monkeypatch.setattr(
        ec, "get_settings",
        lambda: _settings(APP_ENV="production", AI_RESCUE_READD_PROD_ALLOWED=True),
    )
    _mock_llm(monkeypatch, [_sc(0, conf=0.9)])
    ranked = [_row("a1")]
    added = await ec.maybe_rescue_readd("s", "u", "q", ranked, PICOTT)
    assert added == 1
    assert ranked[0]["exclusion_reason"] is None


async def test_skip_when_enough_shown(monkeypatch):
    monkeypatch.setattr(
        ec, "get_settings",
        lambda: _settings(AI_RESCUE_READD_SKIP_IF_INCLUDED_ATLEAST=3),
    )
    ranked = [_row("a1"), _shown("b1"), _shown("b2"), _shown("b3")]
    counter = {"n": 0}
    _mock_llm(monkeypatch, [_sc(0, conf=0.9)], counter)
    added = await ec.maybe_rescue_readd("s", "u", "q", ranked, PICOTT)
    assert added == 0
    assert counter["n"] == 0  # LLM skipped because the shown set is already healthy


async def test_no_anchor_is_permissive(monkeypatch):
    monkeypatch.setattr(ec, "get_settings", lambda: _settings())
    _mock_llm(monkeypatch, [_sc(0, conf=0.9)])
    ranked = [_row("a1", title="unrelated topic", abstract="text")]
    added = await ec.maybe_rescue_readd("s", "u", "q", ranked, {})  # no must_have_concepts
    assert added == 1  # cannot validate anchor => do not block


# ── Fix A: default_selection_summary recount after rescue ────────────────────

def _recount_row(aid, *, rel, included, rescued, cap=None):
    return {
        "_article": {"id": aid, "title": "sepsis", "abstract": "sepsis outcomes"},
        "relevance_score": rel,
        "role_match_score": 100,
        "cap_reasons": list(cap or []),
        "included_by_default": included,
        "llm_rescued": rescued,
        "exclusion_reason": None if (included or rescued) else "low_relevance",
    }


def test_recount_noop_when_nothing_rescued():
    summary = {"selected_count": 1, "excluded_count": 3}
    ranked = [_recount_row("s1", rel=90, included=True, rescued=False)]
    out = recount_summary_after_rescue(summary, ranked)
    assert out == summary  # unchanged
    assert out is not None


def test_recount_updates_counts_for_low_relevance_rescue():
    summary = {
        "selected_count": 1,
        "selected_article_ids": ["s1"],
        "excluded_count": 3,
        "excluded_rationale_summary": {"relevance_below_60": 3},
        "excluded_by_low_relevance_count": 3,
        "excluded_by_cap_count": 0,
        "reason_each_selected": [{"article_id": "s1", "reason": "tier=guideline"}],
        "selected_by_lane": {"primary": 1},
    }
    ranked = [
        _recount_row("s1", rel=90, included=True, rescued=False),
        _recount_row("r1", rel=30, included=True, rescued=True),   # rescued low_relevance
        _recount_row("r2", rel=25, included=True, rescued=True),   # rescued low_relevance
        _recount_row("x1", rel=20, included=False, rescued=False),  # still excluded
    ]
    out = recount_summary_after_rescue(summary, ranked)
    assert out["selected_count"] == 3
    assert set(out["selected_article_ids"]) == {"s1", "r1", "r2"}
    assert out["excluded_rationale_summary"]["relevance_below_60"] == 1  # 3 - 2
    assert out["excluded_count"] == 1
    assert out["excluded_by_low_relevance_count"] == 1
    assert out["llm_rescued_count"] == 2
    # rescued rows got a closed-set rationale appended
    reasons = {x["article_id"]: x["reason"] for x in out["reason_each_selected"]}
    assert reasons["r1"] == "llm_rescued"
    assert reasons["r2"] == "llm_rescued"
    # input not mutated
    assert summary["selected_count"] == 1


def test_recount_decrements_cap_reason():
    summary = {
        "selected_count": 0,
        "excluded_count": 2,
        "excluded_rationale_summary": {"cap_reason_wrong_drug": 2},
        "excluded_by_cap_count": 2,
        "excluded_by_low_relevance_count": 0,
    }
    ranked = [
        _recount_row("r1", rel=95, included=True, rescued=True, cap=["wrong_drug"]),
        _recount_row("x1", rel=95, included=False, rescued=False, cap=["wrong_drug"]),
    ]
    out = recount_summary_after_rescue(summary, ranked)
    assert out["excluded_by_cap_count"] == 1
    assert out["excluded_rationale_summary"]["cap_reason_wrong_drug"] == 1
    assert out["selected_count"] == 1
    assert out["llm_rescued_count"] == 1
