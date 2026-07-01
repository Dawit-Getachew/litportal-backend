"""Unit tests for the three LitPortal AI agents.

No live LLM and no Mongo: the LLM call and the DB writes are monkeypatched, so
these test the AGENT LOGIC (flag gating, schema clamping, the conservative
type-override rule, governance filters for surfacing). pytest.ini sets
``asyncio_mode = auto`` so async tests need no marker.
"""
from __future__ import annotations

import core.config as config
from agents import evidence_curator, query_categorizer, question_interpreter
from agents.schemas import (
    EvidenceCurationResult,
    QueryCategoryResult,
    QuestionUnderstanding,
)


# ── schema clamping / coercion (the fail-soft contract) ─────────────
def test_query_category_clamps_and_coerces():
    m = QueryCategoryResult.model_validate({
        "question_type_hint": "NOT_A_REAL_TYPE",
        "confidence": 5.0,  # out of range
        "enriched_terms": ["a", 1, "  ", "b"],
        "entities": [{"text": "warfarin", "kind": "DRUG"}],
    })
    assert m.question_type_hint == "other"      # unknown type -> safe default
    assert m.confidence == 1.0                  # clamped to [0,1]
    assert m.enriched_terms == ["a", "1", "b"]  # blanks dropped, coerced to str
    assert m.entities[0].kind == "drug"         # kind normalized


def test_surfaced_label_and_index_coerce():
    r = EvidenceCurationResult.model_validate({
        "surfaced": [
            {"index": "2", "advisory_label": "DIRECT_CANDIDATE", "confidence": -1},
            {"index": "x", "advisory_label": "garbage", "confidence": 0.9},
        ],
    })
    assert r.surfaced[0].index == 2
    assert r.surfaced[0].advisory_label == "direct_candidate"
    assert r.surfaced[0].confidence == 0.0          # clamped
    assert r.surfaced[1].index == -1                # bad int -> -1 (skipped downstream)
    assert r.surfaced[1].advisory_label == "keep_excluded"  # unknown -> safe default


# ── Agent 1: query categorizer override rule ────────────────────────
def _fake_run(result):
    async def _run(**kwargs):
        return result, None
    return _run


def _silence(monkeypatch, mod):
    async def _noop(**kwargs):
        return None
    monkeypatch.setattr(mod, "store_agent_decision", _noop)

    async def _log(*a, **k):
        return ""
    monkeypatch.setattr(mod.audit_service, "log", _log)


async def test_categorizer_overrides_only_weak_type(monkeypatch):
    monkeypatch.setattr(config.Settings, "AI_QUERY_CATEGORIZER_ENABLED", True)
    monkeypatch.setattr(config.Settings, "AI_QUERY_CATEGORIZER_MIN_CONFIDENCE", 0.75)
    _silence(monkeypatch, query_categorizer)
    monkeypatch.setattr(
        query_categorizer, "run_json_agent",
        _fake_run(QueryCategoryResult(question_type_hint="pharmacology", confidence=0.9)),
    )
    # Deterministic said "other" + AI confident -> override applied.
    parsed = {"clinical_question_type": "other"}
    await query_categorizer.maybe_categorize("s1", "u1", "warfarin and amiodarone?", parsed)
    assert parsed["clinical_question_type"] == "pharmacology"
    assert parsed["ai_query_categorizer"]["applied"] is True


async def test_categorizer_never_overrides_confident_deterministic(monkeypatch):
    monkeypatch.setattr(config.Settings, "AI_QUERY_CATEGORIZER_ENABLED", True)
    _silence(monkeypatch, query_categorizer)
    monkeypatch.setattr(
        query_categorizer, "run_json_agent",
        _fake_run(QueryCategoryResult(question_type_hint="harm", confidence=0.99)),
    )
    parsed = {"clinical_question_type": "treatment"}  # deterministic already decided
    await query_categorizer.maybe_categorize("s1", "u1", "q", parsed)
    assert parsed["clinical_question_type"] == "treatment"  # untouched
    assert parsed["ai_query_categorizer"]["applied"] is False


async def test_categorizer_low_confidence_no_override(monkeypatch):
    monkeypatch.setattr(config.Settings, "AI_QUERY_CATEGORIZER_ENABLED", True)
    monkeypatch.setattr(config.Settings, "AI_QUERY_CATEGORIZER_MIN_CONFIDENCE", 0.75)
    _silence(monkeypatch, query_categorizer)
    monkeypatch.setattr(
        query_categorizer, "run_json_agent",
        _fake_run(QueryCategoryResult(question_type_hint="diagnosis", confidence=0.5)),
    )
    parsed = {"clinical_question_type": "other"}
    await query_categorizer.maybe_categorize("s1", "u1", "q", parsed)
    assert parsed["clinical_question_type"] == "other"  # below threshold


async def test_categorizer_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(config.Settings, "AI_QUERY_CATEGORIZER_ENABLED", False)
    parsed = {"clinical_question_type": "other"}
    await query_categorizer.maybe_categorize("s1", "u1", "q", parsed)
    assert parsed == {"clinical_question_type": "other"}  # completely untouched


# ── Agent 2: question understanding never mutates parsed ─────────────
async def test_interpreter_is_storage_only(monkeypatch):
    monkeypatch.setattr(config.Settings, "AI_CONTEXT_AUGMENT_ENABLED", True)
    _silence(monkeypatch, question_interpreter)
    monkeypatch.setattr(
        question_interpreter, "run_json_agent",
        _fake_run(QuestionUnderstanding(intent_hint="dosing", confidence=0.9)),
    )
    parsed = {"clinical_question_type": "treatment", "picott_json": {"population": "adults"}}
    before = dict(parsed)
    await question_interpreter.maybe_interpret("s1", "u1", "q", parsed, parsed["picott_json"])
    assert parsed == before  # diagnostics only — never mutates the parse


# ── Agent 3: surfacing governance (hard exclusions, confidence, no mutation) ──
class _FakeColl:
    def __init__(self, sink):
        self._sink = sink

    async def insert_one(self, doc):
        self._sink.append(doc)


class _FakeDB:
    def __init__(self, sink):
        self._sink = sink

    def __getitem__(self, _name):
        return _FakeColl(self._sink)


async def test_curator_surfaces_soft_excluded_only(monkeypatch):
    monkeypatch.setattr(config.Settings, "AI_EVIDENCE_SURFACING_ENABLED", True)
    monkeypatch.setattr(config.Settings, "MIN_AI_SURFACING_CONFIDENCE", 0.65)
    monkeypatch.setattr(config.Settings, "MAX_AI_SURFACING_CANDIDATES", 30)
    _silence(monkeypatch, evidence_curator)
    sink: list[dict] = []
    monkeypatch.setattr(evidence_curator, "get_db", lambda: _FakeDB(sink))

    src = ["pubmed"]  # AI-permitted source so the B16 filter lets candidates through
    ranked = [
        {"exclusion_reason": None, "_article": {"id": "A_included", "title": "t", "abstract": "x", "sources_present": src}},
        {"exclusion_reason": "relevance_below_60", "_article": {"id": "A_soft", "title": "good review", "abstract": "x", "sources_present": src}},
        {"exclusion_reason": "retracted_publication", "_article": {"id": "A_hard", "title": "retracted", "abstract": "x", "sources_present": src}},
    ]
    ranked_snapshot = [dict(r) for r in ranked]

    # Candidate list passed to the agent contains only soft-excluded -> index 0 == A_soft.
    result = EvidenceCurationResult(
        surfaced=[{"index": 0, "advisory_label": "direct_candidate", "confidence": 0.9}],
        broader_guideline_topics=["anticoagulation"],
    )
    monkeypatch.setattr(evidence_curator, "run_json_agent", _fake_run(result))

    await evidence_curator.maybe_curate_evidence("s1", "u1", "q", ranked)

    surfaced_ids = [d["article_id"] for d in sink]
    assert surfaced_ids == ["A_soft"]          # soft surfaced
    assert "A_hard" not in surfaced_ids         # retracted NEVER surfaced
    assert "A_included" not in surfaced_ids     # included articles are not candidates
    assert ranked == ranked_snapshot            # deterministic list untouched


async def test_curator_respects_confidence_floor(monkeypatch):
    monkeypatch.setattr(config.Settings, "AI_EVIDENCE_SURFACING_ENABLED", True)
    monkeypatch.setattr(config.Settings, "MIN_AI_SURFACING_CONFIDENCE", 0.8)
    monkeypatch.setattr(config.Settings, "MAX_AI_SURFACING_CANDIDATES", 30)
    _silence(monkeypatch, evidence_curator)
    sink: list[dict] = []
    monkeypatch.setattr(evidence_curator, "get_db", lambda: _FakeDB(sink))
    ranked = [
        {"exclusion_reason": "cap_reason_wrong_population",
         "_article": {"id": "A1", "title": "t", "abstract": "x", "sources_present": ["pubmed"]}},
    ]
    result = EvidenceCurationResult(
        surfaced=[{"index": 0, "advisory_label": "borderline_candidate", "confidence": 0.7}],
    )
    monkeypatch.setattr(evidence_curator, "run_json_agent", _fake_run(result))
    await evidence_curator.maybe_curate_evidence("s1", "u1", "q", ranked)
    assert sink == []  # 0.7 < 0.8 floor -> not surfaced


async def test_curator_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(config.Settings, "AI_EVIDENCE_SURFACING_ENABLED", False)
    # No monkeypatching of run_json_agent: if it were called it'd hit the real
    # LLM path (not configured -> None), but disabled means it returns early.
    await evidence_curator.maybe_curate_evidence("s1", "u1", "q", [
        {"exclusion_reason": "relevance_below_60", "_article": {"id": "A1"}},
    ])
