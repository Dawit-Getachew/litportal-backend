"""W1.5 — the frozen benchmark set stays frozen and disciplined (pure)."""
from __future__ import annotations

from clinical_engine.benchmark.loader import FROZEN_SHA256, FROZEN_VERSION, discipline_problems, load
from clinical_engine.interpretation import rules


def test_benchmark_is_frozen():
    bench, digest = load()
    assert bench.benchmark_version == FROZEN_VERSION
    # Editing cases requires a deliberate version bump + new hash, never silent tuning.
    assert digest == FROZEN_SHA256


def test_benchmark_discipline():
    bench, _ = load()
    assert discipline_problems(bench) == []
    assert 20 <= len(bench.cases) <= 30
    held_out = [c for c in bench.cases if c.split == "held_out"]
    assert held_out and all(c.references_held_privately and not c.candidate_refs for c in held_out)
    assert any("septic shock" in c.question.lower() and "steroid" in c.question.lower() for c in bench.cases)


def test_every_case_is_interpretable_without_a_model():
    bench, _ = load()
    for case in bench.cases:
        reading = rules.read(case.question)
        assert reading.question_type  # never crashes; type may still need the model
