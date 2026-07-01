"""Phase Q1 — shared benchmark types/constants extracted from
``services.search_benchmark`` to break the circular import with
``services.search_benchmark_phase5c``.

This module contains ONLY the closed-set dataclass + status constants
that both ``search_benchmark.py`` (legacy 7-case set + runner) and
``search_benchmark_phase5c.py`` (provisional Phase 5C cases) need.
No runtime logic lives here — the case fixtures, the runner, and the
phase5c provisional list stay in their respective modules.

Before Q1:
    search_benchmark.py            defines BenchmarkCase
    search_benchmark_phase5c.py    from services.search_benchmark import BenchmarkCase
    search_benchmark.py::list_all  lazy-imports search_benchmark_phase5c
                                   (circular import workaround)

After Q1:
    search_benchmark_common.py     defines BenchmarkCase + status constants  (THIS FILE)
    search_benchmark.py            from .search_benchmark_common import BenchmarkCase
    search_benchmark_phase5c.py    from .search_benchmark_common import BenchmarkCase
    search_benchmark.py::list_all  clean top-level import — no lazy workaround needed
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Phase 5C.1 — closed-set curation status values. Exported for any
# caller that wants to filter cases by curation tier without hardcoding
# the strings.
STATUS_PARSE_ONLY = "parse_only"
STATUS_NEEDS_GOLD_SET = "needs_gold_set"
STATUS_ACTIVE_BENCHMARK = "active_benchmark"
STATUS_RETIRED = "retired"

BENCHMARK_STATUS_CHOICES: frozenset[str] = frozenset({
    STATUS_PARSE_ONLY,
    STATUS_NEEDS_GOLD_SET,
    STATUS_ACTIVE_BENCHMARK,
    STATUS_RETIRED,
})


@dataclass
class BenchmarkCase:
    """Closed-set benchmark case shape shared by Phase 5A/5C/5D/6C.

    Field semantics are unchanged from the pre-Q1 definition in
    ``search_benchmark.py``. This is a pure relocation — every attribute
    keeps its old default and meaning. Tests and importers that previously
    used ``from services.search_benchmark import BenchmarkCase`` continue
    to work because ``search_benchmark`` re-exports the symbol.
    """

    id: str
    question: str
    description: str = ""
    expected_type_of_question: set[str] = field(default_factory=set)
    expected_specialty: str | None = None
    gold_pmids: list[str] = field(default_factory=list)
    must_have_anchors: list[str] = field(default_factory=list)
    negative_concepts: list[str] = field(default_factory=list)
    negative_waiver_concepts: list[str] = field(default_factory=list)
    # Phase 5A — closed-set regex patterns for title-level decoy detection.
    negative_title_patterns: list[str] = field(default_factory=list)
    # Retrieval-side pass thresholds.
    minimum_gold_in_top_10: int = 1
    minimum_gold_in_top_20: int = 1
    max_negative_in_top_10: int = 2
    # Required must-have intervention/index-test concept that, when absent
    # from ``matched_must_have``, qualifies an article as a "negative" for
    # ``negatives_in_top_10_count``. Defaults to the first must-have anchor.
    required_intervention_anchor: str | None = None
    # Phase 5C — provisional cases lack confidently-sourced gold PMIDs.
    # The runner still exercises parser/query-builder behavior on them
    # but excludes them from retrieval pass/fail aggregates.
    provisional: bool = False
    # Phase 5C — specialty tag used for cross-specialty coverage stats.
    specialty_tag: str | None = None
    # Phase 5C — expected candidate tiers that SHOULD appear among the
    # top-K candidates if expansion is healthy (informational only).
    expected_source_tiers: list[str] = field(default_factory=list)
    # Phase 5C.1 — curation status (closed set; see ``BENCHMARK_STATUS_CHOICES``).
    status: str = STATUS_ACTIVE_BENCHMARK
    updated_by_user_id: str | None = None
    updated_at: str | None = None
    # Phase 5D — expected query features (asserted against compiled queries
    # from BooleanQueryCompilerService).
    expected_query_features: dict[str, Any] = field(default_factory=dict)
    # Phase 6C — Closed-set curation flags.
    placeholder_gold: bool = False
