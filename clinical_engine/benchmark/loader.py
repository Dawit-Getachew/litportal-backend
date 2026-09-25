"""W1.5 benchmark: load + validate the frozen case set.

The set is frozen by content hash: any edit must bump ``benchmark_version`` and
``FROZEN_SHA256`` together, deliberately — never silently during tuning.
Held-out references are not stored in the repository (the physician keeps them).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from clinical_engine.interpretation.model_reading import QUESTION_TYPES

CASES_PATH = Path(__file__).with_name("cases_v1.json")
FROZEN_VERSION = "benchmark.v1-draft"
FROZEN_SHA256 = "67d3282ba9d0dff4e50fb59188984bc5ae5bfb3590a0b52da8a743e7ab8e423f"


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pmid: str = Field(pattern=r"^\d{1,9}$")
    title: str
    year: int
    verified: bool
    source: str
    role: str


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^W15-\d{2}$")
    question: str = Field(min_length=3)
    question_type_label: str
    engine_question_type: str
    family: str
    split: Literal["dev", "held_out"]
    split_note: str
    specialty: str
    origin: str
    reference_source: str
    evidence_label: Optional[Literal["historical", "fixture", "live"]] = None
    must_find: list[Reference] = Field(default_factory=list)
    candidate_refs: list[Reference] = Field(default_factory=list)
    hard_negatives_and_fixtures: list[Reference] = Field(default_factory=list)
    references_held_privately: bool = False
    notes: str = ""

    @field_validator("engine_question_type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in QUESTION_TYPES:
            raise ValueError(f"unknown engine question type {v}")
        return v


class BenchmarkSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_version: str
    status: str
    frozen_on: str
    verified_via: str
    must_find_semantics: str
    evidence_label_semantics: str
    cases: list[BenchmarkCase]


def content_hash(raw: dict) -> str:
    canonical = json.dumps(raw["cases"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def discipline_problems(bench: BenchmarkSet) -> list[str]:
    problems: list[str] = []
    ids = [c.id for c in bench.cases]
    if len(ids) != len(set(ids)):
        problems.append("duplicate case ids")
    splits: dict[str, set[str]] = {}
    for c in bench.cases:
        splits.setdefault(c.family, set()).add(c.split)
        refs = c.must_find + c.candidate_refs + c.hard_negatives_and_fixtures
        if c.split == "held_out" and (refs or not c.references_held_privately):
            problems.append(f"{c.id}: held-out references must not be stored in the repository")
        for r in refs:
            if not r.verified:
                problems.append(f"{c.id}: unverified reference {r.pmid}")
    for family, seen in splits.items():
        if len(seen) > 1:
            problems.append(f"family {family} appears in both dev and held-out")
    return problems


def load(path: Path = CASES_PATH) -> tuple[BenchmarkSet, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return BenchmarkSet.model_validate(raw), content_hash(raw)
