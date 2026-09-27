"""Search terms — the second human-in-the-loop step.

After the clinician approves the interpretation (step 1), each concept is mapped
to a MeSH heading and alternative wordings. The clinician reviews and edits the
mapping (step 2) before any evidence source is searched; the approved terms are
exactly what the search uses.

Term kinds decide the default. ``broader`` and ``related`` terms are shown but OFF,
because OR-ing them in is what floods results with a sibling concept (e.g. red
cell transfusion for a whole blood question). ``equivalent`` and ``narrower``
terms are ON only when a lexical check agrees with the label (see
``synonyms.default_on``); the rest are shown OFF for the clinician to opt in.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from models.common import new_id, utcnow

SCHEMA_VERSION = "sterms.v1"

TermKind = Literal["equivalent", "narrower", "broader", "related"]
TermSource = Literal["mesh", "model", "vocabulary", "user"]
DEFAULT_ON_KINDS: frozenset[str] = frozenset({"equivalent", "narrower"})
MAX_TERMS_PER_CONCEPT = 24
MAX_TERM_CHARS = 100


class Term(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    text: str = Field(min_length=1, max_length=MAX_TERM_CHARS)
    kind: TermKind
    source: TermSource
    enabled: bool


class MeshHeading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    descriptor: str                   # preferred MeSH heading, e.g. "Shock, Hemorrhagic"
    ui: Optional[str] = None          # e.g. "D012771"
    scope_note: Optional[str] = None  # NLM definition, shown so the clinician can check the match
    # exact: an entry term of the heading IS the concept. broader: the heading of the
    # concept without its modifier ("COVID-19" for "severe COVID-19") — off by default.
    match: Literal["exact", "broader"] = "exact"
    enabled: bool = True


class TermConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    element_id: str                   # the approved interpretation element it comes from
    role: str
    label: str
    term: str                         # the clinician's concept, always searched
    required: bool                    # AND-ed into every query (else ranking only)
    mesh: Optional[MeshHeading] = None
    terms: list[Term] = Field(default_factory=list)


class TermFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    publication_years: Optional[Literal[5, 10]] = None  # None = all years
    exclude_animal_only: bool = True
    exclude_preprints: bool = True    # always on; shown for transparency


class MappingMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mesh: Literal["ncbi", "unavailable", "not_needed"] = "ncbi"
    synonyms: Literal["model", "rules_only"] = "model"
    model: Optional[str] = None
    fallback_reason: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    rejected: list[str] = Field(default_factory=list)


class SearchTerms(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["sterms.v1"] = SCHEMA_VERSION
    version: int = 1                  # bumps on every edit
    question_version: int
    use_synonyms: bool = True         # off = search each concept's own words (and MeSH) only
    concepts: list[TermConcept] = Field(default_factory=list)
    filters: TermFilters = Field(default_factory=TermFilters)
    mapping: MappingMeta = Field(default_factory=MappingMeta)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def concept(self, concept_id: str) -> Optional[TermConcept]:
        return next((c for c in self.concepts if c.id == concept_id), None)


class TermEdit(BaseModel):
    """One clinician edit to the search terms."""

    model_config = ConfigDict(extra="forbid")

    op: Literal[
        "set_use_synonyms", "set_term_enabled", "add_term", "remove_term", "set_mesh_enabled",
        "set_required", "set_publication_years", "set_exclude_animal_only",
    ]
    concept_id: Optional[str] = None
    term_id: Optional[str] = None
    text: Optional[str] = Field(default=None, max_length=MAX_TERM_CHARS)
    enabled: Optional[bool] = None
    years: Optional[Literal[0, 5, 10]] = None  # 0 = all years


def terms_hash(terms: SearchTerms) -> str:
    """What the approved search depends on (not ids, timestamps or labels)."""
    payload = {
        "use_synonyms": terms.use_synonyms,
        "filters": terms.filters.model_dump(),
        "concepts": sorted(
            (
                c.role, c.term.lower(), c.required,
                (c.mesh.descriptor if c.mesh and c.mesh.enabled else ""),
                sorted(t.text.lower() for t in c.terms if t.enabled),
            )
            for c in terms.concepts
        ),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
