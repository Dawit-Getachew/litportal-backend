"""Phase 6G — clinician-facing PICOTT term mapping schema.

Closed-set Pydantic models for the ``picott_term_mapping_json`` field
returned on GET /api/search-sessions/{id}. Composed as a *pure projection*
from already-persisted session artifacts (primary anchors, PICOTT,
role repairs, ontology lifts, concept translations, multi-intent, boolean
compiler concept blocks). NO LLM, NO external API, NO new search behavior.

All models use ``extra="forbid"`` so unknown keys raise immediately — this
guards against PHI / prompt / completion / URL-with-querystring leakage
into the clinician-facing payload.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Closed-set PICOTT role identifiers. These map 1:1 to the keys of
# ``PicottTermMapping.fields`` and to the user-facing label table in
# the front-end.
PicottRole = Literal[
    "population_problem",
    "intervention_exposure_index_test",
    "comparator_reference_standard",
    "outcomes_target_diagnosis",
    "time_horizon",
    "setting",
    "clinical_action",
]

PICOTT_ROLES: tuple[str, ...] = (
    "population_problem",
    "intervention_exposure_index_test",
    "comparator_reference_standard",
    "outcomes_target_diagnosis",
    "time_horizon",
    "setting",
    "clinical_action",
)

# Closed-set provenance tags for each TermMapEntry.
TermMapSource = Literal[
    "primary_anchor",
    "ontology_lift",
    "role_repair",
    "concept_translation",
    "picott_parser",
    "boolean_block",
]


class TermMapEntry(BaseModel):
    """One mapped clinician-facing PICOTT term.

    NO raw user text fields, NO URLs with query strings, NO prompts,
    NO PHI markers. Free-text fields are sourced exclusively from
    server-derived canonical concepts and rationales (already
    sanitized by upstream services).
    """

    model_config = ConfigDict(extra="forbid")

    original_phrase: str = ""
    canonical_concept: str = ""
    role: PicottRole
    synonyms_used: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    source: TermMapSource
    required_for_high_relevance: bool = False
    query_block: Optional[str] = None
    confidence: float = 0.0
    rationale: str = ""


class CompletenessBlock(BaseModel):
    """Completeness guard verdict for the mapped question."""

    model_config = ConfigDict(extra="forbid")

    has_primary_anchor: bool = False
    has_required_role_mapping: bool = False
    missing_required_roles: list[str] = Field(default_factory=list)
    safe_to_search: bool = False
    clarification_needed: bool = False
    clarification_message: Optional[str] = None


class PicottTermMapping(BaseModel):
    """Top-level mapping object surfaced on the GET /search-sessions/{id}
    response under ``picott_term_mapping_json``.

    Constructed by ``services.picott_term_mapping_service.build_mapping``
    as a pure projection over the persisted session document.
    """

    model_config = ConfigDict(extra="forbid")

    question_scope: str = ""
    type_of_question: str = ""
    question_subtype: Optional[str] = None
    specialty: Optional[str] = None
    fields: dict[str, list[TermMapEntry]] = Field(default_factory=dict)
    primary_clinical_anchors: list[dict] = Field(default_factory=list)
    generic_modifiers: list[dict] = Field(default_factory=list)
    must_have_concepts: list[dict] = Field(default_factory=list)
    ranking_concepts: list[str] = Field(default_factory=list)
    negative_concepts: list[dict] = Field(default_factory=list)
    query_blocks: list[dict] = Field(default_factory=list)
    completeness: CompletenessBlock = Field(default_factory=CompletenessBlock)
    # Phase 6G.4P — post-procedure management routing fields.
    # ``subdomain`` is one of ``services.post_procedure_router.SUBDOMAINS``
    # when applicable, ``None`` otherwise. ``refinement_note`` is a
    # closed-set, server-generated clinician-facing string (never LLM
    # generated). ``post_procedure_routing`` is the structured
    # routing payload (closed-set fields) used by the transparency
    # panel + admin diagnostics.
    subdomain: Optional[str] = None
    refinement_note: Optional[str] = None
    post_procedure_routing: dict = Field(default_factory=dict)
