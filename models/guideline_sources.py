"""Phase K5.0 — Guideline Source Registry + permission model.

Deterministic. NO network. NO LLM. Holds the Northstar v3 §12.1 source
permission model so the system can record, per official-guideline source,
what is permitted (searchable metadata vs link-out vs displayed snippet vs
AI-ingestible full text — these are SEPARATE permissions).

Hard invariant: ``ai_ingestion_allowed`` defaults to ``False`` for every
guideline source. OA / institutional access does NOT imply AI-ingestion
permission. No source-specific AI ingestion until explicitly licensed and
approved in a later phase (K5.2+).

This registry is metadata/link-out-first. Only ``pubmed_guidelines`` is
``live`` (served through the already-approved PubMed/NCBI E-utilities path,
K5.1). USPSTF / CDC-HHS / NICE / WHO / specialty societies are scaffolded
and remain deferred.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import new_id, utcnow


# Closed-set source taxonomy (Northstar §12).
GuidelineSourceType = Literal[
    "pubmed_guideline_lane",     # PubMed-indexed guideline lane (E-utilities)
    "official_guideline_api",    # e.g. USPSTF Prevention TaskForce API
    "syndication_api",           # e.g. NICE / CDC-HHS syndication
    "source_registry",           # link-out registry first (WHO, societies)
    # K5.2C-0 — PubMed-indexed Cochrane systematic-review bibliographic lane
    # (served via the already-approved NCBI E-utilities path; metadata +
    # link-out only; NOT the Cochrane Library / Wiley full-text API).
    "systematic_review_bibliographic_lane",
]

# Closed-set connector lifecycle status.
GuidelineConnectorStatus = Literal[
    "live",            # real retrieval implemented + approved
    "scaffolded",      # interface exists; not configured / not implemented
    "not_configured",  # configured slot but missing key/access
    "deferred",        # intentionally postponed to a later phase
    # K5.2C-0 — live, but served strictly via the PubMed bibliographic lane
    # (no dedicated Cochrane/Wiley connector; no full-text/AI ingestion).
    "live_pubmed_indexed",
]

# Northstar §12.1 ``display_snippet_allowed`` is tri-state (the source's
# terms may require verification before snippets are shown).
DisplaySnippetAllowed = Literal["yes", "no", "verify"]

# How much of a source's text AI may use (B1). Fail-closed default "none".
AllowedTextScope = Literal["none", "title_only", "title_plus_abstract", "full_text"]


class GuidelineSourceDoc(BaseModel):
    """Closed-set registry row for one official-guideline source.

    The four permission planes are SEPARATE per Northstar §4 license gate:
    ``retrieval_allowed`` (searchable metadata) ≠ ``link_out_allowed`` ≠
    ``display_snippet_allowed`` ≠ ``ai_ingestion_allowed``.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    source_name: str  # unique key (e.g. "pubmed_guidelines", "uspstf")
    organization: str = ""
    source_type: GuidelineSourceType = "source_registry"

    # Connector / config state.
    enabled: bool = False
    requires_api_key: bool = False
    connector_status: GuidelineConnectorStatus = "scaffolded"

    # Northstar §12.1 permission model — four separate planes.
    retrieval_allowed: bool = False
    link_out_allowed: bool = True
    display_snippet_allowed: DisplaySnippetAllowed = "verify"
    ai_ingestion_allowed: bool = False  # HARD default false for all sources

    # B1 — granular AI permission planes. Default fail-closed. When unset, the
    # resolver derives these from ``ai_ingestion_allowed`` + per-source policy.
    ai_semantic_screening_allowed: bool = False
    ai_summary_ingestion_allowed: bool = False
    ai_ranker_optimization_allowed: bool = False
    allowed_text_scope: AllowedTextScope = "none"
    source_policy_reason: str = ""

    license_notes: str = ""
    last_checked: Optional[str] = None  # "YYYY-MM-DD"

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


__all__ = [
    "GuidelineSourceType",
    "GuidelineConnectorStatus",
    "DisplaySnippetAllowed",
    "AllowedTextScope",
    "GuidelineSourceDoc",
]
