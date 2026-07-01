"""Research Collections — user-owned, hierarchical collections of saved papers.

Modeled on litbridge-backend's research-collection feature, adapted to MongoDB.
A collection item is mirrored into the central LitHub library (when dual-write is
enabled) and stores the returned canonical ``paper_id`` so collections are shared
with LitPulse and other Scienthesis apps. LitHub is canonical; this is the
user-facing "my collections" layer over it.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import new_id, utcnow


# ----- DB documents -----
class ResearchCollectionDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    user_id: str
    parent_id: Optional[str] = None  # hierarchical (None = root)
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    position: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class ResearchCollectionItemDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    collection_id: str
    user_id: str
    article_id: str  # local articles.id (the record id)
    search_session_id: Optional[str] = None
    title: Optional[str] = None
    notes: Optional[str] = None
    # AI-extracted table-view metadata (study design, sample size, outcomes…).
    # Populated later by an extraction pass; None until then.
    metadata_extracted: Optional[dict[str, Any]] = None
    # Canonical LitHub paper UUID returned when mirrored into the central
    # library. Lets listings enrich from LitHub and keeps the cross-service link.
    paper_id: Optional[str] = None
    # B12 — provenance snapshot captured at save time.
    source_provenance: list[str] = Field(default_factory=list)   # article sources
    permission_snapshot: Optional[dict[str, Any]] = None          # B1 resolver output
    summary_eligible: Optional[bool] = None                       # B16 (ai summary)
    ai_surfaced: bool = False                                     # was this an AI-surfaced candidate
    ranker_advisory: Optional[str] = None                         # advisory label if any
    evidence_packet_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ----- API request models -----
class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: Optional[str] = None
    icon: Optional[str] = Field(default=None, max_length=64)
    color: Optional[str] = Field(default=None, max_length=32)
    parent_id: Optional[str] = None


class CollectionUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    icon: Optional[str] = Field(default=None, max_length=64)
    color: Optional[str] = Field(default=None, max_length=32)
    position: Optional[int] = None


class AddRecordItem(BaseModel):
    article_id: str = Field(min_length=1)
    search_session_id: Optional[str] = None
    title: Optional[str] = Field(default=None, max_length=512)
    notes: Optional[str] = None
    evidence_packet_id: Optional[str] = None


class AddRecordsRequest(BaseModel):
    records: list[AddRecordItem] = Field(min_length=1, max_length=200)


class MoveRecordRequest(BaseModel):
    target_collection_id: str = Field(min_length=1)
