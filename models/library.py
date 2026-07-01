"""Phase 6D — Canonical user library + folder workspace models.

Persistent layer that survives 30-day session expiration. Three new
collections + an evidence-packet save model + a SCAFFOLDED-DISABLED
folder-summary model.

All ``id`` fields are string UUIDs (Phase 1 convention).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import new_id, utcnow


# ============================================================
# user_library_items — persistent saved-article layer
# ============================================================
LibraryItemStatus = Literal["active", "archived"]


class UserLibraryItemDoc(BaseModel):
    """One saved article per (user, article). Survives session expiry."""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    user_id: str
    article_id: str
    canonical_key: str
    note: Optional[str] = Field(default=None, max_length=4000)
    tags: list[str] = Field(default_factory=list, max_length=20)
    is_favorite: bool = False
    status: LibraryItemStatus = "active"
    archived_at: Optional[datetime] = None
    saved_from_search_session_id: Optional[str] = None
    saved_from_rank: Optional[int] = None
    saved_from_question_type: Optional[str] = None
    saved_from_specialty: Optional[str] = None
    saved_at: datetime = Field(default_factory=utcnow)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class LibraryItemCreate(BaseModel):
    article_id: str
    note: Optional[str] = Field(default=None, max_length=4000)
    tags: list[str] = Field(default_factory=list, max_length=20)
    is_favorite: bool = False
    saved_from_search_session_id: Optional[str] = None
    saved_from_rank: Optional[int] = None
    saved_from_question_type: Optional[str] = None
    saved_from_specialty: Optional[str] = None

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, v: list[str]) -> list[str]:
        if v is None:
            return []
        out: list[str] = []
        for t in v:
            if not isinstance(t, str):
                continue
            tt = t.strip()
            if tt and tt not in out:
                out.append(tt)
        return out[:20]


class LibraryItemUpdate(BaseModel):
    note: Optional[str] = Field(default=None, max_length=4000)
    tags: Optional[list[str]] = Field(default=None, max_length=20)
    is_favorite: Optional[bool] = None
    status: Optional[LibraryItemStatus] = None


class LibraryItemImport(BaseModel):
    """Import an article into the library by external identifier.

    Sources hit (closed-set): pubmed, europe_pmc, openalex, crossref.
    No Google Scholar. No publisher scraping.
    """
    pmid: Optional[str] = Field(default=None, max_length=20)
    doi: Optional[str] = Field(default=None, max_length=200)
    pmcid: Optional[str] = Field(default=None, max_length=20)
    note: Optional[str] = Field(default=None, max_length=4000)
    tags: list[str] = Field(default_factory=list, max_length=20)


class LibraryBatchSaveRequest(BaseModel):
    """Save N articles to library and (optionally) to one folder."""
    article_ids: list[str] = Field(min_length=1, max_length=200)
    folder_id: Optional[str] = None
    note: Optional[str] = Field(default=None, max_length=4000)
    tags: list[str] = Field(default_factory=list, max_length=20)
    saved_from_search_session_id: Optional[str] = None


# ============================================================
# saved_evidence_packets — Task 9 (full implement, metadata-only)
# ============================================================
class SavedEvidencePacketDoc(BaseModel):
    """Snapshot of a verified search session's article selection.

    Metadata only — no raw evidence text, no full text, no PHI. Survives
    session expiry. Saved into a folder.
    """
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    user_id: str
    folder_id: str
    title: str
    note: Optional[str] = Field(default=None, max_length=4000)
    source_search_session_id: str
    source_summary_id: Optional[str] = None
    sanitized_question: Optional[str] = None
    topic_label: Optional[str] = None
    clinical_question_type: Optional[str] = None
    article_count: int = 0
    article_ids_json: list[str] = Field(default_factory=list)
    article_canonical_keys_json: list[str] = Field(default_factory=list)
    article_titles_json: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    deleted_at: Optional[datetime] = None


class SavedEvidencePacketCreate(BaseModel):
    folder_id: str
    source_search_session_id: str
    title: str = Field(min_length=1, max_length=300)
    note: Optional[str] = Field(default=None, max_length=4000)


# ============================================================
# folder_summaries — Task 10 (SCAFFOLDED-DISABLED)
# ============================================================
class FolderSummaryDoc(BaseModel):
    """SCAFFOLDED for a future phase. Currently 501 from the router."""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    user_id: str
    folder_id: str
    title: str
    source_summary_id: str
    source_search_session_id: str
    note: Optional[str] = None
    verification_status: Literal["passed"] = "passed"
    created_at: datetime = Field(default_factory=utcnow)
    deleted_at: Optional[datetime] = None


# ============================================================
# folder_articles — EXTEND existing model with Phase 6D fields
# (the extension is added in models/folders.py)
# ============================================================
