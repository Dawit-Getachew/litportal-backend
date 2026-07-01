"""Folder + folder_article schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import new_id, utcnow


class FolderDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    user_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class FolderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)


class FolderUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)


class FolderArticleDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    folder_id: str
    article_id: str
    user_notes: Optional[str] = None
    saved_from_search_session_id: Optional[str] = None
    # Phase 6D — folder-item extensions. Optional so existing rows
    # round-trip unchanged.
    from_library_item_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    pinned_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)


class FolderArticleCreate(BaseModel):
    article_id: str
    user_notes: Optional[str] = Field(default=None, max_length=4000)
    saved_from_search_session_id: Optional[str] = None
    from_library_item_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list, max_length=20)


class FolderArticleBatchCreate(BaseModel):
    """Phase 6D — batch-add N articles to a folder in one request."""
    article_ids: list[str] = Field(min_length=1, max_length=200)
    user_notes: Optional[str] = Field(default=None, max_length=4000)
    saved_from_search_session_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list, max_length=20)


class FolderArticleUpdate(BaseModel):
    user_notes: Optional[str] = Field(default=None, max_length=4000)
    tags: Optional[list[str]] = Field(default=None, max_length=20)
    pinned_at: Optional[datetime] = None
