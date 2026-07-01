"""Article, access location, and text extract schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import new_id, utcnow

FullTextStatus = Literal[
    "full_text_available",
    "free_full_text_link",
    "abstract_only",
    "unknown",
]


class ArticleDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    canonical_key: str
    title: str
    abstract: Optional[str] = None
    journal: Optional[str] = None
    publication_year: Optional[int] = None
    publication_date: Optional[str] = None  # ISO date string
    authors_json: list[dict[str, Any]] = Field(default_factory=list)
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    source_ids_json: dict[str, Any] = Field(default_factory=dict)
    publication_types_json: list[str] = Field(default_factory=list)
    article_type_classification: Optional[str] = None
    is_guideline: bool = False
    is_preprint: bool = False
    is_retracted: bool = False
    has_expression_of_concern: bool = False
    full_text_status: FullTextStatus = "unknown"
    source_urls_json: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


AccessCategory = Literal[
    "oa_pdf",
    "oa_html",
    "pmc_oa",
    "europe_pmc_oa",
    "free_full_text_link",
    "institutional_openathens",
    "institutional_openurl",
    "semantic_scholar_institutional_linkout",
    "publisher_doi_link",
    "user_uploaded_full_text",
    "abstract_only",
    "unknown",
    # Legacy categories kept for backwards compatibility with Phase-1 rows.
    "open_access",
    "free_full_text",
    "subscription_link",
    "institutional",
]

EvidenceTextLevel = Literal[
    "abstract_only",
    "oa_full_text",
    "free_full_text_link",
    "institutional_linkout",
    "user_uploaded_full_text",
    "institution_authorized_full_text",
    "metadata_only",
    "unknown",
]


class ArticleAccessLocationDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    article_id: str
    source: str
    access_category: AccessCategory = "unknown"
    url: Optional[str] = None
    is_direct_pdf: bool = False
    is_oa: bool = False
    requires_login: bool = False
    can_view_in_app: bool = False
    can_ingest_for_summary: bool = False
    license: Optional[str] = None
    version: Optional[str] = None
    evidence_text_level: Optional[str] = None
    last_checked_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


TextSource = Literal[
    "abstract",
    "oa_full_text",
    "user_uploaded_full_text",
    "institution_authorized_full_text",
]


class ArticleTextExtractDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    article_id: str
    uploaded_full_text_asset_id: Optional[str] = None
    search_session_id: Optional[str] = None
    text_source: TextSource = "abstract"
    section_name: Optional[str] = None
    section_text: str = ""
    token_count: int = 0
    include_in_summary: bool = False
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: Optional[datetime] = None
