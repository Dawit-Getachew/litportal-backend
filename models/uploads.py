"""Uploaded full-text asset schema — Phase 2D."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .common import new_id, utcnow

UploadStatus = Literal[
    "pending_review",
    "matched",
    "mismatch",
    "blocked_phi",
    "processed",
    "deleted",
    "error",
]

MatchConfidence = Literal["high", "moderate", "low", "mismatch"]

MatchedBy = Literal[
    "doi",
    "pmid",
    "pmcid",
    "title_year_journal",
    "title_similarity",
    "title_partial",
    "doi_conflict",
    "pmid_conflict",
    "user_confirmed",
    "none",
]

ExtractedTextStatus = Literal["not_started", "complete", "failed"]

RetentionMode = Literal["use_once", "session_30_days", "saved_private"]

ProcessingErrorCode = Literal[
    "not_a_pdf",
    "encrypted_pdf_rejected",
    "active_content_rejected",
    "embedded_attachment_rejected",
    "page_count_exceeded",
    "text_extraction_failed",
    "scanned_pdf_unsupported",
    "safety_scan_failed",
    "match_failed",
    "retention_invalid",
    "attestation_required",
    "server_error",
    # Phase 4B malware-scanner codes.
    "malware_detected",
    "malware_scan_error",
]

MalwareScanStatus = Literal["not_configured", "skipped", "passed", "failed"]
MetaScanStatus = Literal["not_started", "passed", "blocked"]


class UploadedFullTextAssetDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=new_id)
    user_id: str
    search_session_id: Optional[str] = None
    article_id: Optional[str] = None
    original_filename: str
    file_type: Optional[str] = None
    file_size: int = 0
    page_count: Optional[int] = None
    # storage_path is kept for backwards compatibility; new uploads use GridFS.
    storage_path: Optional[str] = None
    gridfs_file_id: Optional[str] = None
    storage_deleted_at: Optional[datetime] = None
    checksum: Optional[str] = None
    extracted_text_checksum: Optional[str] = None

    upload_status: UploadStatus = "pending_review"
    match_confidence: Optional[MatchConfidence] = None
    matched_by: MatchedBy = "none"
    extracted_text_status: ExtractedTextStatus = "not_started"
    evidence_text_level: Literal["user_uploaded_full_text"] = "user_uploaded_full_text"
    retention_mode: RetentionMode = "use_once"

    user_confirmation_required: bool = False
    user_confirmed_match_at: Optional[datetime] = None

    rights_attestation_version: Optional[str] = None
    user_attested_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Closed-set processing diagnostics.
    processing_error_code: Optional[ProcessingErrorCode] = None
    processing_error_message_safe: Optional[str] = None
    malware_scan_status: MalwareScanStatus = "not_configured"
    metadata_scan_status: MetaScanStatus = "not_started"
    annotation_scan_status: MetaScanStatus = "not_started"

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
