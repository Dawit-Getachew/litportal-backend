"""Uploads subpackage — Phase 2D user-uploaded full-text PDF.

NEVER scrapes. NEVER downloads paywalled PDFs. NEVER stores credentials.
NEVER calls an LLM. User-attested, scoped to {user_id, search_session_id,
article_id}. Subscription full text NEVER becomes a global asset.
"""
from __future__ import annotations

RIGHTS_ATTESTATION_VERSION = "v1-2025"

RIGHTS_ATTESTATION_TEXT = (
    "I confirm that I have lawful access to this article and that my license, "
    "institutional policy, or applicable law permits me to upload and process "
    "this PDF in this tool for my own evidence review. I understand this file "
    "will not be redistributed."
)

# Closed user-facing messages — never embed raw extracted text in any error.
USER_MESSAGES = {
    "not_a_pdf": "This file is not a valid PDF.",
    "encrypted_pdf_rejected": "Password-protected PDFs cannot be processed. Please upload an unencrypted copy.",
    "active_content_rejected": "PDFs containing JavaScript actions are not accepted.",
    "embedded_attachment_rejected": "PDFs containing embedded attachments are not accepted.",
    "page_count_exceeded": "This PDF exceeds the page limit. Please upload a shorter document.",
    "text_extraction_failed": "Text could not be extracted from this PDF. Scanned/image-only PDFs are not supported.",
    "scanned_pdf_unsupported": "Scanned (image-only) PDFs are not supported in this phase.",
    "safety_scan_failed": "This PDF appears to contain private or identifying information. Please upload a clean, non-annotated article PDF.",
    "match_failed": "This PDF does not appear to match the selected article. Please upload the correct PDF or select a different article.",
    "retention_invalid": "The selected retention mode is not available.",
    "attestation_required": "Please confirm the rights attestation before uploading.",
    "server_error": "An unexpected error occurred while processing the upload. The file has been deleted.",
}
