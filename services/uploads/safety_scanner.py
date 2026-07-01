"""Upload safety scanner — Phase 2D.

Scans ONLY user-added surfaces of an uploaded PDF for accidental PHI:
  - Original filename
  - PDF metadata (/Title, /Author, /Subject, /Keywords, /Creator, /Producer)
  - Annotations (comments, highlights, free-text)
  - Form fields

Article BODY text is NEVER scanned by this module — bodies legitimately
contain DOIs, author emails, dates, addresses, references; running the
clinical PHI guard on a body would always block.

Returns a closed-set decision. NO raw matched substrings are ever returned,
logged, or audited — only fixed reason codes + the surfaces that were
scanned (e.g., "filename" / "metadata"). The pipeline deletes the GridFS
bytes immediately when a block is decided.
"""
from __future__ import annotations

import re
from typing import Iterable

# --------------------------------------------------------------------------
#  PHI patterns for USER-ADDED surfaces only.
# --------------------------------------------------------------------------
_PATTERNS_ANY_SURFACE: list[tuple[str, re.Pattern]] = [
    # MRN-like patterns
    ("mrn_like", re.compile(r"\bMRN[:\s#]*\d{4,}\b", re.IGNORECASE)),
    ("mrn_like", re.compile(r"\bmedical\s+record\s*(?:no|number|#)?[:\s]*\d{4,}\b", re.IGNORECASE)),
    # DOB / date of birth
    ("dob", re.compile(r"\bDOB[:\s]*\d", re.IGNORECASE)),
    ("dob", re.compile(r"\bdate\s+of\s+birth\b", re.IGNORECASE)),
    # Patient name preface
    ("patient_name", re.compile(r"\b(?:patient|pt|name)[:\s]+(?:[A-Z][a-z]+(?:\s+|$)){1,4}", re.IGNORECASE)),
    # "private note" / "do not share" markers
    ("private_marker", re.compile(r"\b(?:private\s+note|do\s+not\s+share|confidential\s+patient)\b", re.IGNORECASE)),
    # Room numbers (annotation context only)
    ("room_number", re.compile(r"\broom\s*\d+\b", re.IGNORECASE)),
    # Long bare digit run (≥7) — restricted to filename/metadata/annotations
    # because article bodies legitimately have NCT12345678, PMC1234567 etc.
    ("digit_run_7plus", re.compile(r"(?<![\dA-Za-z])\d{7,}\b")),
    # US-style exact dates
    ("us_date", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    # Email — fine in body, suspect in annotations/filename/metadata
    ("email", re.compile(r"[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    # Phone number — US + intl-ish
    ("phone", re.compile(r"\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b")),
    ("phone", re.compile(r"\+\d{1,3}\s?\d{2,4}[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}\b")),
]

# Reason codes are aggregated PER surface for the audit record.
_SURFACE_REASON_CODE = {
    "filename": "filename_phi_detected",
    "metadata": "metadata_phi_detected",
    "annotation": "annotation_phi_detected",
    "formfield": "formfield_phi_detected",
    "watermark": "watermark_phi_detected",
}


def _scan_text(text: str) -> list[str]:
    """Return list of pattern keys that fired on this text fragment."""
    hits: list[str] = []
    if not text:
        return hits
    for key, pat in _PATTERNS_ANY_SURFACE:
        if pat.search(text):
            hits.append(key)
    return hits


def scan_user_added_surfaces(
    *,
    filename: str,
    pdf_metadata: dict | None = None,
    annotations_texts: Iterable[str] | None = None,
    formfield_values: Iterable[str] | None = None,
    watermark_texts: Iterable[str] | None = None,
) -> dict:
    """Run the PHI scan over user-added surfaces only.

    Returns:
        {
          "blocked": bool,
          "reason_codes": [closed-set codes],
          "scanned_surfaces": ["filename", "metadata", "annotation", ...],
        }

    Notes:
      * No raw matched substrings are ever returned.
      * No body text is ever passed into this scanner.
    """
    scanned: list[str] = []
    reason_codes: set[str] = set()

    # Filename — normalize underscores/dashes to spaces so word boundaries
    # work as humans expect (e.g., "patient_MRN1234567_review.pdf").
    if filename:
        scanned.append("filename")
        normalized = re.sub(r"[_\-.]+", " ", filename)
        if _scan_text(normalized):
            reason_codes.add(_SURFACE_REASON_CODE["filename"])

    # Metadata
    if pdf_metadata:
        scanned.append("metadata")
        for k in ("/Title", "/Author", "/Subject", "/Keywords", "/Creator", "/Producer",
                  "title", "author", "subject", "keywords", "creator", "producer"):
            v = pdf_metadata.get(k)
            if isinstance(v, str) and _scan_text(v):
                reason_codes.add(_SURFACE_REASON_CODE["metadata"])
                break

    # Annotations
    if annotations_texts is not None:
        ann_list = [a for a in annotations_texts if a]
        if ann_list:
            scanned.append("annotation")
            for ann_text in ann_list:
                if _scan_text(ann_text):
                    reason_codes.add(_SURFACE_REASON_CODE["annotation"])
                    break

    # Form fields
    if formfield_values is not None:
        ff_list = [v for v in formfield_values if v]
        if ff_list:
            scanned.append("formfield")
            for v in ff_list:
                if _scan_text(v):
                    reason_codes.add(_SURFACE_REASON_CODE["formfield"])
                    break

    # Watermark / overlay text (if separable)
    if watermark_texts is not None:
        wm_list = [w for w in watermark_texts if w]
        if wm_list:
            scanned.append("watermark")
            for wm in wm_list:
                if _scan_text(wm):
                    reason_codes.add(_SURFACE_REASON_CODE["watermark"])
                    break

    return {
        "blocked": len(reason_codes) > 0,
        "reason_codes": sorted(reason_codes),
        "scanned_surfaces": scanned,
    }
