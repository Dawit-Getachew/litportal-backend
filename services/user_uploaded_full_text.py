"""UserUploadedFullTextService — Phase 2D full pipeline.

NEVER calls an LLM. NEVER scrapes. NEVER downloads paywalled PDFs.
NEVER stores credentials. Scoped to {user_id, search_session_id, article_id}.

Pipeline: validate → magic-byte → open via pypdf → encrypted/active-content
guards → page-cap → checksum → GridFS persist → safety scan on user-added
surfaces → text extraction → article match → persist extracts → retention.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PyPdfError

from core.db import Collections, get_client, get_db
from services.audit import audit_service
from services.malware_scan import malware_scan_service
from services.uploads import RIGHTS_ATTESTATION_VERSION, USER_MESSAGES
from services.uploads.safety_scanner import scan_user_added_surfaces

logger = logging.getLogger("clinevidence.uploads")

GRIDFS_BUCKET_NAME = "uploaded_pdfs"

# Re-used by the section extractor in services/evidence_extraction.py.
SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("abstract", re.compile(r"^abstract\s*:?\s*$", re.IGNORECASE)),
    ("introduction", re.compile(r"^introduction\s*:?\s*$", re.IGNORECASE)),
    ("background", re.compile(r"^background\s*:?\s*$", re.IGNORECASE)),
    ("methods", re.compile(r"^(?:methods|materials and methods|study design|methodology)\s*:?\s*$", re.IGNORECASE)),
    ("results", re.compile(r"^(?:results|findings)\s*:?\s*$", re.IGNORECASE)),
    ("discussion", re.compile(r"^discussion\s*:?\s*$", re.IGNORECASE)),
    ("limitations", re.compile(r"^limitations\s*:?\s*$", re.IGNORECASE)),
    ("conclusion", re.compile(r"^conclusions?\s*:?\s*$", re.IGNORECASE)),
    ("references", re.compile(r"^references\s*:?\s*$", re.IGNORECASE)),
    ("acknowledgments", re.compile(r"^acknowledg(?:e)?ments\s*:?\s*$", re.IGNORECASE)),
    ("supplementary", re.compile(r"^(?:supplementary|appendix|supplemental)\s*(?:material|materials|information)?\s*:?\s*$", re.IGNORECASE)),
]

INCLUDE_IN_SUMMARY_DEFAULT = {
    "abstract", "methods", "results", "discussion", "limitations", "conclusion",
    "tables", "figure_captions",
}

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
PMID_RE = re.compile(r"(?:^|[^A-Za-z0-9])(?:PMID|pmid)[:\s]*(\d{6,9})")
PMCID_RE = re.compile(r"\bPMC\d{6,8}\b")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


# ============================================================================
#  Helpers
# ============================================================================
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _gridfs_bucket() -> AsyncIOMotorGridFSBucket:
    client = get_client()
    db = client[os.environ["DB_NAME"]]
    return AsyncIOMotorGridFSBucket(db, bucket_name=GRIDFS_BUCKET_NAME)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _norm_title(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_doi(s: str) -> str:
    return (s or "").lower().strip().rstrip(".,;)")


def _clean_safe_msg(s: str) -> str:
    """Strip newlines and cap to 200 chars for user-facing error messages."""
    s = (s or "").replace("\n", " ").replace("\r", " ").strip()
    return s[:200]


# ============================================================================
#  PDF inspection helpers (pypdf)
# ============================================================================
def _open_pdf(buf: bytes) -> PdfReader:
    return PdfReader(io.BytesIO(buf))


def _has_embedded_files(reader: PdfReader) -> bool:
    try:
        root = reader.trailer["/Root"].get_object()
        names = root.get("/Names") or {}
        if isinstance(names, dict) and names.get("/EmbeddedFiles"):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        if reader.attachments:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _has_javascript(reader: PdfReader) -> bool:
    """Detect /JS or /JavaScript anywhere reachable via /OpenAction / /AA / /Names."""
    try:
        root = reader.trailer["/Root"].get_object()
    except Exception:  # noqa: BLE001
        return False

    def _check(obj) -> bool:
        try:
            from pypdf.generic import IndirectObject
            if isinstance(obj, IndirectObject):
                obj = obj.get_object()
        except Exception:  # noqa: BLE001
            return False
        if obj is None:
            return False
        try:
            keys = list(getattr(obj, "keys", lambda: [])())
        except Exception:  # noqa: BLE001
            return False
        for k in keys:
            sk = str(k)
            if sk in ("/JS", "/JavaScript"):
                return True
        return False

    for key in ("/OpenAction", "/AA"):
        try:
            if key in root and _check(root[key]):
                return True
        except Exception:  # noqa: BLE001
            pass
    # /Names → /JavaScript dict
    try:
        names = root.get("/Names")
        if names and "/JavaScript" in names:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _extract_pdf_metadata(reader: PdfReader) -> dict:
    out = {}
    try:
        md = reader.metadata or {}
        for k, v in md.items():
            if isinstance(v, str):
                out[str(k)] = v
    except Exception:  # noqa: BLE001
        pass
    return out


def _extract_annotation_texts(reader: PdfReader) -> list[str]:
    texts: list[str] = []
    try:
        for page in reader.pages:
            annots = page.get("/Annots") or []
            try:
                annots = annots or []
            except Exception:  # noqa: BLE001
                continue
            for a in annots:
                try:
                    obj = a.get_object() if hasattr(a, "get_object") else a
                except Exception:  # noqa: BLE001
                    continue
                if obj is None:
                    continue
                for k in ("/Contents", "/T", "/RC"):
                    v = obj.get(k) if hasattr(obj, "get") else None
                    if isinstance(v, str) and v.strip():
                        texts.append(v.strip())
    except Exception:  # noqa: BLE001
        return texts
    return texts


def _extract_form_field_values(reader: PdfReader) -> list[str]:
    try:
        fields = reader.get_form_text_fields() or {}
    except Exception:  # noqa: BLE001
        return []
    return [str(v) for v in fields.values() if isinstance(v, str) and v.strip()]


def _extract_full_text(reader: PdfReader) -> str:
    """Concatenate text across pages. Best-effort; scanned PDFs may return empty."""
    chunks: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            t = ""
        if t:
            chunks.append(t)
    return "\n\n".join(chunks)


# ============================================================================
#  Article matching
# ============================================================================
def _match_article(
    article: dict[str, Any],
    *,
    pdf_metadata: dict,
    pdf_text_head: str,
) -> dict[str, Any]:
    """Return {confidence, matched_by, signals}."""
    signals: dict[str, Any] = {}

    art_doi = _norm_doi(article.get("doi") or "")
    art_pmid = (article.get("pmid") or "").strip()
    art_pmcid = (article.get("pmcid") or "").strip().upper()
    art_year = article.get("publication_year")

    # DOI
    doi_in_meta = ""
    for k in ("/doi", "/DOI", "doi"):
        v = pdf_metadata.get(k)
        if isinstance(v, str):
            m = DOI_RE.search(v)
            if m:
                doi_in_meta = m.group(0)
                break
    doi_in_text = ""
    m = DOI_RE.search(pdf_text_head or "")
    if m:
        doi_in_text = m.group(0)
    pdf_doi = _norm_doi(doi_in_meta or doi_in_text)
    signals["pdf_doi"] = pdf_doi
    if pdf_doi and art_doi:
        if pdf_doi == art_doi:
            return {"confidence": "high", "matched_by": "doi", "signals": signals}
        # If both present and they differ, that's a strong mismatch signal.
        signals["doi_conflict"] = True

    # PMID
    pdf_pmid = ""
    m = PMID_RE.search(pdf_text_head or "")
    if m:
        pdf_pmid = m.group(1)
    signals["pdf_pmid"] = pdf_pmid
    if pdf_pmid and art_pmid:
        if pdf_pmid == art_pmid:
            return {"confidence": "high", "matched_by": "pmid", "signals": signals}
        signals["pmid_conflict"] = True

    # PMCID
    pdf_pmcid = ""
    m = PMCID_RE.search(pdf_text_head or "")
    if m:
        pdf_pmcid = m.group(0).upper()
    signals["pdf_pmcid"] = pdf_pmcid
    if pdf_pmcid and art_pmcid and pdf_pmcid == art_pmcid:
        return {"confidence": "high", "matched_by": "pmcid", "signals": signals}

    # Title similarity
    pdf_title = ""
    for k in ("/Title", "title"):
        v = pdf_metadata.get(k)
        if isinstance(v, str) and v.strip():
            pdf_title = v.strip()
            break
    if not pdf_title:
        # Try first line of first page (~200 chars).
        pdf_title = (pdf_text_head or "").split("\n", 1)[0][:200]

    art_title = article.get("title") or ""
    sim = SequenceMatcher(None, _norm_title(pdf_title), _norm_title(art_title)).ratio()
    signals["title_similarity"] = round(sim, 3)

    # Year
    year_ok = False
    if art_year:
        years = [int(x) for x in YEAR_RE.findall(pdf_text_head[:3000] or "")]
        signals["pdf_years_found"] = years[:5]
        if any(abs(y - int(art_year)) <= 1 for y in years):
            year_ok = True

    if sim >= 0.92 and year_ok:
        return {"confidence": "high", "matched_by": "title_year_journal", "signals": signals}
    if sim >= 0.80 and year_ok:
        return {"confidence": "moderate", "matched_by": "title_similarity", "signals": signals}
    if sim >= 0.60:
        return {"confidence": "low", "matched_by": "title_partial", "signals": signals}

    # Strong mismatch signals
    if signals.get("doi_conflict"):
        return {"confidence": "mismatch", "matched_by": "doi_conflict", "signals": signals}
    if signals.get("pmid_conflict"):
        return {"confidence": "mismatch", "matched_by": "pmid_conflict", "signals": signals}
    return {"confidence": "mismatch", "matched_by": "none", "signals": signals}


# ============================================================================
#  Section extraction (used by both upload pipeline + EvidenceExtractionService)
# ============================================================================
def _classify_section_line(line: str) -> Optional[str]:
    line_s = line.strip()
    if not line_s:
        return None
    if re.match(r"^TABLE\s+\d+", line_s, re.IGNORECASE):
        return "tables"
    if re.match(r"^(?:Fig\.|Figure)\s+\d+", line_s, re.IGNORECASE):
        return "figure_captions"
    for name, pat in SECTION_PATTERNS:
        if pat.match(line_s):
            return name
    return None


def extract_sections(full_text: str) -> list[dict[str, Any]]:
    """Heuristic section splitter — never calls an LLM.

    Returns a list of {section_name, section_text, token_count, include_in_summary}.
    Best-effort: lines that match the heading regex demarcate sections.
    Anything before the first heading is treated as "other".

    Phase 2D body fallback: when the heading-based pass produces ZERO
    include_in_summary=true sections AND the cumulative token count is at
    least 200 (excluding references/acknowledgments/author_info), an extra
    ``section_name="body"`` row is emitted with the substantive text
    concatenated. This prevents unstructured-but-text-bearing PDFs from
    silently disqualifying themselves from downstream summarization.
    """
    if not (full_text or "").strip():
        return []
    lines = full_text.splitlines()
    sections: list[dict[str, Any]] = []
    current_name = "other"
    current_lines: list[str] = []

    def _flush():
        if not current_lines:
            return
        text = "\n".join(current_lines).strip()
        if not text:
            return
        if len(text) < 30 and current_name == "other":
            # Skip trivial bursts (e.g., page numbers).
            return
        words = text.split()
        sections.append({
            "section_name": current_name,
            "section_text": text,
            "token_count": (len(words) * 4) // 3,
            "include_in_summary": current_name in INCLUDE_IN_SUMMARY_DEFAULT,
        })

    for ln in lines:
        cat = _classify_section_line(ln)
        if cat is not None:
            _flush()
            current_name = cat
            current_lines = []
        else:
            current_lines.append(ln)
    _flush()

    # ---- Body fallback (Phase 2D) ----
    has_eligible = any(s["include_in_summary"] for s in sections)
    if not has_eligible:
        # Concatenate substantive text, excluding references / acknowledgments /
        # author_info / supplementary (kept consistent with the spec's
        # include_in_summary defaults).
        skip_for_body = {"references", "acknowledgments", "author_info", "supplementary"}
        body_chunks = [s["section_text"] for s in sections if s["section_name"] not in skip_for_body]
        body_text = "\n\n".join(body_chunks).strip()
        if body_text:
            words = body_text.split()
            body_token_count = (len(words) * 4) // 3
            if body_token_count >= 200:
                sections.append({
                    "section_name": "body",
                    "section_text": body_text,
                    "token_count": body_token_count,
                    "include_in_summary": True,
                })
    return sections


# ============================================================================
#  Service
# ============================================================================
class UserUploadedFullTextService:
    """Phase 2D full pipeline. Single public method: ``process``."""

    def __init__(self) -> None:
        self.max_pdf_pages = int(os.environ.get("MAX_PDF_PAGES", "100"))
        self.feature_enabled = (os.environ.get("ENABLE_USER_PDF_UPLOADS", "true").lower() != "false")
        self.malware_scan_enabled = (os.environ.get("PDF_MALWARE_SCAN_ENABLED", "false").lower() == "true")

    async def process(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        user_id: str,
        search_session_id: str,
        article: dict[str, Any],
        session: dict[str, Any],
        retention_mode: str,
        rights_attested: bool,
        rights_attestation_version: str,
    ) -> dict[str, Any]:
        """Run the full pipeline. Returns a dict response shape suitable
        for the upload endpoint. NEVER raises (all paths return a record)."""
        upload_id = uuid.uuid4().hex
        await audit_service.log(
            "upload_received",
            user_id=user_id,
            search_session_id=search_session_id,
            payload={"upload_id": upload_id, "size": len(file_bytes), "filename_len": len(filename or "")},
        )

        # --- Magic bytes ---
        if not file_bytes.startswith(b"%PDF"):
            return await self._finalize_error(
                upload_id, user_id, search_session_id, article["id"],
                filename, len(file_bytes), retention_mode,
                rights_attested, rights_attestation_version,
                code="not_a_pdf", message=USER_MESSAGES["not_a_pdf"],
            )

        # --- Phase 4B malware scan (BEFORE any text extraction) ---
        scan_result = await malware_scan_service.scan(file_bytes=file_bytes)
        # Persist the closed-set result code to audit (never raw bytes).
        await audit_service.log(
            "upload_malware_scan",
            user_id=user_id,
            search_session_id=search_session_id,
            payload={
                "upload_id": upload_id,
                "result": scan_result.result,
                "mode": scan_result.mode,
                "signature_label": scan_result.signature_label,
            },
        )
        if scan_result.result in ("infected", "suspicious"):
            return await self._finalize_error(
                upload_id, user_id, search_session_id, article["id"],
                filename, len(file_bytes), retention_mode,
                rights_attested, rights_attestation_version,
                code="malware_detected",
                message=(
                    scan_result.detail_safe
                    or USER_MESSAGES.get("safety_scan_failed")
                    or "The file failed a malware safety check."
                ),
                malware_scan_status="failed",
            )
        if scan_result.result == "scan_error":
            return await self._finalize_error(
                upload_id, user_id, search_session_id, article["id"],
                filename, len(file_bytes), retention_mode,
                rights_attested, rights_attestation_version,
                code="malware_scan_error",
                message="The document safety scan failed to complete. Please try again.",
                malware_scan_status="failed",
            )
        if scan_result.result == "scanner_unavailable":
            # When scanner is unavailable AND uploads-without-AV are not allowed
            # the gate at the router stops us. Here uploads_allowed() returned
            # True (dev mode), so we record the honest scanner-unavailable
            # status and continue without claiming a scan happened.
            malware_scan_status_value = "skipped"
        elif scan_result.result == "disabled":
            malware_scan_status_value = "not_configured"
        else:
            malware_scan_status_value = "passed"

        # --- Open with pypdf ---
        try:
            reader = _open_pdf(file_bytes)
        except (PdfReadError, PyPdfError, Exception) as exc:  # noqa: BLE001
            logger.warning("pypdf failed to open uploaded PDF: %s", type(exc).__name__)
            return await self._finalize_error(
                upload_id, user_id, search_session_id, article["id"],
                filename, len(file_bytes), retention_mode,
                rights_attested, rights_attestation_version,
                code="not_a_pdf", message=USER_MESSAGES["not_a_pdf"],
            )

        # --- Encryption ---
        if reader.is_encrypted:
            return await self._finalize_error(
                upload_id, user_id, search_session_id, article["id"],
                filename, len(file_bytes), retention_mode,
                rights_attested, rights_attestation_version,
                code="encrypted_pdf_rejected",
                message=USER_MESSAGES["encrypted_pdf_rejected"],
            )

        # --- Embedded files ---
        if _has_embedded_files(reader):
            return await self._finalize_error(
                upload_id, user_id, search_session_id, article["id"],
                filename, len(file_bytes), retention_mode,
                rights_attested, rights_attestation_version,
                code="embedded_attachment_rejected",
                message=USER_MESSAGES["embedded_attachment_rejected"],
            )

        # --- JavaScript / active content ---
        if _has_javascript(reader):
            return await self._finalize_error(
                upload_id, user_id, search_session_id, article["id"],
                filename, len(file_bytes), retention_mode,
                rights_attested, rights_attestation_version,
                code="active_content_rejected",
                message=USER_MESSAGES["active_content_rejected"],
            )

        # --- Page count ---
        try:
            page_count = len(reader.pages)
        except Exception:  # noqa: BLE001
            page_count = 0
        if page_count > self.max_pdf_pages:
            return await self._finalize_error(
                upload_id, user_id, search_session_id, article["id"],
                filename, len(file_bytes), retention_mode,
                rights_attested, rights_attestation_version,
                code="page_count_exceeded",
                message=USER_MESSAGES["page_count_exceeded"],
                page_count=page_count,
            )

        # --- Checksum ---
        checksum = _sha256(file_bytes)

        # --- Persist to GridFS ---
        bucket = _gridfs_bucket()
        gridfs_file_id = await bucket.upload_from_stream(
            filename or f"{upload_id}.pdf",
            io.BytesIO(file_bytes),
            metadata={
                "user_id": user_id,
                "search_session_id": search_session_id,
                "article_id": article["id"],
                "upload_id": upload_id,
                "retention_mode": retention_mode,
                "checksum": checksum,
                "content_type": "application/pdf",
            },
        )
        gridfs_file_id_str = str(gridfs_file_id)
        await audit_service.log(
            "upload_validated",
            user_id=user_id,
            search_session_id=search_session_id,
            payload={"upload_id": upload_id, "page_count": page_count},
        )

        # --- Safety scan (USER-ADDED surfaces only) ---
        pdf_metadata = _extract_pdf_metadata(reader)
        annotation_texts = _extract_annotation_texts(reader)
        formfield_values = _extract_form_field_values(reader)
        scan = scan_user_added_surfaces(
            filename=filename,
            pdf_metadata=pdf_metadata,
            annotations_texts=annotation_texts,
            formfield_values=formfield_values,
        )
        await audit_service.log(
            "upload_safety_scanned",
            user_id=user_id,
            search_session_id=search_session_id,
            payload={
                "upload_id": upload_id,
                "scanned_surfaces": scan["scanned_surfaces"],
                "blocked": scan["blocked"],
                "reason_codes": scan["reason_codes"],
            },
        )
        if scan["blocked"]:
            # Delete bytes immediately, persist a blocked_phi record.
            await self._delete_gridfs(bucket, gridfs_file_id)
            await self._persist_asset(
                upload_id, user_id, search_session_id, article["id"],
                filename, len(file_bytes), retention_mode,
                rights_attested, rights_attestation_version,
                checksum=checksum,
                page_count=page_count,
                upload_status="blocked_phi",
                gridfs_file_id=None,
                storage_deleted_at=_now(),
                processing_error_code="safety_scan_failed",
                processing_error_message_safe=USER_MESSAGES["safety_scan_failed"],
                annotation_scan_status="blocked",
                metadata_scan_status="blocked",
                malware_scan_status=malware_scan_status_value,
            )
            await audit_service.log(
                "upload_blocked",
                user_id=user_id,
                search_session_id=search_session_id,
                payload={
                    "upload_id": upload_id,
                    "reason_codes": scan["reason_codes"],
                    "scanned_surfaces": scan["scanned_surfaces"],
                },
            )
            return {
                "upload_id": upload_id,
                "upload_status": "blocked_phi",
                "match_confidence": None,
                "matched_by": None,
                "extracted_sections_count": 0,
                "user_confirmation_required": False,
                "processing_status": "blocked",
                "user_message": USER_MESSAGES["safety_scan_failed"],
                "reason_codes": scan["reason_codes"],
                "retention_mode": retention_mode,
            }

        # --- Text extraction ---
        full_text = _extract_full_text(reader)
        if not (full_text or "").strip():
            await self._delete_gridfs(bucket, gridfs_file_id)
            return await self._finalize_error(
                upload_id, user_id, search_session_id, article["id"],
                filename, len(file_bytes), retention_mode,
                rights_attested, rights_attestation_version,
                code="scanned_pdf_unsupported",
                message=USER_MESSAGES["scanned_pdf_unsupported"],
                page_count=page_count,
            )
        text_head = full_text[:5000]
        await audit_service.log(
            "upload_text_extracted",
            user_id=user_id,
            search_session_id=search_session_id,
            payload={
                "upload_id": upload_id,
                "char_count": len(full_text),
                "page_count": page_count,
            },
        )

        # --- Article matching ---
        match = _match_article(article, pdf_metadata=pdf_metadata, pdf_text_head=text_head)
        await audit_service.log(
            "upload_matched",
            user_id=user_id,
            search_session_id=search_session_id,
            payload={
                "upload_id": upload_id,
                "match_confidence": match["confidence"],
                "matched_by": match["matched_by"],
                "title_similarity": match["signals"].get("title_similarity"),
            },
        )
        if match["confidence"] == "mismatch":
            await self._delete_gridfs(bucket, gridfs_file_id)
            await self._persist_asset(
                upload_id, user_id, search_session_id, article["id"],
                filename, len(file_bytes), retention_mode,
                rights_attested, rights_attestation_version,
                checksum=checksum,
                page_count=page_count,
                upload_status="mismatch",
                match_confidence="mismatch",
                matched_by=match["matched_by"],
                gridfs_file_id=None,
                storage_deleted_at=_now(),
                processing_error_code="match_failed",
                processing_error_message_safe=USER_MESSAGES["match_failed"],
            )
            return {
                "upload_id": upload_id,
                "upload_status": "mismatch",
                "match_confidence": "mismatch",
                "matched_by": match["matched_by"],
                "extracted_sections_count": 0,
                "user_confirmation_required": False,
                "processing_status": "blocked",
                "user_message": USER_MESSAGES["match_failed"],
                "retention_mode": retention_mode,
            }

        # --- Extract sections ---
        sections = extract_sections(full_text)
        needs_confirm = match["confidence"] in {"moderate", "low"}
        await self._persist_extracts(
            upload_id=upload_id,
            user_id=user_id,
            search_session_id=search_session_id,
            article_id=article["id"],
            sections=sections,
            expires_at=session.get("expires_at"),
            include_default=not needs_confirm,
        )
        extracted_concat = "".join(s["section_text"] for s in sections)
        extracted_checksum = _sha256(extracted_concat.encode("utf-8")) if extracted_concat else None

        # --- Retention ---
        retention_applied = retention_mode
        if retention_mode == "use_once":
            await self._delete_gridfs(bucket, gridfs_file_id)
            gridfs_persisted = None
            storage_deleted_at = _now()
        else:
            gridfs_persisted = gridfs_file_id_str
            storage_deleted_at = None

        upload_status = "matched" if needs_confirm else "processed"
        await self._persist_asset(
            upload_id, user_id, search_session_id, article["id"],
            filename, len(file_bytes), retention_mode,
            rights_attested, rights_attestation_version,
            checksum=checksum,
            extracted_text_checksum=extracted_checksum,
            page_count=page_count,
            upload_status=upload_status,
            match_confidence=match["confidence"],
            matched_by=match["matched_by"],
            extracted_text_status="complete",
            user_confirmation_required=needs_confirm,
            gridfs_file_id=gridfs_persisted,
            storage_deleted_at=storage_deleted_at,
            metadata_scan_status="passed",
            annotation_scan_status="passed",
            malware_scan_status=malware_scan_status_value,
            expires_at=session.get("expires_at"),
        )

        # When extraction yields included sections AND the match is high
        # (auto-included), surface a user-scoped access-locations row so
        # the article card and access-details accordion can show it.
        if not needs_confirm and any(s["include_in_summary"] for s in sections):
            await self._upsert_user_upload_access_location(
                user_id=user_id,
                search_session_id=search_session_id,
                article_id=article["id"],
            )

        await audit_service.log(
            "upload_processed",
            user_id=user_id,
            search_session_id=search_session_id,
            payload={
                "upload_id": upload_id,
                "match_confidence": match["confidence"],
                "matched_by": match["matched_by"],
                "section_count": len(sections),
                "retention_mode": retention_applied,
            },
        )
        return {
            "upload_id": upload_id,
            "upload_status": upload_status,
            "match_confidence": match["confidence"],
            "matched_by": match["matched_by"],
            "extracted_sections_count": len(sections),
            "user_confirmation_required": needs_confirm,
            "processing_status": "needs_confirmation" if needs_confirm else "processed",
            "user_message": (
                "Full text extracted. Please confirm the title matches."
                if needs_confirm
                else "Full text uploaded and extracted successfully."
            ),
            "retention_mode": retention_applied,
        }

    # ------------------------------------------------------------------
    #  Persistence helpers
    # ------------------------------------------------------------------
    async def _delete_gridfs(self, bucket: AsyncIOMotorGridFSBucket, file_id) -> None:
        try:
            await bucket.delete(file_id)
        except Exception:  # noqa: BLE001
            logger.exception("GridFS delete failed for %s", file_id)

    async def _persist_asset(
        self,
        upload_id: str,
        user_id: str,
        search_session_id: str,
        article_id: str,
        filename: str,
        file_size: int,
        retention_mode: str,
        rights_attested: bool,
        rights_attestation_version: str,
        **extra,
    ) -> dict[str, Any]:
        db = get_db()
        doc = {
            "id": upload_id,
            "user_id": user_id,
            "search_session_id": search_session_id,
            "article_id": article_id,
            "original_filename": filename or "",
            "file_type": "application/pdf",
            "file_size": file_size,
            "retention_mode": retention_mode,
            "rights_attestation_version": rights_attestation_version if rights_attested else None,
            "user_attested_at": _now() if rights_attested else None,
            "created_at": _now(),
            "updated_at": _now(),
        }
        doc.update(extra)
        await db[Collections.UPLOADED_FULL_TEXT_ASSETS].insert_one(doc)
        return doc

    async def _finalize_error(
        self,
        upload_id: str,
        user_id: str,
        search_session_id: str,
        article_id: str,
        filename: str,
        file_size: int,
        retention_mode: str,
        rights_attested: bool,
        rights_attestation_version: str,
        *,
        code: str,
        message: str,
        page_count: int | None = None,
        malware_scan_status: str | None = None,
    ) -> dict[str, Any]:
        extras: dict[str, Any] = {
            "upload_status": "error",
            "processing_error_code": code,
            "processing_error_message_safe": _clean_safe_msg(message),
            "page_count": page_count,
        }
        if malware_scan_status is not None:
            extras["malware_scan_status"] = malware_scan_status
        await self._persist_asset(
            upload_id, user_id, search_session_id, article_id,
            filename, file_size, retention_mode,
            rights_attested, rights_attestation_version,
            **extras,
        )
        await audit_service.log(
            "upload_blocked",
            user_id=user_id,
            search_session_id=search_session_id,
            payload={"upload_id": upload_id, "reason_codes": [code]},
        )
        return {
            "upload_id": upload_id,
            "upload_status": "error",
            "match_confidence": None,
            "matched_by": None,
            "extracted_sections_count": 0,
            "user_confirmation_required": False,
            "processing_status": "error",
            "user_message": message,
            "reason_codes": [code],
            "retention_mode": retention_mode,
        }

    async def _persist_extracts(
        self,
        *,
        upload_id: str,
        user_id: str,
        search_session_id: str,
        article_id: str,
        sections: list[dict[str, Any]],
        expires_at: datetime | None,
        include_default: bool,
    ) -> None:
        if not sections:
            return
        db = get_db()
        docs = []
        for s in sections:
            include = s["include_in_summary"] and include_default
            docs.append({
                "id": uuid.uuid4().hex,
                "article_id": article_id,
                "uploaded_full_text_asset_id": upload_id,
                "search_session_id": search_session_id,
                "user_id": user_id,
                "text_source": "user_uploaded_full_text",
                "evidence_text_level": "user_uploaded_full_text",
                "section_name": s["section_name"],
                "section_text": s["section_text"],
                "token_count": s["token_count"],
                "include_in_summary": include,
                "created_at": _now(),
                "expires_at": expires_at,
            })
        if docs:
            await db[Collections.ARTICLE_TEXT_EXTRACTS].insert_many(docs)

    async def _upsert_user_upload_access_location(
        self, *, user_id: str, search_session_id: str, article_id: str,
    ) -> None:
        """Insert a per-user access_locations row of category user_uploaded_full_text.

        Only the owning user can ever read this row (filtered by user_id in
        the access-locations router). Idempotent: skip if one already exists
        for this {user, session, article}.
        """
        db = get_db()
        existing = await db[Collections.ARTICLE_ACCESS_LOCATIONS].find_one(
            {
                "user_id": user_id,
                "search_session_id": search_session_id,
                "article_id": article_id,
                "access_category": "user_uploaded_full_text",
            },
            {"_id": 0, "id": 1},
        )
        if existing:
            return
        await db[Collections.ARTICLE_ACCESS_LOCATIONS].insert_one({
            "id": uuid.uuid4().hex,
            "article_id": article_id,
            "user_id": user_id,
            "search_session_id": search_session_id,
            "source": "user_upload",
            "access_category": "user_uploaded_full_text",
            "url": None,
            "is_direct_pdf": False,
            "is_oa": False,
            "requires_login": True,
            "can_view_in_app": False,
            "can_ingest_for_summary": True,
            "license": "user_attested",
            "version": None,
            "evidence_text_level": "user_uploaded_full_text",
            "last_checked_at": _now(),
            "created_at": _now(),
            "updated_at": _now(),
        })

    # ------------------------------------------------------------------
    #  Confirm / reject moderate-low matches
    # ------------------------------------------------------------------
    async def confirm_match(self, upload_id: str, user_id: str, confirmed: bool) -> dict[str, Any]:
        db = get_db()
        asset = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].find_one(
            {"id": upload_id, "user_id": user_id, "deleted_at": None},
            {"_id": 0},
        )
        if not asset:
            return {"status": "not_found"}
        if asset["upload_status"] not in ("matched",):
            return {"status": "invalid_state"}

        if confirmed:
            # Promote to processed; flip include_in_summary defaults on extracts.
            await db[Collections.ARTICLE_TEXT_EXTRACTS].update_many(
                {"uploaded_full_text_asset_id": upload_id,
                 "section_name": {"$in": list({"abstract", "methods", "results",
                                              "discussion", "limitations", "conclusion",
                                              "tables", "figure_captions"})}},
                {"$set": {"include_in_summary": True}},
            )
            await db[Collections.UPLOADED_FULL_TEXT_ASSETS].update_one(
                {"id": upload_id},
                {"$set": {
                    "upload_status": "processed",
                    "user_confirmed_match_at": _now(),
                    "user_confirmation_required": False,
                    "matched_by": "user_confirmed",
                    "updated_at": _now(),
                }},
            )
            await self._upsert_user_upload_access_location(
                user_id=user_id,
                search_session_id=asset["search_session_id"],
                article_id=asset["article_id"],
            )
            await audit_service.log(
                "upload_match_confirmed",
                user_id=user_id,
                search_session_id=asset["search_session_id"],
                payload={"upload_id": upload_id},
            )
            return {"status": "ok", "upload_status": "processed"}
        else:
            await self.delete_upload(upload_id, user_id)
            await audit_service.log(
                "upload_match_rejected",
                user_id=user_id,
                search_session_id=asset["search_session_id"],
                payload={"upload_id": upload_id},
            )
            return {"status": "ok", "upload_status": "deleted"}

    # ------------------------------------------------------------------
    #  Delete upload (user-initiated OR retention cleanup)
    # ------------------------------------------------------------------
    async def delete_upload(self, upload_id: str, user_id: str | None) -> bool:
        db = get_db()
        match = {"id": upload_id}
        if user_id is not None:
            match["user_id"] = user_id
        asset = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].find_one(match, {"_id": 0})
        if not asset:
            return False
        # GridFS delete (if still persisted)
        if asset.get("gridfs_file_id"):
            try:
                from bson import ObjectId
                bucket = _gridfs_bucket()
                await bucket.delete(ObjectId(asset["gridfs_file_id"]))
            except Exception:  # noqa: BLE001
                logger.exception("GridFS delete failed for %s", asset.get("gridfs_file_id"))
        # Delete extracts
        await db[Collections.ARTICLE_TEXT_EXTRACTS].delete_many(
            {"uploaded_full_text_asset_id": upload_id}
        )
        # Delete the per-user access-location row
        await db[Collections.ARTICLE_ACCESS_LOCATIONS].delete_many({
            "user_id": asset["user_id"],
            "search_session_id": asset["search_session_id"],
            "article_id": asset["article_id"],
            "access_category": "user_uploaded_full_text",
        })
        await db[Collections.UPLOADED_FULL_TEXT_ASSETS].update_one(
            {"id": upload_id},
            {"$set": {
                "upload_status": "deleted",
                "deleted_at": _now(),
                "storage_deleted_at": _now(),
                "gridfs_file_id": None,
                "updated_at": _now(),
            }},
        )
        await audit_service.log(
            "upload_deleted",
            user_id=asset["user_id"],
            search_session_id=asset["search_session_id"],
            payload={"upload_id": upload_id},
        )
        return True


user_uploaded_full_text = UserUploadedFullTextService()
