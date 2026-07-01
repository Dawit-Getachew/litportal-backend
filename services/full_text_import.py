"""B15 / B8 — single-paper full-text finder & import service.

Powers the LitPulse "Full text import" button. The clinician is on a paper in
LitPulse (LitArchive / a saved item / a search result) and wants its FULL TEXT
inside LitPortal so they can run an evidence summary ("chat") over it or save it.

This is NOT a literature search. Given ONE paper's identifiers (PMID and/or DOI,
or — as a fallback — a title), it:

  1. Resolves the exact paper (identifier-first, deterministic) by reusing the
     Known-Article-Finder service (PubMed esummary/efetch + Crossref). PHI guard
     runs before any outbound call.
  2. Creates (or reuses) a lightweight single-article search session owned by the
     user, with the resolved paper attached and user-included. The session is the
     unit every downstream tool (OA fetch, summary, evidence packet) already
     operates on, so reusing it means the imported paper is immediately
     summary/"chat"-ready with no special-casing.
  3. Runs the OA-first access broker (PMC / Europe PMC / Unpaywall / Semantic
     Scholar / Crossref) to discover where the full text lives, then the OA
     full-text fetcher to actually ingest the text WHEN the license permits
     (CC0/BY/BY-SA). Subscription / NC / ND content is link-out only — never
     scraped, never paywalled-PDF-ingested. Per-user institutional link-outs
     (OpenAthens / OpenURL) are built dynamically and never persisted.
  4. Optionally saves the paper to the user's library (a research collection,
     mirrored to central LitHub) so it shows up in LitArchive.

Governance: the OA fetcher's license gate and the summary flow's source-permission
gate (B1/B16) still apply — this service only orchestrates them, it never relaxes
them. AI ingestion of a Cochrane / restricted-source paper remains blocked
downstream even if its full text is reachable for the human to read.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from core.config import get_settings
from core.db import Collections, get_db
from models.research_collections import AddRecordItem, AddRecordsRequest
from models.sessions import SearchSessionDoc
from services.audit import audit_service
from services.canonical_identity import make_canonical_key, normalize_doi
from services.full_text_access_broker import (
    compute_best_access,
    full_text_access_broker,
    institutional_rows_for_user,
)
from services.known_article_lookup import known_article_lookup_service
from services.oa_fetcher import oa_fetcher

logger = logging.getLogger("litportal.full_text_import")

# Confidence ordering (mirrors known_article_lookup.derive_confidence).
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}

# OA-fetch statuses that mean the full text was actually ingested for AI use.
_INGESTED_STATUSES = {"fetched", "cache_hit"}

# Marker stored on import-created sessions so repeat clicks reuse one session.
IMPORT_ORIGIN = "full_text_import"


class FullTextImportError(Exception):
    """Closed-set error for the full-text import flow."""

    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


def _confidence_ok(confidence: Optional[str], minimum: str) -> bool:
    """True if ``confidence`` is at least ``minimum`` on the high>medium>low scale."""
    return _CONFIDENCE_RANK.get(confidence or "low", 3) <= _CONFIDENCE_RANK.get(minimum, 1)


def _build_lookup_text(
    *, pmid: Optional[str], doi: Optional[str], pmcid: Optional[str], title: Optional[str],
) -> Optional[str]:
    """Pick the single strongest identifier string to hand the resolver.

    Identifier-first: an exact PMID/DOI/PMCID yields a high-confidence match;
    title is the deterministic last resort.

    NOTE: the resolver PHI-screens its input after stripping *labelled*
    identifiers. A bare PMID digit-run would otherwise read as a PHI signal
    (7+ digits), so we emit the labelled form ("PMID: <n>") which the
    resolver's identifier-stripper recognizes and removes before screening.
    DOIs / PMCIDs are already stripped by shape.
    """
    if pmid and str(pmid).strip():
        return f"PMID: {str(pmid).strip()}"
    if doi and str(doi).strip():
        return str(doi).strip()
    if pmcid and str(pmcid).strip():
        return str(pmcid).strip()
    if title and str(title).strip():
        return str(title).strip()
    return None


def _public_access_row(row: dict[str, Any]) -> dict[str, Any]:
    """Trim an access_locations row to the frontend-facing fields."""
    return {
        "source": row.get("source"),
        "access_category": row.get("access_category"),
        "url": row.get("url"),
        "is_oa": bool(row.get("is_oa")),
        "is_direct_pdf": bool(row.get("is_direct_pdf")),
        "requires_login": bool(row.get("requires_login")),
        "can_view_in_app": bool(row.get("can_view_in_app")),
        "can_ingest_for_summary": bool(row.get("can_ingest_for_summary")),
        "license": row.get("license"),
        "evidence_text_level": row.get("evidence_text_level"),
    }


class FullTextImportService:
    """Orchestrates resolve → attach → broker → OA-fetch → (optional) save."""

    async def resolve(
        self,
        *,
        user_id: str,
        pmid: Optional[str] = None,
        doi: Optional[str] = None,
        pmcid: Optional[str] = None,
        title: Optional[str] = None,
        session_id: Optional[str] = None,
        accept_low_confidence: bool = False,
        force_refresh: bool = False,
        save: bool = False,
        collection_id: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> dict[str, Any]:
        db = get_db()
        settings = get_settings()

        lookup_text = _build_lookup_text(pmid=pmid, doi=doi, pmcid=pmcid, title=title)
        if not lookup_text:
            raise FullTextImportError("no_identifier", status=422)

        # 1. Resolve the exact paper (PHI-guarded, identifier-first, deterministic).
        search_result = await known_article_lookup_service.search(lookup_text)
        if search_result.get("phi_blocked"):
            return {
                "status": "phi_blocked",
                "reasons": search_result.get("reasons") or [],
                "session_id": None,
                "article_id": None,
            }
        candidates = search_result.get("candidates") or []
        if not candidates:
            return {
                "status": "not_found",
                "lookup_source_provenance": search_result.get("lookup_source_provenance") or [],
                "session_id": None,
                "article_id": None,
            }

        top = candidates[0]
        confidence = top.get("match_confidence")
        # Title-only lookups can be ambiguous; require explicit confirmation
        # unless the caller opted in. Exact PMID/DOI matches score "high".
        if not accept_low_confidence and not _confidence_ok(
            confidence, settings.FULL_TEXT_IMPORT_MIN_CONFIDENCE,
        ):
            return {
                "status": "needs_confirmation",
                "match": {
                    "confidence": confidence,
                    "reason_codes": top.get("match_reason_codes") or [],
                },
                "candidates": candidates[:5],
                "session_id": None,
                "article_id": None,
            }

        # Safety: never auto-import/ingest a retracted or expression-of-concern
        # paper (mirrors the Known-Article-Finder attach guard).
        if top.get("is_retracted") or top.get("has_expression_of_concern"):
            return {
                "status": "blocked_safety",
                "reason": (
                    "article_retracted" if top.get("is_retracted")
                    else "expression_of_concern"
                ),
                "paper": self._paper_view_from_candidate(top),
                "session_id": None,
                "article_id": None,
            }

        # 2. Create or reuse a single-article import session.
        canonical_key = make_canonical_key(
            pmid=top.get("pmid"),
            doi=top.get("doi"),
            pmcid=top.get("pmcid"),
            title=top.get("title"),
            year=top.get("publication_year"),
            journal=top.get("journal"),
            first_author=(top.get("authors") or [None])[0],
        )
        session = await self._get_or_create_import_session(
            db, user_id=user_id, session_id=session_id,
            canonical_key=canonical_key, title=top.get("title"),
            force_refresh=force_refresh,
        )
        sid = session["id"]

        # 3. Attach (or surface) the paper to the session, user-included so the
        #    summary/"chat" flow treats it as selected evidence.
        attach = await known_article_lookup_service.attach_to_session(
            db=db, session_id=sid, user_id=user_id,
            candidate=top, include_in_evidence=True,
        )
        if attach.get("status") in ("session_not_found", "session_blocked_phi"):
            raise FullTextImportError(attach["status"], status=400)
        article_id = attach["article_id"]

        # Enrich the resolved article so downstream eligibility (B16), provenance
        # (B12) and the LitHub mirror have the fields they expect — the
        # known-article upsert is metadata-minimal.
        await self._enrich_article(db, article_id, top)
        # Pin the resolved article onto the session marker for idempotent reuse.
        await db[Collections.SEARCH_SESSIONS].update_one(
            {"id": sid},
            {"$set": {"imported_article_id": article_id,
                      "updated_at": datetime.now(timezone.utc)}},
        )

        # 4. Discover access routes (OA-first cascade) for THIS paper.
        owns_client = http_client is None
        if owns_client:
            http_client = httpx.AsyncClient(timeout=20.0)
        try:
            try:
                await full_text_access_broker.broker_for_session(sid, client=http_client)
            except Exception:  # noqa: BLE001 — discovery is best-effort
                logger.exception("full-text import: access broker failed (non-fatal)")

            # 5. Ingest OA full text when the license permits (license-gated).
            oa_result = await oa_fetcher.fetch_for_article(
                article_id=article_id, session_id=sid, user_id=user_id,
                force_refresh=force_refresh, client=http_client,
            )
        finally:
            if owns_client:
                await http_client.aclose()

        # Assemble the access panel: persisted rows + per-user institutional rows.
        article = await db[Collections.ARTICLES].find_one({"id": article_id}, {"_id": 0}) or {}

        # Seed an abstract extract so the paper is immediately summary/"chat"-ready
        # even when full text was license-blocked. Mirrors what the search pipeline
        # does; no-op when an OA full-text extract already exists (OA is preferred).
        try:
            from services.evidence_extraction import evidence_extraction
            await evidence_extraction.ensure_abstract_extract(
                article=article, search_session_id=sid, user_id=user_id,
                expires_at=session.get("expires_at"),
            )
        except Exception:  # noqa: BLE001 — best-effort; summary will still try
            logger.exception("full-text import: ensure_abstract_extract failed (non-fatal)")
        access_rows: list[dict[str, Any]] = []
        async for row in db[Collections.ARTICLE_ACCESS_LOCATIONS].find(
            {"article_id": article_id}, {"_id": 0},
        ):
            access_rows.append(row)
        inst_settings = await db[Collections.USER_INSTITUTION_SETTINGS].find_one(
            {"user_id": user_id}, {"_id": 0},
        )
        inst_rows = institutional_rows_for_user(article, inst_settings)
        all_rows = access_rows + inst_rows
        best_access = compute_best_access(article, all_rows)

        ingested = (
            oa_result.get("status") in _INGESTED_STATUSES
            and bool(oa_result.get("can_ingest_for_summary"))
        )
        abstract_present = bool((article.get("abstract") or "").strip())

        # 6. Optional save to the user's library (a research collection).
        saved = None
        if save:
            saved = await self._save_to_collection(
                user_id=user_id, article_id=article_id, session_id=sid,
                collection_id=collection_id, title=article.get("title"),
            )

        status = self._top_status(
            ingested=ingested, best_access=best_access, abstract_present=abstract_present,
        )

        await audit_service.log(
            "full_text_import",
            user_id=user_id, search_session_id=sid,
            payload={
                "article_id": article_id,
                "match_confidence": confidence,
                "oa_status": oa_result.get("status"),
                "oa_source": oa_result.get("source"),
                "ingested_for_ai": ingested,
                "best_access_category": best_access.get("access_category"),
                "status": status,
                "saved": bool(saved),
            },
        )

        chat_ready = ingested or abstract_present
        return {
            "status": status,
            "session_id": sid,
            "article_id": article_id,
            "search_session_article_id": attach.get("search_session_article_id"),
            "was_duplicate": bool(attach.get("was_duplicate")),
            "match": {
                "confidence": confidence,
                "reason_codes": top.get("match_reason_codes") or [],
                "source": top.get("source"),
                "lookup_source_provenance": search_result.get("lookup_source_provenance") or [],
            },
            "paper": self._paper_view(article),
            "full_text": {
                "ingested_for_ai": ingested,
                "oa_fetch": oa_result,
                "best_access": best_access,
                "access_locations": [_public_access_row(r) for r in all_rows],
            },
            "chat": {
                "ready": chat_ready,
                "evidence_text_level": oa_result.get("evidence_text_level"),
                "generate_summary_url": f"/api/search-sessions/{sid}/generate-summary",
                "summary_url": f"/api/search-sessions/{sid}/summary",
                "note": (
                    "Evidence chat / AI summary runs over this session's selected "
                    "article. Full text is used when OA-ingestible; otherwise the "
                    "abstract is used and full text is link-out only."
                ),
            },
            "saved": saved,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _top_status(*, ingested: bool, best_access: dict[str, Any], abstract_present: bool) -> str:
        if ingested:
            return "full_text_ingested"
        cat = best_access.get("access_category")
        if not best_access.get("disabled") and cat not in (None, "abstract_only", "unknown"):
            # A real access route exists (OA link, institutional, or DOI) the
            # human can open, but the text was not AI-ingested.
            return "full_text_link_out"
        if abstract_present:
            return "abstract_only"
        return "metadata_only"

    async def _get_or_create_import_session(
        self, db, *, user_id: str, session_id: Optional[str],
        canonical_key: str, title: Optional[str], force_refresh: bool,
    ) -> dict[str, Any]:
        # Explicit reuse of a caller-owned session.
        if session_id:
            owned = await db[Collections.SEARCH_SESSIONS].find_one(
                {"id": session_id, "user_id": user_id, "deleted_at": None}, {"_id": 0},
            )
            if not owned:
                raise FullTextImportError("session_not_found", status=404)
            if owned.get("status") == "blocked_phi":
                raise FullTextImportError("session_blocked_phi", status=400)
            return owned

        # Idempotent reuse: one import session per (user, paper) unless refreshing.
        if not force_refresh:
            existing = await db[Collections.SEARCH_SESSIONS].find_one(
                {"user_id": user_id, "deleted_at": None,
                 "import_origin": IMPORT_ORIGIN,
                 "imported_canonical_key": canonical_key},
                {"_id": 0},
            )
            if existing:
                return existing

        q = f"Full-text import: {title}" if title else "Full-text import"
        s = get_settings()
        doc = SearchSessionDoc(
            user_id=user_id,
            original_question=q[:5000],
            sanitized_question=q[:5000],
            status="awaiting_selection",
            expires_at=datetime.now(timezone.utc) + timedelta(days=s.SESSION_TTL_DAYS),
        ).model_dump()
        for k in ("created_at", "updated_at", "expires_at"):
            doc[k] = doc[k].astimezone(timezone.utc)
        # Markers (ignored by the Pydantic model; persisted on the raw doc).
        doc["import_origin"] = IMPORT_ORIGIN
        doc["imported_canonical_key"] = canonical_key
        await db[Collections.SEARCH_SESSIONS].insert_one(doc)
        await audit_service.log(
            "full_text_import_session_created",
            user_id=user_id, search_session_id=doc["id"],
            payload={"canonical_key": canonical_key},
        )
        return doc

    @staticmethod
    async def _enrich_article(db, article_id: str, candidate: dict[str, Any]) -> None:
        """Backfill source/identity fields the known-article upsert omits, so
        B16 eligibility, B12 provenance and the LitHub mirror behave correctly."""
        art = await db[Collections.ARTICLES].find_one(
            {"id": article_id},
            {"_id": 0, "sources_present": 1, "doi_normalized": 1, "source_ids_json": 1},
        ) or {}
        update: dict[str, Any] = {}
        src = (candidate.get("source") or "pubmed").strip().lower()
        if not art.get("sources_present"):
            update["sources_present"] = [src]
        doi = candidate.get("doi")
        if doi and not art.get("doi_normalized"):
            nd = normalize_doi(doi)
            if nd:
                update["doi_normalized"] = nd
        sids = dict(art.get("source_ids_json") or {})
        changed_sids = False
        if candidate.get("pmid") and not sids.get("pubmed"):
            sids["pubmed"] = candidate["pmid"]; changed_sids = True
        if candidate.get("doi") and not sids.get("doi"):
            sids["doi"] = normalize_doi(candidate["doi"]) or candidate["doi"]; changed_sids = True
        if candidate.get("pmcid") and not sids.get("pmc"):
            sids["pmc"] = candidate["pmcid"]; changed_sids = True
        if changed_sids:
            update["source_ids_json"] = sids
        if update:
            update["updated_at"] = datetime.now(timezone.utc)
            await db[Collections.ARTICLES].update_one({"id": article_id}, {"$set": update})

    async def _save_to_collection(
        self, *, user_id: str, article_id: str, session_id: str,
        collection_id: Optional[str], title: Optional[str],
    ) -> dict[str, Any]:
        from services.research_collection_service import (
            CollectionError,
            research_collection_service,
        )
        try:
            if collection_id:
                col = await research_collection_service.get(user_id, collection_id)
            else:
                col = await research_collection_service.get_or_create_default(user_id)
            result = await research_collection_service.add_records(
                user_id, col["id"],
                AddRecordsRequest(records=[AddRecordItem(
                    article_id=article_id, search_session_id=session_id, title=title,
                )]),
            )
            return {
                "collection_id": col["id"],
                "collection_name": col.get("name"),
                "added_count": result.get("added_count", 0),
                "skipped_count": result.get("skipped_count", 0),
            }
        except CollectionError as exc:
            return {"error": exc.code}
        except Exception:  # noqa: BLE001 — save is best-effort, never fails the import
            logger.exception("full-text import: save to collection failed (non-fatal)")
            return {"error": "save_failed"}

    @staticmethod
    def _paper_view(article: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": article.get("id"),
            "pmid": article.get("pmid"),
            "doi": article.get("doi"),
            "pmcid": article.get("pmcid"),
            "title": article.get("title"),
            "journal": article.get("journal"),
            "publication_year": article.get("publication_year"),
            "abstract_present": bool((article.get("abstract") or "").strip()),
            "is_retracted": bool(article.get("is_retracted")),
            "full_text_status": article.get("full_text_status"),
        }

    @staticmethod
    def _paper_view_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": None,
            "pmid": candidate.get("pmid"),
            "doi": candidate.get("doi"),
            "pmcid": candidate.get("pmcid"),
            "title": candidate.get("title"),
            "journal": candidate.get("journal"),
            "publication_year": candidate.get("publication_year"),
            "abstract_present": bool((candidate.get("abstract") or "").strip()),
            "is_retracted": bool(candidate.get("is_retracted")),
            "full_text_status": None,
        }


full_text_import_service = FullTextImportService()
