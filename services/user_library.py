"""Phase 6D — UserLibraryService.

Persistent saved-article layer + folder workspace helpers. Survives
30-day session expiration. All writes go through closed-set audit
events; no PHI / raw evidence text / secrets are ever logged.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from core.db import Collections, get_db
from models.library import (
    LibraryBatchSaveRequest,
    LibraryItemCreate,
    LibraryItemImport,
    LibraryItemUpdate,
    UserLibraryItemDoc,
)
from services.audit import audit_service
from services.canonical_identity import (
    make_canonical_key,
    normalize_doi,
)


class LibraryError(Exception):
    """Raised with a closed-set error code for the router to translate
    to an HTTP error. ``code`` is snake_case; ``status`` is the HTTP
    status to surface.
    """
    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


class UserLibraryService:
    """Saved-article + folder-workspace operations.

    Responsibilities:
    - save / update / delete / archive / unarchive / favorite library items
    - batch-save N articles to library and (optionally) a folder
    - import by external identifier (PubMed PMID, DOI, PMCID) via the
      existing closed-set connectors (NO Google Scholar, NO scraping)
    - emit closed-set audit events for every mutation
    - emit a PASSIVE ``relevance_event_recorded`` for ranker telemetry
      (never read by the ranker)
    """

    # ------------------------------------------------------------------
    # Library items
    # ------------------------------------------------------------------
    async def list_items(
        self,
        user_id: str,
        *,
        status: Optional[str] = None,
        favorite: Optional[bool] = None,
        tag: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> dict[str, Any]:
        db = get_db()
        q: dict[str, Any] = {"user_id": user_id}
        if status:
            q["status"] = status
        if favorite is True:
            q["is_favorite"] = True
        elif favorite is False:
            q["is_favorite"] = False
        if tag:
            q["tags"] = tag
        cursor = db[Collections.USER_LIBRARY_ITEMS].find(
            q, {"_id": 0},
        ).sort("saved_at", -1).skip(skip).limit(limit)
        items = await cursor.to_list(length=limit)
        # Hydrate with article metadata (title, authors, journal, year,
        # ids) — closed-set fields, no abstract / no PHI.
        article_ids = [it["article_id"] for it in items]
        art_map: dict[str, dict] = {}
        if article_ids:
            proj = {
                "_id": 0, "id": 1, "title": 1, "journal": 1,
                "publication_year": 1, "authors_json": 1,
                "doi": 1, "doi_normalized": 1, "pmid": 1, "pmcid": 1,
                "canonical_key": 1, "article_type_classification": 1,
                "is_guideline": 1, "full_text_status": 1,
                "source_urls_json": 1,
            }
            async for row in db[Collections.ARTICLES].find(
                {"id": {"$in": article_ids}}, proj,
            ):
                art_map[row["id"]] = row
        # Phase H2-tail Task A — defensive author-email scrub on
        # hydrated library cards. Legacy records ingested before the
        # centralized persistence redactor must not leak emails out
        # through the library listing.
        if art_map:
            from services.article_email_redactor import redact_article_emails
            for _aid in list(art_map.keys()):
                art_map[_aid] = redact_article_emails(art_map[_aid])
        if search:
            s = search.lower().strip()
            items = [
                it for it in items
                if s in (art_map.get(it["article_id"], {}).get("title") or "").lower()
                or s in (it.get("note") or "").lower()
                or any(s in (t or "").lower() for t in (it.get("tags") or []))
            ]
        total = await db[Collections.USER_LIBRARY_ITEMS].count_documents(q)
        result = {
            "items": [{**it, "article": art_map.get(it["article_id"])} for it in items],
            "total": total,
            "limit": limit,
            "skip": skip,
        }
        # Best-effort, gated read-through of the central LitHub library so
        # papers saved on other Scienthesis apps (e.g. LitPulse) appear here.
        return await self._maybe_merge_central(result=result, user_id=user_id, status=status, skip=skip)

    async def get_item(self, user_id: str, item_id: str) -> dict[str, Any]:
        db = get_db()
        item = await db[Collections.USER_LIBRARY_ITEMS].find_one(
            {"id": item_id, "user_id": user_id}, {"_id": 0},
        )
        if not item:
            raise LibraryError("library_item_not_found", status=404)
        return item

    async def save_item(
        self, user_id: str, body: LibraryItemCreate,
    ) -> tuple[dict[str, Any], bool]:
        """Save an article to the user's library. Returns ``(doc, created)``
        where ``created`` is True if a NEW row was inserted (False if the
        article was already active in the library — idempotent re-save).
        """
        db = get_db()
        # Resolve the article: a local id (fast path), a central LitHub paper id
        # ("lithub:<id>" / bare), or a PMID/DOI/PMCID — importing/creating it when
        # it is not already a local article. This fixes the integrated frontend's
        # "article_not_found" when saving a paper that lives in LitHub/LitPulse
        # rather than in litportal's local article store.
        art = await self._resolve_article_for_save(body)
        if not art:
            raise LibraryError("article_not_found", status=404)
        article_id = art["id"]
        canonical_key = art.get("canonical_key") or ""
        existing = await db[Collections.USER_LIBRARY_ITEMS].find_one(
            {
                "user_id": user_id, "article_id": article_id,
                "status": "active",
            },
            {"_id": 0},
        )
        if existing:
            # Idempotent — just merge in the (possibly new) note / tags.
            update: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
            if body.note is not None:
                update["note"] = body.note
            if body.tags:
                merged = list(existing.get("tags") or [])
                for t in body.tags:
                    if t not in merged:
                        merged.append(t)
                update["tags"] = merged[:20]
            if body.is_favorite:
                update["is_favorite"] = True
            await db[Collections.USER_LIBRARY_ITEMS].update_one(
                {"id": existing["id"]}, {"$set": update},
            )
            fresh = await db[Collections.USER_LIBRARY_ITEMS].find_one(
                {"id": existing["id"]}, {"_id": 0},
            )
            await audit_service.log(
                "library_item_saved",
                user_id=user_id,
                payload={
                    "library_item_id": existing["id"],
                    "article_id": article_id,
                    "was_duplicate": True,
                },
            )
            await self._maybe_mirror_save(user_id, article_id)
            return fresh, False
        doc = UserLibraryItemDoc(
            user_id=user_id,
            article_id=article_id,
            canonical_key=canonical_key,
            note=body.note,
            tags=body.tags,
            is_favorite=body.is_favorite,
            saved_from_search_session_id=body.saved_from_search_session_id,
            saved_from_rank=body.saved_from_rank,
            saved_from_question_type=body.saved_from_question_type,
            saved_from_specialty=body.saved_from_specialty,
        ).model_dump()
        for k in ("saved_at", "created_at", "updated_at"):
            doc[k] = doc[k].astimezone(timezone.utc)
        await db[Collections.USER_LIBRARY_ITEMS].insert_one(doc)
        await audit_service.log(
            "library_item_saved",
            user_id=user_id,
            payload={
                "library_item_id": doc["id"],
                "article_id": body.article_id,
                "was_duplicate": False,
                "saved_from_search_session_id": body.saved_from_search_session_id,
            },
        )
        # Phase 6D Task 14 — passive relevance event. NOT read by ranker.
        if body.saved_from_search_session_id:
            await audit_service.log(
                "relevance_event_recorded",
                user_id=user_id,
                search_session_id=body.saved_from_search_session_id,
                payload={
                    "event_type": "library_item_saved",
                    "article_id": article_id,
                    "rank_at_save": body.saved_from_rank,
                    "selected_at_save": True,
                    "question_type": body.saved_from_question_type,
                    "specialty": body.saved_from_specialty,
                },
            )
        await self._maybe_mirror_save(user_id, article_id)
        return {k: v for k, v in doc.items() if k != "_id"}, True

    # ------------------------------------------------------------------
    # Article resolution for save (local id / LitHub paper id / identifier)
    # ------------------------------------------------------------------
    _SAVE_PROJ = {"_id": 0, "id": 1, "canonical_key": 1}

    async def _resolve_article_for_save(
        self, body: LibraryItemCreate,
    ) -> Optional[dict[str, Any]]:
        """Resolve ``body.article_id`` (+ optional identifiers/metadata) to a
        local ``articles`` row, importing/creating it when necessary.

        Order: (1) exact local id → (2) PMID/DOI/PMCID/canonical_key lookup →
        (3) import from central LitHub by paper id → (4) find-or-create from any
        provided metadata. Returns ``{id, canonical_key}`` or None.
        """
        db = get_db()
        raw = (body.article_id or "").strip()

        # 1) Exact local article id — the common, fast path (unchanged behavior).
        if raw:
            art = await db[Collections.ARTICLES].find_one({"id": raw}, self._SAVE_PROJ)
            if art:
                return art

        # 1b) The integrated frontend sends the search_session_articles ROW id
        # (the article-card "id" from GET /search-sessions/{id}, a 32-char hex),
        # NOT the article's own id. Map it to the underlying article. Scope to
        # the session when the caller provides saved_from_search_session_id.
        if raw:
            ssa_q: dict[str, Any] = {"id": raw}
            if body.saved_from_search_session_id:
                ssa_q["search_session_id"] = body.saved_from_search_session_id
            ssa = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
                ssa_q, {"_id": 0, "article_id": 1},
            )
            if ssa and ssa.get("article_id"):
                art = await db[Collections.ARTICLES].find_one(
                    {"id": ssa["article_id"]}, self._SAVE_PROJ,
                )
                if art:
                    return art

        # Collect identifiers from the id string + optional body fields.
        paper_id: Optional[str] = None
        if raw.startswith("lithub:"):
            paper_id = raw.split(":", 1)[1].strip() or None
        if body.paper_id:
            paper_id = str(body.paper_id).strip() or paper_id

        pmid = self._norm_pmid(body.pmid)
        if not pmid and raw.isdigit() and 4 <= len(raw) <= 9:
            pmid = raw
        doi_norm = normalize_doi(body.doi or "") or None
        if not doi_norm and raw.lower().startswith("10."):
            doi_norm = normalize_doi(raw) or None
        pmcid: Optional[str] = None
        _raw_pmcid = (body.pmcid or "").strip().upper()
        if _raw_pmcid:
            pmcid = _raw_pmcid if _raw_pmcid.startswith("PMC") else f"PMC{_raw_pmcid}"
        elif raw.upper().startswith("PMC"):
            pmcid = raw.upper()

        # 2) Identifier lookup against existing local articles.
        or_q: list[dict[str, Any]] = []
        if pmid:
            or_q.append({"pmid": pmid})
        if doi_norm:
            or_q.append({"doi_normalized": doi_norm})
        if pmcid:
            or_q.append({"pmcid": pmcid})
        if raw:
            or_q.append({"canonical_key": raw})
        if or_q:
            art = await db[Collections.ARTICLES].find_one({"$or": or_q}, self._SAVE_PROJ)
            if art:
                return art

        # 3) Import from the central LitHub library by paper id.
        if paper_id:
            imported = await self._import_article_from_lithub(paper_id)
            if imported:
                return imported

        # 4) Find-or-create from whatever identifiers/metadata we have.
        if pmid or doi_norm or pmcid or body.title:
            created = await self._create_local_article(
                pmid=pmid, doi=body.doi, doi_norm=doi_norm, pmcid=pmcid,
                title=body.title, journal=body.journal,
                year=body.publication_year, authors=body.authors,
            )
            if created:
                return created
        return None

    async def _import_article_from_lithub(
        self, paper_id: str,
    ) -> Optional[dict[str, Any]]:
        """Fetch a paper's metadata from central LitHub and find-or-create a
        local article. Best-effort; None on any failure / when LitHub is unset."""
        from core.config import get_settings
        s = get_settings()
        if not s.LITHUB_BASE_URL:
            return None
        try:
            from clients.lithub_client import LitHubClient
            async with httpx.AsyncClient(timeout=8.0) as client:
                papers = await LitHubClient(
                    http_client=client, settings=s,
                ).internal_papers_bulk([paper_id])
            p = (papers or [None])[0] if isinstance(papers, list) else None
            if not isinstance(p, dict):
                return None
            return await self._create_local_article(
                pmid=self._norm_pmid(p.get("pmid")),
                doi=p.get("doi"),
                doi_norm=normalize_doi(p.get("doi") or "") or None,
                pmcid=((p.get("pmcid") or "").strip().upper() or None),
                title=p.get("title"),
                journal=p.get("journal"),
                year=self._year_from(
                    p.get("year") or p.get("publication_year") or p.get("pub_date"),
                ),
                authors=p.get("authors"),
            )
        except Exception:  # noqa: BLE001 — best-effort import, never blocks save
            return None

    async def _create_local_article(
        self, *, pmid, doi, doi_norm, pmcid, title, journal, year, authors,
    ) -> Optional[dict[str, Any]]:
        """Find-or-create a local ``articles`` row from identifiers/metadata via
        the pipeline upsert (dedups by canonical_key/pmid/doi)."""
        from models.common import new_id, utcnow
        from services.pipeline import _upsert_article
        first_author = None
        authors_json: list[dict[str, Any]] = []
        for a in (authors or []):
            if isinstance(a, str) and a.strip():
                authors_json.append({"fullName": a.strip()})
                first_author = first_author or a.strip()
            elif isinstance(a, dict):
                authors_json.append(a)
                first_author = first_author or (a.get("fullName") or a.get("name"))
        ck = make_canonical_key(
            pmid=pmid, doi=doi_norm or doi, pmcid=pmcid,
            title=title, year=year, journal=journal, first_author=first_author,
        )
        rec = {
            "id": new_id(), "canonical_key": ck,
            "title": title or "(untitled)", "abstract": None,
            "journal": journal or None, "publication_year": year,
            "doi": doi or doi_norm or None, "doi_normalized": doi_norm,
            "pmid": pmid, "pmcid": pmcid,
            "authors_json": authors_json, "source_ids_json": {},
            "publication_types_json": [], "is_guideline": False,
            "is_preprint": False, "is_retracted": False,
            "has_expression_of_concern": False, "full_text_status": "unknown",
            "sources_present": ["lithub_import"],
            "source_urls_json": [], "created_at": utcnow(), "updated_at": utcnow(),
        }
        try:
            aid = await _upsert_article(rec)
        except Exception:  # noqa: BLE001
            return None
        db = get_db()
        return await db[Collections.ARTICLES].find_one({"id": aid}, self._SAVE_PROJ)

    async def update_item(
        self, user_id: str, item_id: str, body: LibraryItemUpdate,
    ) -> dict[str, Any]:
        db = get_db()
        item = await self.get_item(user_id, item_id)
        update: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        evt: list[str] = []
        if body.note is not None:
            update["note"] = body.note
            evt.append("library_note_updated")
        if body.tags is not None:
            update["tags"] = body.tags
            evt.append("library_tags_updated")
        if body.is_favorite is not None:
            update["is_favorite"] = body.is_favorite
            evt.append(
                "library_item_favorited"
                if body.is_favorite
                else "library_item_unfavorited",
            )
        if body.status is not None and body.status != item.get("status"):
            update["status"] = body.status
            if body.status == "archived":
                update["archived_at"] = datetime.now(timezone.utc)
                evt.append("library_item_archived")
            else:
                update["archived_at"] = None
                evt.append("library_item_unarchived")
        await db[Collections.USER_LIBRARY_ITEMS].update_one(
            {"id": item_id, "user_id": user_id}, {"$set": update},
        )
        for code in evt:
            await audit_service.log(
                code, user_id=user_id,
                payload={"library_item_id": item_id},
            )
        return await self.get_item(user_id, item_id)

    async def delete_item(self, user_id: str, item_id: str) -> dict[str, str]:
        db = get_db()
        item = await self.get_item(user_id, item_id)
        await db[Collections.USER_LIBRARY_ITEMS].delete_one(
            {"id": item_id, "user_id": user_id},
        )
        await audit_service.log(
            "library_item_removed", user_id=user_id,
            payload={"library_item_id": item_id},
        )
        await self._maybe_mirror_delete(user_id, item.get("article_id"))
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Scienthesis LitHub mirror (gated by LITPORTAL_DUAL_WRITE_LITHUB).
    # All helpers are best-effort and NEVER raise into the caller — a
    # LitHub outage must never break a local library operation.
    # ------------------------------------------------------------------
    async def _identity_sub(self, user_id: str) -> Optional[str]:
        """Return the user's Identity ``sub`` (the cross-app key) or None.

        Only identity-provisioned users have an ``identity_id``; native-only
        accounts return None and are not mirrored (they have no central key).
        """
        db = get_db()
        user = await db[Collections.USERS].find_one(
            {"id": user_id}, {"_id": 0, "identity_id": 1},
        )
        return (user or {}).get("identity_id")

    async def _article_lithub_item(self, article_id: str) -> Optional[dict[str, Any]]:
        """Build a LitHub SavePaperRequest item from a local article, or None
        when the article is missing or carries neither a PMID nor a DOI."""
        db = get_db()
        art = await db[Collections.ARTICLES].find_one(
            {"id": article_id},
            {
                "_id": 0, "pmid": 1, "doi": 1, "doi_normalized": 1, "title": 1,
                "journal": 1, "publication_year": 1, "abstract": 1, "authors_json": 1,
            },
        )
        if not art:
            return None
        pmid = (art.get("pmid") or "").strip() or None
        doi = (art.get("doi") or art.get("doi_normalized") or "").strip() or None
        if not (pmid or doi):
            return None
        authors: list[str] = []
        for a in (art.get("authors_json") or []):
            if isinstance(a, dict):
                nm = a.get("fullName") or a.get("full_name") or a.get("name")
                if nm:
                    authors.append(nm)
            elif isinstance(a, str) and a:
                authors.append(a)
        return {
            "pmid": pmid,
            "doi": doi,
            "title": art.get("title") or "Untitled paper",
            "journal": art.get("journal"),
            "abstract": art.get("abstract"),
            "year": art.get("publication_year"),
            "authors": authors,
            "source": "litportal-library",
            # Durable LitPortal-origin marker (LitHub keeps this fill-when-empty on
            # re-save, unlike `source`) so LitPulse can reliably group these under
            # "LitPortal Saved" even after a cross-app re-save.
            "portal_engine_record_id": f"litportal:{article_id}",
        }

    async def _maybe_mirror_save(self, user_id: str, article_id: str) -> None:
        from core.config import get_settings
        s = get_settings()
        if not (s.LITPORTAL_DUAL_WRITE_LITHUB and s.LITHUB_BASE_URL):
            return
        try:
            sub = await self._identity_sub(user_id)
            if not sub:
                return
            item = await self._article_lithub_item(article_id)
            if item is None:
                return
            from services import lithub_sync
            await lithub_sync.mirror_save(sub, item)
        except Exception:  # noqa: BLE001 — mirror is best-effort
            pass

    async def _maybe_mirror_delete(self, user_id: str, article_id: Optional[str]) -> None:
        from core.config import get_settings
        s = get_settings()
        if not (s.LITPORTAL_DUAL_WRITE_LITHUB and s.LITHUB_BASE_URL) or not article_id:
            return
        try:
            sub = await self._identity_sub(user_id)
            if not sub:
                return
            db = get_db()
            art = await db[Collections.ARTICLES].find_one(
                {"id": article_id}, {"_id": 0, "pmid": 1, "doi": 1, "doi_normalized": 1},
            )
            if not art:
                return
            pmid = (art.get("pmid") or "").strip() or None
            doi = (art.get("doi") or art.get("doi_normalized") or "").strip() or None
            if not (pmid or doi):
                return
            from services import lithub_sync
            await lithub_sync.mirror_delete(sub, pmid=pmid, doi=doi)
        except Exception:  # noqa: BLE001 — mirror is best-effort
            pass

    @staticmethod
    def _year_from(pub_date: Any) -> Optional[int]:
        if not pub_date:
            return None
        s = str(pub_date)[:4]
        return int(s) if s.isdigit() else None

    @staticmethod
    def _norm_pmid(v: Any) -> Optional[str]:
        """PMIDs may arrive as int or str — coerce defensively before .strip()."""
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @staticmethod
    def _norm_doi(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip().lower()
        return s or None

    @staticmethod
    def _saved_at_sort_key(item: dict[str, Any]) -> str:
        """Uniform sort key across local (datetime) and central (ISO str) saves.
        None sorts last under reverse=True."""
        v = item.get("saved_at")
        if v is None:
            return ""
        if hasattr(v, "isoformat"):
            try:
                return v.isoformat()
            except Exception:  # noqa: BLE001
                return ""
        return str(v)

    async def _full_local_identifiers(self, user_id: str) -> tuple[set[str], set[str]]:
        """Return (pmids, dois) across the user's ENTIRE active local library.

        Used so the central read-through dedups against all pages, not just the
        current one — otherwise a paper saved locally on page 2+ would be
        re-surfaced as a duplicate 'lithub:' card on page 1.
        """
        db = get_db()
        seen_pmid: set[str] = set()
        seen_doi: set[str] = set()
        art_ids = await db[Collections.USER_LIBRARY_ITEMS].distinct(
            "article_id", {"user_id": user_id, "status": "active"},
        )
        if not art_ids:
            return seen_pmid, seen_doi
        async for a in db[Collections.ARTICLES].find(
            {"id": {"$in": art_ids}},
            {"_id": 0, "pmid": 1, "doi": 1, "doi_normalized": 1},
        ):
            p = self._norm_pmid(a.get("pmid"))
            if p:
                seen_pmid.add(p)
            d = self._norm_doi(a.get("doi_normalized") or a.get("doi"))
            if d:
                seen_doi.add(d)
        return seen_pmid, seen_doi

    def _central_to_item(self, user_id: str, ca: dict[str, Any]) -> dict[str, Any]:
        """Map a LitHub central-library article into the local list item shape
        (an item dict with an embedded hydrated ``article``), marked as central."""
        paper_id = ca.get("paper_id") or ca.get("id")
        entry_id = ca.get("library_entry_id") or paper_id
        authors = [
            {"fullName": a} for a in (ca.get("authors") or []) if isinstance(a, str)
        ]
        return {
            "id": f"lithub:{entry_id}",
            "user_id": user_id,
            "article_id": f"lithub:{paper_id}",
            "canonical_key": None,
            "note": ca.get("notes"),
            "tags": [],
            "is_favorite": False,
            "status": "active",
            "saved_at": ca.get("saved_at"),
            "source": "lithub",
            "is_central": True,
            "lithub_paper_id": paper_id,
            "lithub_library_entry_id": ca.get("library_entry_id"),
            "article": {
                "id": f"lithub:{paper_id}",
                "title": ca.get("title"),
                "journal": ca.get("journal"),
                "publication_year": self._year_from(ca.get("pub_date")),
                "authors_json": authors,
                "doi": ca.get("doi"),
                "pmid": self._norm_pmid(ca.get("pmid")),
                "pmcid": None,
                "canonical_key": None,
                "full_text_status": ca.get("full_text_status"),
            },
        }

    async def _maybe_merge_central(
        self, user_id: str, result: dict[str, Any], *,
        status: Optional[str], skip: int,
    ) -> dict[str, Any]:
        """Append central LitHub papers not already saved locally (best-effort).

        Lets papers a user saved on LitPulse surface in LitPortal's library.
        Only runs on the first page of the active view; any error → local-only.
        """
        from core.config import get_settings
        s = get_settings()
        if not (s.LITPORTAL_DUAL_WRITE_LITHUB and s.LITHUB_BASE_URL):
            return result
        if skip or status not in (None, "active"):
            return result
        try:
            sub = await self._identity_sub(user_id)
            if not sub:
                return result
            from services import lithub_sync
            central = await lithub_sync.list_central(sub, params={"limit": 200})
            if not central:
                return result
            # Dedup against the ENTIRE local library (all pages), not just the
            # current page, so cross-page duplicates are never re-surfaced.
            seen_pmid, seen_doi = await self._full_local_identifiers(user_id)
            extra: list[dict[str, Any]] = []
            for ca in (central.get("articles") or []):
                pmid = self._norm_pmid(ca.get("pmid"))
                doi = self._norm_doi(ca.get("doi"))
                if (pmid and pmid in seen_pmid) or (doi and doi in seen_doi):
                    continue
                extra.append(self._central_to_item(user_id, ca))
            if not extra:
                return result
            merged = dict(result)
            items = list(result.get("items", [])) + extra
            # Interleave cross-app saves by recency (newest first), matching the
            # local saved_at desc ordering.
            items.sort(key=self._saved_at_sort_key, reverse=True)
            merged["items"] = items
            # total reflects the de-duplicated union (local total + central-only).
            # NOTE: page 1 of this unified view may exceed `limit` because it
            # augments the local page with central-only saves (best-effort).
            merged["total"] = int(result.get("total", 0)) + len(extra)
            return merged
        except Exception:  # noqa: BLE001 — read-through is best-effort
            return result

    # ------------------------------------------------------------------
    # Batch save
    # ------------------------------------------------------------------
    async def batch_save(
        self, user_id: str, body: LibraryBatchSaveRequest,
    ) -> dict[str, Any]:
        """Save N articles to library (+ optionally one folder). Idempotent.

        Phase 6D.1 — server-side guard: when ``saved_from_search_session_id``
        is provided, articles whose ``search_session_articles`` row has
        ``exclusion_reason != None`` are skipped. Also: any article with
        ``is_retracted=True`` is skipped regardless of source. Returns
        the skipped article_ids with reason codes so the UI can show a
        "Skipped K unsafe" message.
        """
        db = get_db()

        # Build the exclusion set BEFORE saving.
        skipped: list[dict[str, Any]] = []
        article_ids_to_save: list[str] = []
        # Look up the session's excluded rows (one query).
        excluded_set: set[str] = set()
        if body.saved_from_search_session_id:
            cursor = db[Collections.SEARCH_SESSION_ARTICLES].find(
                {
                    "search_session_id": body.saved_from_search_session_id,
                    "exclusion_reason": {"$nin": [None, ""]},
                },
                {"_id": 0, "article_id": 1, "exclusion_reason": 1},
            )
            async for row in cursor:
                excluded_set.add(row["article_id"])
        # Look up retracted articles in one query.
        retracted_set: set[str] = set()
        if body.article_ids:
            async for art in db[Collections.ARTICLES].find(
                {
                    "id": {"$in": body.article_ids},
                    "is_retracted": True,
                },
                {"_id": 0, "id": 1},
            ):
                retracted_set.add(art["id"])
        for art_id in body.article_ids:
            if art_id in excluded_set:
                skipped.append({
                    "article_id": art_id,
                    "reason": "session_excluded",
                })
                continue
            if art_id in retracted_set:
                skipped.append({
                    "article_id": art_id,
                    "reason": "retracted",
                })
                continue
            article_ids_to_save.append(art_id)

        results: list[dict[str, Any]] = []
        created = 0
        for art_id in article_ids_to_save:
            try:
                doc, was_created = await self.save_item(
                    user_id,
                    LibraryItemCreate(
                        article_id=art_id,
                        note=body.note,
                        tags=body.tags,
                        saved_from_search_session_id=body.saved_from_search_session_id,
                    ),
                )
                results.append(
                    {"article_id": art_id, "library_item_id": doc["id"],
                      "created": was_created},
                )
                if was_created:
                    created += 1
            except LibraryError as e:
                results.append({"article_id": art_id, "error": e.code})
        added_to_folder = 0
        if body.folder_id:
            from models.folders import FolderArticleDoc
            folder = await db[Collections.FOLDERS].find_one(
                {"id": body.folder_id, "user_id": user_id}, {"_id": 0},
            )
            if not folder:
                raise LibraryError("folder_not_found", status=404)
            for r in results:
                if "error" in r:
                    continue
                # Skip if already in folder.
                exists = await db[Collections.FOLDER_ARTICLES].find_one(
                    {"folder_id": body.folder_id, "article_id": r["article_id"]},
                    {"_id": 0, "id": 1},
                )
                if exists:
                    r["folder_article_id"] = exists["id"]
                    continue
                fa = FolderArticleDoc(
                    folder_id=body.folder_id,
                    article_id=r["article_id"],
                    from_library_item_id=r["library_item_id"],
                    user_notes=body.note,
                    tags=list(body.tags),
                    saved_from_search_session_id=body.saved_from_search_session_id,
                ).model_dump()
                fa["created_at"] = fa["created_at"].astimezone(timezone.utc)
                await db[Collections.FOLDER_ARTICLES].insert_one(fa)
                r["folder_article_id"] = fa["id"]
                added_to_folder += 1
        await audit_service.log(
            "library_batch_saved",
            user_id=user_id,
            search_session_id=body.saved_from_search_session_id,
            payload={
                "n_articles_requested": len(body.article_ids),
                "n_articles_saved": len(article_ids_to_save),
                "n_library_created": created,
                "n_added_to_folder": added_to_folder,
                "n_skipped": len(skipped),
                "folder_id": body.folder_id,
            },
        )
        return {
            "n_articles": len(body.article_ids),
            "n_articles_attempted": len(article_ids_to_save),
            "n_library_created": created,
            "n_added_to_folder": added_to_folder,
            "n_skipped": len(skipped),
            "folder_id": body.folder_id,
            "results": results,
            "skipped": skipped,
        }

    # ------------------------------------------------------------------
    # Import by identifier (PMID / DOI / PMCID)
    # ------------------------------------------------------------------
    async def import_by_identifier(
        self, user_id: str, body: LibraryItemImport,
    ) -> dict[str, Any]:
        """Import a single article by external id and add to library.

        Strategy (closed-set, NO Scholar / NO scraping):
        1. Build candidate canonical_key from the provided ids.
        2. If the article already exists by ``pmid`` / ``doi_normalized``
           / ``pmcid``, use it.
        3. Else fetch via PubMed (PMID) or Europe PMC (DOI/PMCID) using
           the existing connectors; persist; then save to library.
        """
        db = get_db()
        pmid = (body.pmid or "").strip() or None
        doi_norm = normalize_doi(body.doi or "") or None
        pmcid = (body.pmcid or "").strip().upper() or None
        if not (pmid or doi_norm or pmcid):
            raise LibraryError("identifier_required", status=400)
        # Step 1 — lookup.
        q: dict[str, Any] = {"$or": []}
        if pmid:
            q["$or"].append({"pmid": pmid})
        if doi_norm:
            q["$or"].append({"doi_normalized": doi_norm})
        if pmcid:
            q["$or"].append({"pmcid": pmcid})
        existing = await db[Collections.ARTICLES].find_one(
            q, {"_id": 0, "id": 1},
        )
        article_id: Optional[str] = existing["id"] if existing else None

        # Step 2 — fetch if missing.
        source_hit: Optional[str] = None
        if not article_id:
            article_id, source_hit = await self._fetch_and_persist(
                pmid=pmid, doi=body.doi, doi_norm=doi_norm, pmcid=pmcid,
            )
        # Step 3 — save to library.
        doc, was_created = await self.save_item(
            user_id,
            LibraryItemCreate(
                article_id=article_id,
                note=body.note,
                tags=body.tags,
            ),
        )
        await audit_service.log(
            "library_article_imported",
            user_id=user_id,
            payload={
                "library_item_id": doc["id"],
                "article_id": article_id,
                "source_hit": source_hit or "existing",
                "was_new_article": source_hit is not None,
            },
        )
        return {"library_item": doc, "created": was_created, "source": source_hit or "existing"}

    async def _fetch_and_persist(
        self, *,
        pmid: Optional[str], doi: Optional[str],
        doi_norm: Optional[str], pmcid: Optional[str],
    ) -> tuple[str, str]:
        """Try Europe PMC (supports PMID/DOI/PMCID) → Crossref (DOI only).

        Persists the article doc and returns ``(article_id, source_hit)``.
        Raises LibraryError(``import_source_unavailable``) if nothing
        returns a record.
        """
        from services.metadata_normalizer import metadata_normalizer
        from services.pipeline import _upsert_article  # noqa: WPS437

        # Europe PMC search supports PMID, DOI, and PMCID via the same
        # ``query`` syntax.
        if pmid or doi_norm or pmcid:
            try:
                if pmid:
                    query = f"EXT_ID:{pmid} AND SRC:MED"
                    source_hit = "pubmed_via_europe_pmc"
                elif doi_norm:
                    query = f"DOI:{doi_norm}"
                    source_hit = "europe_pmc"
                else:
                    query = f"PMCID:{pmcid}"
                    source_hit = "europe_pmc"
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(
                        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                        params={
                            "query": query,
                            "format": "json",
                            "pageSize": 1,
                            "resultType": "core",
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    raws = (
                        ((data.get("resultList") or {}).get("result") or [])
                    )
                    if raws:
                        normalized = metadata_normalizer.normalize(raws[0])
                        art_id = await _upsert_article(normalized)
                        return art_id, source_hit
            except Exception:  # noqa: BLE001
                pass

        # Crossref by DOI (metadata-only).
        if doi_norm:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(
                        f"https://api.crossref.org/works/{doi_norm}",
                    )
                    if resp.status_code == 200:
                        msg = (resp.json() or {}).get("message") or {}
                        title = (msg.get("title") or [""])[0]
                        year = ((msg.get("issued") or {}).get("date-parts")
                                or [[None]])[0][0]
                        journal = (msg.get("container-title") or [""])[0]
                        first_author = ""
                        for a in (msg.get("author") or []):
                            if a.get("family"):
                                first_author = a["family"]
                                break
                        ck = make_canonical_key(
                            doi=doi_norm, title=title, year=year,
                            journal=journal, first_author=first_author,
                        )
                        from models.common import new_id, utcnow
                        rec = {
                            "id": new_id(),
                            "canonical_key": ck,
                            "title": title or "(untitled)",
                            "abstract": None,
                            "journal": journal or None,
                            "publication_year": year,
                            "doi": doi,
                            "doi_normalized": doi_norm,
                            "pmid": None,
                            "pmcid": None,
                            "authors_json": [
                                {"fullName": (a.get("family") or "")}
                                for a in (msg.get("author") or [])[:5]
                            ],
                            "source_ids_json": {"crossref": doi_norm},
                            "publication_types_json": (msg.get("type") and [msg["type"]]) or [],
                            "is_guideline": False,
                            "is_preprint": False,
                            "is_retracted": False,
                            "has_expression_of_concern": False,
                            "full_text_status": "unknown",
                            "source_urls_json": [],
                            "created_at": utcnow(),
                            "updated_at": utcnow(),
                        }
                        art_id = await _upsert_article(rec)
                        return art_id, "crossref"
            except Exception:  # noqa: BLE001
                pass

        raise LibraryError("import_source_unavailable", status=502)


library_service = UserLibraryService()
