"""B14 — LitArchive: a filterable repository/table view over a user's saved
articles. LitArchive is a VIEW, not a separate library: it unions the user's
saves across the local library, research collections, and folders, dedupes by
article, hydrates the table columns from the ``articles`` collection, and applies
filters. (When LitHub dual-write is on, these are the same papers mirrored to the
canonical central library.)

Filters: source, journal, year, pmid, doi, saved_from, collection_id, folder_id,
full_text_available, ai_summary_eligible, free-text q. No PHI is read.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from core.db import Collections, get_db
from services.downstream_eligibility import can_use_article_for_tool

logger = logging.getLogger("litportal.litarchive")

_GATHER_CAP = 5000
_FULLTEXT_UNAVAILABLE = {None, "", "unknown", "not_found", "unavailable", "none", "pending"}

_ARTICLE_PROJECTION = {
    "_id": 0, "id": 1, "title": 1, "authors_json": 1, "journal": 1,
    "publication_year": 1, "doi": 1, "doi_normalized": 1, "pmid": 1, "pmcid": 1,
    "canonical_key": 1, "article_type_classification": 1, "is_guideline": 1,
    "study_design": 1, "full_text_status": 1, "source_urls_json": 1,
    "sources_present": 1,
}


def _is_full_text_available(status: Any) -> bool:
    return str(status).strip().lower() not in _FULLTEXT_UNAVAILABLE if status is not None else False


class LitArchiveService:
    async def _gather_saved(
        self, user_id: str, *, saved_from: Optional[str],
        collection_id: Optional[str], folder_id: Optional[str],
    ) -> dict[str, dict[str, Any]]:
        """article_id -> provenance {saved_from:set, collection_ids, folder_ids, saved_at}."""
        db = get_db()
        prov: dict[str, dict[str, Any]] = {}

        def _add(aid: str, src: str, saved_at: Any, *, col: str = None, fol: str = None):
            p = prov.setdefault(aid, {
                "saved_from": set(), "collection_ids": set(), "folder_ids": set(), "saved_at": None,
            })
            p["saved_from"].add(src)
            if col:
                p["collection_ids"].add(col)
            if fol:
                p["folder_ids"].add(fol)
            if saved_at and (p["saved_at"] is None or saved_at > p["saved_at"]):
                p["saved_at"] = saved_at

        want = saved_from
        # Library items
        if want in (None, "library") and not collection_id and not folder_id:
            async for it in db[Collections.USER_LIBRARY_ITEMS].find(
                {"user_id": user_id, "status": "active"},
                {"_id": 0, "article_id": 1, "saved_at": 1},
            ).limit(_GATHER_CAP):
                _add(it["article_id"], "library", it.get("saved_at"))
        # Research collection items
        if want in (None, "collection") or collection_id:
            q: dict[str, Any] = {"user_id": user_id}
            if collection_id:
                q["collection_id"] = collection_id
            async for it in db[Collections.RESEARCH_COLLECTION_ITEMS].find(
                q, {"_id": 0, "article_id": 1, "collection_id": 1, "created_at": 1},
            ).limit(_GATHER_CAP):
                _add(it["article_id"], "collection", it.get("created_at"), col=it.get("collection_id"))
        # Folder articles (folders carry the user_id)
        if want in (None, "folder") or folder_id:
            fq: dict[str, Any] = {"user_id": user_id}
            if folder_id:
                fq["id"] = folder_id
            folder_ids = [
                f["id"] async for f in db[Collections.FOLDERS].find(fq, {"_id": 0, "id": 1})
            ]
            if folder_ids:
                async for fa in db[Collections.FOLDER_ARTICLES].find(
                    {"folder_id": {"$in": folder_ids}},
                    {"_id": 0, "article_id": 1, "folder_id": 1, "created_at": 1},
                ).limit(_GATHER_CAP):
                    _add(fa["article_id"], "folder", fa.get("created_at"), fol=fa.get("folder_id"))
        return prov

    async def query(
        self,
        user_id: str,
        *,
        source: Optional[str] = None,
        journal: Optional[str] = None,
        year: Optional[int] = None,
        pmid: Optional[str] = None,
        doi: Optional[str] = None,
        saved_from: Optional[str] = None,
        collection_id: Optional[str] = None,
        folder_id: Optional[str] = None,
        full_text_available: Optional[bool] = None,
        ai_summary_eligible: Optional[bool] = None,
        q: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> dict[str, Any]:
        db = get_db()
        prov = await self._gather_saved(
            user_id, saved_from=saved_from, collection_id=collection_id, folder_id=folder_id,
        )
        article_ids = list(prov.keys())
        if not article_ids:
            return {"items": [], "total": 0, "limit": limit, "skip": skip}

        art_map: dict[str, dict] = {}
        async for row in db[Collections.ARTICLES].find(
            {"id": {"$in": article_ids}}, _ARTICLE_PROJECTION,
        ):
            art_map[row["id"]] = row
        if art_map:
            from services.article_email_redactor import redact_article_emails
            for _aid in list(art_map.keys()):
                art_map[_aid] = redact_article_emails(art_map[_aid])

        rows: list[dict[str, Any]] = []
        s_journal = (journal or "").strip().lower()
        s_doi = (doi or "").strip().lower()
        s_q = (q or "").strip().lower()
        for aid, p in prov.items():
            art = art_map.get(aid)
            if not art:
                continue
            # Article-level filters.
            if source and source.strip().lower() not in {
                str(x).strip().lower() for x in (art.get("sources_present") or [])
            }:
                continue
            if s_journal and s_journal not in (art.get("journal") or "").lower():
                continue
            if year is not None and art.get("publication_year") != year:
                continue
            if pmid and (art.get("pmid") or "") != pmid:
                continue
            if s_doi and s_doi not in {
                (art.get("doi") or "").lower(), (art.get("doi_normalized") or "").lower(),
            }:
                continue
            if s_q and s_q not in (art.get("title") or "").lower():
                continue
            fta = _is_full_text_available(art.get("full_text_status"))
            if full_text_available is not None and fta != full_text_available:
                continue
            summary_ok = can_use_article_for_tool(art, "summary")["allowed"]
            if ai_summary_eligible is not None and summary_ok != ai_summary_eligible:
                continue
            rows.append({
                "article_id": aid,
                "saved_from": sorted(p["saved_from"]),
                "collection_ids": sorted(p["collection_ids"]),
                "folder_ids": sorted(p["folder_ids"]),
                "saved_at": p["saved_at"],
                "article": art,
                "badges": {
                    "full_text_available": fta,
                    "ai_summary_eligible": summary_ok,
                    "is_guideline": bool(art.get("is_guideline")),
                },
            })

        rows.sort(key=lambda r: (r["saved_at"] is not None, r["saved_at"]), reverse=True)
        total = len(rows)
        page = rows[skip: skip + limit]
        return {"items": page, "total": total, "limit": limit, "skip": skip}


litarchive_service = LitArchiveService()
