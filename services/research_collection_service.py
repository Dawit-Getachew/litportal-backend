"""Research Collections service (MongoDB).

User-owned hierarchical collections of saved papers. Mirrors litbridge's
research-collection behavior: on add, each paper is mirrored into the central
LitHub library (when ``LITPORTAL_DUAL_WRITE_LITHUB`` is on) keyed by the user's
Identity ``sub``, and the returned canonical ``paper_id`` is persisted on the
item so collections are shared with LitPulse. Mirroring is best-effort and never
blocks the local save (falls back to the reliable outbox).

All operations are user-scoped. PHI screening of free-text fields happens in the
router (mirrors routers/folders.py). Errors use the closed-set ``CollectionError``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from core.config import get_settings
from core.db import Collections, get_db
from models.research_collections import (
    AddRecordsRequest,
    CollectionCreate,
    CollectionUpdate,
    ResearchCollectionDoc,
    ResearchCollectionItemDoc,
)
from services.audit import audit_service

logger = logging.getLogger("litportal.research_collections")

_LITHUB_TIMEOUT_S = 8.0
# Article fields hydrated into list-view items (the "columns").
_ARTICLE_PROJECTION = {
    "_id": 0, "id": 1, "title": 1, "authors_json": 1, "journal": 1,
    "publication_year": 1, "doi": 1, "doi_normalized": 1, "pmid": 1, "pmcid": 1,
    "canonical_key": 1, "article_type_classification": 1, "is_guideline": 1,
    "study_design": 1, "full_text_status": 1, "source_urls_json": 1,
}


class CollectionError(Exception):
    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


def _serialize(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "_id"}


DEFAULT_IMPORTS_COLLECTION = "LitPortal Imports"


class ResearchCollectionService:
    # ── Collections ────────────────────────────────────────────────
    async def get_or_create_default(self, user_id: str) -> dict[str, Any]:
        """B13 — the user's default 'LitPortal Imports' collection (idempotent)."""
        db = get_db()
        existing = await db[Collections.RESEARCH_COLLECTIONS].find_one(
            {"user_id": user_id, "name": DEFAULT_IMPORTS_COLLECTION, "parent_id": None},
            {"_id": 0},
        )
        if existing:
            return existing
        return await self.create(user_id, CollectionCreate(
            name=DEFAULT_IMPORTS_COLLECTION,
            description="Articles imported from LitPortal.",
        ))

    async def create(self, user_id: str, body: CollectionCreate) -> dict[str, Any]:
        db = get_db()
        if body.parent_id:
            await self._owned_collection(user_id, body.parent_id)  # validate parent
        col = ResearchCollectionDoc(
            user_id=user_id,
            parent_id=body.parent_id,
            name=body.name.strip(),
            description=body.description,
            icon=body.icon,
            color=body.color,
        )
        doc = col.model_dump()
        for k in ("created_at", "updated_at"):
            doc[k] = doc[k].astimezone(timezone.utc)
        await db[Collections.RESEARCH_COLLECTIONS].insert_one(doc)
        await audit_service.log(
            "research_collection_created", user_id=user_id,
            payload={"collection_id": col.id, "parent_id": body.parent_id},
        )
        return _serialize(doc)

    async def list(self, user_id: str) -> list[dict[str, Any]]:
        db = get_db()
        cols = await db[Collections.RESEARCH_COLLECTIONS].find(
            {"user_id": user_id}, {"_id": 0},
        ).sort("created_at", -1).to_list(length=1000)
        for c in cols:
            c["item_count"] = await db[Collections.RESEARCH_COLLECTION_ITEMS].count_documents(
                {"collection_id": c["id"]},
            )
            c["children_count"] = await db[Collections.RESEARCH_COLLECTIONS].count_documents(
                {"parent_id": c["id"]},
            )
        return cols

    async def tree(self, user_id: str) -> dict[str, Any]:
        """Full collection hierarchy + each collection's papers' canonical ids,
        in ONE call. Consumed cross-app (LitArchive / LitPulse) to render the
        LitPortal sub-folders and match papers to central LitHub by
        pmid / doi / paper_id. 3 queries total (no N+1)."""
        db = get_db()
        cols = await db[Collections.RESEARCH_COLLECTIONS].find(
            {"user_id": user_id}, {"_id": 0},
        ).sort([("parent_id", 1), ("position", 1), ("created_at", 1)]).to_list(length=1000)
        col_ids = [c["id"] for c in cols]
        items_by_col: dict[str, list[dict]] = {}
        article_ids: set[str] = set()
        if col_ids:
            async for it in db[Collections.RESEARCH_COLLECTION_ITEMS].find(
                {"collection_id": {"$in": col_ids}},
                {"_id": 0, "collection_id": 1, "article_id": 1, "paper_id": 1, "title": 1},
            ):
                items_by_col.setdefault(it["collection_id"], []).append(it)
                if it.get("article_id"):
                    article_ids.add(it["article_id"])
        art_map: dict[str, dict] = {}
        if article_ids:
            async for a in db[Collections.ARTICLES].find(
                {"id": {"$in": list(article_ids)}},
                {"_id": 0, "id": 1, "pmid": 1, "doi": 1, "doi_normalized": 1,
                 "canonical_key": 1, "title": 1},
            ):
                art_map[a["id"]] = a
        out: list[dict[str, Any]] = []
        for c in cols:
            papers: list[dict[str, Any]] = []
            for it in items_by_col.get(c["id"], []):
                a = art_map.get(it.get("article_id")) or {}
                papers.append({
                    "article_id": it.get("article_id"),
                    "paper_id": it.get("paper_id"),  # central LitHub paper id (when mirrored)
                    "pmid": a.get("pmid"),
                    "doi": a.get("doi") or a.get("doi_normalized"),
                    "canonical_key": a.get("canonical_key"),
                    "title": it.get("title") or a.get("title"),
                })
            out.append({
                "id": c["id"], "name": c["name"], "parent_id": c.get("parent_id"),
                "position": c.get("position", 0), "icon": c.get("icon"),
                "color": c.get("color"), "item_count": len(papers),
                "papers": papers,
            })
        return {"collections": out}

    async def _owned_collection(self, user_id: str, collection_id: str) -> dict[str, Any]:
        db = get_db()
        col = await db[Collections.RESEARCH_COLLECTIONS].find_one(
            {"id": collection_id, "user_id": user_id}, {"_id": 0},
        )
        if not col:
            raise CollectionError("collection_not_found", status=404)
        return col

    async def get(self, user_id: str, collection_id: str) -> dict[str, Any]:
        col = await self._owned_collection(user_id, collection_id)
        db = get_db()
        col["item_count"] = await db[Collections.RESEARCH_COLLECTION_ITEMS].count_documents(
            {"collection_id": collection_id},
        )
        return col

    async def update(
        self, user_id: str, collection_id: str, body: CollectionUpdate,
    ) -> dict[str, Any]:
        await self._owned_collection(user_id, collection_id)
        db = get_db()
        update = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
        if "name" in update:
            update["name"] = str(update["name"]).strip()
        if update:
            update["updated_at"] = datetime.now(timezone.utc)
            await db[Collections.RESEARCH_COLLECTIONS].update_one(
                {"id": collection_id}, {"$set": update},
            )
            await audit_service.log(
                "research_collection_updated", user_id=user_id,
                payload={"collection_id": collection_id, "fields": list(update.keys())},
            )
        return await db[Collections.RESEARCH_COLLECTIONS].find_one(
            {"id": collection_id}, {"_id": 0},
        )

    async def delete(self, user_id: str, collection_id: str) -> dict[str, str]:
        await self._owned_collection(user_id, collection_id)
        db = get_db()
        # Promote direct children to root so no subtree is orphaned/hidden.
        await db[Collections.RESEARCH_COLLECTIONS].update_many(
            {"parent_id": collection_id, "user_id": user_id},
            {"$set": {"parent_id": None, "updated_at": datetime.now(timezone.utc)}},
        )
        await db[Collections.RESEARCH_COLLECTION_ITEMS].delete_many(
            {"collection_id": collection_id},
        )
        await db[Collections.RESEARCH_COLLECTIONS].delete_one({"id": collection_id})
        await audit_service.log(
            "research_collection_deleted", user_id=user_id,
            payload={"collection_id": collection_id},
        )
        return {"status": "ok"}

    # ── Items ──────────────────────────────────────────────────────
    async def add_records(
        self, user_id: str, collection_id: str, body: AddRecordsRequest,
    ) -> dict[str, Any]:
        await self._owned_collection(user_id, collection_id)
        db = get_db()
        created: list[dict[str, Any]] = []
        skipped: list[str] = []
        for rec in body.records:
            existing = await db[Collections.RESEARCH_COLLECTION_ITEMS].find_one(
                {"collection_id": collection_id, "article_id": rec.article_id},
                {"_id": 0, "id": 1},
            )
            if existing:
                skipped.append(rec.article_id)
                continue
            prov = await self._provenance(user_id, rec.article_id, rec.search_session_id)
            paper_id = await self._mirror_item_to_lithub(user_id, rec.article_id)
            item = ResearchCollectionItemDoc(
                collection_id=collection_id,
                user_id=user_id,
                article_id=rec.article_id,
                search_session_id=rec.search_session_id,
                title=rec.title,
                notes=rec.notes,
                paper_id=paper_id,
                source_provenance=prov["source_provenance"],
                permission_snapshot=prov["permission_snapshot"],
                summary_eligible=prov["summary_eligible"],
                ai_surfaced=prov["ai_surfaced"],
                ranker_advisory=prov["ranker_advisory"],
                evidence_packet_id=rec.evidence_packet_id,
            )
            doc = item.model_dump()
            for k in ("created_at", "updated_at"):
                doc[k] = doc[k].astimezone(timezone.utc)
            try:
                await db[Collections.RESEARCH_COLLECTION_ITEMS].insert_one(doc)
            except Exception:  # noqa: BLE001 — unique race → treat as skipped
                skipped.append(rec.article_id)
                continue
            created.append({"id": doc["id"], "article_id": rec.article_id, "paper_id": paper_id})
        await audit_service.log(
            "research_collection_records_added", user_id=user_id,
            payload={
                "collection_id": collection_id,
                "added_count": len(created),
                "skipped_count": len(skipped),
            },
        )
        return {
            "collection_id": collection_id,
            "added_count": len(created),
            "skipped_count": len(skipped),
            "created": created,
            "skipped_article_ids": skipped,
        }

    async def list_items(self, user_id: str, collection_id: str) -> list[dict[str, Any]]:
        await self._owned_collection(user_id, collection_id)
        db = get_db()
        rows = await db[Collections.RESEARCH_COLLECTION_ITEMS].find(
            {"collection_id": collection_id}, {"_id": 0},
        ).sort("created_at", -1).to_list(length=1000)
        article_ids = [r["article_id"] for r in rows]
        art_map: dict[str, dict] = {}
        if article_ids:
            async for row in db[Collections.ARTICLES].find(
                {"id": {"$in": article_ids}}, _ARTICLE_PROJECTION,
            ):
                art_map[row["id"]] = row
        if art_map:
            from services.article_email_redactor import redact_article_emails
            for _aid in list(art_map.keys()):
                art_map[_aid] = redact_article_emails(art_map[_aid])
        for r in rows:
            r["article"] = art_map.get(r["article_id"])
        return rows

    async def remove_record(
        self, user_id: str, collection_id: str, item_id: str,
    ) -> dict[str, str]:
        await self._owned_collection(user_id, collection_id)
        db = get_db()
        res = await db[Collections.RESEARCH_COLLECTION_ITEMS].delete_one(
            {"id": item_id, "collection_id": collection_id},
        )
        if res.deleted_count == 0:
            raise CollectionError("collection_item_not_found", status=404)
        await audit_service.log(
            "research_collection_record_removed", user_id=user_id,
            payload={"collection_id": collection_id, "item_id": item_id},
        )
        return {"status": "ok"}

    async def move_record(
        self, user_id: str, collection_id: str, item_id: str, target_collection_id: str,
    ) -> dict[str, str]:
        await self._owned_collection(user_id, collection_id)
        await self._owned_collection(user_id, target_collection_id)
        db = get_db()
        item = await db[Collections.RESEARCH_COLLECTION_ITEMS].find_one(
            {"id": item_id, "collection_id": collection_id}, {"_id": 0},
        )
        if not item:
            raise CollectionError("collection_item_not_found", status=404)
        # Respect the destination's uniqueness (collection_id, article_id).
        dup = await db[Collections.RESEARCH_COLLECTION_ITEMS].find_one(
            {"collection_id": target_collection_id, "article_id": item["article_id"]},
            {"_id": 0, "id": 1},
        )
        if dup:
            raise CollectionError("already_in_target_collection", status=409)
        await db[Collections.RESEARCH_COLLECTION_ITEMS].update_one(
            {"id": item_id},
            {"$set": {"collection_id": target_collection_id, "updated_at": datetime.now(timezone.utc)}},
        )
        await audit_service.log(
            "research_collection_record_moved", user_id=user_id,
            payload={"from": collection_id, "to": target_collection_id, "item_id": item_id},
        )
        return {"status": "ok"}

    # ── Provenance (B12) ───────────────────────────────────────────
    async def _provenance(
        self, user_id: str, article_id: str, search_session_id: Optional[str],
    ) -> dict[str, Any]:
        """Capture a provenance snapshot at save time: source provenance,
        permission snapshot (B1), summary eligibility (B16), and whether the
        article was AI-surfaced (+ its advisory label) for the session."""
        from services.downstream_eligibility import article_sources, can_use_article_for_tool
        from services.source_permission_resolver import resolve_for_sources

        out = {
            "source_provenance": [], "permission_snapshot": None,
            "summary_eligible": None, "ai_surfaced": False, "ranker_advisory": None,
        }
        try:
            db = get_db()
            art = await db[Collections.ARTICLES].find_one(
                {"id": article_id}, {"_id": 0, "sources_present": 1, "source_ids_json": 1},
            ) or {}
            sources = article_sources(art)
            out["source_provenance"] = sources
            out["permission_snapshot"] = resolve_for_sources(sources).model_dump()
            out["summary_eligible"] = can_use_article_for_tool(art, "summary")["allowed"]
            if search_session_id:
                surfaced = await db[Collections.AI_SURFACED_ARTICLES].find_one(
                    {"search_session_id": search_session_id, "article_id": article_id},
                    {"_id": 0, "advisory_label": 1},
                )
                if surfaced:
                    out["ai_surfaced"] = True
                    out["ranker_advisory"] = surfaced.get("advisory_label")
        except Exception:  # noqa: BLE001 — provenance is best-effort
            logger.exception("research_collection provenance failed (non-fatal)")
        return out

    # ── LitHub mirror ──────────────────────────────────────────────
    async def _mirror_item_to_lithub(self, user_id: str, article_id: str) -> Optional[str]:
        """Mirror a paper into the central LitHub library and return the canonical
        ``paper_id``. Best-effort: returns None (and enqueues the reliable outbox)
        on any failure. No-op (None) when dual-write is disabled or the user has
        no Identity ``sub`` or the article lacks a PMID/DOI."""
        s = get_settings()
        if not (s.LITPORTAL_DUAL_WRITE_LITHUB and s.LITHUB_BASE_URL):
            return None
        try:
            db = get_db()
            user = await db[Collections.USERS].find_one(
                {"id": user_id}, {"_id": 0, "identity_id": 1},
            )
            sub = (user or {}).get("identity_id")
            if not sub:
                return None
            art = await db[Collections.ARTICLES].find_one(
                {"id": article_id},
                {"_id": 0, "pmid": 1, "doi": 1, "doi_normalized": 1, "title": 1,
                 "journal": 1, "publication_year": 1, "abstract": 1, "authors_json": 1},
            )
            if not art:
                return None
            pmid = (art.get("pmid") or "").strip() or None
            doi = (art.get("doi") or art.get("doi_normalized") or "").strip() or None
            if not (pmid or doi):
                return None
            authors = [
                (a.get("fullName") or a.get("full_name") or a.get("name"))
                for a in (art.get("authors_json") or []) if isinstance(a, dict)
            ]
            item = {
                "pmid": pmid, "doi": doi,
                "title": art.get("title") or "Untitled paper",
                "journal": art.get("journal"),
                "abstract": art.get("abstract"),
                "year": art.get("publication_year"),
                "authors": [a for a in authors if a],
                "source": "litportal-collection",
                # Durable LitPortal-origin marker (LitHub keeps this fill-when-empty
                # on re-save, unlike `source` which is overwritten) so LitPulse can
                # reliably group these under "LitPortal Saved". See FE_OPTION_A spec.
                "portal_engine_record_id": f"litportal:{article_id}",
            }
            try:
                from clients.lithub_client import LitHubClient
                async with httpx.AsyncClient(timeout=_LITHUB_TIMEOUT_S) as client:
                    resp = await LitHubClient(http_client=client, settings=s).internal_save_paper(sub, item)
                pid = resp.get("paper_id") if isinstance(resp, dict) else None
                return str(pid) if pid else None
            except Exception:  # noqa: BLE001 — fall back to the reliable outbox
                from services import lithub_sync
                await lithub_sync.mirror_save(sub, item)
                return None
        except Exception:  # noqa: BLE001 — mirror is best-effort, never blocks the save
            logger.exception("research_collection lithub mirror failed (non-fatal)")
            return None


research_collection_service = ResearchCollectionService()
