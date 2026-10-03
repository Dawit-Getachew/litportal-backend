"""Write the engine's evidence list into the existing session records, so every
existing action (selection, summaries, open-access fetch, saving) keeps working
on engine searches.

The shared ``articles`` collection is written identity-safely — unlike the
legacy upsert, which merges on ANY shared identifier (a record found only by a
DOI it borrowed could be overwritten with another paper's fields):

* a record is matched by PMID first; by DOI or PMCID only when the stored
  record has no PMID or the same PMID;
* if another stored record already holds this paper's DOI or PMCID (a legacy
  identity error), that identifier is not copied and the collision is noted —
  nothing is merged and the other record is left untouched.

Session rows carry only the fields the existing API already returns; the
engine's own detail (state, reasons, lineage, access) stays in
``engine_publications`` and is served by the engine API.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from clinical_engine import store
from services import engine_concept_plan as ecp

LEGACY_SOURCE = {"pubmed": "pubmed", "europepmc": "europe_pmc"}
# A legacy relevance number for consumers that read one; the engine's state is
# the actual judgment (shown by the engine API).
_STATE_SCORE = {"direct": 90.0, "broad_answering": 80.0, "supportive": 60.0, "uncertain": 50.0,
                "indirect": 40.0, "excluded": 0.0}


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def create_session_shell(run: dict[str, Any], version: dict[str, Any], plan: Optional[dict[str, Any]]) -> str:
    """The session the results page opens while the engine searches."""
    from core.db import Collections, get_db
    from models.sessions import SearchSessionDoc

    cqm = version["map"]
    picott: dict[str, Any] = {}
    if plan:
        validated = ecp.validate_plan(plan)
        if validated:
            ecp.apply_to_picott(picott, {}, validated)
    doc = SearchSessionDoc(
        user_id=run["user_id"], original_question=cqm.get("question_text"),
        sanitized_question=cqm.get("question_text"), topic_label=(cqm.get("summary") or "")[:120] or None,
        clinical_question_type=(plan or {}).get("legacy_type_of_question"), status="searching",
    ).model_dump()
    doc.update({"picott_json": picott, "engine_concept_plan": plan, "engine_run_id": run["id"],
                "retrieval_mode": "engine", "sources_searched": ["pubmed", "europe_pmc"]})
    await get_db()[Collections.SEARCH_SESSIONS].insert_one(doc)
    return doc["id"]


async def set_session_status(session_id: str, status: str, **extra: Any) -> None:
    from core.db import Collections, get_db

    await get_db()[Collections.SEARCH_SESSIONS].update_one(
        {"id": session_id}, {"$set": {"status": status, "updated_at": _now(), **extra}})


async def upsert_article(article: dict[str, Any]) -> tuple[str, list[str]]:
    """(article id, notes). Identity-safe; see the module docstring."""
    from pymongo.errors import DuplicateKeyError

    from core.db import Collections, get_db
    from services.article_email_redactor import redact_article_emails

    db = get_db()[Collections.ARTICLES]
    art = redact_article_emails(dict(article))
    art.pop("_retrieved_by", None)
    pmid, doi_n, pmcid = art.get("pmid"), art.get("doi_normalized"), art.get("pmcid")
    notes: list[str] = []
    existing = None
    if pmid:
        existing = await db.find_one({"pmid": pmid}, {"_id": 0, "id": 1, "pmid": 1})
    for field, value in (("doi_normalized", doi_n), ("pmcid", pmcid)):
        if existing or not value:
            continue
        holder = await db.find_one({field: value}, {"_id": 0, "id": 1, "pmid": 1})
        if holder and (not holder.get("pmid") or holder.get("pmid") == pmid):
            existing = holder
    if existing is None:
        existing = await db.find_one({"canonical_key": art.get("canonical_key")}, {"_id": 0, "id": 1, "pmid": 1})
        if existing and existing.get("pmid") and pmid and existing["pmid"] != pmid:
            existing = None

    fields = {k: v for k, v in art.items() if k not in ("id", "created_at", "canonical_key") and v not in (None, [], {}, "")}
    fields["updated_at"] = _now()
    # Never take an identifier another stored record already holds.
    for field, value in (("doi_normalized", doi_n), ("pmcid", pmcid)):
        if not value:
            continue
        holder = await db.find_one({field: value}, {"_id": 0, "id": 1})
        if holder and (existing is None or holder["id"] != existing["id"]):
            fields.pop(field, None)
            if field == "doi_normalized":
                fields.pop("doi", None)
            notes.append(f"{field}_held_by_another_record")
    if existing:
        try:
            await db.update_one({"id": existing["id"]}, {"$set": fields})
        except DuplicateKeyError:
            notes.append("identifier_collision_on_update")
            safe = {k: v for k, v in fields.items() if k not in ("pmid", "doi", "doi_normalized", "pmcid")}
            await db.update_one({"id": existing["id"]}, {"$set": safe})
        return existing["id"], notes
    doc = {**fields, "id": art.get("id") or uuid.uuid4().hex, "canonical_key": art.get("canonical_key"),
           "created_at": _now()}
    try:
        await db.insert_one(doc)
    except DuplicateKeyError:
        notes.append("identifier_collision_on_insert")
        race = await db.find_one({"canonical_key": doc["canonical_key"]}, {"_id": 0, "id": 1})
        if race:
            return race["id"], notes
        doc["canonical_key"] = f"engine:{doc['id']}"
        for field in ("pmid", "doi_normalized", "pmcid"):
            doc.pop(field, None)
        await db.insert_one(doc)
    return doc["id"], notes


async def project(session_id: str, pubs: list[dict[str, Any]], sections: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Write session rows for every listed publication (main, other, more,
    excluded); not-assessed and held-back records stay engine-side."""
    from core.db import Collections, get_db

    db = get_db()
    await db[Collections.SEARCH_SESSION_ARTICLES].delete_many({"search_session_id": session_id})
    order = [(name, p) for name in ("main", "other", "more", "excluded") for p in sections.get(name, [])]
    rows = []
    collisions = 0
    for position, (name, p) in enumerate(order, start=1):
        adm = p.get("admission") or {}
        article_id, notes = await upsert_article(p["article"])
        collisions += bool(notes)
        row = {
            "id": uuid.uuid4().hex, "search_session_id": session_id, "article_id": article_id,
            "relevance_score": _STATE_SCORE.get(adm.get("state"), 50.0), "evidence_design_score": 0.0,
            "overall_rank": max(1, 10_000 - position),
            "rank_explanation": adm.get("reason"),
            "included_by_default": False, "user_included": False,
            "exclusion_reason": (f"engine:{adm.get('exclusion') or 'excluded'}" if name == "excluded" else None),
            "query_types_matched": sorted({f"engine_{h['family']}" for h in p.get("hits") or []}),
            "query_family_matches": sorted({h["family"] for h in p.get("hits") or []}),
            "sources_present": sorted({LEGACY_SOURCE.get(s, s) for s in p.get("sources") or []}),
            "flags": [], "created_at": _now(), "updated_at": _now(),
        }
        rows.append(row)
        await store.update_publication(p["id"], {
            "section": name, "position": position, "legacy_article_id": article_id, "legacy_row_id": row["id"],
            "projection_notes": notes,
        })
    if rows:
        await db[Collections.SEARCH_SESSION_ARTICLES].insert_many(rows, ordered=False)
    for name in ("held_back", "not_assessed"):
        for p in sections.get(name, []):
            await store.update_publication(p["id"], {"section": name, "position": None})
    return {"rows": len(rows), "identifier_collisions": collisions}
