"""Europe PMC REST search for the engine: cursor paging, the source's own hit
count, and the open-access / licence flags the access step uses."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from clinical_engine.retrieval.http import SourceError, request

SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PAGE_SIZE = 100


@dataclass
class SearchPage:
    count: int
    records: list[dict[str, Any]]
    next_cursor: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


async def search_page(client: httpx.AsyncClient, query: str, *, cursor: str = "*",
                      page_size: int = PAGE_SIZE, sort: str | None = None) -> SearchPage:
    params = {"query": query, "format": "json", "resultType": "core",
              "pageSize": str(page_size), "cursorMark": cursor}
    if sort:
        params["sort"] = sort  # e.g. "CITED desc": most-cited first
    resp = await request(client, "GET", SEARCH, host="europepmc", params=params)
    try:
        body = resp.json()
    except ValueError as exc:
        raise SourceError("invalid_response", "Europe PMC returned non-JSON") from exc
    records = list((body.get("resultList") or {}).get("result") or [])
    nxt = body.get("nextCursorMark")
    return SearchPage(
        count=int(body.get("hitCount") or 0), records=records,
        next_cursor=nxt if nxt and nxt != cursor and records else None,
    )


def record_extras(raw: dict[str, Any]) -> dict[str, Any]:
    """Access-relevant flags as Europe PMC reports them (Y/N → bool)."""
    def yn(key: str) -> Optional[bool]:
        v = raw.get(key)
        return None if v is None else str(v).upper() == "Y"

    return {
        "epmc_id": raw.get("id"), "epmc_source": (raw.get("source") or "").upper(),
        "is_open_access": yn("isOpenAccess"), "in_epmc": yn("inEPMC"), "in_pmc": yn("inPMC"),
        "has_pdf": yn("hasPDF"), "license": (raw.get("license") or "").strip().lower() or None,
        "first_publication_date": raw.get("firstPublicationDate"), "first_index_date": raw.get("firstIndexDate"),
        "full_text_urls": [
            {"url": u.get("url"), "availability": u.get("availability"), "code": u.get("availabilityCode"),
             "style": u.get("documentStyle"), "site": u.get("site")}
            for u in ((raw.get("fullTextUrlList") or {}).get("fullTextUrl") or []) if u.get("url")
        ],
        "mesh": [m.get("descriptorName") for m in ((raw.get("meshHeadingList") or {}).get("meshHeading") or [])
                 if m.get("descriptorName")],
    }


_TAG = re.compile(r"<[^>]+>")


def to_article(raw: dict[str, Any]) -> dict[str, Any]:
    """The LitPortal article shape (same normalizer the pipeline uses). Europe PMC
    abstracts carry HTML markup (<h4>, <i>, …); it is reduced to text."""
    from services.metadata_normalizer import metadata_normalizer

    article = metadata_normalizer.normalize(raw)
    for key in ("abstract", "title"):
        if article.get(key):
            article[key] = " ".join(html.unescape(_TAG.sub(" ", article[key])).split()) or None
    return article
