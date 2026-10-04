"""Citation counts for PubMed records, used ONLY to choose which records the
"most-cited" lane reads (landmark papers that relevance ranking buries) — never to
rank the evidence list or judge relevance.

Source: NIH iCite (public; no key). GET /api/pubs?pmids=…&fl=… — batches of 200
PMIDs (1,000 makes the URL too long: HTTP 413).
"""
from __future__ import annotations

import asyncio

import httpx

from clinical_engine.retrieval.http import SourceError, request

ICITE = "https://icite.od.nih.gov/api/pubs"
BATCH = 200


async def citation_counts(client: httpx.AsyncClient, pmids: list[str]) -> dict[str, int]:
    """{pmid: citation count}; raises SourceError when iCite cannot be reached."""
    out: dict[str, int] = {}

    async def one(chunk: list[str]) -> None:
        resp = await request(client, "GET", ICITE, host="icite", max_attempts=3,
                             params={"pmids": ",".join(chunk), "fl": "pmid,citation_count"})
        try:
            data = resp.json().get("data") or []
        except ValueError as exc:
            raise SourceError("invalid_response", "iCite returned non-JSON") from exc
        for d in data:
            if d.get("pmid") is not None:
                out[str(d["pmid"])] = int(d.get("citation_count") or 0)

    await asyncio.gather(*(one(pmids[i:i + BATCH]) for i in range(0, len(pmids), BATCH)))
    return out
