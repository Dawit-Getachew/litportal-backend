"""Verified full-text access (W2.5), progressive and after identity checks.

Four facts are kept apart for every paper:

  readable   a link the clinician can open (PubMed / PMC page, open-access copy,
             publisher page — which may need a subscription)
  acquired   text LitPortal actually fetched, with the route, the version
             (published / accepted manuscript), an identity check against the
             verified record and a completeness check
  permitted  what AI may process: the source policy (Cochrane and unknown
             sources fail closed) and, for full text, the licence
  used       what a summary actually consumed — set by the summary step

Routes (permitted only; no scraping, no paywall bypass, no stored credentials):
  1. Europe PMC fullTextXML (open-access subset; HTTP 500 = not available)
  2. PMC Cloud (NCBI's open-access dataset on AWS, replacing the retired OA
     web service): metadata gives PMID/DOI/title, licence, manuscript flag,
     retraction flag and the JATS file — the alternate route for the same paper
  3. Unpaywall (only when UNPAYWALL_EMAIL is set): open-access locations, their
     version and licence — recorded as readable links
A fetched text whose identifiers or title do not match the verified record is
rejected and that link is blocked for the paper. Access never changes ranking.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import httpx

from clinical_engine.identity.resolver import titles_compatible
from clinical_engine.retrieval.http import SourceError, request, transport_or_default

EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
PMC_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
UNPAYWALL = "https://api.unpaywall.org/v2/{doi}"
PER_PAPER_SECONDS = 25.0
_BLOCKLIST_FILE = Path(__file__).with_name("blocked_links.json")


def blocked_links() -> dict[str, list[dict[str, str]]]:
    """Known wrong links, keyed by PMID: {"<pmid>": [{"url": ..., "reason": ...}]}."""
    try:
        return json.loads(_BLOCKLIST_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _jats_ids(root: ET.Element) -> dict[str, Optional[str]]:
    meta = root.find(".//front/article-meta")
    ids: dict[str, Optional[str]] = {"pmid": None, "doi": None, "pmcid": None, "title": None}
    if meta is None:
        return ids
    for el in meta.findall("article-id"):
        kind = (el.attrib.get("pub-id-type") or "").lower()
        val = (el.text or "").strip()
        if kind in ("pmid", "doi") and val:
            ids[kind] = val
        elif kind in ("pmcid", "pmc") and val:
            ids["pmcid"] = val if val.upper().startswith("PMC") else f"PMC{val}"
    title = meta.find("title-group/article-title")
    ids["title"] = " ".join("".join(title.itertext()).split()) if title is not None else None
    return ids


def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if isinstance(el.tag, str) and el.tag.startswith("{"):
            el.tag = el.tag.split("}", 1)[1]
    return root


def completeness(root: ET.Element) -> dict[str, Any]:
    """Full text = a body with methods/results-like sections, not just front matter."""
    body = root.find(".//body")
    if body is None:
        return {"status": "abstract_only", "sections": 0, "words": 0, "has": []}
    titles = [" ".join("".join(t.itertext()).split()).lower() for t in body.iter("title")]
    words = len(" ".join(body.itertext()).split())
    has = sorted({k for k, cues in (("methods", ("method", "patients and", "study design", "materials")),
                                     ("results", ("result", "finding")), ("discussion", ("discussion",)),
                                     ("conclusion", ("conclusion",)))
                  if any(any(c in t for c in cues) for t in titles)})
    status = "complete" if words >= 1500 and ("results" in has or "discussion" in has) else \
        "partial" if words >= 300 else "abstract_only"
    return {"status": status, "sections": len(body.findall("sec")), "words": words, "has": has}


def identity_check(article: dict[str, Any], found: dict[str, Optional[str]]) -> dict[str, Any]:
    problems = []
    if found.get("pmid") and article.get("pmid") and str(found["pmid"]) != str(article["pmid"]):
        problems.append(f"PMID {found['pmid']} ≠ {article['pmid']}")
    if found.get("doi") and article.get("doi") and found["doi"].lower() != str(article["doi"]).lower():
        problems.append(f"DOI {found['doi']} ≠ {article['doi']}")
    if found.get("title") and article.get("title") and not titles_compatible(found["title"], article["title"]):
        problems.append("title does not match")
    return {"ok": not problems, "problems": problems,
            "checked": [k for k in ("pmid", "doi", "title") if found.get(k) and article.get(k)]}


def _licence_code(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower().replace(" ", "-")
    return s if s.startswith("cc") or s.startswith("public") else raw.strip().lower()


async def _pmc_cloud_meta(client: httpx.AsyncClient, pmcid: str) -> Optional[dict[str, Any]]:
    resp = await request(client, "GET", f"{PMC_BUCKET}/", host="pmc_cloud",
                         params={"list-type": "2", "prefix": f"metadata/{pmcid}."}, max_attempts=2)
    keys = sorted(re.findall(r"<Key>([^<]+)</Key>", resp.text),
                  key=lambda k: int(re.search(r"\.(\d+)\.json$", k).group(1)) if re.search(r"\.(\d+)\.json$", k) else 0)
    if not keys:
        return None
    meta = await request(client, "GET", f"{PMC_BUCKET}/{keys[-1]}", host="pmc_cloud", max_attempts=2)
    return meta.json()


async def _fetch_xml(client: httpx.AsyncClient, url: str, host: str) -> Optional[ET.Element]:
    try:
        resp = await request(client, "GET", url, host=host, max_attempts=2, max_retry_wait_s=2.0)
    except SourceError as exc:
        if exc.code in ("http_404", "http_500"):  # Europe PMC answers 500 for non-OA articles
            return None
        raise
    if "xml" not in (resp.headers.get("content-type") or "") and not resp.text.lstrip().startswith("<"):
        return None
    try:
        return _strip_ns(ET.fromstring(resp.text))
    except ET.ParseError:
        return None


async def _unpaywall(client: httpx.AsyncClient, doi: str, email: str) -> Optional[dict[str, Any]]:
    try:
        resp = await request(client, "GET", UNPAYWALL.format(doi=doi), host="unpaywall",
                             params={"email": email}, max_attempts=2)
    except SourceError as exc:
        if exc.code == "http_404":
            return None
        raise
    return resp.json()


def _permission(pub: dict[str, Any], licence: Optional[str], *, full_text: bool) -> dict[str, Any]:
    from services.downstream_eligibility import can_use_article_for_tool
    from services.oa_fetcher.license_gate import determine_ingest_permission

    sources = ["pubmed" if s == "pubmed" else "europepmc" for s in pub.get("sources") or []]
    source_verdict = can_use_article_for_tool({**(pub.get("article") or {}), "sources_present": sources}, "summary")
    if not source_verdict["allowed"]:
        return {"text_scope": "none", "licence": licence, "basis": source_verdict["reason"]}
    if full_text:
        gate = determine_ingest_permission(source="pmc_oa", license=licence, access_category="pmc_oa")
        if gate["can_ingest_for_summary"]:
            return {"text_scope": "full_text", "licence": gate["license"], "basis": gate["reason"]}
        return {"text_scope": "abstract", "licence": gate["license"],
                "basis": f"Full text not permitted ({gate['reason']}); the abstract is ({source_verdict['reason']})."}
    return {"text_scope": "abstract", "licence": licence, "basis": source_verdict["reason"]}


async def resolve(pub: dict[str, Any], client: httpx.AsyncClient, *,
                  unpaywall_email: Optional[str] = None) -> dict[str, Any]:
    art = pub.get("article") or {}
    epmc = pub.get("epmc") or {}
    pmid, pmcid, doi = art.get("pmid"), art.get("pmcid"), art.get("doi")
    attempts: list[dict[str, Any]] = []
    links: list[dict[str, str]] = []
    blocked_for_paper = blocked_links().get(str(pmid), [])
    blocked = {b["url"] for b in blocked_for_paper}
    if pmid:
        links.append({"kind": "abstract", "label": "PubMed record", "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"})
    acquired: dict[str, Any] = {"status": "abstract_only" if art.get("abstract") else "none", "route": None,
                                "version": None, "completeness": None, "identity": None}
    licence: Optional[str] = epmc.get("license")
    cloud: Optional[dict[str, Any]] = None

    async def attempt(route: str, fn: Any) -> Optional[ET.Element]:
        started = time.perf_counter()
        try:
            root = await fn()
            attempts.append({"route": route, "status": "ok" if root is not None else "not_available",
                             "ms": int((time.perf_counter() - started) * 1000)})
            return root
        except (SourceError, httpx.HTTPError, ValueError) as exc:
            attempts.append({"route": route, "status": "failed", "retryable": True,
                             "error": getattr(exc, "code", type(exc).__name__),
                             "ms": int((time.perf_counter() - started) * 1000)})
            return None

    def accept(route: str, root: ET.Element, version: Optional[str]) -> bool:
        found = _jats_ids(root)
        ident = identity_check(art, found)
        comp = completeness(root)
        if not ident["ok"]:
            attempts[-1].update(status="identity_mismatch", problems=ident["problems"])
            return False
        if comp["status"] == "abstract_only":
            attempts[-1].update(status="incomplete", completeness=comp)
            return False
        acquired.update(status="full_text", route=route, version=version, completeness=comp, identity=ident)
        return True

    if pmcid:
        epmc_url = EPMC_FULLTEXT.format(pmcid=pmcid)
        if epmc_url not in blocked and (epmc.get("in_epmc") or epmc.get("is_open_access") or not epmc):
            root = await attempt("europepmc", lambda: _fetch_xml(client, epmc_url, "europepmc"))
            if root is not None:
                lic = root.find(".//license")
                href = None
                if lic is not None:
                    href = lic.attrib.get("{http://www.w3.org/1999/xlink}href") or lic.attrib.get("href")
                licence = licence or _licence_code(href)
                accept("europepmc", root, "published (PMC open access)")
        started = time.perf_counter()
        try:
            cloud = await _pmc_cloud_meta(client, pmcid)
            attempts.append({"route": "pmc_cloud_metadata", "status": "ok" if cloud else "not_available",
                             "ms": int((time.perf_counter() - started) * 1000)})
        except (SourceError, httpx.HTTPError, ValueError) as exc:
            attempts.append({"route": "pmc_cloud_metadata", "status": "failed", "retryable": True,
                             "error": getattr(exc, "code", type(exc).__name__)})
        if cloud:
            licence = licence or _licence_code(cloud.get("license_code"))
            meta_ident = identity_check(art, {"pmid": str(cloud.get("pmid") or "") or None, "doi": cloud.get("doi"),
                                              "title": cloud.get("title")})
            version = "accepted manuscript (PMC author manuscript)" if cloud.get("is_manuscript") \
                else "published (PMC open access)"
            if acquired["status"] == "full_text":
                acquired["version"] = version
            if cloud.get("is_retracted"):
                acquired["retracted_in_pmc"] = True
            if not meta_ident["ok"]:
                attempts[-1].update(status="identity_mismatch", problems=meta_ident["problems"])
            elif acquired["status"] != "full_text" and cloud.get("is_pmc_openaccess") and cloud.get("xml_url"):
                url = cloud["xml_url"].split("?")[0].replace("s3://pmc-oa-opendata", PMC_BUCKET)
                if url not in blocked:
                    root = await attempt("pmc_cloud", lambda: _fetch_xml(client, url, "pmc_cloud"))
                    if root is not None:
                        accept("pmc_cloud", root, version)
        if cloud and cloud.get("is_pmc_openaccess") or epmc.get("in_pmc"):
            links.append({"kind": "free_full_text", "label": "Free full text (PMC)",
                          "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"})
    unpaywall_state = "not_configured"
    best_oa: Optional[dict[str, Any]] = None
    if doi and unpaywall_email:
        started = time.perf_counter()
        try:
            data = await _unpaywall(client, doi, unpaywall_email)
            unpaywall_state = "checked"
            best_oa = (data or {}).get("best_oa_location")
            attempts.append({"route": "unpaywall", "status": "ok" if best_oa else "not_available",
                             "ms": int((time.perf_counter() - started) * 1000)})
            if best_oa and best_oa.get("url") and best_oa["url"] not in blocked:
                links.append({"kind": "open_access_copy",
                              "label": f"Open-access copy ({best_oa.get('host_type') or 'host'}, "
                                       f"{(best_oa.get('version') or 'version unknown').replace('Version', ' version')})",
                              "url": best_oa["url"]})
                licence = licence or best_oa.get("license")
        except (SourceError, httpx.HTTPError, ValueError) as exc:
            unpaywall_state = "failed"
            attempts.append({"route": "unpaywall", "status": "failed", "retryable": True,
                             "error": getattr(exc, "code", type(exc).__name__)})
    if doi and f"https://doi.org/{doi}" not in blocked:
        links.append({"kind": "publisher", "label": "Publisher page (may need a subscription)",
                      "url": f"https://doi.org/{doi}"})
    oa = any(link["kind"] in ("free_full_text", "open_access_copy") for link in links)
    readable = ("free_full_text" if oa else "publisher_link" if doi else "abstract_only" if pmid else "unknown")
    return {
        "readable": {"status": readable, "links": links, "unpaywall": unpaywall_state,
                     "blocked_links": blocked_for_paper,
                     "oa_status": "open" if oa else ("closed" if unpaywall_state == "checked" else None)},
        "acquired": acquired,
        "permitted": _permission(pub, licence, full_text=acquired["status"] == "full_text"),
        "used": {"status": "not_used", "note": "Set when a summary consumes this paper."},
        "attempts": attempts,
        "checked_at": time.time(),
    }


async def resolve_many(pubs: list[dict[str, Any]], *, stage_seconds: float = 60.0, concurrency: int = 6,
                       transport: Optional[httpx.AsyncBaseTransport] = None,
                       on_result: Any = None, call: Any = None) -> dict[str, Any]:
    """Resolve access for *pubs* (identity-checked only) within a stage deadline;
    papers not reached in time are reported as such — never left pending."""
    email = os.environ.get("UNPAYWALL_EMAIL", "").strip() or None
    sem = asyncio.Semaphore(concurrency)
    done: dict[str, dict[str, Any]] = {}

    async def one(pub: dict[str, Any]) -> None:
        if pub.get("identity_status") == "quarantined":
            return
        async with sem:
            try:
                job = (call(resolve, pub, client, unpaywall_email=email) if call is not None
                       else resolve(pub, client, unpaywall_email=email))
                result = await asyncio.wait_for(job, timeout=PER_PAPER_SECONDS)
            except asyncio.TimeoutError:
                result = {"acquired": {"status": "not_checked", "reason": "timed out — retry later"},
                          "readable": {"status": "unknown", "links": []},
                          "permitted": _permission(pub, None, full_text=False),
                          "used": {"status": "not_used"}, "attempts": [{"route": "all", "status": "timeout"}]}
            done[pub["id"]] = result
            if on_result:
                await on_result(pub, result)

    async with httpx.AsyncClient(timeout=15.0, transport=transport_or_default(transport),
                                 follow_redirects=True) as client:
        try:
            await asyncio.wait_for(asyncio.gather(*(one(p) for p in pubs)), timeout=stage_seconds)
        except asyncio.TimeoutError:
            pass
    pending = [p for p in pubs if p["id"] not in done and p.get("identity_status") != "quarantined"]
    for p in pending:
        result = {"acquired": {"status": "not_checked", "reason": "access stage time limit — retry later"},
                  "readable": {"status": "unknown", "links": []}, "permitted": _permission(p, None, full_text=False),
                  "used": {"status": "not_used"}, "attempts": []}
        done[p["id"]] = result
        if on_result:
            await on_result(p, result)
    return {"checked": len(done), "full_text": sum(1 for r in done.values() if r["acquired"].get("status") == "full_text"),
            "not_checked": len(pending), "unpaywall": "configured" if email else "not_configured"}


__all__ = ["resolve", "resolve_many", "identity_check", "completeness", "blocked_links"]
