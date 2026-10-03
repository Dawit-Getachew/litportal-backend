"""PubMed (NCBI E-utilities) for the engine: paged ESearch in Best Match order
with PubMed's own translation of the query, and one EFetch (XML) call per
batch of records — every field is read from the record itself, never from the
papers it cites (PubmedData/ReferenceList)."""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from clinical_engine.retrieval.http import SourceError, ncbi_host, ncbi_params, request

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PAGE_SIZE = 100
FETCH_BATCH = 200
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_NCT = re.compile(r"\bNCT\d{8}\b", re.I)


@dataclass
class SearchPage:
    count: int
    ids: list[str]
    translation: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


async def esearch(client: httpx.AsyncClient, term: str, *, retstart: int = 0, retmax: int = PAGE_SIZE,
                  sort: str = "relevance", extra: Optional[dict[str, str]] = None) -> SearchPage:
    params = ncbi_params(db="pubmed", term=term, retmode="json", retstart=str(retstart), retmax=str(retmax),
                         sort=sort, **(extra or {}))
    if len(term) > 1500:  # long boolean queries go in the body
        resp = await request(client, "POST", ESEARCH, host=ncbi_host(), data=params)
    else:
        resp = await request(client, "GET", ESEARCH, host=ncbi_host(), params=params)
    try:
        # NCBI error strings can contain raw newlines (not valid strict JSON).
        body = json.loads(resp.text, strict=False).get("esearchresult") or {}
    except ValueError as exc:
        raise SourceError("invalid_response", "esearch returned non-JSON") from exc
    if "ERROR" in body:
        raise SourceError("invalid_response", str(body.get("ERROR"))[:200])
    warn = body.get("warninglist") or {}
    err = body.get("errorlist") or {}
    return SearchPage(
        count=int(body.get("count") or 0),
        ids=[str(i) for i in body.get("idlist") or []],
        translation=body.get("querytranslation") or None,
        warnings=[f"{k}: {v}" for k, vals in warn.items() for v in (vals or []) if v],
        errors=[f"{k}: {v}" for k, vals in err.items() for v in (vals or []) if v],
    )


async def efetch(client: httpx.AsyncClient, pmids: list[str]) -> list[dict[str, Any]]:
    """Full records for *pmids* (POST, batched), in the order given."""
    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(pmids), FETCH_BATCH):
        batch = pmids[i:i + FETCH_BATCH]
        resp = await request(client, "POST", EFETCH, host=ncbi_host(),
                             data=ncbi_params(db="pubmed", id=",".join(batch), rettype="abstract", retmode="xml"))
        for rec in parse_efetch(resp.text):
            out[rec["pmid"]] = rec
    return [out[p] for p in pmids if p in out]


def _text(el: Optional[ET.Element]) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _date(el: Optional[ET.Element]) -> dict[str, Any]:
    """{raw, year, month, day} from a PubMed date element (Year/Month/Day or MedlineDate)."""
    if el is None:
        return {}
    year = el.findtext("Year")
    month = el.findtext("Month")
    day = el.findtext("Day")
    medline = el.findtext("MedlineDate")
    if not year and medline:
        m = re.match(r"(\d{4})(?:\s+([A-Za-z]{3}))?", medline)
        if m:
            year, month = m.group(1), m.group(2)
    if not year:
        return {}
    mnum: Optional[int] = None
    if month:
        mnum = int(month) if month.isdigit() else _MONTHS.get(month[:3].lower())
    dnum = int(day) if day and day.isdigit() else None
    parts = [year] + ([f"{mnum:02d}"] if mnum else []) + ([f"{dnum:02d}"] if mnum and dnum else [])
    return {"raw": "-".join(parts), "year": int(year), "month": mnum, "day": dnum}


def parse_efetch(xml_text: str) -> list[dict[str, Any]]:
    """Engine records from EFetch XML. Identifiers come only from the article's
    own PubmedData/ArticleIdList (+ ELocationID DOI); never from ReferenceList."""
    try:
        root = ET.fromstring(xml_text or "")
    except ET.ParseError as exc:
        raise SourceError("invalid_response", "efetch XML did not parse") from exc
    out: list[dict[str, Any]] = []
    for art in root.iter("PubmedArticle"):
        mc = art.find("MedlineCitation")
        pmid = (mc.findtext("PMID") or "").strip() if mc is not None else ""
        if not pmid:
            continue
        a = mc.find("Article")
        own_ids = {(e.attrib.get("IdType") or "").lower(): (e.text or "").strip()
                   for e in art.findall("PubmedData/ArticleIdList/ArticleId") if (e.text or "").strip()}
        doi = own_ids.get("doi") or next(
            ((e.text or "").strip() for e in (a.findall("ELocationID") if a is not None else [])
             if (e.attrib.get("EIdType") or "").lower() == "doi" and (e.text or "").strip()), "")
        pmcid = own_ids.get("pmc", "")
        if pmcid and not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        abstract_parts = []
        for ab in (a.findall("Abstract/AbstractText") if a is not None else []):
            txt = _text(ab)
            if txt:
                label = ab.attrib.get("Label")
                abstract_parts.append(f"{label}: {txt}" if label else txt)
        journal = a.find("Journal") if a is not None else None
        pub_date = _date(journal.find("JournalIssue/PubDate")) if journal is not None else {}
        article_date = _date(a.find("ArticleDate")) if a is not None else {}
        history = {(d.attrib.get("PubStatus") or ""): _date(d)
                   for d in art.findall("PubmedData/History/PubMedPubDate")}
        authors = []
        for au in (a.findall("AuthorList/Author") if a is not None else []):
            if au.findtext("CollectiveName"):
                authors.append({"name": _text(au.find("CollectiveName")), "collective": True})
            elif au.findtext("LastName"):
                authors.append({"name": f"{au.findtext('LastName')} {au.findtext('Initials') or ''}".strip()})
        nct = set()
        for acc in mc.findall("Article/DataBankList/DataBank/AccessionNumberList/AccessionNumber"):
            nct.update(m.upper() for m in _NCT.findall(acc.text or ""))
        for sid in mc.findall("Article/SecondaryIdList/SecondaryId") + mc.findall("OtherID"):
            nct.update(m.upper() for m in _NCT.findall(sid.text or ""))
        links = [{"type": cc.attrib.get("RefType"), "pmid": (cc.findtext("PMID") or "").strip(),
                  "note": _text(cc.find("RefSource"))[:200]}
                 for cc in mc.findall("CommentsCorrectionsList/CommentsCorrections")]
        out.append({
            "pmid": pmid,
            "title": _text(a.find("ArticleTitle")) if a is not None else "",
            "abstract": "\n".join(abstract_parts) or None,
            "journal": _text(journal.find("Title")) if journal is not None else "",
            "journal_iso": _text(journal.find("ISOAbbreviation")) if journal is not None else "",
            "pub_date": pub_date,
            "article_date": article_date,
            "entrez_date": history.get("entrez") or history.get("pubmed") or {},
            "authors": authors,
            "publication_types": [_text(p) for p in (a.findall("PublicationTypeList/PublicationType")
                                                    if a is not None else []) if _text(p)],
            "mesh": [_text(m.find("DescriptorName")) for m in mc.findall("MeshHeadingList/MeshHeading")
                     if m.find("DescriptorName") is not None],
            "keywords": [_text(k) for k in mc.findall("KeywordList/Keyword") if _text(k)][:20],
            "language": (a.findtext("Language") or "").strip() if a is not None else "",
            "doi": doi,
            "pmcid": pmcid,
            "nct": sorted(nct),
            "links": [x for x in links if x["pmid"] and x["type"]],
        })
    return out


_MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def to_article(rec: dict[str, Any]) -> dict[str, Any]:
    """The LitPortal article shape (same normalizer as the existing connector)."""
    from services.connectors.pubmed import pubmed_connector

    d = rec.get("pub_date") or rec.get("article_date") or {}
    pubdate = " ".join(str(x) for x in (d.get("year"), _MON[d["month"] - 1] if d.get("month") else None,
                                       d.get("day")) if x)
    raw = {
        "pmid": rec["pmid"], "title": rec.get("title") or "", "pubdate": pubdate,
        "authors": [{"name": a["name"]} for a in rec.get("authors") or []],
        "fulljournalname": rec.get("journal") or "", "source": rec.get("journal_iso") or "",
        "articleids": [], "pubtype": [],
        "abstractText_xml": rec.get("abstract"), "pubTypeList_xml": rec.get("publication_types") or [],
        "doi_xml": rec.get("doi") or None, "pmcid_xml": rec.get("pmcid") or None,
        "language": rec.get("language") or None,
    }
    article = pubmed_connector._normalize(raw)  # noqa: SLF001 — shared, tested normalizer
    if article is None:
        raise SourceError("invalid_response", "record without PMID")
    article["publication_date"] = (rec.get("pub_date") or {}).get("raw") or article.get("publication_date")
    return article
