"""A fake PubMed / Europe PMC / PMC Cloud for engine retrieval tests (no network).

Records are defined once; every source answers from them:
  * PubMed ESearch: ids of records whose text contains EVERY quoted phrase group
    of the query (an AND of OR-groups, roughly like PubMed), Best-Match order =
    the order records are defined; count, querytranslation, paging.
  * PubMed EFetch: XML for the requested ids (own ids + a reference list whose
    DOIs must never leak onto the record).
  * Europe PMC search: the same matching, core JSON, cursor paging.
  * Europe PMC fullTextXML / PMC Cloud: JATS for open-access records.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import parse_qs

import httpx


@dataclass
class Rec:
    pmid: str
    title: str
    abstract: str
    year: int = 2020
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    types: list[str] = field(default_factory=lambda: ["Randomized Controlled Trial"])
    journal: str = "N Engl J Med"
    nct: Optional[str] = None
    mesh: list[str] = field(default_factory=list)
    epmc_doi: Optional[str] = None      # Europe PMC's DOI (to create conflicts)
    oa_licence: Optional[str] = None    # open access in PMC with this licence
    jats_title: Optional[str] = None    # full text with a different title (wrong article)
    entrez: str = "2020/05/01"
    cites: int = 0                      # citation count (iCite / Europe PMC)


def _groups(query: str) -> list[list[str]]:
    """OR-groups of quoted phrases/MeSH names joined by AND at the top level (approximate)."""
    core = query.split(" NOT ")[0]
    out = []
    for grp in re.findall(r"\(([^()]*)\)", core):
        terms = [t.lower() for t in re.findall(r'"([^"]+)"', grp)]
        if terms:
            out.append(terms)
    return out


def _match(rec: Rec, query: str) -> bool:
    text = f"{rec.title} {rec.abstract} {' '.join(rec.mesh)} {' '.join(rec.types)}".lower()
    groups = _groups(query)
    return bool(groups) and all(any(t in text for t in g) for g in groups)


class FakeSources:
    def __init__(self, records: list[Rec], *, fail: set[str] = frozenset()) -> None:
        self.records = records
        self.fail = set(fail)  # "pubmed" | "europepmc" | "fulltext" | "icite"
        self.calls: list[str] = []

    def by_pmid(self, pmid: str) -> Optional[Rec]:
        return next((r for r in self.records if r.pmid == pmid), None)

    def efetch_xml(self, pmids: list[str]) -> str:
        arts = []
        for pmid in pmids:
            r = self.by_pmid(pmid)
            if not r:
                continue
            ids = f'<ArticleId IdType="pubmed">{r.pmid}</ArticleId>'
            ids += f'<ArticleId IdType="doi">{r.doi}</ArticleId>' if r.doi else ""
            ids += f'<ArticleId IdType="pmc">{r.pmcid}</ArticleId>' if r.pmcid else ""
            types = "".join(f"<PublicationType>{t}</PublicationType>" for t in r.types)
            mesh = "".join(f"<MeshHeading><DescriptorName>{m}</DescriptorName></MeshHeading>" for m in r.mesh)
            bank = (f"<DataBankList><DataBank><DataBankName>ClinicalTrials.gov</DataBankName><AccessionNumberList>"
                    f"<AccessionNumber>{r.nct}</AccessionNumber></AccessionNumberList></DataBank></DataBankList>"
                    if r.nct else "")
            y, m, d = r.entrez.split("/")
            arts.append(
                f"<PubmedArticle><MedlineCitation><PMID>{r.pmid}</PMID><Article>"
                f"<Journal><Title>{r.journal}</Title><ISOAbbreviation>{r.journal}</ISOAbbreviation>"
                f"<JournalIssue><PubDate><Year>{r.year}</Year><Month>Mar</Month></PubDate></JournalIssue></Journal>"
                f"<ArticleTitle>{r.title}</ArticleTitle><Abstract><AbstractText>{r.abstract}</AbstractText></Abstract>"
                f"<AuthorList><Author><LastName>Smith</LastName><Initials>J</Initials></Author></AuthorList>"
                f"<Language>eng</Language>{bank}<PublicationTypeList>{types}</PublicationTypeList></Article>"
                f"<MeshHeadingList>{mesh}</MeshHeadingList></MedlineCitation>"
                f"<PubmedData><History><PubMedPubDate PubStatus=\"entrez\"><Year>{y}</Year><Month>{m}</Month>"
                f"<Day>{d}</Day></PubMedPubDate></History><ArticleIdList>{ids}</ArticleIdList>"
                f"<ReferenceList><Reference><Citation>Cited</Citation><ArticleIdList>"
                f"<ArticleId IdType=\"doi\">10.9999/cited.reference</ArticleId>"
                f"<ArticleId IdType=\"pmc\">PMC9999999</ArticleId></ArticleIdList></Reference></ReferenceList>"
                f"</PubmedData></PubmedArticle>")
        return "<?xml version=\"1.0\"?><PubmedArticleSet>" + "".join(arts) + "</PubmedArticleSet>"

    def jats(self, r: Rec) -> str:
        body = "".join(f"<sec><title>{t}</title><p>{' '.join(['word'] * 450)}</p></sec>"
                       for t in ("Introduction", "Methods", "Results", "Discussion"))
        lic = (f'<permissions><license xmlns:xlink="http://www.w3.org/1999/xlink" '
               f'xlink:href="https://creativecommons.org/licenses/{r.oa_licence}/4.0/"/></permissions>'
               if r.oa_licence else "")
        return (f"<article><front><article-meta><article-id pub-id-type=\"pmid\">{r.pmid}</article-id>"
                f"<article-id pub-id-type=\"doi\">{r.doi or ''}</article-id>"
                f"<title-group><article-title>{r.jats_title or r.title}</article-title></title-group>{lic}"
                f"</article-meta></front><body>{body}</body></article>")

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append(url)
        if "esearch.fcgi" in url:
            if "pubmed" in self.fail:
                return httpx.Response(500, text="boom")
            params = dict(request.url.params)
            if request.method == "POST":
                params.update({k: v[0] for k, v in parse_qs(request.content.decode()).items()})
            term = params.get("term", "")
            uid = re.search(r"\s+AND\s+(\d+)\[uid\]\s*$", term)
            if uid:  # "(<query>) AND <pmid>[uid]": does the query match this record?
                inner = term[:uid.start()]
                rec = self.by_pmid(uid.group(1))
                ok = rec is not None and (inner == "(all[sb])" or _match(rec, inner))
                return httpx.Response(200, json={"esearchresult": {"count": "1" if ok else "0", "idlist": []}})
            hits = [r.pmid for r in self.records if _match(r, term)]
            start, size = int(params.get("retstart", 0)), int(params.get("retmax", 20))
            return httpx.Response(200, json={"esearchresult": {
                "count": str(len(hits)), "idlist": hits[start:start + size], "querytranslation": f"T({term[:40]})"}})
        if "icite.od.nih.gov/api/pubs" in url:
            if "icite" in self.fail:
                return httpx.Response(503, text="down")
            pmids = (request.url.params.get("pmids") or "").split(",")
            return httpx.Response(200, json={"data": [{"pmid": int(r.pmid), "citation_count": r.cites}
                                                      for p in pmids if (r := self.by_pmid(p))]})
        if "efetch.fcgi" in url:
            ids = parse_qs(request.content.decode()).get("id", [""])[0].split(",")
            return httpx.Response(200, text=self.efetch_xml(ids))
        if "europepmc/webservices/rest/search" in url:
            if "europepmc" in self.fail:
                return httpx.Response(503, text="down")
            q = request.url.params.get("query", "")
            hits = [r for r in self.records if _match(r, q.replace("TITLE_ABS:", ""))]
            if (request.url.params.get("sort") or "").upper().startswith("CITED"):
                hits = sorted(hits, key=lambda r: -r.cites)
            cursor = request.url.params.get("cursorMark", "*")
            start = 0 if cursor == "*" else int(cursor)
            size = int(request.url.params.get("pageSize", 25))
            page = hits[start:start + size]
            results = [{
                "id": r.pmid, "source": "MED", "pmid": r.pmid, "doi": r.epmc_doi or r.doi, "pmcid": r.pmcid,
                "title": r.title, "abstractText": f"<p>{r.abstract}</p>", "pubYear": str(r.year),
                "journalInfo": {"journal": {"title": r.journal}}, "pubTypeList": {"pubType": r.types},
                "isOpenAccess": "Y" if r.oa_licence else "N", "inEPMC": "Y" if r.oa_licence else "N",
                "inPMC": "Y" if r.pmcid else "N", "license": r.oa_licence,
                "firstIndexDate": r.entrez.replace("/", "-"),
            } for r in page]
            nxt = str(start + size) if start + size < len(hits) else None
            return httpx.Response(200, json={"hitCount": len(hits), "nextCursorMark": nxt,
                                             "resultList": {"result": results}})
        m = re.search(r"/rest/(PMC\d+)/fullTextXML", url)
        if m:
            r = next((x for x in self.records if x.pmcid == m.group(1)), None)
            if "fulltext" in self.fail or not r or not r.oa_licence:
                return httpx.Response(500, json={"errCode": 404})
            return httpx.Response(200, text=self.jats(r), headers={"content-type": "application/xml"})
        if "pmc-oa-opendata" in url:
            prefix = request.url.params.get("prefix")
            if prefix:
                pmcid = prefix.split("/")[1].rstrip(".")
                r = next((x for x in self.records if x.pmcid == pmcid), None)
                body = f"<ListBucketResult><Key>metadata/{pmcid}.1.json</Key></ListBucketResult>" if r and r.oa_licence \
                    else "<ListBucketResult></ListBucketResult>"
                return httpx.Response(200, text=body)
            m = re.search(r"metadata/(PMC\d+)\.1\.json", url)
            if m:
                r = next(x for x in self.records if x.pmcid == m.group(1))
                return httpx.Response(200, json={"pmcid": r.pmcid, "pmid": int(r.pmid), "doi": r.doi, "title": r.title,
                                                 "is_pmc_openaccess": True, "is_manuscript": False,
                                                 "is_retracted": False, "license_code": f"CC {r.oa_licence.upper()}",
                                                 "xml_url": f"s3://pmc-oa-opendata/{r.pmcid}.1/{r.pmcid}.1.xml?md5=x"})
            m = re.search(r"/(PMC\d+)\.1/", url)
            if m:
                r = next(x for x in self.records if x.pmcid == m.group(1))
                return httpx.Response(200, text=self.jats(r), headers={"content-type": "application/xml"})
        return httpx.Response(404, text="not mocked: " + url)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


class FakeModel:
    """Admission model stand-in: answers per candidate from a rule on its title."""

    model = "fake-admission"

    def __init__(self, decide) -> None:
        self.decide = decide
        self.calls = 0

    def is_configured(self) -> bool:
        return True

    async def complete_json(self, system: str, user: str, *, max_tokens: int, timeout_s: float):
        from clinical_engine.gateway import ModelCallResult

        self.calls += 1
        items = []
        for cid, title in re.findall(r"^\[(c\d+)\] (.+)$", user, flags=re.M):
            items.append({"id": cid, **self.decide(title)})
        return ModelCallResult(data={"items": items}, model=self.model, tokens_in=500, tokens_out=200)


def dims(**kw: str) -> dict[str, Any]:
    base = {"population": "match", "condition": "match", "intervention": "match", "comparison": "match",
            "outcome": "match", "setting": "match", "role": "primary_result", "state": "direct",
            "exclusion": None, "reason": "Model reason.", "confidence": "high"}
    base.update(kw)
    return base


def dump(obj: Any) -> str:
    return json.dumps(obj, default=str)[:500]
