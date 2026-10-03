"""Phase K4B-light — Known Article Finder service.

Deterministic, identifier-first lookup of a single known article from
clinician-supplied raw text. Used by the "Find known article" modal on
the search results page.

HARD CONSTRAINTS (do NOT relax in subclasses or extensions):
  - No Google Scholar. No publisher-page scraping. No paywall bypass.
  - No LLM call to invent or force-pick an article. Identifier
    resolution is deterministic.
  - PHI guard MUST be invoked on the raw input BEFORE any outbound
    network call. If blocked, the service short-circuits and returns
    a closed-set blocked result.
  - Closed-set match reason codes (see ``_REASON_CODES``).
  - Closed-set access status labels + a SEPARATE closed-set AI-summary
    text source label so the clinician sees "I can read it" is NOT
    the same as "AI can ingest it for summary."
  - No new schema migration; persists nothing beyond an optional
    ephemeral lookup record (admin diagnostics only, no raw text).

Outbound source allowlist (the only hosts ever called):
  - eutils.ncbi.nlm.nih.gov     (PubMed esearch / esummary / efetch)
  - api.ncbi.nlm.nih.gov        (PMC ID converter)
  - api.crossref.org            (Crossref /works/{doi})
  - www.ebi.ac.uk               (Europe PMC search)
"""
from __future__ import annotations

import os
import re
import logging
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Optional, Iterable

import httpx
from pydantic import BaseModel, Field

from services.phi_guard import PHIGuardService
from services.canonical_identity import normalize_doi, make_canonical_key
from services.audit import audit_service
from core.db import Collections

logger = logging.getLogger(__name__)

# Closed-set reason codes per spec.
_REASON_CODES = {
    "exact_doi_match",
    "exact_pmid_match",
    "exact_pmcid_match",
    "exact_title_year_match",
    "fuzzy_title_author_year_match",
    "ambiguous_title_only_match",
}

# Closed-set access status labels (what the human user can do).
ACCESS_STATUS_LABELS = (
    "Can open full text",
    "Link-out only",
    "Abstract only",
)

# Closed-set AI summary text source labels (what the summarizer can
# actually ingest). MUST be visually distinct from access status.
AI_SUMMARY_TEXT_SOURCE_LABELS = (
    "Abstract only",
    "OA full text eligible",
    "User-uploaded PDF eligible",
    "Institutional link-out only, not AI-ingestible",
    "No eligible text",
)

# Outbound host allowlist — used by tests to assert no other host is ever called.
ALLOWED_HOSTS = frozenset({
    "eutils.ncbi.nlm.nih.gov",
    "api.ncbi.nlm.nih.gov",
    "api.crossref.org",
    "www.ebi.ac.uk",
})

# ---------------------------------------------------------------------
# Identifier detection — deterministic regex.
# ---------------------------------------------------------------------
_RE_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)
_RE_PMID = re.compile(r"\bPMID[:\s]*([0-9]{4,10})\b", re.IGNORECASE)
_RE_PMID_BARE = re.compile(r"^\s*([0-9]{4,10})\s*$")
_RE_PMCID = re.compile(r"\b(PMC[0-9]{3,10})\b", re.IGNORECASE)
_RE_NCT = re.compile(r"\b(NCT[0-9]{6,10})\b", re.IGNORECASE)
# Citation-string detector — looks for journal-style "; YYYY[;:]" or
# "vol(issue):pages" patterns.
_RE_CITATION = re.compile(
    r"(?:[A-Z][a-z]+\s+[A-Z]+(?:,\s*[A-Z][a-z]+\s+[A-Z]+)*?\.\s*)?"
    r".+?\.\s*(?P<journal>[A-Za-z][A-Za-z &.\-]+)\.\s*"
    r"(?P<year>\d{4})\s*[;:]\s*"
    r"(?:(?P<volume>\d+)(?:\((?P<issue>[\dA-Za-z]+)\))?:)?"
    r"(?P<pages>\d+[-\u2013\u2014]\d+)",
)


# Phase K4B-light — pre-strip patterns. These are the IDENTIFIER-shaped
# substrings that legitimately appear in the known-article-lookup
# workflow but would otherwise trip the base PHI guard. We strip them
# from the raw text BEFORE PHI screening, so the PHI guard sees the
# clean residual (authors, journal, free text). PHI in the residual
# still blocks unchanged.
_IDENTIFIER_STRIP_PATTERNS = (
    # DOI URLs / bare DOIs.
    re.compile(r"https?://(?:dx\.)?doi\.org/\S+", re.IGNORECASE),
    re.compile(r"\bdoi:\s*10\.\d{4,9}/\S+", re.IGNORECASE),
    re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE),
    # PubMed URLs.
    re.compile(r"https?://(?:www\.)?(?:pubmed|ncbi)\.\S+", re.IGNORECASE),
    # Europe PMC URLs.
    re.compile(r"https?://(?:www\.|europepmc\.)\S*europepmc\S*", re.IGNORECASE),
    # PMID labels + digit runs.
    re.compile(r"\bPMID[:\s]*[0-9]{4,10}\b", re.IGNORECASE),
    # PMCIDs.
    re.compile(r"\bPMC[0-9]{3,10}\b", re.IGNORECASE),
    re.compile(r"\bPMCID[:\s]*PMC[0-9]{3,10}\b", re.IGNORECASE),
    # ClinicalTrials.gov NCT IDs.
    re.compile(r"\bNCT[0-9]{6,10}\b", re.IGNORECASE),
    # Standalone 4-digit years (common in citations; safe to strip since
    # we never echo or store the raw text).
    re.compile(r"\b(?:19|20)\d{2}\b"),
    # Volume(Issue):Pages citation tail like ``;385:1451-1461``.
    re.compile(r";\s*\d{1,4}(?:\(\S+?\))?:\d+\s*[-\u2013\u2014]\s*\d+"),
)


def _strip_identifiers_for_phi_screen(text: str) -> str:
    """Return a copy of ``text`` with identifier-shaped substrings
    replaced by spaces. Pure function — used only by the PHI-screen
    step of the known-article-lookup workflow. Never persisted, never
    sent outbound; the original ``text`` is the only string passed
    to identifier parsers."""
    if not text:
        return ""
    residual = text
    for pat in _IDENTIFIER_STRIP_PATTERNS:
        residual = pat.sub(" ", residual)
    return residual


class ParsedInput(BaseModel):
    """Parsed lookup input — pure derivation, no outbound calls."""
    raw_text_preview: str  # first 200 chars; never echoed in diagnostics
    detected_kind: str  # closed-set: doi | pmid | pmcid | nct | title | citation | unknown
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    nct: Optional[str] = None
    title_guess: Optional[str] = None
    journal_guess: Optional[str] = None
    year_guess: Optional[int] = None
    volume_guess: Optional[str] = None
    issue_guess: Optional[str] = None
    pages_guess: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


def parse_input(raw: str) -> ParsedInput:
    """Detect what the clinician pasted. Pure function. No outbound calls."""
    raw = (raw or "").strip()
    preview = raw[:200]
    warnings: list[str] = []
    if not raw:
        return ParsedInput(
            raw_text_preview="", detected_kind="unknown",
            warnings=["Empty input."],
        )
    if len(raw) > 4000:
        warnings.append("Input truncated to 4000 chars.")
        raw = raw[:4000]

    # Identifier-first detection.
    m_doi = _RE_DOI.search(raw)
    m_pmid = _RE_PMID.search(raw)
    m_pmcid = _RE_PMCID.search(raw)
    m_nct = _RE_NCT.search(raw)
    bare_pmid = _RE_PMID_BARE.match(raw)

    if m_doi:
        doi = normalize_doi(m_doi.group(1))
        if doi:
            return ParsedInput(
                raw_text_preview=preview, detected_kind="doi", doi=doi,
                pmid=m_pmid.group(1) if m_pmid else None,
                pmcid=m_pmcid.group(1).upper() if m_pmcid else None,
                warnings=warnings,
            )
    if m_pmid:
        return ParsedInput(
            raw_text_preview=preview, detected_kind="pmid",
            pmid=m_pmid.group(1),
            pmcid=m_pmcid.group(1).upper() if m_pmcid else None,
            warnings=warnings,
        )
    if bare_pmid:
        return ParsedInput(
            raw_text_preview=preview, detected_kind="pmid",
            pmid=bare_pmid.group(1), warnings=warnings,
        )
    if m_pmcid:
        return ParsedInput(
            raw_text_preview=preview, detected_kind="pmcid",
            pmcid=m_pmcid.group(1).upper(), warnings=warnings,
        )
    if m_nct:
        return ParsedInput(
            raw_text_preview=preview, detected_kind="nct",
            nct=m_nct.group(1).upper(), warnings=warnings,
        )

    # Citation-string vs free title.
    m_cite = _RE_CITATION.search(raw)
    if m_cite:
        try:
            year_int = int(m_cite.group("year"))
        except (TypeError, ValueError):
            year_int = None
        # The citation regex captures a year + pages tail. The
        # journal is the period-delimited segment IMMEDIATELY BEFORE
        # the year; the title is the segment before THAT (skipping
        # any author-list prefix). We re-derive both via a
        # deterministic split rather than relying on the greedy
        # regex sub-captures.
        title_guess: Optional[str] = None
        journal_str: Optional[str] = None
        # Find the year position; split everything before it on ". ".
        year_idx = raw.find(str(year_int)) if year_int else -1
        if year_idx > 0:
            head = raw[:year_idx].strip().rstrip(",.;: ")
            segs = [s.strip().rstrip(",") for s in head.split(".") if s.strip()]
            # Drop the trailing semicolon segment if present.
            if segs and segs[-1].endswith(";"):
                segs[-1] = segs[-1][:-1].strip()
            # Expect: [authors], title, journal — drop authors if 3+ segs.
            if len(segs) >= 3:
                journal_str = segs[-1][:200]
                title_guess = segs[-2][:300]
            elif len(segs) == 2:
                # title + journal (no author list).
                journal_str = segs[-1][:200]
                title_guess = segs[0][:300]
            elif len(segs) == 1:
                title_guess = segs[0][:300]
        if not title_guess:
            title_guess = raw.split(".")[0].strip()[:300] or None
        return ParsedInput(
            raw_text_preview=preview, detected_kind="citation",
            title_guess=title_guess,
            journal_guess=journal_str,
            year_guess=year_int,
            volume_guess=m_cite.group("volume"),
            issue_guess=m_cite.group("issue"),
            pages_guess=m_cite.group("pages"),
            warnings=warnings,
        )
    # Fallback: treat as title.
    if len(raw) >= 8:
        return ParsedInput(
            raw_text_preview=preview, detected_kind="title",
            title_guess=raw[:300], warnings=warnings,
        )
    return ParsedInput(
        raw_text_preview=preview, detected_kind="unknown",
        warnings=["Input too short or ambiguous to identify."],
    )


# ---------------------------------------------------------------------
# Source-call wrappers — kept narrow + named for easy mocking in tests.
# ---------------------------------------------------------------------
async def _pubmed_efetch_by_pmid(pmid: str) -> Optional[dict[str, Any]]:
    """Single-PMID PubMed lookup via esummary + efetch.

    Returns a normalized candidate dict on success, ``None`` otherwise.
    Subject to test-side monkey-patching.
    """
    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    email = os.environ.get("NCBI_EMAIL", "").strip()
    tool = os.environ.get("NCBI_TOOL", "ClinEvidenceSearch").strip()
    common = {"tool": tool}
    if email:
        common["email"] = email
    if api_key:
        common["api_key"] = api_key
    async with httpx.AsyncClient(timeout=15.0) as client:
        sum_r = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": pmid, "retmode": "json", **common},
        )
        if sum_r.status_code != 200:
            return None
        body = (sum_r.json().get("result") or {}).get(pmid) or {}
        if not body:
            return None
        fetch_r = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": pmid, "rettype": "abstract",
                    "retmode": "xml", **common},
        )
        xml_text = fetch_r.text if fetch_r.status_code == 200 else ""
    return _normalize_pubmed_record(pmid, body, xml_text)


async def _pubmed_esearch_by_title(title: str, year: Optional[int] = None) -> list[str]:
    """Title-based PubMed esearch. Returns a list of PMIDs."""
    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    email = os.environ.get("NCBI_EMAIL", "").strip()
    tool = os.environ.get("NCBI_TOOL", "ClinEvidenceSearch").strip()
    common = {"tool": tool}
    if email:
        common["email"] = email
    if api_key:
        common["api_key"] = api_key
    q = f'"{title}"[Title]'
    if year:
        q += f" AND {year}[PDAT]"
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": q, "retmode": "json",
                    "retmax": "10", **common},
        )
        if r.status_code != 200:
            return []
        return list((r.json().get("esearchresult") or {}).get("idlist") or [])


async def _pubmed_esearch_by_doi(doi: str) -> Optional[str]:
    """Resolve a DOI to a PMID via PubMed's ``[AID]`` field. Returns PMID or None."""
    common: dict[str, str] = {"tool": os.environ.get("NCBI_TOOL", "ClinEvidenceSearch").strip()}
    email = os.environ.get("NCBI_EMAIL", "").strip()
    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    if email:
        common["email"] = email
    if api_key:
        common["api_key"] = api_key
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": f"{doi}[AID]", "retmode": "json",
                    "retmax": "1", **common},
        )
        if r.status_code != 200:
            return None
        ids = list((r.json().get("esearchresult") or {}).get("idlist") or [])
        return ids[0] if ids else None


async def _crossref_by_doi(doi: str) -> Optional[dict[str, Any]]:
    """Single-DOI Crossref lookup. Returns a Crossref message dict on success."""
    email = os.environ.get("CROSSREF_EMAIL", "").strip()
    headers = {"User-Agent": f"ClinEvidenceSearch ({email})" if email else "ClinEvidenceSearch"}
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/')}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url, headers=headers)
        if r.status_code != 200:
            return None
        return (r.json() or {}).get("message")


async def _pmc_idconv(pmcid: str) -> Optional[dict[str, str]]:
    """PMCID → PMID + DOI via NCBI ID converter."""
    pmcid_clean = pmcid.upper()
    if not pmcid_clean.startswith("PMC"):
        pmcid_clean = "PMC" + pmcid_clean
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": f"{pmcid_clean}[pmcid]",
                    "retmode": "json", "retmax": "1"},
        )
        if r.status_code != 200:
            return None
        ids = list((r.json().get("esearchresult") or {}).get("idlist") or [])
        if not ids:
            return None
        return {"pmcid": pmcid_clean, "pmid": ids[0]}


# ---------------------------------------------------------------------
# Normalization helpers.
# ---------------------------------------------------------------------
def _normalize_pubmed_record(pmid: str, summary: dict, xml_text: str) -> dict[str, Any]:
    """Compose a closed-set candidate dict from PubMed esummary + efetch."""
    title = (summary.get("title") or "").strip()
    journal = (summary.get("fulljournalname") or summary.get("source") or "").strip()
    pubdate = (summary.get("pubdate") or "").strip()
    year = None
    if pubdate:
        try:
            year = int(pubdate.split()[0])
        except (ValueError, IndexError):
            year = None
    authors_list = summary.get("authors") or []
    authors = [
        a.get("name", "").strip() for a in authors_list
        if isinstance(a, dict) and a.get("name")
    ]
    article_ids = summary.get("articleids") or []
    doi = ""
    pmcid: Optional[str] = None
    for aid in article_ids:
        if not isinstance(aid, dict):
            continue
        idtype = (aid.get("idtype") or "").lower()
        v = (aid.get("value") or "").strip()
        if idtype == "doi" and v:
            doi = normalize_doi(v)
        elif idtype == "pmc" and v:
            pmcid = v.upper() if v.upper().startswith("PMC") else f"PMC{v}"
    # Parse abstract + pub types from XML if present.
    abstract = None
    pub_types: list[str] = []
    is_retracted = False
    has_expression_of_concern = False
    if xml_text:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            root = None
        if root is not None:
            for ab in root.iter("AbstractText"):
                label = ab.attrib.get("Label")
                text = (ab.text or "").strip()
                if not text:
                    continue
                if label:
                    pub_types  # avoid lint
                    abstract = (abstract or "") + ("\n" if abstract else "") + f"{label}: {text}"
                else:
                    abstract = (abstract or "") + ("\n" if abstract else "") + text
            for pt in root.iter("PublicationType"):
                if pt.text:
                    txt = pt.text.strip()
                    if txt:
                        pub_types.append(txt)
                        if txt.lower() == "retracted publication":
                            is_retracted = True
                        if "expression of concern" in txt.lower():
                            has_expression_of_concern = True
    return {
        "source": "pubmed",
        "pmid": pmid,
        "doi": doi or None,
        "pmcid": pmcid,
        "title": title or None,
        "journal": journal or None,
        "publication_year": year,
        "authors": authors,
        "abstract": abstract,
        "publication_types": pub_types,
        "is_retracted": is_retracted,
        "has_expression_of_concern": has_expression_of_concern,
    }


def _normalize_crossref(msg: dict[str, Any]) -> dict[str, Any]:
    title_list = msg.get("title") or []
    title = (title_list[0] if title_list else "").strip()
    journal_list = msg.get("container-title") or []
    journal = (journal_list[0] if journal_list else "").strip()
    issued = (msg.get("issued") or {}).get("date-parts") or [[None]]
    year = None
    try:
        if issued and issued[0]:
            year = int(issued[0][0])
    except (ValueError, TypeError, IndexError):
        year = None
    authors = []
    for a in msg.get("author") or []:
        if isinstance(a, dict):
            full = " ".join(filter(None, [a.get("given"), a.get("family")])).strip()
            if full:
                authors.append(full)
    return {
        "source": "crossref",
        "pmid": None,
        "doi": normalize_doi(msg.get("DOI")),
        "pmcid": None,
        "title": title or None,
        "journal": journal or None,
        "publication_year": year,
        "authors": authors,
        "abstract": None,
        "publication_types": [msg.get("type")] if msg.get("type") else [],
        "is_retracted": False,
        "has_expression_of_concern": False,
    }


# ---------------------------------------------------------------------
# Confidence + reason-code derivation.
# ---------------------------------------------------------------------
def _norm_for_compare(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()


def derive_confidence(
    *,
    parsed: ParsedInput,
    candidate: dict[str, Any],
    num_candidates: int,
) -> tuple[str, list[str]]:
    """Return (closed-set confidence, closed-set reason_codes)."""
    reasons: list[str] = []
    # Exact identifier matches win.
    if parsed.doi and candidate.get("doi") and normalize_doi(parsed.doi) == candidate.get("doi"):
        reasons.append("exact_doi_match")
        return "high", reasons
    if parsed.pmid and candidate.get("pmid") and parsed.pmid == candidate.get("pmid"):
        reasons.append("exact_pmid_match")
        return "high", reasons
    if parsed.pmcid and candidate.get("pmcid") and parsed.pmcid.upper() == candidate.get("pmcid", "").upper():
        reasons.append("exact_pmcid_match")
        return "high", reasons
    # Title-based matches.
    title_norm_p = _norm_for_compare(parsed.title_guess)
    title_norm_c = _norm_for_compare(candidate.get("title"))
    title_match = bool(title_norm_p and title_norm_c and (
        title_norm_p == title_norm_c or title_norm_p in title_norm_c
        or title_norm_c in title_norm_p
    ))
    year_match = bool(
        parsed.year_guess and candidate.get("publication_year")
        and parsed.year_guess == candidate.get("publication_year")
    )
    if title_match and year_match:
        reasons.append("exact_title_year_match")
        return "high", reasons
    if title_match and candidate.get("authors") and parsed.title_guess:
        # Title matches and there's at least one author + maybe year — medium fuzzy.
        reasons.append("fuzzy_title_author_year_match")
        return "medium" if num_candidates == 1 else "low", reasons
    if title_norm_p and title_norm_c:
        reasons.append("ambiguous_title_only_match")
        return "low", reasons
    return "low", reasons or ["ambiguous_title_only_match"]


# ---------------------------------------------------------------------
# Access-status / AI-summary-source derivation.
# ---------------------------------------------------------------------
def derive_access_panel(
    *,
    article: dict[str, Any],
    access_rows: Iterable[dict[str, Any]] = (),
    has_user_upload: bool = False,
) -> dict[str, Any]:
    """Closed-set access status + AI-summary text source. Two SEPARATE fields."""
    rows = list(access_rows or [])
    is_oa = any(r.get("is_oa") for r in rows)
    has_institutional = any(
        (r.get("access_category") or "").lower() == "institutional" for r in rows
    )
    has_oa_direct_pdf = any(r.get("is_direct_pdf") and r.get("is_oa") for r in rows)
    # full_text_status from PubMed normalization stub.
    fts = (article.get("full_text_status") or "").lower()
    if not fts and rows:
        if is_oa:
            fts = "full_text_available"
        elif has_institutional:
            fts = "free_full_text_link"
        else:
            fts = "abstract_only" if article.get("abstract") else "unknown"
    abstract_present = bool(article.get("abstract"))

    # ACCESS STATUS — what the human user can do.
    if is_oa or has_oa_direct_pdf:
        access_status = "Can open full text"
    elif has_institutional:
        access_status = "Link-out only"
    elif abstract_present:
        access_status = "Abstract only"
    else:
        access_status = "Abstract only"  # neutral default; UI still says "no full text"

    # AI SUMMARY TEXT SOURCE — what the summarizer can actually ingest.
    can_ingest_oa = any(
        r.get("can_ingest_for_summary") for r in rows
        if (r.get("access_category") or "").lower() == "oa"
    )
    institutional_only_no_oa = (has_institutional and not is_oa
                                and not has_oa_direct_pdf
                                and not has_user_upload)
    if has_user_upload:
        ai_source = "User-uploaded PDF eligible"
    elif can_ingest_oa or has_oa_direct_pdf:
        ai_source = "OA full text eligible"
    elif institutional_only_no_oa:
        ai_source = "Institutional link-out only, not AI-ingestible"
    elif abstract_present:
        ai_source = "Abstract only"
    else:
        ai_source = "No eligible text"

    return {
        "access_status": access_status,
        "ai_summary_text_source": ai_source,
        "can_view_in_app": bool(is_oa or has_oa_direct_pdf or abstract_present),
        "can_ingest_for_summary": ai_source not in (
            "No eligible text",
            "Institutional link-out only, not AI-ingestible",
        ),
        "has_oa": is_oa,
        "has_institutional_linkout": has_institutional,
        "has_user_upload": has_user_upload,
    }


# ---------------------------------------------------------------------
# Service class — orchestrates the whole flow with PHI guard at front.
# ---------------------------------------------------------------------
class KnownArticleLookupService:
    """Orchestrates parse → PHI guard → identifier-first source cascade
    → candidate scoring → optional access broker → attach to session.
    """

    def __init__(self, *, phi_guard: Optional[PHIGuardService] = None) -> None:
        self.phi_guard = phi_guard or PHIGuardService()

    async def search(self, raw_text: str) -> dict[str, Any]:
        """Top-level orchestration. Returns closed-set result dict."""
        raw_text = raw_text or ""
        # Phase K4B-light — the known-article-lookup workflow EXPECTS
        # identifier-shaped input (DOI URLs, PMID digit-runs, PMCID,
        # NCT, 4-digit years). The base PHI guard correctly flags
        # ``https://`` URLs and 7+ digit sequences as PHI signals for
        # the clinical-question context — but those are legitimate
        # identifier inputs here. We pre-strip detected identifiers
        # (and only identifiers) from the raw text BEFORE running
        # the existing PHI guard on the residual. Any real PHI in
        # the residual (emails, SSNs, names, addresses, dates outside
        # citation year context) is still blocked unchanged.
        residual = _strip_identifiers_for_phi_screen(raw_text)
        scr = self.phi_guard.screen(residual)
        if scr.is_blocked:
            return {
                "phi_blocked": True,
                "reasons": list(scr.reason_codes),
                "parsed": None,
                "candidates": [],
                "lookup_source_provenance": [],
            }
        parsed = parse_input(raw_text)
        if parsed.detected_kind == "unknown":
            return {
                "phi_blocked": False, "reasons": [],
                "parsed": parsed.model_dump(),
                "candidates": [],
                "lookup_source_provenance": [],
            }

        candidates: list[dict[str, Any]] = []
        provenance: list[str] = []
        # 2. Identifier-first cascade.
        if parsed.detected_kind == "pmid" and parsed.pmid:
            rec = await _pubmed_efetch_by_pmid(parsed.pmid)
            provenance.append("pubmed")
            if rec:
                candidates.append(rec)
        elif parsed.detected_kind == "doi" and parsed.doi:
            # Try PubMed via [AID] first; fall back to Crossref.
            pmid = await _pubmed_esearch_by_doi(parsed.doi)
            provenance.append("pubmed")
            if pmid:
                rec = await _pubmed_efetch_by_pmid(pmid)
                if rec:
                    candidates.append(rec)
            if not candidates:
                msg = await _crossref_by_doi(parsed.doi)
                provenance.append("crossref")
                if msg:
                    candidates.append(_normalize_crossref(msg))
        elif parsed.detected_kind == "pmcid" and parsed.pmcid:
            id_pair = await _pmc_idconv(parsed.pmcid)
            provenance.append("pubmed_pmcid_converter")
            if id_pair and id_pair.get("pmid"):
                rec = await _pubmed_efetch_by_pmid(id_pair["pmid"])
                if rec:
                    # Ensure pmcid is preserved on the record.
                    rec["pmcid"] = id_pair["pmcid"]
                    candidates.append(rec)
        elif parsed.detected_kind in ("title", "citation") and parsed.title_guess:
            pmids = await _pubmed_esearch_by_title(
                parsed.title_guess, year=parsed.year_guess,
            )
            provenance.append("pubmed")
            # Limit fan-out to top 5 to keep latency bounded.
            for pmid in pmids[:5]:
                rec = await _pubmed_efetch_by_pmid(pmid)
                if rec:
                    candidates.append(rec)

        # 3. Score + assemble closed-set candidate result.
        out_candidates: list[dict[str, Any]] = []
        for c in candidates:
            confidence, reasons = derive_confidence(
                parsed=parsed, candidate=c, num_candidates=len(candidates),
            )
            out_candidates.append({
                "match_confidence": confidence,
                "match_reason_codes": reasons,
                "pmid": c.get("pmid"),
                "doi": c.get("doi"),
                "pmcid": c.get("pmcid"),
                "title": c.get("title"),
                "journal": c.get("journal"),
                "publication_year": c.get("publication_year"),
                "authors": c.get("authors") or [],
                "abstract": c.get("abstract"),
                "publication_types": c.get("publication_types") or [],
                "is_retracted": bool(c.get("is_retracted")),
                "has_expression_of_concern": bool(c.get("has_expression_of_concern")),
                "source": c.get("source") or "pubmed",
            })
        # Sort by confidence (high > medium > low), then by source preference.
        rank = {"high": 0, "medium": 1, "low": 2}
        out_candidates.sort(key=lambda x: (
            rank.get(x["match_confidence"], 3),
            0 if x.get("pmid") else 1,
        ))
        return {
            "phi_blocked": False,
            "reasons": [],
            "parsed": parsed.model_dump(),
            "candidates": out_candidates,
            "lookup_source_provenance": provenance,
        }

    async def confirm_access(
        self, *, db, candidate: dict[str, Any], user_id: str,
    ) -> dict[str, Any]:
        """Resolve access for a single candidate. Reuses
        FullTextAccessBrokerService for OA discovery if the article
        already lives in the DB; otherwise derives a conservative
        access panel directly from candidate metadata + the candidate's
        own fields (no scraping)."""
        article_id = await self._upsert_article(db, candidate)
        # Fetch any existing access locations for this article.
        rows: list[dict[str, Any]] = []
        async for r in db[Collections.ARTICLE_ACCESS_LOCATIONS].find(
            {"article_id": article_id},
            {"_id": 0},
        ):
            rows.append(r)
        # User uploads for this article (any session).
        upload = await db[Collections.UPLOADED_FULL_TEXT_ASSETS].find_one(
            {"article_id": article_id, "user_id": user_id, "upload_status": "processed"},
            {"_id": 0, "id": 1},
        )
        panel = derive_access_panel(
            article={
                "full_text_status": candidate.get("full_text_status"),
                "abstract": candidate.get("abstract"),
            },
            access_rows=rows,
            has_user_upload=bool(upload),
        )
        panel["article_id"] = article_id
        panel["access_rows_total"] = len(rows)
        return panel

    async def attach_to_session(
        self,
        *,
        db,
        session_id: str,
        user_id: str,
        candidate: dict[str, Any],
        include_in_evidence: bool = False,
    ) -> dict[str, Any]:
        """Attach (or surface existing) candidate as a search_session_articles row.

        Deduplicates against existing session articles using
        canonical_key. NEVER attaches a low-confidence candidate
        silently — the caller must ensure confidence is acceptable
        BEFORE invoking this method; this method does not gate on
        confidence (gating is the router's job so the closed-set
        rule remains visible).
        """
        session = await db[Collections.SEARCH_SESSIONS].find_one(
            {"id": session_id, "user_id": user_id, "deleted_at": None},
            {"_id": 0, "id": 1, "status": 1},
        )
        if not session:
            return {"status": "session_not_found"}
        if session.get("status") == "blocked_phi":
            return {"status": "session_blocked_phi"}

        article_id = await self._upsert_article(db, candidate)
        # Dedup vs existing session articles.
        existing = await db[Collections.SEARCH_SESSION_ARTICLES].find_one(
            {"search_session_id": session_id, "article_id": article_id},
            {"_id": 0, "id": 1, "user_included": 1},
        )
        if existing:
            if include_in_evidence and not existing.get("user_included"):
                await db[Collections.SEARCH_SESSION_ARTICLES].update_one(
                    {"id": existing["id"]},
                    {"$set": {"user_included": True,
                              "updated_at": datetime.now(timezone.utc)}},
                )
            await audit_service.log(
                "known_article_lookup_duplicate_in_session",
                user_id=user_id, search_session_id=session_id,
                payload={"article_id": article_id,
                         "search_session_article_id": existing["id"]},
            )
            return {
                "status": "already_in_session",
                "article_id": article_id,
                "search_session_article_id": existing["id"],
                "was_duplicate": True,
                "user_included": bool(include_in_evidence) or bool(existing.get("user_included")),
            }
        # Create the attachment.
        import uuid
        now = datetime.now(timezone.utc)
        ssa_id = str(uuid.uuid4())
        await db[Collections.SEARCH_SESSION_ARTICLES].insert_one({
            "id": ssa_id,
            "search_session_id": session_id,
            "article_id": article_id,
            "relevance_score": None,
            "evidence_design_score": None,
            "overall_rank": None,
            "rank_explanation": None,
            "included_by_default": False,
            "user_included": bool(include_in_evidence),
            "exclusion_reason": None,
            "source": "known_article_lookup",
            "created_at": now,
            "updated_at": now,
        })
        await audit_service.log(
            "known_article_lookup_attached",
            user_id=user_id, search_session_id=session_id,
            payload={
                "article_id": article_id,
                "search_session_article_id": ssa_id,
                "match_confidence": candidate.get("match_confidence"),
                "match_reason_codes": candidate.get("match_reason_codes") or [],
                "include_in_evidence": bool(include_in_evidence),
            },
        )
        return {
            "status": "attached",
            "article_id": article_id,
            "search_session_article_id": ssa_id,
            "was_duplicate": False,
            "user_included": bool(include_in_evidence),
        }

    async def _upsert_article(self, db, candidate: dict[str, Any]) -> str:
        """Find-or-create an ``articles`` row keyed by canonical_key."""
        ck = make_canonical_key(
            pmid=candidate.get("pmid"),
            doi=candidate.get("doi"),
            pmcid=candidate.get("pmcid"),
            title=candidate.get("title"),
            year=candidate.get("publication_year"),
            journal=candidate.get("journal"),
            first_author=(candidate.get("authors") or [None])[0],
        )
        existing = await db[Collections.ARTICLES].find_one(
            {"canonical_key": ck}, {"_id": 0, "id": 1},
        )
        if existing:
            return existing["id"]
        import uuid
        now = datetime.now(timezone.utc)
        aid = str(uuid.uuid4())
        await db[Collections.ARTICLES].insert_one({
            "id": aid,
            "canonical_key": ck,
            "title": candidate.get("title"),
            "abstract": candidate.get("abstract"),
            "journal": candidate.get("journal"),
            "publication_year": candidate.get("publication_year"),
            "publication_date": None,
            "authors_json": candidate.get("authors") or [],
            "doi": candidate.get("doi"),
            "pmid": candidate.get("pmid"),
            "pmcid": candidate.get("pmcid"),
            # Where the metadata came from — the AI-use policy is per source.
            "source_ids_json": (
                {candidate["source"]: candidate.get("pmid") or candidate.get("doi") or ""}
                if candidate.get("source") in ("pubmed", "crossref") else {}
            ),
            "publication_types_json": candidate.get("publication_types") or [],
            "article_type_classification": None,
            "is_guideline": False,
            "is_preprint": False,
            "is_retracted": bool(candidate.get("is_retracted")),
            "has_expression_of_concern": bool(candidate.get("has_expression_of_concern")),
            "full_text_status": (
                "abstract_only" if candidate.get("abstract") else "unknown"
            ),
            "source_urls_json": [],
            "created_at": now,
            "updated_at": now,
        })
        return aid


# Singleton — same pattern as other service modules.
known_article_lookup_service = KnownArticleLookupService()
