"""MetadataNormalizerService — Europe PMC → canonical articles schema."""
from __future__ import annotations

import re
from typing import Any

from models.common import new_id, utcnow
from services.article_email_redactor import redact_article_emails
from services.canonical_identity import make_canonical_key, normalize_doi

_GUIDELINE_PUB_TYPES = {"guideline", "practice guideline", "consensus development conference"}
_RETRACTED_TOKENS = ("retracted publication",)
_RETRACTION_NOTICE_TOKENS = ("retraction of publication", "retraction notice")
_EOC_TOKENS = ("expression of concern",)
_PREPRINT_TOKENS = ("preprint",)


def _slug(s: str, maxlen: int = 64) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:maxlen]


def _first_author_last_name(authors_json: list[dict[str, Any]]) -> str:
    if not authors_json:
        return ""
    first = authors_json[0]
    # Europe PMC author entries usually have fullName ("Smith J") and lastName.
    last = first.get("lastName") or ""
    if last:
        return last
    full = first.get("fullName") or first.get("name") or ""
    # Heuristic: assume "Lastname Initials" or "Lastname FN"
    return (full.split(" ")[0] if full else "").strip()


def _classify_article_type(pub_types_lower: list[str]) -> str:
    pts = " | ".join(pub_types_lower)
    if any(g in pts for g in _GUIDELINE_PUB_TYPES):
        return "guideline"
    if "systematic review" in pts:
        return "systematic_review"
    if "meta analysis" in pts or "meta-analysis" in pts:
        return "meta_analysis"
    if "randomized controlled trial" in pts:
        return "rct"
    if "diagnostic accuracy" in pts:
        return "diagnostic_accuracy_study"
    if "cohort" in pts:
        return "cohort_study"
    if "case-control" in pts or "case control" in pts:
        return "case_control"
    if "cross-sectional" in pts or "cross sectional" in pts:
        return "cross_sectional"
    if "review" in pts:
        return "narrative_review"
    return "other"


def _full_text_status(raw: dict[str, Any], pmcid: str) -> str:
    if (raw.get("inEPMC") or "").upper() == "Y" or (raw.get("inPMC") or "").upper() == "Y" or pmcid:
        return "full_text_available"
    full_urls = ((raw.get("fullTextUrlList") or {}).get("fullTextUrl") or [])
    for u in full_urls:
        if (u.get("availabilityCode") or "").upper() in {"OA", "F"}:
            return "free_full_text_link"
    if raw.get("abstractText"):
        return "abstract_only"
    return "unknown"


def _canonical_key(pmid: str, doi: str, pmcid: str,
                   title: str, year: int | None,
                   journal: str, first_author: str) -> str:
    if pmid:
        return f"pmid:{pmid}"
    if doi:
        return f"doi:{doi.strip().lower()}"
    if pmcid:
        return f"pmcid:{pmcid}"
    return (
        "title_year_journal_firstauthor:"
        f"{_slug(title)}_{year or 0}_{_slug(journal, 32)}_{_slug(first_author, 32)}"
    )


class MetadataNormalizerService:
    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        title = (raw.get("title") or "").strip()
        abstract = (raw.get("abstractText") or "").strip() or None
        journal_info = raw.get("journalInfo") or {}
        journal = ((journal_info.get("journal") or {}).get("title") or "").strip() or None
        pub_year_raw = raw.get("pubYear") or journal_info.get("yearOfPublication")
        try:
            pub_year = int(pub_year_raw) if pub_year_raw else None
        except (TypeError, ValueError):
            pub_year = None
        pub_date = (raw.get("firstPublicationDate") or raw.get("electronicPublicationDate") or "").strip() or None

        authors_raw = ((raw.get("authorList") or {}).get("author") or [])
        authors_json: list[dict[str, Any]] = []
        for a in authors_raw:
            authors_json.append({
                "fullName": a.get("fullName"),
                "lastName": a.get("lastName"),
                "firstName": a.get("firstName"),
                "initials": a.get("initials"),
                "affiliation": (a.get("authorAffiliationDetailsList") or {})
                    .get("authorAffiliation", [{}])[0].get("affiliation") if a.get("authorAffiliationDetailsList") else None,
            })
        first_author = _first_author_last_name(authors_json)

        doi = (raw.get("doi") or "").strip() or None
        pmid = (raw.get("pmid") or "").strip() or None
        pmcid = (raw.get("pmcid") or "").strip() or None

        pub_types = ((raw.get("pubTypeList") or {}).get("pubType") or [])
        if isinstance(pub_types, str):
            pub_types = [pub_types]
        pub_types_lower = [p.lower() for p in pub_types if isinstance(p, str)]

        source_code = (raw.get("source") or "").strip().upper()
        in_epmc = (raw.get("inEPMC") or "").upper() == "Y"
        source_ids_json = {
            "europe_pmc_source_code": source_code or None,
            "europe_pmc_id": raw.get("id"),
            "inEPMC": in_epmc,
        }

        full_text_urls = []
        full_text_urls_meta: list[dict[str, Any]] = []
        for u in ((raw.get("fullTextUrlList") or {}).get("fullTextUrl") or []):
            if u.get("url"):
                full_text_urls.append(u["url"])
            # Preserve availabilityCode + documentStyle for the access broker
            # (Phase 2C). Only structured metadata, never raw HTML.
            full_text_urls_meta.append({
                "url": u.get("url"),
                "availabilityCode": u.get("availabilityCode"),
                "availability": u.get("availability"),
                "documentStyle": u.get("documentStyle"),
                "site": u.get("site"),
            })

        is_preprint = (
            source_code == "PPR"
            or any(t in p for p in pub_types_lower for t in _PREPRINT_TOKENS)
        )
        is_guideline = any(p in _GUIDELINE_PUB_TYPES for p in pub_types_lower)
        is_retracted = any(any(tok in p for tok in _RETRACTED_TOKENS) for p in pub_types_lower)
        is_retraction_notice = any(
            any(tok in p for tok in _RETRACTION_NOTICE_TOKENS) for p in pub_types_lower
        )
        has_eoc = any(any(tok in p for tok in _EOC_TOKENS) for p in pub_types_lower)

        article_type = _classify_article_type(pub_types_lower)
        if is_guideline:
            article_type = "guideline"

        full_text_status = _full_text_status(raw, pmcid or "")

        canonical_key = make_canonical_key(
            pmid=pmid or "", doi=doi or "", pmcid=pmcid or "",
            title=title, year=pub_year,
            journal=journal or "", first_author=first_author,
        )
        doi_normalized = normalize_doi(doi or "") or None

        article = {
            "id": new_id(),
            "canonical_key": canonical_key,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "publication_year": pub_year,
            "publication_date": pub_date,
            "authors_json": authors_json,
            "doi": doi,
            "doi_normalized": doi_normalized,
            "pmid": pmid,
            "pmcid": pmcid,
            "source_ids_json": source_ids_json,
            "publication_types_json": pub_types_lower,
            "article_type_classification": article_type,
            "is_guideline": is_guideline,
            "is_preprint": is_preprint,
            "is_retracted": is_retracted,
            "is_retraction_notice": is_retraction_notice,
            "has_expression_of_concern": has_eoc,
            "full_text_status": full_text_status,
            "source_urls_json": full_text_urls,
            "full_text_urls_json": full_text_urls_meta,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        # Phase H2-tail Task A — strip author / contact / corresponding-
        # author emails before the article dict is persisted or
        # returned. Closed-set, idempotent.
        return redact_article_emails(article)


metadata_normalizer = MetadataNormalizerService()
