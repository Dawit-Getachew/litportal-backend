"""Phase H2-tail Task A — author / contact email redaction.

Strips author / corresponding-author email addresses from upstream
PubMed / Europe PMC / OpenAlex / Crossref-derived metadata BEFORE the
normalized article dict is persisted or returned by any API.

This is leaked-PII hardening, NOT a PHI / HIPAA boundary. Author
emails are public-source metadata (PubMed publishes them with
articles) but our app contract is to keep client-facing payloads
clean.

Rules:
  * Redact in ``abstract`` (Europe PMC + PubMed both occasionally
    embed corresponding-author emails in abstract footers).
  * Redact in every ``authors_json[*].affiliation`` string (Europe
    PMC author affiliations often carry the corresponding author's
    email).
  * Redact in every ``full_text_urls_json[*].url`` and
    ``source_urls_json[*]`` value (defensive — URLs sometimes embed
    ``mailto:`` segments or ``?email=`` query parameters).
  * NEVER touch: DOI, PMID, PMCID, journal title, article title,
    author NAMES, author affiliation institutional names (only the
    email segment is stripped, the rest of the affiliation string
    is preserved), license, retraction flags, dates.

The redaction is in-place — the article dict's identity is unchanged.
"""
from __future__ import annotations

import re
from typing import Any


# Closed-set redaction regex. RFC-5322 is intentionally not used —
# the goal is to catch the 99%-case pattern that surfaces in NCBI /
# Europe PMC metadata, NOT to validate addresses.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE)

REDACTED_TOKEN: str = ""


def _redact(s: Any) -> Any:
    """Replace every email pattern in ``s`` with the closed-set
    REDACTED_TOKEN. Non-string inputs are returned unchanged.
    """
    if not isinstance(s, str) or not s:
        return s
    return _EMAIL_RE.sub(REDACTED_TOKEN, s)


def redact_article_emails(article: dict[str, Any]) -> dict[str, Any]:
    """In-place email redaction over the normalized article dict.

    Returns the same dict for chaining. NEVER raises. NEVER modifies
    DOI / PMID / PMCID / IDs / dates / flags.
    """
    if not isinstance(article, dict):
        return article
    # Abstract.
    if "abstract" in article and isinstance(article["abstract"], str):
        article["abstract"] = _redact(article["abstract"])
    # Authors → affiliation strings (Europe PMC carries email here
    # most often).
    authors = article.get("authors_json")
    if isinstance(authors, list):
        for a in authors:
            if not isinstance(a, dict):
                continue
            if "affiliation" in a:
                a["affiliation"] = _redact(a.get("affiliation"))
            # Some upstream sources put a free-text "raw" author
            # string here too.
            if "raw" in a:
                a["raw"] = _redact(a.get("raw"))
    # source_urls_json — list of plain strings.
    urls = article.get("source_urls_json")
    if isinstance(urls, list):
        article["source_urls_json"] = [
            (_redact(u) if isinstance(u, str) else u) for u in urls
        ]
    # full_text_urls_json — list of dicts.
    meta = article.get("full_text_urls_json")
    if isinstance(meta, list):
        for entry in meta:
            if isinstance(entry, dict) and "url" in entry:
                entry["url"] = _redact(entry.get("url"))
    return article


__all__ = ["redact_article_emails", "REDACTED_TOKEN"]
