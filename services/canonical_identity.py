"""Phase 6D — canonical article identity.

ONE source of truth for:
- ``normalize_doi(s)``    — strip ``https://doi.org/`` / ``https://dx.doi.org/`` /
                            ``doi:`` prefixes, trim, lowercase. Returns ``""``
                            if no DOI shape remains.
- ``normalize_trial_id(s)`` — strip whitespace, uppercase ``NCT`` prefix, keep
                              digits. Returns ``""`` if not a valid NCT id.
- ``make_canonical_key(...)`` — priority order:
    1. ``pmid:<digits>``
    2. ``doi:<doi_normalized>``
    3. ``pmcid:<PMCID>``
    4. ``trial:<NCTxxxxxxxx>``
    5. ``title_year_journal_firstauthor:<slug>``

Replaces the duplicate private ``_canonical_key`` copies that previously
lived in ``services/metadata_normalizer.py`` + ``services/connectors/pubmed.py``.

Pure functions. No external calls. Unit-testable.
"""
from __future__ import annotations

import re


_DOI_PREFIX_RE = re.compile(
    r"^\s*(?:https?://)?(?:dx\.)?doi\.org/|^\s*doi:\s*", re.IGNORECASE,
)
_NCT_RE = re.compile(r"^NCT\d{6,10}$")


def normalize_doi(value: str | None) -> str:
    """Return a normalized DOI suitable for indexing.

    Strips ``https://doi.org/`` / ``https://dx.doi.org/`` / ``doi:`` prefixes,
    trims surrounding whitespace, and lowercases. Returns ``""`` if the
    cleaned value doesn't look like a DOI (``10.<reg>/<suffix>``).
    """
    if not value:
        return ""
    s = _DOI_PREFIX_RE.sub("", str(value)).strip().lower()
    # Guard against junk like "n/a" / empty / partial.
    if "/" not in s or not s.startswith("10."):
        return ""
    return s


def normalize_trial_id(value: str | None) -> str:
    """Return ``NCTxxxxxxxx`` (uppercase) or ``""``.

    Accepts trailing/leading whitespace and lowercase ``nct`` prefixes.
    Rejects anything that isn't ``NCT`` + 6–10 digits.
    """
    if not value:
        return ""
    s = str(value).strip().upper().replace(" ", "")
    if not s.startswith("NCT"):
        return ""
    return s if _NCT_RE.match(s) else ""


def _slug(text: str, max_len: int = 64) -> str:
    """Lowercase, alphanumeric-only slug. Mirrors the prior helper."""
    if not text:
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return cleaned[:max_len]


def make_canonical_key(
    *,
    pmid: str | None = None,
    doi: str | None = None,
    pmcid: str | None = None,
    trial_id: str | None = None,
    title: str | None = None,
    year: int | None = None,
    journal: str | None = None,
    first_author: str | None = None,
) -> str:
    """Compute the canonical_key for an article.

    Priority order: pmid > doi > pmcid > trial_id > title+year+journal+author.

    All identifier args are normalized inside this function — callers may
    pass raw values.
    """
    if pmid:
        pmid_clean = str(pmid).strip()
        # Strip a leading ``PMID:`` prefix (case-insensitive) — common
        # in user input from PubMed copy/paste — but preserve the rest
        # so synthetic test fixtures (e.g., ``dx1``) still hash through.
        if pmid_clean.lower().startswith("pmid:"):
            pmid_clean = pmid_clean[5:].strip()
        if pmid_clean:
            return f"pmid:{pmid_clean}"
    doi_norm = normalize_doi(doi)
    if doi_norm:
        return f"doi:{doi_norm}"
    if pmcid:
        pc = str(pmcid).strip().upper()
        if pc:
            if not pc.startswith("PMC"):
                pc = f"PMC{pc.lstrip('PMC')}"
            return f"pmcid:{pc}"
    trial_norm = normalize_trial_id(trial_id)
    if trial_norm:
        return f"trial:{trial_norm}"
    return (
        "title_year_journal_firstauthor:"
        f"{_slug(title or '')}_{year or 0}_"
        f"{_slug(journal or '', 32)}_{_slug(first_author or '', 32)}"
    )
