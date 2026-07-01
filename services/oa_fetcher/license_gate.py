"""Conservative license gate for the Phase 4A OA fetcher.

Rules (default env):
  CC0 / Public domain / CC BY / CC BY-SA  → ingest. Cache article-level.
  CC BY-NC, CC BY-NC-SA, CC BY-NC-ND      → skip (unless OA_ALLOW_NONCOMMERCIAL_LICENSES=true).
  CC BY-ND                                → skip (unless OA_ALLOW_NO_DERIVATIVES_LICENSES=true).
  Unknown / other                         → skip.

Institutional link-out / publisher-DOI / OpenAthens / Semantic Scholar
institutional-linkout access categories ALWAYS resolve to `not ingestible`
regardless of license (those URLs are link-out only — Phase 2C boundary).

User-uploaded full text is NOT governed by this gate: the user's
attestation is the signal there. The gate returns the existing
`can_ingest_for_summary` decision unchanged for `user_uploaded_full_text`.
"""
from __future__ import annotations

import os
import re
from typing import Optional, TypedDict


LicenseCategory = str  # "commercial_reuse_allowed" | "noncommercial_only" | "restricted" | "unknown" | "other"


_CC_CANONICAL = {
    "cc0": "cc0",
    "publicdomain": "public-domain",
    "public-domain": "public-domain",
    "cc-by": "cc-by",
    "by": "cc-by",
    "cc-by-sa": "cc-by-sa",
    "by-sa": "cc-by-sa",
    "cc-by-nc": "cc-by-nc",
    "by-nc": "cc-by-nc",
    "cc-by-nc-sa": "cc-by-nc-sa",
    "by-nc-sa": "cc-by-nc-sa",
    "cc-by-nc-nd": "cc-by-nc-nd",
    "by-nc-nd": "cc-by-nc-nd",
    "cc-by-nd": "cc-by-nd",
    "by-nd": "cc-by-nd",
}


LICENSE_CATEGORY_BY_NORMALIZED: dict[str, LicenseCategory] = {
    "cc0": "commercial_reuse_allowed",
    "public-domain": "commercial_reuse_allowed",
    "cc-by": "commercial_reuse_allowed",
    "cc-by-sa": "commercial_reuse_allowed",
    "cc-by-nc": "noncommercial_only",
    "cc-by-nc-sa": "noncommercial_only",
    "cc-by-nc-nd": "noncommercial_only",
    "cc-by-nd": "restricted",
}


_NEVER_INGESTIBLE_CATEGORIES = {
    "institutional_openathens",
    "institutional_openurl",
    "semantic_scholar_institutional_linkout",
    "semantic_scholar_institutional",  # legacy alias
    "publisher_doi_link",
    "publisher_url",
    "publisher_html",  # belt-and-braces
}


def _env_flag(key: str, default: bool = False) -> bool:
    raw = (os.environ.get(key) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def normalize_license(raw: Optional[str]) -> Optional[str]:
    """Map a raw license string / URL → our canonical short form, or None."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None
    # Strip URL prefix.
    s = s.replace("https://", "").replace("http://", "")
    s = s.replace("creativecommons.org/licenses/", "").replace("creativecommons.org/", "")
    s = s.replace("creativecommons.org/publicdomain/", "publicdomain")
    # Drop trailing /<version>/ or query strings.
    s = re.sub(r"[/?].*$", "", s)
    s = s.strip("/ ").replace(" ", "-")
    if s in _CC_CANONICAL:
        return _CC_CANONICAL[s]
    # Sometimes we see "cc-by-4.0" or "BY-NC-ND-3.0" — strip versions.
    s = re.sub(r"-?\d+(\.\d+)?$", "", s)
    return _CC_CANONICAL.get(s)


class IngestPermission(TypedDict):
    can_ingest_for_summary: bool
    license: Optional[str]
    license_category: LicenseCategory
    reason: str


def determine_ingest_permission(
    *,
    source: str,
    license: Optional[str],
    access_category: str,
    metadata: Optional[dict] = None,
) -> IngestPermission:
    """Return whether the LLM summarizer may receive text from this source.

    `access_category` examples: ``pmc_oa``, ``europe_pmc_oa``,
    ``oa_html``, ``oa_pdf``, ``publisher_doi_link``,
    ``institutional_openathens``, ``user_uploaded_full_text``.
    """
    # User uploads have a separate attestation path (Phase 2D). The gate
    # returns the existing permission unchanged: callers pass the prior
    # value via metadata["existing_can_ingest"] if needed.
    if access_category == "user_uploaded_full_text":
        existing = bool((metadata or {}).get("existing_can_ingest", True))
        return {
            "can_ingest_for_summary": existing,
            "license": "user_attested",
            "license_category": "other",
            "reason": (
                "User-uploaded full text governed by user attestation; "
                "license gate intentionally not applied."
            ),
        }

    # Institutional / publisher landing pages are NEVER ingestible.
    if access_category in _NEVER_INGESTIBLE_CATEGORIES:
        return {
            "can_ingest_for_summary": False,
            "license": normalize_license(license),
            "license_category": "restricted",
            "reason": (
                "Institutional or publisher link-out; subscription full text is "
                "never ingested. Use Upload Full Text if you have lawful access."
            ),
        }

    norm = normalize_license(license)
    if norm is None:
        return {
            "can_ingest_for_summary": False,
            "license": None,
            "license_category": "unknown",
            "reason": "License could not be confidently determined; skipping ingestion.",
        }
    category = LICENSE_CATEGORY_BY_NORMALIZED.get(norm, "other")
    if category == "commercial_reuse_allowed":
        return {
            "can_ingest_for_summary": True,
            "license": norm,
            "license_category": category,
            "reason": "Permissive Creative Commons license; full text may be summarized.",
        }
    if category == "noncommercial_only":
        if _env_flag("OA_ALLOW_NONCOMMERCIAL_LICENSES", False):
            return {
                "can_ingest_for_summary": True,
                "license": norm,
                "license_category": category,
                "reason": "Noncommercial CC license allowed by OA_ALLOW_NONCOMMERCIAL_LICENSES.",
            }
        return {
            "can_ingest_for_summary": False,
            "license": norm,
            "license_category": category,
            "reason": "Noncommercial license; not ingested by default.",
        }
    if category == "restricted":
        if _env_flag("OA_ALLOW_NO_DERIVATIVES_LICENSES", False):
            return {
                "can_ingest_for_summary": True,
                "license": norm,
                "license_category": category,
                "reason": "No-derivatives CC license allowed by OA_ALLOW_NO_DERIVATIVES_LICENSES.",
            }
        return {
            "can_ingest_for_summary": False,
            "license": norm,
            "license_category": category,
            "reason": "No-derivatives license; not ingested by default.",
        }
    return {
        "can_ingest_for_summary": False,
        "license": norm,
        "license_category": "other",
        "reason": "License not in the allow-list; skipping ingestion.",
    }


def is_article_eligible(article: dict) -> tuple[bool, str]:
    """Pre-flight eligibility check (no network). Returns (eligible, reason)."""
    if article.get("is_retracted"):
        return False, "article_retracted"
    if article.get("is_retraction_notice"):
        return False, "article_retraction_notice"
    if article.get("is_preprint") and not _env_flag("OA_ALLOW_PREPRINTS", False):
        return False, "article_is_preprint"
    pmcid = (article.get("pmcid") or "").strip()
    sids = article.get("source_ids_json") or {}
    in_epmc = str(sids.get("inEPMC", "")).upper() == "Y"
    is_oa_flag = str(sids.get("isOpenAccess", "")).upper() == "Y"
    has_oa_access = any(
        (loc.get("is_oa") and loc.get("access_category") in {
            "pmc_oa", "europe_pmc_oa", "oa_pdf", "oa_html",
        })
        for loc in (article.get("_access_locations") or [])
    )
    if pmcid or in_epmc or is_oa_flag or has_oa_access:
        return True, "eligible"
    return False, "no_oa_signal"
