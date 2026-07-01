"""Phase K5.2C-0 — PubMed-indexed Cochrane Review lane (deterministic helpers).

DETERMINISTIC. NO LLM. NO network. NO scraping. NO full-text ingestion.

Discovery + labeling + bounded ranking support for **PubMed-indexed** Cochrane
systematic reviews, using the already-approved PubMed / NCBI E-utilities path
ONLY. This module never touches the Cochrane Library / Wiley API, never scrapes,
and never ingests or summarizes Cochrane full text.

Three responsibilities (all pure):

  1. ``article_cochrane_provenance(article_doc)`` — does this PubMed-indexed
     record look like a Cochrane systematic review? Detected ONLY from existing
     metadata already stamped by the PubMed connector (journal title / source
     and the validated normalized DOI). Returns the closed-set provenance label
     or ``None``.

  2. ``cochrane_query_applicable(picott)`` — should we emit a Cochrane-scoped
     PubMed query family for this question lane? True for therapy / prevention /
     screening / management / diagnosis / drug-safety / prognosis / etiology
     lanes; False for mechanism / background / cost / generic.

  3. ``cochrane_boost_tier(picott)`` / ``cochrane_boost_points(picott)`` — the
     closed-set, BOUNDED preview ranker boost magnitude for a *directly
     relevant* Cochrane review, gated by question lane. The boost is applied by
     ``picott_ranker`` ONLY when the article carries no relevance cap and all
     must-have anchors matched, so Cochrane status can NEVER override
     wrong-disease / wrong-outcome / wrong-framework / missing-anchor / policy /
     retraction / preprint caps. Caps always dominate.

PREVIEW ONLY. No production activation. No clinician-visible production ranking
change.
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

# --- closed-set constants -------------------------------------------------

# The registry source id (K5.0 registry ``source_name``) that authorises this
# lane. Served via the existing PubMed/NCBI E-utilities path (bibliographic
# metadata + link-out only; NO Cochrane Library API; NO AI ingestion).
COCHRANE_LANE_SOURCE_ID = "cochrane_pubmed_indexed"

# Closed-set per-article provenance label. Explicitly "PubMed-indexed" — NOT
# official Cochrane Library full-text connector coverage.
COCHRANE_LANE_PROVENANCE = "pubmed_indexed_cochrane_review"

# Clinician-facing lane label + tooltip.
COCHRANE_LANE_LABEL = "Cochrane systematic review (PubMed-indexed)"
COCHRANE_LANE_TOOLTIP = (
    "Cochrane systematic review discovered via PubMed (journal 'Cochrane "
    "Database Syst Rev'). Bibliographic metadata + link-out only. Not Cochrane "
    "Library / Wiley full-text connector coverage; no full-text ingestion."
)

# PubMed journal-title-abbreviation tag value used to scope the query family.
COCHRANE_PUBMED_JOURNAL_TAG = '"Cochrane Database Syst Rev"[ta]'

# Cochrane Database of Systematic Reviews — Wiley DOI registrant prefix. Used
# ONLY against the validated normalized DOI already on the article record.
COCHRANE_DOI_PREFIX = "10.1002/14651858"


def _normalize_journal(name: str) -> str:
    n = (name or "").strip().lower()
    n = re.sub(r"\s+", " ", n)
    return n.rstrip(". ")


def article_cochrane_provenance(article_doc: dict[str, Any]) -> Optional[str]:
    """Return ``COCHRANE_LANE_PROVENANCE`` when the PubMed-indexed record looks
    like a Cochrane systematic review; else ``None``.

    Detection uses ONLY metadata already stamped upstream by the PubMed
    connector — the journal title / source and the validated normalized DOI.
    NEVER calls the network. Pure.
    """
    if not isinstance(article_doc, dict):
        return None

    journal = _normalize_journal(
        str(article_doc.get("journal") or article_doc.get("source") or "")
    )
    if journal:
        # The PubMed full journal name is "Cochrane Database of Systematic
        # Reviews"; the indexed abbreviation (``[ta]``) is "Cochrane Database
        # Syst Rev". Accept either, robustly.
        if "cochrane database" in journal and (
            "systematic review" in journal or "syst rev" in journal
        ):
            return COCHRANE_LANE_PROVENANCE

    # Validated DOI only (already normalized by the connector). No URL fetch.
    doi = str(
        article_doc.get("doi_normalized") or article_doc.get("doi") or ""
    ).strip().lower()
    if doi.startswith(COCHRANE_DOI_PREFIX):
        return COCHRANE_LANE_PROVENANCE

    return None


# --- bounded preview ranking support -------------------------------------

# Closed-set boost tier per ``type_of_question``. The query family + boost are
# emitted only for the lanes where a Cochrane systematic review is a directly
# useful evidence type.
_TIER_BY_QUESTION_TYPE: dict[str, str] = {
    # Intervention / prevention / management / dosing / diagnostic accuracy:
    # a Cochrane SR is a top-tier directly-relevant evidence type.
    "therapy": "strong",
    "prevention": "strong",
    "dosing_protocol": "strong",
    "pharmacology": "strong",
    "diagnosis": "strong",
    "clinical_exam": "strong",
    # Prognosis / risk / complication / harm: moderate — a Cochrane *treatment*
    # SR must not dominate a prognosis question, so the boost is smaller and
    # still gated on direct relevance (anchors matched, no caps).
    "etiology_harm": "moderate",
    "prognosis": "moderate",
    "epidemiology": "moderate",
    # Mechanism / background / general overview: weak, cautious.
    "background": "weak",
    # No Cochrane boost / no active Cochrane query for these lanes.
    "guideline": "none",  # the K5.1 PubMed guideline lane already owns this.
    "cost": "none",
    "other": "none",
}

# Closed-set boost points per tier. BOUNDED and small — caps (≤ 20/30/…) and
# the wrong-anchor machinery always dominate.
_BOOST_BY_TIER: dict[str, int] = {
    "strong": 8,
    "drug_safety": 7,
    "moderate": 5,
    "weak": 2,
    "none": 0,
}

# Question subtypes that get a dedicated (moderate/strong) drug-safety tier even
# when ``type_of_question`` resolves to ``etiology_harm``/``pharmacology``.
_DRUG_SAFETY_SUBTYPES = frozenset({"drug_safety"})


def cochrane_boost_tier(picott: dict[str, Any]) -> str:
    """Return the closed-set Cochrane boost tier for the question lane."""
    if not isinstance(picott, dict):
        return "none"
    subtype = str(picott.get("question_subtype") or "").strip().lower()
    if subtype in _DRUG_SAFETY_SUBTYPES:
        return "drug_safety"
    type_of_q = str(picott.get("type_of_question") or "other").strip().lower()
    return _TIER_BY_QUESTION_TYPE.get(type_of_q, "none")


def cochrane_boost_points(picott: dict[str, Any]) -> int:
    """Return the BOUNDED preview boost magnitude (points) for a directly
    relevant Cochrane review on this question lane. 0 for non-applicable lanes.
    """
    return _BOOST_BY_TIER.get(cochrane_boost_tier(picott), 0)


def cochrane_query_applicable(picott: dict[str, Any]) -> bool:
    """Should we emit the PubMed-only ``cochrane_review`` query family for this
    question? True for the strong / drug-safety / moderate lanes; False for the
    weak (mechanism/background) / none (cost/guideline/generic) lanes."""
    return cochrane_boost_tier(picott) in {"strong", "drug_safety", "moderate"}


# --- production gate ------------------------------------------------------
# Preview/dev → the bounded Cochrane preview boost is ACTIVE (clinician-visible
# in preview). Production → the boost is DISABLED (shadow-only) by default; the
# ranker still records the would-be boost in ``cochrane_review_shadow_boost``
# but does NOT apply it to the visible score. Only the explicit
# ``COCHRANE_PROD_BOOST_ALLOWED=true`` flag re-enables application in
# production. This mirrors the L1 soft-boost and ImpactRank shadow production
# gates so K5.2C can NEVER alter production ranking without explicit opt-in.
def cochrane_boost_enabled() -> bool:
    """Return True when the Cochrane preview boost may be APPLIED to the visible
    score. Always True in non-production. In production, only when
    ``COCHRANE_PROD_BOOST_ALLOWED=true``."""
    is_prod = (
        os.environ.get("APP_ENV", "development").strip().lower() == "production"
    )
    if not is_prod:
        return True
    return (
        os.environ.get("COCHRANE_PROD_BOOST_ALLOWED", "false").strip().lower()
        == "true"
    )


__all__ = [
    "COCHRANE_LANE_SOURCE_ID",
    "COCHRANE_LANE_PROVENANCE",
    "COCHRANE_LANE_LABEL",
    "COCHRANE_LANE_TOOLTIP",
    "COCHRANE_PUBMED_JOURNAL_TAG",
    "COCHRANE_DOI_PREFIX",
    "article_cochrane_provenance",
    "cochrane_query_applicable",
    "cochrane_boost_tier",
    "cochrane_boost_points",
    "cochrane_boost_enabled",
]
