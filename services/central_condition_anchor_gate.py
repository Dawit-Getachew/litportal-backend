"""Phase RankPrecision-1 — Central Condition Anchor Precision Gate.

Deterministic (NO LLM, NO network). For a SINGLE-CONDITION cancer question, an
article whose TITLE is about a DIFFERENT specific cancer — and that does NOT
name the target cancer in its title — must not earn high relevance merely
because it mentions the target cancer incidentally in its abstract / full text.
Such an article is capped ``wrong_condition`` (the title's primary disease
focus dominates an incidental body mention).

Genuine multi-cancer registries (no single competing cancer in the title) are
NOT touched here — they fall through to the existing missing/anchor machinery.
"""
from __future__ import annotations

import re
from typing import Any, Optional

PRECISION_CAP = 30

_CANCER_HEAD = re.compile(
    r"\b(?:cancers?|carcinomas?|adenocarcinomas?|tumou?rs?|neoplasms?|"
    r"malignanc(?:y|ies)|sarcomas?|melanomas?|lymphomas?|leuk[ae]mias?|"
    r"myelomas?|glioblastomas?|gliomas?|mesotheliomas?|blastomas?|"
    r"oncolog\w*)\b",
    re.IGNORECASE,
)

# Standalone cancer terms (a site by themselves; no "cancer" head needed).
_STANDALONE_SITE = {
    "melanoma": "skin", "lymphoma": "lymphoma", "leukemia": "leukemia",
    "leukaemia": "leukemia", "myeloma": "myeloma", "glioblastoma": "brain",
    "glioma": "brain", "astrocytoma": "brain", "mesothelioma": "pleura",
    "nephroblastoma": "kidney", "neuroblastoma": "nerve",
    "retinoblastoma": "eye", "hepatoma": "liver",
}

# Site token → canonical site key (only counted as a cancer site when a cancer
# head is also present in the title; see ``_extract_title_sites``).
_SITE_TOKEN = {
    "prostate": "prostate", "prostatic": "prostate",
    "colon": "colorectal", "colonic": "colorectal", "colorectal": "colorectal",
    "rectal": "colorectal", "rectum": "colorectal", "bowel": "colorectal",
    "breast": "breast", "mammary": "breast",
    "lung": "lung", "bronchogenic": "lung", "nsclc": "lung", "sclc": "lung",
    "pancreatic": "pancreas", "pancreas": "pancreas",
    "gastric": "stomach", "stomach": "stomach",
    "ovarian": "ovary", "ovary": "ovary",
    "cervical": "cervix", "cervix": "cervix",
    "endometrial": "uterus", "uterine": "uterus", "uterus": "uterus",
    "bladder": "bladder", "urothelial": "bladder",
    "renal": "kidney", "kidney": "kidney",
    "hepatocellular": "liver", "liver": "liver", "hepatic": "liver",
    "esophageal": "esophagus", "oesophageal": "esophagus", "esophagus": "esophagus",
    "thyroid": "thyroid",
    "testicular": "testis", "testis": "testis",
    "laryngeal": "head_neck", "oropharyngeal": "head_neck",
    "nasopharyngeal": "head_neck", "hypopharyngeal": "head_neck",
    "cutaneous": "skin",
}


def _word(token: str, text: str) -> bool:
    return bool(re.search(rf"\b{re.escape(token)}\b", text))


def _extract_title_sites(title: str) -> set[str]:
    """Canonical cancer sites named in a title. Site tokens (prostate, colon,
    …) count only when a cancer head is also present; standalone terms
    (melanoma, glioma, …) always count."""
    t = (title or "").lower()
    sites: set[str] = set()
    for term, site in _STANDALONE_SITE.items():
        if _word(term, t):
            sites.add(site)
    if _CANCER_HEAD.search(t):
        for token, site in _SITE_TOKEN.items():
            if _word(token, t):
                sites.add(site)
    return sites


def _target_cancer_site(picott: dict[str, Any]) -> Optional[str]:
    """Target cancer site for a single-condition cancer question, else None."""
    anchor = (picott.get("prognosis_stage_cancer_condition_anchor") or "").strip().lower()
    candidates: list[str] = []
    if anchor:
        candidates.append(anchor)
    for mh in picott.get("must_have_concepts") or []:
        if isinstance(mh, dict) and (mh.get("role") or "").lower() in {"condition", "population"}:
            candidates.append((mh.get("concept") or "").strip().lower())
    for c in candidates:
        if not c:
            continue
        # must itself be a cancer concept
        if not (_CANCER_HEAD.search(c) or any(s in c for s in _STANDALONE_SITE)):
            continue
        for term, site in _STANDALONE_SITE.items():
            if _word(term, c):
                return site
        for token, site in _SITE_TOKEN.items():
            if _word(token, c):
                return site
    return None


def evaluate(title: str, picott: dict[str, Any]) -> dict[str, Any]:
    """Return precision-gate verdict for one article TITLE in the context of a
    (possibly cancer-) single-condition question.

    Fires (``wrong_condition``) when the title names a DIFFERENT specific
    cancer and does NOT name the target cancer.
    """
    target = _target_cancer_site(picott)
    if not target:
        return {"fires": False, "reason": None, "target_site": None, "title_sites": []}
    title_sites = _extract_title_sites(title)
    other = title_sites - {target}
    fires = bool(other) and (target not in title_sites)
    return {
        "fires": fires,
        "reason": "wrong_condition" if fires else None,
        "target_site": target,
        "title_sites": sorted(title_sites),
        "other_sites": sorted(other),
    }


__all__ = ["evaluate", "PRECISION_CAP"]
