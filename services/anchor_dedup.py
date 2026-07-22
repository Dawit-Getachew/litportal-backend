"""Redundant-anchor collapse for the LLM-resolver PICOTT merge.

The clinical-context resolver sometimes emits a concept that is really a
narrower/overlapping duplicate of an anchor already present — e.g. the parent
molecule "Glucagon-Like Peptide 1" alongside the drug-class intervention anchor
"GLP-1 receptor agonists". The merge in services/pipeline.py only deduped by
EXACT concept string, so the duplicate became a THIRD required anchor and caused
genuinely on-topic articles to be capped/rejected for "missing" it.

`concept_is_redundant()` catches this deterministically: a candidate concept is
redundant when its significant tokens are a subset of some existing anchor's
terms (concept + synonyms). Conservative — requires >=2 significant tokens and
FULL token containment, so distinct concepts ("heart failure" vs "GLP-1 receptor
agonists") are never collapsed. Pure + side-effect free for unit testing.

Default-ON; set PICOTT_ANCHOR_DEDUP=false to disable.
"""
from __future__ import annotations

import os
import re


def anchor_dedup_enabled() -> bool:
    return os.environ.get("PICOTT_ANCHOR_DEDUP", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


def significant_tokens(text: str) -> set[str]:
    """Lowercased alphanumeric tokens >=3 chars, excluding pure numbers."""
    toks = re.split(r"[^a-z0-9]+", (text or "").lower())
    return {t for t in toks if len(t) >= 3 and not t.isdigit()}


def _termset_tokens(terms) -> set[str]:
    out: set[str] = set()
    for t in terms or []:
        out |= significant_tokens(t)
    return out


def concept_is_redundant(concept: str, existing_termsets) -> bool:
    """True when ``concept`` is a narrower/overlapping duplicate of some existing
    anchor.

    ``existing_termsets`` is an iterable of term lists — one per existing anchor,
    each being ``[concept] + synonyms``. Returns True iff the candidate has at
    least 2 significant tokens and ALL of them appear within a single existing
    anchor's tokens (full containment)."""
    ctoks = significant_tokens(concept)
    if len(ctoks) < 2:
        return False
    for terms in existing_termsets:
        etoks = _termset_tokens(terms)
        if etoks and ctoks <= etoks:
            return True
    return False


__all__ = [
    "anchor_dedup_enabled",
    "significant_tokens",
    "concept_is_redundant",
]
