"""Concept term-set builder shared by the deterministic scorer + LLM rescorer.

The deterministic ``evidence_ranker._relevance`` used to match articles only
against the SPARSE top-level parsed fields (``synonyms`` is often 1-2 terms), so
on-topic papers that phrase the concept differently ("GLP-1RA", "semaglutide",
"NSTEMI", "ACS") scored far too low and were excluded (<60). This collects the
FULL term set — the top-level fields PLUS the rich PICOTT must-have/should-have
synonym lists (drug-class members, MeSH descriptors, acronyms) — de-duplicated.
Pure + side-effect free for unit testing.
"""
from __future__ import annotations

from typing import Any

# PICOTT concept groups whose concept + synonyms feed keyword matching.
_PICOTT_GROUPS = ("must_have_concepts", "should_have_concepts")
_TOP_LEVEL_KEYS = ("key_concepts", "mesh_terms", "synonyms", "drug_names_and_classes")


def concept_term_set(parsed: dict[str, Any]) -> list[str]:
    """Full, order-preserving, case-insensitively de-duplicated concept term
    set for keyword relevance matching."""
    terms: list[str] = []

    for k in _TOP_LEVEL_KEYS:
        v = parsed.get(k)
        if isinstance(v, list):
            terms.extend(str(t) for t in v)

    picott = parsed.get("picott_json") or {}
    if isinstance(picott, dict):
        for grp in _PICOTT_GROUPS:
            for c in (picott.get(grp) or []):
                if not isinstance(c, dict):
                    continue
                if c.get("concept"):
                    terms.append(str(c["concept"]))
                syn = c.get("synonyms")
                if isinstance(syn, list):
                    terms.extend(str(s) for s in syn)

    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        tl = (t or "").strip().lower()
        if tl and tl not in seen:
            seen.add(tl)
            out.append(t)
    return out


__all__ = ["concept_term_set"]
