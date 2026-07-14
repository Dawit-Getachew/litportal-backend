"""AI clinical-synonym expansion — default-OFF, flag-gated.

Research-backed (SIGIR 2023 arXiv:2302.03495; JMIR 2020 PMC7303830; ontology-
guided BMQExpander arXiv:2508.11784): the "No results found" dead-end on an
over-specific anchor ("stage 4 prostate cancer") is fixed by expanding that
anchor into a controlled-vocabulary OR-group of synonyms / staged variants, so
recall-first retrieval actually finds the right papers instead of matching a
single literal phrase. The terms are injected as OR-within-concept ``synonyms``
on the existing ``must_have_concepts`` — which ``concept_block_query_builder``
already OR-expands within a block and AND-gates across blocks, so precision is
preserved (SIGIR/systematic-review structure).

Entirely gated by ``AI_QUERY_EXPANSION_ENABLED`` (default false). When off, or
when the LLM is not configured, ``expand_concept_synonyms`` returns ``[]`` — a
no-op — so this module can ship dormant and be A/B-tested via the env flag.
"""
from __future__ import annotations

import os

from services.llm_provider import llm_provider


def is_enabled() -> bool:
    return os.environ.get("AI_QUERY_EXPANSION_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


_SYSTEM = (
    "You are a clinical search librarian building a PubMed / Europe PMC search. "
    "Given ONE clinical concept, return the controlled-vocabulary synonyms and "
    "clinical variants a comprehensive literature search should OR together — so "
    "an over-specific, staged, or lay phrasing still retrieves the right papers.\n"
    "\n"
    "Rules:\n"
    "- INCLUDE: the core disease/drug/procedure with the stage/qualifier removed "
    "(e.g. 'stage 4 prostate cancer' -> 'prostate cancer'); standard MeSH-style "
    "names; common abbreviations/acronyms; and clinically-equivalent staged "
    "variants (e.g. 'stage 4 X' -> 'metastatic X', 'advanced X', 'stage IV X'; "
    "for prostate specifically also 'castration-resistant prostate cancer', "
    "'mCRPC').\n"
    "- DO NOT invent conditions, drugs, or facts. Only real, widely-used clinical "
    "terms. If unsure, return fewer terms.\n"
    "- DO NOT include question-intent words (treatment, prognosis, diagnosis, "
    "management) or generic words (patients, adults, therapy, disease).\n"
    "- 4 to 8 terms, most useful first. No duplicates of the input.\n"
    "\n"
    'Return JSON: {"synonyms": ["term1", "term2", ...]}.'
)


async def expand_concept_synonyms(concept: str, max_terms: int = 8) -> list[str]:
    """Return LLM-generated clinical synonyms for ``concept`` (OR-group members),
    or ``[]`` when disabled / unconfigured / on any failure. Never raises."""
    if not is_enabled() or not llm_provider.is_configured():
        return []
    concept = (concept or "").strip()
    if not concept or len(concept) > 120:
        return []
    try:
        result = await llm_provider.complete_json(
            _SYSTEM, concept, schema_hint={"synonyms": ["string"]}, max_tokens=300,
        )
    except Exception:  # noqa: BLE001 — never propagate to the pipeline
        return []
    if not isinstance(result, dict):
        return []
    raw = result.get("synonyms") or result.get("terms") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen = {concept.strip().lower()}
    for s in raw:
        s = str(s).strip()
        sl = s.lower()
        if s and sl not in seen and 1 < len(s) <= 80:
            seen.add(sl)
            out.append(s)
    return out[:max_terms]
