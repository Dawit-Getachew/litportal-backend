"""Phase 6G.4P — ProcedureOntologyMapperService.

Deterministic mapper that resolves a procedure phrase (e.g. "aortic
valve surgery", "hip replacement", "PCI") to a closed-set
``ProcedureMapping``. Distinct from ``PrimaryClinicalEntityAnchorService``:
this service is COMPOSED into the anchor service for the post-procedure
management routing path. It does NOT replace any existing service.

Resolution order (deterministic-first):

1. Pack lookup over every entry in ``services.procedure_packs.all_entries()``.
   Match priorities: canonical_concept → synonym → acronym.
2. Curated MeSH resolver: only when the descriptor is in the existing
   ``services.mesh.mesh_resolver``. Never hallucinated.
3. UMLS-like local mapping — skipped cleanly when not available.
4. ``llm_repair_shadow`` — SHADOW-ONLY behind
   ``PROCEDURE_ANCHOR_CRITIC_ENABLED=true``. See
   ``services.procedure_anchor_critic``.

NO LLM. NO external API. NO PHI. NO fabrication.
"""
from __future__ import annotations

import logging
import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from services.procedure_packs import (
    MANAGEMENT_DOMAINS,
    PROCEDURE_FAMILIES,
    all_entries,
)

logger = logging.getLogger("clinevidence.procedure_ontology_mapper")


ProcedureMappingSource = Literal[
    "procedure_pack",
    "mesh_curated",
    "umls_like",
    "llm_repair_shadow",
]


class ProcedureMapping(BaseModel):
    """Closed-set procedure ontology mapping.

    ``extra="forbid"`` prevents accidental key drift.
    """

    model_config = ConfigDict(extra="forbid")

    original_phrase: str
    canonical_concept: str
    entity_type: Literal["procedure_or_surgery"] = "procedure_or_surgery"
    procedure_family: str
    specialty: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    related_but_distinct: list[str] = Field(default_factory=list)
    management_domains: list[str] = Field(default_factory=list)
    role_hint: Literal["procedure_context"] = "procedure_context"
    mesh_terms_curated: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    source: ProcedureMappingSource
    rationale: str = ""


# Helper — case-insensitive, word-boundary match for short tokens.
def _phrase_contains(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    n = needle.strip().lower()
    h = haystack.strip().lower()
    if not n:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", h))


def _score_entry(phrase: str, entry: dict) -> tuple[float, str, str]:
    """Score a pack entry against ``phrase``. Returns
    ``(score, matched_token, match_kind)`` where match_kind is one of
    ``canonical`` / ``synonym`` / ``acronym``.

    Score components:
      * canonical phrase exact substring → 1.0
      * synonym exact substring → 0.92
      * acronym whole-word match → 0.90
      * canonical token-set overlap ≥0.6 → 0.78
    """
    if not phrase or not entry:
        return 0.0, "", ""
    p = phrase.strip().lower()
    canonical = (entry.get("canonical_concept") or "").strip()
    if canonical and _phrase_contains(p, canonical):
        return 1.0, canonical, "canonical"
    for syn in entry.get("synonyms") or []:
        if isinstance(syn, str) and _phrase_contains(p, syn):
            return 0.92, syn, "synonym"
    for ac in entry.get("acronyms") or []:
        if not isinstance(ac, str):
            continue
        # Acronyms must match as standalone tokens (avoid "tap" matching "PCI"-style).
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(ac)}(?![A-Za-z0-9])", phrase):
            return 0.9, ac, "acronym"
    # Conservative token-overlap fallback.
    if canonical:
        canonical_tokens = {t for t in re.findall(r"[a-z0-9]+", canonical.lower()) if t}
        phrase_tokens = {t for t in re.findall(r"[a-z0-9]+", p) if t}
        if canonical_tokens and phrase_tokens:
            overlap = len(canonical_tokens & phrase_tokens) / max(
                len(canonical_tokens), 1,
            )
            if overlap >= 0.6:
                return 0.78, canonical, "canonical"
    return 0.0, "", ""


class ProcedureOntologyMapperService:
    """Closed-set procedure ontology mapper. Stateless / pure."""

    def resolve_procedure(self, phrase: str) -> Optional[ProcedureMapping]:
        if not phrase:
            return None
        # Step 1 — pack lookup. Score ties are broken in favor of the
        # entry with the LONGEST canonical_concept so that more
        # specific procedures (e.g., "colonoscopy with polypectomy")
        # outrank their less specific parents (e.g., "colonoscopy")
        # when both are substring-present in the phrase.
        best: tuple[float, dict, str, str] | None = None
        for entry in all_entries():
            if not isinstance(entry, dict):
                continue
            score, _matched, kind = _score_entry(phrase, entry)
            if score <= 0:
                continue
            if best is None:
                best = (score, entry, _matched, kind)
                continue
            cur_canonical_len = len((best[1].get("canonical_concept") or ""))
            new_canonical_len = len((entry.get("canonical_concept") or ""))
            if score > best[0] or (
                score == best[0] and new_canonical_len > cur_canonical_len
            ):
                best = (score, entry, _matched, kind)
        if best is not None:
            score, entry, matched, kind = best
            fam = (entry.get("procedure_family") or "").strip()
            if fam not in PROCEDURE_FAMILIES:
                # Defensive: never return a non-closed-set family. Drop.
                logger.debug(
                    "procedure_pack_invalid_family",
                    extra={"family": fam, "pack": entry.get("_pack")},
                )
                fam = ""
            domains = [
                d for d in (entry.get("management_domains") or [])
                if isinstance(d, str) and d in MANAGEMENT_DOMAINS
            ]
            mesh = [
                m for m in (entry.get("mesh_terms_curated") or [])
                if isinstance(m, str) and m
            ]
            return ProcedureMapping(
                original_phrase=phrase.strip(),
                canonical_concept=entry.get("canonical_concept") or matched,
                procedure_family=fam or "general_abdominal_surgery",
                specialty=[
                    s for s in (entry.get("specialty") or [])
                    if isinstance(s, str) and s
                ],
                synonyms=[
                    s for s in (entry.get("synonyms") or [])
                    if isinstance(s, str) and s
                ],
                related_but_distinct=[
                    r for r in (entry.get("related_but_distinct") or [])
                    if isinstance(r, str) and r
                ],
                management_domains=domains,
                mesh_terms_curated=mesh,
                confidence=float(score),
                source="procedure_pack",
                rationale=f"pack_match:{kind}",
            )
        # Step 2 — curated MeSH (only if the existing resolver has it).
        try:
            from services.mesh import mesh_resolver  # local import
            res = mesh_resolver.resolve(phrase.strip())
        except Exception:  # noqa: BLE001
            res = None
        if res is not None and getattr(res, "mesh_descriptor", None):
            # Best-effort family inference for MeSH hits. Without a
            # pack entry we cannot confidently assign a family →
            # return None (caller falls back to conservative anchor).
            return None
        # Step 3 / 4 — skip cleanly.
        return None


procedure_ontology_mapper = ProcedureOntologyMapperService()


__all__ = [
    "ProcedureMapping",
    "ProcedureOntologyMapperService",
    "procedure_ontology_mapper",
]
