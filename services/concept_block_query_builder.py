"""Phase 5A — Concept-block query builder.

Generates 6 query families per enabled retrieval source (PubMed + Europe PMC):

  - high_precision        — AND of mandatory anchors with at least 3 blocks
  - broad_recall          — OR over all anchors + ranking concepts
  - study_type_filtered   — high_precision narrowed by [Publication Type]
  - guideline_review      — narrowed to guideline / SR / MA / review
  - recent_title_abstract — high_precision restricted to last 5y, [tiab]
  - rescue_query          — narrowed must-haves + key synonyms ONLY
                             (only emitted on demand from the evaluator)

Each output ``ConceptBlockQuery`` carries:
  - source, query_family, query_string
  - concept blocks used
  - mandatory anchors
  - optional terms
  - rationale
  - max_results quota

Per-family default quotas come from environment variables so deployments
can tune them without code changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from services.study_type_router import study_type_router
# Phase 5D — apply the same STOP_PHRASES + STOP_TOKENS filter to legacy
# queries that the new BooleanQueryCompilerService applies. Otherwise
# prose phrases like "difference between" and single-token stopwords
# like "tell" / "about" / "me" leak verbatim into Europe PMC
# ``TITLE_ABS`` expressions via the legacy builder.
from services.query_stop_phrases import (  # noqa: E402
    strip_stop_phrases,
    is_query_term_meaningful,
)

# ---------------------------------------------------------------------------
# Default per-family quotas (overridable via env vars).
# ---------------------------------------------------------------------------
def _quota(env_var: str, default: int) -> int:
    try:
        return int(os.environ.get(env_var, str(default)))
    except (TypeError, ValueError):
        return default


DEFAULT_QUOTAS = {
    "high_precision": _quota("Q_QUOTA_HIGH_PRECISION", 50),
    "broad_recall": _quota("Q_QUOTA_BROAD_RECALL", 75),
    "study_type_filtered": _quota("Q_QUOTA_STUDY_TYPE_FILTERED", 50),
    "guideline_review": _quota("Q_QUOTA_GUIDELINE_REVIEW", 50),
    "recent_title_abstract": _quota("Q_QUOTA_RECENT_TIAB", 50),
    "rescue_query": _quota("Q_QUOTA_RESCUE", 50),
}

# Max-length safeguards mirror the legacy planner.
MAX_PHRASE_CHARS = 100
MAX_QUERY_LEN = 1500


@dataclass
class ConceptBlockQuery:
    source: str
    query_family: str
    query_string: str
    concept_blocks: dict[str, list[str]]
    mandatory_anchors: list[str]
    optional_terms: list[str]
    rationale: str
    max_results: int
    # Legacy column for BC with the existing ``search_queries`` collection.
    query_type: str = field(default="")

    def to_query_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "query_type": self.query_type or self.query_family,
            "query_family": self.query_family,
            "query_string": self.query_string,
            "concept_blocks_json": self.concept_blocks,
            "mandatory_anchors": self.mandatory_anchors,
            "optional_terms": self.optional_terms,
            "rationale": self.rationale,
            "max_results": self.max_results,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _trim(s: str) -> str:
    # Phase 5D — strip STOP_PHRASES before length-trim so prose noise
    # ("difference between", "what is the dose of", ...) never leaks into
    # a TITLE_ABS / [tiab] expression. Applies to ALL legacy families
    # (high_precision, broad_recall, study_type_filtered, guideline_review,
    # recent_title_abstract, rescue_query) for both PubMed and Europe PMC.
    s = strip_stop_phrases(s or "").strip().strip(",;:.!?")
    if not s:
        return ""
    # Phase 5D — drop single-token stopwords ("tell", "about", "me",
    # "what", etc.) so a "tell me about diabetes" question can't leak
    # them into ``TITLE_ABS:"tell"`` style emissions. Multi-word phrases
    # that still contain at least one meaningful token survive.
    if not is_query_term_meaningful(s):
        return ""
    return s[:MAX_PHRASE_CHARS] if len(s) > MAX_PHRASE_CHARS else s


def _truncate(q: str) -> str:
    return q if len(q) <= MAX_QUERY_LEN else q[:MAX_QUERY_LEN]


def _ddp_quote(term: str) -> str:
    """Quote a phrase, escaping inner quotes."""
    t = _trim(term)
    if not t:
        return ""
    return '"' + t.replace('"', "") + '"'


def _epmc_ta(term: str) -> str:
    q = _ddp_quote(term)
    return f"TITLE_ABS:{q}" if q else ""


def _pubmed_tiab(term: str) -> str:
    q = _ddp_quote(term)
    return f"{q}[tiab]" if q else ""


def _or_block(terms: list[str], emitter) -> str:
    cleaned = [t for t in (emitter(x) for x in terms) if t]
    if not cleaned:
        return ""
    return "(" + " OR ".join(cleaned) + ")"


def _and_block(blocks: list[str]) -> str:
    blocks = [b for b in blocks if b]
    if not blocks:
        return ""
    return "(" + " AND ".join(blocks) + ")"


def _dedupe_preserve(seq):
    out, seen = [], set()
    for s in seq:
        if not s:
            continue
        k = s.lower().strip()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Block extraction from a PICOTT dict.
# ---------------------------------------------------------------------------
def _build_blocks(picott: dict[str, Any]) -> dict[str, list[str]]:
    """Extract concept blocks from PICOTT. Each block is the canonical
    concept + its expanded synonyms — used as an OR group."""
    pico = picott.get("pico") or {}
    must = picott.get("must_have_concepts") or []
    blocks: dict[str, list[str]] = {
        "population": list(pico.get("population") or []),
        "intervention": [],
        "comparator": list(pico.get("comparator_or_reference_standard") or []),
        "outcome": list(pico.get("outcomes_or_target_diagnosis") or []),
        "condition": [],
        "context": [],
    }
    # Merge must-have concepts by role into the matching block (concept +
    # synonyms together).
    for mh in must:
        role = (mh.get("role") or "other").lower()
        concept = mh.get("concept") or ""
        terms = [concept] + list(mh.get("synonyms") or [])
        if role == "intervention":
            blocks["intervention"].extend(terms)
        elif role == "condition":
            blocks["condition"].extend(terms)
        elif role == "outcome":
            blocks["outcome"].extend(terms)
        elif role == "comparator":
            blocks["comparator"].extend(terms)
        elif role == "population":
            blocks["population"].extend(terms)
        elif role == "context":
            blocks["context"].extend(terms)
    # Add ranking concepts (context-role only) into context block.
    for rc in (picott.get("ranking_concepts") or []):
        if (rc.get("role") or "").lower() == "context":
            blocks["context"].append(rc.get("concept") or "")
            blocks["context"].extend(rc.get("synonyms") or [])
        # Drug ranking concepts → also push to intervention block as
        # optional terms (we'll dedupe later).
        elif (rc.get("role") or "").lower() == "intervention":
            blocks["intervention"].append(rc.get("concept") or "")
    # Dedupe each block.
    for k in list(blocks.keys()):
        blocks[k] = _dedupe_preserve(blocks[k])
    return blocks


def _mandatory_anchor_terms(picott: dict[str, Any]) -> list[str]:
    """The canonical concept names of must-haves (used in queries to AND-
    together as the high-precision core)."""
    return _dedupe_preserve(
        m.get("concept") or "" for m in (picott.get("must_have_concepts") or [])
    )


# ---------------------------------------------------------------------------
# Per-source emitters.
# ---------------------------------------------------------------------------
def _epmc_pub_type_block(type_of_q: str) -> str:
    pathway = study_type_router.pathway(type_of_q)
    pts = pathway.europe_pmc_pub_types or []
    if not pts:
        return ""
    return "(" + " OR ".join(f'PUB_TYPE:{_ddp_quote(pt)}' for pt in pts) + ")"


def _pubmed_pub_type_block(type_of_q: str) -> str:
    pathway = study_type_router.pathway(type_of_q)
    pts = pathway.pubmed_pub_types or []
    if not pts:
        return ""
    return "(" + " OR ".join(f'"{pt}"[Publication Type]' for pt in pts) + ")"


_EPMC_GUIDELINE_PT = '(PUB_TYPE:"practice guideline" OR PUB_TYPE:"guideline" OR PUB_TYPE:"systematic review" OR PUB_TYPE:"meta analysis" OR PUB_TYPE:"review")'
_PUBMED_GUIDELINE_PT = '("Practice Guideline"[Publication Type] OR "Guideline"[Publication Type] OR "Systematic Review"[Publication Type] OR "Meta-Analysis"[Publication Type] OR "Review"[Publication Type])'
_EPMC_EXCL = 'NOT SRC:PPR NOT PUB_TYPE:"preprint"'
_PUBMED_EXCL = 'NOT "preprint"[Publication Type]'


def _build_for_source(
    *,
    source: str,
    picott: dict[str, Any],
    blocks: dict[str, list[str]],
    rescue_must_haves: list[str] | None = None,
    rescue_only: bool = False,
) -> list[ConceptBlockQuery]:
    type_of_q = (picott.get("type_of_question") or "other").lower()
    emitter = _epmc_ta if source == "europe_pmc" else _pubmed_tiab
    excl = _EPMC_EXCL if source == "europe_pmc" else _PUBMED_EXCL
    guideline_pt = (_EPMC_GUIDELINE_PT if source == "europe_pmc"
                    else _PUBMED_GUIDELINE_PT)
    pub_type_block = (_epmc_pub_type_block(type_of_q) if source == "europe_pmc"
                      else _pubmed_pub_type_block(type_of_q))
    if source == "pubmed":
        date_recent = 'AND ("last 5 years"[PDat])'
    else:
        date_recent = 'AND (FIRST_PDATE:[NOW-5YEARS TO NOW])'

    queries: list[ConceptBlockQuery] = []
    mandatory_anchors = _mandatory_anchor_terms(picott)

    # ---- Rescue query (only when requested by the evaluator) ----
    if rescue_only:
        synonyms = []
        for mh in (picott.get("must_have_concepts") or []):
            synonyms.extend(mh.get("synonyms") or [])
        rescue_terms = _dedupe_preserve(
            (rescue_must_haves or []) + mandatory_anchors + synonyms[:6]
        )
        if not rescue_terms:
            return queries
        # Build per-anchor OR groups for each role.
        per_role_groups: list[str] = []
        for role in ("intervention", "condition"):
            if blocks.get(role):
                per_role_groups.append(_or_block(blocks[role], emitter))
        if blocks.get("context"):
            per_role_groups.append(_or_block(blocks["context"][:6], emitter))
        core = _and_block(per_role_groups) or _or_block(rescue_terms, emitter)
        q = _truncate(f"{core} {excl}".strip())
        queries.append(ConceptBlockQuery(
            source=source,
            query_family="rescue_query",
            query_type="rescue_query",
            query_string=q,
            concept_blocks={k: blocks.get(k, []) for k in ("intervention", "condition", "context")},
            mandatory_anchors=mandatory_anchors,
            optional_terms=blocks.get("context", [])[:6],
            rationale=(
                "Rescue query: AND of must-have intervention + condition + "
                "key context terms (synonym-expanded). Triggered by retrieval "
                "evaluator due to missing anchor coverage in top results."
            ),
            max_results=DEFAULT_QUOTAS["rescue_query"],
        ))
        return queries

    # ---- high_precision: AND of must-haves + at-least intervention OR ----
    must_or_blocks: list[str] = []
    for role in ("intervention", "condition", "population"):
        terms = blocks.get(role) or []
        if terms:
            must_or_blocks.append(_or_block(terms, emitter))
    # Context as optional supplement only if we have at least 2 mandatory groups.
    if blocks.get("context") and len(must_or_blocks) >= 2:
        must_or_blocks.append(_or_block(blocks["context"][:8], emitter))
    high_precision_core = _and_block(must_or_blocks)
    if high_precision_core:
        queries.append(ConceptBlockQuery(
            source=source,
            query_family="high_precision",
            query_type="high_precision",
            query_string=_truncate(f"{high_precision_core} {excl}".strip()),
            concept_blocks=dict(blocks),
            mandatory_anchors=mandatory_anchors,
            optional_terms=blocks.get("context", [])[:8],
            rationale=(
                "Concept-block AND across mandatory roles "
                "(intervention/condition/population). Preprints excluded."
            ),
            max_results=DEFAULT_QUOTAS["high_precision"],
        ))

    # ---- broad_recall: OR over ALL anchors + ranking concepts ----
    all_recall = (
        blocks.get("intervention", [])
        + blocks.get("condition", [])
        + blocks.get("outcome", [])
        + blocks.get("context", [])[:6]
    )
    recall_block = _or_block(_dedupe_preserve(all_recall), emitter)
    if recall_block:
        queries.append(ConceptBlockQuery(
            source=source,
            query_family="broad_recall",
            query_type="broad_recall",
            query_string=_truncate(f"{recall_block} {excl}".strip()),
            concept_blocks=dict(blocks),
            mandatory_anchors=mandatory_anchors,
            optional_terms=all_recall,
            rationale=(
                "Broad TITLE_ABS OR across anchors + ranking concepts. "
                "Used for recall — does NOT boost articles missing must-haves."
            ),
            max_results=DEFAULT_QUOTAS["broad_recall"],
        ))

    # ---- study_type_filtered: high_precision AND pub-type ----
    if high_precision_core and pub_type_block:
        queries.append(ConceptBlockQuery(
            source=source,
            query_family="study_type_filtered",
            query_type="study_type_filtered",
            query_string=_truncate(
                f"{high_precision_core} AND {pub_type_block} {excl}".strip()
            ),
            concept_blocks=dict(blocks),
            mandatory_anchors=mandatory_anchors,
            optional_terms=[],
            rationale=(
                f"High-precision core narrowed to study types for "
                f"type_of_question='{type_of_q}'."
            ),
            max_results=DEFAULT_QUOTAS["study_type_filtered"],
        ))

    # ---- guideline_review: high_precision AND guideline pub-types ----
    if high_precision_core and guideline_pt:
        queries.append(ConceptBlockQuery(
            source=source,
            query_family="guideline_review",
            query_type="guideline_review",
            query_string=_truncate(
                f"{high_precision_core} AND {guideline_pt} {excl}".strip()
            ),
            concept_blocks=dict(blocks),
            mandatory_anchors=mandatory_anchors,
            optional_terms=[],
            rationale=(
                "High-precision core narrowed to guideline / SR / MA / review "
                "publication types."
            ),
            max_results=DEFAULT_QUOTAS["guideline_review"],
        ))

    # ---- recent_title_abstract: last 5y high_precision ----
    if high_precision_core:
        queries.append(ConceptBlockQuery(
            source=source,
            query_family="recent_title_abstract",
            query_type="recent_title_abstract",
            query_string=_truncate(
                f"{high_precision_core} {date_recent} {excl}".strip()
            ),
            concept_blocks=dict(blocks),
            mandatory_anchors=mandatory_anchors,
            optional_terms=[],
            rationale=(
                "High-precision core restricted to publications from the last "
                "5 years on title/abstract terms."
            ),
            max_results=DEFAULT_QUOTAS["recent_title_abstract"],
        ))

    return queries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class ConceptBlockQueryBuilderService:
    def build(
        self,
        picott: dict[str, Any],
        *,
        sources: list[str] | None = None,
    ) -> list[ConceptBlockQuery]:
        sources = sources or ["europe_pmc"]
        blocks = _build_blocks(picott)
        out: list[ConceptBlockQuery] = []
        for src in sources:
            out.extend(_build_for_source(source=src, picott=picott, blocks=blocks))
        return out

    def build_rescue(
        self,
        picott: dict[str, Any],
        *,
        sources: list[str] | None = None,
        rescue_must_haves: list[str] | None = None,
    ) -> list[ConceptBlockQuery]:
        sources = sources or ["europe_pmc"]
        blocks = _build_blocks(picott)
        out: list[ConceptBlockQuery] = []
        for src in sources:
            out.extend(
                _build_for_source(
                    source=src, picott=picott, blocks=blocks,
                    rescue_must_haves=rescue_must_haves, rescue_only=True,
                )
            )
        return out


concept_block_query_builder = ConceptBlockQueryBuilderService()
