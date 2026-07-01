"""SearchPlanningService — Phase 2B.

Generates query variants for each enabled retrieval source. Each connector
receives queries tailored to its own search syntax:

* Europe PMC — TITLE_ABS phrases, PUB_TYPE clauses, ``NOT SRC:PPR`` exclusion
* PubMed — NCBI fielded syntax with [tiab], [pt] tags, preprint excluded via
  ``NOT preprint[Publication Type]``

The 4 query variants per source are conceptually identical (high_recall,
high_precision, evidence_filtered, guideline_focused) so the downstream
pipeline can attribute evidence by query_type across sources.

All variants exclude preprints at-source. Per-phrase length is capped to
avoid Europe PMC / NCBI errors. Final query length is truncated.

Phase 5D Fix 4 (round 4) — every emitted term is now passed through the
STOP_PHRASES + STOP_TOKENS filter so prose noise like ``"tell"`` /
``"about"`` / ``"difference between"`` cannot leak into a
``TITLE_ABS:"…"`` or ``"…"[tiab]`` expression. The filter mirrors the
one wired into the Phase 5A ``concept_block_query_builder`` and the
Phase 5D ``boolean_query_compiler`` — see
``services/query_stop_phrases.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from services.query_stop_phrases import (
    is_query_term_meaningful,
    strip_stop_phrases,
)

logger = logging.getLogger("clinevidence.search_planning")

MAX_TERMS = 12
MAX_PHRASE_CHARS = 100
MAX_QUERY_LEN = 1500


@dataclass
class PlannedQuery:
    query_type: str
    query_string: str
    rationale: str
    source: str = "europe_pmc"


# ============================================================================
# Shared helpers
# ============================================================================
# Closed-set log key for empty-block skips (Phase 5D Fix 4).
EMPTY_BLOCK_LOG = "legacy_planner_block_empty_after_filter"


def _strip_edge_stopwords(s: str) -> str:
    """Trim leading/trailing single-token stopwords from a multi-token
    phrase. E.g. "between high hemoglobin" → "high hemoglobin",
    "the diabetes" → "diabetes". Pure stopword phrases reduce to "".

    Internal stopwords (in the middle of a phrase) are intentionally
    preserved because they're often grammatically required in valid
    medical terms (e.g., ``"acute infarction of the myocardium"``).
    """
    if not s:
        return ""
    # Import inline to avoid circulars and to share the canonical set.
    from services.query_stop_phrases import STOP_TOKENS

    tokens = s.split()
    while tokens and tokens[0].lower().strip('"\'') in STOP_TOKENS:
        tokens.pop(0)
    while tokens and tokens[-1].lower().strip('"\'') in STOP_TOKENS:
        tokens.pop()
    return " ".join(tokens)


def _trim_phrase(s: str) -> str:
    # Phase 5D Fix 4 — apply the same STOP_PHRASES + STOP_TOKENS filter
    # the Phase 5A and Phase 5D builders apply. Multi-step:
    #   1. strip multi-word STOP_PHRASES ("difference between",
    #      "what is the dose of", "tell me about", ...)
    #   2. strip leading/trailing single-token stopwords ("between high"
    #      → "high", "the diabetes" → "diabetes")
    #   3. drop the term entirely if it's not meaningful (single token
    #      stopword like "tell" / "about" / "me")
    # Without this, the legacy planner emits e.g. ``TITLE_ABS:"tell"``
    # or ``"difference between"[tiab]`` — see
    # ``tests/test_search_planning_stop_filter.py``.
    s = (s or "").strip().strip(",;:.!?")
    s = strip_stop_phrases(s)
    s = _strip_edge_stopwords(s)
    if not s or not is_query_term_meaningful(s):
        return ""
    if len(s) <= MAX_PHRASE_CHARS:
        return s
    return s[:MAX_PHRASE_CHARS].rstrip()


def _clean_terms(terms: list[str]) -> list[str]:
    """Run a list of candidate terms through ``_trim_phrase`` and drop
    the falsy ones. Deduplicates case-insensitively while preserving
    order. Applied at planner *assembly* time so every downstream
    builder (``_epmc_*``, ``_pubmed_*``) sees a clean list."""
    seen: set[str] = set()
    out: list[str] = []
    for t in terms or []:
        cleaned = _trim_phrase(t)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _truncate(q: str) -> str:
    return q if len(q) <= MAX_QUERY_LEN else q[:MAX_QUERY_LEN]


def _short_population(pop: str) -> str:
    if not pop:
        return ""
    lower = pop.lower()
    idx = lower.find(" with ")
    if idx >= 0:
        return _trim_phrase(pop[idx + 6:])
    return _trim_phrase(pop)


def _signal_outcome(outcomes: list[str]) -> str:
    if not outcomes:
        return ""
    for o in outcomes:
        ol = o.lower()
        if any(skip in ol for skip in ("diagnostic accuracy",)):
            continue
        if len(o) > 0 and len(o) <= 40:
            return o
    return ""


# ============================================================================
# Europe PMC syntax
# ============================================================================
def _epmc_quote(term: str) -> str:
    term = _trim_phrase(term)
    if not term:
        return ""
    return '"' + term.replace('"', '\\"') + '"'


def _epmc_ta(term: str) -> str:
    q = _epmc_quote(term)
    return f"TITLE_ABS:{q}" if q else ""


def _epmc_or_block(terms: list[str]) -> str:
    cleaned = [t for t in (_epmc_ta(x) for x in terms) if t]
    if not cleaned:
        return ""
    return "(" + " OR ".join(cleaned) + ")"


def _epmc_and_block(terms: list[str]) -> str:
    cleaned = [t for t in (_epmc_ta(x) for x in terms) if t]
    if not cleaned:
        return ""
    return "(" + " AND ".join(cleaned) + ")"


_EPMC_PUB_TYPE_BY_QTYPE: dict[str, list[str]] = {
    "treatment": ["systematic review", "meta analysis", "randomized controlled trial"],
    "prevention": ["systematic review", "meta analysis", "randomized controlled trial"],
    "screening": ["systematic review", "meta analysis", "randomized controlled trial"],
    "diagnosis": ["systematic review", "meta analysis", "diagnostic accuracy"],
    "harm": ["cohort studies", "case-control studies", "observational study"],
    "prognosis": ["cohort studies", "observational study"],
    "etiology": ["cohort studies", "case-control studies", "observational study"],
    "guideline": ["practice guideline", "guideline"],
    "epidemiology": ["cross-sectional studies", "cohort studies", "observational study"],
    "pharmacology": ["randomized controlled trial", "review"],
    "other": ["systematic review", "review"],
}

_EPMC_GUIDELINE_PUB_TYPES = ["practice guideline", "guideline", "consensus development conference"]


def _epmc_pub_type_block(qtype: str) -> str:
    pts = _EPMC_PUB_TYPE_BY_QTYPE.get(qtype) or _EPMC_PUB_TYPE_BY_QTYPE["other"]
    parts = [f'PUB_TYPE:{_epmc_quote(pt)}' for pt in pts]
    return "(" + " OR ".join(parts) + ")"


_EPMC_EXCL = 'NOT SRC:PPR NOT PUB_TYPE:"preprint"'


def _build_epmc_plans(
    *,
    recall_terms: list[str],
    precision_terms: list[str],
    used_fields: list[str],
    qtype: str,
    normalized: str,
    population_short: str,
    intervention: str,
) -> list[PlannedQuery]:
    """Phase 5D Fix 4 — emit families ONLY when their block is non-empty
    after STOP_PHRASES/STOP_TOKENS filtering. Skip + log otherwise, so
    we never emit naked ``()`` clauses."""
    plans: list[PlannedQuery] = []

    # high_recall
    hr_block = _epmc_or_block(recall_terms)
    if not hr_block:
        # Fallback: try the normalized question only if it survives the filter.
        nq_clean = _trim_phrase(normalized[:80])
        if nq_clean:
            hr_block = _epmc_ta(nq_clean)
    if hr_block:
        plans.append(PlannedQuery(
            source="europe_pmc",
            query_type="high_recall",
            query_string=_truncate(f"{hr_block} {_EPMC_EXCL}"),
            rationale=(
                "Europe PMC: broad TITLE_ABS OR over key concepts, synonyms, "
                "and drug names. Preprints excluded at source."
            ),
        ))
    else:
        logger.info(
            "%s source=europe_pmc family=high_recall reason=no_recall_terms",
            EMPTY_BLOCK_LOG,
        )

    # high_precision
    precision_core = _epmc_and_block(precision_terms)
    if precision_core:
        plans.append(PlannedQuery(
            source="europe_pmc",
            query_type="high_precision",
            query_string=_truncate(f"{precision_core} {_EPMC_EXCL}"),
            rationale=(
                "Europe PMC: AND of " + ", ".join(used_fields)
                + ". Preprints excluded at source."
            ),
        ))
        # evidence_filtered — uses the same precision core.
        ef_query = _truncate(
            f"{precision_core} AND {_epmc_pub_type_block(qtype)} {_EPMC_EXCL} OPEN_ACCESS:Y"
        )
        plans.append(PlannedQuery(
            source="europe_pmc",
            query_type="evidence_filtered",
            query_string=ef_query,
            rationale=(
                f"Europe PMC: precision core narrowed to pub-types appropriate "
                f"for clinical_question_type='{qtype}'. OA preference; "
                f"preprints excluded."
            ),
        ))
    else:
        logger.info(
            "%s source=europe_pmc families=high_precision,evidence_filtered "
            "reason=no_precision_terms", EMPTY_BLOCK_LOG,
        )

    # guideline_focused
    gp_parts = [f'PUB_TYPE:{_epmc_quote(pt)}' for pt in _EPMC_GUIDELINE_PUB_TYPES]
    gp_block = "(" + " OR ".join(gp_parts) + ")"
    guideline_topic_terms = [t for t in [population_short, intervention] if t]
    guideline_topic_terms = list(dict.fromkeys(guideline_topic_terms))
    guideline_topic = _epmc_and_block(guideline_topic_terms) if guideline_topic_terms else precision_core
    society_block = (
        '(TITLE_ABS:"recommendation" OR TITLE_ABS:"society" OR '
        'TITLE_ABS:"clinical practice")'
    )
    if guideline_topic:
        plans.append(PlannedQuery(
            source="europe_pmc",
            query_type="guideline_focused",
            query_string=_truncate(
                f"{gp_block} AND ({guideline_topic} OR {society_block}) {_EPMC_EXCL}"
            ),
            rationale=(
                "Europe PMC: guideline pub-types AND'd with population + "
                "intervention (or society/recommendation language fallback). "
                "Preprints excluded."
            ),
        ))
    else:
        # Fall back to society-language only — guarantees we still have a
        # guideline family without emitting a naked ``()`` topic block.
        plans.append(PlannedQuery(
            source="europe_pmc",
            query_type="guideline_focused",
            query_string=_truncate(
                f"{gp_block} AND {society_block} {_EPMC_EXCL}"
            ),
            rationale=(
                "Europe PMC: guideline pub-types AND'd with society/"
                "recommendation language fallback. Topic block was empty "
                "after STOP_PHRASES filter."
            ),
        ))
    return plans


# ============================================================================
# PubMed (NCBI E-utilities) syntax
# ============================================================================
def _pubmed_quote(term: str) -> str:
    """Quote a phrase for NCBI search; escape inner quotes; cap length."""
    term = _trim_phrase(term)
    if not term:
        return ""
    return '"' + term.replace('"', "") + '"'


def _pubmed_tiab(term: str) -> str:
    q = _pubmed_quote(term)
    return f"{q}[tiab]" if q else ""


def _pubmed_or_block(terms: list[str]) -> str:
    cleaned = [t for t in (_pubmed_tiab(x) for x in terms) if t]
    if not cleaned:
        return ""
    return "(" + " OR ".join(cleaned) + ")"


def _pubmed_and_block(terms: list[str]) -> str:
    cleaned = [t for t in (_pubmed_tiab(x) for x in terms) if t]
    if not cleaned:
        return ""
    return "(" + " AND ".join(cleaned) + ")"


_PUBMED_PUB_TYPE_BY_QTYPE: dict[str, list[str]] = {
    "treatment": ["randomized controlled trial", "systematic review", "meta-analysis"],
    "prevention": ["randomized controlled trial", "systematic review", "meta-analysis"],
    "screening": ["randomized controlled trial", "systematic review", "meta-analysis"],
    "diagnosis": ["systematic review", "meta-analysis"],
    "harm": ["cohort studies", "case-control studies", "observational study"],
    "prognosis": ["cohort studies", "observational study"],
    "etiology": ["cohort studies", "case-control studies", "observational study"],
    "guideline": ["practice guideline", "guideline"],
    "epidemiology": ["cohort studies", "cross-sectional studies", "observational study"],
    "pharmacology": ["randomized controlled trial", "review"],
    "other": ["systematic review", "review"],
}

_PUBMED_GUIDELINE_PUB_TYPES = ["practice guideline", "guideline", "consensus development conference"]

_PUBMED_EXCL = 'NOT "preprint"[Publication Type]'


def _pubmed_pub_type_block(qtype: str) -> str:
    pts = _PUBMED_PUB_TYPE_BY_QTYPE.get(qtype) or _PUBMED_PUB_TYPE_BY_QTYPE["other"]
    parts = [f'"{pt}"[Publication Type]' for pt in pts]
    return "(" + " OR ".join(parts) + ")"


def _build_pubmed_plans(
    *,
    recall_terms: list[str],
    precision_terms: list[str],
    used_fields: list[str],
    qtype: str,
    normalized: str,
    population_short: str,
    intervention: str,
) -> list[PlannedQuery]:
    """Phase 5D Fix 4 — emit families ONLY when their block is non-empty
    after STOP_PHRASES/STOP_TOKENS filtering. Skip + log otherwise."""
    plans: list[PlannedQuery] = []

    # high_recall
    hr_block = _pubmed_or_block(recall_terms)
    if not hr_block:
        nq_clean = _trim_phrase(normalized[:80])
        if nq_clean:
            hr_block = _pubmed_tiab(nq_clean)
    if hr_block:
        plans.append(PlannedQuery(
            source="pubmed",
            query_type="high_recall",
            query_string=_truncate(f"{hr_block} {_PUBMED_EXCL}"),
            rationale=(
                "PubMed: broad [tiab] OR over key concepts, synonyms, drug names. "
                "Preprints excluded at source."
            ),
        ))
    else:
        logger.info(
            "%s source=pubmed family=high_recall reason=no_recall_terms",
            EMPTY_BLOCK_LOG,
        )

    # high_precision
    precision_core = _pubmed_and_block(precision_terms)
    if precision_core:
        plans.append(PlannedQuery(
            source="pubmed",
            query_type="high_precision",
            query_string=_truncate(f"{precision_core} {_PUBMED_EXCL}"),
            rationale=(
                "PubMed: AND of " + ", ".join(used_fields)
                + " on title/abstract. Preprints excluded at source."
            ),
        ))
        # evidence_filtered — uses the same precision core.
        plans.append(PlannedQuery(
            source="pubmed",
            query_type="evidence_filtered",
            query_string=_truncate(
                f"{precision_core} AND {_pubmed_pub_type_block(qtype)} {_PUBMED_EXCL}"
            ),
            rationale=(
                f"PubMed: precision core narrowed by [Publication Type] for "
                f"clinical_question_type='{qtype}'. Preprints excluded."
            ),
        ))
    else:
        logger.info(
            "%s source=pubmed families=high_precision,evidence_filtered "
            "reason=no_precision_terms", EMPTY_BLOCK_LOG,
        )

    # guideline_focused
    gp_parts = [f'"{pt}"[Publication Type]' for pt in _PUBMED_GUIDELINE_PUB_TYPES]
    gp_block = "(" + " OR ".join(gp_parts) + ")"
    guideline_topic_terms = [t for t in [population_short, intervention] if t]
    guideline_topic_terms = list(dict.fromkeys(guideline_topic_terms))
    guideline_topic = _pubmed_and_block(guideline_topic_terms) if guideline_topic_terms else precision_core
    if guideline_topic:
        plans.append(PlannedQuery(
            source="pubmed",
            query_type="guideline_focused",
            query_string=_truncate(
                f"{gp_block} AND {guideline_topic} {_PUBMED_EXCL}"
            ),
            rationale=(
                "PubMed: guideline [Publication Type] AND'd with population + "
                "intervention. Preprints excluded."
            ),
        ))
    else:
        # Fall back to pub-type filter only — preserves the guideline
        # family without emitting a naked AND().
        plans.append(PlannedQuery(
            source="pubmed",
            query_type="guideline_focused",
            query_string=_truncate(f"{gp_block} {_PUBMED_EXCL}"),
            rationale=(
                "PubMed: guideline [Publication Type] only — topic block "
                "was empty after STOP_PHRASES filter."
            ),
        ))
    return plans


# ============================================================================
# Public API
# ============================================================================
class SearchPlanningService:
    def plan(
        self,
        parsed: dict[str, Any],
        *,
        sources: list[str] | None = None,
    ) -> list[PlannedQuery]:
        """Return PlannedQuery list spanning the requested sources.

        ``sources`` defaults to ['europe_pmc'] for backwards compatibility
        with existing tests. The pipeline passes a list of enabled sources.

        Phase 5D Fix 4 — every candidate term is filtered at assembly via
        ``_clean_terms`` so STOP_PHRASES and STOP_TOKENS never reach the
        Boolean emitters downstream.
        """
        if sources is None:
            sources = ["europe_pmc"]

        # Raw candidate lists (capped to MAX_TERMS, dedup-via-dict).
        key_raw = [t for t in (parsed.get("key_concepts") or []) if t][:MAX_TERMS]
        syn_raw = [t for t in (parsed.get("synonyms") or []) if t][:MAX_TERMS]
        drugs_raw = [t for t in (parsed.get("drug_names_and_classes") or []) if t][:MAX_TERMS]
        intervention_raw = parsed.get("intervention_or_exposure") or ""
        comparator_raw = parsed.get("comparator") or ""
        population_raw = parsed.get("population") or ""
        population_short_raw = _short_population(population_raw)
        outcomes_raw = parsed.get("outcomes") or []
        signal_outcome_raw = _signal_outcome(outcomes_raw)
        qtype = parsed.get("clinical_question_type") or "other"
        normalized = parsed.get("normalized_question") or ""

        # Phase 5D Fix 4 — filter every input term through the STOP_PHRASES
        # + STOP_TOKENS filter at the assembly boundary, ONCE. Downstream
        # emitters then never see noise terms.
        key = _clean_terms(key_raw)
        syn = _clean_terms(syn_raw)
        drugs = _clean_terms(drugs_raw)
        intervention = _trim_phrase(intervention_raw)
        comparator = _trim_phrase(comparator_raw)
        population_short = _trim_phrase(population_short_raw)
        signal_outcome = _trim_phrase(signal_outcome_raw)

        recall_terms = list(dict.fromkeys(key + syn + drugs))
        if not recall_terms and normalized:
            # Fallback: emit the normalized question as a single term —
            # but only if it survives the filter.
            nq_clean = _trim_phrase(normalized[:80])
            recall_terms = [nq_clean] if nq_clean else []

        precision_terms: list[str] = []
        used_fields: list[str] = []
        if intervention:
            precision_terms.append(intervention)
            used_fields.append("intervention")
        if comparator:
            precision_terms.append(comparator)
            used_fields.append("comparator")
        if population_short:
            precision_terms.append(population_short)
            used_fields.append("population")
        if signal_outcome:
            precision_terms.append(signal_outcome)
            used_fields.append("outcome")
        seen: set[str] = set()
        precision_terms = [
            t for t in precision_terms
            if t and (t.lower() not in seen and not seen.add(t.lower()))
        ]
        if not precision_terms:
            precision_terms = key[:3]
            used_fields = ["key_concepts_fallback"]

        plans: list[PlannedQuery] = []
        if "europe_pmc" in sources:
            plans.extend(_build_epmc_plans(
                recall_terms=recall_terms,
                precision_terms=precision_terms,
                used_fields=used_fields,
                qtype=qtype,
                normalized=normalized,
                population_short=population_short,
                intervention=intervention,
            ))
        if "pubmed" in sources:
            plans.extend(_build_pubmed_plans(
                recall_terms=recall_terms,
                precision_terms=precision_terms,
                used_fields=used_fields,
                qtype=qtype,
                normalized=normalized,
                population_short=population_short,
                intervention=intervention,
            ))
        return plans


search_planning = SearchPlanningService()
