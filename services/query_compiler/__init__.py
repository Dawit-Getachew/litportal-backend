"""Phase 5D — Formal Boolean Query Compiler.

Extends the Phase 5A ``concept_block_query_builder`` with:
- 3 new query families (mesh_keyword, clinical_queries, trial_acronym)
- STOP_PHRASES filter
- Syntax validation (closed-set error codes)
- MeSH resolver integration
- Outcome/comparator role demotion (ranking, not mandatory, unless the
  question structure demands it)
- PubMed Clinical Queries filter strings (NCBI verbatim, public)

The output type is the existing ``ConceptBlockQuery`` (so the rest of
the pipeline doesn't need to change). New families surface as the
``query_family`` attribute and persist on ``search_queries.query_family``.

Closed-set syntax-error codes (matching the spec):
    invalid_syntax | empty_block | unbalanced_parens | invalid_field_tag
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from services.concept_block_query_builder import (
    ConceptBlockQuery,
    DEFAULT_QUOTAS,
    MAX_QUERY_LEN,
    _build_blocks,
    _dedupe_preserve,
    _mandatory_anchor_terms,
    _truncate,
)
from services.mesh import mesh_resolver

logger = logging.getLogger("clinevidence.query_compiler")


# ---------------------------------------------------------------------------
# Phase 5D — extended quotas for the 3 new query families.
# ---------------------------------------------------------------------------
import os


def _quota(env_var: str, default: int) -> int:
    try:
        return int(os.environ.get(env_var, str(default)))
    except (TypeError, ValueError):
        return default


PHASE5D_QUOTAS = {
    "mesh_keyword": _quota("Q_QUOTA_MESH_KEYWORD", 50),
    "clinical_queries": _quota("Q_QUOTA_CLINICAL_QUERIES", 50),
    "trial_acronym": _quota("Q_QUOTA_TRIAL_ACRONYM_FAMILY", 50),
    # Phase 5F — PubMed Automatic Term Mapping family. ATM only fires on
    # UNTAGGED queries — i.e., no ``[tiab]`` / ``[MeSH Terms]`` field tags.
    # PubMed's NCBI server then auto-maps free-text tokens against MeSH,
    # journal names, and author indexes. Documented at
    # https://pubmed.ncbi.nlm.nih.gov/help/#automatic-term-mapping
    "pubmed_atm": _quota("Q_QUOTA_PUBMED_ATM", 50),
    # Phase K5.2C-0 — PubMed-indexed Cochrane systematic-review lane.
    "cochrane_review": _quota("Q_QUOTA_COCHRANE_REVIEW", 50),
}


# ---------------------------------------------------------------------------
# STOP_PHRASES — stripped from concept terms before query emission.
# Now lives in ``services/query_stop_phrases.py`` so the Phase 5A legacy
# concept_block_query_builder can apply the same filter without importing
# this module (which would create a cycle).
# ---------------------------------------------------------------------------
from services.query_stop_phrases import (  # noqa: E402
    STOP_PHRASES,
    _STOP_RE,
    strip_stop_phrases,
    is_query_term_meaningful,
)


# ---------------------------------------------------------------------------
# Syntax validation — closed-set error codes.
# ---------------------------------------------------------------------------
SYNTAX_ERROR_CODES = {
    "invalid_syntax", "empty_block",
    "unbalanced_parens", "invalid_field_tag",
    # Phase K4A.2.1 — anchor-quality pre-dispatch QA error codes.
    "malformed_required_anchor", "generic_only_required_anchor",
}

# Phase K4A.2.1 — closed-set typo / non-word denylist. Any quoted
# anchor that matches an entry here triggers ``malformed_required_anchor``
# and the query is dropped pre-dispatch. Deterministic, NO LLM.
_MALFORMED_ANCHOR_TOKENS: frozenset[str] = frozenset({
    "iilness", "iillness", "illnes",
    "diseasse", "disesase",
    "pancreatits", "pancretitis",  # typo regressions seen on similar Qs
})

# Phase K4A.2.1 — closed-set generic clinical tokens. A query whose
# DISTINCT quoted clinical anchors are all in this set carries no real
# disease/drug/test/procedure/vaccine anchor and must NOT be dispatched.
# Triggers ``generic_only_required_anchor``. Closed-set, deterministic.
# Multi-word disease names that contain these tokens (e.g.,
# "Crohn's disease", "sickle cell disease", "Cushing's disease",
# "Addison's disease", "celiac disease") ARE NOT in the set — the
# check is on the EXACT normalized quoted phrase.
_GENERIC_ANCHOR_TOKENS: frozenset[str] = frozenset({
    "illness", "illnesses",
    "disease", "diseases",
    "condition", "conditions",
    "problem", "problems",
    "severity",
    "marker", "markers",
    "predictor", "predictors", "prediction",
    "risk", "risks",
    "patient", "patients",
    "outcome", "outcomes",
    # Generic catch-all tokens that should never be the sole anchor.
    "treatment", "therapy", "management",
    # Phase K4A.2.1 — multi-word generic phrases that are also never
    # safe as the SOLE quoted anchor. These multi-word phrases appear
    # in the K4A.2 severity OR-block alongside a real disease anchor —
    # they should pass when paired (e.g., "acute pancreatitis" + "severity
    # of illness") but fail when standalone.
    "severity of illness",
    "severity of disease",
    "disease severity",
    "severity prediction",
    "severity assessment",
    "risk stratification",
    "prognostic marker", "prognostic markers",
    "prognostic factor", "prognostic factors",
    "scoring system", "scoring systems",
    "risk score", "risk scores",
    # Phase K4A.3 — diagnostic rule-out / test-utility outcome cues.
    # These pair with a real disease + index_test anchor and never
    # carry a query on their own.
    "rule out",
    "rule-out",
    "rules out",
    "ruling out",
    "predictive value",
    "negative predictive value",
    "diagnostic accuracy",
    "diagnostic utility",
    "test utility",
    "test accuracy",
    "sensitivity",
    "specificity",
    "likelihood ratio",
    "nasal swab",
    "nares swab",
    "nasal screening",
    "nares screening",
    # Phase K4A.4 — incidence / risk epidemiology generic phrases.
    # These pair with a real outcome (aspiration pneumonia) + risk-
    # context (low GCS / Glasgow Coma Scale) anchor and never carry a
    # query on their own. The pre-fix bug emitted a literal
    # ``"patients with GCS 8 or less"`` AS THE SOLE high-precision
    # query, and a spurious ``"complications prevention"`` as a
    # required outcome — both are now blocked pre-dispatch.
    "patients with gcs 8 or less",
    "patients with gcs less than 8",
    "patients with gcs below 8",
    "patients with gcs under 8",
    "patients with low gcs",
    "patients with depressed gcs",
    "patients with low glasgow coma scale",
    "patients with depressed consciousness",
    "patients with depressed level of consciousness",
    "patients with decreased level of consciousness",
    "patients with altered mental status",
    "patients with loss of consciousness",
    "complications prevention",
    "incidence",
    "prevalence",
    "frequency",
    "occurrence",
    "rate",
    "risk factor",
    "risk factors",
})

_QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"')

_PUBMED_FIELD_TAGS = {
    "tiab", "Title/Abstract", "MeSH Terms", "MeSH Subheading",
    "Publication Type", "Date - Publication", "PDat", "Language",
    # Phase K5.2C-0 — journal-title-abbreviation tags so journal-scoped
    # Cochrane queries (``"Cochrane Database Syst Rev"[ta]``) pass query QA.
    "ta", "jour", "Journal",
}
_EPMC_FIELD_PREFIXES = {
    "TITLE", "ABSTRACT", "TITLE_ABS", "KW", "AUTH", "PUB_TYPE",
    "SRC", "FIRST_PDATE", "DOI", "PMID", "PMCID", "MESH",
}


@dataclass
class SyntaxValidation:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_query_syntax(
    query_string: str, *, source: str = "europe_pmc",
) -> SyntaxValidation:
    """Validate a compiled query string. Returns ``SyntaxValidation``.

    Hard failures (``valid=False``):
      - unbalanced_parens
      - empty_block (e.g., ``()`` or trailing AND/OR)
      - invalid_field_tag (closed-set check)
      - malformed_required_anchor (Phase K4A.2.1) — any quoted clinical
        token in the query matches the closed-set typo / non-word
        denylist (e.g., ``iilness``).
      - generic_only_required_anchor (Phase K4A.2.1) — every distinct
        quoted clinical token in the query is in the closed-set
        generic-token list (illness / severity / marker / predictor /
        risk / outcome / disease / condition / problem / patient /
        ...) AND the query has no other anchor clause.

    Soft warnings include ``query_string_truncated`` if the string was
    trimmed by the caller.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not query_string or not query_string.strip():
        return SyntaxValidation(valid=False, errors=["empty_block"])

    # Parens balance
    depth = 0
    for ch in query_string:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                errors.append("unbalanced_parens")
                break
    if depth != 0 and "unbalanced_parens" not in errors:
        errors.append("unbalanced_parens")

    # Empty blocks: () or trailing/leading operators
    if re.search(r"\(\s*\)", query_string):
        errors.append("empty_block")
    if re.search(r"\b(AND|OR|NOT)\s*$", query_string.strip(), re.IGNORECASE):
        errors.append("invalid_syntax")
    if re.search(r"^\s*(AND|OR)\b", query_string.strip(), re.IGNORECASE):
        errors.append("invalid_syntax")
    if re.search(r"\b(AND|OR)\s+(AND|OR|NOT)\b", query_string, re.IGNORECASE):
        errors.append("invalid_syntax")

    # Field tag validation per source
    if source == "pubmed":
        # PubMed tags look like ``"…"[Field Name]`` or ``term[tiab]``.
        for m in re.finditer(r"\[([^\[\]]+?)\]", query_string):
            tag = m.group(1).strip()
            if tag not in _PUBMED_FIELD_TAGS:
                errors.append("invalid_field_tag")
                warnings.append(f"unknown_pubmed_tag:{tag}")
        # Reject Europe PMC-style prefixes leaking into a PubMed query.
        if re.search(r"\b(TITLE_ABS|TITLE|KW|PUB_TYPE|FIRST_PDATE|SRC):", query_string):
            errors.append("invalid_field_tag")
            warnings.append("europe_pmc_prefix_in_pubmed_query")
    elif source == "europe_pmc":
        # Europe PMC uses ``PREFIX:value`` syntax. Verify only allowed prefixes.
        for m in re.finditer(r"\b([A-Z_]{2,16}):", query_string):
            prefix = m.group(1)
            if prefix not in _EPMC_FIELD_PREFIXES:
                errors.append("invalid_field_tag")
                warnings.append(f"unknown_epmc_prefix:{prefix}")
        # Reject PubMed-only `[Tag]` syntax leaking into Europe PMC.
        if re.search(r"\[[A-Za-z][^\]]*\]", query_string):
            errors.append("invalid_field_tag")
            warnings.append("pubmed_tag_in_epmc_query")

    # Phase K4A.2.1 — anchor-quality pre-dispatch QA.
    # Extract every quoted phrase, lower-case + strip, then:
    #   1. If ANY quoted phrase is a closed-set typo/non-word
    #      (``iilness`` etc.) → ``malformed_required_anchor``.
    #   2. If ALL DISTINCT quoted phrases are in the closed-set
    #      generic-token set AND there are no other clinical anchors
    #      visible → ``generic_only_required_anchor``.
    # Closed-set, deterministic, NO LLM. Multi-word disease names
    # that contain a generic token (``Crohn's disease`` etc.) are
    # NOT in the generic set so they continue to validate.
    quoted_phrases = []
    for m in _QUOTED_PHRASE_RE.finditer(query_string):
        ph = (m.group(1) or "").strip().lower()
        if ph:
            quoted_phrases.append(ph)
    if quoted_phrases:
        if any(ph in _MALFORMED_ANCHOR_TOKENS for ph in quoted_phrases):
            errors.append("malformed_required_anchor")
            warnings.append("query_has_malformed_anchor_token")
        else:
            distinct = set(quoted_phrases)
            if distinct and all(
                ph in _GENERIC_ANCHOR_TOKENS for ph in distinct
            ):
                errors.append("generic_only_required_anchor")
                warnings.append("query_has_only_generic_anchor_tokens")

    return SyntaxValidation(
        valid=not errors,
        errors=_dedupe_preserve(errors),
        warnings=_dedupe_preserve(warnings),
    )


# ---------------------------------------------------------------------------
# Outcome / comparator role policy (Phase 5D pre-committed decision #5).
# ---------------------------------------------------------------------------
_OUTCOME_FOCUSED_QUESTION_TYPES = {
    "prognosis", "epidemiology", "etiology_harm",
}
_OUTCOME_FOCUSED_PHRASES = re.compile(
    r"\b(?:mortality rate|prevalence|incidence|five[- ]year survival)\b",
    re.IGNORECASE,
)
_COMPARISON_SUBTYPES = {
    "transfusion_threshold_comparison",
}


def outcomes_are_mandatory(picott: dict[str, Any]) -> bool:
    """Return True only when the question is explicitly outcome-focused.

    By default, outcomes are ranking concepts (not query-time anchors),
    so the compiler emits them in the broad_recall block but never
    inside the high_precision AND-of-anchors. This keeps high-precision
    queries narrow to the population × intervention × condition core.
    """
    type_of_q = (picott.get("type_of_question") or "").lower()
    if type_of_q in _OUTCOME_FOCUSED_QUESTION_TYPES:
        return True
    norm = picott.get("normalized_question") or ""
    return bool(_OUTCOME_FOCUSED_PHRASES.search(norm))


def comparator_is_mandatory(picott: dict[str, Any]) -> bool:
    """Comparator is mandatory only on explicit-comparison questions."""
    subtype = (picott.get("question_subtype") or "")
    if subtype in _COMPARISON_SUBTYPES:
        return True
    # comparator must-have explicitly marked in PICOTT.
    for mh in (picott.get("must_have_concepts") or []):
        if (mh.get("role") or "").lower() == "comparator":
            return True
    return False


# ---------------------------------------------------------------------------
# PubMed Clinical Queries filters — NCBI verbatim (public documentation).
# https://pubmed.ncbi.nlm.nih.gov/help/#clinical-queries-filters
# We ship 4 categories (therapy / diagnosis / etiology / prognosis),
# each with broad+narrow scope.
# ---------------------------------------------------------------------------
CLINICAL_QUERIES_FILTERS: dict[str, dict[str, str]] = {
    "therapy": {
        "broad": (
            '((clinical[Title/Abstract] AND trial[Title/Abstract]) OR '
            '"clinical trials as topic"[MeSH Terms] OR clinical trial[Publication Type] OR '
            'random*[Title/Abstract] OR randomly[Title/Abstract] OR groups[Title/Abstract])'
        ),
        "narrow": (
            '(randomized controlled trial[Publication Type] OR '
            '(randomized[Title/Abstract] AND controlled[Title/Abstract] AND trial[Title/Abstract]))'
        ),
    },
    "diagnosis": {
        "broad": (
            '(sensitiv*[Title/Abstract] OR sensitivity and specificity[MeSH Terms] OR '
            'diagnos*[Title/Abstract] OR diagnosis[MeSH:noexp] OR '
            'diagnostic *[MeSH:noexp] OR diagnosis,differential[MeSH:noexp] OR '
            'diagnosis[Subheading:noexp])'
        ),
        "narrow": (
            '(specificity[Title/Abstract])'
        ),
    },
    "etiology": {
        "broad": (
            '(risk*[Title/Abstract] OR risk[MeSH:noexp] OR risk factors[MeSH:noexp] OR '
            'cohort studies[MeSH Terms] OR group[Title/Abstract] OR groups[Title/Abstract] OR '
            'groups\'[Title/Abstract])'
        ),
        "narrow": (
            '((relative[Title/Abstract] AND risk*[Title/Abstract]) OR '
            '(relative risk[Title/Abstract]) OR (risks[Title/Abstract]) OR '
            '(cohort studies[MeSH:noexp]) OR (risk period*[Title/Abstract]))'
        ),
    },
    "prognosis": {
        "broad": (
            '(incidence[MeSH:noexp] OR mortality[MeSH Terms] OR follow up studies[MeSH:noexp] OR '
            'prognos*[Title/Abstract] OR predict*[Title/Abstract] OR course[Title/Abstract])'
        ),
        "narrow": (
            '(prognos*[Title/Abstract] OR (first[Title/Abstract] AND episode[Title/Abstract]) OR '
            'cohort[Title/Abstract])'
        ),
    },
}

_QUESTION_TYPE_TO_CQ_CATEGORY = {
    "therapy": "therapy",
    "pharmacology": "therapy",
    "dosing_protocol": "therapy",
    "prevention": "therapy",
    "diagnosis": "diagnosis",
    "clinical_exam": "diagnosis",
    "etiology_harm": "etiology",
    "prognosis": "prognosis",
    "epidemiology": "prognosis",
}


def clinical_queries_category(picott: dict[str, Any]) -> str | None:
    type_of_q = (picott.get("type_of_question") or "").lower()
    return _QUESTION_TYPE_TO_CQ_CATEGORY.get(type_of_q)


# ---------------------------------------------------------------------------
# Compiler — extends the Phase 5A builder with 3 new families.
# ---------------------------------------------------------------------------
@dataclass
class CompiledQuery:
    source: str
    query_family: str
    query_string: str
    concept_blocks: dict[str, list[str]]
    mandatory_anchors: list[str]
    optional_terms: list[str]
    rationale: str
    max_results: int
    field_tags_used: list[str] = field(default_factory=list)
    mesh_terms_used: list[dict[str, Any]] = field(default_factory=list)
    syntax_validation: SyntaxValidation = field(
        default_factory=lambda: SyntaxValidation(valid=True),
    )
    cq_category: str | None = None
    cq_scope: str | None = None

    def to_concept_block_query(self) -> ConceptBlockQuery:
        return ConceptBlockQuery(
            source=self.source,
            query_family=self.query_family,
            query_string=self.query_string,
            concept_blocks=self.concept_blocks,
            mandatory_anchors=self.mandatory_anchors,
            optional_terms=self.optional_terms,
            rationale=self.rationale,
            max_results=self.max_results,
            query_type=self.query_family,
        )


def _pubmed_mesh_or_tiab(concept: str) -> tuple[str, dict[str, Any] | None, str]:
    """Return (rendered_term, mesh_meta, field_tag_used) for a concept.

    Falls back to ``"concept"[tiab]`` when no confident MeSH match.
    Drops single-token stopwords (e.g. "tell", "about", "me") so they
    can't leak into a ``[tiab]`` expression.
    """
    cleaned = strip_stop_phrases(concept).strip()
    if not cleaned or not is_query_term_meaningful(cleaned):
        return "", None, ""
    res = mesh_resolver.resolve(cleaned)
    if res:
        return (
            f'"{res.mesh_descriptor}"[MeSH Terms] OR "{cleaned}"[tiab]',
            {"concept": cleaned, "mesh_descriptor": res.mesh_descriptor,
             "confidence": res.confidence, "source": res.source},
            "MeSH Terms",
        )
    return f'"{cleaned}"[tiab]', None, "tiab"


class BooleanQueryCompilerService:
    """Phase 5D — formal Boolean compiler. Wraps the legacy
    ``concept_block_query_builder`` and adds 3 new families + MeSH
    resolution + syntax validation."""

    def compile(
        self,
        picott: dict[str, Any],
        *,
        sources: list[str] | None = None,
        is_phi_blocked: bool = False,
    ) -> list[CompiledQuery]:
        if is_phi_blocked:
            return []
        sources = sources or ["europe_pmc"]
        out: list[CompiledQuery] = []
        for src in sources:
            out.extend(self._mesh_keyword_for_source(src, picott))
            out.extend(self._trial_acronym_for_source(src, picott))
            if src == "pubmed":
                out.extend(self._clinical_queries_for_pubmed(picott))
                # Phase 5F — PubMed Automatic Term Mapping (ATM) variant.
                # Generates an UNTAGGED PubMed query so NCBI's ATM engine
                # can auto-map free-text tokens to MeSH / synonyms.
                out.extend(self._pubmed_atm(picott))
                # Phase K5.2C-0 — PubMed-indexed Cochrane systematic-review
                # lane (journal-scoped). Emitted only for applicable question
                # lanes (therapy/prevention/diagnosis/drug-safety/prognosis/
                # etiology). PubMed-only; metadata + link-out; NO scraping.
                out.extend(self._cochrane_review_for_pubmed(picott))
        # Filter out any compiled query whose syntax validation FAILED.
        kept: list[CompiledQuery] = []
        for q in out:
            v = validate_query_syntax(q.query_string, source=q.source)
            q.syntax_validation = v
            if v.valid:
                kept.append(q)
            else:
                logger.info(
                    "phase5d_query_dropped",
                    extra={"source": q.source, "family": q.query_family,
                            "errors": v.errors},
                )
        return kept

    # ------------------------------------------------------------------
    # mesh_keyword
    # ------------------------------------------------------------------
    def _mesh_keyword_for_source(
        self, source: str, picott: dict[str, Any],
    ) -> list[CompiledQuery]:
        mandatory = _mandatory_anchor_terms(picott)
        if not mandatory:
            return []
        blocks = _build_blocks(picott)
        if source == "pubmed":
            and_blocks: list[str] = []
            mesh_meta_all: list[dict[str, Any]] = []
            for anchor in mandatory[:4]:  # cap blocks to keep query short
                cleaned = strip_stop_phrases(anchor)
                rendered, meta, _ = _pubmed_mesh_or_tiab(cleaned)
                if rendered:
                    and_blocks.append(f"({rendered})")
                    if meta:
                        mesh_meta_all.append(meta)
            if not and_blocks:
                return []
            qs = _truncate(
                " AND ".join(and_blocks) + ' NOT "preprint"[Publication Type]'
            )
            return [CompiledQuery(
                source=source,
                query_family="mesh_keyword",
                query_string=qs,
                concept_blocks=dict(blocks),
                mandatory_anchors=mandatory,
                optional_terms=[],
                rationale=(
                    "MeSH+Title/Abstract hybrid AND of mandatory anchors. "
                    f"Used {len(mesh_meta_all)} verified MeSH descriptor(s); "
                    "rest fall back to title/abstract keywords."
                ),
                max_results=PHASE5D_QUOTAS["mesh_keyword"],
                field_tags_used=_dedupe_preserve(
                    ["MeSH Terms", "tiab", "Publication Type"]
                    if mesh_meta_all else ["tiab", "Publication Type"]
                ),
                mesh_terms_used=mesh_meta_all,
            )]
        # europe_pmc — Europe PMC handles MeSH via the MESH: prefix.
        and_blocks_e: list[str] = []
        mesh_meta_e: list[dict[str, Any]] = []
        for anchor in mandatory[:4]:
            cleaned = strip_stop_phrases(anchor)
            if not cleaned:
                continue
            res = mesh_resolver.resolve(cleaned)
            if res:
                and_blocks_e.append(
                    f'(MESH:"{res.mesh_descriptor}" OR TITLE_ABS:"{cleaned}")'
                )
                mesh_meta_e.append({
                    "concept": cleaned, "mesh_descriptor": res.mesh_descriptor,
                    "confidence": res.confidence, "source": res.source,
                })
            else:
                and_blocks_e.append(f'TITLE_ABS:"{cleaned}"')
        if not and_blocks_e:
            return []
        qs = _truncate(" AND ".join(and_blocks_e) + ' NOT SRC:PPR')
        return [CompiledQuery(
            source=source,
            query_family="mesh_keyword",
            query_string=qs,
            concept_blocks=dict(blocks),
            mandatory_anchors=mandatory,
            optional_terms=[],
            rationale=(
                "Europe PMC MESH + TITLE_ABS hybrid AND of mandatory anchors."
            ),
            max_results=PHASE5D_QUOTAS["mesh_keyword"],
            field_tags_used=["MESH", "TITLE_ABS", "SRC"],
            mesh_terms_used=mesh_meta_e,
        )]

    # ------------------------------------------------------------------
    # clinical_queries (PubMed only)
    # ------------------------------------------------------------------
    def _clinical_queries_for_pubmed(
        self, picott: dict[str, Any],
    ) -> list[CompiledQuery]:
        cat = clinical_queries_category(picott)
        if not cat:
            return []
        mandatory = _mandatory_anchor_terms(picott)
        if not mandatory:
            return []
        and_blocks: list[str] = []
        for anchor in mandatory[:3]:
            rendered, _meta, _tag = _pubmed_mesh_or_tiab(anchor)
            if rendered:
                and_blocks.append(f"({rendered})")
        if not and_blocks:
            return []
        core = " AND ".join(and_blocks)
        out: list[CompiledQuery] = []
        for scope in ("broad", "narrow"):
            filter_str = CLINICAL_QUERIES_FILTERS[cat][scope]
            qs = _truncate(
                f"{core} AND {filter_str} NOT \"preprint\"[Publication Type]",
            )
            out.append(CompiledQuery(
                source="pubmed",
                query_family="clinical_queries",
                query_string=qs,
                concept_blocks=_build_blocks(picott),
                mandatory_anchors=mandatory,
                optional_terms=[],
                rationale=(
                    f"PubMed Clinical Queries ({cat}, {scope}) filter AND "
                    "mandatory anchors. Filter text is NCBI-verbatim."
                ),
                max_results=PHASE5D_QUOTAS["clinical_queries"],
                field_tags_used=["MeSH Terms", "Title/Abstract",
                                  "Publication Type"],
                cq_category=cat,
                cq_scope=scope,
            ))
        return out

    # ------------------------------------------------------------------
    # pubmed_atm (PubMed only — UNTAGGED query for Automatic Term Mapping)
    # ------------------------------------------------------------------
    def _pubmed_atm(
        self, picott: dict[str, Any],
    ) -> list[CompiledQuery]:
        """Build a PubMed UNTAGGED query so NCBI's Automatic Term Mapping
        engine can auto-map free-text tokens against MeSH descriptors,
        entry terms, and journal/author indexes.

        ATM is ONLY engaged when the query contains zero field tags. We
        therefore strip the ``[tiab]`` / ``[MeSH Terms]`` tags entirely
        and AND together the cleaned anchor texts as plain free-text
        terms (quoted only when multi-word).

        Reference: https://pubmed.ncbi.nlm.nih.gov/help/#automatic-term-mapping
        """
        mandatory = _mandatory_anchor_terms(picott)
        if not mandatory:
            return []
        blocks = _build_blocks(picott)
        free_terms: list[str] = []
        for anchor in mandatory[:5]:
            cleaned = strip_stop_phrases(anchor).strip()
            if not cleaned:
                continue
            # Quote multi-word terms so PubMed treats them as a phrase.
            free_terms.append(
                f"\"{cleaned}\"" if " " in cleaned else cleaned
            )
        if not free_terms:
            return []
        qs = _truncate(" AND ".join(free_terms))
        return [CompiledQuery(
            source="pubmed",
            query_family="pubmed_atm",
            query_string=qs,
            concept_blocks=blocks,
            mandatory_anchors=mandatory,
            optional_terms=[],
            rationale=(
                "PubMed Automatic Term Mapping (ATM) variant — UNTAGGED "
                "free-text query so PubMed auto-maps tokens against MeSH "
                "and entry-term indexes. Critical for short / shorthand "
                "questions where field-tagged queries return zero hits."
            ),
            max_results=PHASE5D_QUOTAS["pubmed_atm"],
            field_tags_used=[],
        )]

    # ------------------------------------------------------------------
    # cochrane_review (PubMed only — journal-scoped systematic reviews)
    # ------------------------------------------------------------------
    def _cochrane_review_for_pubmed(
        self, picott: dict[str, Any],
    ) -> list[CompiledQuery]:
        """Build a PubMed query scoped to the Cochrane Database of Systematic
        Reviews (journal abbreviation ``Cochrane Database Syst Rev``) ANDed
        with the question's mandatory anchors.

        Emitted ONLY for applicable question lanes (therapy / prevention /
        diagnosis / drug-safety / prognosis / etiology) per
        ``cochrane_lane.cochrane_query_applicable``. PubMed-only. Bibliographic
        metadata + link-out only — NO Cochrane Library / Wiley API, NO
        scraping, NO full-text ingestion.
        """
        from services.cochrane_lane import (
            COCHRANE_PUBMED_JOURNAL_TAG,
            cochrane_query_applicable,
        )
        if not cochrane_query_applicable(picott):
            return []
        mandatory = _mandatory_anchor_terms(picott)
        if not mandatory:
            return []
        and_blocks: list[str] = []
        mesh_meta_all: list[dict[str, Any]] = []
        for anchor in mandatory[:3]:
            rendered, meta, _ = _pubmed_mesh_or_tiab(anchor)
            if rendered:
                and_blocks.append(f"({rendered})")
                if meta:
                    mesh_meta_all.append(meta)
        if not and_blocks:
            return []
        qs = _truncate(
            f"{COCHRANE_PUBMED_JOURNAL_TAG} AND " + " AND ".join(and_blocks)
        )
        return [CompiledQuery(
            source="pubmed",
            query_family="cochrane_review",
            query_string=qs,
            concept_blocks=_build_blocks(picott),
            mandatory_anchors=mandatory,
            optional_terms=[],
            rationale=(
                "PubMed-indexed Cochrane systematic reviews (journal-scoped to "
                "'Cochrane Database Syst Rev') ANDed with mandatory anchors. "
                "Bibliographic metadata + link-out only; no Cochrane Library / "
                "Wiley API; no scraping; no full-text ingestion."
            ),
            max_results=PHASE5D_QUOTAS["cochrane_review"],
            field_tags_used=_dedupe_preserve(
                ["ta", "MeSH Terms", "tiab"]
                if mesh_meta_all else ["ta", "tiab"]
            ),
            mesh_terms_used=mesh_meta_all,
        )]

    # ------------------------------------------------------------------
    # trial_acronym (both sources)
    # ------------------------------------------------------------------
    def _trial_acronym_for_source(
        self, source: str, picott: dict[str, Any],
    ) -> list[CompiledQuery]:
        # Trial acronyms come from ranking_concepts that look like
        # ALLCAPS tokens (e.g. "MINT", "REALITY") or end in " trial".
        ranking = picott.get("ranking_concepts") or []
        acronyms: list[str] = []
        for rc in ranking:
            c = (rc.get("concept") or "").strip()
            if not c:
                continue
            if c.endswith(" trial") or (c.isupper() and 2 <= len(c) <= 8):
                acronyms.append(c)
        acronyms = _dedupe_preserve(acronyms)
        if not acronyms:
            return []
        if source == "pubmed":
            ac_or = " OR ".join(f'"{a}"[Title/Abstract]' for a in acronyms[:6])
            # Constrain to the case's condition / intervention so we
            # don't pull unrelated trials with the same acronym.
            blocks = _build_blocks(picott)
            cond_terms = (blocks.get("condition") or [])[:3]
            int_terms = (blocks.get("intervention") or [])[:3]
            ctx_or = " OR ".join(
                f'"{t}"[Title/Abstract]'
                for t in _dedupe_preserve(cond_terms + int_terms)
            )
            qs = f"({ac_or}) AND ({ctx_or})" if ctx_or else f"({ac_or})"
            qs = _truncate(qs + ' NOT "preprint"[Publication Type]')
            return [CompiledQuery(
                source=source,
                query_family="trial_acronym",
                query_string=qs,
                concept_blocks=blocks,
                mandatory_anchors=acronyms,
                optional_terms=cond_terms + int_terms,
                rationale=(
                    "PubMed trial-acronym query: ANDed with condition/"
                    "intervention to avoid acronym collision with "
                    "unrelated trials."
                ),
                max_results=PHASE5D_QUOTAS["trial_acronym"],
                field_tags_used=["Title/Abstract", "Publication Type"],
            )]
        # europe_pmc
        ac_or = " OR ".join(f'TITLE_ABS:"{a}"' for a in acronyms[:6])
        blocks = _build_blocks(picott)
        cond_terms = (blocks.get("condition") or [])[:3]
        int_terms = (blocks.get("intervention") or [])[:3]
        ctx_or = " OR ".join(
            f'TITLE_ABS:"{t}"'
            for t in _dedupe_preserve(cond_terms + int_terms)
        )
        qs = f"({ac_or}) AND ({ctx_or})" if ctx_or else f"({ac_or})"
        qs = _truncate(qs + ' NOT SRC:PPR')
        return [CompiledQuery(
            source=source,
            query_family="trial_acronym",
            query_string=qs,
            concept_blocks=blocks,
            mandatory_anchors=acronyms,
            optional_terms=cond_terms + int_terms,
            rationale=(
                "Europe PMC trial-acronym query: ANDed with condition/"
                "intervention to avoid acronym collision."
            ),
            max_results=PHASE5D_QUOTAS["trial_acronym"],
            field_tags_used=["TITLE_ABS", "SRC"],
        )]


boolean_query_compiler = BooleanQueryCompilerService()
