"""Phase M4-shadow — Full-Question LLM ClinicalQuestionMap Interpreter.

Generalizes the M0/M1/M2 bounded semantic anchor repair beyond the
narrow grammar/shape allowlist: the bounded LLM may map the ENTIRE
PHI-cleared question into a full ClinicalQuestionMap candidate using
broad semantic understanding. Its authority remains bounded —
deterministic validation decides whether the candidate is clean
enough to (eventually) influence retrieval.

SHADOW ONLY. PREVIEW/DEV ONLY. The candidate NEVER changes
clinician-visible retrieval, ranking, selection, summaries, evidence
packets, or citation verification in this phase. It is recorded into
session diagnostics + an admin-only sidecar for adjudication.

Hard safety invariants (same family as semantic_anchor_repair):
  * NEVER searches sources / selects / ranks articles.
  * NEVER emits PMIDs / DOIs / NCT IDs / article references.
  * NEVER produces validated MeSH (only ``*_unvalidated`` fields).
  * NEVER references Google Scholar / ResearchGate / Sci-Hub.
  * NEVER runs in production (mode resolver forces ``disabled``).
  * NEVER runs when PHI was detected upstream (pipeline short-circuits
    to ``blocked_phi`` long before this module). The validator ALSO
    scans every output string for PHI as defense-in-depth.

NO raw prompt or raw LLM completion is persisted anywhere. Only the
validated structured candidate, closed-set trigger/rejection reasons,
the deterministic baseline snapshot, the would-be compiler projection,
and latency counters reach diagnostics.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from services.diagnostics import SessionDiagnostics, get_current as diag_current
from services.llm_provider import llm_provider

# Reuse the closed-set enums + scrub helpers from the M0/M1/M2 module so
# M4 can NEVER disagree with the upstream bounded-repair guards (DRY).
from services.semantic_anchor_repair import (
    ALLOWED_FRAMEWORKS,
    ALLOWED_PRIMARY_INTENTS,
    _ANSWER_TEXT_INDICATORS,
    _ARTICLE_REF_RE,
    _BARE_LONG_DIGITS_RE,
    _DOI_RE,
    _FRAMEWORK_ALIASES,
    _PHI_DATE_RE,
    _PHI_EMAIL_RE,
    _PHI_MRN_RE,
    _PHI_PHONE_RE,
    _PHI_SSN_RE,
    _PMID_RE,
    _PRIMARY_INTENT_ALIASES,
    _UNAPPROVED_SOURCE_RE,
    _canonicalize_str_enum,
    _generic_only_required_anchor,
    _iter_strings,
    _load_generic_denylist,
)
from services.semantic_repair_active_projection import (
    project_would_be_compiler_input,
)

logger = logging.getLogger("clinevidence.m4_full_question_interpreter")


M4_PROVENANCE: str = "m4_full_question_interpreter_shadow"


# ---------------------------------------------------------------------
# Mode resolver — env-driven, hard-off in production.
# ---------------------------------------------------------------------
M4_MODES: frozenset[str] = frozenset({"disabled", "shadow"})


def resolve_m4_mode(
    *, is_production: Optional[bool] = None, env: Optional[dict] = None,
) -> str:
    """Return the resolved M4 mode. ALWAYS ``disabled`` in production
    (defense-in-depth — independent of any env value). In non-production
    reads ``M4_FULL_QUESTION_INTERPRETER_MODE`` (default ``shadow``);
    ``active`` is NOT a permitted value in M4 and collapses to
    ``shadow``.
    """
    if is_production is None:
        try:
            from core.config import settings  # type: ignore
            is_production = bool(settings.IS_PRODUCTION)
        except Exception:  # noqa: BLE001
            is_production = False
    if is_production:
        return "disabled"
    src = env if env is not None else os.environ
    raw = (src.get("M4_FULL_QUESTION_INTERPRETER_MODE") or "shadow").strip().lower()
    if raw not in M4_MODES:
        # Any unknown / 'active' value collapses to shadow (never active).
        return "shadow"
    return raw


# ---------------------------------------------------------------------
# Closed-set TRIGGER reasons (deterministic detector, no LLM).
# ---------------------------------------------------------------------
M4_TRIGGER_REASONS: frozenset[str] = frozenset({
    "long_literal_phrase_anchor",
    "intent_word_fused_in_anchor",
    "what_is_forced_disease_overview",
    "low_baseline_confidence",
    "no_primary_entity_but_clinically_meaningful",
    "stage_biomarker_population_fused",
    "stop_grammar_required_anchor",
})


# ---------------------------------------------------------------------
# Closed-set REJECTION reasons.
# ---------------------------------------------------------------------
M4_REJECTION_REASONS: frozenset[str] = frozenset({
    # Schema / structural.
    "empty_output",
    "malformed_json",
    "refused",
    "schema_invalid",
    "missing_required_field",
    "extra_field",
    "invalid_enum",
    "invalid_slot_type",
    "invalid_confidence",
    "provenance_mismatch",
    "invalid_repair_reason",
    # Clinical bounding.
    "confidence_below_threshold",
    "no_disease_problem_anchor",
    "generic_only_required_anchor",
    "would_remove_central_anchor",
    "literal_full_phrase_required_anchor",
    "stage_not_separated_from_disease",
    "not_cleaner_than_baseline",
    "overbroad_synonym_expansion",
    "unvalidated_mesh_marked_validated",
    # Safety.
    "phi_pattern_detected",
    "article_reference_detected",
    "unapproved_source_referenced",
    "unsafe_text_detected",
})


# Intent words whose presence INSIDE a primary anchor (or in the
# question while baseline forced an overview) signals a stronger intent
# than disease_overview/background. Closed-set.
_INTENT_WORDS: tuple[str, ...] = (
    "prognosis", "prognostic", "survival", "recurrence", "recur",
    "mortality", "mechanism", "pathophysiology", "pathogenesis",
    "risk", "cause", "causes", "etiology", "aetiology",
    "treatment", "therapy", "dose", "dosing", "complication",
    "incidence", "prevalence", "screening",
)

_INTENT_WORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _INTENT_WORDS) + r")\b",
    re.IGNORECASE,
)

# "what is / what are" lead-in that the deterministic parser often maps
# to a disease/drug overview even when a stronger intent word follows.
_WHAT_IS_RE = re.compile(r"^\s*what\s+(?:is|are|was|were)\b", re.IGNORECASE)

# Stage / biomarker tokens that are frequently fused into a single
# literal anchor (e.g., "stage 1 colon cancer", "HER2 positive ...").
_STAGE_RE = re.compile(
    r"\bstage\s*(?:[0-4]|i{1,3}v?|iv)\b", re.IGNORECASE,
)
_BIOMARKER_RE = re.compile(
    r"\b(?:her2|er[\s/+-]*pr|egfr|alk|braf|kras|pd[\s-]?l1|"
    r"brca\d?|msi[\s-]?h|tmb)\b",
    re.IGNORECASE,
)

# Overview-ish baseline intents/subtypes that "what is" tends to force.
_OVERVIEW_INTENTS: frozenset[str] = frozenset({
    "disease_overview", "drug_overview", "definition",
    "background", "background_overview",
})

# Grammar connectives that, with length, mark a fused literal anchor.
_GRAMMAR_CONNECTIVES: tuple[str, ...] = (
    " in ", " of ", " for ", " after ", " following ", " with ",
    " among ", " during ", " from ",
)

# K4A.* fired subtypes — M4 must NOT trigger (cross-phase preservation).
_K4A_FIRED_SUBTYPES: frozenset[str] = frozenset({
    "severity_marker_overview",
    "rule_out_test_utility",
    "epidemiology_incidence_complication_risk",
})

# Stop-grammar prefixes that should never survive as a required anchor.
_STOP_GRAMMAR_RE = re.compile(
    r"^(?:what\s+(?:is|are)\s+the\s+)?(?:prognosis|survival|recurrence|"
    r"mortality|mechanism|risk|cause|causes|incidence|prevalence|"
    r"treatment|dose)\s+(?:of|for|in|after)\b",
    re.IGNORECASE,
)


def _anchor_text(a: Any) -> str:
    if not isinstance(a, dict):
        return ""
    return (a.get("canonical_concept") or a.get("text") or "").strip()


def _is_long_literal(text: str) -> bool:
    low = (text or "").strip().lower()
    if not low:
        return False
    words = low.split()
    return len(words) >= 5 and any(c in f" {low} " for c in _GRAMMAR_CONNECTIVES)


def detect_m4_triggers(
    *,
    normalized_question: str,
    baseline_primary_intent: Optional[str],
    baseline_subtype: Optional[str],
    primary_anchors: List[dict],
    generic_modifiers: List[str],
    anchor_failure_reason: Optional[str],
    baseline_confidence: Optional[float] = None,
) -> List[str]:
    """Closed-set deterministic detector. Returns the matched
    ``M4_TRIGGER_REASONS`` (broad on purpose — full-question
    interpretation is the point; downstream validation does the
    bounding). NEVER fires when a K4A.* route already owns the session.
    """
    q = (normalized_question or "").strip()
    if not q:
        return []
    st = (baseline_subtype or "").strip().lower()
    if st in _K4A_FIRED_SUBTYPES:
        return []

    triggers: list[str] = []
    intent_low = (baseline_primary_intent or "").strip().lower()

    anchors = primary_anchors or []
    required_texts = [
        _anchor_text(a)
        for a in anchors
        if isinstance(a, dict) and a.get("required_for_search") is True
    ]
    all_anchor_texts = [_anchor_text(a) for a in anchors]

    # (a) Long literal fused anchor.
    if any(_is_long_literal(t) for t in all_anchor_texts):
        triggers.append("long_literal_phrase_anchor")

    # (b) Intent word fused inside an anchor.
    if any(_INTENT_WORD_RE.search(t or "") for t in all_anchor_texts):
        triggers.append("intent_word_fused_in_anchor")

    # (c) "what is" forced an overview despite a stronger intent word.
    if _WHAT_IS_RE.search(q) and _INTENT_WORD_RE.search(q):
        if (intent_low in _OVERVIEW_INTENTS) or (st in _OVERVIEW_INTENTS):
            triggers.append("what_is_forced_disease_overview")

    # (d) Low baseline confidence (when available).
    if baseline_confidence is not None:
        try:
            if float(baseline_confidence) < 0.5:
                triggers.append("low_baseline_confidence")
        except (TypeError, ValueError):
            pass

    # (e) Clinically meaningful question with no primary entity.
    if (anchor_failure_reason or "").strip().lower() == "no_primary_clinical_entity":
        # Only when the question is more than a bare generic stub.
        if len(q.split()) >= 4 and _INTENT_WORD_RE.search(q):
            triggers.append("no_primary_entity_but_clinically_meaningful")

    # (f) Stage / biomarker / population fused into one anchor.
    if any(
        (_STAGE_RE.search(t or "") or _BIOMARKER_RE.search(t or ""))
        and len((t or "").split()) >= 3
        for t in all_anchor_texts
    ):
        triggers.append("stage_biomarker_population_fused")

    # (g) Required anchor contains stop-grammar ("what is the prognosis of…").
    if any(_STOP_GRAMMAR_RE.search(t or "") for t in required_texts):
        triggers.append("stop_grammar_required_anchor")

    # Dedupe, closed-set filter.
    seen: set[str] = set()
    out: list[str] = []
    for t in triggers:
        if t in seen or t not in M4_TRIGGER_REASONS:
            continue
        seen.add(t)
        out.append(t)
    return out


# ---------------------------------------------------------------------
# Schema-locked candidate (extra="forbid").
# ---------------------------------------------------------------------
class M4Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_intent: str
    secondary_intents: List[str] = Field(default_factory=list, max_length=4)
    frameworks: List[str] = Field(default_factory=list, max_length=4)
    clinical_problem: List[str] = Field(default_factory=list)
    stage: List[str] = Field(default_factory=list)
    biomarker: List[str] = Field(default_factory=list)
    population: List[str] = Field(default_factory=list)
    exposure: List[str] = Field(default_factory=list)
    intervention: List[str] = Field(default_factory=list)
    procedure: List[str] = Field(default_factory=list)
    medication: List[str] = Field(default_factory=list)
    index_test: List[str] = Field(default_factory=list)
    outcome: List[str] = Field(default_factory=list)
    setting: List[str] = Field(default_factory=list)
    timeframe: List[str] = Field(default_factory=list)
    mechanism: List[str] = Field(default_factory=list)
    comparator: List[str] = Field(default_factory=list)
    required_anchors: List[str] = Field(default_factory=list)
    optional_or_ranking_only_terms: List[str] = Field(default_factory=list)
    candidate_synonyms_unvalidated: List[str] = Field(default_factory=list)
    candidate_mesh_terms_unvalidated: List[str] = Field(default_factory=list)
    clarification_required: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    repair_reason: str
    provenance: Literal["m4_full_question_interpreter_shadow"]


CONFIDENCE_THRESHOLD: float = float(
    (os.environ.get("M4_CONFIDENCE_THRESHOLD") or "0.7").strip() or "0.7"
)
MAX_REQUIRED_ANCHORS: int = 6
MAX_SYNONYMS: int = 15


_PYDANTIC_TYPE_TO_CLOSED_SET: dict[str, str] = {
    "missing": "missing_required_field",
    "extra_forbidden": "extra_field",
    "literal_error": "invalid_enum",
    "enum": "invalid_enum",
    "string_type": "invalid_slot_type",
    "list_type": "invalid_slot_type",
    "dict_type": "invalid_slot_type",
    "bool_type": "invalid_slot_type",
    "bool_parsing": "invalid_slot_type",
    "float_parsing": "invalid_confidence",
    "int_parsing": "invalid_confidence",
    "float_type": "invalid_confidence",
    "greater_than_equal": "invalid_confidence",
    "less_than_equal": "invalid_confidence",
    "json_invalid": "malformed_json",
    "too_long": "overbroad_synonym_expansion",
}


def _pydantic_errors_to_closed_set(ve: ValidationError) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    try:
        errs = ve.errors()
    except Exception:  # noqa: BLE001
        return ["schema_invalid"]
    for err in errs or []:
        etype = (err.get("type") or "").strip()
        loc = ".".join(str(x) for x in (err.get("loc") or ()))
        if etype == "literal_error" and "provenance" in loc:
            mapped = "provenance_mismatch"
        else:
            mapped = _PYDANTIC_TYPE_TO_CLOSED_SET.get(etype, "schema_invalid")
        if mapped not in seen:
            seen.add(mapped)
            out.append(mapped)
    return out or ["schema_invalid"]


def canonicalize_m4_candidate(raw: Optional[dict]) -> Optional[dict]:
    """Closed-set enum normalization BEFORE strict validation. Never
    mutates input. Never invents new enum values."""
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    if "primary_intent" in out:
        out["primary_intent"] = _canonicalize_str_enum(
            out.get("primary_intent"), _PRIMARY_INTENT_ALIASES,
        )
    if isinstance(out.get("secondary_intents"), list):
        out["secondary_intents"] = [
            _canonicalize_str_enum(s, _PRIMARY_INTENT_ALIASES)
            for s in out["secondary_intents"]
        ]
    if isinstance(out.get("frameworks"), list):
        out["frameworks"] = [
            _canonicalize_str_enum(s, _FRAMEWORK_ALIASES)
            for s in out["frameworks"]
        ]
    return out


def validate_m4_candidate(
    raw: Optional[dict],
    *,
    deterministic_primary_anchors: List[dict],
    baseline_required_anchors: List[str],
) -> tuple[Optional[M4Candidate], list[str]]:
    """Deterministic validator. Returns ``(candidate, [])`` on success
    or ``(None, [rejection_reason, ...])``. Every reason is a member of
    ``M4_REJECTION_REASONS``.
    """
    if not raw:
        return None, ["empty_output"]
    if not isinstance(raw, dict):
        return None, ["malformed_json"]
    if raw.get("refused") is True:
        return None, ["refused"]

    canon = canonicalize_m4_candidate(raw)
    if not isinstance(canon, dict):
        return None, ["malformed_json"]

    try:
        cand = M4Candidate.model_validate(canon)
    except ValidationError as ve:
        return None, _pydantic_errors_to_closed_set(ve)
    except Exception:  # noqa: BLE001
        return None, ["schema_invalid"]

    if cand.provenance != M4_PROVENANCE:
        return None, ["provenance_mismatch"]

    # Enum validations.
    if cand.primary_intent not in ALLOWED_PRIMARY_INTENTS:
        return None, ["invalid_enum"]
    for si in cand.secondary_intents:
        if si not in ALLOWED_PRIMARY_INTENTS:
            return None, ["invalid_enum"]
    for fw in cand.frameworks:
        if fw not in ALLOWED_FRAMEWORKS:
            return None, ["invalid_enum"]
    if cand.repair_reason not in M4_TRIGGER_REASONS:
        return None, ["invalid_repair_reason"]

    # Confidence floor.
    if cand.confidence < CONFIDENCE_THRESHOLD:
        return None, ["confidence_below_threshold"]

    # Overbroad expansion guard.
    if (
        len(cand.required_anchors) > MAX_REQUIRED_ANCHORS
        or len(cand.candidate_synonyms_unvalidated) > MAX_SYNONYMS
    ):
        return None, ["overbroad_synonym_expansion"]

    # At least one non-generic required anchor.
    if not cand.required_anchors:
        return None, ["no_disease_problem_anchor"]
    denylist = _load_generic_denylist()
    if _generic_only_required_anchor(
        cand.required_anchors, generic_denylist=denylist,
    ):
        return None, ["generic_only_required_anchor"]

    # Required anchors must NOT be a fused literal full phrase.
    for a in cand.required_anchors:
        if _is_long_literal(a):
            return None, ["literal_full_phrase_required_anchor"]

    # Stage must be SEPARATED from the disease anchor — a required
    # anchor must not still carry a stage token.
    for a in cand.required_anchors:
        if _STAGE_RE.search(a or ""):
            return None, ["stage_not_separated_from_disease"]

    # Unvalidated MeSH must NOT be promoted.
    for s in _iter_strings(canon):
        if isinstance(s, str) and s.lower().startswith("mesh:") and not s.lower().endswith("_unvalidated"):
            return None, ["unvalidated_mesh_marked_validated"]

    # Safety scrub — PHI / article refs / unapproved sources / answer-text.
    for s in _iter_strings(cand.model_dump()):
        if not isinstance(s, str):
            continue
        if _PMID_RE.search(s) or _DOI_RE.search(s) or _ARTICLE_REF_RE.search(s):
            return None, ["article_reference_detected"]
        if _BARE_LONG_DIGITS_RE.search(s):
            return None, ["article_reference_detected"]
        if _UNAPPROVED_SOURCE_RE.search(s):
            return None, ["unapproved_source_referenced"]
        if (
            _PHI_EMAIL_RE.search(s) or _PHI_PHONE_RE.search(s)
            or _PHI_SSN_RE.search(s) or _PHI_MRN_RE.search(s)
            or _PHI_DATE_RE.search(s)
        ):
            return None, ["phi_pattern_detected"]
        s_low = " " + s.lower() + " "
        if any(ind in s_low for ind in _ANSWER_TEXT_INDICATORS):
            return None, ["unsafe_text_detected"]

    # Central deterministic anchor preservation (short disease/drug
    # tokens only — long literal phrases were the trigger signal).
    central_terms: list[str] = []
    for a in (deterministic_primary_anchors or []):
        concept = _anchor_text(a).lower()
        if not concept or concept in denylist:
            continue
        if 1 <= len(concept.split()) <= 4 and not _is_long_literal(concept):
            central_terms.append(concept)
    if central_terms:
        cand_terms_low = {
            (t or "").strip().lower()
            for t in (
                cand.required_anchors
                + cand.optional_or_ranking_only_terms
                + cand.clinical_problem
            )
        }
        cand_terms_low.discard("")
        for ct in central_terms:
            if not any(ct in t or t in ct for t in cand_terms_low):
                return None, ["would_remove_central_anchor"]

    # Cleaner-than-baseline: when the baseline carried a fused literal
    # required anchor, the candidate must NOT simply reproduce it.
    baseline_low = {(b or "").strip().lower() for b in (baseline_required_anchors or [])}
    baseline_low.discard("")
    baseline_had_literal = any(_is_long_literal(b) for b in baseline_low)
    if baseline_had_literal:
        cand_req_low = {(a or "").strip().lower() for a in cand.required_anchors}
        # Must not keep the exact fused literal anchor.
        if baseline_low & cand_req_low:
            return None, ["not_cleaner_than_baseline"]

    return cand, []


# ---------------------------------------------------------------------
# System prompt — module-level constant (tests can hash/inspect it).
# ---------------------------------------------------------------------
SYSTEM_PROMPT: str = """You are the Full-Question ClinicalQuestionMap Interpreter for a clinician-facing evidence retrieval product.

You may use BROAD semantic understanding of the entire clinical question, but your authority is bounded — a deterministic validator decides whether your candidate may influence retrieval.

Hard contract — you are NOT allowed to:
- answer the clinical question
- search literature or any source
- select, rank, or score articles
- generate citations, PMIDs, DOIs, NCT IDs, or any article reference
- generate final/validated MeSH terms or final Boolean queries
- recommend treatment or produce clinical advice
- reference Google Scholar, ResearchGate, or Sci-Hub

You ONLY produce a structured ClinicalQuestionMap candidate as STRICT JSON for deterministic validation.

Output discipline:
- Return STRICT JSON only. No prose. No markdown code fences.
- Use ONLY values from the closed-set enums below. Never invent enum values.
- All listed fields are required. ``provenance`` MUST equal ``m4_full_question_interpreter_shadow``.
- Put the disease/problem in ``clinical_problem`` AND in ``required_anchors`` as a clean canonical concept (e.g., ``colon cancer``). NEVER keep a fused literal phrase such as ``prognosis of stage 1 colon cancer`` as a required anchor.
- Put stage in ``stage`` (e.g., ``stage I``), biomarker in ``biomarker`` (e.g., ``HER2-positive``), population context in ``population``. NEVER fuse stage/biomarker/population into a required anchor.
- Outcome / ranking words (prognosis, survival, overall survival, disease-free survival, recurrence, mortality) go in ``outcome`` and/or ``optional_or_ranking_only_terms`` — NEVER as a required anchor.
- ``what is`` must NOT force a disease overview when a stronger intent (prognosis, risk/etiology, mechanism, complication, dosing) is present.

Refusal envelope — refuse ONLY when the question is genuinely ambiguous, off-topic, unsafe, or lacks enough information to identify ANY clinical concept. If the question contains ANY stronger clinical intent (mechanism, pathophysiology, prognosis, prognostic, survival, recurrence, mortality, risk, cause, etiology, complication, dosing, screening, incidence, prevalence), you MUST produce a candidate — do NOT refuse. To refuse, return exactly:
  {"refused": true, "refusal_reason": "<reason>"}
with closed-set reason in {ambiguous_question, off_topic, unsafe_request, insufficient_information}.

Closed-set enums:
  primary_intent / secondary_intents elements ∈ {background_overview, disease_overview, drug_overview, definition,
    therapy, therapy_safety, drug_safety,
    diagnostic_accuracy, diagnosis, rule_out_test_utility, screening,
    risk_etiology, risk_etiology_harm, etiology,
    complication_risk, complication_after_procedure,
    epidemiology_prevalence_incidence,
    prognosis, prognosis_natural_history, prognostic_factor,
    mechanism, pathophysiology,
    severity_marker_overview, other}
  frameworks elements ∈ {PICO, PICOT, PICOTT, CoCoPop, PIRD, PFO,
    risk_etiology, mechanism, prognosis, complication, diagnostic_accuracy, narrative_overview}
  repair_reason ∈ {long_literal_phrase_anchor, intent_word_fused_in_anchor, what_is_forced_disease_overview,
    low_baseline_confidence, no_primary_entity_but_clinically_meaningful,
    stage_biomarker_population_fused, stop_grammar_required_anchor}

Schema fields (all required; use the empty defaults shown when not applicable):
  primary_intent: enum.
  secondary_intents: list[enum] (default []).
  frameworks: list[enum] (max 4, default []).
  clinical_problem: list[str] — the disease/problem (e.g., ["colon cancer"]).
  stage: list[str] — disease stage separated out (e.g., ["stage I"]).
  biomarker: list[str] — molecular markers (e.g., ["HER2-positive"]).
  population: list[str] — population/context (e.g., ["patients with stage I colon cancer"]).
  exposure / intervention / procedure / medication / index_test / outcome / setting / timeframe / mechanism / comparator: list[str] (default []).
  required_anchors: list[str] — clean canonical disease/drug/test/procedure concepts that MUST be retrievable. NEVER generic tokens alone, NEVER fused literal phrases, NEVER stage/outcome words.
  optional_or_ranking_only_terms: list[str] — boost/ranking-only concepts (default []).
  candidate_synonyms_unvalidated: list[str] — candidate search synonyms; explicitly NOT validated.
  candidate_mesh_terms_unvalidated: list[str] — candidate MeSH-like terms; NOT validated. Never prefix with ``MeSH:``.
  clarification_required: boolean — true only when the question is too generic to anchor.
  confidence: float 0.0..1.0. Use ≥0.7 only when you can cleanly separate the clinical concepts.
  repair_reason: copy ONE closed-set repair reason supplied in the user payload.
  provenance: MUST be ``m4_full_question_interpreter_shadow``.

Few-shot examples (match their structure):

Example A — prognosis with stage:
  Input: normalized_question: What is the prognosis of stage 1 colon cancer?
         repair_reason: long_literal_phrase_anchor
  Output: {"primary_intent":"prognosis_natural_history","secondary_intents":["prognostic_factor"],"frameworks":["PFO","prognosis"],"clinical_problem":["colon cancer"],"stage":["stage I"],"biomarker":[],"population":["patients with stage I colon cancer"],"exposure":[],"intervention":[],"procedure":[],"medication":[],"index_test":[],"outcome":["overall survival","disease-free survival","recurrence","mortality"],"setting":[],"timeframe":[],"mechanism":[],"comparator":[],"required_anchors":["colon cancer"],"optional_or_ranking_only_terms":["prognosis","survival","stage I"],"candidate_synonyms_unvalidated":["colon carcinoma","colorectal cancer"],"candidate_mesh_terms_unvalidated":["Colonic Neoplasms","Neoplasm Staging","Survival Rate"],"clarification_required":false,"confidence":0.9,"repair_reason":"long_literal_phrase_anchor","provenance":"m4_full_question_interpreter_shadow"}

Example D — prognosis with biomarker + stage:
  Input: normalized_question: What is the prognosis for HER2 positive stage 2 breast cancer?
         repair_reason: stage_biomarker_population_fused
  Output: {"primary_intent":"prognosis_natural_history","secondary_intents":[],"frameworks":["PFO","prognosis"],"clinical_problem":["breast cancer"],"stage":["stage II"],"biomarker":["HER2-positive"],"population":["patients with HER2-positive stage II breast cancer"],"exposure":[],"intervention":[],"procedure":[],"medication":[],"index_test":[],"outcome":["overall survival","recurrence","mortality"],"setting":[],"timeframe":[],"mechanism":[],"comparator":[],"required_anchors":["breast cancer"],"optional_or_ranking_only_terms":["prognosis","HER2-positive","stage II"],"candidate_synonyms_unvalidated":["breast carcinoma"],"candidate_mesh_terms_unvalidated":["Breast Neoplasms","Receptor, ErbB-2","Neoplasm Staging"],"clarification_required":false,"confidence":0.88,"repair_reason":"stage_biomarker_population_fused","provenance":"m4_full_question_interpreter_shadow"}

Example G — mechanism (note: ``what is`` lead-in must NOT cause refusal when a stronger intent like mechanism is present):
  Input: normalized_question: What is the mechanism of pruritus in cholestatic liver disease?
         repair_reason: what_is_forced_disease_overview
  Output: {"primary_intent":"mechanism","secondary_intents":["pathophysiology"],"frameworks":["mechanism"],"clinical_problem":["cholestatic liver disease"],"stage":[],"biomarker":[],"population":[],"exposure":[],"intervention":[],"procedure":[],"medication":[],"index_test":[],"outcome":[],"setting":[],"timeframe":[],"mechanism":["pruritus"],"comparator":[],"required_anchors":["cholestatic liver disease"],"optional_or_ranking_only_terms":["pruritus","itch"],"candidate_synonyms_unvalidated":["cholestasis","itching"],"candidate_mesh_terms_unvalidated":["Cholestasis","Pruritus"],"clarification_required":false,"confidence":0.86,"repair_reason":"what_is_forced_disease_overview","provenance":"m4_full_question_interpreter_shadow"}

Example H — procedure complication risk:
  Input: normalized_question: What is the risk of pneumothorax after central line placement?
         repair_reason: intent_word_fused_in_anchor
  Output: {"primary_intent":"complication_after_procedure","secondary_intents":["complication_risk"],"frameworks":["complication","PFO"],"clinical_problem":[],"stage":[],"biomarker":[],"population":[],"exposure":[],"intervention":[],"procedure":["central venous catheterization"],"medication":[],"index_test":[],"outcome":["pneumothorax"],"setting":[],"timeframe":[],"mechanism":[],"comparator":[],"required_anchors":["central venous catheterization"],"optional_or_ranking_only_terms":["pneumothorax","central line placement"],"candidate_synonyms_unvalidated":["central venous catheter insertion","central line"],"candidate_mesh_terms_unvalidated":["Catheterization, Central Venous","Pneumothorax"],"clarification_required":false,"confidence":0.85,"repair_reason":"intent_word_fused_in_anchor","provenance":"m4_full_question_interpreter_shadow"}

Note on bare overviews: the deterministic detector NEVER sends a bare ``What is X?`` disease/drug overview (no stronger intent) to you — so if you ARE invoked, assume a stronger intent exists and map it; do not refuse merely because the question begins with ``what is``.

Final reminder: JSON only. No code fences. Copy ONE supplied repair_reason value.
"""


def _build_user_payload(
    *, normalized_question: str, trigger_reasons: List[str],
) -> str:
    fr = (trigger_reasons[0] if trigger_reasons else "long_literal_phrase_anchor").strip()
    return (
        f"normalized_question: {normalized_question}\n"
        f"repair_reason: {fr}\n"
        "\n"
        "Emit a single JSON object matching the schema above. Return JSON only."
    )


async def run_m4_interpreter_shadow(
    *,
    normalized_question: str,
    trigger_reasons: List[str],
    deterministic_primary_anchors: List[dict],
    baseline_required_anchors: List[str],
    diagnostics: Optional[SessionDiagnostics] = None,
) -> tuple[Optional[M4Candidate], list[str], int]:
    """Run the bounded LLM and validate. Returns
    ``(candidate, rejection_reasons, latency_ms)``. Never raises.
    """
    if not trigger_reasons:
        return None, [], 0
    _ = diagnostics or diag_current()
    user_payload = _build_user_payload(
        normalized_question=normalized_question or "",
        trigger_reasons=trigger_reasons,
    )
    t0 = time.perf_counter()
    raw: Optional[dict] = None
    try:
        raw = await llm_provider.complete_json(
            system=SYSTEM_PROMPT,
            user=user_payload,
            temperature=0.0,
            max_tokens=1800,
        )
    except Exception:  # noqa: BLE001
        logger.exception("m4_full_question_interpreter llm call failed (non-fatal)")
        raw = None
    latency_ms = int(round((time.perf_counter() - t0) * 1000))
    if raw is None:
        return None, ["empty_output"], latency_ms
    candidate, rejections = validate_m4_candidate(
        raw,
        deterministic_primary_anchors=deterministic_primary_anchors,
        baseline_required_anchors=baseline_required_anchors,
    )
    if rejections:
        try:
            logger.warning(
                "m4_full_question_interpreter rejected reasons=%s",
                ",".join(sorted(set(rejections))),
            )
        except Exception:  # noqa: BLE001
            pass
    return candidate, rejections, latency_ms


def project_m4_compiler_input(
    *,
    candidate: Optional[M4Candidate],
    baseline_primary_intent: Optional[str],
    baseline_frameworks: List[str],
    baseline_required_anchors: List[str],
    baseline_branch_count: int,
) -> dict[str, Any]:
    """Reuse the deterministic would-be compiler-input projection. The
    candidate is mapped onto the projector's expected shape. Never
    invokes the compiler; never dispatches retrieval.
    """
    if candidate is None:
        return {}
    cand_dict = {
        "primary_intent": candidate.primary_intent,
        "frameworks": list(candidate.frameworks),
        "required_anchors": list(candidate.required_anchors),
        "branches": [],
        "repair_reason": candidate.repair_reason,
    }
    proj = project_would_be_compiler_input(
        candidate_dict=cand_dict,
        current_primary_intent=baseline_primary_intent,
        current_frameworks=list(baseline_frameworks or []),
        current_required_anchors=list(baseline_required_anchors or []),
        current_branch_count=int(baseline_branch_count or 0),
    )
    return proj


__all__ = [
    "M4_PROVENANCE",
    "M4_MODES",
    "M4_TRIGGER_REASONS",
    "M4_REJECTION_REASONS",
    "CONFIDENCE_THRESHOLD",
    "M4Candidate",
    "SYSTEM_PROMPT",
    "resolve_m4_mode",
    "detect_m4_triggers",
    "canonicalize_m4_candidate",
    "validate_m4_candidate",
    "run_m4_interpreter_shadow",
    "project_m4_compiler_input",
]
