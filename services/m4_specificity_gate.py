"""Phase M4.4 — Specificity Preservation / Do-No-Harm Gate (preview-only).

DETERMINISTIC. NO LLM. NO retrieval of its own. Pure functions.

Context
-------
The M4 active-rewrite physician adjudication (preview, no-UI) showed M4
*over-broadens* precise clinical questions: it drops or demotes central
clinical concepts (e.g. ``troponin`` from "does troponin elevation
predict mortality in sepsis"), generalizes a biomarker/stage condition
("HER2-positive stage 2 breast cancer" → bare "breast cancer"), and
broadens an already-precise prognosis literal phrase to the bare disease
("prognosis of idiopathic pulmonary fibrosis" → "idiopathic pulmonary
fibrosis"), which re-introduces wrong-disease noise.

This module is the deterministic do-no-harm gate that runs *before* M4
can replace the baseline query families. It NEVER calls an LLM and NEVER
runs retrieval — it only inspects:
  * the deterministic parser slots (baseline ``must_have_concepts`` with
    roles),
  * the baseline literal anchors,
  * the already-produced + validated M4 candidate schema (slots +
    ``required_anchors``),
  * deterministic question-family centrality policies.

Hard precedence (enforced by callers, restated here): PHI guard and the
K4A.2/.3/.4 protected-route exclusion DOMINATE this gate. This gate only
ever makes M4 *more* conservative — it can BLOCK an otherwise-eligible
promotion (preserve baseline), never force one.

Rule of thumb (closed): **M4 may apply only when the baseline is
degraded AND every central clinical concept for the question family is
preserved AND query-family diversity does not decrease AND there is no
baseline precision veto. If anything is uncertain → preserve baseline.**

Retrieval-level non-inferiority (overlap@10, anchor coverage,
wrong-anchor rate) is intentionally NOT computed here — per the M4.4
decision it remains an A/B adjudication-harness metric, not a runtime
criterion (the live clinician path chooses query families before
retrieval and must not double-retrieve).
"""
from __future__ import annotations

import re
from typing import Any, Optional

M4_SPECIFICITY_PHASE: str = "M4.5"

# ---- Closed-set final decisions surfaced in diagnostics. ----
FINAL_BASELINE_PRESERVED: str = "baseline_preserved"
FINAL_M4_APPLIED: str = "m4_applied"
FINAL_BLOCKED_K4A: str = "blocked_k4a"
FINAL_CLARIFICATION: str = "clarification"
FINAL_BLOCKED_PHI: str = "blocked_phi"

M4_SPECIFICITY_FINAL_DECISIONS: frozenset[str] = frozenset({
    FINAL_BASELINE_PRESERVED,
    FINAL_M4_APPLIED,
    FINAL_BLOCKED_K4A,
    FINAL_CLARIFICATION,
    FINAL_BLOCKED_PHI,
})

# ---- Phase M4.5 — closed-set application modes + query strategies. ----
APP_BASELINE_PRESERVED: str = "baseline_preserved"
APP_ADDITIVE_RESCUE: str = "additive_rescue"
APP_REPLACEMENT_BLOCKED: str = "replacement_blocked"

M4_APPLICATION_MODES: frozenset[str] = frozenset({
    APP_BASELINE_PRESERVED, APP_ADDITIVE_RESCUE, APP_REPLACEMENT_BLOCKED,
})

STRAT_BASELINE_ONLY: str = "baseline_only"
STRAT_BASELINE_PLUS_M4_RESCUE: str = "baseline_plus_m4_rescue"

M4_FINAL_QUERY_STRATEGIES: frozenset[str] = frozenset({
    STRAT_BASELINE_ONLY, STRAT_BASELINE_PLUS_M4_RESCUE,
})

# ---- Closed-set block reasons. ----
BLK_BASELINE_PRECISION_VETO: str = "baseline_precision_veto"
BLK_CENTRAL_CONCEPT_DROPPED: str = "central_concept_dropped"
BLK_CENTRAL_CONCEPT_DEMOTED: str = "central_concept_demoted"
BLK_MECHANISM_CONCEPT_DROPPED: str = "mechanism_concept_dropped"
BLK_PROGNOSTIC_FACTOR_DROPPED: str = "prognostic_factor_dropped"
BLK_OUTCOME_DROPPED: str = "outcome_dropped"
BLK_PROCEDURE_OR_DEVICE_DROPPED: str = "procedure_or_device_dropped"
BLK_COMPLICATION_DROPPED: str = "complication_dropped"
BLK_BIOMARKER_OR_STAGE_DEMOTED: str = "biomarker_or_stage_demoted"
BLK_PROGNOSIS_INTENT_DROPPED: str = "prognosis_intent_dropped"
BLK_QUERY_FAMILY_DIVERSITY_WOULD_DECREASE: str = "query_family_diversity_would_decrease"
BLK_HIGH_PRECISION_SPECIFICITY_WOULD_DECREASE: str = "high_precision_specificity_would_decrease"
BLK_ROUTE_OWNED_BY_K4A_SPECIALTY: str = "route_owned_by_k4a_specialty"
BLK_CANDIDATE_NOT_ACCEPTED: str = "candidate_not_accepted"
BLK_PHI_BLOCKED: str = "phi_blocked"
BLK_GENERIC_ONLY: str = "generic_only"
BLK_TIMEOUT_OR_SOURCE_UNAVAILABLE: str = "timeout_or_source_unavailable"
BLK_BASELINE_NOT_DEGRADED: str = "baseline_not_degraded"

M4_SPECIFICITY_BLOCK_REASONS: frozenset[str] = frozenset({
    BLK_BASELINE_PRECISION_VETO,
    BLK_CENTRAL_CONCEPT_DROPPED,
    BLK_CENTRAL_CONCEPT_DEMOTED,
    BLK_MECHANISM_CONCEPT_DROPPED,
    BLK_PROGNOSTIC_FACTOR_DROPPED,
    BLK_OUTCOME_DROPPED,
    BLK_PROCEDURE_OR_DEVICE_DROPPED,
    BLK_COMPLICATION_DROPPED,
    BLK_BIOMARKER_OR_STAGE_DEMOTED,
    BLK_PROGNOSIS_INTENT_DROPPED,
    BLK_QUERY_FAMILY_DIVERSITY_WOULD_DECREASE,
    BLK_HIGH_PRECISION_SPECIFICITY_WOULD_DECREASE,
    BLK_ROUTE_OWNED_BY_K4A_SPECIALTY,
    BLK_CANDIDATE_NOT_ACCEPTED,
    BLK_PHI_BLOCKED,
    BLK_GENERIC_ONLY,
    BLK_TIMEOUT_OR_SOURCE_UNAVAILABLE,
    BLK_BASELINE_NOT_DEGRADED,
})

# ---- Closed-set pass reasons. ----
PASS_BASELINE_DEGRADED_LITERAL_ANCHOR: str = "baseline_degraded_literal_anchor"
PASS_GENERIC_INTENT_REMOVED: str = "generic_intent_removed"
PASS_CENTRAL_CONCEPTS_PRESERVED: str = "central_concepts_preserved"
PASS_LITERAL_PHRASE_REMOVED: str = "literal_phrase_removed"
PASS_HIGH_PRECISION_FAMILY_PRESERVED: str = "high_precision_family_preserved"
PASS_QUERY_FAMILY_DIVERSITY_PRESERVED: str = "query_family_diversity_preserved"
PASS_REQUIRED_ANCHOR_SPECIFICITY_PRESERVED: str = "required_anchor_specificity_preserved"
PASS_M4_RESCUE_ALLOWED_BY_FAMILY_POLICY: str = "m4_rescue_allowed_by_family_policy"

M4_SPECIFICITY_PASS_REASONS: frozenset[str] = frozenset({
    PASS_BASELINE_DEGRADED_LITERAL_ANCHOR,
    PASS_GENERIC_INTENT_REMOVED,
    PASS_CENTRAL_CONCEPTS_PRESERVED,
    PASS_LITERAL_PHRASE_REMOVED,
    PASS_HIGH_PRECISION_FAMILY_PRESERVED,
    PASS_QUERY_FAMILY_DIVERSITY_PRESERVED,
    PASS_REQUIRED_ANCHOR_SPECIFICITY_PRESERVED,
    PASS_M4_RESCUE_ALLOWED_BY_FAMILY_POLICY,
})

# ---- Closed-set question families. ----
FAM_ETIOLOGY: str = "etiology"
FAM_MECHANISM: str = "mechanism"
FAM_PROGNOSIS: str = "prognosis"
FAM_PROGNOSTIC_FACTOR: str = "prognostic_factor"
FAM_BIOMARKER_STAGE_PROGNOSIS: str = "biomarker_stage_prognosis"
FAM_THERAPY: str = "therapy"
FAM_PROCEDURE_DEVICE_COMPLICATION: str = "procedure_device_complication"
FAM_BACKGROUND: str = "background"

# Families for which a single-clinical-concept literal phrase is treated
# as a *precise* phrase query (broadening to the bare disease loses
# precision → veto). Etiology is intentionally NOT here (a "causes of X"
# question genuinely benefits from a clean disease anchor).
_PRECISION_VETO_FAMILIES: frozenset[str] = frozenset({
    FAM_PROGNOSIS, FAM_MECHANISM, FAM_BACKGROUND,
})

# ---- Intent lexicons (lowercase, closed). ----
_PROGNOSIS_INTENT: frozenset[str] = frozenset({
    "prognosis", "prognostic", "survival", "mortality", "natural history",
    "life expectancy", "disease progression", "progression-free",
    "progression free", "overall survival", "recurrence", "death",
    "fatal", "outcome", "outcomes", "long-term outcome",
})
_MECHANISM_INTENT: frozenset[str] = frozenset({
    "mechanism", "mechanisms", "pathophysiology", "pathophysiologic",
    "pathogenesis", "molecular mechanism", "underlying mechanism",
})
_ETIOLOGY_INTENT: frozenset[str] = frozenset({
    "cause", "causes", "causing", "etiology", "aetiology", "risk factor",
    "risk factors", "leads to", "lead to",
})
_PREDICT_INTENT: frozenset[str] = frozenset({
    "predict", "predicts", "predictor", "predictors", "prognostic",
    "associated with mortality", "predictive of",
})
_COMPLICATION_INTENT: frozenset[str] = frozenset({
    "risk", "complication", "complications", "infection", "adverse",
})

# Stage / biomarker token detectors (mirror m4_active_pipeline closed set).
_STAGE_RE = re.compile(
    r"\bstage\s+(0|i{1,3}v?|iv|1|2|3|4)\b|\bt[0-4][a-c]?\b|\bn[0-3][a-c]?\b"
    r"|\bm[0-1]\b|\bgrade\s+(1|2|3|4|i{1,3}v?)\b",
    re.IGNORECASE,
)
_BIOMARKER_TOKENS: frozenset[str] = frozenset({
    "her2", "her-2", "her2-positive", "her2 positive", "er-positive",
    "er positive", "pr-positive", "egfr", "alk", "pd-l1", "pdl1",
    "braf", "kras", "ros1", "msi", "msi-high", "brca", "brca1", "brca2",
    "estrogen receptor", "ki-67", "ki67", "p53", "triple-negative",
    "triple negative",
})

# Bare generic/intent-only terms that must never be the sole required
# anchor of a rewritten question (shared shape with the M4.1.3 rescue
# rejection list).
_GENERIC_ANCHOR_TERMS: frozenset[str] = frozenset({
    "dose", "doses", "dosage", "dosing", "treatment", "treatments",
    "therapy", "therapies", "management", "risk", "risks", "prognosis",
    "survival", "recurrence", "cause", "causes", "etiology", "aetiology",
    "mechanism", "pathophysiology", "diagnosis", "outcome", "outcomes",
    "side effect", "side effects", "adverse effect", "adverse effects",
    "guideline", "guidelines", "incidence", "prevalence", "rate",
    "marker", "markers", "symptom", "symptoms", "sign", "signs",
    "infection", "complication", "complications",
})

# Population / context tokens that do NOT count as a distinct clinical
# concept when deciding single-vs-multi-concept for the precision veto.
_POPULATION_TOKENS: frozenset[str] = frozenset({
    "young", "adults", "adult", "children", "child", "elderly", "older",
    "patients", "patient", "women", "woman", "men", "man", "pregnancy",
    "pregnant", "neonates", "neonatal", "infants", "infant", "adolescents",
    "adolescent", "geriatric", "icu", "outpatient", "inpatient", "people",
    "population", "individuals",
})

_INTENT_LEAD_PHRASES: tuple[str, ...] = (
    "what is the prognosis of", "what is the mechanism of",
    "what are the causes of", "what is the cause of",
    "prognosis of", "mechanism of", "mechanisms of",
    "pathophysiology of", "pathogenesis of", "causes of", "cause of",
    "etiology of", "aetiology of", "natural history of",
    "risk factors for", "risk factors of", "risk of",
    "markers of severity in", "markers of severity of", "markers of",
    "incidence of", "prevalence of", "complications of", "complication of",
)
_CONNECTIVES: tuple[str, ...] = (
    " in ", " after ", " among ", " due to ", " following ",
    " with ", " of ",
)

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\- ]+")
_CORE_STOP: frozenset[str] = frozenset({
    "of", "in", "the", "a", "an", "with", "and", "or", "to", "for",
    "after", "among", "due", "is", "are", "what", "does", "do",
})


def _norm(s: Any) -> str:
    txt = str(s or "").lower().strip()
    txt = _PUNCT_RE.sub(" ", txt)
    return _WS_RE.sub(" ", txt).strip()


def _toks(s: Any) -> list[str]:
    return [t for t in _norm(s).split() if t]


def _core_tokens(s: Any) -> list[str]:
    return [t for t in _toks(s) if t not in _CORE_STOP]


def _contains_any(text: str, terms) -> bool:
    nt = _norm(text)
    tt = set(nt.split())
    for term in terms:
        nterm = _norm(term)
        if not nterm:
            continue
        if " " in nterm or "-" in nterm:
            if nterm in nt:
                return True
        elif nterm in tt:
            return True
    return False


def _concept_in_text(concept: str, text: str) -> bool:
    """True if ``concept`` is represented in ``text`` — either as a
    normalized-substring phrase match or all of its core tokens present.
    """
    nc = _norm(concept)
    if not nc:
        return False
    nt = _norm(text)
    if nc in nt:
        return True
    ct = _core_tokens(concept)
    if not ct:
        return False
    tt = set(nt.split())
    return all(t in tt for t in ct)


def _concept_in_any(concept: str, texts) -> bool:
    return any(_concept_in_text(concept, t) for t in (texts or []))


def _slot_list(candidate: dict, key: str) -> list[str]:
    v = candidate.get(key)
    return [str(x) for x in v if isinstance(x, str) and x.strip()] if isinstance(v, list) else []


def _candidate_hard_terms(candidate: dict) -> list[str]:
    """Every concept the M4 candidate carries in a *hard* slot (i.e.
    one the compiler ANDs to the required core in at least one family),
    EXCLUDING ``optional_or_ranking_only_terms``.
    """
    out: list[str] = []
    for key in (
        "clinical_problem", "stage", "biomarker", "population", "exposure",
        "intervention", "procedure", "medication", "index_test", "outcome",
        "mechanism", "comparator", "required_anchors",
    ):
        out.extend(_slot_list(candidate, key))
    return out


def decompose_literal_anchor(phrase: str) -> dict[str, Any]:
    """Deterministically decompose a fused literal-phrase anchor into
    its clinical fragments. Pure.

    Returns ``{"fragments": [...], "clinical_fragments": [...],
    "intent_removed": bool}``. ``clinical_fragments`` excludes
    population-only fragments (used for single-vs-multi-concept veto
    detection).
    """
    p = _norm(phrase)
    intent_removed = False
    for lead in _INTENT_LEAD_PHRASES:
        nlead = _norm(lead)
        if p.startswith(nlead + " ") or p == nlead:
            p = p[len(nlead):].strip()
            intent_removed = True
            break
    # Split on connectives.
    frags = [p]
    for conn in _CONNECTIVES:
        nconn = conn  # already spaced
        new: list[str] = []
        for fr in frags:
            new.extend(re.split(nconn, f" {fr} "))
        frags = [f.strip() for f in new if f.strip()]
    frags = [f for f in frags if f]
    clinical = [
        f for f in frags
        if not all(t in _POPULATION_TOKENS for t in _toks(f))
    ]
    return {
        "fragments": frags,
        "clinical_fragments": clinical,
        "intent_removed": intent_removed,
    }


def detect_question_family(
    *,
    normalized_question: str,
    baseline_type_of_question: Optional[str],
    baseline_question_subtype: Optional[str],
    candidate: dict,
) -> str:
    """Closed-set, deterministic question-family detection from all
    available signals (subtype, type-of-question, candidate intent,
    intent lexicons over the question text). Pure."""
    q = _norm(normalized_question)
    subtype = _norm(baseline_question_subtype)
    toq = _norm(baseline_type_of_question)
    cand_intent = _norm(candidate.get("primary_intent"))
    blob = " ".join([subtype, toq, cand_intent])

    has_biomarker_or_stage = bool(
        _slot_list(candidate, "biomarker")
        or _slot_list(candidate, "stage")
        or _STAGE_RE.search(q)
        or _contains_any(q, _BIOMARKER_TOKENS)
    )
    factor_present = bool(
        _slot_list(candidate, "index_test") or _slot_list(candidate, "exposure")
    )
    outcome_present = bool(_slot_list(candidate, "outcome"))

    # 1) Prognostic factor — "does <factor> predict <outcome> in <cond>".
    if (
        "prognostic factor" in blob
        or (_contains_any(q, _PREDICT_INTENT) and factor_present and outcome_present)
    ):
        return FAM_PROGNOSTIC_FACTOR

    prognosis_signal = (
        "prognosis" in blob or "natural history" in blob
        or _contains_any(q, _PROGNOSIS_INTENT)
    )
    mechanism_signal = (
        "mechanism" in blob or "pathophysiology" in blob
        or "pathogenesis" in blob or _contains_any(q, _MECHANISM_INTENT)
    )
    etiology_signal = (
        "etiology" in blob or "etiology harm" in blob or "harm" in blob
        or _contains_any(q, _ETIOLOGY_INTENT)
    )

    # 2) Biomarker/stage prognosis dominates plain prognosis.
    if prognosis_signal and has_biomarker_or_stage:
        return FAM_BIOMARKER_STAGE_PROGNOSIS
    if prognosis_signal:
        return FAM_PROGNOSIS
    if mechanism_signal:
        return FAM_MECHANISM
    if etiology_signal:
        return FAM_ETIOLOGY

    # 3) Procedure / device complication.
    procedure_present = bool(
        _slot_list(candidate, "procedure") or _slot_list(candidate, "medication")
    )
    if procedure_present and _contains_any(q, _COMPLICATION_INTENT):
        return FAM_PROCEDURE_DEVICE_COMPLICATION

    # 4) Therapy when an intervention is actually named.
    if ("therapy" in blob or "treatment" in blob) and _slot_list(candidate, "intervention"):
        return FAM_THERAPY

    return FAM_BACKGROUND



# ---- Central-concept categories (closed set). ----
CAT_CONDITION: str = "condition"
CAT_PROGNOSTIC_FACTOR: str = "prognostic_factor"
CAT_OUTCOME: str = "outcome"
CAT_MECHANISM: str = "mechanism"
CAT_NAMED_PROCESS: str = "named_process"
CAT_BIOMARKER: str = "biomarker"
CAT_STAGE: str = "stage"
CAT_PROCEDURE_OR_DEVICE: str = "procedure_or_device"
CAT_COMPLICATION: str = "complication"
CAT_EXPOSURE: str = "exposure"
CAT_INTERVENTION: str = "intervention"

# Baseline must-have roles → central category. Roles outside this map
# (population/context/setting/comparator/study_design) are NOT central.
_ROLE_TO_CATEGORY: dict[str, str] = {
    "condition": CAT_CONDITION,
    "outcome": CAT_OUTCOME,
    "index_test": CAT_PROGNOSTIC_FACTOR,
    "exposure": CAT_EXPOSURE,
    "drug_or_drug_class": CAT_EXPOSURE,
    "intervention": CAT_INTERVENTION,
    "procedure": CAT_PROCEDURE_OR_DEVICE,
    "mechanism": CAT_MECHANISM,
    "biomarker": CAT_BIOMARKER,
}


def build_central_concept_ledger(
    *,
    question_family: str,
    normalized_question: str,
    baseline_must_have_concepts: list[dict],
    baseline_literal_anchors: list[str],
    candidate: dict,
) -> dict[str, Any]:
    """Phase M4.4 — deterministically map parsed slots + candidate schema
    to the closed-set central clinical concepts the question family
    requires. NO LLM. Pure.

    Returns ``{"family": str, "concepts": [{"text","category",
    "source"}...], "condition": str|None}``.
    """
    concepts: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(text: str, category: str, source: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        key = (_norm(t), category)
        if key in seen:
            return
        seen.add(key)
        concepts.append({"text": t, "category": category, "source": source})

    # 1) Baseline parser must-have concepts (authoritative roles).
    for mh in (baseline_must_have_concepts or []):
        if not isinstance(mh, dict):
            continue
        concept = mh.get("concept") or ""
        role = _norm(mh.get("role"))
        cat = _ROLE_TO_CATEGORY.get(role)
        n = _norm(concept)
        # A fused literal phrase carrying an intent verb is decomposed
        # below; do not register the whole phrase as a condition.
        is_fused_literal = (
            len(_core_tokens(concept)) >= 4
            or _contains_any(n, _PROGNOSIS_INTENT | _MECHANISM_INTENT
                             | _ETIOLOGY_INTENT)
        )
        if cat and not (cat == CAT_CONDITION and is_fused_literal):
            _add(concept, cat, "baseline_must_have")

    # 2) Candidate structured slots (the M4 decomposition of the
    #    question — used to know what concepts the question is about).
    for txt in _slot_list(candidate, "clinical_problem"):
        _add(txt, CAT_CONDITION, "candidate_slot")
    for txt in _slot_list(candidate, "biomarker"):
        _add(txt, CAT_BIOMARKER, "candidate_slot")
    for txt in _slot_list(candidate, "stage"):
        _add(txt, CAT_STAGE, "candidate_slot")
    for txt in _slot_list(candidate, "index_test"):
        _add(txt, CAT_PROGNOSTIC_FACTOR, "candidate_slot")
    for txt in _slot_list(candidate, "exposure"):
        _add(txt, CAT_EXPOSURE, "candidate_slot")
    for txt in _slot_list(candidate, "intervention"):
        _add(txt, CAT_INTERVENTION, "candidate_slot")
    for txt in _slot_list(candidate, "procedure"):
        _add(txt, CAT_PROCEDURE_OR_DEVICE, "candidate_slot")
    for txt in _slot_list(candidate, "medication"):
        _add(txt, CAT_PROCEDURE_OR_DEVICE, "candidate_slot")
    for txt in _slot_list(candidate, "mechanism"):
        _add(txt, CAT_MECHANISM, "candidate_slot")
    for txt in _slot_list(candidate, "outcome"):
        # An outcome that names a complication (infection) is tagged as
        # such for procedure/device questions.
        cat = (
            CAT_COMPLICATION
            if (question_family == FAM_PROCEDURE_DEVICE_COMPLICATION
                and _contains_any(txt, _COMPLICATION_INTENT))
            else CAT_OUTCOME
        )
        _add(txt, cat, "candidate_slot")

    # 3) Decompose fused literal anchors → condition + named process.
    cand_condition = (_slot_list(candidate, "clinical_problem") or [None])[0]
    for anc in (baseline_literal_anchors or []):
        dec = decompose_literal_anchor(anc)
        clinical = dec["clinical_fragments"]
        if not dec["intent_removed"]:
            continue
        for frag in clinical:
            # The disease fragment (matches candidate condition) → condition,
            # any other clinical fragment → named process.
            if cand_condition and _concept_in_text(frag, cand_condition):
                _add(frag, CAT_CONDITION, "literal_decomposition")
            elif cand_condition and _concept_in_text(cand_condition, frag):
                _add(frag, CAT_CONDITION, "literal_decomposition")
            elif all(t in _POPULATION_TOKENS for t in _toks(frag)):
                continue
            else:
                # Decide condition vs named process: the LAST clinical
                # fragment is conventionally the disease ("X of Y in Z").
                cat = (
                    CAT_CONDITION if frag == clinical[-1] and not cand_condition
                    else CAT_NAMED_PROCESS
                )
                _add(frag, cat, "literal_decomposition")

    condition_txt = None
    for c in concepts:
        if c["category"] == CAT_CONDITION:
            condition_txt = c["text"]
            break

    return {
        "family": question_family,
        "concepts": concepts,
        "condition": condition_txt,
    }


def assess_baseline_precision_veto(
    *,
    question_family: str,
    baseline_literal_anchors: list[str],
    ledger: dict,
    m4_required_anchors: list[str],
) -> dict[str, Any]:
    """Phase M4.4 — veto broadening of an already-precise baseline.

    Fires (veto=True → preserve baseline) when the question family is
    prognosis / mechanism / background AND the baseline anchor is a
    single-clinical-concept fused literal phrase (e.g. "prognosis of
    idiopathic pulmonary fibrosis") AND the M4 rewrite would merely
    generalize to the bare disease. A two-clinical-concept phrase
    (e.g. "mechanism of thrombosis in antiphospholipid syndrome") does
    NOT veto here — it is gated by the central-concept check instead.
    Pure.
    """
    reasons: list[str] = []
    if question_family not in _PRECISION_VETO_FAMILIES:
        return {"veto": False, "reasons": reasons}
    for anc in (baseline_literal_anchors or []):
        dec = decompose_literal_anchor(anc)
        if not dec["intent_removed"]:
            continue
        clinical = dec["clinical_fragments"]
        if len(clinical) != 1:
            continue
        single = clinical[0]
        # M4 must be broadening to (a generalization of) that single
        # concept — i.e. every M4 required anchor is contained within
        # the single baseline clinical concept.
        req = [a for a in (m4_required_anchors or []) if str(a).strip()]
        if req and all(_concept_in_text(a, single) for a in req):
            if question_family == FAM_PROGNOSIS:
                reasons.append("single_concept_prognosis_literal_phrase_baseline_precise")
            elif question_family == FAM_MECHANISM:
                reasons.append("single_concept_mechanism_literal_phrase_baseline_precise")
            else:
                reasons.append("single_concept_overview_literal_phrase_baseline_precise")
    return {"veto": bool(reasons), "reasons": reasons}


def _in_required(concept: str, required: list[str]) -> bool:
    return _concept_in_any(concept, required)


def _in_hard_slot(concept: str, hard_terms: list[str]) -> bool:
    return _concept_in_any(concept, hard_terms)


def _only_optional(concept: str, optional_terms: list[str],
                   required: list[str], hard_terms: list[str]) -> bool:
    return (
        _concept_in_any(concept, optional_terms)
        and not _in_required(concept, required)
        and not _in_hard_slot(concept, hard_terms)
    )


def evaluate_specificity_delta(
    *,
    question_family: str,
    ledger: dict,
    candidate: dict,
) -> dict[str, Any]:
    """Phase M4.4 — closed-set central-concept preservation + specificity
    delta against the M4 candidate structure. Pure. NO LLM.

    A central concept is:
      * REQUIRED-preserved  → in ``required_anchors`` (ANDed everywhere)
      * PRECISION-preserved → in a hard slot (ANDed to core in ≥1 family)
      * DEMOTED             → only in ``optional_or_ranking_only_terms``
      * DROPPED             → present nowhere
    """
    required = _slot_list(candidate, "required_anchors")
    optional = _slot_list(candidate, "optional_or_ranking_only_terms")
    hard_terms = _candidate_hard_terms(candidate)
    concepts = ledger.get("concepts") or []

    status: dict[str, str] = {}

    def classify(text: str) -> str:
        if _in_required(text, required):
            return "required"
        if _in_hard_slot(text, hard_terms):
            return "precision"
        if _concept_in_any(text, optional):
            return "demoted"
        return "dropped"

    by_cat: dict[str, list[tuple[str, str]]] = {}
    for c in concepts:
        st = classify(c["text"])
        status[f"{c['category']}:{c['text']}"] = st
        by_cat.setdefault(c["category"], []).append((c["text"], st))

    block_reasons: list[str] = []
    pass_reasons: list[str] = []

    def cat_states(cat: str) -> list[str]:
        return [st for (_t, st) in by_cat.get(cat, [])]

    def cond_ok() -> bool:
        states = cat_states(CAT_CONDITION)
        # Condition must be preserved at required level (ANDed everywhere)
        # for any non-background family. If no condition concept was
        # detected at all, treat as not-ok (preserve baseline).
        return bool(states) and any(s == "required" for s in states)

    # ---- Condition is mandatory + required for every clinical family. ----
    if question_family != FAM_BACKGROUND:
        if not cond_ok():
            block_reasons.append(BLK_CENTRAL_CONCEPT_DROPPED)
        else:
            pass_reasons.append(PASS_REQUIRED_ANCHOR_SPECIFICITY_PRESERVED)

    if question_family == FAM_PROGNOSTIC_FACTOR:
        # condition + prognostic factor + outcome ALL required (ANDed).
        factor_states = cat_states(CAT_PROGNOSTIC_FACTOR) + cat_states(CAT_EXPOSURE) + cat_states(CAT_BIOMARKER)
        outcome_states = cat_states(CAT_OUTCOME)
        if not factor_states or not any(s == "required" for s in factor_states):
            block_reasons.append(BLK_PROGNOSTIC_FACTOR_DROPPED)
        if not outcome_states or not any(s == "required" for s in outcome_states):
            block_reasons.append(BLK_OUTCOME_DROPPED)

    elif question_family == FAM_BIOMARKER_STAGE_PROGNOSIS:
        # condition + biomarker + stage must BOTH be required (ANDed
        # together) — a population OR-group demotes them.
        bm = cat_states(CAT_BIOMARKER)
        stg = cat_states(CAT_STAGE)
        bm_req = bool(bm) and any(s == "required" for s in bm)
        stg_req = bool(stg) and any(s == "required" for s in stg)
        if (bm and not bm_req) or (stg and not stg_req) or (not bm and not stg):
            block_reasons.append(BLK_BIOMARKER_OR_STAGE_DEMOTED)
        prog = cat_states(CAT_OUTCOME) + cat_states(CAT_MECHANISM)
        if not any(s in ("required", "precision") for s in prog):
            block_reasons.append(BLK_PROGNOSIS_INTENT_DROPPED)

    elif question_family == FAM_PROGNOSIS:
        # condition required + prognosis intent preserved (precision ok).
        prog = cat_states(CAT_OUTCOME)
        if not prog or not any(s in ("required", "precision") for s in prog):
            block_reasons.append(BLK_PROGNOSIS_INTENT_DROPPED)
        else:
            pass_reasons.append(PASS_HIGH_PRECISION_FAMILY_PRESERVED)

    elif question_family == FAM_MECHANISM:
        # condition required + named process preserved (precision ok)
        # + mechanism intent term preserved (precision ok).
        proc = cat_states(CAT_NAMED_PROCESS) + cat_states(CAT_OUTCOME)
        if proc and not any(s in ("required", "precision") for s in proc):
            block_reasons.append(BLK_CENTRAL_CONCEPT_DROPPED)
        mech = cat_states(CAT_MECHANISM)
        if not mech or not any(s in ("required", "precision") for s in mech):
            block_reasons.append(BLK_MECHANISM_CONCEPT_DROPPED)
        else:
            pass_reasons.append(PASS_HIGH_PRECISION_FAMILY_PRESERVED)

    elif question_family == FAM_ETIOLOGY:
        # condition required; named exposure (if present) preserved.
        exp = cat_states(CAT_EXPOSURE)
        if exp and not any(s in ("required", "precision") for s in exp):
            block_reasons.append(BLK_CENTRAL_CONCEPT_DROPPED)

    elif question_family == FAM_PROCEDURE_DEVICE_COMPLICATION:
        proc = cat_states(CAT_PROCEDURE_OR_DEVICE)
        comp = cat_states(CAT_COMPLICATION) + cat_states(CAT_OUTCOME)
        if not proc or not any(s == "required" for s in proc):
            block_reasons.append(BLK_PROCEDURE_OR_DEVICE_DROPPED)
        if not comp or not any(s in ("required", "precision") for s in comp):
            block_reasons.append(BLK_COMPLICATION_DROPPED)

    elif question_family == FAM_THERAPY:
        interv = cat_states(CAT_INTERVENTION)
        if interv and not any(s in ("required", "precision") for s in interv):
            block_reasons.append(BLK_CENTRAL_CONCEPT_DEMOTED)

    # Generic demotion sweep: any central concept demoted to optional-only
    # is a specificity loss (skip population/comparator which are not in
    # the ledger anyway).
    for c in concepts:
        st = status.get(f"{c['category']}:{c['text']}")
        if st == "demoted" and BLK_CENTRAL_CONCEPT_DEMOTED not in block_reasons:
            block_reasons.append(BLK_CENTRAL_CONCEPT_DEMOTED)

    if not block_reasons:
        pass_reasons.append(PASS_CENTRAL_CONCEPTS_PRESERVED)

    # Dedupe preserving order.
    def _dd(seq):
        s: set[str] = set()
        return [x for x in seq if not (x in s or s.add(x))]

    return {
        "central_concept_preservation_status": status,
        "specificity_delta": {
            cat: [{"concept": t, "status": st} for (t, st) in items]
            for cat, items in by_cat.items()
        },
        "block_reasons": _dd(block_reasons),
        "pass_reasons": _dd(pass_reasons),
    }


def run_specificity_gate(
    *,
    normalized_question: str,
    baseline_type_of_question: Optional[str],
    baseline_question_subtype: Optional[str],
    baseline_must_have_concepts: list[dict],
    baseline_literal_anchors: list[str],
    m4_candidate: dict,
    baseline_degraded: bool,
    baseline_degradation_reasons: Optional[list[str]] = None,
    baseline_query_family_count: Optional[int] = None,
    m4_query_family_count: int = 0,
    route_owned_by_k4a_specialty: bool = False,
    candidate_accepted: bool = True,
    upstream_timed_out: bool = False,
    additive_mode: bool = True,
    m4_rescue_family_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Phase M4.4/M4.5 — top-level deterministic specificity-preservation /
    do-no-harm gate. NEVER raises (best-effort closed-set output).

    Phase M4.5: when ``additive_mode=True`` (default) M4 does NOT replace
    the baseline query families — it PRESERVES them and ADDS validated
    rescue families. Query-family diversity is therefore preserved or
    increased, so the diversity guard only blocks a degenerate rescue that
    would collapse a multi-concept question into a broad disease-only
    family. Returns ``allow`` + the full closed-set diagnostics block.
    """
    candidate = m4_candidate if isinstance(m4_candidate, dict) else {}
    deg_reasons = list(baseline_degradation_reasons or [])
    m4_required = _slot_list(candidate, "required_anchors")
    rescue_names = [str(n) for n in (m4_rescue_family_names or []) if str(n).strip()]

    block_reasons: list[str] = []
    pass_reasons: list[str] = []

    # ---- Dominant precedence guards (do not even build the ledger). ----
    if route_owned_by_k4a_specialty:
        return _gate_result(
            allow=False, final=FINAL_BLOCKED_K4A,
            family=None, ledger={"concepts": []},
            baseline_degraded=baseline_degraded, deg_reasons=deg_reasons,
            precision_veto=False, precision_reasons=[],
            preservation={}, block_reasons=[BLK_ROUTE_OWNED_BY_K4A_SPECIALTY],
            pass_reasons=[], diversity_before=baseline_query_family_count,
            diversity_after=baseline_query_family_count or 0,
            application_mode=APP_BASELINE_PRESERVED, added_rescue_families=[],
            baseline_families_preserved=True,
            final_query_strategy=STRAT_BASELINE_ONLY, diversity_guard_passed=True,
        )
    if upstream_timed_out:
        return _gate_result(
            allow=False, final=FINAL_BASELINE_PRESERVED, family=None,
            ledger={"concepts": []}, baseline_degraded=baseline_degraded,
            deg_reasons=deg_reasons, precision_veto=False, precision_reasons=[],
            preservation={}, block_reasons=[BLK_TIMEOUT_OR_SOURCE_UNAVAILABLE],
            pass_reasons=[], diversity_before=baseline_query_family_count,
            diversity_after=baseline_query_family_count or 0,
            application_mode=APP_BASELINE_PRESERVED, added_rescue_families=[],
            baseline_families_preserved=True,
            final_query_strategy=STRAT_BASELINE_ONLY, diversity_guard_passed=True,
        )
    if not candidate_accepted or not candidate:
        return _gate_result(
            allow=False, final=FINAL_BASELINE_PRESERVED, family=None,
            ledger={"concepts": []}, baseline_degraded=baseline_degraded,
            deg_reasons=deg_reasons, precision_veto=False, precision_reasons=[],
            preservation={}, block_reasons=[BLK_CANDIDATE_NOT_ACCEPTED],
            pass_reasons=[], diversity_before=baseline_query_family_count,
            diversity_after=baseline_query_family_count or 0,
            application_mode=APP_BASELINE_PRESERVED, added_rescue_families=[],
            baseline_families_preserved=True,
            final_query_strategy=STRAT_BASELINE_ONLY, diversity_guard_passed=True,
        )

    family = detect_question_family(
        normalized_question=normalized_question,
        baseline_type_of_question=baseline_type_of_question,
        baseline_question_subtype=baseline_question_subtype,
        candidate=candidate,
    )
    ledger = build_central_concept_ledger(
        question_family=family,
        normalized_question=normalized_question,
        baseline_must_have_concepts=baseline_must_have_concepts or [],
        baseline_literal_anchors=baseline_literal_anchors or [],
        candidate=candidate,
    )

    # ---- Baseline must be explicitly degraded to permit any rewrite. ----
    if not baseline_degraded:
        block_reasons.append(BLK_BASELINE_NOT_DEGRADED)
    else:
        pass_reasons.append(PASS_BASELINE_DEGRADED_LITERAL_ANCHOR)

    # ---- Generic-only required anchor sweep. ----
    if m4_required and all(
        _norm(a) in _GENERIC_ANCHOR_TERMS for a in m4_required
    ):
        block_reasons.append(BLK_GENERIC_ONLY)

    # ---- Baseline precision veto. ----
    veto = assess_baseline_precision_veto(
        question_family=family,
        baseline_literal_anchors=baseline_literal_anchors or [],
        ledger=ledger,
        m4_required_anchors=m4_required,
    )
    if veto["veto"]:
        block_reasons.append(BLK_BASELINE_PRECISION_VETO)
    else:
        pass_reasons.append(PASS_LITERAL_PHRASE_REMOVED)

    # ---- Phase M4.5 — query-family diversity (additive-aware). ----
    diversity_before = baseline_query_family_count
    diversity_guard_passed = True
    if additive_mode:
        # Baseline families are PRESERVED and M4 rescue families are ADDED,
        # so diversity is preserved or increased. The only diversity harm is
        # a degenerate rescue that collapses a multi-concept question to a
        # broad disease-only family (no high-precision rescue family).
        diversity_after = int(diversity_before or 0) + int(m4_query_family_count)
        named_central = len({
            c["category"] for c in (ledger.get("concepts") or [])
        })
        has_high_precision_rescue = (
            len(rescue_names) >= 2 or int(m4_query_family_count) >= 2
        )
        if named_central >= 2 and not has_high_precision_rescue:
            block_reasons.append(BLK_HIGH_PRECISION_SPECIFICITY_WOULD_DECREASE)
            diversity_guard_passed = False
        else:
            pass_reasons.append(PASS_QUERY_FAMILY_DIVERSITY_PRESERVED)
    else:
        diversity_after = int(m4_query_family_count)
        if (
            diversity_before is not None
            and int(m4_query_family_count) < int(diversity_before)
        ):
            block_reasons.append(BLK_QUERY_FAMILY_DIVERSITY_WOULD_DECREASE)
            diversity_guard_passed = False
        else:
            pass_reasons.append(PASS_QUERY_FAMILY_DIVERSITY_PRESERVED)

    # ---- Central-concept preservation + specificity delta. ----
    preservation = evaluate_specificity_delta(
        question_family=family, ledger=ledger, candidate=candidate,
    )
    block_reasons.extend(preservation["block_reasons"])
    pass_reasons.extend(preservation["pass_reasons"])

    def _dd(seq):
        s: set[str] = set()
        return [x for x in seq if not (x in s or s.add(x))]

    block_reasons = _dd(block_reasons)
    pass_reasons = _dd(pass_reasons)
    allow = not block_reasons
    if allow:
        pass_reasons = _dd(pass_reasons + [PASS_M4_RESCUE_ALLOWED_BY_FAMILY_POLICY])

    # ---- Closed-set application mode + final strategy. ----
    if allow:
        application_mode = APP_ADDITIVE_RESCUE
        final_query_strategy = STRAT_BASELINE_PLUS_M4_RESCUE
        added_rescue_families = list(rescue_names)
    else:
        diversity_block = (
            BLK_HIGH_PRECISION_SPECIFICITY_WOULD_DECREASE in block_reasons
            or BLK_QUERY_FAMILY_DIVERSITY_WOULD_DECREASE in block_reasons
        )
        application_mode = (
            APP_REPLACEMENT_BLOCKED if diversity_block else APP_BASELINE_PRESERVED
        )
        final_query_strategy = STRAT_BASELINE_ONLY
        added_rescue_families = []

    return _gate_result(
        allow=allow,
        final=FINAL_M4_APPLIED if allow else FINAL_BASELINE_PRESERVED,
        family=family, ledger=ledger, baseline_degraded=baseline_degraded,
        deg_reasons=deg_reasons, precision_veto=bool(veto["veto"]),
        precision_reasons=list(veto["reasons"]),
        preservation=preservation, block_reasons=block_reasons,
        pass_reasons=pass_reasons,
        diversity_before=diversity_before,
        diversity_after=diversity_after,
        application_mode=application_mode,
        added_rescue_families=added_rescue_families,
        baseline_families_preserved=True,
        final_query_strategy=final_query_strategy,
        diversity_guard_passed=diversity_guard_passed,
    )


def _gate_result(
    *, allow: bool, final: str, family: Optional[str], ledger: dict,
    baseline_degraded: bool, deg_reasons: list[str], precision_veto: bool,
    precision_reasons: list[str], preservation: dict,
    block_reasons: list[str], pass_reasons: list[str],
    diversity_before: Optional[int], diversity_after: int,
    application_mode: str = APP_BASELINE_PRESERVED,
    added_rescue_families: Optional[list[str]] = None,
    baseline_families_preserved: bool = True,
    final_query_strategy: str = STRAT_BASELINE_ONLY,
    diversity_guard_passed: bool = True,
) -> dict[str, Any]:
    """Assemble the closed-set diagnostics block."""
    before = diversity_before if diversity_before is not None else 0
    return {
        "allow": bool(allow),
        "m4_specificity_phase": M4_SPECIFICITY_PHASE,
        "final_decision": final,
        "question_family": family,
        "central_concept_ledger": ledger,
        "baseline_degraded": bool(baseline_degraded),
        "baseline_degradation_reasons": list(deg_reasons),
        "baseline_precision_veto": bool(precision_veto),
        "baseline_precision_veto_reasons": list(precision_reasons),
        "central_concept_preservation_status": (
            preservation.get("central_concept_preservation_status") or {}
        ),
        "specificity_delta": preservation.get("specificity_delta") or {},
        "query_family_diversity_before": diversity_before,
        "query_family_diversity_after": int(diversity_after),
        "diversity_delta": int(diversity_after) - int(before),
        "diversity_guard_passed": bool(diversity_guard_passed),
        # ---- Phase M4.5 — additive rescue diagnostics. ----
        "m4_application_mode": application_mode,
        "baseline_families_preserved": bool(baseline_families_preserved),
        "added_m4_rescue_families": list(added_rescue_families or []),
        "final_query_strategy": final_query_strategy,
        "m4_specificity_block_reasons": list(block_reasons),
        "m4_specificity_pass_reasons": list(pass_reasons),
    }


__all__ = [
    "M4_SPECIFICITY_PHASE",
    "M4_SPECIFICITY_FINAL_DECISIONS",
    "M4_SPECIFICITY_BLOCK_REASONS",
    "M4_SPECIFICITY_PASS_REASONS",
    "M4_APPLICATION_MODES",
    "M4_FINAL_QUERY_STRATEGIES",
    "APP_BASELINE_PRESERVED",
    "APP_ADDITIVE_RESCUE",
    "APP_REPLACEMENT_BLOCKED",
    "STRAT_BASELINE_ONLY",
    "STRAT_BASELINE_PLUS_M4_RESCUE",
    "FINAL_BASELINE_PRESERVED",
    "FINAL_M4_APPLIED",
    "FINAL_BLOCKED_K4A",
    "FINAL_CLARIFICATION",
    "FINAL_BLOCKED_PHI",
    "decompose_literal_anchor",
    "detect_question_family",
    "build_central_concept_ledger",
    "assess_baseline_precision_veto",
    "evaluate_specificity_delta",
    "run_specificity_gate",
]
