"""Phase M4.1 — Preview-only Active M4 orchestrator.

DETERMINISTIC promotion layer. NO NEW LLM CALL. NO NEW RETRIEVAL of
its own — it only compiles the validator-accepted M4 candidate into
Boolean query records that the main pipeline then dispatches via the
SAME source path the deterministic A-arm uses.

This is the **active** counterpart to the M4 shadow interpreter. The
bounded LLM candidate has ALREADY been produced + validated upstream
(M4 shadow) and the structured compiler-input contract has ALREADY
been projected + validated (M4.0.1). M4.1 just decides whether to
PROMOTE that candidate's queries into clinician-visible retrieval, for
preview/dev only, behind a two-source kill switch.

Hard rules (closed-set):
    1. Preview/dev/staging env only. Production fail-closed via
       ``services.m4_active_guard`` at startup AND the env-allowlist
       gate here AND the M4 mode resolver (which forces ``disabled`` in
       production, so no candidate even exists).
    2. The M4 candidate must be ACCEPTED (validation passed) AND the
       M4.0.1 projection validation must be ``passed`` (no flags).
    3. Two-source kill switch: env ``M4_ACTIVE_KILL_SWITCH`` (default
       engaged) OR Mongo ``feature_flags.m4_active_kill_switch``. Either
       engaged = promotion NOT attempted, baseline retrieval unchanged.
    4. NEVER runs two active semantic-repair engines in one session.
       If M3.2 active applied OR M3.2 is not dormant, M4.1 blocks with
       the closed-set decision ``m3_2_already_applied_or_not_dormant``.
    5. PHI dominance: the pipeline reaches this code only when the PHI
       guard upstream passed; the candidate fields are validator-
       scrubbed. Defense in depth: a leak scan is a HARD gate before
       promotion.
    6. Only the small clinician indicator string "Clinical question
       interpretation refined" is exposed to clinicians, and only when
       applied. No prompts / completions / PHI / LLM specifics.

The decision is recorded as a closed-set sidecar row in
``m4_shadow_decisions`` (TTL 30d) with ``phase="M4.1"`` and
``active_retrieval_changed`` reflecting whether queries were swapped.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from core.db import Collections, get_db
from services.audit import audit_service
from services.m3_1_benchmark import compile_b_arm_queries, leak_scan_passes
from services.m4_active_config import M4ActiveConfig, load_m4_active_config
from services.m4_active_feature_flags import is_m4_mongo_kill_engaged

logger = logging.getLogger("clinevidence.m4_active")


# Closed-set decision strings.
M4_ACTIVE_DECISIONS: frozenset[str] = frozenset({
    "eligible",
    "ineligible_candidate_not_accepted",
    "ineligible_projection_flagged",
    "ineligible_no_required_anchor",
    "ineligible_master_disabled",
    "ineligible_kill_switch_engaged",
    "ineligible_env_not_allowlisted",
    "ineligible_production_hard_off",
    "m3_2_already_applied_or_not_dormant",
    "ineligible_no_query_records",
    "ineligible_leak_scan_failed",
    "ineligible_internal_error",
    # Phase M4.1.5 — hard end-to-end latency timeout fallback.
    "m4_latency_timeout",
    # Phase M4.1.1 — active promotion eligibility guard.
    "baseline_already_clean",
    "query_family_diversity_would_decrease_without_baseline_defect",
    # Phase M4.1 — explicit K4A protected-route exclusion (dominant).
    "route_owned_by_k4a_specialty",
    # Phase M4.4 — specificity-preservation / do-no-harm gate block.
    "specificity_gate_blocked",
})


# ---- Phase M4.1.1 — closed-set baseline-degradation reasons. ----
DEG_FULL_LITERAL_PHRASE: str = "full_literal_phrase_required_anchor"
DEG_INTENT_WORD_FUSED: str = "intent_word_fused_in_disease_anchor"
DEG_STAGE_BIOMARKER_POP_FUSED: str = "stage_biomarker_population_fused_in_anchor"
DEG_OUTCOME_TERM_AS_ANCHOR: str = "outcome_or_intent_modifier_as_required_anchor"
DEG_NO_CLEAN_PRIMARY_ENTITY: str = "no_clean_primary_entity_in_baseline"
DEG_WHATIS_OVERVIEW_MISFORCE: str = "what_is_forced_overview_contradicts_m4_intent"
DEG_LOW_CONFIDENCE: str = "baseline_low_confidence_meaningful_question"

M4_BASELINE_DEGRADATION_REASONS: frozenset[str] = frozenset({
    DEG_FULL_LITERAL_PHRASE,
    DEG_INTENT_WORD_FUSED,
    DEG_STAGE_BIOMARKER_POP_FUSED,
    DEG_OUTCOME_TERM_AS_ANCHOR,
    DEG_NO_CLEAN_PRIMARY_ENTITY,
    DEG_WHATIS_OVERVIEW_MISFORCE,
    DEG_LOW_CONFIDENCE,
})

# Closed-set lexicons (lowercase). Conservative on purpose: a clean
# disease/symptom/entity anchor must NOT match any of these so that an
# already-correct baseline (e.g. "cholestatic liver disease",
# "pruritus") is never treated as degraded.
_INTENT_TERMS: frozenset[str] = frozenset({
    "prognosis", "prognostic", "survival", "recurrence", "recur",
    "mortality", "cause", "causes", "etiology", "aetiology",
    "risk factor", "risk factors", "mechanism", "pathophysiology",
    "pathogenesis", "incidence", "prevalence", "differential diagnosis",
    "natural history", "complication", "complications",
})
# Outcome / intent-modifier phrases that should never be a required
# anchor on their own (e.g. "survival rate", "recurrence risk").
_OUTCOME_MODIFIER_PHRASES: frozenset[str] = frozenset({
    "survival rate", "recurrence risk", "recurrence rate",
    "mortality rate", "survival", "recurrence", "five-year survival",
    "5-year survival", "overall survival", "disease-free survival",
})
_BIOMARKER_TOKENS: frozenset[str] = frozenset({
    "her2", "her-2", "her2-positive", "her2 positive", "er-positive",
    "er positive", "pr-positive", "egfr", "alk", "pd-l1", "pdl1",
    "braf", "kras", "ros1", "msi", "msi-high", "brca", "brca1", "brca2",
    "estrogen receptor", "ki-67", "ki67", "p53", "triple-negative",
    "triple negative",
})
# Stage / grade tokens via regex (built once).
_STAGE_RE = re.compile(
    r"\bstage\s+(0|i{1,3}v?|iv|1|2|3|4)\b|\bt[0-4][a-c]?\b|\bn[0-3][a-c]?\b"
    r"|\bm[0-1]\b|\bgrade\s+(1|2|3|4|i{1,3}v?)\b",
    re.IGNORECASE,
)
# Baseline frameworks that represent a background / disease-overview
# misforce (NOT mechanism — concept_mechanism is the CORRECT framework
# for a genuine mechanism question and must never be treated as a
# misforce on its own).
_OVERVIEW_FRAMEWORKS: frozenset[str] = frozenset({
    "disease_overview", "background", "overview", "concept_overview",
})
# M4 intents that are MORE specific than a background/overview parse.
_SPECIFIC_M4_INTENTS: frozenset[str] = frozenset({
    "prognosis_natural_history", "etiology", "diagnosis", "therapy",
    "prognostic_factor", "harm_etiology",
})
_BASELINE_CONFIDENCE_FLOOR: float = 0.35


def _contains_term(text: str, terms) -> bool:
    t = f" {text.lower()} "
    for term in terms:
        tt = term.lower()
        # word-ish boundary match for short tokens; substring for phrases
        if " " in tt or "-" in tt:
            if tt in text.lower():
                return True
        elif f" {tt} " in t or t.strip() == tt:
            return True
    return False


def assess_baseline_degradation(
    *,
    baseline_required_anchors: list[str],
    baseline_framework,
    baseline_intent,
    baseline_confidence,
    m4_candidate: dict,
) -> dict[str, Any]:
    """Phase M4.1.1 — deterministic, closed-set assessment of whether
    the DETERMINISTIC baseline parse is degraded enough to justify
    promoting the M4 candidate into active retrieval.

    Pure function. NO LLM. Returns
    ``{"baseline_degraded": bool, "reasons": [closed-set...]}``.

    A clean disease/symptom/entity anchor (no fused intent verb, no
    fused stage/biomarker, reasonable length) yields NO reasons → the
    caller will NOT promote (preserving the richer baseline families).
    """
    reasons: list[str] = []
    anchors = [a for a in (baseline_required_anchors or []) if (a or "").strip()]

    for a in anchors:
        al = a.strip().lower()
        wc = len(al.split())
        # D1 — full literal / fused question phrase as a required anchor.
        if wc >= 5:
            reasons.append(DEG_FULL_LITERAL_PHRASE)
        # D2 — intent verb/noun fused into the anchor.
        if _contains_term(al, _INTENT_TERMS):
            reasons.append(DEG_INTENT_WORD_FUSED)
        # D3 — stage / biomarker / population fused into the anchor.
        if _STAGE_RE.search(al) or _contains_term(al, _BIOMARKER_TOKENS):
            reasons.append(DEG_STAGE_BIOMARKER_POP_FUSED)
        # D4 — a bare outcome / intent-modifier phrase used as anchor.
        if al in _OUTCOME_MODIFIER_PHRASES:
            reasons.append(DEG_OUTCOME_TERM_AS_ANCHOR)

    # D5 — baseline recovered no clean primary entity but M4 did.
    cand_problem = [
        x for x in (m4_candidate.get("clinical_problem") or [])
        if isinstance(x, str) and x.strip()
    ]
    cand_required = [
        x for x in (m4_candidate.get("required_anchors") or [])
        if isinstance(x, str) and x.strip()
    ]
    if not anchors and (cand_problem or cand_required):
        reasons.append(DEG_NO_CLEAN_PRIMARY_ENTITY)

    # D6 — "what is" forced a background/overview parse that contradicts
    # a stronger M4 intent. Fires ONLY when the baseline framework is an
    # overview/background framework (NOT concept_mechanism) AND the M4
    # intent is a specific non-overview intent. This is what spares B8:
    # baseline concept_mechanism + M4 mechanism → no misforce.
    fw_list = baseline_framework if isinstance(baseline_framework, (list, tuple)) else [baseline_framework]
    fw_norm = {str(f).strip().lower() for f in fw_list if f}
    m4_intent = str(m4_candidate.get("primary_intent") or "").strip().lower()
    if (fw_norm & _OVERVIEW_FRAMEWORKS) and m4_intent in _SPECIFIC_M4_INTENTS:
        reasons.append(DEG_WHATIS_OVERVIEW_MISFORCE)

    # D7 — low baseline confidence on a clinically meaningful question
    # (M4 produced a clean candidate with a required anchor).
    try:
        conf = float(baseline_confidence) if baseline_confidence is not None else None
    except (TypeError, ValueError):
        conf = None
    if conf is not None and conf < _BASELINE_CONFIDENCE_FLOOR and cand_required:
        reasons.append(DEG_LOW_CONFIDENCE)

    # Dedupe preserving order.
    seen: set[str] = set()
    deduped = [r for r in reasons if not (r in seen or seen.add(r))]
    return {"baseline_degraded": bool(deduped), "reasons": deduped}


# ---- Phase M4.1.3 — no-primary-entity rescue. ----
# Closed-set rescue trigger reason.
RESCUE_NO_PRIMARY_MEANINGFUL: str = "no_primary_entity_but_clinically_meaningful"
# Closed-set rescue rejection reasons.
RESCUE_REJECT_NO_REQUIRED_ANCHOR: str = "rescue_no_required_anchor"
RESCUE_REJECT_GENERIC_ANCHOR: str = "rescue_generic_required_anchor"
RESCUE_REJECT_FULL_LITERAL_ANCHOR: str = "rescue_full_literal_required_anchor"
RESCUE_REJECT_NO_CLINICAL_ENTITY: str = "rescue_no_clinical_entity"
RESCUE_REJECT_NOT_ACCEPTED: str = "rescue_candidate_not_accepted"
RESCUE_REJECT_PROJECTION_FLAGGED: str = "rescue_projection_flagged"
RESCUE_REJECT_LOW_CONFIDENCE: str = "rescue_low_confidence"

M4_RESCUE_REJECTIONS: frozenset[str] = frozenset({
    RESCUE_REJECT_NO_REQUIRED_ANCHOR,
    RESCUE_REJECT_GENERIC_ANCHOR,
    RESCUE_REJECT_FULL_LITERAL_ANCHOR,
    RESCUE_REJECT_NO_CLINICAL_ENTITY,
    RESCUE_REJECT_NOT_ACCEPTED,
    RESCUE_REJECT_PROJECTION_FLAGGED,
    RESCUE_REJECT_LOW_CONFIDENCE,
})

# Generic / intent-only terms that must NEVER be the required anchor of a
# rescued question (so "What is the dose/treatment/risk?" still clarifies).
_GENERIC_ANCHOR_TERMS: frozenset[str] = frozenset({
    "dose", "doses", "dosage", "dosing", "treatment", "treatments",
    "therapy", "therapies", "management", "risk", "risks", "prognosis",
    "survival", "recurrence", "cause", "causes", "etiology", "aetiology",
    "mechanism", "pathophysiology", "diagnosis", "outcome", "outcomes",
    "side effect", "side effects", "adverse effect", "adverse effects",
    "guideline", "guidelines", "incidence", "prevalence", "rate",
    "marker", "markers", "symptom", "symptoms", "sign", "signs",
})
_RESCUE_CONFIDENCE_FLOOR: float = 0.4


def validate_no_primary_rescue_candidate(
    candidate: dict,
) -> tuple[bool, str]:
    """Phase M4.1.3 — STRICT, deterministic validation that an M4
    candidate is safe to rescue a baseline that found no primary entity.

    Returns ``(ok, reason)``. ``reason`` is the closed-set rescue
    trigger reason on success, or a closed-set rejection reason on
    failure. Pure function.
    """
    c = candidate if isinstance(candidate, dict) else {}
    req = [str(x).strip() for x in (c.get("required_anchors") or [])
           if isinstance(x, str) and str(x).strip()]
    if not req:
        return False, RESCUE_REJECT_NO_REQUIRED_ANCHOR
    for a in req:
        al = a.lower()
        # required anchor must not be a bare generic / intent term
        if al in _GENERIC_ANCHOR_TERMS:
            return False, RESCUE_REJECT_GENERIC_ANCHOR
        # required anchor must not be the full literal question
        if len(al.split()) >= 6:
            return False, RESCUE_REJECT_FULL_LITERAL_ANCHOR
    # must carry a real clinical entity (problem / drug / procedure / test)
    has_entity = any(
        bool(c.get(k)) for k in
        ("clinical_problem", "medication", "procedure", "index_test")
    )
    if not has_entity:
        return False, RESCUE_REJECT_NO_CLINICAL_ENTITY
    # confidence threshold
    try:
        conf = float(c.get("confidence")) if c.get("confidence") is not None else None
    except (TypeError, ValueError):
        conf = None
    if conf is not None and conf < _RESCUE_CONFIDENCE_FLOOR:
        return False, RESCUE_REJECT_LOW_CONFIDENCE
    return True, RESCUE_NO_PRIMARY_MEANINGFUL


def is_m4_no_primary_rescue_eligible(
    m4_block: dict,
) -> tuple[bool, str]:
    """Phase M4.1.3 — gate: is this accepted+validated M4 candidate
    eligible to rescue a no-primary-entity baseline? Checks acceptance,
    projection validation, and the strict candidate validation. Does NOT
    consider the kill switch / env — those are enforced by
    ``run_m4_active_decision``. Pure function."""
    if not isinstance(m4_block, dict) or not m4_block.get("accepted"):
        return False, RESCUE_REJECT_NOT_ACCEPTED
    projection = m4_block.get("m4_0_1_projection") or {}
    if (projection.get("projection_validation_status")
            if isinstance(projection, dict) else None) != "passed":
        return False, RESCUE_REJECT_PROJECTION_FLAGGED
    candidate = m4_block.get("candidate") or {}
    return validate_no_primary_rescue_candidate(candidate)


# Exact clinician-facing string per the frontend safety guideline.
M4_ACTIVE_CHIP_PRIMARY_TEXT: str = "Clinical question interpretation refined"
M4_ACTIVE_CHIP_TOOLTIP_TEXT: str = (
    "The clinical question interpretation was refined before evidence "
    "retrieval. Source selection and ranking are unchanged."
)


def m4_active_chip_payload() -> dict[str, str]:
    """Closed-set chip payload. No LLM/prompt specifics."""
    return {
        "primary_text": M4_ACTIVE_CHIP_PRIMARY_TEXT,
        "tooltip_text": M4_ACTIVE_CHIP_TOOLTIP_TEXT,
    }


def _dedupe(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq or []:
        k = (s or "").strip()
        if not k:
            continue
        kl = k.lower()
        if kl in seen:
            continue
        seen.add(kl)
        out.append(k)
    return out


def m4_candidate_to_compiler_candidate(candidate: dict) -> dict[str, Any]:
    """Map an ACCEPTED M4 candidate (``M4Candidate.model_dump``) onto
    the ``compile_b_arm_queries`` candidate shape (required_anchors +
    slots). Pure, deterministic.

    Stage / biomarker are folded into the population context slot so
    they survive into the precision query family WITHOUT shrinking the
    broad-recall anchor family — preventing a stage-specific question
    from collapsing to the bare disease (the M4.0.1 caveat).
    """
    c = candidate if isinstance(candidate, dict) else {}

    def _g(key: str) -> list[str]:
        v = c.get(key)
        return [str(x) for x in v if isinstance(x, str) and x.strip()] if isinstance(v, list) else []

    required = _dedupe(_g("required_anchors"))
    slots = {
        "condition": _dedupe(_g("clinical_problem")),
        "population": _dedupe(_g("population") + _g("stage") + _g("biomarker")),
        "intervention_or_exposure": _dedupe(_g("intervention") + _g("exposure")),
        "outcome": _dedupe(_g("outcome")),
        "index_test": _dedupe(_g("index_test")),
        "target_condition": [],
        "procedure_or_device": _dedupe(_g("procedure") + _g("medication")),
        "mechanism": _dedupe(_g("mechanism")),
    }
    return {"required_anchors": required, "slots": slots}


def compile_m4_active_cb_records(
    candidate: dict,
    sources: Iterable[str],
) -> list[dict[str, Any]]:
    """Compile the accepted M4 candidate into cb_query records the main
    pipeline understands. Each record carries
    ``planner_source="m4_active"`` so the per-row audit trail in
    ``search_queries`` is unambiguous. Pure: deterministic, no I/O.
    """
    compiler_candidate = m4_candidate_to_compiler_candidate(candidate)
    bq = compile_b_arm_queries(compiler_candidate)
    sources_set = {s.lower() for s in sources}
    src_map = {"pubmed": "pubmed", "epmc": "europe_pmc"}
    out: list[dict[str, Any]] = []
    for raw_src, qs in bq.items():
        normalized_src = src_map.get(raw_src.lower(), raw_src.lower())
        if normalized_src not in sources_set:
            continue
        for q in qs:
            qstr = q.get("query_string") or ""
            if not qstr.strip():
                continue
            family = q.get("query_type") or "b_required_AND"
            out.append({
                "source": normalized_src,
                "query_family": f"m4_active_{family}",
                "query_type": f"m4_active_{family}",
                "candidate_tier": "m4_active",
                "query_string": qstr,
                "rationale": "phase_m4_1_active_interpretation",
                "concept_blocks_json": {},
                "mandatory_anchors": list(
                    compiler_candidate.get("required_anchors") or []
                ),
                "optional_terms": [],
                "planner_source": "m4_active",
                # ---- Phase M4.5 — additive-rescue provenance. ----
                "query_family_source": "m4_rescue",
                "m4_rescue": True,
                "m4_rescue_reason": "additive_diversity_preserving_rescue",
                "is_primary_strategy": True,
                "is_fallback_strategy": False,
                "fallback_reason": None,
                "max_results": 50,
                "field_tags_used": [],
                "mesh_terms_used": [],
                "cq_category": None,
                "cq_scope": None,
            })
    return out


# ---------------------------------------------------------------------
# Phase M4.1 — explicit K4A protected-route exclusion (deterministic).
#
# K4A.* specialty routes OWN their question shape. M4 active query
# rewriting must NEVER fire on them — not even with an otherwise valid,
# accepted, eligible M4 candidate (the central failure the operator
# flagged). The authoritative signal is the deterministic parser's
# ``question_subtype`` (mirrors the closed set in
# ``services.semantic_repair_active_eligibility._K4A_OWNED_SUBTYPES``).
# ---------------------------------------------------------------------
K4A_OWNED_SUBTYPES: frozenset[str] = frozenset({
    "severity_marker_overview",                  # K4A.2
    "rule_out_test_utility",                     # K4A.3
    "epidemiology_incidence_complication_risk",  # K4A.4
})

# Closed-set active-route IDs surfaced in diagnostics so operators never
# have to infer K4A route identity from directness labels / anchor shape.
ACTIVE_ROUTE_IDS: frozenset[str] = frozenset({
    "k4a_2_severity_marker_overview",
    "k4a_3_rule_out_diagnostic_accuracy",
    "k4a_4_incidence_risk_epidemiology",
    "default",
    "unknown",
})

PROTECTED_ROUTE_FAMILY_K4A: str = "k4a"

_K4A_SUBTYPE_TO_ROUTE_ID: dict[str, str] = {
    "severity_marker_overview": "k4a_2_severity_marker_overview",
    "rule_out_test_utility": "k4a_3_rule_out_diagnostic_accuracy",
    "epidemiology_incidence_complication_risk": "k4a_4_incidence_risk_epidemiology",
}


def resolve_active_route(
    *,
    question_subtype: Optional[str],
    route_owned_hint: Optional[bool] = None,
) -> dict[str, Any]:
    """Deterministic, closed-set active-route resolution.

    Returns ``{active_route_id, protected_route_family,
    route_owned_by_k4a_specialty}``. A K4A-owned ``question_subtype`` is
    authoritative; an explicit upstream ``route_owned_hint`` (e.g. from
    M3/M4 diagnostics) is honored as defense-in-depth.
    """
    st = (question_subtype or "").strip().lower()
    if st in K4A_OWNED_SUBTYPES:
        return {
            "active_route_id": _K4A_SUBTYPE_TO_ROUTE_ID[st],
            "protected_route_family": PROTECTED_ROUTE_FAMILY_K4A,
            "route_owned_by_k4a_specialty": True,
        }
    if route_owned_hint is True:
        return {
            "active_route_id": "unknown",
            "protected_route_family": PROTECTED_ROUTE_FAMILY_K4A,
            "route_owned_by_k4a_specialty": True,
        }
    return {
        "active_route_id": "default" if st else "unknown",
        "protected_route_family": None,
        "route_owned_by_k4a_specialty": False,
    }


def _decide(
    *,
    candidate: Optional[dict],
    accepted: bool,
    projection_status: Optional[str],
    required_anchors: list[str],
    cfg: M4ActiveConfig,
    combined_kill: bool,
    m3_2_applied: bool,
    m3_2_dormant: bool,
    route_owned_by_k4a_specialty: bool = False,
) -> str:
    """Closed-set eligibility decision. Order encodes precedence."""
    # Phase M4.1 — protected-route exclusion DOMINATES every other
    # verdict. A K4A specialty route owns its question shape; M4 active
    # query rewriting must never fire on it, even with a valid /
    # accepted / otherwise-eligible candidate.
    if route_owned_by_k4a_specialty:
        return "route_owned_by_k4a_specialty"
    if not accepted or not isinstance(candidate, dict):
        return "ineligible_candidate_not_accepted"
    # Never run two active engines in one session.
    if m3_2_applied or not m3_2_dormant:
        return "m3_2_already_applied_or_not_dormant"
    if cfg.is_production_env and not cfg.prod_allow_active:
        return "ineligible_production_hard_off"
    if not cfg.enabled:
        return "ineligible_master_disabled"
    if combined_kill:
        return "ineligible_kill_switch_engaged"
    if not cfg.env_in_allowlist:
        return "ineligible_env_not_allowlisted"
    # M4.0.1 projection must have validated clean (fail-closed otherwise).
    if projection_status != "passed":
        return "ineligible_projection_flagged"
    if not required_anchors:
        return "ineligible_no_required_anchor"
    return "eligible"


async def run_m4_active_decision(
    *,
    session_id: str,
    user_id: Optional[str],
    m4_block: dict[str, Any],
    picott_payload: dict[str, Any],
    sources: list[str],
    a_arm_cqm: dict[str, Any],
    m3_2_applied: bool,
    m3_2_kill_combined_engaged: Optional[bool],
    route_owned_hint: Optional[bool] = None,
    baseline_degraded: bool = False,
    baseline_degradation_reasons: Optional[list[str]] = None,
    baseline_query_family_count: Optional[int] = None,
    a_arm_boolean_queries: Optional[list[str]] = None,
    rescue_path: Optional[str] = None,
    normalized_question: str = "",
    additive_mode: bool = True,
    upstream_latency_ms: Optional[int] = None,
    upstream_latency_breakdown: Optional[dict[str, Any]] = None,
    upstream_timed_out: bool = False,
    config: Optional[M4ActiveConfig] = None,
    mongo_kill_engaged: Optional[bool] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Main orchestrator. NEVER raises. Returns a closed-set dict with
    ``applied`` / ``decision`` / ``cb_records`` / ``chip`` / etc.

    Phase M4.1.1: promotion is additionally gated on
    ``baseline_degraded`` — an already-clean baseline is NOT replaced
    (``baseline_already_clean``), and a promotion that would shrink
    query-family diversity without a real baseline defect is blocked
    (``query_family_diversity_would_decrease_without_baseline_defect``).

    Side effects: writes a closed-set sidecar row to
    ``m4_shadow_decisions`` (phase=M4.1.1) and emits a closed-set audit
    event.
    """
    t_start = time.perf_counter()
    now_dt = now if now is not None else datetime.now(timezone.utc)
    cfg = config if config is not None else load_m4_active_config()
    if mongo_kill_engaged is None:
        try:
            mongo_kill_engaged = await is_m4_mongo_kill_engaged()
        except Exception:  # noqa: BLE001
            mongo_kill_engaged = True
    kill_state = {
        "env_kill_switch_engaged": bool(cfg.kill_switch_engaged),
        "mongo_kill_switch_engaged": bool(mongo_kill_engaged),
        "combined_engaged": bool(cfg.kill_switch_engaged or mongo_kill_engaged),
        "fail_closed_default": True,
    }
    combined_kill = kill_state["combined_engaged"]

    accepted = bool(m4_block.get("accepted"))
    candidate = m4_block.get("candidate") if accepted else None
    candidate = candidate if isinstance(candidate, dict) else None
    projection = m4_block.get("m4_0_1_projection") or {}
    projection_status = (
        projection.get("projection_validation_status")
        if isinstance(projection, dict) else None
    )
    projected_input = (
        projection.get("projected_compiler_input")
        if isinstance(projection, dict) else None
    )
    required_anchors = _dedupe(
        (projected_input or {}).get("required_anchors")
        or (candidate or {}).get("required_anchors")
        or []
    )

    # M3.2 dormancy: dormant only when it did NOT apply AND its combined
    # kill switch is engaged. Unknown combined state → treat as NOT
    # dormant (fail-closed: block M4 rather than risk a double engine).
    m3_2_dormant = (not m3_2_applied) and bool(m3_2_kill_combined_engaged)

    # Phase M4.1 — deterministic, closed-set protected-route resolution.
    # Authoritative signal = the parser's ``question_subtype``. A K4A
    # specialty route blocks active query rewriting regardless of
    # candidate validity / eligibility.
    route_info = resolve_active_route(
        question_subtype=(picott_payload or {}).get("question_subtype"),
        route_owned_hint=route_owned_hint,
    )
    route_owned = bool(route_info["route_owned_by_k4a_specialty"])

    decision_str = _decide(
        candidate=candidate,
        accepted=accepted,
        projection_status=projection_status,
        required_anchors=required_anchors,
        cfg=cfg,
        combined_kill=combined_kill,
        m3_2_applied=m3_2_applied,
        m3_2_dormant=m3_2_dormant,
        route_owned_by_k4a_specialty=route_owned,
    )

    # ---- Phase M4.1.5 — hard latency timeout dominates every other
    # eligibility verdict. When the upstream bounded-LLM M4 decision
    # exceeded its budget, we NEVER promote (no partial M4 retrieval);
    # the pipeline falls back to baseline retrieval / clarification.
    if upstream_timed_out and decision_str != "route_owned_by_k4a_specialty":
        decision_str = "m4_latency_timeout"

    applied = False
    cb_records: list[dict[str, Any]] = []
    leak_ok = True
    m4_family_count = 0
    deg_reasons = list(baseline_degradation_reasons or [])
    spec_gate: Optional[dict[str, Any]] = None

    if decision_str == "eligible" and candidate is not None:
        try:
            cb_records = compile_m4_active_cb_records(candidate, sources)
            m4_family_count = len({
                r.get("query_family") for r in cb_records if r.get("query_family")
            })
            if not cb_records:
                decision_str = "ineligible_no_query_records"
            elif not baseline_degraded:
                # ---- Phase M4.1.1 — active promotion eligibility guard.
                # The baseline parse has no detected defect → do NOT
                # replace it. Preserve the (richer) baseline retrieval.
                if (
                    baseline_query_family_count is not None
                    and m4_family_count < baseline_query_family_count
                ):
                    decision_str = (
                        "query_family_diversity_would_decrease_"
                        "without_baseline_defect"
                    )
                else:
                    decision_str = "baseline_already_clean"
                cb_records = []
                logger.info(
                    "m4_active_promotion_blocked session_id=%s reason=%s "
                    "m4_families=%d baseline_families=%s",
                    session_id, decision_str, m4_family_count,
                    baseline_query_family_count,
                )
            else:
                leak_ok = leak_scan_passes({
                    "cb_records": cb_records,
                    "required_anchors": required_anchors,
                    "projected_compiler_input": projected_input,
                })
                if not leak_ok:
                    decision_str = "ineligible_leak_scan_failed"
                    cb_records = []
                    logger.warning(
                        "m4_active_leak_scan_failed session_id=%s — blocking",
                        session_id,
                    )
                else:
                    # ---- Phase M4.4 — Specificity Preservation /
                    # Do-No-Harm gate. Deterministic, NO LLM, NO
                    # retrieval. BLOCKS promotion (preserve baseline)
                    # when M4 would drop/demote a central clinical
                    # concept for the question family, generalize a
                    # biomarker/stage condition, broaden an
                    # already-precise literal phrase (precision veto),
                    # or shrink query-family diversity. ----
                    from services.m4_specificity_gate import (
                        run_specificity_gate as _m4_4_gate,
                    )
                    spec_gate = _m4_4_gate(
                        normalized_question=normalized_question or "",
                        baseline_type_of_question=(
                            picott_payload.get("type_of_question")
                        ),
                        baseline_question_subtype=(
                            picott_payload.get("question_subtype")
                        ),
                        baseline_must_have_concepts=list(
                            picott_payload.get("must_have_concepts") or []
                        ),
                        baseline_literal_anchors=list(
                            (m4_block.get("baseline") or {}).get(
                                "required_anchors"
                            ) or []
                        ),
                        m4_candidate=candidate or {},
                        baseline_degraded=bool(baseline_degraded),
                        baseline_degradation_reasons=list(deg_reasons),
                        baseline_query_family_count=baseline_query_family_count,
                        m4_query_family_count=int(m4_family_count),
                        route_owned_by_k4a_specialty=route_owned,
                        candidate_accepted=True,
                        upstream_timed_out=bool(upstream_timed_out),
                        additive_mode=bool(additive_mode),
                        m4_rescue_family_names=sorted({
                            r.get("query_family")
                            for r in cb_records if r.get("query_family")
                        }),
                    )
                    if spec_gate.get("allow"):
                        applied = True
                        # ---- Phase M4.5 — tag the rescue reason on each
                        # cb record per the active rescue path. ----
                        _rescue_reason = (
                            "no_primary_entity_rescue"
                            if rescue_path == "no_primary_entity"
                            else "additive_diversity_preserving_rescue"
                        )
                        for _r in cb_records:
                            _r["m4_rescue_reason"] = _rescue_reason
                    else:
                        applied = False
                        cb_records = []
                        decision_str = "specificity_gate_blocked"
                        logger.info(
                            "m4_active_specificity_blocked session_id=%s "
                            "family=%s reasons=%s",
                            session_id,
                            spec_gate.get("question_family"),
                            spec_gate.get("m4_specificity_block_reasons"),
                        )
        except Exception:  # noqa: BLE001
            decision_str = "ineligible_internal_error"
            cb_records = []
            applied = False
            logger.exception(
                "m4_active_apply_failed session_id=%s — falling back",
                session_id,
            )

    # Phase M4.1 — closed-set route/rewrite diagnostics.
    active_query_rewritten = bool(applied)
    m4_block_reasons: list[str] = [] if applied else [decision_str]

    promotion_decision_ms = int((time.perf_counter() - t_start) * 1000)
    latency_ms = promotion_decision_ms  # BC alias (promotion-decision only)

    # ---- Phase M4.1.5 — closed-set NUMERIC end-to-end latency breakdown.
    # Only integer milliseconds + closed-set keys are recorded. NO
    # prompts / completions / PHI / credentials / evidence text.
    _ub = upstream_latency_breakdown if isinstance(upstream_latency_breakdown, dict) else {}
    latency_breakdown = {
        "trigger_detect_ms": int(_ub.get("trigger_detect_ms") or 0),
        "llm_candidate_ms": int(_ub.get("llm_candidate_ms") or 0),
        "validation_ms": int(_ub.get("validation_ms") or 0),
        "would_be_projection_ms": int(_ub.get("would_be_projection_ms") or 0),
        "compiler_projection_ms": int(_ub.get("compiler_projection_ms") or 0),
        "promotion_decision_ms": promotion_decision_ms,
    }
    m4_total_latency_ms = int(upstream_latency_ms or 0) + promotion_decision_ms
    latency_breakdown["total_ms"] = m4_total_latency_ms

    # ---- Phase M4.4 — closed-set specificity-gate diagnostics. ----
    m4_specificity_diag = {
        "m4_specificity_phase": (spec_gate or {}).get(
            "m4_specificity_phase", "M4.4"
        ),
        "final_decision": (spec_gate or {}).get("final_decision"),
        "question_family": (spec_gate or {}).get("question_family"),
        "central_concept_ledger": (spec_gate or {}).get(
            "central_concept_ledger", {"concepts": []}
        ),
        "baseline_precision_veto": bool(
            (spec_gate or {}).get("baseline_precision_veto")
        ),
        "baseline_precision_veto_reasons": list(
            (spec_gate or {}).get("baseline_precision_veto_reasons") or []
        ),
        "central_concept_preservation_status": (spec_gate or {}).get(
            "central_concept_preservation_status", {}
        ),
        "specificity_delta": (spec_gate or {}).get("specificity_delta", {}),
        "query_family_diversity_before": (spec_gate or {}).get(
            "query_family_diversity_before", baseline_query_family_count
        ),
        "query_family_diversity_after": (spec_gate or {}).get(
            "query_family_diversity_after", int(m4_family_count)
        ),
        "diversity_delta": (spec_gate or {}).get("diversity_delta", 0),
        "diversity_guard_passed": (spec_gate or {}).get(
            "diversity_guard_passed", True
        ),
        "m4_application_mode": (spec_gate or {}).get(
            "m4_application_mode", "baseline_preserved"
        ),
        "baseline_families_preserved": (spec_gate or {}).get(
            "baseline_families_preserved", True
        ),
        "added_m4_rescue_families": list(
            (spec_gate or {}).get("added_m4_rescue_families") or []
        ),
        "final_query_strategy": (spec_gate or {}).get(
            "final_query_strategy", "baseline_only"
        ),
        "m4_specificity_block_reasons": list(
            (spec_gate or {}).get("m4_specificity_block_reasons") or []
        ),
        "m4_specificity_pass_reasons": list(
            (spec_gate or {}).get("m4_specificity_pass_reasons") or []
        ),
    }

    # ---- Sidecar (append-only, TTL 30d, phase=M4.1). ----
    sidecar_doc_id = uuid.uuid4().hex
    sidecar = {
        "id": sidecar_doc_id,
        "search_session_id": session_id,
        "user_id": user_id,
        "phase": "M4.1.1",
        "mode": "active",
        "triggered": True,
        "trigger_reasons": list(m4_block.get("trigger_reasons") or []),
        "validation_status": m4_block.get("validation_status"),
        "accepted": accepted,
        "active_decision": decision_str,
        "active_retrieval_changed": bool(applied),
        "active_route_id": route_info["active_route_id"],
        "protected_route_family": route_info["protected_route_family"],
        "route_owned_by_k4a_specialty": route_owned,
        "m4_block_reasons": list(m4_block_reasons),
        "active_query_rewritten": active_query_rewritten,
        "rescue_path": rescue_path,
        "baseline_degraded": bool(baseline_degraded),
        "baseline_degradation_reasons": deg_reasons,
        "baseline_query_family_count": baseline_query_family_count,
        "m4_query_family_count": int(m4_family_count),
        "required_anchors": list(required_anchors),
        "projection_validation_status": projection_status,
        "a_arm_cqm": dict(a_arm_cqm or {}),
        "a_arm_boolean_queries": list(a_arm_boolean_queries or []),
        "m4_active_boolean_queries": sorted({
            r.get("query_string") for r in cb_records if r.get("query_string")
        }),
        "kill_switch_state": kill_state,
        "m3_2_applied": bool(m3_2_applied),
        "m3_2_dormant": bool(m3_2_dormant),
        "safety_flags": {
            "leak_scan_passed": bool(leak_ok),
        },
        "m4_specificity": dict(m4_specificity_diag),
        "config_snapshot": {
            "enabled": bool(cfg.enabled),
            "current_env": cfg.current_env,
            "env_allowlist": list(cfg.env_allowlist),
            "max_latency_ms": int(cfg.max_latency_ms),
            "prod_allow_active": bool(cfg.prod_allow_active),
            "effective_enabled": bool(cfg.effective_enabled),
        },
        "latency_ms": int(latency_ms),
        "promotion_decision_latency_ms": int(promotion_decision_ms),
        "m4_total_latency_ms": int(m4_total_latency_ms),
        "latency_breakdown": dict(latency_breakdown),
        "timed_out": bool(upstream_timed_out),
        "created_at": now_dt,
    }
    try:
        await get_db()[Collections.M4_SHADOW_DECISIONS].insert_one(dict(sidecar))
    except Exception:  # noqa: BLE001
        logger.exception(
            "m4_active sidecar insert failed (non-fatal) session_id=%s",
            session_id,
        )

    # ---- Audit. ----
    try:
        await audit_service.log(
            "m4_active_decision_applied" if applied
            else "m4_active_decision_fallback",
            user_id=user_id,
            search_session_id=session_id,
            payload={
                "decision": decision_str,
                "applied": bool(applied),
                "kill_switch_combined_engaged": combined_kill,
                "m3_2_applied": bool(m3_2_applied),
                "m3_2_dormant": bool(m3_2_dormant),
                "projection_validation_status": projection_status,
                "leak_scan_passed": bool(leak_ok),
                "baseline_degraded": bool(baseline_degraded),
                "baseline_degradation_reasons": deg_reasons,
                "m4_query_family_count": int(m4_family_count),
                "baseline_query_family_count": baseline_query_family_count,
                "latency_ms": int(latency_ms),
                "m4_total_latency_ms": int(m4_total_latency_ms),
                "timed_out": bool(upstream_timed_out),
                "phase": "M4.1.1",
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception("m4_active audit event failed (non-fatal)")

    return {
        "applied": bool(applied),
        "decision": decision_str,
        "active_route_id": route_info["active_route_id"],
        "protected_route_family": route_info["protected_route_family"],
        "route_owned_by_k4a_specialty": route_owned,
        "m4_block_reasons": list(m4_block_reasons),
        "active_query_rewritten": active_query_rewritten,
        "kill_switch_state": kill_state,
        "cb_records": cb_records,
        "application_mode": m4_specificity_diag.get(
            "m4_application_mode", "baseline_preserved"
        ),
        "added_m4_rescue_families": list(
            m4_specificity_diag.get("added_m4_rescue_families") or []
        ),
        "final_query_strategy": m4_specificity_diag.get(
            "final_query_strategy", "baseline_only"
        ),
        "chip": m4_active_chip_payload() if applied else None,
        "projection_validation_status": projection_status,
        "required_anchors": list(required_anchors),
        "m3_2_dormant": bool(m3_2_dormant),
        "rescue_path": rescue_path,
        "baseline_degraded": bool(baseline_degraded),
        "baseline_degradation_reasons": deg_reasons,
        "m4_specificity": dict(m4_specificity_diag),
        "m4_query_family_count": int(m4_family_count),
        "baseline_query_family_count": baseline_query_family_count,
        "sidecar_doc_id": sidecar_doc_id,
        "latency_ms": int(latency_ms),
        "promotion_decision_latency_ms": int(promotion_decision_ms),
        "m4_total_latency_ms": int(m4_total_latency_ms),
        "latency_breakdown": dict(latency_breakdown),
        "timed_out": bool(upstream_timed_out),
        "phase": "M4.1.1",
    }


__all__ = [
    "M4_ACTIVE_DECISIONS",
    "M4_ACTIVE_CHIP_PRIMARY_TEXT",
    "M4_BASELINE_DEGRADATION_REASONS",
    "K4A_OWNED_SUBTYPES",
    "ACTIVE_ROUTE_IDS",
    "PROTECTED_ROUTE_FAMILY_K4A",
    "resolve_active_route",
    "m4_active_chip_payload",
    "m4_candidate_to_compiler_candidate",
    "compile_m4_active_cb_records",
    "assess_baseline_degradation",
    "validate_no_primary_rescue_candidate",
    "is_m4_no_primary_rescue_eligible",
    "RESCUE_NO_PRIMARY_MEANINGFUL",
    "M4_RESCUE_REJECTIONS",
    "run_m4_active_decision",
]
