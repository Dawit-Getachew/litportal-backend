# DEPRECATED (Phase M4.2 — 2026-06-09): Legacy M3.2 active semantic-repair
# decision, superseded by services.m4_active_pipeline.run_m4_active_decision. NOT
# invoked in the active pipeline (gated off via
# pipeline._LEGACY_SEMANTIC_REPAIR_RUNTIME_ENABLED=False) — the pipeline now feeds
# M4 a static DORMANT M3.2 decision. Retained for service-level unit tests.
# M3.2 stays dormant + production-hard-off. Do not re-wire into the forward path.

"""Phase M3.2 — Preview-only Active Semantic Repair orchestrator.

DETERMINISTIC. NO NEW LLM CALL. NO NEW RETRIEVAL.

This module is the **active** counterpart to the M3-preflight
``semantic_repair_active_*`` family. It DOES NOT introduce a new
LLM call: the validator-accepted semantic-repair candidate has
ALREADY been produced upstream in the shadow path (M2.1). M3.2
just decides whether to **promote** that candidate into the
clinician-visible active retrieval pipeline for a narrow allowlist
of grammar shapes and writes an audit sidecar capturing the
decision.

Hard rules (closed-set):
    1. Preview/dev/staging env only. Production fail-closed via
       ``semantic_repair_active_guard`` at startup.
    2. Allowlist enforced via the hard-coded shape constant in
       ``semantic_repair_active_config._DEFAULT_SHAPE_ALLOWLIST``.
       Any non-allowlisted shape emits a closed-set warning log
       ``semantic_repair_active_shape_not_allowlisted`` and falls
       back to A-arm.
    3. K4A.2 / K4A.3 / K4A.4 routes are NEVER promoted. The
       eligibility decision in
       ``semantic_repair_active_eligibility`` already enforces this
       (``route_owned_by_k4a_specialty`` block reason). Defense in
       depth: we re-check ``question_subtype`` here.
    4. Kill switch is two-sourced: env var
       ``SEMANTIC_REPAIR_ACTIVE_KILL_SWITCH`` (default true =
       engaged) AND Mongo flag
       ``feature_flags.m3_2_active_repair_kill_switch``. Either
       engaged = repair NOT attempted. Repair only runs if BOTH
       report disengaged.
    5. Hard timeout ceiling ``SEMANTIC_REPAIR_ACTIVE_MAX_LATENCY_MS``
       (default 3500 ms). On timeout → A-arm fallback, audit
       sidecar records ``timeout_status="m3_2_active_timeout"`` and
       ``fallback_status="fallback_to_a_arm"``.
    6. The A-arm CQM / boolean queries are ALWAYS computed and
       stored in the sidecar, even when B-arm takes over. The
       physician can audit the diff at any time.
    7. PHI dominance: the pipeline reaches this code path only when
       the PHI guard upstream passed. The closed-set candidate
       fields have ALREADY been scrubbed (validator-side). Defense
       in depth: we re-scan the candidate-side strings here too.

The audit sidecar is written to the ``m3_2_session_decisions``
Mongo collection (TTL 30 days). One row per active-pipeline
decision (the same session may be re-searched and produce a new
sidecar row; sidecar is append-only).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from core.db import Collections, get_db
from services.audit import audit_service
from services.m3_1_benchmark import (
    compile_b_arm_queries,
    leak_scan_passes,
)
from services.m3_2_feature_flags import is_mongo_kill_engaged
from services.semantic_repair_active_config import (
    SemanticRepairActiveConfig,
    load_active_config,
)
from services.semantic_repair_active_eligibility import (
    EligibilityDecision,
    is_semantic_repair_active_eligible,
)

logger = logging.getLogger("clinevidence.m3_2_active")


# ---------------------------------------------------------------------
# Closed-set diagnostic enums
# ---------------------------------------------------------------------

#: ``eligibility_decision`` closed set on the sidecar.
M3_2_ELIGIBILITY_DECISIONS: frozenset[str] = frozenset({
    "eligible",
    "ineligible_kill_switch_engaged",
    "ineligible_active_flag_disabled",
    "ineligible_env_not_in_allowlist",
    "ineligible_shape_not_in_allowlist",
    "ineligible_route_owned_by_k4a_specialty",
    "ineligible_detector_not_triggered",
    "ineligible_candidate_not_validated",
    "ineligible_central_anchor_removed",
    "ineligible_phi_blocked",
    "ineligible_latency_above_threshold",
    "ineligible_unknown_clinical_block",
})

#: ``timeout_status`` closed set on the sidecar.
M3_2_TIMEOUT_STATUSES: frozenset[str] = frozenset({
    "none",
    "m3_2_active_timeout",
})

#: ``fallback_status`` closed set on the sidecar.
M3_2_FALLBACK_STATUSES: frozenset[str] = frozenset({
    "none",
    "fallback_to_a_arm",
})

#: Chip variants surfaced to the clinician UI. The primary text is
#: shape-agnostic; the tooltip pulls the human-facing shape label.
M3_2_CHIP_PRIMARY_TEXT: str = "Clinical interpretation refined"

#: Human-facing shape labels — exposed via the tooltip ONLY. Match
#: the closed-set shape allowlist names in
#: ``semantic_repair_active_config._DEFAULT_SHAPE_ALLOWLIST``.
M3_2_SHAPE_HUMAN_LABELS: dict[str, str] = {
    "etiology_risk": "etiology/risk",
    "pathophysiology_mechanism": "pathophysiology/mechanism",
    "procedure_complication": "procedure/device/drug complication risk",
}

#: K4A.* protected subtypes — defense-in-depth. Must match the set in
#: ``services.semantic_repair_active_eligibility._K4A_OWNED_SUBTYPES``.
_K4A_PROTECTED_SUBTYPES: frozenset[str] = frozenset({
    "severity_marker_overview",
    "rule_out_test_utility",
    "epidemiology_incidence_complication_risk",
})


# ---------------------------------------------------------------------
# Pure projection helpers
# ---------------------------------------------------------------------

def chip_payload_for(shape: Optional[str]) -> dict[str, Any]:
    """Closed-set chip payload exposed via the clinician
    ``session_diagnostics`` surface and read by the UI. Returns the
    same shape regardless of whether the chip should render — the UI
    gates visibility on ``applied=True``.
    """
    human = M3_2_SHAPE_HUMAN_LABELS.get((shape or "").strip().lower())
    return {
        "primary_text": M3_2_CHIP_PRIMARY_TEXT,
        "shape_machine": shape,
        "shape_human": human,
        "tooltip_text": (
            "The active interpretation was refined by bounded clinical "
            f"semantic repair ({human}). Source retrieval is unchanged."
            if human else
            "The active interpretation was refined by bounded clinical "
            "semantic repair. Source retrieval is unchanged."
        ),
    }


def m_repaired_cqm_from_candidate(candidate: Optional[dict]) -> dict[str, Any]:
    """Project the validator-accepted candidate into a closed-set
    M-repaired CQM shape suitable for the audit sidecar and the
    clinician transparency surface.

    Returns the closed-set keys ``primary_intent`` / ``frameworks`` /
    ``required_anchors`` / ``branch_count`` / ``semantic_repair_reason``
    / ``confidence``. Never echoes ``slots`` (that's
    ``m_repaired_compiler_input``).

    Pure function. Returns empty placeholders when ``candidate`` is
    None.
    """
    cand = candidate if isinstance(candidate, dict) else {}
    branches = cand.get("branches") or []
    return {
        "primary_intent": cand.get("primary_intent"),
        "frameworks": list(cand.get("frameworks") or []),
        "required_anchors": [
            str(a) for a in (cand.get("required_anchors") or [])
            if isinstance(a, str) and a.strip()
        ],
        "branch_count": int(len(branches)) if isinstance(branches, list) else 0,
        "semantic_repair_reason": cand.get("semantic_repair_reason"),
        "confidence": (
            float(cand.get("confidence"))
            if isinstance(cand.get("confidence"), (int, float)) else None
        ),
    }


def m_repaired_compiler_input_from_candidate(
    candidate: Optional[dict],
) -> dict[str, Any]:
    """Project the candidate fields fed to the Boolean compiler.

    Returns ``primary_intent`` / ``frameworks`` / ``required_anchors``
    / ``slots`` / ``semantic_repair_reason``.
    """
    cand = candidate if isinstance(candidate, dict) else {}
    return {
        "primary_intent": cand.get("primary_intent"),
        "frameworks": list(cand.get("frameworks") or []),
        "required_anchors": [
            str(a) for a in (cand.get("required_anchors") or [])
            if isinstance(a, str) and a.strip()
        ],
        "slots": dict(cand.get("slots") or {}),
        "semantic_repair_reason": cand.get("semantic_repair_reason"),
    }


def compile_m_repaired_cb_records(
    candidate: dict,
    sources: Iterable[str],
) -> list[dict[str, Any]]:
    """Adapter — call ``compile_b_arm_queries`` (from m3_1_benchmark)
    and project its output into the cb_query record shape expected
    by ``services.pipeline``. Each record carries
    ``planner_source="semantic_repair_active"`` so the per-row
    audit trail in ``search_queries`` is unambiguous.

    Pure function: deterministic, no I/O.
    """
    bq = compile_b_arm_queries(candidate)
    sources_set = {s.lower() for s in sources}
    out: list[dict[str, Any]] = []
    src_map = {"pubmed": "pubmed", "epmc": "europe_pmc"}
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
                "query_family": f"semantic_repair_{family}",
                "query_type": f"semantic_repair_{family}",
                "candidate_tier": "semantic_repair_active",
                "query_string": qstr,
                "rationale": "phase_m3_2_active_semantic_repair",
                "concept_blocks_json": {},
                "mandatory_anchors": list(
                    candidate.get("required_anchors") or []
                ),
                "optional_terms": [],
                "planner_source": "semantic_repair_active",
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
# Decision function
# ---------------------------------------------------------------------

def _map_eligibility_to_decision_str(
    elig: EligibilityDecision,
    *,
    kill_engaged: bool,
    config: SemanticRepairActiveConfig,
) -> str:
    """Collapse the closed-set block_reasons into a single closed-set
    ``eligibility_decision`` string suitable for the sidecar."""
    if kill_engaged:
        return "ineligible_kill_switch_engaged"
    block = set(elig.block_reasons or [])
    if "active_flag_disabled" in block:
        return "ineligible_active_flag_disabled"
    if "env_not_in_allowlist" in block:
        return "ineligible_env_not_in_allowlist"
    if "shape_not_in_allowlist" in block:
        return "ineligible_shape_not_in_allowlist"
    if "route_owned_by_k4a_specialty" in block:
        return "ineligible_route_owned_by_k4a_specialty"
    if "detector_did_not_trigger" in block:
        return "ineligible_detector_not_triggered"
    if "candidate_schema_invalid" in block or "candidate_refused_or_unsafe" in block \
            or "candidate_confidence_below_threshold" in block \
            or "generic_only_required_anchor" in block \
            or "unvalidated_mesh_treated_as_validated" in block:
        return "ineligible_candidate_not_validated"
    if "central_anchor_removed" in block:
        return "ineligible_central_anchor_removed"
    if "phi_blocked" in block:
        return "ineligible_phi_blocked"
    if "latency_above_threshold" in block:
        return "ineligible_latency_above_threshold"
    if elig.eligible:
        return "eligible"
    return "ineligible_unknown_clinical_block"


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------

async def run_m3_2_active_decision(
    *,
    session_id: str,
    user_id: Optional[str],
    semantic_repair_block: dict[str, Any],
    picott_payload: dict[str, Any],
    deterministic_primary_anchors: list[dict],
    sources: list[str],
    a_arm_cqm: dict[str, Any],
    a_arm_boolean_queries: list[str],
    config: Optional[SemanticRepairActiveConfig] = None,
    mongo_kill_engaged: Optional[bool] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Main orchestrator. Returns a closed-set dict:

        {
          "applied": bool,                    # True only when B-arm injected
          "decision": str,                    # closed-set eligibility decision
          "shape": str | None,
          "timeout_status": str,              # closed set
          "fallback_status": str,             # closed set
          "latency_ms": int,
          "kill_switch_state": {env, mongo, combined_engaged},
          "a_arm_cqm": dict,
          "a_arm_boolean_queries": list[str],
          "m_repaired_cqm": dict | None,
          "m_repaired_compiler_input": dict | None,
          "m_repaired_boolean_queries": dict | None,
          "m_repaired_cb_records": list[dict] | None,
          "route_owned_by_k4a_specialty": bool,
          "safety_flags": {leak_scan_passed: bool, ...},
          "chip": dict,
          "sidecar_doc_id": str,
        }

    Side effects:
      * Writes a row to ``m3_2_session_decisions``.
      * Emits closed-set audit events ``m3_2_active_decision_applied``
        / ``m3_2_active_decision_fallback`` / ``m3_2_active_timeout``.

    NEVER raises. On any internal error → fallback to A-arm, sidecar
    records the failure, audit event emitted.
    """
    t_start = time.perf_counter()
    now_dt = now if now is not None else datetime.now(timezone.utc)
    cfg = config if config is not None else load_active_config()
    if mongo_kill_engaged is None:
        try:
            mongo_kill_engaged = await is_mongo_kill_engaged()
        except Exception:  # noqa: BLE001
            mongo_kill_engaged = True
    kill_state = {
        "env_kill_switch_engaged": bool(cfg.kill_switch_engaged),
        "mongo_kill_switch_engaged": bool(mongo_kill_engaged),
        "combined_engaged": bool(
            cfg.kill_switch_engaged or mongo_kill_engaged,
        ),
        "fail_closed_default": True,
    }
    kill_engaged = kill_state["combined_engaged"]

    # ---- Build the decision inputs from the upstream shadow block. ----
    triggered = bool(semantic_repair_block.get("semantic_repair_triggered"))
    trigger_reasons = list(
        semantic_repair_block.get("semantic_repair_trigger_reasons") or [],
    )
    cand_dict = semantic_repair_block.get("semantic_repair_candidate")
    cand_validated = (
        semantic_repair_block.get("semantic_repair_validation_status") == "passed"
    )
    candidate = cand_dict if isinstance(cand_dict, dict) and cand_validated else None
    candidate_rejections = list(
        semantic_repair_block.get("semantic_repair_rejection_reasons") or [],
    )
    candidate_latency_ms = int(
        semantic_repair_block.get("semantic_repair_latency_ms") or 0,
    )
    question_subtype = (picott_payload or {}).get("question_subtype")
    route_owned = (
        (question_subtype or "").strip().lower() in _K4A_PROTECTED_SUBTYPES
    )

    # ---- Eligibility decision (clinical gates + operator gates). ----
    elig: EligibilityDecision = is_semantic_repair_active_eligible(
        session_blocked_phi=False,
        detector_triggered=triggered,
        trigger_reasons=trigger_reasons,
        candidate_validated=cand_validated,
        candidate_rejection_reasons=candidate_rejections,
        candidate_dict=cand_dict if isinstance(cand_dict, dict) else None,
        deterministic_primary_anchors=deterministic_primary_anchors or [],
        question_subtype=question_subtype,
        latency_ms=candidate_latency_ms,
        config=cfg,
    )
    decision_str = _map_eligibility_to_decision_str(
        elig, kill_engaged=kill_engaged, config=cfg,
    )

    # Non-allowlisted shape → emit closed-set warning log AND fall back.
    shape = elig.shape
    if shape is not None and not cfg.shape_in_allowlist(shape):
        logger.warning(
            "semantic_repair_active_shape_not_allowlisted attempted_shape=%s "
            "session_id=%s",
            shape, session_id,
        )

    # ---- Try to apply (with timeout). ----
    applied = False
    timeout_status = "none"
    fallback_status = "none"
    m_cqm: Optional[dict[str, Any]] = None
    m_compiler: Optional[dict[str, Any]] = None
    m_queries_struct: Optional[dict[str, Any]] = None
    m_cb_records: Optional[list[dict[str, Any]]] = None

    should_attempt = (
        decision_str == "eligible"
        and not kill_engaged
        and not route_owned
        and candidate is not None
    )

    if should_attempt:
        timeout_s = max(0.001, float(cfg.max_latency_ms) / 1000.0)
        try:
            (m_cqm, m_compiler, m_queries_struct, m_cb_records) = (
                await asyncio.wait_for(
                    _build_m_repaired_artifacts(candidate, sources),
                    timeout=timeout_s,
                )
            )
            applied = True
            timeout_status = "none"
            fallback_status = "none"
        except asyncio.TimeoutError:
            timeout_status = "m3_2_active_timeout"
            fallback_status = "fallback_to_a_arm"
            applied = False
            logger.warning(
                "m3_2_active_timeout session_id=%s timeout_ms=%d shape=%s",
                session_id, cfg.max_latency_ms, shape,
            )
        except Exception:  # noqa: BLE001
            timeout_status = "none"
            fallback_status = "fallback_to_a_arm"
            applied = False
            logger.exception(
                "m3_2_active_apply_failed session_id=%s shape=%s — falling back",
                session_id, shape,
            )

    # When eligible-but-not-attempted (kill switch / route_owned) we
    # still record fallback_status=none (no attempt → no fallback
    # event). The audit chain stays clean: applied=False with the
    # decision string carries the reason.
    if not applied and should_attempt is False:
        timeout_status = "none"
        fallback_status = "none"

    # ---- Leak scan defense in depth. ----
    leak_doc: dict[str, Any] = {
        "decision": decision_str,
        "shape": shape,
        "m_repaired_cqm": m_cqm,
        "m_repaired_compiler_input": m_compiler,
        "m_repaired_boolean_queries": m_queries_struct,
    }
    leak_ok = leak_scan_passes(leak_doc)
    # Phase M3.2 code-review patch — leak scan is now a HARD gate.
    # If the validator-accepted candidate projections somehow
    # contain a PHI / credential / prompt-boilerplate signal, we
    # MUST fall back to A-arm and NOT promote the chip. Inputs are
    # validator-scrubbed upstream so practical risk ≈ 0, but the
    # contract "leak scan rejects PHI-shaped patterns" requires
    # belt-and-suspenders enforcement.
    if applied and not leak_ok:
        logger.warning(
            "m3_2_active_leak_scan_failed session_id=%s shape=%s — falling back",
            session_id, shape,
        )
        applied = False
        fallback_status = "fallback_to_a_arm"
        # Strip the candidate-derived artifacts from the response
        # AND the sidecar so nothing flagged ever lands persisted.
        m_cqm = None
        m_compiler = None
        m_queries_struct = None
        m_cb_records = None

    safety_flags = {
        "leak_scan_passed": bool(leak_ok),
        "kill_switch_state": kill_state,
        "K4A_route_protected": bool(route_owned),
    }

    latency_ms = int((time.perf_counter() - t_start) * 1000)

    # ---- Write audit sidecar. ----
    sidecar_doc_id = uuid.uuid4().hex
    sidecar = {
        "id": sidecar_doc_id,
        "search_session_id": session_id,
        "user_id": user_id,
        "shape": shape,
        "semantic_repair_active_applied": bool(applied),
        "eligibility_decision": decision_str,
        "timeout_status": timeout_status,
        "fallback_status": fallback_status,
        "a_arm_cqm": _safe_copy(a_arm_cqm),
        "a_arm_boolean_queries": list(a_arm_boolean_queries or []),
        "m_repaired_cqm": _safe_copy(m_cqm),
        "m_repaired_compiler_input": _safe_copy(m_compiler),
        "m_repaired_boolean_queries": _safe_copy(m_queries_struct),
        "route_owned_by_k4a_specialty": bool(route_owned),
        "latency_ms": int(latency_ms),
        "kill_switch_state": kill_state,
        "safety_flags": safety_flags,
        "config_snapshot": {
            "enabled": bool(cfg.enabled),
            "current_env": cfg.current_env,
            "env_allowlist": list(cfg.env_allowlist),
            "shapes_allowlist": list(cfg.shapes_allowlist),
            "max_latency_ms": int(cfg.max_latency_ms),
            "prod_allow_active": bool(cfg.prod_allow_active),
            "effective_enabled": bool(cfg.effective_enabled),
        },
        "phase": "M3.2",
        "created_at": now_dt,
    }
    try:
        await get_db()[Collections.M3_2_SESSION_DECISIONS].insert_one(
            dict(sidecar),
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "m3_2_session_decisions insert failed (non-fatal) session_id=%s",
            session_id,
        )

    # ---- Audit event. ----
    audit_event_type = (
        "m3_2_active_decision_applied" if applied
        else (
            "m3_2_active_timeout" if timeout_status == "m3_2_active_timeout"
            else "m3_2_active_decision_fallback"
        )
    )
    try:
        await audit_service.log(
            audit_event_type,
            user_id=user_id,
            search_session_id=session_id,
            payload={
                "eligibility_decision": decision_str,
                "shape": shape,
                "applied": bool(applied),
                "timeout_status": timeout_status,
                "fallback_status": fallback_status,
                "route_owned_by_k4a_specialty": bool(route_owned),
                "kill_switch_combined_engaged": kill_engaged,
                "latency_ms": int(latency_ms),
                "leak_scan_passed": bool(leak_ok),
                "phase": "M3.2",
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception("m3_2 audit event failed (non-fatal)")

    return {
        "applied": bool(applied),
        "decision": decision_str,
        "shape": shape,
        "timeout_status": timeout_status,
        "fallback_status": fallback_status,
        "latency_ms": int(latency_ms),
        "kill_switch_state": kill_state,
        "a_arm_cqm": _safe_copy(a_arm_cqm),
        "a_arm_boolean_queries": list(a_arm_boolean_queries or []),
        "m_repaired_cqm": m_cqm,
        "m_repaired_compiler_input": m_compiler,
        "m_repaired_boolean_queries": m_queries_struct,
        "m_repaired_cb_records": m_cb_records,
        "route_owned_by_k4a_specialty": bool(route_owned),
        "safety_flags": safety_flags,
        "chip": chip_payload_for(shape) if applied else None,
        "sidecar_doc_id": sidecar_doc_id,
    }


async def _build_m_repaired_artifacts(
    candidate: dict,
    sources: list[str],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Deterministic build of (m_cqm, m_compiler_input, m_boolean_queries
    struct, cb_records). Wrapped in a coroutine so the orchestrator
    can ``asyncio.wait_for`` it. No external I/O — purely CPU.
    """
    m_cqm = m_repaired_cqm_from_candidate(candidate)
    m_compiler = m_repaired_compiler_input_from_candidate(candidate)
    m_bq = compile_b_arm_queries(candidate)
    m_queries_struct: dict[str, Any] = {
        "pubmed": [
            {"query_type": q.get("query_type"),
             "query_string": q.get("query_string")}
            for q in (m_bq.get("pubmed") or [])
        ],
        "epmc": [
            {"query_type": q.get("query_type"),
             "query_string": q.get("query_string")}
            for q in (m_bq.get("epmc") or [])
        ],
    }
    cb_records = compile_m_repaired_cb_records(candidate, sources)
    return m_cqm, m_compiler, m_queries_struct, cb_records


def _safe_copy(d: Any) -> Any:
    """Defensive deep-shallow copy for dict/list to avoid leaking refs
    into the audit sidecar (consumers may mutate). ``None``-safe."""
    if d is None:
        return None
    if isinstance(d, dict):
        return {k: _safe_copy(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_safe_copy(v) for v in d]
    return d


# ---------------------------------------------------------------------
# Sidecar read helpers (admin diagnostics)
# ---------------------------------------------------------------------

async def get_session_decision_latest(
    session_id: str,
) -> Optional[dict[str, Any]]:
    """Return the most-recent sidecar row for ``session_id`` or None.

    Read-only. Strips Mongo ``_id`` and re-renders ``created_at`` as
    ISO-8601 for the admin surface.
    """
    db = get_db()
    doc = await db[Collections.M3_2_SESSION_DECISIONS].find_one(
        {"search_session_id": session_id},
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    if not doc:
        return None
    ca = doc.get("created_at")
    if isinstance(ca, datetime):
        doc["created_at"] = ca.isoformat()
    return doc


async def list_recent_session_decisions(
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return the most-recent sidecar rows across all sessions.

    Closed-set projection (the full doc minus ``_id``). Used by the
    admin preview-adjudication panel.
    """
    db = get_db()
    cursor = (
        db[Collections.M3_2_SESSION_DECISIONS]
        .find({}, {"_id": 0})
        .sort([("created_at", -1)])
        .limit(int(limit))
    )
    out: list[dict[str, Any]] = []
    async for d in cursor:
        ca = d.get("created_at")
        if isinstance(ca, datetime):
            d["created_at"] = ca.isoformat()
        out.append(d)
    return out


# ---------------------------------------------------------------------
# Phase M3.2 — Adjudication Export.
# ---------------------------------------------------------------------

#: Closed-set physician verdict values.
M3_2_PHYSICIAN_VERDICTS: frozenset[str] = frozenset({
    "better", "same", "worse", "not_adjudicable",
})

#: Closed-set physician acceptable_yes_no values.
M3_2_ACCEPTABLE_VALUES: frozenset[str] = frozenset({"yes", "no"})

#: Maximum length for the physician reason free-text field.
_PHYSICIAN_REASON_MAX_LEN: int = 1200

#: Maximum chars echoed back for the post-PHI-guard sanitized
#: question preview. We hard-cap so even if the upstream sanitized
#: field is unexpectedly long the export response stays bounded.
_SANITIZED_QUESTION_MAX_LEN: int = 600

#: Closed-set source-error summary keys exposed in the export.
_SOURCE_ERROR_CLOSED_SET: frozenset[str] = frozenset({
    "ok", "timeout", "rate_limited", "auth_failed", "no_results",
    "unknown_error",
})


def _sanitize_question_preview(s: Any) -> Optional[str]:
    """Closed-set echo of the post-PHI-guard sanitized question. The
    upstream PHI guard has already replaced PHI spans with
    ``[REDACTED]`` markers; we ADDITIONALLY:
      * cap length at ``_SANITIZED_QUESTION_MAX_LEN`` chars
      * collapse multiple whitespace runs
      * return ``None`` on non-str / empty input
    """
    if not isinstance(s, str):
        return None
    text = " ".join(s.split())
    if not text:
        return None
    if len(text) > _SANITIZED_QUESTION_MAX_LEN:
        return text[:_SANITIZED_QUESTION_MAX_LEN].rstrip() + "…"
    return text


def _project_top_articles_for_export(
    article_docs: list[dict],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Closed-set article projection — NO abstract, NO full text,
    NO evidence text. Only PMID / title / journal / year / source
    / lane / overall_rank fields. Inputs come from the existing
    ``search_session_articles`` JOIN already used by the admin
    diagnostics endpoint. Defensive against missing fields.
    """
    out: list[dict[str, Any]] = []
    for d in (article_docs or [])[: int(limit)]:
        if not isinstance(d, dict):
            continue
        out.append({
            "pmid": d.get("pmid"),
            "doi": d.get("doi"),
            "title": (d.get("title") or "")[:300] or None,
            "journal": d.get("journal"),
            "publication_year": d.get("publication_year"),
            "source": d.get("source"),
            "lane": d.get("lane") or d.get("article_lane"),
            "overall_rank": d.get("overall_rank"),
        })
    return out


def _summarize_source_errors(
    queries: list[dict],
) -> dict[str, list[str]]:
    """Closed-set per-source error summary. Reads from
    ``search_queries.fallback_reason`` (set when a source fell
    back to legacy or returned zero results) and bins into the
    closed set above. NO provider auth detail, NO API tokens, NO
    URL fragments. Returns ``{source: [closed_set_reasons]}``.
    """
    summary: dict[str, list[str]] = {}
    for q in (queries or []):
        if not isinstance(q, dict):
            continue
        src = q.get("source")
        if not isinstance(src, str):
            continue
        raw_reason = (q.get("fallback_reason") or "").lower()
        if not raw_reason:
            mapped = "ok"
        elif "timeout" in raw_reason:
            mapped = "timeout"
        elif "rate" in raw_reason or "429" in raw_reason:
            mapped = "rate_limited"
        elif "auth" in raw_reason or "401" in raw_reason or "403" in raw_reason:
            mapped = "auth_failed"
        elif "no_results" in raw_reason or "empty" in raw_reason or "zero" in raw_reason:
            mapped = "no_results"
        else:
            mapped = "unknown_error"
        # Defensive — only emit closed-set values.
        if mapped not in _SOURCE_ERROR_CLOSED_SET:
            mapped = "unknown_error"
        summary.setdefault(src, [])
        if mapped not in summary[src]:
            summary[src].append(mapped)
    return summary


async def build_adjudication_export_rows(
    *,
    limit: int = 50,
    shape_filter: Optional[str] = None,
    applied_filter: Optional[bool] = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Phase M3.2 — admin adjudication export. Returns:

      (rows, count_returned, count_leaked_redacted)

    Each row is a closed-set projection joining:
      * the M3.2 sidecar (a_arm_cqm, a_arm_boolean_queries,
        m_repaired_*, shape, applied, fallback_status,
        eligibility_decision, timeout_status, latency_ms)
      * the search session (sanitized_question preview only)
      * the top-10 search_session_articles (PMID/title/journal/
        year/source/lane only — NO abstracts/full text)
      * the per-source error summary (closed-set bin only)
      * the persisted ``physician_adjudication`` block (defaults
        to ``{verdict: None, reason: None, acceptable_yes_no: None}``)

    Defense in depth: every row is run through ``leak_scan_passes``
    BEFORE being added to the output. Failing rows are skipped and
    counted in the redacted-count; the response status will reflect
    this so the operator knows to look at audit_events for the
    closed-set ``m3_2_adjudication_export_leak_blocked`` event.

    NO PHI / prompts / completions / credentials / abstracts /
    full text / evidence text ever surface.
    """
    db = get_db()
    query: dict[str, Any] = {}
    if shape_filter:
        query["shape"] = shape_filter.strip().lower()
    if applied_filter is True:
        query["semantic_repair_active_applied"] = True
    elif applied_filter is False:
        query["semantic_repair_active_applied"] = False

    cursor = (
        db[Collections.M3_2_SESSION_DECISIONS]
        .find(query, {"_id": 0})
        .sort([("created_at", -1)])
        .limit(int(limit))
    )
    sidecars: list[dict[str, Any]] = []
    async for d in cursor:
        ca = d.get("created_at")
        if isinstance(ca, datetime):
            d["created_at"] = ca.isoformat()
        sidecars.append(d)

    out: list[dict[str, Any]] = []
    leaked_count = 0
    for side in sidecars:
        session_id = side.get("search_session_id")
        if not session_id:
            continue
        # ---- Session join (sanitized_question only). ----
        sdoc = await db[Collections.SEARCH_SESSIONS].find_one(
            {"id": session_id},
            {
                "_id": 0,
                "sanitized_question": 1,
                "original_question": 1,
                "status": 1,
                "created_at": 1,
            },
        ) or {}
        sanitized = _sanitize_question_preview(
            sdoc.get("sanitized_question")
            or sdoc.get("original_question"),
        )

        # ---- Top-10 articles join (closed-set fields). ----
        article_cur = (
            db[Collections.SEARCH_SESSION_ARTICLES]
            .find(
                {"search_session_id": session_id},
                {
                    "_id": 0,
                    "pmid": 1, "doi": 1, "title": 1, "journal": 1,
                    "publication_year": 1, "source": 1,
                    "lane": 1, "article_lane": 1, "overall_rank": 1,
                },
            )
            .sort([("overall_rank", 1)])
            .limit(10)
        )
        article_docs: list[dict] = []
        async for ad in article_cur:
            article_docs.append(ad)
        top_articles = _project_top_articles_for_export(article_docs)

        # ---- Per-source error summary (closed-set bin). ----
        query_cur = db[Collections.SEARCH_QUERIES].find(
            {"search_session_id": session_id},
            {"_id": 0, "source": 1, "fallback_reason": 1},
        )
        query_docs: list[dict] = []
        async for qd in query_cur:
            query_docs.append(qd)
        source_errors = _summarize_source_errors(query_docs)

        # ---- Physician adjudication block (default blank). ----
        adj = side.get("physician_adjudication") or {}
        physician_adjudication = {
            "verdict": adj.get("verdict")
                if adj.get("verdict") in M3_2_PHYSICIAN_VERDICTS else None,
            "reason": (
                (adj.get("reason") or "").strip()[:_PHYSICIAN_REASON_MAX_LEN]
                if isinstance(adj.get("reason"), str) else None
            ) or None,
            "acceptable_yes_no": adj.get("acceptable_yes_no")
                if adj.get("acceptable_yes_no") in M3_2_ACCEPTABLE_VALUES else None,
            "updated_by_user_id": adj.get("updated_by_user_id"),
            "updated_at": adj.get("updated_at"),
        }

        row = {
            "search_session_id": session_id,
            "session_status": sdoc.get("status"),
            "session_created_at": (
                sdoc.get("created_at").isoformat()
                if isinstance(sdoc.get("created_at"), datetime)
                else sdoc.get("created_at")
            ),
            "sanitized_question_preview": sanitized,
            "shape": side.get("shape"),
            "applied": bool(side.get("semantic_repair_active_applied")),
            "eligibility_decision": side.get("eligibility_decision"),
            "fallback_status": side.get("fallback_status"),
            "timeout_status": side.get("timeout_status"),
            "latency_ms": int(side.get("latency_ms") or 0),
            "route_owned_by_k4a_specialty": bool(
                side.get("route_owned_by_k4a_specialty")
            ),
            "clinician_chip_shown": bool(
                side.get("semantic_repair_active_applied")
            ),  # chip renders iff applied=true; closed-set boolean.
            "a_arm_cqm": side.get("a_arm_cqm"),
            "m_repaired_cqm": side.get("m_repaired_cqm"),
            "a_arm_boolean_queries": list(
                side.get("a_arm_boolean_queries") or []
            ),
            "m_repaired_boolean_queries": side.get("m_repaired_boolean_queries"),
            "top_10_articles": top_articles,
            "source_errors": source_errors,
            "kill_switch_state": side.get("kill_switch_state"),
            "config_snapshot": side.get("config_snapshot"),
            "physician_adjudication": physician_adjudication,
            "sidecar_created_at": side.get("created_at"),
        }

        # ---- Hard leak gate. ----
        if not leak_scan_passes(row):
            leaked_count += 1
            try:
                await audit_service.log(
                    "m3_2_adjudication_export_leak_blocked",
                    user_id=None,
                    search_session_id=session_id,
                    payload={
                        "shape": side.get("shape"),
                        "applied": bool(
                            side.get("semantic_repair_active_applied")
                        ),
                        "phase": "M3.2",
                    },
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "m3_2_adjudication_export_leak_blocked audit failed",
                )
            out.append({
                "search_session_id": session_id,
                "row_redacted": True,
                "redaction_reason": "leak_scan_failed",
                "shape": side.get("shape"),
            })
            continue
        out.append(row)
    return out, len(out), leaked_count


async def set_physician_adjudication(
    *,
    search_session_id: str,
    actor_user_id: Optional[str],
    verdict: Optional[str] = None,
    reason: Optional[str] = None,
    acceptable_yes_no: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Phase M3.2 — admin PATCH of the physician adjudication
    block on the LATEST sidecar row for a given session.

    Strict closed-set validation:
      * ``verdict`` must be in ``M3_2_PHYSICIAN_VERDICTS`` or None
      * ``acceptable_yes_no`` must be in ``M3_2_ACCEPTABLE_VALUES``
        or None
      * ``reason`` is truncated to ``_PHYSICIAN_REASON_MAX_LEN``
      * ONLY the four ``physician_adjudication.*`` subfields are
        mutated; A-arm CQM / M-repaired CQM / boolean queries /
        latency / kill_switch_state / config_snapshot are
        immutable post-write.

    Returns the updated sidecar row (closed-set projection) or
    ``None`` if no sidecar exists for the session.
    """
    if verdict is not None and verdict not in M3_2_PHYSICIAN_VERDICTS:
        raise ValueError(
            f"verdict must be one of {sorted(M3_2_PHYSICIAN_VERDICTS)} "
            f"or null"
        )
    if acceptable_yes_no is not None and acceptable_yes_no not in M3_2_ACCEPTABLE_VALUES:
        raise ValueError(
            f"acceptable_yes_no must be one of "
            f"{sorted(M3_2_ACCEPTABLE_VALUES)} or null"
        )
    db = get_db()
    side = await db[Collections.M3_2_SESSION_DECISIONS].find_one(
        {"search_session_id": search_session_id},
        {"_id": 0, "id": 1, "physician_adjudication": 1},
        sort=[("created_at", -1)],
    )
    if not side:
        return None
    sidecar_doc_id = side.get("id")
    if not sidecar_doc_id:
        return None
    prev = side.get("physician_adjudication") or {}
    now = datetime.now(timezone.utc)
    next_block = {
        "verdict": verdict if verdict is not None else prev.get("verdict"),
        "reason": (
            (reason or "").strip()[:_PHYSICIAN_REASON_MAX_LEN]
            if reason is not None else prev.get("reason")
        ),
        "acceptable_yes_no": (
            acceptable_yes_no if acceptable_yes_no is not None
            else prev.get("acceptable_yes_no")
        ),
        "updated_by_user_id": actor_user_id,
        "updated_at": now.isoformat(),
    }
    # Closed-set whitelist of fields we touch on the sidecar.
    await db[Collections.M3_2_SESSION_DECISIONS].update_one(
        {"id": sidecar_doc_id},
        {"$set": {"physician_adjudication": next_block}},
    )
    try:
        await audit_service.log(
            "m3_2_physician_adjudication_patched",
            user_id=actor_user_id,
            search_session_id=search_session_id,
            payload={
                "sidecar_doc_id": sidecar_doc_id,
                "fields_changed": [
                    k for k, v in (
                        ("verdict", verdict),
                        ("reason", reason),
                        ("acceptable_yes_no", acceptable_yes_no),
                    ) if v is not None
                ],
                "phase": "M3.2",
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "m3_2_physician_adjudication_patched audit failed (non-fatal)",
        )
    return {
        "search_session_id": search_session_id,
        "sidecar_doc_id": sidecar_doc_id,
        "physician_adjudication": next_block,
    }


__all__ = [
    "M3_2_ELIGIBILITY_DECISIONS",
    "M3_2_TIMEOUT_STATUSES",
    "M3_2_FALLBACK_STATUSES",
    "M3_2_CHIP_PRIMARY_TEXT",
    "M3_2_SHAPE_HUMAN_LABELS",
    "M3_2_PHYSICIAN_VERDICTS",
    "M3_2_ACCEPTABLE_VALUES",
    "chip_payload_for",
    "m_repaired_cqm_from_candidate",
    "m_repaired_compiler_input_from_candidate",
    "compile_m_repaired_cb_records",
    "run_m3_2_active_decision",
    "get_session_decision_latest",
    "list_recent_session_decisions",
    "build_adjudication_export_rows",
    "set_physician_adjudication",
]
