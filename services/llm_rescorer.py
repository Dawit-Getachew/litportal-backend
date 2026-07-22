"""LLM-based relevance rescorer.

The deterministic ``evidence_ranker`` keyword-scores every article and excludes
those scoring < 60. Keyword scoring inevitably misses genuinely-relevant papers
that phrase concepts differently, so a broad retrieval (200 real papers) can
collapse to a handful of shown results. This pass asks an LLM to judge the true
clinical relevance of the borderline / excluded candidates and LIFTS the ones
it rates relevant — boosting their relevance so they are ranked and shown by
default-selection instead of hidden.

Governance mirrors the other AI agents (Northstar posture):
  * ONE batched LLM call per search; the model refers to candidates by integer
    index only — it never emits PMIDs/DOIs/scores/identifiers.
  * Closed-set relevance labels + confidence only; deterministic validation is
    authoritative and only ever LIFTS a score (never lowers a deterministic
    one), and only clears the soft ``low_relevance`` exclusion.
  * Feature-flagged, fail-closed, no-op when the LLM provider is unconfigured.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.base import run_json_agent
from services.audit import audit_service

logger = logging.getLogger("clinevidence.llm_rescorer")

PROMPT_VERSION = "llmrescore-v1"

# Closed-set relevance labels → the relevance score each lifts an article TO.
# "off_topic" never lifts. Thresholds are chosen so 'relevant'+ clear the 60
# exclusion floor while 'marginally_relevant' stays below it (boosted, still
# not surfaced by default).
_LABEL_TARGET_SCORE: dict[str, int] = {
    "highly_relevant": 88,
    "relevant": 72,
    "marginally_relevant": 55,
    "off_topic": 0,
}
_LABELS = tuple(_LABEL_TARGET_SCORE.keys())
_LIFT_FLOOR = 60  # clear low_relevance exclusion only when lifted to >= this


# ---------------------------------------------------------------- flags
def rescorer_enabled() -> bool:
    """Default-ON so no deployment env var is required; set
    LLM_RESCORER_ENABLED=false to disable."""
    return os.environ.get("LLM_RESCORER_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


def rescorer_prod_allowed() -> bool:
    return os.environ.get("LLM_RESCORER_PROD_ALLOWED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _min_confidence() -> float:
    try:
        return float(os.environ.get("LLM_RESCORER_MIN_CONFIDENCE", "0.7"))
    except (TypeError, ValueError):
        return 0.7


def _max_candidates() -> int:
    try:
        return max(1, int(os.environ.get("LLM_RESCORER_MAX_CANDIDATES", "40")))
    except (TypeError, ValueError):
        return 40


# ---------------------------------------------------------------- schema
class RescoreItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    index: int = -1
    label: str = "off_topic"
    confidence: float = 0.0

    @field_validator("label", mode="before")
    @classmethod
    def _coerce_label(cls, v: Any) -> str:
        s = str(v or "").strip().lower().replace(" ", "_")
        return s if s in _LABEL_TARGET_SCORE else "off_topic"

    @field_validator("index", mode="before")
    @classmethod
    def _coerce_index(cls, v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return -1

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_conf(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


class RescoreResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    items: list[RescoreItem] = Field(default_factory=list)

    @field_validator("items", mode="before")
    @classmethod
    def _cap(cls, v: Any) -> Any:
        return v[:200] if isinstance(v, list) else []


_SYSTEM = (
    "You are a clinical-evidence relevance RESCORER. You are given a clinician's "
    "question and a numbered list of candidate articles (title + abstract) that "
    "a keyword ranker scored as low-relevance. For EACH candidate decide how "
    "relevant it truly is to the question and assign a closed-set label:\n"
    "  'highly_relevant'  — directly answers/addresses the question,\n"
    "  'relevant'         — clearly on-topic and useful,\n"
    "  'marginally_relevant' — related background only,\n"
    "  'off_topic'        — not relevant.\n"
    "Refer to each candidate by its given integer index ONLY. Do NOT invent or "
    "echo PMIDs/DOIs/URLs/scores. Return STRICT JSON: an 'items' array of "
    "{index, label, confidence} where confidence is a number in [0,1]."
)

_SCHEMA_HINT = {
    "items": [{"index": 0, "label": "highly_relevant|relevant|marginally_relevant|off_topic", "confidence": 0.0}],
}


# ---------------------------------------------------------------- helpers (pure)
def collect_rescore_candidates(
    ranked: list[dict[str, Any]], cap: int
) -> list[dict[str, Any]]:
    """The deterministically-excluded ``low_relevance`` rows, highest current
    score first (closest to the floor = most likely genuinely relevant), capped.
    Only these soft-excluded rows are eligible — safety/cap exclusions are left
    untouched."""
    pool = [
        r for r in ranked
        if r.get("exclusion_reason") == "low_relevance"
        and (r.get("_article") or {}).get("id")
    ]
    pool.sort(key=lambda r: -(float(r.get("overall_rank_score") or 0.0)))
    out: list[dict[str, Any]] = []
    for r in pool[:cap]:
        art = r["_article"]
        out.append({
            "index": len(out),
            "article_id": art["id"],
            "title": (art.get("title") or "")[:300],
            "abstract": (art.get("abstract") or "")[:900],
        })
    return out


def candidates_block(candidates: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{c['index']}] {c['title']}\n    {c['abstract'] or '(no abstract)'}"
        for c in candidates
    )


def apply_judgments(
    ranked: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    result: RescoreResult,
    *,
    min_confidence: float,
) -> int:
    """Deterministically apply the LLM judgments to ``ranked`` IN PLACE. Only
    LIFTS a row's relevance_score / overall_rank_score (never lowers), stamps
    provenance, and clears the ``low_relevance`` exclusion when lifted to the
    surfacing floor. Returns the number of rows lifted. Pure w.r.t. I/O."""
    by_id: dict[str, dict[str, Any]] = {}
    for r in ranked:
        aid = (r.get("_article") or {}).get("id")
        if aid:
            by_id[aid] = r

    lifted = 0
    for item in result.items:
        target = _LABEL_TARGET_SCORE.get(item.label, 0)
        if target <= 0 or item.confidence < min_confidence:
            continue
        if item.index < 0 or item.index >= len(candidates):
            continue
        r = by_id.get(candidates[item.index]["article_id"])
        if r is None:
            continue
        # Only touch a SOFT low_relevance exclusion; never a safety/cap one.
        if r.get("exclusion_reason") not in (None, "low_relevance"):
            continue
        cur_rel = float(r.get("relevance_score") or 0.0)
        cur_overall = float(r.get("overall_rank_score") or 0.0)
        if target <= cur_overall and target <= cur_rel:
            continue  # deterministic score already >= LLM target
        r["relevance_score"] = float(max(cur_rel, target))
        r["overall_rank_score"] = int(max(cur_overall, target))
        r["llm_rescored"] = True
        r["llm_relevance_label"] = item.label
        r["llm_rescore_confidence"] = round(float(item.confidence), 3)
        if r["overall_rank_score"] >= _LIFT_FLOOR and r.get("exclusion_reason") == "low_relevance":
            r["exclusion_reason"] = None  # un-hide; default-selection decides inclusion
        lifted += 1
    return lifted


# ---------------------------------------------------------------- orchestrator
async def maybe_llm_rescore(
    session_id: str,
    user_id: str,
    question: str,
    ranked: list[dict[str, Any]],
) -> int:
    """Run the LLM rescorer (if enabled) over the deterministically-excluded
    candidates and lift the genuinely-relevant ones. Mutates ``ranked`` in
    place; returns the number lifted. Never raises; no-op when disabled, in
    prod without opt-in, or when the LLM provider is unconfigured."""
    if not rescorer_enabled():
        return 0
    is_prod = (os.environ.get("APP_ENV", "development") or "").strip().lower() == "production"
    if is_prod and not rescorer_prod_allowed():
        return 0

    candidates = collect_rescore_candidates(ranked, _max_candidates())
    if not candidates:
        return 0

    user_prompt = (
        f"Clinical question:\n{question}\n\n"
        f"Candidate articles to rescore (by index):\n{candidates_block(candidates)}"
    )
    model, reason = await run_json_agent(
        system=_SYSTEM,
        user=user_prompt,
        schema_hint=_SCHEMA_HINT,
        validator=RescoreResult.model_validate,
        max_tokens=1800,
        session_id=session_id,
    )
    if model is None:
        return 0

    lifted = apply_judgments(
        ranked, candidates, model, min_confidence=_min_confidence()
    )
    if lifted:
        logger.info(
            "llm_rescorer: session=%s lifted %d of %d candidates",
            session_id, lifted, len(candidates),
        )
    try:
        await audit_service.log(
            "llm_rescore_decision",
            user_id=user_id,
            search_session_id=session_id,
            payload={
                "candidates_reviewed": len(candidates),
                "lifted_count": lifted,
                "min_confidence": _min_confidence(),
                "failure_reason": reason,
                "prompt_version": PROMPT_VERSION,
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception("llm_rescore audit failed (non-fatal)")
    return lifted


__all__ = [
    "PROMPT_VERSION",
    "rescorer_enabled",
    "collect_rescore_candidates",
    "candidates_block",
    "apply_judgments",
    "maybe_llm_rescore",
    "RescoreResult",
    "RescoreItem",
]
