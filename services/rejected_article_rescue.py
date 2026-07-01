"""Phase R2A — Rejected-Article Rescue Classifier (SHADOW ONLY).

A bounded-AI, admin-only, shadow diagnostic that reviews the
REJECTED / capped-as-low-relevance article list for a session and flags
records that may deserve borderline/review status rather than outright
rejection. It NEVER ranks, selects, surfaces, or re-orders articles for
clinicians, and NEVER calls source APIs or generates identifiers.

Hard guarantees (Northstar v4.0 §4/§7.3):
  * No clinician-visible ranking / default-selection / evidence-packet /
    summary / citation-verifier change. Output is admin-only shadow.
  * The LLM returns CLOSED-SET labels/reasons + confidence + a short
    rationale ONLY. It never returns article IDs, PMIDs, DOIs, citations,
    or ranking scores — every identifier is supplied deterministically by
    us.
  * Deterministic validation is the authority: schema, closed-set enums,
    confidence threshold, policy-exclusion override, central-disease
    match, and a no-identifier guard.
  * Policy-excluded records (retracted / preprint / expression-of-concern)
    stay excluded regardless of the LLM.
  * No PHI / prompt / completion / credential / full-text / evidence-text
    leakage: the shadow OUTPUT carries only bibliographic metadata +
    closed-set tokens (never abstracts/full text). Abstracts may be sent
    to the LLM as INPUT only (already-retrieved, display-allowed).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from services.llm_provider import llm_provider

logger = logging.getLogger("clinevidence.r2a")

SCHEMA_VERSION = "R2A.1.0"

# --- Closed-set vocabularies ------------------------------------------
RESCUE_LABELS = (
    "direct_relevance_candidate",
    "borderline_relevance_candidate",
    "indirect_relevance_candidate",
    "keep_rejected",
    "policy_excluded",
)
RESCUE_CANDIDATE_LABELS = frozenset({
    "direct_relevance_candidate",
    "borderline_relevance_candidate",
    "indirect_relevance_candidate",
})

RELEVANCE_REASONS = (
    "stage_specific_prognosis",
    "survival_outcome",
    "recurrence_outcome",
    "cancer_specific_mortality",
    "prognostic_model",
    "localized_high_risk_context",
    "advanced_metastatic_wrong_stage",
    "treatment_not_prognosis",
    "diagnostic_not_prognosis",
    "screening_not_prognosis",
    "mechanism_not_prognosis",
    "wrong_disease",
    "wrong_population",
    "non_article",
    "preprint_policy",
    "retracted_policy",
)

# Deterministic priority for ordering the top rescue candidates (higher
# first). Non-candidate labels never appear in the top list.
_LABEL_PRIORITY = {
    "direct_relevance_candidate": 3,
    "borderline_relevance_candidate": 2,
    "indirect_relevance_candidate": 1,
}


class RescueClassification(BaseModel):
    """Strict closed-set classifier output. ``extra='forbid'`` rejects any
    LLM-injected field (e.g. an article id, pmid, doi, score)."""

    model_config = ConfigDict(extra="forbid")

    rescue_label: Literal[RESCUE_LABELS]  # type: ignore[valid-type]
    relevance_reason: Literal[RELEVANCE_REASONS]  # type: ignore[valid-type]
    confidence: float = Field(ge=0.0, le=1.0)
    short_rationale: str = Field(default="", max_length=240)


SYSTEM_PROMPT = (
    "You are a SHADOW relevance triage classifier for a clinical "
    "literature search tool. You are given ONE rejected or low-relevance "
    "article plus the clinician's question and its structured "
    "interpretation. Decide whether the article might deserve review "
    "rather than outright rejection.\n\n"
    "You MUST NOT: rank or score articles, select articles, invent or "
    "echo PMIDs / DOIs / URLs / article identifiers, fabricate citations, "
    "answer the clinical question, or call any external source.\n\n"
    "Return ONLY a strict JSON object with EXACTLY these keys: "
    "rescue_label, relevance_reason, confidence, short_rationale.\n"
    f"rescue_label MUST be one of: {', '.join(RESCUE_LABELS)}.\n"
    f"relevance_reason MUST be one of: {', '.join(RELEVANCE_REASONS)}.\n"
    "confidence is a number between 0 and 1. short_rationale is at most "
    "240 characters and must NOT contain patient identifiers, quotes from "
    "the abstract, PMIDs, DOIs, or URLs.\n"
    "Prefer keep_rejected when the article is clearly off-topic, and "
    "policy_excluded for retracted / preprint / non-article records."
)


def _confidence_threshold() -> float:
    try:
        return float(os.environ.get("R2A_CONFIDENCE_THRESHOLD", "0.6"))
    except ValueError:
        return 0.6


def _max_articles() -> int:
    try:
        return max(1, int(os.environ.get("R2A_MAX_ARTICLES", "30")))
    except ValueError:
        return 30


def r2a_shadow_enabled() -> bool:
    """Preview/dev only by default. In production this shadow classifier
    is OFF unless explicitly enabled via ``R2A_RESCUE_PROD_ENABLED=true``
    (NOT set). Mirrors the L1.0.1 / M4 production-safety posture."""
    is_prod = os.environ.get("APP_ENV", "development").strip().lower() == "production"
    if not is_prod:
        return True
    return os.environ.get("R2A_RESCUE_PROD_ENABLED", "false").strip().lower() == "true"


# --- Deterministic helpers --------------------------------------------

def _article_text(article_doc: dict) -> str:
    return " ".join([
        str(article_doc.get("title") or ""),
        str(article_doc.get("abstract") or ""),
    ]).lower()


def _pub_types(article_doc: dict) -> list[str]:
    pts = (
        article_doc.get("publication_types")
        or article_doc.get("publication_types_json")
        or article_doc.get("pub_types")
        or []
    )
    return [str(p).strip().lower() for p in pts if str(p).strip()]


def is_policy_excluded(article_doc: dict, sa_row: dict) -> Optional[str]:
    """Return a closed-set policy-exclusion reason, or ``None``. These
    records stay excluded regardless of the LLM."""
    cap_reasons = {str(c).strip().lower() for c in (sa_row.get("cap_reasons") or [])}
    if article_doc.get("is_retracted") or cap_reasons & {
        "retracted", "expression_of_concern",
    }:
        return "retracted_policy"
    if article_doc.get("is_preprint") or "preprint_for_clinical_recommendation" in cap_reasons:
        return "preprint_policy"
    pts = _pub_types(article_doc)
    if pts and not any(
        t in p for p in pts for t in (
            "journal article", "review", "trial", "study", "guideline",
            "meta-analysis", "comparative",
        )
    ):
        # No recognizable article-type token → treat as non-article.
        if any("retracted" in p for p in pts):
            return "retracted_policy"
    return None


def central_disease_tokens(cqm: dict) -> list[str]:
    """Closed set of disease/anchor tokens from the ClinicalQuestionMap,
    plus their individual long words, used for the deterministic
    central-disease match guard."""
    cqm = cqm if isinstance(cqm, dict) else {}
    phrases: list[str] = []
    for key in ("clinical_problem", "required_anchors", "primary_clinical_anchors"):
        v = cqm.get(key)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    item = item.get("text") or item.get("canonical_concept") or ""
                s = str(item).strip().lower()
                if s:
                    phrases.append(s)
    tokens: set[str] = set()
    for p in phrases:
        tokens.add(p)
        for w in p.split():
            if len(w) > 4:
                tokens.add(w)
    return sorted(tokens)


def _central_disease_present(article_doc: dict, tokens: list[str]) -> bool:
    if not tokens:
        return True  # cannot validate → do not block on this guard
    text = _article_text(article_doc)
    return any(t in text for t in tokens)


def build_classifier_input(
    *,
    question: str,
    cqm_summary: dict,
    article_doc: dict,
    sa_row: dict,
    include_abstract: bool,
) -> str:
    """Closed-set, bounded LLM input. NO full text; abstract included only
    when allowed + truncated. NO PHI (question is already PHI-cleared)."""
    abstract = ""
    if include_abstract:
        abstract = str(article_doc.get("abstract") or "").strip()[:1200]
    lines = [
        f"user_question: {str(question or '').strip()[:600]}",
        f"question_intent: {cqm_summary.get('primary_intent') or 'unspecified'}",
        f"question_clinical_problem: {cqm_summary.get('clinical_problem') or []}",
        f"question_required_anchors: {cqm_summary.get('required_anchors') or []}",
        f"article_title: {str(article_doc.get('title') or '').strip()[:400]}",
        f"article_journal: {str(article_doc.get('journal') or '').strip()[:200]}",
        f"article_year: {article_doc.get('publication_year') or article_doc.get('year') or 'unknown'}",
        f"article_publication_types: {_pub_types(article_doc)[:8]}",
        f"article_source: {list(article_doc.get('sources_present') or [])[:6]}",
        f"current_relevance_score: {sa_row.get('overall_rank')}",
        f"current_rejection_reason: {sa_row.get('exclusion_reason') or 'low_relevance'}",
        f"current_cap_reasons: {list(sa_row.get('cap_reasons') or [])[:8]}",
        f"current_lane_labels: {list(sa_row.get('flags') or [])[:8]}",
    ]
    if abstract:
        lines.append(f"article_abstract: {abstract}")
    lines.append("\nClassify this article. Return JSON only.")
    return "\n".join(lines)


def validate_classification(
    raw: Optional[dict],
    *,
    central_disease_present: bool,
    policy_excluded_reason: Optional[str],
) -> tuple[Optional[RescueClassification], list[str]]:
    """Deterministic validation gate. Returns ``(classification, rejections)``.
    Even on a 'valid' parse we OVERRIDE the label for policy-excluded
    records and for central-disease mismatches — the deterministic layer
    is the authority."""
    rejections: list[str] = []
    if not isinstance(raw, dict):
        return None, ["empty_output"]

    # No-identifier guard: the LLM must never emit article identifiers.
    for forbidden in ("article_id", "id", "pmid", "doi", "url", "score", "rank"):
        if forbidden in raw:
            rejections.append("llm_emitted_identifier_or_score")
            break

    try:
        parsed = RescueClassification(**raw)
    except ValidationError:
        return None, (rejections + ["schema_invalid"]) or ["schema_invalid"]

    if rejections:
        return None, rejections

    # Confidence threshold → downgrade to keep_rejected.
    if parsed.confidence < _confidence_threshold():
        return (
            RescueClassification(
                rescue_label="keep_rejected",
                relevance_reason=parsed.relevance_reason,
                confidence=parsed.confidence,
                short_rationale="below_confidence_threshold",
            ),
            ["below_confidence_threshold"],
        )

    # Policy-excluded records stay excluded regardless of the LLM.
    if policy_excluded_reason is not None:
        return (
            RescueClassification(
                rescue_label="policy_excluded",
                relevance_reason=policy_excluded_reason,
                confidence=parsed.confidence,
                short_rationale="deterministic_policy_exclusion",
            ),
            ["policy_excluded_override"],
        )

    # Central-disease mismatch → a rescue candidate cannot be promoted.
    if parsed.rescue_label in RESCUE_CANDIDATE_LABELS and not central_disease_present:
        return (
            RescueClassification(
                rescue_label="keep_rejected",
                relevance_reason="wrong_disease",
                confidence=parsed.confidence,
                short_rationale="central_disease_mismatch",
            ),
            ["central_disease_mismatch"],
        )

    return parsed, []


async def _classify_one(
    *,
    question: str,
    cqm_summary: dict,
    article_doc: dict,
    sa_row: dict,
    central_present: bool,
    policy_reason: Optional[str],
) -> tuple[Optional[RescueClassification], list[str], int]:
    # Skip the LLM entirely for policy-excluded records (deterministic).
    if policy_reason is not None:
        return (
            RescueClassification(
                rescue_label="policy_excluded",
                relevance_reason=policy_reason,
                confidence=1.0,
                short_rationale="deterministic_policy_exclusion",
            ),
            ["policy_excluded_override"],
            0,
        )
    user_payload = build_classifier_input(
        question=question, cqm_summary=cqm_summary, article_doc=article_doc,
        sa_row=sa_row, include_abstract=True,
    )
    t0 = time.perf_counter()
    raw: Optional[dict] = None
    try:
        raw = await llm_provider.complete_json(
            system=SYSTEM_PROMPT, user=user_payload,
            temperature=0.0, max_tokens=320,
        )
    except Exception:  # noqa: BLE001
        logger.exception("r2a classify llm call failed (non-fatal)")
        raw = None
    latency_ms = int(round((time.perf_counter() - t0) * 1000))
    classification, rejections = validate_classification(
        raw, central_disease_present=central_present,
        policy_excluded_reason=policy_reason,
    )
    return classification, rejections, latency_ms


def _priority_key(label: str, confidence: float, relevance_score: Any) -> tuple:
    try:
        rs = float(relevance_score or 0.0)
    except (TypeError, ValueError):
        rs = 0.0
    return (_LABEL_PRIORITY.get(label, 0), float(confidence or 0.0), rs)


def _simulate_rescue_landing(
    candidates: list[dict], *, included_floor: float = 60.0,
) -> dict[str, Any]:
    """Shadow-only simulation: describe (deterministically) where rescue
    candidates WOULD land if allowed back into the candidate pool. Does
    NOT alter live ranking / selection / packet."""
    would_reach_review = [
        c for c in candidates
        if c["rescue_label"] in RESCUE_CANDIDATE_LABELS
    ]
    return {
        "note": "shadow_only_no_live_change",
        "included_floor_score": included_floor,
        "candidates_that_would_enter_review_pool": len(would_reach_review),
        "by_label": {
            lbl: sum(1 for c in would_reach_review if c["rescue_label"] == lbl)
            for lbl in _LABEL_PRIORITY
        },
    }


async def run_rescue_shadow(
    *,
    question: str,
    cqm: dict,
    rejected_rows: list[dict],
    articles_by_id: dict[str, dict],
    max_articles: Optional[int] = None,
) -> dict[str, Any]:
    """Orchestrate the shadow rescue classification over a session's
    rejected/capped articles. Returns an admin-only, closed-set
    diagnostics block. Never raises."""
    limit = max_articles or _max_articles()
    cqm = cqm if isinstance(cqm, dict) else {}
    cqm_summary = {
        "primary_intent": cqm.get("primary_intent") or cqm.get("type_of_question"),
        "clinical_problem": (cqm.get("clinical_problem") or [])[:6]
        if isinstance(cqm.get("clinical_problem"), list) else [],
        "required_anchors": (cqm.get("required_anchors") or [])[:8]
        if isinstance(cqm.get("required_anchors"), list) else [],
    }
    disease_tokens = central_disease_tokens(cqm)

    rows = list(rejected_rows)[:limit]

    async def _work(sa_row: dict):
        art = articles_by_id.get(sa_row.get("article_id"))
        if not art:
            return None
        policy_reason = is_policy_excluded(art, sa_row)
        central_present = _central_disease_present(art, disease_tokens)
        classification, rejections, latency = await _classify_one(
            question=question, cqm_summary=cqm_summary, article_doc=art,
            sa_row=sa_row, central_present=central_present,
            policy_reason=policy_reason,
        )
        if classification is None:
            return {
                "article_id": sa_row.get("article_id"),
                "pmid": art.get("pmid"),
                "title": str(art.get("title") or "")[:300],
                "journal": str(art.get("journal") or "")[:160],
                "year": art.get("publication_year") or art.get("year"),
                "current_relevance_score": sa_row.get("overall_rank"),
                "current_rejection_reason": sa_row.get("exclusion_reason") or "low_relevance",
                "rescue_label": "keep_rejected",
                "relevance_reason": "wrong_disease",
                "confidence": 0.0,
                "short_rationale": "classifier_unavailable_or_rejected",
                "validation_rejections": rejections,
                "latency_ms": latency,
            }
        return {
            "article_id": sa_row.get("article_id"),
            "pmid": art.get("pmid"),
            "title": str(art.get("title") or "")[:300],
            "journal": str(art.get("journal") or "")[:160],
            "year": art.get("publication_year") or art.get("year"),
            "current_relevance_score": sa_row.get("overall_rank"),
            "current_rejection_reason": sa_row.get("exclusion_reason") or "low_relevance",
            "rescue_label": classification.rescue_label,
            "relevance_reason": classification.relevance_reason,
            "confidence": round(classification.confidence, 3),
            "short_rationale": classification.short_rationale,
            "validation_rejections": rejections,
            "latency_ms": latency,
        }

    sem = asyncio.Semaphore(5)

    async def _bounded(sa_row):
        async with sem:
            return await _work(sa_row)

    results = [r for r in await asyncio.gather(*[_bounded(r) for r in rows]) if r]

    counts = {lbl: 0 for lbl in RESCUE_LABELS}
    for r in results:
        counts[r["rescue_label"]] = counts.get(r["rescue_label"], 0) + 1

    rescue_candidates = [
        r for r in results if r["rescue_label"] in RESCUE_CANDIDATE_LABELS
    ]
    rescue_candidates.sort(
        key=lambda r: _priority_key(
            r["rescue_label"], r["confidence"], r["current_relevance_score"],
        ),
        reverse=True,
    )
    for i, r in enumerate(rescue_candidates):
        r["priority_rank"] = i + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "shadow",
        "no_clinician_visible_change": True,
        "llm_configured": llm_provider.is_configured(),
        "model": llm_provider.summary_model,
        "confidence_threshold": _confidence_threshold(),
        "total_rejected_reviewed": len(results),
        "direct_rescue_candidates": counts.get("direct_relevance_candidate", 0),
        "borderline_candidates": counts.get("borderline_relevance_candidate", 0),
        "indirect_candidates": counts.get("indirect_relevance_candidate", 0),
        "keep_rejected": counts.get("keep_rejected", 0),
        "policy_excluded": counts.get("policy_excluded", 0),
        "top_rescue_candidates": rescue_candidates[:15],
        "rejection_reason_vs_rescue": [
            {
                "article_id": r["article_id"],
                "pmid": r["pmid"],
                "current_rejection_reason": r["current_rejection_reason"],
                "rescue_label": r["rescue_label"],
                "rescue_reason": r["relevance_reason"],
            }
            for r in results
        ],
        "rescue_simulation": _simulate_rescue_landing(rescue_candidates),
    }


__all__ = [
    "SCHEMA_VERSION",
    "RESCUE_LABELS",
    "RESCUE_CANDIDATE_LABELS",
    "RELEVANCE_REASONS",
    "RescueClassification",
    "r2a_shadow_enabled",
    "is_policy_excluded",
    "central_disease_tokens",
    "build_classifier_input",
    "validate_classification",
    "run_rescue_shadow",
]
