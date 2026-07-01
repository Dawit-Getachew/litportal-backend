"""Phase 3 SummarizerService — produces a structured DraftSummary that is
GROUNDED ONLY in the provided evidence spans. No model memory, no
fabrication, no extra-textual claims.

The LLM is treated as a structural transformer: it reorganizes the evidence
spans into the required JSON shape. The verifier service (verifier.py) then
independently checks every claim.

NEVER include raw evidence text in audit logs. NEVER include raw prompts in
audit logs. NEVER include any LLM key fragment.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import ValidationError

from services.llm_provider import llm_provider as default_provider

from .evidence_spans import EvidenceSpan
from .prompts import SUMMARIZER_OUTPUT_SHAPE, SUMMARIZER_SYSTEM_PROMPT
from .schemas import DraftSummary

logger = logging.getLogger("clinevidence.summarizer")


def _format_pico_block(pico: dict | None) -> str:
    """Phase 6B.2 — the summarizer prompt's "Question type" line must
    come from the canonical interpretation helper. Cascade:
    PICOTT → role_weights → multi_intent → legacy → "other"."""
    from services.interpretation_canonical import (  # noqa: WPS433
        canonical_label_for_rank_explanation,
    )
    p = pico or {}
    picott = p.get("picott_json") or {}
    role_weights = (
        p.get("role_weights_json") or p.get("role_weights") or {}
    )
    multi_intent = (
        p.get("multi_intent_json") or p.get("multi_intent") or {}
    )
    legacy = (
        p.get("clinical_question_type") or p.get("question_type") or ""
    )
    qtype_label = canonical_label_for_rank_explanation(
        picott=picott,
        role_weights=role_weights,
        multi_intent=multi_intent,
        legacy_qtype=legacy,
    )
    if qtype_label == "other":
        qtype_label = "(not specified)"
    return (
        "Parsed PICO:\n"
        f"- Population: {p.get('population') or '(not specified)'}\n"
        f"- Intervention/Exposure: {p.get('intervention') or p.get('exposure') or '(not specified)'}\n"
        f"- Comparator: {p.get('comparator') or '(not specified)'}\n"
        f"- Outcomes: {p.get('outcomes') or '(not specified)'}\n"
        f"- Question type: {qtype_label}\n"
    )


def _format_selected_studies_block(studies: list[dict]) -> str:
    lines: list[str] = [f"Selected studies ({len(studies)}):"]
    for i, s in enumerate(studies, 1):
        cit = s.get("citation_label") or "article:?"
        title = (s.get("title") or "(untitled)").strip()
        journal = s.get("journal") or ""
        year = s.get("publication_year") or ""
        stype = s.get("study_type") or s.get("article_type_classification") or "unspecified"
        level = s.get("evidence_text_level") or "abstract"
        lines.append(
            f"{i}. [article_id={s['article_id']} | {cit}] {title} — "
            f"{journal} {year} — study_type={stype} — evidence_text_level={level}"
        )
    return "\n".join(lines)


def _format_spans_block(spans: list[EvidenceSpan]) -> str:
    lines: list[str] = ["Evidence spans (these are the ONLY texts you may use):"]
    for sp in spans:
        head = (
            f"[evidence_span_id={sp.evidence_span_id}]\n"
            f"  Article: {sp.citation_label} ({sp.title}"
        )
        if sp.journal or sp.publication_year:
            head += f", {sp.journal or ''} {sp.publication_year or ''}".rstrip()
        head += f") — section: {sp.section_name} — level: {sp.evidence_text_level}\n"
        lines.append(head + f'  Text: """{sp.span_text}"""')
    return "\n\n".join(lines)


def _format_override_caveats_block(caveats: list[dict] | None) -> str:
    """Phase 6B.2 — when one or more selected articles are user-override
    or indirect (i.e., had cap_reasons from the picott_aware_ranker
    but were still selected), the summarizer MUST be told so the model
    hedges claims that lean on those articles instead of treating them
    as direct evidence.
    """
    if not caveats:
        return ""
    lines: list[str] = [
        "⚠ Override / indirect evidence caveats:",
        "Some selected articles are NOT direct evidence for this clinical",
        "question. When a claim depends on any of the items below, the",
        "claim text MUST include explicit hedging language (e.g., \"indirect",
        "evidence from <citation> suggests…\") and the certainty MUST be",
        "low / very_low / insufficient — never high or moderate. Treating",
        "any of these as direct evidence is a verifier-blocking error.",
    ]
    for c in caveats:
        cap = ", ".join(c.get("cap_reasons") or []) or "indirectness_flag"
        lines.append(
            f"- [article_id={c['article_id']} | {c.get('citation_label')}] "
            f"directness={c.get('directness')} ; cap_reasons={cap}"
        )
    return "\n".join(lines)


def _build_user_prompt(
    *,
    question: str,
    pico: dict | None,
    studies: list[dict],
    spans: list[EvidenceSpan],
    revision_hint: Optional[str] = None,
    override_caveats: Optional[list[dict]] = None,
) -> str:
    parts: list[str] = [
        f"Clinical question: {question}",
        "",
        _format_pico_block(pico),
        "",
        _format_selected_studies_block(studies),
    ]
    caveat_block = _format_override_caveats_block(override_caveats)
    if caveat_block:
        parts.extend(["", caveat_block])
    parts.extend([
        "",
        _format_spans_block(spans),
        "",
        "Return JSON only.",
    ])
    if revision_hint:
        parts.append("")
        parts.append("REVISION REQUIRED. Previous attempt failed verification:")
        parts.append(revision_hint)
        parts.append("Produce a corrected JSON that addresses every violation above. Do not introduce new claims.")
    return "\n".join(parts)


class SummarizerService:
    """Calls the LLM, parses + validates the response into a DraftSummary.

    On any failure (LLM not configured, network error, parse failure,
    schema mismatch) returns `None` and the caller treats it as a safe
    failure.
    """

    def __init__(self, provider=None):
        self.provider = provider or default_provider

    async def draft(
        self,
        *,
        sanitized_question: str,
        pico: dict | None,
        selected_studies: list[dict],
        spans: list[EvidenceSpan],
        revision_hint: Optional[str] = None,
        session_id: Optional[str] = None,
        override_caveats: Optional[list[dict]] = None,
    ) -> tuple[Optional[DraftSummary], Optional[str]]:
        """Return (draft, failure_reason).

        - If LLM not configured → (None, "llm_not_configured").
        - If LLM returned non-JSON or schema fail → (None, "parser_validation_failed").
        - On other exceptions → (None, "other").

        Phase 6B.2 — ``override_caveats`` is an optional list of
        ``{article_id, citation_label, directness, cap_reasons, ...}``
        dicts. When present, the model is explicitly told these
        articles are NOT direct evidence; the prompt instructs the
        model to add hedging language and forbids high/moderate
        certainty on claims that depend on them. The verifier
        independently enforces the same rule.
        """
        if not self.provider.is_configured():
            return None, "llm_not_configured"
        user_prompt = _build_user_prompt(
            question=sanitized_question,
            pico=pico,
            studies=selected_studies,
            spans=spans,
            revision_hint=revision_hint,
            override_caveats=override_caveats,
        )
        try:
            raw = await self.provider.complete_json(
                system=SUMMARIZER_SYSTEM_PROMPT,
                user=user_prompt,
                schema_hint=SUMMARIZER_OUTPUT_SHAPE,
                model=self.provider.summary_model,
                temperature=0.0,
                session_id=session_id,
            )
        except Exception:  # noqa: BLE001
            logger.exception("summarizer LLM call raised")
            return None, "other"
        if raw is None:
            return None, "parser_validation_failed"
        try:
            return DraftSummary.model_validate(raw), None
        except ValidationError:
            logger.warning("summarizer output failed schema validation")
            return None, "parser_validation_failed"
        except Exception:  # noqa: BLE001
            logger.exception("summarizer parse raised")
            return None, "parser_validation_failed"


summarizer = SummarizerService()
