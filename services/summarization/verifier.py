"""Phase 3 CitationVerifierService — independent LLM call PLUS a deterministic
mechanical floor that catches the most dangerous failure modes (fabricated
identifiers, citations to unselected/excluded articles, missing span
citations, overstated certainty given only abstract-level evidence).

The mechanical floor is the FLOOR — even if the LLM verifier returns
`verification_status="passed"`, the floor can override to `failed` with
required revisions. The reverse is never true: if the floor rejects, the
LLM cannot rescue.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import ValidationError

from services.llm_provider import llm_provider as default_provider

from .evidence_spans import EvidenceSpan
from .prompts import VERIFIER_OUTPUT_SHAPE, VERIFIER_SYSTEM_PROMPT
from .schemas import (
    DraftSummary,
    IncorrectCitation,
    UnsupportedClaim,
    VerifierResult,
)

logger = logging.getLogger("clinevidence.verifier")


def _normalize_id(value: str | None) -> str:
    return re.sub(r"\s+", "", (value or "").strip().lower())


def _norm_doi(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v.startswith("doi:"):
        v = v[4:]
    if v.startswith("https://doi.org/"):
        v = v[len("https://doi.org/"):]
    return v


def _norm_pmid(value: str | None) -> str:
    v = re.sub(r"\D", "", (value or ""))
    return v


def _norm_pmcid(value: str | None) -> str:
    v = (value or "").strip().upper()
    if v.startswith("PMCID:"):
        v = v[len("PMCID:"):]
    return v


def _format_floor_revisions(violations: list[str]) -> str:
    """Build a structured revision hint string for the summarizer."""
    if not violations:
        return ""
    lines = ["The previous draft failed mechanical verification:"]
    for v in violations:
        lines.append(f"- {v}")
    lines.append(
        "Fix every issue above. Do not invent new claims. Cite only the "
        "provided evidence_span_id values. Use the citation labels exactly "
        "as provided in the studies list."
    )
    return "\n".join(lines)


class MechanicalVerificationFloor:
    """Deterministic checks that ALWAYS run, regardless of the LLM verdict."""

    def __init__(
        self,
        *,
        spans: list[EvidenceSpan],
        selected_studies: list[dict],
    ) -> None:
        self.spans_by_id = {s.evidence_span_id: s for s in spans}
        self.spans = spans
        self.studies = selected_studies
        self.studies_by_id = {s["article_id"]: s for s in selected_studies}
        self.selected_article_ids = set(self.studies_by_id.keys())
        self.excluded_article_ids = {
            s["article_id"] for s in selected_studies if s.get("_is_excluded")
        }
        # Phase 6B.2 — articles whose ``directness`` is anything other
        # than "direct" must be hedged in the draft. We pre-compute the
        # set + a regex of hedging tokens so the rule can be enforced
        # mechanically on every claim that cites these articles.
        self.indirect_article_ids = {
            s["article_id"] for s in selected_studies
            if s.get("directness") in ("indirect", "override")
        }
        self._HEDGE_RX = re.compile(
            r"\b(indirect(ly)?|extrapolat(?:e|ed|ion)|hedg(?:e|ed|ing)|"
            r"caveat|caution|may not be directly applicable|"
            r"limited applicability|outside the question's population|"
            r"different population|surrogate|inferred|extrapolated"
            r"|cannot be directly|requires caution|tentatively)\b",
            re.IGNORECASE,
        )

    def check(self, draft: DraftSummary) -> tuple[bool, list[str]]:
        """Return (passed, violations).

        `violations` are short, deterministic strings safe to feed back to
        the summarizer as a revision hint (no raw evidence text).
        """
        violations: list[str] = []
        certainty = draft.certainty

        # Rule 1: claim_map must be non-empty unless certainty=insufficient.
        if certainty != "insufficient" and not draft.claim_map:
            violations.append("claim_map is empty but certainty != insufficient")

        # Rule 2: every claim's supporting_evidence_span_ids non-empty + valid.
        for claim in draft.claim_map:
            if not claim.supporting_evidence_span_ids:
                violations.append(
                    f"claim_id={claim.claim_id} has no supporting evidence_span_ids"
                )
                continue
            for sid in claim.supporting_evidence_span_ids:
                span = self.spans_by_id.get(sid)
                if span is None:
                    violations.append(
                        f"claim_id={claim.claim_id} cites unknown evidence_span_id={sid}"
                    )
                    continue
                if span.article_id not in self.selected_article_ids:
                    violations.append(
                        f"claim_id={claim.claim_id} cites span from unselected article"
                    )
                if span.article_id in self.excluded_article_ids:
                    violations.append(
                        f"claim_id={claim.claim_id} cites span from excluded/retracted article"
                    )
            # supporting_article_ids must all be selected (when provided).
            for art_id in (claim.supporting_article_ids or []):
                if art_id not in self.selected_article_ids:
                    violations.append(
                        f"claim_id={claim.claim_id} cites unselected article_id={art_id}"
                    )

        # Rule 3: selected_studies article_id and identifiers must match
        # the provided study metadata exactly.
        for s in draft.selected_studies:
            study = self.studies_by_id.get(s.article_id)
            if not study:
                violations.append(
                    f"selected_studies entry article_id={s.article_id} is not in the selected set"
                )
                continue
            if study.get("_is_excluded"):
                violations.append(
                    f"selected_studies entry article_id={s.article_id} is excluded/retracted"
                )
            # Identifier consistency.
            if s.pmid and _norm_pmid(s.pmid) != _norm_pmid(study.get("pmid")):
                violations.append(
                    f"PMID mismatch on article_id={s.article_id}: model emitted a PMID different from the provided record"
                )
            if s.doi and _norm_doi(s.doi) != _norm_doi(study.get("doi")):
                violations.append(
                    f"DOI mismatch on article_id={s.article_id}: model emitted a DOI different from the provided record"
                )
            if s.pmcid and _norm_pmcid(s.pmcid) != _norm_pmcid(study.get("pmcid")):
                violations.append(
                    f"PMCID mismatch on article_id={s.article_id}: model emitted a PMCID different from the provided record"
                )
            if not s.supporting_evidence_span_ids:
                violations.append(
                    f"selected_studies entry article_id={s.article_id} has no supporting_evidence_span_ids"
                )
            for sid in s.supporting_evidence_span_ids:
                span = self.spans_by_id.get(sid)
                if span is None:
                    violations.append(
                        f"selected_studies article_id={s.article_id} cites unknown evidence_span_id={sid}"
                    )
                elif span.article_id != s.article_id:
                    violations.append(
                        f"selected_studies article_id={s.article_id} cites a span belonging to a different article"
                    )

        # Rule 4: certainty downgrade check.
        # If EVERY provided span is abstract-level, certainty must be
        # low/very_low/insufficient.
        if self.spans:
            all_abstract = all(s.evidence_text_level == "abstract" for s in self.spans)
            if all_abstract and certainty in {"high", "moderate"}:
                violations.append(
                    "Downgrade certainty when only abstract-level evidence is available."
                )

        # Phase 6B.2 — Rule 5: ``indirect_treated_as_direct``.
        # If the draft cites an article whose ``directness`` is
        # ``indirect`` or ``override`` (user override of a capped
        # article) WITHOUT explicit hedging language in the claim text,
        # OR if the overall ``certainty`` is high/moderate while ANY
        # claim cites indirect/override articles, the floor rejects
        # with ``indirect_treated_as_direct``.
        if self.indirect_article_ids:
            for claim in draft.claim_map:
                cited_articles: set[str] = set()
                for sid in (claim.supporting_evidence_span_ids or []):
                    span = self.spans_by_id.get(sid)
                    if span is not None:
                        cited_articles.add(span.article_id)
                for aid in (claim.supporting_article_ids or []):
                    cited_articles.add(aid)
                indirect_cited = cited_articles & self.indirect_article_ids
                if not indirect_cited:
                    continue
                claim_text = (claim.claim_text or "")
                has_hedge = bool(self._HEDGE_RX.search(claim_text))
                claim_high = certainty in {"high", "moderate"}
                if not has_hedge or claim_high:
                    cited_str = ", ".join(sorted(indirect_cited))
                    violations.append(
                        f"indirect_treated_as_direct: claim_id={claim.claim_id} "
                        f"cites indirect/override article(s) [{cited_str}] "
                        f"without explicit hedging language and/or with "
                        f"overall certainty=high/moderate. Add hedging "
                        f"(\"indirect evidence from <citation> suggests…\") "
                        f"and downgrade certainty to low/very_low/insufficient."
                    )

        passed = len(violations) == 0
        return passed, violations


# ---- LLM verifier -----------------------------------------------------------


def _format_studies_block(studies: list[dict]) -> str:
    lines: list[str] = [f"Selected studies ({len(studies)}):"]
    for s in studies:
        lines.append(
            f"- article_id={s['article_id']} | citation_label={s.get('citation_label')} | "
            f"pmid={s.get('pmid') or 'null'} | doi={s.get('doi') or 'null'} | "
            f"pmcid={s.get('pmcid') or 'null'} | evidence_text_level={s.get('evidence_text_level')} | "
            f"title={s.get('title','')!r}"
        )
    return "\n".join(lines)


def _format_spans_block(spans: list[EvidenceSpan]) -> str:
    lines: list[str] = ["Evidence spans:"]
    for sp in spans:
        lines.append(
            f"[evidence_span_id={sp.evidence_span_id}] article_id={sp.article_id} "
            f"section={sp.section_name} level={sp.evidence_text_level}\n"
            f'Text: """{sp.span_text}"""'
        )
    return "\n\n".join(lines)


class CitationVerifierService:
    """Independent verification of a DraftSummary.

    Calls a separate LLM completion (different system prompt) on the SAME
    evidence spans + draft, parses the response into ``VerifierResult``,
    and then enforces the mechanical floor.
    """

    def __init__(self, provider=None):
        self.provider = provider or default_provider

    async def verify(
        self,
        *,
        draft: DraftSummary,
        spans: list[EvidenceSpan],
        selected_studies: list[dict],
        session_id: Optional[str] = None,
    ) -> VerifierResult:
        floor = MechanicalVerificationFloor(
            spans=spans, selected_studies=selected_studies
        )
        floor_pass, floor_violations = floor.check(draft)

        # Always attempt to consult the LLM for semantic verification — even
        # if the floor already failed; it can add useful detail. But if the
        # LLM is not configured, the floor is the only signal we have.
        llm_result: Optional[VerifierResult] = None
        if self.provider.is_configured():
            user_prompt = (
                _format_studies_block(selected_studies)
                + "\n\n"
                + _format_spans_block(spans)
                + "\n\nDraft summary JSON to verify:\n"
                + draft.model_dump_json(indent=None)
                + "\n\nReturn JSON only."
            )
            try:
                raw = await self.provider.complete_json(
                    system=VERIFIER_SYSTEM_PROMPT,
                    user=user_prompt,
                    schema_hint=VERIFIER_OUTPUT_SHAPE,
                    model=self.provider.verifier_model,
                    temperature=0.0,
                    session_id=session_id,
                )
            except Exception:  # noqa: BLE001
                logger.exception("verifier LLM call raised")
                raw = None
            if raw is not None:
                try:
                    llm_result = VerifierResult.model_validate(raw)
                except (ValidationError, Exception):  # noqa: BLE001
                    logger.warning("verifier output failed schema validation")
                    llm_result = None

        # Merge: floor failures override LLM verdict.
        if not floor_pass:
            return VerifierResult(
                verification_status="failed",
                final_answer_safe_to_show=False,
                unsupported_claims=(llm_result.unsupported_claims if llm_result else []),
                incorrect_citations=(llm_result.incorrect_citations if llm_result else []),
                required_revisions=floor_violations + (
                    list(llm_result.required_revisions) if llm_result else []
                ),
                verifier_notes=(
                    "Mechanical verification floor rejected the draft."
                    + (f" LLM notes: {llm_result.verifier_notes}" if (llm_result and llm_result.verifier_notes) else "")
                ),
                verified_summary_json=None,
            )

        # Floor passed. Defer to the LLM verdict — but only if available.
        if llm_result is None:
            # No LLM verifier signal — without it we cannot semantically
            # verify, so we must FAIL safely.
            return VerifierResult(
                verification_status="failed",
                final_answer_safe_to_show=False,
                unsupported_claims=[],
                incorrect_citations=[],
                required_revisions=[
                    "Verifier LLM unavailable; cannot complete semantic verification."
                ],
                verifier_notes="Verifier LLM unavailable — failing safe.",
                verified_summary_json=None,
            )

        # LLM verifier passed and floor passed → accept.
        if llm_result.verification_status == "passed":
            return VerifierResult(
                verification_status="passed",
                final_answer_safe_to_show=True,
                unsupported_claims=[],
                incorrect_citations=[],
                required_revisions=[],
                verifier_notes=llm_result.verifier_notes or "Citation verifier passed.",
                verified_summary_json=draft.model_dump(),
            )

        # LLM rejected — propagate its findings.
        return VerifierResult(
            verification_status="failed",
            final_answer_safe_to_show=False,
            unsupported_claims=list(llm_result.unsupported_claims),
            incorrect_citations=list(llm_result.incorrect_citations),
            required_revisions=list(llm_result.required_revisions),
            verifier_notes=llm_result.verifier_notes or "Citation verifier rejected the draft.",
            verified_summary_json=None,
        )


def build_revision_hint(result: VerifierResult) -> str:
    """Compose a revision message for the summarizer from a failed verify."""
    parts: list[str] = []
    if result.required_revisions:
        parts.append("Required revisions:")
        for r in result.required_revisions:
            parts.append(f"- {r}")
    if result.unsupported_claims:
        parts.append("\nUnsupported claims:")
        for c in result.unsupported_claims:
            parts.append(f"- claim_id={c.claim_id}: {c.reason}")
    if result.incorrect_citations:
        parts.append("\nIncorrect citations:")
        for c in result.incorrect_citations:
            parts.append(
                f"- claim_id={c.claim_id} cites {c.evidence_span_id}: {c.reason}"
            )
    return "\n".join(parts) if parts else "Verifier rejected the draft (no detail)."


citation_verifier = CitationVerifierService()
