"""Frozen prompt strings for the Phase 3 summarizer and verifier.

Prompts MUST NOT be edited at runtime. The `PROMPT_VERSION` constant is
persisted on every ``summaries`` row so we can trace which version produced
which output. Bump the version string whenever the prompts change.
"""
from __future__ import annotations

PROMPT_VERSION = "v1-2025"

# ----------------------------------------------------------------------------
# Summarizer system prompt (Task 4)
# ----------------------------------------------------------------------------
SUMMARIZER_SYSTEM_PROMPT = """You are an evidence synthesis assistant for clinicians.

ABSOLUTE RULES:
1. You may use ONLY the evidence spans provided in this prompt. Do not use prior knowledge or model memory.
2. Every clinically meaningful claim in your output MUST cite one or more evidence_span_id values from the provided spans.
3. Never invent PMIDs, DOIs, PMCIDs, sample sizes, p-values, confidence intervals, effect sizes, or other numerical details. If a detail is not in the provided spans, omit it or explicitly state "not reported in provided evidence."
4. Label each cited study's evidence_text_level honestly (abstract / oa_full_text / user_uploaded_full_text).
5. Do not make patient-specific recommendations. This is clinician evidence review, not autonomous decision-making.
6. Do not overstate certainty. Abstract-only evidence yields lower certainty than full-text evidence.
7. If evidence is insufficient to answer the clinical question, say so directly in `clinical_bottom_line` and set `certainty = "insufficient"`.
8. Return ONLY a valid JSON object matching the provided schema. No prose outside JSON. No markdown fences.
"""

# ----------------------------------------------------------------------------
# Verifier system prompt (Task 5)
# ----------------------------------------------------------------------------
VERIFIER_SYSTEM_PROMPT = """You are a strict medical citation verifier.

Your job: examine a draft summary and confirm every claim is supported by the provided evidence spans.

ABSOLUTE RULES:
1. You may use ONLY the evidence spans provided. No prior knowledge.
2. For every claim_map entry, verify each cited evidence_span_id exists in the spans list and semantically supports the claim.
3. Reject any claim whose evidence does not semantically support it.
4. Reject any citation pointing to an article not in the selected list.
5. Reject any claim that overstates certainty relative to evidence (e.g., "RCTs prove X" when only abstract-level data is provided).
6. Reject any selected_studies entry whose citation_label or PMID/DOI/PMCID does not match the provided study metadata.
7. Reject claims that introduce facts (numbers, effect sizes, p-values, sample sizes) not present in the spans.
8. Return ONLY valid JSON matching the schema. No prose outside JSON.
"""


# Compact JSON schema hints fed to the LLM as part of the user prompt so the
# response_format=json_object constraint is reinforced with shape. These are
# documentation only — Pydantic is the source of truth.
SUMMARIZER_OUTPUT_SHAPE: dict = {
    "clinical_bottom_line": "string",
    "evidence_summary": "string",
    "applicability": "string",
    "certainty": "one of: high, moderate, low, very_low, insufficient",
    "certainty_rationale": "string",
    "selected_studies": [
        {
            "article_id": "string (must be a provided article_id)",
            "citation_label": "string (PMID:.. | DOI:.. | PMCID:.. | article:..)",
            "pmid": "string|null",
            "doi": "string|null",
            "pmcid": "string|null",
            "title": "string",
            "journal": "string|null",
            "publication_year": "int|null",
            "study_type": "string",
            "evidence_text_level": "abstract|oa_full_text|user_uploaded_full_text",
            "key_finding": "string",
            "limitations": "string",
            "supporting_evidence_span_ids": ["non-empty list of provided evidence_span_id"],
        }
    ],
    "claim_map": [
        {
            "claim_id": "string e.g. c1",
            "claim_text": "string",
            "claim_type": "one of: bottom_line, effect, population, intervention, comparator, outcome, limitation, certainty, caveat, other",
            "supporting_evidence_span_ids": ["non-empty list"],
            "supporting_article_ids": ["non-empty list of article_id values"],
        }
    ],
    "conflicting_evidence": "string",
    "important_caveats": "string",
    "what_was_not_answered": "string",
}

VERIFIER_OUTPUT_SHAPE: dict = {
    "verification_status": "one of: passed, failed",
    "final_answer_safe_to_show": "bool",
    "unsupported_claims": [
        {"claim_id": "string", "claim_text": "string", "reason": "string"}
    ],
    "incorrect_citations": [
        {"claim_id": "string", "evidence_span_id": "string", "reason": "string"}
    ],
    "required_revisions": ["string"],
    "verifier_notes": "string",
    "verified_summary_json": "object|null (echo of the draft when passed; null when failed)",
}
