"""ClinEvidence Search — Phase 3 grounded summarizer + citation verifier.

ABSOLUTE SAFETY INVARIANTS (enforced by code, not by the LLM):

1. The summarizer LLM sees ONLY the structured spans built from
   ``article_text_extracts`` rows where ``include_in_summary=true`` AND whose
   article is in the SELECTED set for this session. No other text — no
   institutional link-out text, no PDF bytes, no unselected article
   abstracts, no excluded/retracted/preprint articles.

2. Every clinically meaningful claim must cite at least one
   ``evidence_span_id`` belonging to a selected article. Mechanical floor
   (verifier.py) rejects fabricated PMIDs/DOIs/PMCIDs, unselected article
   citations, missing span citations, and overstated certainty (e.g.
   abstract-only + certainty=high).

3. The verifier is an INDEPENDENT LLM call with a different system prompt.
   If the verifier fails twice, the draft is NEVER displayed to the user.

4. PHI guard runs FIRST at session creation. The summarizer pipeline is
   never reached for blocked_phi sessions.
"""
from .schemas import (
    DraftSummary,
    ClaimMapEntry,
    SelectedStudyEntry,
    VerifierResult,
    UnsupportedClaim,
    IncorrectCitation,
)
from .evidence_spans import EvidenceSpan, build_evidence_spans
from .summarizer import SummarizerService, summarizer
from .verifier import CitationVerifierService, citation_verifier
from .prompts import PROMPT_VERSION

__all__ = [
    "DraftSummary",
    "ClaimMapEntry",
    "SelectedStudyEntry",
    "VerifierResult",
    "UnsupportedClaim",
    "IncorrectCitation",
    "EvidenceSpan",
    "build_evidence_spans",
    "SummarizerService",
    "summarizer",
    "CitationVerifierService",
    "citation_verifier",
    "PROMPT_VERSION",
]
