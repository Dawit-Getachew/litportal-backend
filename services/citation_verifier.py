"""Phase 3 CitationVerifierService — see ``services.summarization.verifier``.

Kept for back-compat with earlier import sites.
"""
from __future__ import annotations

from services.summarization.verifier import (
    CitationVerifierService,
    citation_verifier,
)

__all__ = ["CitationVerifierService", "citation_verifier"]
