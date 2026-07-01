"""Phase 3 SummarizerService — see ``services.summarization.summarizer``.

Kept for back-compat with earlier import sites that did
``from services.summarizer import summarizer``.
"""
from __future__ import annotations

from services.summarization.summarizer import SummarizerService, summarizer

__all__ = ["SummarizerService", "summarizer"]
