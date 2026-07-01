"""Build EvidenceSpan records from `article_text_extracts` rows.

These spans are the ONLY text the summarizer LLM sees. Anything outside the
selected/eligible set is excluded by construction here. No span ever crosses
articles. No span ever carries text from an unselected, excluded, retracted,
or preprint article. No span ever carries text from a different user's
user-uploaded PDF.

Deterministic chunking — no LLM call, no randomness.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("clinevidence.spans")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


MAX_EXTRACT_CHARS_PER_ARTICLE = _env_int("MAX_EXTRACT_CHARS_PER_ARTICLE", 8000)
MAX_CHUNK_CHARS = 2000


# ---- model ------------------------------------------------------------------


class EvidenceSpan(BaseModel):
    """One chunk of one section of one extract belonging to one article.

    `evidence_span_id` is deterministic — re-running on the same packet
    produces the same span ids so retries are stable.
    """

    model_config = ConfigDict(extra="ignore")

    evidence_span_id: str
    article_id: str
    search_session_id: str
    section_name: str
    text_source: str  # abstract | oa_full_text | user_uploaded_full_text
    evidence_text_level: str  # abstract | oa_full_text | user_uploaded_full_text
    start_char: int
    end_char: int
    span_text: str
    citation_label: str
    pmid: Optional[str] = None
    doi: Optional[str] = None
    pmcid: Optional[str] = None
    title: str = ""
    journal: Optional[str] = None
    publication_year: Optional[int | str] = None
    article_type_classification: Optional[str] = None


# ---- helpers ----------------------------------------------------------------


def _normalize_level(extract: dict) -> str:
    """Map a stored extract row to the public evidence_text_level enum.

    `article_text_extracts.evidence_text_level` is permissive (e.g.
    "abstract_only"); the public-facing enum is the short form.
    """
    src = (extract.get("text_source") or "").lower()
    raw = (extract.get("evidence_text_level") or "").lower()
    if src == "user_uploaded_full_text" or raw == "user_uploaded_full_text":
        return "user_uploaded_full_text"
    if src == "oa_full_text" or raw == "oa_full_text":
        return "oa_full_text"
    return "abstract"


def citation_label_for(article: dict) -> str:
    """Deterministic citation label, per Task 3 spec."""
    pmid = (article.get("pmid") or "").strip()
    if pmid:
        return f"PMID:{pmid}"
    doi = (article.get("doi") or "").strip()
    if doi:
        return f"DOI:{doi}"
    pmcid = (article.get("pmcid") or "").strip()
    if pmcid:
        return f"PMCID:{pmcid}"
    return f"article:{article.get('id') or '?'} (metadata-limited)"


_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _chunk_section_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[tuple[int, int, str]]:
    """Return [(start, end, chunk_text), ...] for paragraph-aware chunks.

    Tiny paragraphs (<200 chars) are merged with the previous chunk. Chunks
    exceeding `max_chars` are split at a sentence boundary when possible,
    falling back to a hard cut at the limit.
    """
    if not text:
        return []
    cleaned = text.replace("\r\n", "\n")
    paragraphs: list[tuple[int, int, str]] = []
    cursor = 0
    for para in _PARA_SPLIT.split(cleaned):
        if not para:
            cursor += len(para) + 2  # rough advance for separator
            continue
        # locate in original
        start = cleaned.find(para, cursor)
        if start == -1:
            start = cursor
        end = start + len(para)
        cursor = end
        paragraphs.append((start, end, para))

    # Merge tiny paragraphs.
    merged: list[tuple[int, int, str]] = []
    for s, e, p in paragraphs:
        if merged and len(p) < 200:
            ps, pe, pt = merged[-1]
            merged[-1] = (ps, e, (pt + "\n\n" + p))
        else:
            merged.append((s, e, p))

    # Hard cap each chunk at max_chars.
    capped: list[tuple[int, int, str]] = []
    for s, e, p in merged:
        if len(p) <= max_chars:
            capped.append((s, e, p))
            continue
        # Split at sentence boundaries.
        remaining = p
        local_offset = s
        while len(remaining) > max_chars:
            window = remaining[:max_chars]
            # Try to back off to the last sentence boundary in the window.
            split_matches = list(_SENTENCE_END.finditer(window))
            if split_matches and split_matches[-1].end() >= max_chars // 2:
                cut = split_matches[-1].end()
            else:
                cut = max_chars
            capped.append((local_offset, local_offset + cut, remaining[:cut]))
            local_offset += cut
            remaining = remaining[cut:]
        if remaining:
            capped.append((local_offset, local_offset + len(remaining), remaining))
    return capped


# ---- builder ----------------------------------------------------------------


def build_evidence_spans(
    *,
    session_id: str,
    selected_article_ids: list[str],
    articles_by_id: dict[str, dict],
    extracts_by_article: dict[str, list[dict]],
    max_chars_per_article: int = MAX_EXTRACT_CHARS_PER_ARTICLE,
) -> list[EvidenceSpan]:
    """Construct deterministic EvidenceSpan records for the summarizer.

    Excludes:
      - any article not in selected_article_ids
      - any extract row with include_in_summary != True
      - any text beyond `max_chars_per_article` per article (truncated;
        truncated spans are dropped silently and counted via the caller)

    The selection of "user-uploaded preferred over abstract" is done by the
    caller (evidence_extraction prepare_evidence_packet) — this builder
    accepts whatever rows the caller already filtered to.
    """
    spans: list[EvidenceSpan] = []
    for aid in selected_article_ids:
        art = articles_by_id.get(aid)
        rows = extracts_by_article.get(aid) or []
        if not art or not rows:
            continue
        # Preserve provided order; trust the caller filtered to eligible rows.
        char_budget = max_chars_per_article
        for row in rows:
            if not row.get("include_in_summary"):
                continue
            text = (row.get("section_text") or "").strip()
            if not text:
                continue
            section_name = row.get("section_name") or "section"
            text_source = (row.get("text_source") or "abstract").lower()
            level = _normalize_level(row)
            chunks = _chunk_section_text(text)
            for idx, (s, e, chunk_text) in enumerate(chunks):
                if char_budget <= 0:
                    break
                if len(chunk_text) > char_budget:
                    chunk_text = chunk_text[:char_budget]
                    e = s + len(chunk_text)
                char_budget -= len(chunk_text)
                span_id = f"span:{row.get('id', aid)}:{idx}"
                spans.append(EvidenceSpan(
                    evidence_span_id=span_id,
                    article_id=aid,
                    search_session_id=session_id,
                    section_name=section_name,
                    text_source=text_source,
                    evidence_text_level=level,
                    start_char=s,
                    end_char=e,
                    span_text=chunk_text,
                    citation_label=citation_label_for(art),
                    pmid=(art.get("pmid") or None),
                    doi=(art.get("doi") or None),
                    pmcid=(art.get("pmcid") or None),
                    title=art.get("title") or "",
                    journal=art.get("journal"),
                    publication_year=art.get("publication_year"),
                    article_type_classification=art.get("article_type_classification"),
                ))
            if char_budget <= 0:
                break
    return spans
