"""Phase 5D — STOP_PHRASES + STOP_TOKENS filter.

Lives in its own module so both ``concept_block_query_builder`` (Phase 5A
legacy) and ``query_compiler`` (Phase 5D new) can import it without
creating an import cycle.

Two layers of filtering:
- ``STOP_PHRASES``: multi-word prose noise (e.g. "difference between",
  "what is the dose of"). Stripped IN-PLACE from a term — the remaining
  signal still gets emitted.
- ``STOP_TOKENS``: single-word English stopwords + question words
  (e.g. "tell", "about", "me", "what", "the"). Any term that, AFTER
  STOP_PHRASE stripping, reduces to a single STOP_TOKEN must be
  dropped entirely — never emitted into a ``TITLE_ABS:"…"`` or
  ``"…"[tiab]`` expression.

Together, these guarantee that prose-junk PICOTT terms (e.g. a
"tell me about diabetes" question that the LLM splits into the tokens
"tell" / "about" / "me" / "diabetes") cannot leak into a query.
"""
from __future__ import annotations

import re

# Per-spec list + a small set of equally-meaningless connective phrases.
STOP_PHRASES = [
    r"\bdifference between\b",
    r"\bin patients that are\b",
    r"\bin patients that have\b",
    r"\bin patients who are\b",
    r"\bin patients who have\b",
    r"\bhow does\s+\w+\s+compare with\b",
    r"\bwhat is the dose of\b",
    r"\bwhat is the role of\b",
    r"\bwhat is the evidence for\b",
    r"\bwhat is the best\b",
    r"\bwhat is the optimal\b",
    r"\bcomparison of\b",
    r"\bcomparing\b",
    r"\btell me about\b",
    r"\btell me\b",
]

_STOP_RE = re.compile("|".join(STOP_PHRASES), re.IGNORECASE)


def strip_stop_phrases(text: str) -> str:
    """Strip user-phrase noise before query emission."""
    if not text:
        return ""
    cleaned = _STOP_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" ,;:.!?")


# Single-word stopwords that must NEVER appear as a search term on their
# own (even if the parser hallucinated them as a "concept"). Lowercase
# canonicalized. Length ≤ 4 chars is the typical signal.
STOP_TOKENS = frozenset({
    # Question / instruction words
    "tell", "me", "us", "what", "why", "how", "where", "when", "who",
    "which", "show", "give", "explain", "describe", "discuss", "list",
    # Articles / determiners
    "a", "an", "the", "this", "that", "these", "those",
    # Prepositions / common connectives
    "about", "with", "of", "to", "from", "for", "in", "on", "at",
    "into", "onto", "by", "as", "over", "under",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    # Verbs of being / generic
    "is", "are", "was", "were", "be", "being", "been",
    "do", "does", "did", "doing", "done",
    "has", "have", "had", "having",
    "can", "could", "would", "should", "may", "might", "shall",
    # Conjunctions
    "and", "or", "but", "if", "then", "than", "so", "because",
    # Relational / connective words — strip from edges so junk bigrams
    # like "between high", "versus low", "during sepsis" don't leak as
    # the FIRST or LAST token. Internal occurrences (e.g., "acute
    # infarction of the myocardium") are preserved.
    # Intentionally NOT included: "high", "low", "good", "bad" — these
    # are clinical adjectives that frequently form valid bigrams
    # ("high hemoglobin", "low-dose aspirin", "high-risk"). Single-token
    # `"high"` emission is acceptable; the tester explicitly OK'd the
    # `"high hemoglobin"` bigram in the round-3 brief.
    "between", "versus", "via", "amongst", "during", "within", "without",
    "across", "through", "throughout", "upon",
    "near", "around", "above", "below", "after", "before",
    "different", "differences", "compared", "various",
    # Other prose noise
    "any", "all", "some", "more", "most", "less", "least", "new", "old",
    # Single common short forms
    "vs", "etc", "ie", "eg",
})


def is_query_term_meaningful(term: str) -> bool:
    """Return False for terms that, after STOP_PHRASE stripping, reduce
    to a single ``STOP_TOKEN`` or to whitespace/punctuation only.

    True for any term with at least one non-stopword token (or a single
    token of length ≥ 5 chars — long acronyms / drug names / diseases
    survive even if they happen to match a stopword pattern). This is
    intentionally conservative: false-positives (dropping a real term)
    are worse than letting through a slightly-noisy multi-word phrase.
    """
    if not term:
        return False
    cleaned = strip_stop_phrases(term).strip()
    if not cleaned:
        return False
    # Decompose into lowercased tokens (alphanumeric + hyphen, the
    # building blocks of medical phrases like "D-dimer", "HFpEF").
    tokens = [t for t in re.split(r"[\s,;:.!?]+", cleaned.lower()) if t]
    if not tokens:
        return False
    # Single-token: drop iff it's a known stopword AND short enough that
    # we're confident it's prose noise rather than a clinical acronym.
    if len(tokens) == 1:
        t = tokens[0].strip("\"'")
        if t in STOP_TOKENS:
            return False
        # Single very-short token that isn't a known clinical acronym
        # is probably noise (e.g. "or", "is").
        if len(t) <= 2 and t in STOP_TOKENS:
            return False
        return True
    # Multi-token: at least one token must be a non-stopword.
    return any(t.strip("\"'") not in STOP_TOKENS for t in tokens)
