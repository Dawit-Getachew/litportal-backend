"""Phase K5.1 — PubMed guideline lane (deterministic helpers).

DETERMINISTIC. NO LLM. NO network. NO ranking impact.

Three pure helpers used to *label and surface* the already-retrieved
PubMed-indexed guideline evidence as a distinct lane. This module does NOT
perform retrieval (the existing PubMed/NCBI E-utilities path already emits a
``guideline_focused`` query family) and does NOT change ranking, selection,
summaries, or citation verification.

  1. ``classify_guideline_intent(...)`` — is this question a guideline /
     management / protocol / stabilization / monitoring / prevention /
     screening / diagnostic-workup question? Gates whether the guideline
     lane is *surfaced* for the session. Closed-set output.

  2. ``article_guideline_provenance(article_doc)`` — does this article look
     like a PubMed-indexed guideline / consensus / guideline-like review?
     Returns the closed-set provenance label or ``None``.

  3. ``run_guideline_query_qa(query_string)`` — Northstar §10.4 query QA:
     balanced parentheses, no empty blocks, no dangling boolean operators,
     recognised field tags. Returns a closed-set pass/warn result.

Provenance is explicitly labelled "PubMed-indexed", NOT official-society
connector coverage (those connectors remain deferred to K5.2+).
"""
from __future__ import annotations

import re
from typing import Any, Optional

# --- closed-set constants -------------------------------------------------

# The registry source id that authorises this lane (K5.0).
GUIDELINE_LANE_SOURCE_ID = "pubmed_guidelines"

# Closed-set per-article provenance label.
GUIDELINE_LANE_PROVENANCE = "pubmed_indexed_guideline"

# Clinician-facing lane label + tooltip. MUST make clear this is
# PubMed-indexed evidence, not official-source connector coverage.
GUIDELINE_LANE_LABEL = "PubMed guideline / consensus lane"
GUIDELINE_LANE_TOOLTIP = (
    "PubMed-indexed practice guidelines, consensus statements, and "
    "guideline-like reviews. Not official-society-source connector coverage."
)

# Closed-set CQM intents that activate the guideline lane surfacing.
GUIDELINE_LANE_INTENTS = frozenset({
    "guideline_recommendation",
    "guideline_stabilization",
    "stabilization_emergency_management",
    "monitoring_followup_surveillance",
    "prevention",
    "prevention_screening",
    "screening",
    "diagnostic_workup",
})

# Closed-set keyword signals (fallback when the CQM intent is coarse or
# missing). Each maps a closed-set signal name → regex. Deliberately narrow
# so non-guideline questions (mechanism, incidence, severity markers,
# diagnostic rule-out, generic-only) do NOT trip.
_KEYWORD_SIGNALS: dict[str, re.Pattern[str]] = {
    "guideline_term": re.compile(r"\bguidelines?\b|\bpractice\s+guideline\b", re.I),
    "recommendation_term": re.compile(r"\brecommendations?\b", re.I),
    "consensus_term": re.compile(r"\bconsensus\b", re.I),
    "protocol_term": re.compile(r"\bprotocols?\b|\balgorithms?\b", re.I),
    "management_term": re.compile(r"\bmanage(?:d|ment|s)?\b", re.I),
    "stabilization_term": re.compile(
        r"\bstabili[sz]ation\b|\bunstable\b|\bemergency\s+management\b|"
        r"\bacute\s+management\b", re.I,
    ),
    "evaluation_workup_term": re.compile(
        r"\bwork[\s-]?up\b|\bhow\s+should\b[^.?]*\bbe\s+evaluated\b|"
        r"\bevaluation\s+of\b|\bbe\s+evaluated\b|\bapproach\s+to\b|"
        r"\bbe\s+diagnosed\b|\bdiagnostic\s+approach\b", re.I,
    ),
    "screening_term": re.compile(r"\bscreening\b", re.I),
    "prevention_term": re.compile(r"\bprevention\b|\bprophylaxis\b", re.I),
    "surveillance_monitoring_term": re.compile(
        r"\bsurveillance\b|\bmonitoring\b|\bfollow[\s-]?up\b|"
        r"\bhow\s+often\b|\brepeated?\b|\binterval(?:\s+testing)?\b", re.I,
    ),
}


def classify_guideline_intent(
    *,
    primary_intent: Optional[str] = None,
    secondary_intents: Optional[list] = None,
    framework: Optional[str] = None,
    question: Optional[str] = None,
) -> dict[str, Any]:
    """Return a closed-set result describing whether the guideline lane
    should be SURFACED for this session.

    {
      "applicable": bool,
      "matched_intents": [closed-set intent strings],
      "matched_keyword_signals": [closed-set signal names],
      "reason_code": "intent_match" | "keyword_signal_match" |
                     "intent_and_keyword_match" | "no_guideline_signal",
    }

    Pure / deterministic. NEVER stores raw question text.
    """
    matched_intents: list[str] = []
    pi = str(primary_intent or "").strip().lower()
    if pi in GUIDELINE_LANE_INTENTS:
        matched_intents.append(pi)
    for si in (secondary_intents or []):
        s = str(si or "").strip().lower()
        if s in GUIDELINE_LANE_INTENTS and s not in matched_intents:
            matched_intents.append(s)
    fw = str(framework or "").strip().lower()
    if fw in {"guideline_stabilization", "guideline"} and fw not in matched_intents:
        matched_intents.append(fw)

    matched_signals: list[str] = []
    q = str(question or "")
    if q:
        for name, patt in _KEYWORD_SIGNALS.items():
            if patt.search(q):
                matched_signals.append(name)

    has_intent = bool(matched_intents)
    has_keyword = bool(matched_signals)
    if has_intent and has_keyword:
        reason = "intent_and_keyword_match"
    elif has_intent:
        reason = "intent_match"
    elif has_keyword:
        reason = "keyword_signal_match"
    else:
        reason = "no_guideline_signal"

    return {
        "applicable": bool(has_intent or has_keyword),
        "matched_intents": matched_intents,
        "matched_keyword_signals": matched_signals,
        "reason_code": reason,
    }


_GUIDELINE_PUBTYPE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bpractice\s+guideline\b", re.I),
    re.compile(r"\bguideline\b", re.I),
    re.compile(r"\bconsensus\s+development\s+conference\b", re.I),
    re.compile(r"\bconsensus\s+statement\b", re.I),
)


def article_guideline_provenance(article_doc: dict[str, Any]) -> Optional[str]:
    """Return ``GUIDELINE_LANE_PROVENANCE`` when the article looks like a
    PubMed-indexed guideline / consensus / guideline-like review; else
    ``None``.

    Reuses the existing deterministic signals already stamped upstream
    (``is_guideline`` from the PubMed connector, ``article_type``,
    ``publication_types``). NEVER calls the network. Pure.
    """
    if not isinstance(article_doc, dict):
        return None
    if bool(article_doc.get("is_guideline")):
        return GUIDELINE_LANE_PROVENANCE
    atype = str(article_doc.get("article_type") or "").strip().lower()
    if atype == "guideline":
        return GUIDELINE_LANE_PROVENANCE
    pubs = (
        article_doc.get("publication_types_json")
        or article_doc.get("publication_types")
        or []
    )
    blob = " ".join(str(p) for p in pubs if isinstance(p, str))
    if blob and any(p.search(blob) for p in _GUIDELINE_PUBTYPE_PATTERNS):
        return GUIDELINE_LANE_PROVENANCE
    return None


# --- query QA (Northstar §10.4) ------------------------------------------

# Closed-set of recognised PubMed/EPMC field tags (lower-cased, normalised).
_ALLOWED_FIELD_TAGS = frozenset({
    "tiab", "title/abstract", "title", "ti", "ab", "mesh", "mesh terms",
    "mesh:noexp", "majr", "mh", "pt", "publication type", "tw", "text word",
    "la", "language", "dp", "date - publication", "pdat", "sb", "subheading",
    "sh", "all fields", "nm", "rn", "au", "ta", "journal", "so",
})

_LEADING_BOOL_RE = re.compile(r"^\s*(?:AND|OR|NOT)\b", re.I)
_TRAILING_BOOL_RE = re.compile(r"\b(?:AND|OR|NOT)\s*$", re.I)
_EMPTY_PAREN_RE = re.compile(r"\(\s*\)")
_FIELD_TAG_RE = re.compile(r"\[([^\[\]]*)\]")


def run_guideline_query_qa(query_string: str) -> dict[str, Any]:
    """Deterministic query QA. Returns
    ``{"passed": bool, "warnings": [closed-set codes]}``.

    Structural failures (``passed=False``): empty query, unbalanced
    parentheses, empty ``()`` block, empty ``[]`` tag, leading/trailing
    boolean operator. Unrecognised field tags are an ADVISORY warning only
    (do not fail), so valid-but-unlisted tags are not rejected.
    """
    warnings: list[str] = []
    passed = True
    q = "" if query_string is None else str(query_string)

    if not q.strip():
        return {"passed": False, "warnings": ["empty_query"]}

    # Balanced parentheses.
    depth = 0
    balanced = True
    for ch in q:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                balanced = False
                break
    if not balanced or depth != 0:
        warnings.append("unbalanced_parentheses")
        passed = False

    if _EMPTY_PAREN_RE.search(q):
        warnings.append("empty_paren_block")
        passed = False

    if _LEADING_BOOL_RE.search(q):
        warnings.append("leading_boolean_operator")
        passed = False
    if _TRAILING_BOOL_RE.search(q):
        warnings.append("trailing_boolean_operator")
        passed = False

    for m in _FIELD_TAG_RE.finditer(q):
        tag = m.group(1).strip().lower()
        if not tag:
            warnings.append("empty_field_tag")
            passed = False
        elif tag not in _ALLOWED_FIELD_TAGS:
            warnings.append("unrecognized_field_tag")

    # Stable de-dup preserving first occurrence.
    seen: set[str] = set()
    ordered = [w for w in warnings if not (w in seen or seen.add(w))]
    return {"passed": bool(passed), "warnings": ordered}


__all__ = [
    "GUIDELINE_LANE_SOURCE_ID",
    "GUIDELINE_LANE_PROVENANCE",
    "GUIDELINE_LANE_LABEL",
    "GUIDELINE_LANE_TOOLTIP",
    "GUIDELINE_LANE_INTENTS",
    "classify_guideline_intent",
    "article_guideline_provenance",
    "run_guideline_query_qa",
]
