"""Deterministic PHI screen for clinician-submitted questions.

NO AI. NO external calls. Pure regex + heuristics so the result is reviewable
and unit-testable.

Detection categories emit SNAKE_CASE reason codes (no free text):
    email_detected, phone_detected, url_detected, ssn_detected,
    mrn_pattern_detected, long_digit_sequence_detected, exact_date_detected,
    street_address_detected, name_title_detected

The sanitized version replaces every detected span with the literal
``[REDACTED]`` — but per Phase 1 policy the *caller* must NOT persist either
the original_question or the sanitized_question when is_blocked=True.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from core.config import settings


# ---- detection patterns ----
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# Phone: US and intl variants.
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?|\d{2,4}[\s.\-])\d{2,4}[\s.\-]?\d{3,4}"
)

_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

_MRN_PREFIX_RE = re.compile(r"\bMRN[:\s#]*\d+\b", re.IGNORECASE)

_DATE_NUMERIC_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_DATE_ISO_RE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")
_MONTH_NAMES = (
    "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    "january|february|march|april|june|july|august|september|october|november|december"
)
_DATE_WORD_RE = re.compile(
    rf"\b(?:{_MONTH_NAMES})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{2,4}})?\b",
    re.IGNORECASE,
)
_DOB_LABEL_RE = re.compile(r"\b(?:DOB|D\.O\.B\.|date\s+of\s+birth)\b[^\n]{0,40}", re.IGNORECASE)

_NAME_RE = re.compile(
    # Phase R1A — Part 2B — patient-name-like PHI policy.
    # Honorific is case-sensitive (rejects lowercase running text);
    # trailing ``.`` is OPTIONAL (real clinical narrative routinely
    # omits it — see soak finding `phi_guard_name_pattern_optional_period`).
    # The trailing comma form (``Mr,`` / ``Mrs,``) is also accepted so
    # patterns like ``Mr, James`` are matched. 1-3 capitalized name
    # tokens to avoid over-matching long titles.
    r"\b(?:Mr|Mrs|Ms|Miss|Mister|Sir|Dr|Mx)[\.,]?\s+"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
)

# Phase R1A — citation-context allowlist. If a name-like match
# falls inside a citation context, we treat it as a literature
# reference rather than PHI. Patterns are deliberately narrow so
# real clinical narrative still trips.
_CITATION_CONTEXT_RES: tuple[re.Pattern[str], ...] = (
    # "Smith et al." / "Smith and Jones et al"
    re.compile(r"\bet\s+al\.?", re.IGNORECASE),
    # "(2024)" / "(2019-2020)" — citation year in parens.
    re.compile(r"\(\s*\d{4}(?:\s*[-–]\s*\d{4})?\s*\)"),
    # "Smith JAMA 2024" / "Smith N Engl J Med 2024" — author + journal + year
    re.compile(
        r"\b(?:JAMA|Lancet|BMJ|N\s+Engl\s+J\s+Med|NEJM|Nat\s+Rev|"
        r"Nature|Science|Cell|Circulation|JACC|Blood|Gastroenterology)\b",
        re.IGNORECASE,
    ),
    # "according to Smith" / "reported by Smith"
    re.compile(
        r"\b(?:according\s+to|reported\s+by|cited\s+by|as\s+per)\s+[A-Z]",
        re.IGNORECASE,
    ),
    # Common scientific-name compound forms (Dr Strangelove etc.) —
    # not a real clinician finding but defensive.
    re.compile(r"\bvon\s+[A-Z]"),
)


def _is_citation_context(text: str, match: re.Match[str]) -> bool:
    """Phase R1A — return True if the matched name span falls within
    a citation context (author + journal, ``et al.``, ``(year)``).
    Looks ±60 chars on each side of the match.

    Pure function. No state.
    """
    start = max(0, match.start() - 60)
    end = min(len(text), match.end() + 60)
    window = text[start:end]
    return any(cr.search(window) for cr in _CITATION_CONTEXT_RES)


# Phase P0 — patient-name-like PHI WITHOUT an honorific. Catches the
# label form ``Patient John Doe has ...`` which the honorific
# ``_NAME_RE`` cannot match (no Mr/Mrs/Dr token). TWO Cap-case name
# tokens are required because single Cap-case tokens after ``Patient``
# are overwhelmingly clinical phrases (``Patient Safety``,
# ``Patient Care``). A closed set of phrase heads that legitimately
# follow ``Patient`` (``Patient Reported Outcomes``,
# ``Patient Health Questionnaire`` …) is excluded so the guard does
# not over-block real clinical questions. Group 1 = first name token.
_PATIENT_LABEL_NAME_RE = re.compile(
    r"\b[Pp]atient\s+([A-Z][a-z]+)\s+[A-Z][a-z]+\b"
)

# Explicit naming markers — ``patient named John``, ``patient's name
# is John Doe``, ``patient called Jane``. Unambiguous personal-name
# signal; 1-3 Cap-case tokens captured.
_PATIENT_NAMED_RE = re.compile(
    r"\b[Pp]atient(?:'s)?\s+(?:named|name\s+is|called)\s+"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
)

# Closed-set clinical phrase heads that legitimately follow ``Patient``.
# When the FIRST token after ``Patient`` is one of these, the bigram is
# a clinical phrase, not a personal name, and is NOT flagged.
_PATIENT_PHRASE_HEADS = frozenset({
    "Reported", "Health", "Global", "Centered", "Centred", "Safety",
    "Education", "Generated", "Care", "Derived", "Specific", "Level",
    "Population", "Preference", "Preferences", "Satisfaction",
    "Experience", "Portal", "Navigator", "Advocate", "Selection",
    "Demographics", "Characteristics", "Outcome", "Outcomes",
    "Engagement", "Activation", "Flow", "Monitoring", "Record",
    "Records", "Data", "Consent", "Age", "Sex", "Gender", "Group",
    "Groups", "Cohort", "Cohorts", "Factors", "Related", "Provider",
    "Physician", "Facing", "Centric", "Acuity", "Adherence",
    "Compliance", "Empowerment", "Eligibility", "Enrollment", "Risk",
    "Stratification", "Disposition", "Identification", "Identifier",
    "Identifiers", "Volume", "Days", "Throughput", "Mix",
})


_STREET_RE = re.compile(
    r"\b\d+\s+[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s+"
    r"(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Lane|Ln|Dr|Drive|Ct|Court|Way|Pl|Place)\b",
    re.IGNORECASE,
)

# Long bare digit runs (excluding obvious years 1900-2099 when exactly 4 digits)
_LONG_DIGITS_RE_TMPL = r"\b\d{{{n},}}\b"


# Fixed snake_case reason codes — the only values that should ever appear in
# the API response or audit payload. Order matters for stable output.
REASON_CODES = (
    "email_detected",
    "phone_detected",
    "url_detected",
    "ssn_detected",
    "mrn_pattern_detected",
    "long_digit_sequence_detected",
    "exact_date_detected",
    "street_address_detected",
    "name_title_detected",
)


# Fixed user-facing message — never echoes the user's input.
USER_PHI_MESSAGE = (
    "Your question appears to include patient identifiers (e.g., dates of birth, "
    "contact info, or ID numbers). Please remove these and resubmit. This tool "
    "must not receive PHI."
)


# Phase 6D — closed-set context types for ``screen_user_text``.
LIBRARY_CONTEXT_TYPES = frozenset({
    "folder_name",
    "folder_description",
    "library_item_note",
    "folder_item_note",
    "evidence_packet_title",
    "evidence_packet_note",
    "summary_title",
    "summary_note",
    "library_tag",
    # Research collections (B12/B13).
    "collection_name",
    "collection_description",
    "collection_item_title",
    "collection_item_note",
})


# Closed-set user-facing message for library text rejections.
LIBRARY_PHI_MESSAGE = (
    "This library text appears to contain private or identifying information. "
    "Please remove identifiers and try again."
)


@dataclass
class PHIScreenResult:
    is_blocked: bool
    reason_codes: list[str]
    sanitized_question: str

    def to_dict(self) -> dict:
        return {
            "is_blocked": self.is_blocked,
            "reason_codes": self.reason_codes,
            "sanitized_question": self.sanitized_question,
        }


def _redact(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    spans = sorted(spans)
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out: list[str] = []
    cursor = 0
    for s, e in merged:
        out.append(text[cursor:s])
        out.append("[REDACTED]")
        cursor = e
    out.append(text[cursor:])
    return "".join(out)


def _is_year(token: str) -> bool:
    if len(token) != 4 or not token.isdigit():
        return False
    return 1900 <= int(token) <= 2099


class PHIGuardService:
    """Stateless, deterministic PHI screen."""

    def __init__(self, min_digit_run: int | None = None):
        self.min_digit_run = min_digit_run or settings.PHI_MIN_DIGIT_RUN
        self._long_digits_re = re.compile(_LONG_DIGITS_RE_TMPL.format(n=self.min_digit_run))

    def screen(self, question: str) -> PHIScreenResult:
        text = "" if question is None else str(question)
        codes: set[str] = set()
        spans: list[tuple[int, int]] = []

        def _scan(pattern: re.Pattern[str], code: str) -> None:
            for m in pattern.finditer(text):
                spans.append((m.start(), m.end()))
                codes.add(code)

        _scan(_EMAIL_RE, "email_detected")
        _scan(_URL_RE, "url_detected")
        _scan(_SSN_RE, "ssn_detected")
        _scan(_MRN_PREFIX_RE, "mrn_pattern_detected")
        _scan(_DATE_NUMERIC_RE, "exact_date_detected")
        _scan(_DATE_ISO_RE, "exact_date_detected")
        _scan(_DATE_WORD_RE, "exact_date_detected")
        _scan(_DOB_LABEL_RE, "exact_date_detected")
        # Phase R1A — Part 2B — patient-name-like PHI scan with
        # citation-context allowlist. Each name-shaped match is
        # window-checked for a literature-citation context
        # (``et al.``, ``(year)``, journal name, ``according to``);
        # such matches are skipped to avoid over-blocking citation
        # references.
        for m in _NAME_RE.finditer(text):
            if _is_citation_context(text, m):
                continue
            spans.append((m.start(), m.end()))
            codes.add("name_title_detected")
        # Phase P0 — patient-name-like label form (no honorific). The
        # ``Patient <First> <Last>`` form is flagged UNLESS the first
        # token after ``Patient`` is a known clinical phrase head
        # (``Patient Reported Outcomes`` etc.). The explicit
        # ``patient named X`` marker form is always flagged. No citation
        # allowlist applies here: a literature reference never reads
        # ``Patient John Doe``, so suppressing on citation context would
        # only create false negatives.
        for m in _PATIENT_LABEL_NAME_RE.finditer(text):
            if m.group(1) in _PATIENT_PHRASE_HEADS:
                continue
            spans.append((m.start(), m.end()))
            codes.add("name_title_detected")
        for m in _PATIENT_NAMED_RE.finditer(text):
            spans.append((m.start(), m.end()))
            codes.add("name_title_detected")
        _scan(_STREET_RE, "street_address_detected")
        _scan(_PHONE_RE, "phone_detected")

        for m in self._long_digits_re.finditer(text):
            tok = m.group(0)
            if _is_year(tok):
                continue
            if any(s <= m.start() < e for s, e in spans):
                continue
            spans.append((m.start(), m.end()))
            codes.add("long_digit_sequence_detected")

        # Stable, fixed ordering.
        ordered = [c for c in REASON_CODES if c in codes]
        return PHIScreenResult(
            is_blocked=bool(codes),
            reason_codes=ordered,
            sanitized_question=_redact(text, spans),
        )


phi_guard = PHIGuardService()


# ============================================================
# Phase 6D — screen_user_text(text, context_type)
# ============================================================
@dataclass
class UserTextScreenResult:
    is_blocked: bool
    reason_codes: list[str]
    context_type: str

    def to_dict(self) -> dict:
        return {
            "is_blocked": self.is_blocked,
            "reason_codes": self.reason_codes,
            "context_type": self.context_type,
        }


def screen_user_text(text: str, context_type: str) -> UserTextScreenResult:
    """Phase 6D — deterministic PHI screen for library / folder text inputs.

    Reuses the patterns from ``PHIGuardService.screen`` so library, folder,
    saved-packet, and tag inputs apply the SAME closed-set detection. Adds
    a context-specific rule for ``library_tag`` (single token, <30 chars,
    no whitespace beyond surrounding trim) — when the tag rule fails the
    extra code ``invalid_tag_shape`` is added to the reason_codes.

    NEVER echoes the input back in the result. Closed-set reason codes
    only. Caller is expected to use ``LIBRARY_PHI_MESSAGE`` as the
    user-facing string.
    """
    if context_type not in LIBRARY_CONTEXT_TYPES:
        raise ValueError(
            f"invalid_context_type: {context_type!r} not in closed set",
        )
    if text is None:
        return UserTextScreenResult(
            is_blocked=False, reason_codes=[], context_type=context_type,
        )
    s = str(text)
    # Tag shape rule fires BEFORE PHI checks for ``library_tag``.
    tag_violations: list[str] = []
    if context_type == "library_tag":
        bare = s.strip()
        if len(bare) > 30 or " " in bare or "\t" in bare or "\n" in bare:
            tag_violations.append("invalid_tag_shape")
    # Always run the PHI screen.
    phi_result = phi_guard.screen(s)
    codes: list[str] = list(phi_result.reason_codes) + tag_violations
    return UserTextScreenResult(
        is_blocked=bool(codes),
        reason_codes=codes,
        context_type=context_type,
    )
