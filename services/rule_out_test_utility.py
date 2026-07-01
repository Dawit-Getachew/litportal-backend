"""Phase K4A.3 — Diagnostic Rule-Out / Test Utility deterministic
detection + ranking helpers.

Scope (closed-set, deterministic, NO LLM):

  • Recognize the question shape ``"Does a negative [test] rule out
    [condition]?"`` and equivalents (NPV, sensitivity/specificity,
    rule-out / exclude, predictive value, diagnostic utility,
    de-escalation).

  • Extract index_test + target_condition anchors deterministically.

  • Closed-set acronym expansion for MRSA.

  • Closed-set 10-label directness classifier the picott_ranker uses
    to boost direct diagnostic-meta-analysis / SR / primary
    diagnostic-accuracy articles over treatment-only or
    colonization-only articles, scoped to THIS question family only.

Cross-phase interaction notes:
  • K4A.2.1 generic-anchor denylist is concept-level exact-match —
    multi-word disease names like ``MRSA pneumonia`` are NOT in the
    denylist. We rely on that contract here.
  • K4A.2 severity-marker overview behavior is preserved — this
    module only activates when the question matches the rule-out /
    test-utility patterns.
"""
from __future__ import annotations

import re
from typing import Literal


# =====================================================================
# Closed-set MRSA acronym + synonym expansion.
#
# Used by:
#   - the question detector (so "MRSA" tokens are recognized)
#   - the OR-block builder (so emitted Boolean queries surface every
#     accepted variant alongside the canonical expansion)
# =====================================================================
MRSA_SYNONYMS: tuple[str, ...] = (
    "MRSA",
    "methicillin-resistant Staphylococcus aureus",
    "methicillin resistant Staphylococcus aureus",
    "methicillin-resistant S. aureus",
    "methicillin resistant S aureus",
    "methicillin-resistant Staph aureus",
    "methicillin resistant Staph aureus",
)

NASAL_SCREENING_SYNONYMS: tuple[str, ...] = (
    "MRSA nasal screening",
    "MRSA nares screening",
    "MRSA nasal screen",
    "MRSA nares screen",
    "MRSA nasal swab",
    "MRSA nares swab",
    "MRSA nasal PCR",
    "MRSA nares PCR",
    "MRSA PCR nasal swab",
    "MRSA surveillance swab",
    "nasal MRSA PCR",
    "nares MRSA PCR",
    "nasal culture for MRSA",
    "nares culture for MRSA",
)

# Multi-word target-condition synonyms — explicitly NOT generic.
MRSA_PNEUMONIA_SYNONYMS: tuple[str, ...] = (
    "MRSA pneumonia",
    "methicillin-resistant Staphylococcus aureus pneumonia",
    "staphylococcal pneumonia",
    "MRSA lower respiratory tract infection",
    "MRSA respiratory infection",
)

# Closed-set "rule out / NPV / diagnostic accuracy" outcome cues.
DIAGNOSTIC_RULE_OUT_OUTCOMES: tuple[str, ...] = (
    "rule out",
    "rule-out",
    "rules out",
    "ruling out",
    "exclude",
    "negative predictive value",
    "NPV",
    "predictive value",
    "negative test",
    "negative swab",
    "negative PCR",
    "sensitivity",
    "specificity",
    "likelihood ratio",
    "diagnostic accuracy",
    "diagnostic utility",
    "test utility",
    "test accuracy",
    "diagnostic meta-analysis",
)


# =====================================================================
# Question detection — closed-set regex patterns.
# =====================================================================
# A "test-like" phrase recognizer used in patterns. Closed-set so we
# don't accidentally match unrelated wording.
_TEST_LIKE_RE = (
    r"(?:[a-z][a-z0-9 \-]*?\s+)?"
    r"(?:nasal\s+(?:swab|pcr|screen(?:ing)?|culture)|"
    r"nares\s+(?:swab|pcr|screen(?:ing)?|culture)|"
    r"nasal\s+mrsa\s+(?:pcr|swab|screen(?:ing)?)|"
    r"nares\s+mrsa\s+(?:pcr|swab|screen(?:ing)?)|"
    r"surveillance\s+swab|"
    r"mrsa\s+(?:nasal|nares)\s+(?:swab|pcr|screen(?:ing)?|culture)|"
    r"mrsa\s+pcr|mrsa\s+screen(?:ing)?|"
    r"d-?dimer|procalcitonin|troponin|brain\s+natriuretic\s+peptide|bnp|"
    r"rapid\s+(?:antigen|strep|flu)\s+test)"
)

# A "condition-like" phrase recognizer. Disease names with MRSA
# prefix are explicitly accepted; bare generic words (pneumonia,
# infection) accepted only when paired with a real organism qualifier
# or the rule-out shape's full grammar.
_CONDITION_LIKE_RE = (
    r"(?P<condition>"
    r"mrsa\s+(?:pneumonia|lower\s+respiratory\s+tract\s+infection|"
    r"respiratory\s+infection|bacteremia|bloodstream\s+infection|"
    r"infection)|"
    r"methicillin[\-\s]resistant\s+(?:staphylococcus\s+aureus|s\.?\s*aureus|"
    r"staph\s+aureus)\s+(?:pneumonia|infection)|"
    r"staphylococcal\s+pneumonia|"
    r"pneumonia|"
    r"infection|"
    r"sepsis|"
    r"bacteremia|"
    r"bloodstream\s+infection|"
    r"pulmonary\s+embolism|pe|"
    r"acute\s+myocardial\s+infarction|ami|"
    r"deep\s+vein\s+thrombosis|dvt)"
)

_RULE_OUT_QUESTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 1. "Does/Can a negative [test] rule out / exclude [condition]?"
    re.compile(
        r"\b(?:does|can|is|will)\s+(?:a\s+|an\s+)?negative\s+"
        r"(?P<test1>" + _TEST_LIKE_RE + r")\s+"
        r"(?:rule\s*out|rules\s+out|exclude|reliably\s+rule\s*out)\s+"
        + _CONDITION_LIKE_RE,
        re.IGNORECASE,
    ),
    # 2. "Does a negative [test] EXCLUDE [condition]?"  (already covered
    #    by pattern 1 — kept as an explicit variant for completeness.)
    # 3. "What is the negative predictive value of [test] for [condition]?"
    re.compile(
        r"\bwhat\s+is\s+(?:the\s+)?(?:negative\s+predictive\s+value|npv|"
        r"sensitivity|specificity|diagnostic\s+accuracy|likelihood\s+ratio|"
        r"predictive\s+value|diagnostic\s+utility|test\s+utility|"
        r"test\s+accuracy)"
        r"\s+(?:of|for)\s+(?P<test3>" + _TEST_LIKE_RE + r")\s+(?:for|to\s+"
        r"(?:rule\s*out|exclude|diagnose|detect))\s+"
        + _CONDITION_LIKE_RE,
        re.IGNORECASE,
    ),
    # 4. "Is [test] useful (to|for) rule(ing) out [condition]?"
    re.compile(
        r"\b(?:is|are)\s+(?P<test4>" + _TEST_LIKE_RE + r")\s+"
        r"(?:useful|helpful|reliable|accurate|sensitive|specific)\s+"
        r"(?:to|for)\s+(?:rule\s*out|rules\s+out|ruling\s+out|exclude|"
        r"excluding|diagnose|detect|diagnosing|detecting)\s+"
        + _CONDITION_LIKE_RE,
        re.IGNORECASE,
    ),
    # 5. "How accurate/sensitive/specific is [test] for [condition]?"
    re.compile(
        r"\bhow\s+(?:accurate|sensitive|specific|useful|reliable)\s+is\s+"
        r"(?P<test5>" + _TEST_LIKE_RE + r")\s+for\s+(?:diagnosing|"
        r"detecting|ruling\s+out|excluding)?\s*"
        + _CONDITION_LIKE_RE,
        re.IGNORECASE,
    ),
    # 6. "Can [test] guide de-escalation of [therapy] in [condition]?"
    #    Stewardship phrasing — still routes to test-utility family.
    re.compile(
        r"\b(?:can|does|do)\s+(?P<test6>" + _TEST_LIKE_RE + r")\s+"
        r"(?:guide|inform|support|drive|enable)\s+"
        r"(?:de[\-\s]?escalation|discontinuation|withdrawal|stopping|"
        r"stewardship)\s+"
        r"(?:of\s+[a-z][a-z0-9 \-]*?\s+)?(?:in|for)\s+"
        + _CONDITION_LIKE_RE,
        re.IGNORECASE,
    ),
)


def is_rule_out_test_utility_question(text: str) -> bool:
    """Closed-set deterministic detection of the diagnostic
    rule-out / test-utility question family.

    Returns ``True`` when the text matches one of 6 closed-set
    shape patterns. NO LLM, NO network.
    """
    if not text:
        return False
    norm = text.strip().lower()
    for pat in _RULE_OUT_QUESTION_PATTERNS:
        if pat.search(norm):
            return True
    return False


def extract_rule_out_components(text: str) -> dict[str, str | None]:
    """Extract closed-set index_test + target_condition anchors from
    the rule-out / test-utility question.

    Returns a dict with keys:
      - ``index_test``        canonical index-test phrase, or None
      - ``target_condition``  canonical target-condition phrase, or None
      - ``mrsa_route``        ``True`` if either anchor mentions MRSA
                              (closed-set acronym/synonym match)
      - ``matched_pattern``   index of the matched closed-set pattern
                              (0..5), or ``-1`` if no match
    """
    out: dict[str, str | None] = {
        "index_test": None,
        "target_condition": None,
        "mrsa_route": False,
        "matched_pattern": -1,
    }
    if not text:
        return out
    norm = text.strip().lower()
    for idx, pat in enumerate(_RULE_OUT_QUESTION_PATTERNS):
        m = pat.search(norm)
        if not m:
            continue
        out["matched_pattern"] = idx
        for name in ("test1", "test3", "test4", "test5", "test6"):
            try:
                val = m.group(name)
            except Exception:  # noqa: BLE001
                val = None
            if val:
                out["index_test"] = _canonicalize_index_test(val)
                break
        try:
            cond = m.group("condition")
            if cond:
                out["target_condition"] = _canonicalize_target_condition(cond)
        except Exception:  # noqa: BLE001
            pass
        break
    # MRSA route flag — closed-set acronym/synonym test.
    blob = (
        (out.get("index_test") or "")
        + " "
        + (out.get("target_condition") or "")
    ).lower()
    out["mrsa_route"] = (
        "mrsa" in blob
        or "methicillin-resistant" in blob
        or "methicillin resistant" in blob
        or "staphylococcal" in blob
    )
    return out


def _canonicalize_index_test(raw: str) -> str:
    """Closed-set canonicalization of an index-test phrase."""
    r = (raw or "").strip().lower()
    if not r:
        return ""
    # MRSA nasal family → canonicalize to "MRSA nasal screening" for
    # the picott must_have. Surface the user-friendly subtype as
    # synonyms during query construction.
    if (
        ("mrsa" in r and ("nasal" in r or "nares" in r))
        or ("nasal" in r and "mrsa" in r)
        or ("nares" in r and "mrsa" in r)
    ):
        return "MRSA nasal screening"
    if "surveillance swab" in r:
        return "MRSA nasal screening"
    # Fall back: preserve casing-aware version of the raw match.
    return raw.strip()


def _canonicalize_target_condition(raw: str) -> str:
    """Closed-set canonicalization of a target-condition phrase."""
    r = (raw or "").strip().lower()
    if not r:
        return ""
    if "mrsa pneumonia" in r or (
        "methicillin" in r and "pneumonia" in r
    ) or "staphylococcal pneumonia" in r:
        return "MRSA pneumonia"
    if "mrsa lower respiratory" in r or "mrsa respiratory" in r:
        return "MRSA pneumonia"
    # Bare "pneumonia" — caller will decide if this is enough for an
    # anchor (the rule-out grammar requires it appear inside the
    # closed-set shape above; without the MRSA qualifier the picott
    # condition is the bare "pneumonia", which the K4A.2.1 generic
    # denylist would block. The pipeline therefore relies on the
    # rule_out_components.target_condition + the MRSA orblock synonyms
    # to produce a usable anchor.)
    return raw.strip()


# =====================================================================
# Closed-set OR-blocks for the Boolean query compiler.
# =====================================================================
def mrsa_or_block_synonyms() -> list[str]:
    """Closed-set MRSA acronym + synonym OR-block list."""
    seen: set[str] = set()
    out: list[str] = []
    for s in MRSA_SYNONYMS:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def nasal_screening_or_block_synonyms() -> list[str]:
    """Closed-set MRSA nasal screening OR-block list."""
    seen: set[str] = set()
    out: list[str] = []
    for s in NASAL_SCREENING_SYNONYMS:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def mrsa_pneumonia_or_block_synonyms() -> list[str]:
    """Closed-set MRSA pneumonia OR-block list."""
    seen: set[str] = set()
    out: list[str] = []
    for s in MRSA_PNEUMONIA_SYNONYMS:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def diagnostic_rule_out_or_block_synonyms() -> list[str]:
    """Closed-set diagnostic-accuracy / rule-out OR-block list."""
    return [
        "negative predictive value",
        "NPV",
        "rule out",
        "rule-out",
        "predictive value",
        "sensitivity",
        "specificity",
        "likelihood ratio",
        "diagnostic accuracy",
        "diagnostic utility",
        "test utility",
        "test accuracy",
        "diagnostic meta-analysis",
    ]


# =====================================================================
# Closed-set 10-label directness classifier.
# =====================================================================
DiagnosticDirectnessLabel = Literal[
    "diagnostic_meta_analysis_rule_out_mrsa_pneumonia",
    "systematic_review_mrsa_nasal_screening_pneumonia",
    "diagnostic_accuracy_primary_mrsa_nasal_screening",
    "stewardship_deescalation_mrsa_nasal_screening",
    "timing_of_negative_mrsa_screen",
    "wrong_target_condition",
    "wrong_index_test",
    "colonization_only_not_pneumonia",
    "general_mrsa_colonization_not_pneumonia",
    "treatment_only_without_test_accuracy",
]

DIAGNOSTIC_DIRECTNESS_RANKED: tuple[DiagnosticDirectnessLabel, ...] = (
    "diagnostic_meta_analysis_rule_out_mrsa_pneumonia",
    "systematic_review_mrsa_nasal_screening_pneumonia",
    "diagnostic_accuracy_primary_mrsa_nasal_screening",
    "stewardship_deescalation_mrsa_nasal_screening",
    "timing_of_negative_mrsa_screen",
    "general_mrsa_colonization_not_pneumonia",
    "colonization_only_not_pneumonia",
    "treatment_only_without_test_accuracy",
    "wrong_target_condition",
    "wrong_index_test",
)

DIAGNOSTIC_DIRECTNESS_BOOSTS: dict[DiagnosticDirectnessLabel, int] = {
    "diagnostic_meta_analysis_rule_out_mrsa_pneumonia": 14,
    "systematic_review_mrsa_nasal_screening_pneumonia": 12,
    "diagnostic_accuracy_primary_mrsa_nasal_screening": 9,
    "stewardship_deescalation_mrsa_nasal_screening": 6,
    "timing_of_negative_mrsa_screen": 4,
    "general_mrsa_colonization_not_pneumonia": 0,
    "colonization_only_not_pneumonia": 0,
    "treatment_only_without_test_accuracy": 0,
    "wrong_target_condition": 0,
    "wrong_index_test": 0,
}

DIAGNOSTIC_DIRECTNESS_CAPS: dict[DiagnosticDirectnessLabel, int | None] = {
    "diagnostic_meta_analysis_rule_out_mrsa_pneumonia": None,
    "systematic_review_mrsa_nasal_screening_pneumonia": None,
    "diagnostic_accuracy_primary_mrsa_nasal_screening": None,
    "stewardship_deescalation_mrsa_nasal_screening": None,
    "timing_of_negative_mrsa_screen": None,
    "general_mrsa_colonization_not_pneumonia": 30,
    "colonization_only_not_pneumonia": 30,
    "treatment_only_without_test_accuracy": 30,
    "wrong_target_condition": 20,
    "wrong_index_test": 20,
}

DIAGNOSTIC_DIRECTNESS_CAP_REASONS: dict[DiagnosticDirectnessLabel, str | None] = {
    "diagnostic_meta_analysis_rule_out_mrsa_pneumonia": None,
    "systematic_review_mrsa_nasal_screening_pneumonia": None,
    "diagnostic_accuracy_primary_mrsa_nasal_screening": None,
    "stewardship_deescalation_mrsa_nasal_screening": None,
    "timing_of_negative_mrsa_screen": None,
    "general_mrsa_colonization_not_pneumonia": "lane_mismatch_colonization_only",
    "colonization_only_not_pneumonia": "lane_mismatch_colonization_only",
    "treatment_only_without_test_accuracy": "lane_mismatch_treatment_only",
    "wrong_target_condition": "wrong_target_condition",
    "wrong_index_test": "wrong_index_test",
}


_MRSA_BLOB_RE = re.compile(
    r"\b(?:mrsa|methicillin[\-\s]?resistant|staphylococcal|staph\s+aureus)\b",
    re.IGNORECASE,
)
_NASAL_BLOB_RE = re.compile(
    r"\b(?:nasal|nares|nostril)\b",
    re.IGNORECASE,
)
_PNEUMONIA_BLOB_RE = re.compile(
    r"\b(?:pneumonia|lower\s+respiratory|hap|vap)\b",
    re.IGNORECASE,
)
_RULE_OUT_BLOB_RE = re.compile(
    r"\b(?:rule\s*out|rules\s+out|ruling\s+out|exclude|excluding|"
    r"negative\s+predictive\s+value|\bnpv\b|predictive\s+value|"
    r"sensitivity|specificity|likelihood\s+ratio|diagnostic\s+accuracy|"
    r"diagnostic\s+utility|test\s+utility|test\s+accuracy|"
    r"de[\-\s]?escalat|stewardship)\b",
    re.IGNORECASE,
)
_DIAGNOSTIC_META_RE = re.compile(
    r"\b(?:diagnostic\s+meta[\-\s]?analysis|meta[\-\s]?analysis\s+of\s+diagnostic|"
    r"systematic\s+review\s+and\s+meta[\-\s]?analysis)\b",
    re.IGNORECASE,
)
_SR_RE = re.compile(
    r"\b(?:systematic\s+review|umbrella\s+review|scoping\s+review)\b",
    re.IGNORECASE,
)
_COLONIZATION_ONLY_RE = re.compile(
    r"\bcolonization\b|\bcolonisation\b|\bcarrier\s+screening\b",
    re.IGNORECASE,
)
_TREATMENT_ONLY_RE = re.compile(
    r"\b(?:treatment|therapy|vancomycin|linezolid|ceftaroline|daptomycin|"
    r"tedizolid|clindamycin)\b",
    re.IGNORECASE,
)
_TIMING_RE = re.compile(
    r"\b(?:duration\s+of\s+validity|how\s+long|time[\-\s]?to|temporal\s+validity|"
    r"timing\s+of\s+swab|swab\s+timing)\b",
    re.IGNORECASE,
)


def classify_diagnostic_directness(
    article: dict,
    *,
    index_test_anchor: str = "MRSA nasal screening",
    target_condition_anchor: str = "MRSA pneumonia",
) -> DiagnosticDirectnessLabel:
    """Closed-set deterministic directness classifier for the
    diagnostic rule-out / test-utility question family.

    Mirrors the K4A.2 ``classify_severity_marker_directness`` pattern:
    inputs are an article dict with closed-set keys (``title``,
    ``abstract``, ``publication_types_json``) and the rule-out
    family's index_test + target_condition anchors. Output is a
    closed-set ``DiagnosticDirectnessLabel``.
    """
    title = str((article or {}).get("title") or "")
    abstract = str((article or {}).get("abstract") or "")
    pub_types = [
        str(p).lower() for p in ((article or {}).get("publication_types_json") or [])
    ]
    is_guideline = bool((article or {}).get("is_guideline"))
    blob = f"{title}\n{abstract}".lower()

    has_mrsa = bool(_MRSA_BLOB_RE.search(blob))
    has_nasal = bool(_NASAL_BLOB_RE.search(blob))
    has_pneumonia = bool(_PNEUMONIA_BLOB_RE.search(blob))
    has_rule_out = bool(_RULE_OUT_BLOB_RE.search(blob))
    is_diag_meta = bool(_DIAGNOSTIC_META_RE.search(blob)) or any(
        "meta-analysis" in p or "meta analysis" in p for p in pub_types
    )
    is_sr = bool(_SR_RE.search(blob)) or any(
        "systematic review" in p for p in pub_types
    )
    is_review_shape = (
        is_sr or is_diag_meta
        or any(p.strip() == "review" or p.strip().startswith("review ")
               for p in pub_types)
    )
    is_colonization_only = bool(_COLONIZATION_ONLY_RE.search(blob)) and not has_pneumonia
    is_treatment_only = (
        bool(_TREATMENT_ONLY_RE.search(blob))
        and not has_rule_out
        and not has_nasal
    )
    is_timing = bool(_TIMING_RE.search(blob)) and has_mrsa and has_nasal

    # 1. Wrong target condition — if the article is about a non-MRSA
    # disease entirely (and the rule-out family is MRSA-route), cap.
    if not has_mrsa and target_condition_anchor.lower().startswith("mrsa"):
        return "wrong_target_condition"

    # 3. Diagnostic meta-analysis directly about MRSA nasal rule-out
    # for pneumonia — top of the lane.
    if is_diag_meta and has_mrsa and has_nasal and has_pneumonia and has_rule_out:
        return "diagnostic_meta_analysis_rule_out_mrsa_pneumonia"

    # 4. Systematic review of MRSA nasal screening utility for
    # pneumonia.
    if is_sr and has_mrsa and has_nasal and has_pneumonia:
        return "systematic_review_mrsa_nasal_screening_pneumonia"

    # 5. Stewardship / de-escalation studies — MRSA nasal screen
    # guides vancomycin discontinuation in pneumonia. Checked BEFORE
    # the generic primary-accuracy bucket because stewardship articles
    # also satisfy the (MRSA + nasal + pneumonia + rule-out) shape but
    # belong in a more specific lane.
    if has_mrsa and has_nasal and has_pneumonia and (
        "de-escalat" in blob or "de escalat" in blob
        or "discontinu" in blob or "stewardship" in blob
        or "duration of therapy" in blob
        or "duration of mrsa-targeted therapy" in blob
        or "duration of mrsa targeted therapy" in blob
        or "anti-mrsa" in blob or "anti mrsa" in blob
    ):
        return "stewardship_deescalation_mrsa_nasal_screening"

    # 6. Primary diagnostic accuracy study of MRSA nasal PCR/swab
    # for MRSA pneumonia.
    if has_mrsa and has_nasal and has_pneumonia and has_rule_out:
        return "diagnostic_accuracy_primary_mrsa_nasal_screening"

    # 7. Timing of negative MRSA screen validity.
    if is_timing:
        return "timing_of_negative_mrsa_screen"

    # 8. Colonization-only studies (no pneumonia, no rule-out test
    # accuracy claim). Checked BEFORE the wrong_index_test catch-all
    # so colonization articles don't get misclassified as "off-lane".
    if has_mrsa and is_colonization_only:
        return "colonization_only_not_pneumonia"

    # 9. General MRSA colonization (no pneumonia in scope).
    # Checked BEFORE wrong_index_test so MRSA colonization articles
    # without nasal/rule-out wording still land in the correct lane.
    if has_mrsa and not has_pneumonia and not is_treatment_only:
        return "general_mrsa_colonization_not_pneumonia"

    # 10. Treatment-only article (no test accuracy / no nasal screen).
    if has_mrsa and is_treatment_only:
        return "treatment_only_without_test_accuracy"

    # 2. Wrong index test — catch-all for off-lane articles that
    # don't fit any of the lanes above. Closed-set fallback.
    if not has_nasal and not has_rule_out and not is_treatment_only:
        return "wrong_index_test"

    # Final fallback closed-set bucket.
    if is_guideline and has_mrsa:
        return "diagnostic_accuracy_primary_mrsa_nasal_screening"
    return "wrong_index_test"
