"""Phase K4A.4 — Incidence / Risk / Epidemiology deterministic
detection + ranking helpers for the
``"What is the incidence/prevalence/risk of OUTCOME in patients with
RISK_CONTEXT?"`` question family.

Scope (closed-set, deterministic, NO LLM):

  • Recognize the question shape (incidence/prevalence/risk/frequency
    of a complication in a population/risk-context).

  • Extract outcome (e.g., ``aspiration pneumonia``) + population
    qualifier (e.g., ``low Glasgow Coma Scale``, ``coma``, ``loss of
    consciousness``) deterministically.

  • Closed-set GCS / Glasgow Coma Scale acronym + threshold expansion
    that recognizes "GCS 8 or less", "GCS ≤8", "low GCS", "depressed
    consciousness", etc.

  • Closed-set 10-label directness classifier the picott_ranker uses
    to boost direct incidence/risk studies + cohort studies above
    pneumonia-without-aspiration / GCS-without-aspiration articles.
    Scoped to THIS question family only.

Cross-phase contracts:
  • K4A.2.1 generic-anchor denylist is concept-level exact-match —
    multi-word phrases like ``aspiration pneumonia`` and ``low Glasgow
    Coma Scale`` are NOT in the denylist. We rely on that.
  • K4A.2 (severity-marker) + K4A.3 (rule-out) routes remain intact;
    the K4A.4 route is mutually exclusive with both (only one fires
    per session).
  • DO NOT touch the K4A.3 module — it is read-only reference here.
"""
from __future__ import annotations

import re
from typing import Literal


# =====================================================================
# Closed-set GCS / Glasgow Coma Scale acronym + threshold expansion.
# =====================================================================
GCS_SYNONYMS: tuple[str, ...] = (
    "low Glasgow Coma Scale",
    "Glasgow Coma Scale",
    "Glasgow coma score",
    "Glasgow coma scale score",
    "GCS",
    "low GCS",
    "depressed GCS",
    "GCS less than 8",
    "GCS below 8",
    "GCS under 8",
    "GCS <= 8",
    "GCS \u2264 8",
    "GCS \u22648",
    "GCS <8",
    "GCS 8 or less",
    "GCS 8",
)

# Closed-set related depressed-consciousness terms — broad-recall
# friendly, NOT a separate disease anchor.
DEPRESSED_CONSCIOUSNESS_SYNONYMS: tuple[str, ...] = (
    "coma",
    "comatose",
    "unconscious",
    "unconsciousness",
    "loss of consciousness",
    "depressed consciousness",
    "decreased level of consciousness",
    "impaired consciousness",
    "altered mental status",
    "altered level of consciousness",
)

# Closed-set aspiration-pneumonia synonyms.
ASPIRATION_PNEUMONIA_SYNONYMS: tuple[str, ...] = (
    "aspiration pneumonia",
    "aspiration pneumonitis",
    "aspiration-related pneumonia",
)

# Closed-set incidence / risk / epidemiology method cues.
INCIDENCE_RISK_OUTCOMES: tuple[str, ...] = (
    "incidence",
    "prevalence",
    "frequency",
    "occurrence",
    "rate",
    "risk",
    "risk factor",
    "risk factors",
    "predictor",
    "predictors",
    "predict",
    "predicting",
)


# =====================================================================
# Question detection — closed-set regex patterns.
# =====================================================================
_OUTCOME_RE = (
    r"(?P<outcome>"
    # Aspiration-pneumonia family (closed-set canonical anchors).
    r"aspiration\s+pneumonia|aspiration\s+pneumonitis|aspiration[\-\s]"
    r"related\s+pneumonia|"
    # Other common closed-set outcomes (pneumonia + ICU complications).
    r"ventilator[\-\s]associated\s+pneumonia|vap|"
    r"hospital[\-\s]acquired\s+pneumonia|hap|"
    r"pulmonary\s+aspiration|"
    r"sepsis|severe\s+sepsis|septic\s+shock|"
    r"acute\s+respiratory\s+distress\s+syndrome|ards|"
    r"deep\s+vein\s+thrombosis|dvt|"
    r"pulmonary\s+embolism|pe|"
    r"pressure\s+(?:ulcer|injury|injuries)|"
    r"delirium|"
    r"acute\s+kidney\s+injury|aki|"
    r"(?:severe\s+)?acute\s+pancreatitis|"
    r"stroke|"
    r"pneumonia|"
    # Generic fallback: any short noun phrase (≤ 6 words) up to the
    # closed-set glue word (``in/among/after/following``). This is
    # the safety net for diseases not enumerated above — preserves
    # K4A.4 routing on ``"incidence of severe X in patients with low
    # GCS"``. Closed-set boundary; never wildcard.
    r"(?:severe\s+|acute\s+|chronic\s+|new[\-\s]onset\s+)?"
    r"[a-z][a-z\-]+(?:\s+[a-z][a-z\-]+){0,4})"
)

# Capture a population/risk-context phrase. We accept "patients with
# X", "X patients", "after X", "in X" — closed-set glue words to
# avoid greedy over-capture.
_RISK_CONTEXT_RE = (
    r"(?P<context>"
    # GCS family.
    r"(?:patients?\s+with\s+)?(?:gcs|glasgow\s+coma\s+scale|glasgow\s+coma\s+score)"
    r"(?:\s+(?:score|level))?\s*"
    r"(?:(?:of\s+)?(?:less\s+than|below|under|<=|<|\u2264|\u22648)\s*\d+|"
    r"\d+\s+or\s+less|"
    r"\d+|"
    r")|"
    # Low GCS / depressed-consciousness family.
    r"(?:patients?\s+with\s+)?(?:low|depressed)\s+gcs|"
    r"(?:patients?\s+with\s+)?(?:low|depressed)\s+glasgow\s+coma\s+scale|"
    r"(?:patients?\s+with\s+)?(?:low|depressed|decreased|impaired|altered)"
    r"\s+(?:level\s+of\s+)?consciousness|"
    r"(?:patients?\s+with\s+)?altered\s+mental\s+status|"
    # Coma / comatose family.
    r"comatose(?:\s+patients?)?|"
    r"(?:patients?\s+(?:in|with)\s+)?coma|"
    r"loss\s+of\s+consciousness|"
    r"unconscious(?:ness)?(?:\s+patients?)?|"
    # Generic "patients with X" risk context — only matched when the
    # outer pattern enforces a real risk word.
    r"after\s+loss\s+of\s+consciousness|"
    r"after\s+(?:coma|comatose|unconsciousness))"
)

_INCIDENCE_RISK_QUESTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 1. "What is the incidence/prevalence/risk of [outcome] in
    #    [patients with] [risk_context]?"
    re.compile(
        r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:incidence|prevalence|"
        r"frequency|occurrence|rate|risk)\s+of\s+"
        + _OUTCOME_RE
        + r"\s+(?:in|among|with(?:in)?|after|following)\s+"
        + _RISK_CONTEXT_RE,
        re.IGNORECASE,
    ),
    # 2. "How often does [outcome] occur in/after [risk_context]?"
    re.compile(
        r"\bhow\s+(?:often|frequently|commonly|likely)\s+(?:does|do|is|are)"
        r"\s+" + _OUTCOME_RE + r"\s+(?:occur|develop|happen|arise|"
        r"present)\s+(?:in|among|after|following|with)\s+"
        + _RISK_CONTEXT_RE,
        re.IGNORECASE,
    ),
    # 3. "Does [risk_context] predict [outcome]?"
    re.compile(
        r"\bdoes\s+" + _RISK_CONTEXT_RE + r"\s+predict\s+" + _OUTCOME_RE,
        re.IGNORECASE,
    ),
    # 4. "Is [risk_context] a risk factor for [outcome]?"
    re.compile(
        r"\bis\s+" + _RISK_CONTEXT_RE + r"\s+(?:a\s+)?(?:risk\s+factor|"
        r"predictor|risk)\s+(?:for|of)\s+" + _OUTCOME_RE,
        re.IGNORECASE,
    ),
    # 5. "What are the risk factors / predictors of [outcome]
    #    in [risk_context]?"
    re.compile(
        r"\bwhat\s+(?:are|is)\s+(?:the\s+)?(?:risk\s+factors?|"
        r"predictors?|determinants?)\s+(?:of|for)\s+"
        + _OUTCOME_RE
        + r"\s+(?:in|among|after|following|with)\s+"
        + _RISK_CONTEXT_RE,
        re.IGNORECASE,
    ),
)


def is_incidence_risk_question(text: str) -> bool:
    """Closed-set deterministic detection of the incidence/risk
    epidemiology question family. NO LLM, NO network."""
    if not text:
        return False
    norm = text.strip().lower()
    for pat in _INCIDENCE_RISK_QUESTION_PATTERNS:
        if pat.search(norm):
            return True
    return False


def extract_incidence_risk_components(text: str) -> dict[str, str | bool | int]:
    """Closed-set extraction of the (outcome, risk_context) pair.

    Returns:
      - ``outcome``: canonical outcome phrase, e.g. ``aspiration pneumonia``
      - ``risk_context``: canonical risk-context phrase, e.g.
        ``low Glasgow Coma Scale``
      - ``gcs_route``: True if the risk_context references GCS / Glasgow
        Coma Scale / depressed-consciousness terms
      - ``matched_pattern``: index of the matched closed-set pattern
        (0..4), or ``-1`` if no match
    """
    out: dict[str, str | bool | int] = {
        "outcome": None,
        "risk_context": None,
        "gcs_route": False,
        "matched_pattern": -1,
    }
    if not text:
        return out
    norm = text.strip().lower()
    for idx, pat in enumerate(_INCIDENCE_RISK_QUESTION_PATTERNS):
        m = pat.search(norm)
        if not m:
            continue
        out["matched_pattern"] = idx
        try:
            raw_outcome = m.group("outcome")
        except Exception:  # noqa: BLE001
            raw_outcome = None
        try:
            raw_context = m.group("context")
        except Exception:  # noqa: BLE001
            raw_context = None
        if raw_outcome:
            out["outcome"] = _canonicalize_outcome(raw_outcome)
        if raw_context:
            out["risk_context"] = _canonicalize_risk_context(raw_context)
        break
    blob = (
        (out.get("risk_context") or "")
        + " "
        + (text or "")
    ).lower()
    out["gcs_route"] = bool(
        "gcs" in blob
        or "glasgow coma" in blob
        or "coma" in blob
        or "comatose" in blob
        or "consciousness" in blob
        or "altered mental status" in blob
        or "unconscious" in blob
    )
    return out


def _canonicalize_outcome(raw: str) -> str:
    """Closed-set canonicalization for outcome strings."""
    r = (raw or "").strip().lower()
    if not r:
        return ""
    if "aspiration pneumonia" in r or "aspiration pneumonitis" in r or (
        "aspiration-related" in r or "aspiration related" in r
    ):
        return "aspiration pneumonia"
    if "pulmonary aspiration" in r:
        return "aspiration pneumonia"
    if "ventilator-associated pneumonia" in r or "vap" == r:
        return "ventilator-associated pneumonia"
    if "hospital-acquired pneumonia" in r or "hap" == r:
        return "hospital-acquired pneumonia"
    if r == "pneumonia":
        return "pneumonia"
    return raw.strip()


def _canonicalize_risk_context(raw: str) -> str:
    """Closed-set canonicalization for risk-context phrases."""
    r = (raw or "").strip().lower()
    if not r:
        return ""
    # GCS / Glasgow Coma Scale family — collapse all GCS expressions
    # to a single canonical phrase. The OR-block downstream surfaces
    # every accepted variant.
    if "gcs" in r or "glasgow coma" in r:
        return "low Glasgow Coma Scale"
    # Coma / comatose / unconscious / loss-of-consciousness all
    # collapse to "depressed consciousness".
    if any(tok in r for tok in (
        "coma", "comatose", "unconscious", "loss of consciousness",
        "depressed consciousness", "decreased level of consciousness",
        "impaired consciousness", "altered mental status",
        "altered level of consciousness",
    )):
        return "depressed consciousness"
    return raw.strip()


# =====================================================================
# OR-block synonyms for the Boolean query compiler.
# =====================================================================
def aspiration_pneumonia_or_block_synonyms() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in ASPIRATION_PNEUMONIA_SYNONYMS:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def low_consciousness_or_block_synonyms() -> list[str]:
    """Closed-set OR-block list combining GCS expressions + depressed-
    consciousness synonyms (used for broad-recall queries)."""
    closed: list[str] = list(GCS_SYNONYMS) + list(DEPRESSED_CONSCIOUSNESS_SYNONYMS)
    seen: set[str] = set()
    out: list[str] = []
    for s in closed:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def incidence_risk_or_block_synonyms() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in INCIDENCE_RISK_OUTCOMES:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


# =====================================================================
# Closed-set 10-label directness classifier.
# =====================================================================
IncidenceRiskDirectnessLabel = Literal[
    "direct_incidence_aspiration_pneumonia_low_gcs",
    "aspiration_pneumonia_low_consciousness_cohort",
    "aspiration_pneumonia_low_gcs_systematic_review",
    "gcs_aspiration_risk_study",
    "intubation_low_gcs_aspiration_prevention_related",
    "aspiration_pneumonia_general_incidence",
    "pneumonia_without_aspiration",
    "low_gcs_without_aspiration_pneumonia",
    "wrong_outcome",
    "wrong_population",
]

INCIDENCE_RISK_DIRECTNESS_RANKED: tuple[IncidenceRiskDirectnessLabel, ...] = (
    "direct_incidence_aspiration_pneumonia_low_gcs",
    "aspiration_pneumonia_low_consciousness_cohort",
    "aspiration_pneumonia_low_gcs_systematic_review",
    "gcs_aspiration_risk_study",
    "intubation_low_gcs_aspiration_prevention_related",
    "aspiration_pneumonia_general_incidence",
    "pneumonia_without_aspiration",
    "low_gcs_without_aspiration_pneumonia",
    "wrong_outcome",
    "wrong_population",
)

INCIDENCE_RISK_DIRECTNESS_BOOSTS: dict[IncidenceRiskDirectnessLabel, int] = {
    "direct_incidence_aspiration_pneumonia_low_gcs": 14,
    "aspiration_pneumonia_low_consciousness_cohort": 12,
    "aspiration_pneumonia_low_gcs_systematic_review": 11,
    "gcs_aspiration_risk_study": 9,
    "intubation_low_gcs_aspiration_prevention_related": 5,
    "aspiration_pneumonia_general_incidence": 3,
    "pneumonia_without_aspiration": 0,
    "low_gcs_without_aspiration_pneumonia": 0,
    "wrong_outcome": 0,
    "wrong_population": 0,
}

INCIDENCE_RISK_DIRECTNESS_CAPS: dict[IncidenceRiskDirectnessLabel, int | None] = {
    "direct_incidence_aspiration_pneumonia_low_gcs": None,
    "aspiration_pneumonia_low_consciousness_cohort": None,
    "aspiration_pneumonia_low_gcs_systematic_review": None,
    "gcs_aspiration_risk_study": None,
    "intubation_low_gcs_aspiration_prevention_related": None,
    "aspiration_pneumonia_general_incidence": 50,
    "pneumonia_without_aspiration": 30,
    "low_gcs_without_aspiration_pneumonia": 25,
    "wrong_outcome": 15,
    "wrong_population": 20,
}

INCIDENCE_RISK_DIRECTNESS_CAP_REASONS: dict[
    IncidenceRiskDirectnessLabel, str | None,
] = {
    "direct_incidence_aspiration_pneumonia_low_gcs": None,
    "aspiration_pneumonia_low_consciousness_cohort": None,
    "aspiration_pneumonia_low_gcs_systematic_review": None,
    "gcs_aspiration_risk_study": None,
    "intubation_low_gcs_aspiration_prevention_related": None,
    "aspiration_pneumonia_general_incidence": "lane_mismatch_general_incidence",
    "pneumonia_without_aspiration": "lane_mismatch_pneumonia_without_aspiration",
    "low_gcs_without_aspiration_pneumonia": "lane_mismatch_gcs_without_aspiration",
    "wrong_outcome": "wrong_outcome",
    "wrong_population": "wrong_population",
}


_ASPIRATION_RE = re.compile(
    r"\b(?:aspiration\s+pneumonia|aspiration\s+pneumonitis|aspiration[\-\s]"
    r"related\s+pneumonia|pulmonary\s+aspiration)\b",
    re.IGNORECASE,
)
_PNEUMONIA_RE = re.compile(r"\bpneumonia\b", re.IGNORECASE)
_LOW_CONSCIOUSNESS_RE = re.compile(
    r"\b(?:gcs|glasgow\s+coma\s+scale|glasgow\s+coma\s+score|low\s+gcs|"
    r"depressed\s+gcs|coma|comatose|unconscious|loss\s+of\s+consciousness|"
    r"depressed\s+consciousness|decreased\s+level\s+of\s+consciousness|"
    r"impaired\s+consciousness|altered\s+mental\s+status|"
    r"altered\s+level\s+of\s+consciousness)\b",
    re.IGNORECASE,
)
_INCIDENCE_RE = re.compile(
    r"\b(?:incidence|prevalence|frequency|occurrence|rate|risk|"
    r"risk\s+factor[s]?|predictor[s]?|predict(?:ing|ed)?|"
    r"epidemiology|cohort)\b",
    re.IGNORECASE,
)
_SR_RE = re.compile(
    r"\b(?:systematic\s+review|meta[\-\s]?analysis|umbrella\s+review|"
    r"scoping\s+review)\b",
    re.IGNORECASE,
)
_COHORT_RE = re.compile(
    r"\b(?:cohort|retrospective|prospective|observational|case[\-\s]control|"
    r"longitudinal|follow[\-\s]?up)\b",
    re.IGNORECASE,
)
_INTUBATION_PREVENTION_RE = re.compile(
    r"\b(?:intubation|airway\s+management|endotracheal|airway\s+protection|"
    r"prevent|prevention|reduce|reducing)\b",
    re.IGNORECASE,
)


def classify_incidence_risk_directness(
    article: dict,
    *,
    outcome_anchor: str = "aspiration pneumonia",
    risk_context_anchor: str = "low Glasgow Coma Scale",
) -> IncidenceRiskDirectnessLabel:
    """Closed-set deterministic directness classifier for the
    incidence/risk epidemiology question family.

    Mirrors the K4A.2 / K4A.3 patterns: inputs are an article dict
    plus closed-set anchors; output is a closed-set directness label.
    """
    title = str((article or {}).get("title") or "")
    abstract = str((article or {}).get("abstract") or "")
    pub_types = [
        str(p).lower() for p in ((article or {}).get("publication_types_json") or [])
    ]
    blob = f"{title}\n{abstract}".lower()

    has_aspiration = bool(_ASPIRATION_RE.search(blob))
    has_pneumonia = bool(_PNEUMONIA_RE.search(blob))
    has_low_consciousness = bool(_LOW_CONSCIOUSNESS_RE.search(blob))
    has_incidence = bool(_INCIDENCE_RE.search(blob))
    is_sr = bool(_SR_RE.search(blob)) or any(
        "systematic review" in p or "meta-analysis" in p
        or "meta analysis" in p for p in pub_types
    )
    is_cohort = bool(_COHORT_RE.search(blob)) or any(
        "observational" in p or "cohort" in p for p in pub_types
    )
    has_intubation_prevention = bool(_INTUBATION_PREVENTION_RE.search(blob))

    # 1. Direct incidence/risk study about aspiration pneumonia in
    #    a low-GCS / depressed-consciousness population.
    if (
        has_aspiration and has_low_consciousness and has_incidence
        and not is_sr
    ):
        return "direct_incidence_aspiration_pneumonia_low_gcs"

    # 2. Systematic review of aspiration pneumonia + low GCS /
    #    consciousness — slightly distinct from the direct primary
    #    study lane (placed above purely "cohort" because SR has
    #    higher evidence-design weight).
    if is_sr and has_aspiration and has_low_consciousness:
        return "aspiration_pneumonia_low_gcs_systematic_review"

    # 3. Aspiration pneumonia cohort in a low-consciousness
    #    population (e.g., stroke, TBI, overdose, post-op).
    if has_aspiration and has_low_consciousness and is_cohort:
        return "aspiration_pneumonia_low_consciousness_cohort"

    # 4. Intubation / airway-prevention article relevant to low-GCS
    #    aspiration prevention. Checked BEFORE the generic
    #    ``gcs_aspiration_risk_study`` fallback so the more specific
    #    prevention lane wins when intubation/airway-management
    #    framing is explicit.
    if (
        has_aspiration and has_low_consciousness
        and has_intubation_prevention
    ):
        return "intubation_low_gcs_aspiration_prevention_related"

    # 5. GCS / aspiration risk study without explicit incidence/risk
    #    method wording or intubation framing — falls into the GCS
    #    aspiration risk lane.
    if has_aspiration and has_low_consciousness:
        return "gcs_aspiration_risk_study"

    # 6. General aspiration pneumonia incidence / risk-factor article
    #    without the low-consciousness population qualifier.
    if has_aspiration and has_incidence:
        return "aspiration_pneumonia_general_incidence"

    # 7. Pneumonia article that doesn't mention aspiration.
    if has_pneumonia and not has_aspiration:
        return "pneumonia_without_aspiration"

    # 8. Low GCS / consciousness article that doesn't mention
    #    aspiration pneumonia.
    if has_low_consciousness and not has_aspiration:
        return "low_gcs_without_aspiration_pneumonia"

    # 9. Wrong outcome — neither aspiration nor pneumonia.
    if not has_aspiration and not has_pneumonia:
        return "wrong_outcome"

    # 10. Fallback closed-set bucket — wrong population.
    return "wrong_population"
