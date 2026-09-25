"""Model interpretation of a question, validated against the question text.

The model output is untrusted input:
  * strict schema; unknown question types / roles are rejected;
  * an element may claim ``stated`` only with a quote copied from the question —
    otherwise it is demoted to an inactive suggestion;
  * age groups must be backed by age words in the question;
  * values that look like citations / identifiers / URLs are dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from clinical_engine.contracts.question import AGE_GROUPS, QuestionElement

QUESTION_TYPES = (
    "treatment", "prevention", "harm", "etiology_risk", "diagnosis", "screening",
    "prognosis", "epidemiology", "clinical_management", "background", "other",
)
ROLES = (
    "condition", "population", "age_group", "sex", "pregnancy", "intervention", "exposure",
    "index_test", "comparator", "reference_standard", "prognostic_factor", "outcome",
    "setting", "time_frame", "study_design", "publication_window",
)
MAX_ELEMENTS = 16

SYSTEM_PROMPT = """You interpret a clinician's question for a clinical-evidence search.
Return ONLY a JSON object:
{"question_type": <type>, "elements": [{"role": <role>, "value": <short canonical clinical term>,
  "stated": <true|false>, "quote": <exact words copied from the question, or null>,
  "importance": <"critical"|"important"|"unspecified"|null>}], "ambiguities": [<short string>]}

question_type — one of:
  treatment (does an intervention work), prevention (does an intervention prevent a condition),
  harm (adverse effects of an intervention/exposure), etiology_risk (does an exposure raise risk),
  diagnosis (accuracy of a test), screening (should a population be screened),
  prognosis (likely course / prognostic factors), epidemiology (incidence / prevalence),
  clinical_management (how to manage a condition, no single intervention named),
  background (definition / mechanism), other.
  "Impact/effect/efficacy of <drug or procedure> on <outcome>" is treatment (or prevention), not prognosis.

roles:
  condition — the disease or state the patients HAVE (e.g. septic shock). In diagnosis questions it is
    the target condition being diagnosed (e.g. sepsis), not the reason for testing.
  population — who the patients are beyond condition and age (e.g. critically ill, mechanically ventilated,
    postmenopausal women, suspected infection). Never an age alone: use age_group for age.
  intervention / comparator — the treatments or strategies being compared (e.g. hydrocortisone taper vs
    abrupt discontinuation). exposure — a non-treatment exposure, or a drug studied for harm.
  index_test / reference_standard — diagnosis and screening questions only.
  outcome — what is measured (e.g. 28-day mortality). For prevention or incidence questions the prevented
    or incident condition is an outcome, not a condition.
  setting — where care happens (e.g. intensive care unit, emergency department, primary care, nursing home,
    low-income countries). Timing, dose or duration is never a setting.
  prognostic_factor, time_frame, study_design, publication_window, sex, pregnancy.
  age_group values: neonate, infant, child, adolescent, pediatric, adult, older_adult, all_ages.

Rules:
- "stated": true ONLY if the element is in the question; then "quote" MUST be copied verbatim from it.
- Put anything you add that the question does not say (a likely comparator, an important outcome,
  an age group) as "stated": false. NEVER state an age group the question does not mention.
- The condition the patients have is NOT an outcome, unless the question is about preventing it
  or its incidence.
- Values are short noun phrases (max 8 words), canonical clinical terms, e.g. "steroids" -> "corticosteroids";
  no verbs such as "diagnosing", and never another element inside a value.
- Mark patient-important outcomes such as mortality as "critical".
- No citations, identifiers, URLs, study names or recommendations. JSON only."""


class _ModelElement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    value: str = Field(min_length=1, max_length=200)
    stated: bool = False
    quote: Optional[str] = Field(default=None, max_length=400)
    importance: Optional[Literal["critical", "important", "unspecified"]] = None


class _ModelOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_type: str
    elements: list[_ModelElement] = Field(default_factory=list, max_length=40)
    ambiguities: list[str] = Field(default_factory=list, max_length=10)


_CITATIONISH = re.compile(
    r"(https?://|www\.|\bdoi\b|10\.\d{4,}/|\bpmid\b|\bpmc\d+|\bnct\d{6,}|\bet al\b|\[\d+\])", re.I,
)
_AGE_SYNONYMS = {
    "adults": "adult", "adult": "adult", "children": "child", "child": "child", "paediatric": "pediatric",
    "pediatric": "pediatric", "pediatrics": "pediatric", "elderly": "older_adult", "older adults": "older_adult",
    "older adult": "older_adult", "older_adult": "older_adult", "geriatric": "older_adult",
    "neonates": "neonate", "neonate": "neonate", "newborn": "neonate", "newborns": "neonate",
    "infants": "infant", "infant": "infant", "adolescents": "adolescent", "adolescent": "adolescent",
    "all ages": "all_ages", "all_ages": "all_ages",
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


_AGE_ONLY = re.compile(
    r"(?i)^(?:(?:the|all|any)\s+)?(?:adults?|adult patients|children|child|kids|infants?|neonates?|newborns?|"
    r"adolescents?|teenagers?|elderly|older adults?|older patients|pediatric(?: patients)?|paediatric(?: patients)?|"
    r"patients?|people|persons?|individuals?|all ages)$"
)
# "patients over 80", "aged 65 and older", ">= 18 years": an age, not a patient group.
_AGE_EXPRESSION = re.compile(
    r"(?i)^(?:(?:patients?|people|persons?|individuals?|adults?)\s+)?(?:(?:aged|age)\s+)?"
    r"(?:(?:over|above|older than|under|below|younger than|at least)\s+|[<>]=?\s*|[≥≤]\s*)?"
    r"\d{1,3}(?:\s*(?:-|–|to)\s*\d{1,3})?\s*(?:\+|years?|yrs?)?(?:\s+(?:old|of age))?"
    r"(?:\s+(?:and|or)\s+(?:older|over|above|younger|under))?$"
)
# "treatment options", "best therapy": the question's wording, not a treatment to evaluate.
_GENERIC_INTERVENTION = re.compile(
    r"(?i)^(?:the\s+)?(?:(?:best|optimal|current|first[- ]line|second[- ]line|available|possible|different|other|"
    r"new|newer|latest|recommended|appropriate|effective)\s+)?"
    r"(?:treatments?|therap(?:y|ies)|management|interventions?|options?|strateg(?:y|ies)|approach(?:es)?|"
    r"modalit(?:y|ies))(?:\s+(?:options?|strateg(?:y|ies)|approach(?:es)?|choices?|modalit(?:y|ies)))?$"
)
_GENERIC_OUTCOME = re.compile(
    r"(?i)^(?:the\s+)?(?:impact|effects?|efficacy|effectiveness|benefits?|outcomes?|clinical outcomes?|results?|"
    r"role|use|usefulness|value|evidence|response)$"
)
_DIAGNOSTIC_VERB = re.compile(r"(?i)^(?:diagnosing|detecting|diagnosis of|detection of|ruling out|rule out)\s+")
_EMBEDDED_COMPARISON = re.compile(r"(?i)\s+(?:compared (?:with|to)|versus|vs\.?)\s+.*$")
# A setting is a place or context of care; anything else ("timing and dose") is misfiled.
_CARE_PLACE = re.compile(
    r"(?i)\b(?:units?|icu|nicu|picu|ccu|wards?|departments?|ed|a&e|hospitals?|clinics?|care|homes?|community|"
    r"settings?|rooms?|theat(?:re|er)s?|practices?|cent(?:re|er)s?|facilit(?:y|ies)|pre-?hospital|out-of-hospital|"
    r"in-hospital|outpatients?|inpatients?|ambulatory|emergency|nursing|primary|intensive|critical|perioperative|"
    r"postoperative|intraoperative|operating|schools?|prisons?|military|rural|urban|tele(?:health|medicine)|"
    r"ambulances?|offices?|hospice|palliative|rehabilitation|dialysis|countries|country|lmics?|low-income|"
    r"middle-income|high-income|resource|regions?|nationwide|national|global|worldwide)\b"
)


_PLACEHOLDER = re.compile(
    r"(?i)^(?:the\s+|a\s+|an\s+)?(?:intervention|exposure|outcome|condition|population|treatment|comparator|"
    r"control|test|index test|drug|medication|disease|patients?|setting|unknown|none|n/?a|not specified)s?$"
)


def clean_element(role: str, value: str) -> tuple[Optional[str], str]:
    """Deterministic clean-up of a model element → (role or None to drop, value)."""
    if _PLACEHOLDER.match(value.strip()):
        return None, value  # a role name is not a value ("intervention?")
    if role == "population":
        v = value.strip().replace("_", " ")  # the model sometimes echoes an age code ("older_adult")
        if _AGE_ONLY.match(v) or _AGE_EXPRESSION.match(v):
            return None, value  # age belongs in age_group; generic nouns add nothing
    if role == "intervention" and _GENERIC_INTERVENTION.match(value.strip()):
        return None, value
    if role == "outcome" and _GENERIC_OUTCOME.match(value.strip()):
        return None, value  # "impact"/"effect" is the question's verb, not a measurable outcome
    if role in ("outcome", "condition") and _DIAGNOSTIC_VERB.match(value):
        return "condition", _DIAGNOSTIC_VERB.sub("", value).strip()
    if role in ("outcome", "intervention", "exposure", "condition"):
        value = _EMBEDDED_COMPARISON.sub("", value).strip()
    return role, value


def quote_in_question(quote: str, question: str) -> bool:
    q = _norm(quote)
    return bool(q) and f" {q} " in f" {_norm(question)} "


_COORDINATORS = ("versus", "vs", "or", "and")
# The words between the arms describe the other arm ("delayed"); a preposition or article
# means the phrase has moved on ("… antibiotics for sepsis").
_NOT_A_MODIFIER = {
    "for", "in", "of", "with", "to", "on", "among", "after", "before", "at", "by", "from", "the", "a", "an",
    "and", "or", "versus", "vs", "compared", "than", "as", "per", "into", "within", "without", "during",
    "over", "under", "who", "that", "which", "is", "are", "does", "do",
}


def elliptical_span(value: str, question: str) -> Optional[str]:
    """'Early versus delayed antibiotics' states 'early antibiotics' with the shared noun
    left out. Accept a value whose first words appear verbatim just before
    versus/vs/or/and/compared with|to and whose remaining words appear verbatim after one
    to three modifier words of the other arm; return that span of the question. Nothing
    else is accepted."""
    v, q = _norm(value).split(), _norm(question).split()
    for k in range(1, len(v)):
        head, tail = v[:k], v[k:]
        for i in range(len(q) - k + 1):
            if q[i:i + k] != head:
                continue
            j = i + k
            if j < len(q) and q[j] in _COORDINATORS:
                j += 1
            elif q[j:j + 2] in (["compared", "with"], ["compared", "to"]):
                j += 2
            else:
                continue
            for gap in (1, 2, 3):
                end = j + gap + len(tail)
                if any(w in _NOT_A_MODIFIER for w in q[j:j + gap]):
                    break
                if q[j + gap:end] == tail:
                    words = q[i:end]
                    m = re.search(r"\b" + r"\W+".join(words) + r"\b", question, re.I)
                    return m.group(0) if m else " ".join(words)
    return None


@dataclass
class ModelReading:
    question_type: Optional[str]
    elements: list[QuestionElement]
    ambiguities: list[str]
    rejected: list[str] = field(default_factory=list)


def validate_model_output(question: str, raw: dict[str, Any]) -> ModelReading:
    """Turn raw model JSON into verified elements. Never raises."""
    rejected: list[str] = []
    raw_elements = raw.get("elements", []) if isinstance(raw, dict) else None
    if not isinstance(raw_elements, list):
        return ModelReading(None, [], [], ["schema_invalid"])
    try:
        parsed = _ModelOutput.model_validate({**raw, "elements": []})
    except ValidationError:
        return ModelReading(None, [], [], ["schema_invalid"])
    # One malformed element (e.g. "value": null for an unknown reference standard)
    # drops that element only, not the whole reading.
    items: list[_ModelElement] = []
    for item in raw_elements[:40]:
        try:
            items.append(_ModelElement.model_validate(item))
        except ValidationError:
            rejected.append("element_invalid")

    qtype = parsed.question_type.strip().lower()
    if qtype not in QUESTION_TYPES:
        rejected.append(f"question_type:{qtype[:30]}")
        qtype = None

    from clinical_engine.interpretation.rules import _age_elements  # local: shared age lexicon

    elements: list[QuestionElement] = []
    for item in items[: MAX_ELEMENTS * 2]:
        role = item.role.strip().lower()
        value = " ".join(item.value.split())[:120]
        if role not in ROLES:
            rejected.append(f"role:{role[:30]}")
            continue
        if _CITATIONISH.search(value):
            rejected.append(f"citation_like:{role}")
            continue
        cleaned_role, value = clean_element(role, value)
        if cleaned_role is None or not value:
            continue
        role = cleaned_role
        if role == "age_group":
            mapped = _AGE_SYNONYMS.get(value.lower(), value.lower())
            if mapped not in AGE_GROUPS:
                rejected.append(f"age_value:{value[:30]}")
                continue
            value = mapped
        stated = bool(item.stated)
        span = None
        if stated:
            quote = (item.quote or "").strip()
            if quote and quote_in_question(quote, question):
                span = quote
            elif quote_in_question(value, question):
                span = value
            elif ellipsis := elliptical_span(value, question):
                span = ellipsis
            else:
                rejected.append(f"unsupported_quote:{role}")
                stated = False
        if stated and role == "age_group" and not _age_elements(span or ""):
            rejected.append("age_group_without_age_words")
            stated = False
        if stated and role == "setting" and not _CARE_PLACE.search(value):
            rejected.append("setting_not_a_place")
            stated = False
        elements.append(QuestionElement(
            role=role, value=value,
            source="user_text" if stated else "ai_inferred",
            active=stated,
            span_text=span,
            confidence=0.85 if stated else 0.5,
            importance=item.importance if role == "outcome" else None,
        ))
        if len(elements) >= MAX_ELEMENTS:
            break
    ambiguities = [" ".join(a.split())[:160] for a in parsed.ambiguities if a and a.strip()][:4]
    return ModelReading(qtype, elements, ambiguities, rejected)
