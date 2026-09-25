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
  prognostic_factor, setting, time_frame, study_design, publication_window, sex, pregnancy.
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
    r"patients?|people|persons?|individuals?)$"
)
_GENERIC_OUTCOME = re.compile(
    r"(?i)^(?:the\s+)?(?:impact|effects?|efficacy|effectiveness|benefits?|outcomes?|clinical outcomes?|results?|"
    r"role|use|usefulness|value|evidence|response)$"
)
_DIAGNOSTIC_VERB = re.compile(r"(?i)^(?:diagnosing|detecting|diagnosis of|detection of|ruling out|rule out)\s+")
_EMBEDDED_COMPARISON = re.compile(r"(?i)\s+(?:compared (?:with|to)|versus|vs\.?)\s+.*$")


def clean_element(role: str, value: str) -> tuple[Optional[str], str]:
    """Deterministic clean-up of a model element → (role or None to drop, value)."""
    if role == "population" and _AGE_ONLY.match(value.strip()):
        return None, value  # age belongs in age_group; generic nouns add nothing
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


@dataclass
class ModelReading:
    question_type: Optional[str]
    elements: list[QuestionElement]
    ambiguities: list[str]
    rejected: list[str] = field(default_factory=list)


def validate_model_output(question: str, raw: dict[str, Any]) -> ModelReading:
    """Turn raw model JSON into verified elements. Never raises."""
    rejected: list[str] = []
    try:
        parsed = _ModelOutput.model_validate(raw)
    except ValidationError:
        return ModelReading(None, [], [], ["schema_invalid"])

    qtype = parsed.question_type.strip().lower()
    if qtype not in QUESTION_TYPES:
        rejected.append(f"question_type:{qtype[:30]}")
        qtype = None

    from clinical_engine.interpretation.rules import _age_elements  # local: shared age lexicon

    elements: list[QuestionElement] = []
    for item in parsed.elements[: MAX_ELEMENTS * 2]:
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
            else:
                rejected.append(f"unsupported_quote:{role}")
                stated = False
        if stated and role == "age_group" and not _age_elements(span or ""):
            rejected.append("age_group_without_age_words")
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
