"""Deterministic question reading: the always-available baseline (W1.4).

Conservative by design: it only emits what the words of the question support,
with the supporting span, and leaves a slot empty rather than guessing. The
model interpretation (when configured) is layered on top and verified against
the question text; without a model this alone produces the interpretation.

Deliberately independent of the legacy parser: that one misread "impact of X on
Y" as prognosis and appended conditions into outcomes (see the reuse register).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from clinical_engine.contracts.question import QuestionElement

_WS = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _WS.sub(" ", s).strip(" \t\n.,;:?!\"'()[]")


# ── Age groups (with negation) ──────────────────────────────────────────────

_AGE_PATTERNS: list[tuple[str, str]] = [
    ("neonate", r"\b(neonat\w*|newborns?|preterm (?:infants?|neonates?)|premature infants?)\b"),
    ("infant", r"\b(infants?|infancy)\b"),
    ("child", r"\b(child(?:ren)?|kids?|school[- ]age(?:d)? children|toddlers?)\b"),
    ("pediatric", r"\b(p(?:a)?ediatrics?|p(?:a)?ediatric (?:patients?|population))\b"),
    ("adolescent", r"\b(adolescen\w*|teen(?:ager)?s?|youths?)\b"),
    ("older_adult", r"\b(elderly|older (?:adults?|people|patients|persons|women|men)|geriatric\w*|seniors?|aged (?:65|70|75|80)|over (?:65|70|75|80)(?: years)?|octogenarians?|nonagenarians?)\b"),
    ("adult", r"\b(adults?|adult (?:patients?|population))\b"),
]
_NEGATION = re.compile(r"\b(?:excluding|except|not|non[- ]|other than|without)\s*$", re.I)


def _age_elements(text: str) -> list[QuestionElement]:
    found: list[QuestionElement] = []
    taken: list[tuple[int, int]] = []
    for group, pattern in _AGE_PATTERNS:
        for m in re.finditer(pattern, text, re.I):
            if any(a <= m.start() < b for a, b in taken):
                continue
            # "older adults" must not also count as "adult"
            taken.append((m.start(), m.end()))
            prefix = text[max(0, m.start() - 14):m.start()]
            negated = bool(_NEGATION.search(prefix)) or text[max(0, m.start() - 4):m.start()].lower().endswith("non-")
            found.append(QuestionElement(
                role="age_group", value=group, source="user_text", span_text=m.group(0),
                negated=negated, confidence=0.95,
            ))
            break
    return found


# ── Setting ─────────────────────────────────────────────────────────────────

_SETTINGS: list[tuple[str, str]] = [
    ("intensive care unit", r"\b(icu|intensive care(?: units?)?|critical care|critically ill)\b"),
    ("emergency department", r"\b(emergency department|emergency room|\bed\b|a&e|emergency setting)\b"),
    ("primary care", r"\b(primary care|general practice|family practice)\b"),
    ("outpatient", r"\b(outpatients?|ambulatory|community[- ]dwelling)\b"),
    ("inpatient", r"\b(inpatients?|hospitali[sz]ed(?: patients?)?)\b"),
    ("prehospital", r"\b(pre-?hospital|out-of-hospital|ems|paramedic)\b"),
    ("perioperative", r"\b(perioperative|postoperative|post-operative|intraoperative|surgical patients?)\b"),
    ("nursing home", r"\b(nursing homes?|long[- ]term care|care homes?)\b"),
]


def _setting_elements(text: str) -> list[QuestionElement]:
    out = []
    for value, pattern in _SETTINGS:
        m = re.search(pattern, text, re.I)
        if m:
            out.append(QuestionElement(
                role="setting", value=value, source="user_text", span_text=m.group(0), confidence=0.9,
            ))
    return out[:2]


# ── Time frame and publication window ──────────────────────────────────────

_TIME_FRAME = re.compile(
    r"\b(\d{1,3}[- ](?:day|week|month|year)s?(?: all[- ]cause)?|short[- ]term|long[- ]term|in[- ]hospital|within \d{1,3} (?:hours?|days?|weeks?))\b",
    re.I,
)
_PUB_WINDOW = re.compile(
    r"\b((?:in|over|from) the (?:last|past) \d{1,2} years?|since (?:19|20)\d{2}|after (?:19|20)\d{2}|recent (?:studies|trials|evidence|literature))\b",
    re.I,
)


def _time_elements(text: str) -> list[QuestionElement]:
    out = []
    m = _TIME_FRAME.search(text)
    if m:
        out.append(QuestionElement(role="time_frame", value=_clean(m.group(1)).lower(), source="user_text",
                                   span_text=m.group(0), confidence=0.9))
    m = _PUB_WINDOW.search(text)
    if m:
        out.append(QuestionElement(role="publication_window", value=_clean(m.group(1)).lower(),
                                   source="user_text", span_text=m.group(0), confidence=0.9))
    return out


# ── Question type ───────────────────────────────────────────────────────────

_TYPE_CUES: list[tuple[str, str]] = [
    ("diagnosis", r"\b(sensitivity|specificity|diagnostic (?:accuracy|performance|yield|value)|accuracy of|how accurate|to (?:diagnose|detect|rule out)|for (?:diagnosing|detecting|the diagnosis of|the detection of)|rule[- ]out|predictive value|likelihood ratio|detect(?:ing|ion)? of)\b"),
    ("screening", r"\b(screening|screen(?:ed)? for|be screened)\b"),
    ("prevention", r"\b(prevent(?:s|ing|ion)?|prophyla\w*|reduce the (?:risk|incidence) of|risk reduction)\b"),
    ("harm", r"\b(side[- ]effects?|adverse (?:events?|effects?|reactions?)|harms?|toxicit\w*|safety of|complications? of)\b"),
    ("etiology_risk", r"\b(risk factors?|associated with|association (?:between|of)|link(?:ed)? (?:to|between)|caus(?:e|es|ed) by|exposure to|increase(?:s|d)? the risk|risk of \w+ (?:with|after|from|in))\b"),
    ("prognosis", r"\b(prognos\w*|predictors? of|natural history|life expectancy|survival rates?|long[- ]term outcomes? of|recurrence rates?|course of)\b"),
    ("epidemiology", r"\b(incidence|prevalence|how common|epidemiolog\w*|how many (?:patients|people))\b"),
    ("treatment", r"\b(treat(?:s|ing|ment)?|therap(?:y|ies|eutic)|efficacy|effectiveness|effective|impact of|effects? of|benefits? of|role of|use of|dos(?:e|ing|age)|timing of|versus|vs\.?|compared (?:with|to)|should .{0,40} (?:be used|be given|receive)|does .{0,60} (?:reduce|improve|lower|decrease|increase|shorten|prolong|affect|change)|do .{0,60} (?:reduce|improve|lower|decrease|increase|shorten|prolong|affect|change)|is .{0,60} (?:effective|superior|better|beneficial))\b"),
    ("clinical_management", r"\b(management of|manag(?:e|ing)|approach to|how (?:to|should) .{0,30}(?:treat|manage)|treatment options|best treatment|first[- ]line)\b"),
]
_BACKGROUND = re.compile(r"^\s*(what (?:is|are)|define|definition of|explain|describe|tell me about)\b", re.I)
_EFFECT_WORDS = re.compile(
    r"\b(impact|effects?|efficacy|effectiveness|outcomes?|risks?|treat\w*|therap\w*|reduce|improve|versus|vs|compared|mortality|benefit|harm|accuracy|sensitivity|dose|dosing|timing|prevent\w*|associated|prognos\w*|incidence|prevalence)\b",
    re.I,
)


@dataclass
class TypeReading:
    question_type: str
    confidence: float
    cues: list[str] = field(default_factory=list)


def classify(text: str) -> TypeReading:
    if _BACKGROUND.search(text) and not _EFFECT_WORDS.search(text):
        return TypeReading("background", 0.8, ["what-is"])
    hits: list[tuple[str, str]] = []
    for qtype, pattern in _TYPE_CUES:
        m = re.search(pattern, text, re.I)
        if m:
            hits.append((qtype, m.group(0)))
    if not hits:
        return TypeReading("other", 0.3, [])
    types = [h[0] for h in hits]
    # Specific designs beat generic therapy cues ("accuracy of X" is diagnosis
    # even though "of" phrases also look like therapy questions).
    for specific in ("diagnosis", "screening", "epidemiology", "prognosis", "prevention", "harm", "etiology_risk"):
        if specific in types:
            # "effect of X on risk of Y" with a therapy cue → still prevention/harm/etiology
            return TypeReading(specific, 0.7, [h[1] for h in hits])
    if "treatment" in types:
        # "treatment options for X" / "management of X" without a specific agent → management
        if "clinical_management" in types and not re.search(
            r"\b(versus|vs\.?|compared|does|do|is|should|impact of|effects? of|efficacy|dose|timing)\b", text, re.I,
        ):
            return TypeReading("clinical_management", 0.6, [h[1] for h in hits])
        return TypeReading("treatment", 0.7, [h[1] for h in hits])
    return TypeReading(types[0], 0.6, [h[1] for h in hits])


# ── Slot patterns ────────────────────────────────────────────────────────────

_POP_TAIL = r"(?:\s+(?:in|among|for)\s+(?P<pop>.+))?"
_PATTERNS: list[re.Pattern[str]] = [
    # impact/effect of X on Y (in P)
    re.compile(r"\b(?:impact|effects?|efficacy|effectiveness|role|benefits?|influence|use)\s+of\s+(?P<i>.+?)\s+(?:on|in reducing|in improving|for)\s+(?P<o>.+?)" + _POP_TAIL + r"\s*[?.]?$", re.I),
    # does X reduce/improve Y (in P)
    re.compile(r"\b(?:does|do|did|can|could|will|would)\s+(?P<i>.+?)\s+(?:reduce|improve|lower|decrease|increase|shorten|prolong|affect|change|prevent)\s+(?P<o>.+?)" + _POP_TAIL + r"\s*[?.]?$", re.I),
    # how accurate is X for diagnosing Y (in P)
    re.compile(r"\bhow (?:accurate|sensitive|specific|reliable|useful) (?:is|are)\s+(?P<t>.+?)\s+(?:for|in|at|to)\s+(?:diagnos(?:e|ing|is of)|detect(?:ing|ion of)?|the diagnosis of|the detection of|rul(?:e|ing) out)\s+(?P<c>.+?)" + _POP_TAIL + r"\s*[?.]?$", re.I),
    # accuracy of X for diagnosing Y (in P)
    re.compile(r"\b(?:accuracy|sensitivity|specificity|performance|utility|value)\s+of\s+(?P<t>.+?)\s+(?:for|in|to)\s+(?:diagnos(?:e|ing|is of)|detect(?:ing|ion of)?|the diagnosis of|the detection of|rul(?:e|ing) out)\s+(?P<c>.+?)" + _POP_TAIL + r"\s*[?.]?$", re.I),
    # is X associated with Y (in P)
    re.compile(r"\b(?:is|are|was|were)\s+(?:exposure to\s+)?(?P<e>.+?)\s+(?:associated with|linked to|a risk factor for)\s+(?P<o>.+?)" + _POP_TAIL + r"\s*[?.]?$", re.I),
    # X vs Y for/in P
    re.compile(r"^(?:.*?\b(?:of|is|are|between)\s+)?(?P<i>[^?]+?)\s+(?:versus|vs\.?|compared (?:with|to))\s+(?P<cmp>.+?)\s+(?:for|in)\s+(?P<pop>.+?)\s*[?.]?$", re.I),
]
_IS_EFFECTIVE = re.compile(r"\b(?:is|are)\s+(?P<i>.+?)\s+(?:effective|efficacious|beneficial|useful|superior|safe and effective)\s+(?:for|in|at)\s+(?:(?P<verb>preventing|reducing|treating|improving|managing|lowering)\s+)?(?P<o>.+?)" + _POP_TAIL + r"\s*[?.]?$", re.I)
_RISK_OF_WITH = re.compile(r"\brisks? of\s+(?P<o>.+?)\s+(?:with|from|after|associated with|due to|on)\s+(?P<e>.+?)" + _POP_TAIL + r"\s*[?.]?$", re.I)
_TIMING_DOSE = re.compile(r"\b(?:optimal\s+)?(?:timing|dose|doses|dosing|dosage|duration|choice|type)(?:\s+(?:and|or|&)\s+(?:timing|dose|dosing|dosage|duration))?\s+of\s+(?P<i>.+?)" + _POP_TAIL + r"\s*[?.]?$", re.I)
_SHOULD_RECEIVE = re.compile(r"\bshould\s+(?P<pop>.+?)\s+(?:receive|be given|be treated with|get|be started on|be offered)\s+(?P<i>.+?)\s*[?.]?$", re.I)
_MANAGEMENT = re.compile(r"\b(?:management|treatment(?: options)?|therapy|approach)\s+(?:of|for|to)\s+(?P<c>.+?)" + _POP_TAIL + r"\s*[?.]?$", re.I)
_BACKGROUND_TOPIC = re.compile(r"^\s*(?:what (?:is|are)|define|definition of|explain|describe|tell me about)\s+(?P<c>.+?)\s*[?.]?$", re.I)
_PROGNOSIS = re.compile(r"\b(?:prognosis|outcomes?|survival|mortality)\s+(?:of|in|for|after)\s+(?P<c>.+?)\s*[?.]?$", re.I)
_EPI = re.compile(r"\b(?:incidence|prevalence|epidemiology)\s+of\s+(?P<c>.+?)" + _POP_TAIL + r"\s*[?.]?$", re.I)
_WITH_CONDITION = re.compile(r"\b(?:patients?|people|persons?|adults?|children|women|men|infants?|neonates?|individuals?|those|subjects?)\s+(?:with|having|diagnosed with|who have|suffering from)\s+(?P<c>.+)$", re.I)
_CONDITION_PATIENTS = re.compile(r"^(?P<c>.+?)\s+patients?$", re.I)
_LEAD_ARTICLES = re.compile(r"^(?:the|a|an|any|all)\s+", re.I)
_TRAILING_POP = re.compile(r"\s+(?:in|among|for)\s+(?:patients?|people|adults?|children|women|men).*$", re.I)


def _strip(value: str) -> str:
    v = _clean(value)
    v = _LEAD_ARTICLES.sub("", v)
    return v[:120]


# Words that make one arm of a comparison: "Early versus delayed antibiotics".
_ARM_MODIFIERS = {
    "early", "late", "delayed", "immediate", "deferred", "prompt", "rapid", "high", "low", "higher", "lower",
    "high-dose", "low-dose", "standard-dose", "short", "long", "short-course", "long-course", "restrictive",
    "liberal", "conservative", "aggressive", "intensive", "continuous", "intermittent", "oral", "intravenous",
    "iv", "topical", "inhaled", "subcutaneous", "intramuscular", "fixed", "weight-based", "single", "multiple",
    "daily", "weekly",
}
# "acute kidney injury after cardiac surgery": the condition, then who the patients are.
_AFTER_EVENT = re.compile(r"^(?P<c>.+?)\s+(?:after|following)\s+(?P<e>.+)$", re.I)


def _fill_elided_head(arm: str, other: str) -> str:
    """'Early' vs 'delayed antibiotics' → 'Early antibiotics' (the shared noun is left out)."""
    a, o = arm.split(), other.split()
    if len(a) == 1 and len(o) >= 2 and a[0].lower() in _ARM_MODIFIERS and o[0].lower() in _ARM_MODIFIERS:
        return f"{arm} {' '.join(o[1:])}"
    return arm


_GENERIC_POP = re.compile(
    r"(?i)^(?:(?:the|all|any)\s+)?(?:patients?|people|persons?|individuals?|subjects?|population|cases?|those|"
    r"adults?|children|infants?|neonates?|women|men|elderly|older adults?|adolescents?)$"
)
_AGE_WORDS = re.compile(
    r"(?i)\b(?:older|elderly|geriatric|adult|adults|children|child|pediatric|paediatric|infants?|neonates?|"
    r"newborns?|adolescents?|teenagers?|aged? (?:\d+|over \d+)|over \d+(?: years)?(?: old)?|\d+ years? (?:and|or) older)\b"
)


_STOPWORDS_ONLY = re.compile(r"(?i)^(?:(?:the|a|an|in|at|on|of|and|or|with|for|to|admitted)\b\s*)+$")


def _remove_settings(text: str) -> str:
    out = text
    for _, pattern in _SETTINGS:
        out = re.sub(pattern, " ", out, flags=re.I)
    out = re.sub(r"(?i)\b(?:in|at|admitted to|on|within)\s+(?:the\s+)?(?=\s|$)", " ", out)
    out = _clean(out)
    return "" if _STOPWORDS_ONLY.match(out) else out


def _condition_from_population(pop: str) -> tuple[Optional[str], Optional[str]]:
    """Split a population phrase into (condition, population descriptor).

    'adults with septic shock in the ICU'     → ('septic shock', None)
    'septic shock patients'                    → ('septic shock', None)
    'mechanically ventilated patients'         → (None, 'mechanically ventilated')
    'critically ill older adults'              → (None, None)   # setting + age only
    'atrial fibrillation in patients over 80'  → ('atrial fibrillation', None)
    """
    pop = _clean(pop)
    # "X in patients over 80" → the condition is X; the tail is age/population.
    head = re.split(r"(?i)\s+(?:in|among|for)\s+(?=(?:\w+\s+){0,3}(?:patients?|people|adults?|children|infants?|neonates?|women|men|individuals?)\b)", pop)[0]
    m = _WITH_CONDITION.search(head)
    if m:
        return _strip(_remove_settings(m.group("c"))) or None, None
    head = _remove_settings(head)
    if not head:
        return None, None
    m = _CONDITION_PATIENTS.match(head)
    if m:
        cond = _clean(_AGE_WORDS.sub(" ", m.group("c")))
        if cond and not _GENERIC_POP.match(cond):
            return _strip(cond), None
        return None, None
    rest = _clean(_AGE_WORDS.sub(" ", head))
    rest = re.sub(r"(?i)\b(?:patients?|people|persons?|individuals?|subjects?|population)\b", " ", rest)
    rest = _clean(rest)
    if not rest or _GENERIC_POP.match(rest):
        return None, None
    if head.lower().endswith(("patients", "people", "individuals", "adults", "children", "women", "men")):
        return None, _strip(rest)
    return _strip(rest), None


def _elem(role: str, value: str, span: str, conf: float = 0.75) -> QuestionElement:
    return QuestionElement(role=role, value=value, source="user_text", span_text=span[:300], confidence=conf)


@dataclass
class RulesReading:
    question_type: str
    type_confidence: float
    elements: list[QuestionElement]
    notes: list[str] = field(default_factory=list)


def read(question: str) -> RulesReading:
    text = _WS.sub(" ", question).strip()
    reading = classify(text)
    qtype = reading.question_type
    elements: list[QuestionElement] = []

    def add(role: str, raw: Optional[str], conf: float = 0.75) -> None:
        if not raw:
            return
        if role in ("outcome", "intervention", "exposure", "condition"):
            raw = re.split(r"(?i)\s+(?:compared (?:with|to)|versus|vs\.?)\s+", raw)[0]
        value = _strip(raw)
        if value and not any(e.role == role and e.value.lower() == value.lower() for e in elements):
            elements.append(_elem(role, value, raw, conf))

    matched = False
    if qtype == "background":
        m = _BACKGROUND_TOPIC.search(text)
        if m:
            add("condition", m.group("c"), 0.7)
            matched = True
    if not matched and qtype == "clinical_management":
        m = _MANAGEMENT.search(text)
        if m:
            add("condition", m.group("c"))
            if m.group("pop"):
                cond, desc = _condition_from_population(m.group("pop"))
                add("condition", cond)
                add("population", desc)
            matched = True
    if not matched and qtype == "epidemiology":
        m = _EPI.search(text)
        if m:
            after = _AFTER_EVENT.match(m.group("c"))
            if after:
                add("condition", after.group("c"))
                add("population", after.group("e"))
            else:
                add("condition", m.group("c"))
            if m.group("pop"):
                cond, desc = _condition_from_population(m.group("pop"))
                add("population", cond or desc)
            matched = True
    if not matched:
        m = _IS_EFFECTIVE.search(text)
        if m:
            add("intervention", m.group("i"))
            if (m.group("verb") or "").lower() in ("treating", "managing"):
                add("condition", m.group("o"))
            else:
                add("outcome", m.group("o"))
            if m.group("pop"):
                cond, desc = _condition_from_population(m.group("pop"))
                add("condition", cond)
                add("population", desc)
            matched = True
    if not matched:
        m = _RISK_OF_WITH.search(text)
        if m:
            add("outcome", m.group("o"))
            add("exposure", m.group("e"))
            if m.group("pop"):
                cond, desc = _condition_from_population(m.group("pop"))
                add("condition", cond)
                add("population", desc)
            matched = True
    if not matched:
        m = _SHOULD_RECEIVE.search(text)
        if m:
            add("intervention", _TRAILING_POP.sub("", m.group("i")))
            cond, desc = _condition_from_population(m.group("pop"))
            add("condition", cond)
            add("population", desc)
            matched = True
    if not matched:
        m = _TIMING_DOSE.search(text)
        if m:
            add("intervention", m.group("i"))
            if m.group("pop"):
                cond, desc = _condition_from_population(m.group("pop"))
                add("condition", cond)
                add("population", desc)
            matched = True
    if not matched:
        for pattern in _PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            groups = m.groupdict()
            if groups.get("t"):  # diagnostic pattern
                add("index_test", groups["t"])
                add("condition", groups.get("c"))
            elif groups.get("e"):  # association pattern
                add("exposure", groups["e"])
                add("outcome", groups.get("o"))
            else:
                intervention = groups.get("i")
                outcome = groups.get("o")
                if intervention:
                    intervention = _TRAILING_POP.sub("", intervention)
                    if groups.get("cmp"):
                        intervention = _fill_elided_head(intervention, groups["cmp"])
                    role = "exposure" if qtype in ("harm", "etiology_risk") else "intervention"
                    add(role, intervention)
                if groups.get("cmp"):
                    add("comparator", groups["cmp"])
                if outcome:
                    # "on mortality in adults with X" may have been swallowed by the lazy group
                    add("outcome", _TRAILING_POP.sub("", outcome))
            if groups.get("pop"):
                cond, desc = _condition_from_population(groups["pop"])
                add("condition", cond)
                add("population", desc)
            matched = True
            break
    if not matched and qtype == "prognosis":
        m = _PROGNOSIS.search(text)
        if m:
            cond, desc = _condition_from_population(m.group("c"))
            add("condition", cond or desc or m.group("c"))
            matched = True

    # Comparator phrased separately: "... compared with placebo ..."
    if not any(e.role == "comparator" for e in elements):
        m = re.search(r"\b(?:versus|vs\.?|compared (?:with|to)|rather than)\s+(?P<c>[^,?.;]+?)(?=\s+(?:in|for|on|among)\b|[,?.;]|$)", text, re.I)
        if m:
            add("comparator", m.group("c"), 0.7)

    # Condition phrased as "in patients with X" anywhere.
    if not any(e.role == "condition" for e in elements):
        m = re.search(r"\b(?:in|among|for)\s+(?P<p>(?:[a-z-]+\s+){0,3}(?:patients?|people|adults?|children|infants?|neonates?|women|men|individuals?)\s+(?:with|having|diagnosed with)\s+[^,?.;]+)", text, re.I)
        if m:
            cond, desc = _condition_from_population(m.group("p"))
            add("condition", cond)
            add("population", desc)

    # Drop outcomes that merely repeat the condition (legacy defect: condition as outcome),
    # except where the condition *is* the outcome being prevented/predicted.
    if qtype not in ("prevention", "harm", "etiology_risk", "screening", "epidemiology"):
        conditions = {e.value.lower() for e in elements if e.role == "condition"}
        elements = [e for e in elements if not (e.role == "outcome" and e.value.lower() in conditions)]

    elements.extend(_age_elements(text))
    elements.extend(_setting_elements(text))
    elements.extend(_time_elements(text))
    # Age/setting words inside a population slot are not a separate condition.
    elements = [
        e for e in elements
        if not (e.role == "condition" and re.fullmatch(r"(?i)(?:adults?|children|infants?|neonates?|patients?|elderly|older adults?)", e.value))
    ]
    return RulesReading(question_type=qtype, type_confidence=reading.confidence, elements=elements,
                        notes=reading.cues)
