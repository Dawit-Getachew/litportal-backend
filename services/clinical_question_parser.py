"""ClinicalQuestionParserService — Phase 2A.1.

Deterministic-first PICO + clinical_question_type extraction. The LLM path is
optional enrichment and always gates through the same canonicalization.

The canonical clinical_question_type enum is defined in `models/pico.py`.
All output JSON conforms to `ParsedPICO`.
"""
from __future__ import annotations

import re
from typing import Any

from models.pico import CLINICAL_QUESTION_TYPES, ParsedPICO
from services.llm_provider import llm_provider
from services.picott_extractor import picott_extractor
from services.spelling_normalizer import spelling_normalizer


# --------------------------------------------------------------------------
# Synonyms & drug families (small, auditable, extend later)
# --------------------------------------------------------------------------
_SYNONYMS: dict[str, list[str]] = {
    "hypertension": ["htn", "high blood pressure"],
    "htn": ["hypertension", "high blood pressure"],
    "myocardial infarction": ["mi", "heart attack", "stemi", "nstemi"],
    "mi": ["myocardial infarction", "heart attack"],
    "atrial fibrillation": ["af", "afib", "a-fib", "nvaf", "nonvalvular atrial fibrillation"],
    "af": ["atrial fibrillation", "afib"],
    "type 2 diabetes": ["t2dm", "type 2 diabetes mellitus", "niddm"],
    "t2dm": ["type 2 diabetes", "type 2 diabetes mellitus"],
    "heart failure": ["hf", "chf", "congestive heart failure"],
    "hf": ["heart failure", "chf"],
    "hfpef": ["heart failure with preserved ejection fraction"],
    "hfref": ["heart failure with reduced ejection fraction"],
    "copd": ["chronic obstructive pulmonary disease"],
    "ckd": ["chronic kidney disease"],
    "esrd": ["end-stage renal disease"],
    "covid-19": ["covid", "sars-cov-2", "coronavirus disease"],
    "stroke": ["cerebrovascular accident", "cva", "ischemic stroke"],
    "dvt": ["deep vein thrombosis"],
    "pe": ["pulmonary embolism"],
    "vte": ["venous thromboembolism"],
    "pulmonary embolism": ["pe"],
}

_DRUG_CLASSES: dict[str, list[str]] = {
    "sglt2_inhibitor": ["empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin"],
    "glp1_agonist": ["semaglutide", "liraglutide", "dulaglutide", "exenatide", "tirzepatide"],
    "doac": ["apixaban", "rivaroxaban", "edoxaban", "dabigatran"],
    "vka": ["warfarin"],
    "statin": ["atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "pitavastatin"],
    "acei": ["lisinopril", "enalapril", "ramipril", "captopril"],
    "arb": ["losartan", "valsartan", "candesartan", "irbesartan", "telmisartan"],
    "arni": ["sacubitril", "sacubitril/valsartan"],
    "beta_blocker": ["metoprolol", "carvedilol", "bisoprolol", "atenolol"],
    "ccb": ["amlodipine", "diltiazem", "verapamil", "nifedipine"],
    "thiazide": ["hydrochlorothiazide", "chlorthalidone", "indapamide"],
    "loop_diuretic": ["furosemide", "bumetanide", "torsemide"],
    "mra": ["spironolactone", "eplerenone", "finerenone"],
    "ppi": ["omeprazole", "pantoprazole", "esomeprazole", "lansoprazole"],
    "antiplatelet": ["aspirin", "clopidogrel", "ticagrelor", "prasugrel"],
    "nsaid": ["ibuprofen", "naproxen", "diclofenac", "celecoxib", "indomethacin", "meloxicam", "ketorolac"],
}

_ALL_DRUGS: list[str] = sorted({d for vs in _DRUG_CLASSES.values() for d in vs})

# Drug-class display names (case-preserving, so "SGLT2 inhibitors" stays canonical).
_CLASS_DISPLAY_NAMES: dict[str, str] = {
    "sglt2_inhibitor": "SGLT2 inhibitors",
    "glp1_agonist": "GLP-1 agonists",
    "doac": "DOACs",
    "vka": "VKAs",
    "statin": "statins",
    "acei": "ACE inhibitors",
    "arb": "ARBs",
    "arni": "ARNIs",
    "beta_blocker": "beta blockers",
    "ccb": "calcium channel blockers",
    "thiazide": "thiazides",
    "loop_diuretic": "loop diuretics",
    "mra": "mineralocorticoid receptor antagonists",
    "ppi": "proton pump inhibitors",
    "antiplatelet": "antiplatelets",
    "nsaid": "NSAIDs",
}


# --------------------------------------------------------------------------
# Question-type heuristics + canonicalizer
# --------------------------------------------------------------------------
_QTYPE_SYNONYM_MAP: dict[str, str] = {
    # canonical → canonical (identity)
    **{v: v for v in CLINICAL_QUESTION_TYPES},
    # synonyms
    "management": "treatment",
    "therapy": "treatment",
    "intervention": "treatment",
    "drug_comparison": "treatment",
    "drug-comparison": "treatment",
    "diagnostic": "diagnosis",
    "diagnose": "diagnosis",
    "safety": "harm",
    "risk_factor": "etiology",
    "risk-factor": "etiology",
    "cause": "etiology",
    "incidence": "epidemiology",
    "prevalence": "epidemiology",
    "rec": "guideline",
    "recommendation": "guideline",
}


def _canonicalize_qtype(raw: str) -> str:
    if not raw:
        return "other"
    key = str(raw).strip().lower().replace(" ", "_")
    return _QTYPE_SYNONYM_MAP.get(key, "other")


def _classify_type(q: str) -> str:
    s = q.lower()
    # 1) The strongest treatment signal: "how does X compare with Y for OUTCOME"
    if re.search(r"\bhow does\b.+?\bcompare (?:with|to)\b", s):
        return "treatment"
    # 2) Harm / risk language has priority over generic "vs" wording.
    if re.search(r"\b(risk of|harm|harmful|adverse|side effect|bleeding|safety)\b", s):
        return "harm"
    # 3) Explicit guideline ask wins over comparisons.
    if re.search(r"\b(guideline|recommend|recommendation|practice guideline)\b", s):
        return "guideline"
    # 4) Diagnosis terminology.
    if re.search(
        r"\b(diagnostic accuracy|sensitivity|specificity|likelihood ratio|diagnose|diagnosis of)\b",
        s,
    ):
        return "diagnosis"
    # 5) Screening / prevention asks.
    if re.search(r"\bscreening\b|screen for\b", s):
        return "screening"
    if re.search(r"\bprevent(?:ion)?\b|prophylax", s) and not re.search(
        r"\b(vs\.?|versus|compared (?:with|to))\b", s
    ):
        # Pure prevention question (no comparison frame).
        return "prevention"
    # 6) Prognosis & etiology.
    if re.search(r"\b(prognosis|mortality|recurrence|survival)\b", s):
        return "prognosis"
    if re.search(r"\b(etiology|cause of|causes of|risk factors?)\b", s):
        return "etiology"
    if re.search(r"\b(prevalence|incidence|epidemiology)\b", s):
        return "epidemiology"
    if re.search(r"\b(pharmacokinetics|pharmacodynamics|drug.drug interaction)\b", s):
        return "pharmacology"
    # 7) Comparison phrasing → treatment.
    if re.search(r"\b(vs\.?|versus)\b", s):
        return "treatment"
    if re.search(r"\bcompared (?:with|to)\b", s):
        return "treatment"
    # 8) Generic treatment cues.
    if re.search(
        r"\b(treatment of|treat|therapy|effective|efficacy|effectiveness|trial|reduce|improve)\b",
        s,
    ):
        return "treatment"
    return "other"


# --------------------------------------------------------------------------
# Specialty inference
# --------------------------------------------------------------------------
_SPECIALTY_PATTERNS: list[tuple[str, str]] = [
    ("cardiology", r"\b(heart|cardiac|cardio|coronary|hf|chf|af|nvaf|stemi|nstemi|hypertens|aortic|valvul|atrial fibrillation)"),
    ("endocrinology", r"\b(diabet|t2dm|t1dm|thyroid|hypothyroid|hyperthyroid|adrenal|insulin)"),
    ("pulmonology", r"\b(copd|asthma|pulmon|lung|bronch|pneumon|pulmonary embolism)"),
    ("nephrology", r"\b(ckd|kidney|renal|dialysis|esrd|nephro)"),
    ("oncology", r"\b(cancer|tumor|tumour|chemo|onco|carcinoma|leukemia|lymphoma|metasta)"),
    ("infectious_disease", r"\b(sepsis|infection|antibiotic|antifungal|antiviral|covid|hiv|tb|tuberculos)"),
    ("neurology", r"\b(stroke|tia|seizure|epileps|parkinson|alzheim|dement|migraine)"),
    # Phase 6F — psychiatry: capture quetiapine/Seroquel + common
    # psychiatric drug classes BEFORE the generic disease pattern below.
    ("psychiatry", r"\b(depress|anxiety|bipolar|schizophren|ptsd|ocd|adhd|"
     r"quetiapine|seroquel|olanzapine|zyprexa|risperidone|risperdal|"
     r"aripiprazole|abilify|clozapine|clozaril|ziprasidone|geodon|"
     r"lurasidone|latuda|atypical antipsychotic|second[-\s]generation antipsychotic)"),
    ("gastroenterology", r"\b(gerd|ibd|crohn|ulcerative|hepati|cirrho|pancreatit|gastrointestinal bleeding|nsaid)"),
    ("hematology", r"\b(anemia|leukemia|lymphoma|thrombocyto|coagul|dvt|pe|vte|bleed|warfarin|apixaban|doac)"),
    ("rheumatology", r"\b(rheumat|lupus|arthritis|gout|spondyl|vasculit)"),
    # Phase 6F — dermatology: psoriasis + topical/biologic terms commonly
    # appearing in derm questions. Listed AFTER rheumatology so
    # "psoriatic arthritis" is still rheum, but plain "psoriasis" wins
    # for derm.
    ("dermatology", r"\b(psoriasis|plaque psoriasis|eczema|atopic dermatitis|"
     r"dermatology|dermatologic|cutaneous|topical corticosteroid|"
     r"phototherapy|adalimumab|etanercept|infliximab|secukinumab|"
     r"ixekizumab|ustekinumab|guselkumab|risankizumab|tildrakizumab|"
     r"brodalumab|methotrexate|apremilast|otezla)"),
]


def _infer_specialty(q: str) -> str:
    s = q.lower()
    for label, patt in _SPECIALTY_PATTERNS:
        if re.search(patt, s):
            return label
    return "primary_care"


# --------------------------------------------------------------------------
# PICO extraction
# --------------------------------------------------------------------------
_PUNCT_TRIM = " \t\n\r,.;:!?"
_LEADING_STOP = {"the", "a", "an", "of", "in", "to", "for", "and"}


def _trim_span(s: str) -> str:
    s = s.strip(_PUNCT_TRIM)
    # strip leading articles
    parts = s.split()
    while parts and parts[0].lower() in _LEADING_STOP:
        parts.pop(0)
    return " ".join(parts).strip()


def _orig_slice(text: str, start: int, end: int) -> str:
    """Return the case-preserved slice from the original text."""
    return _trim_span(text[start:end])


def _split_outcomes(s: str) -> list[str]:
    if not s:
        return []
    # Split on " and " or commas, normalize.
    parts = re.split(r"\s*(?:,|\sand\s|/)\s*", s, flags=re.IGNORECASE)
    out: list[str] = []
    for p in parts:
        t = _trim_span(p)
        if t and t.lower() not in out:
            out.append(t)
    return out


# Population patterns
_POP_PATTERNS = [
    # "in adults with X" / "in patients with X" / "in children with X"
    re.compile(
        r"\bin\s+((?:low-risk|high-risk|elderly|pregnant|adolescent|geriatric|pediatric|critically\s+ill)?\s*"
        r"(?:adults?|children|patients|women|men|infants|adolescents|individuals|persons|subjects)"
        r"(?:\s+with\s+[^,?.]{2,80}?)?)\s*(?:,|[.?]|$)",
        re.IGNORECASE,
    ),
    # "in patients on long-term X"
    re.compile(
        r"\bin\s+(patients\s+on\s+[^,?.]{2,80}?)(?:,|[.?]|$)",
        re.IGNORECASE,
    ),
    # "among adults with X"
    re.compile(
        r"\bamong\s+([^,?.]{2,100}?)(?:,|[.?]|$)",
        re.IGNORECASE,
    ),
]


def _extract_population(text: str) -> tuple[str, tuple[int, int] | None]:
    for patt in _POP_PATTERNS:
        m = patt.search(text)
        if m:
            return _orig_slice(text, m.start(1), m.end(1)), (m.start(1), m.end(1))
    return "", None


# Diagnostic-accuracy pattern
_DX_ACC = re.compile(
    r"diagnostic accuracy of\s+([^,.?]+?)\s+for\s+([^,?.]+?)(?:\s+in\s+([^,?.]+))?(?:[.?]|$)",
    re.IGNORECASE,
)

# Harm/risk pattern
_HARM = re.compile(
    r"risk of\s+([^,?.]+?)\s+compared\s+(?:with|to)\s+([^,?.]+?)(?:[.?]|$)",
    re.IGNORECASE,
)
# "in patients on X" exposure
_EXPOSURE = re.compile(
    r"\bin\s+patients\s+on\s+([^,?.]{2,80}?)(?:,|[.?]|$)",
    re.IGNORECASE,
)

# Comparison patterns — try most specific first.
_CMP_HOW_DOES = re.compile(
    r"how does\s+([^,?.]+?)\s+compare\s+(?:with|to)\s+([^,?.]+?)\s+for\s+([^?.]+?)(?:[.?]|$)",
    re.IGNORECASE,
)
_CMP_DO_OUTCOME = re.compile(
    r"\b(?:do(?:es)?|can|will)\s+([^,?.]+?)\s+(?:improve|reduce|prevent|increase|lower|raise|detect)\s+"
    r"([^?.]+?)\s+compared\s+(?:with|to)\s+([^,?.]+?)(?:[.?]|$)",
    re.IGNORECASE,
)
_CMP_XPLUSY_VS_Z = re.compile(
    r"([^,?.]+?\s+plus\s+[^,?.]+?)\s+(?:vs\.?|versus|v\.?|compared\s+(?:with|to))\s+([^,?.]+?)(?:[.?]|$)",
    re.IGNORECASE,
)
_CMP_X_VS_Y = re.compile(
    r"([\w][\w\-/\s.]*?)\s+(?:vs\.?|versus|v\.?|compared\s+(?:with|to))\s+([\w][\w\-/\s.]*?)(?:[.?]|$)",
    re.IGNORECASE,
)


def _extract_comparison(text: str) -> dict[str, Any]:
    """Return dict with intervention / comparator / outcomes_from_pattern."""
    out: dict[str, Any] = {"intervention": "", "comparator": "", "outcomes": []}
    m = _CMP_HOW_DOES.search(text)
    if m:
        out["intervention"] = _orig_slice(text, m.start(1), m.end(1))
        out["comparator"] = _orig_slice(text, m.start(2), m.end(2))
        out["outcomes"] = _split_outcomes(_orig_slice(text, m.start(3), m.end(3)))
        return out
    m = _CMP_DO_OUTCOME.search(text)
    if m:
        out["intervention"] = _orig_slice(text, m.start(1), m.end(1))
        out["outcomes"] = _split_outcomes(_orig_slice(text, m.start(2), m.end(2)))
        out["comparator"] = _orig_slice(text, m.start(3), m.end(3))
        return out
    m = _CMP_XPLUSY_VS_Z.search(text)
    if m:
        out["intervention"] = _orig_slice(text, m.start(1), m.end(1))
        out["comparator"] = _orig_slice(text, m.start(2), m.end(2))
        return out
    m = _CMP_X_VS_Y.search(text)
    if m:
        out["intervention"] = _orig_slice(text, m.start(1), m.end(1))
        out["comparator"] = _orig_slice(text, m.start(2), m.end(2))
        return out
    return out


# Limit intervention/comparator span length to avoid runaway captures.
_MAX_PICO_FIELD = 80


def _cap(s: str) -> str:
    s = (s or "").strip()
    if len(s) <= _MAX_PICO_FIELD:
        return s
    return s[:_MAX_PICO_FIELD].rstrip()


# --------------------------------------------------------------------------
# Key concepts / synonyms / drugs
# --------------------------------------------------------------------------
def _extract_key_concepts(q: str) -> list[str]:
    s = q.lower()
    # Allow digits so "sglt2" stays intact.
    tokens = re.findall(r"[a-z][a-z0-9\-/]+", s)
    stop = {
        "the", "a", "an", "of", "in", "for", "on", "to", "and", "or",
        "is", "are", "was", "were", "with", "as", "by", "at", "from",
        "what", "which", "who", "whom", "how", "why", "best", "evidence",
        "compared", "compare", "versus", "vs", "than", "do", "does", "did",
        "should", "could", "would", "be", "into", "out", "this", "that",
        "their", "them", "they", "we", "you", "i", "use", "no",
    }
    keep: list[str] = []
    for t in tokens:
        if len(t) > 2 and t not in stop and t not in keep:
            keep.append(t)
    bigrams: list[str] = []
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if a in stop or b in stop or len(a) <= 2 or len(b) <= 2:
            continue
        bg = f"{a} {b}"
        if bg not in bigrams:
            bigrams.append(bg)
    return (bigrams + keep)[:14]


def _expand_synonyms(concepts: list[str]) -> list[str]:
    out: list[str] = []
    for c in concepts:
        cl = c.lower()
        for k, vs in _SYNONYMS.items():
            if cl == k:
                for v in vs:
                    if v not in out:
                        out.append(v)
    return out


def _detect_drugs(q: str) -> list[str]:
    """Detect drug names + class display names. Preserves canonical casing."""
    s = q.lower()
    found_lower: list[str] = []
    found_display: list[str] = []
    for d in _ALL_DRUGS:
        if re.search(rf"\b{re.escape(d)}\b", s):
            if d not in found_lower:
                found_lower.append(d)
                found_display.append(d)
    # Class detection by drug membership.
    for cls, members in _DRUG_CLASSES.items():
        if any(d in found_lower for d in members):
            display = _CLASS_DISPLAY_NAMES.get(cls, cls)
            if display not in found_display:
                found_display.append(display)
    # Also detect class names directly mentioned (e.g. "SGLT2 inhibitors", "NSAIDs").
    for cls, display in _CLASS_DISPLAY_NAMES.items():
        if display.lower() in s and display not in found_display:
            found_display.append(display)
        # Common variants
        if cls == "sglt2_inhibitor" and re.search(r"\bsglt[ -]?2\b", q, re.IGNORECASE):
            if "SGLT2 inhibitors" not in found_display:
                found_display.append("SGLT2 inhibitors")
        if cls == "nsaid" and re.search(r"\bnsaids?\b", q, re.IGNORECASE):
            if "NSAIDs" not in found_display:
                found_display.append("NSAIDs")
    return found_display


# --------------------------------------------------------------------------
# Service
# --------------------------------------------------------------------------
_LLM_SYSTEM = (
    "You are a clinical librarian's structuring assistant. You DO NOT answer "
    "the clinical question. You DO NOT invent medical facts. You ONLY restate "
    "the question and extract PICO + question type. Never include patient "
    "identifiers.\n"
    "\n"
    "Return ONE flat JSON object with EXACTLY these top-level keys. Do NOT "
    "nest them under a 'PICO' (or any other) object, and do NOT rename them:\n"
    '  "normalized_question": string (the question restated in one sentence),\n'
    '  "clinical_question_type": one of treatment|diagnosis|prognosis|harm|'
    "screening|prevention|etiology|guideline|epidemiology|pharmacology|other,\n"
    '  "specialty": string,\n'
    '  "population": string — the bare patient group or condition ONLY '
    '(e.g. "colon cancer", "adults with sepsis"). NEVER a descriptive phrase '
    'about the question such as "diagnostic methods for X", "prognosis of X", '
    'or "potential causes of X" — put just the condition/patient here,\n'
    '  "intervention_or_exposure": string (the I/E),\n'
    '  "comparator": string (the C),\n'
    '  "outcomes": array of strings (the O),\n'
    '  "setting": string,\n'
    '  "time_horizon": string,\n'
    '  "key_concepts": array of strings,\n'
    '  "mesh_terms": array of strings,\n'
    '  "synonyms": array of strings,\n'
    '  "drug_names_and_classes": array of strings,\n'
    '  "exclusions": array of strings,\n'
    '  "rationale": string (brief note on how you extracted the fields).\n'
    "Every key MUST be present. Use an empty string \"\" or empty array [] for "
    "any field you cannot determine. String fields must be strings (never "
    "arrays); array fields must be arrays (never strings)."
)

# Concrete typed example sent alongside ``_LLM_SYSTEM`` as the schema hint so
# the model emits the flat ParsedPICO shape (strings vs arrays) rather than a
# freelanced ``{"PICO": {...}}`` nesting, which ``_validate_llm_output`` would
# otherwise have to salvage.
_LLM_SCHEMA_HINT: dict[str, Any] = {
    "normalized_question": "string",
    "clinical_question_type": (
        "treatment|diagnosis|prognosis|harm|screening|prevention|"
        "etiology|guideline|epidemiology|pharmacology|other"
    ),
    "specialty": "string",
    "population": "string",
    "intervention_or_exposure": "string",
    "comparator": "string",
    "outcomes": ["string"],
    "setting": "string",
    "time_horizon": "string",
    "key_concepts": ["string"],
    "mesh_terms": ["string"],
    "synonyms": ["string"],
    "drug_names_and_classes": ["string"],
    "exclusions": ["string"],
    "rationale": "string",
}


# Question-stem phrases the LLM (or deterministic extractor) sometimes folds
# INTO the population/condition, e.g. "prognosis of colon cancer". That yields
# an unmatchable primary anchor — no paper's title/abstract contains the literal
# phrase — which the ranker then caps out, so a perfectly valid question returns
# zero results. Strip the leading stem so the anchor is the bare clinical
# subject ("colon cancer"), which the Boolean compiler and MeSH resolver match.
# Adjectives/qualifiers that can precede the aspect noun in a descriptive lead
# ("DIAGNOSTIC methods for X", "POTENTIAL causes of X").
_STEM_ADJ = (
    r"(?:potential|various|possible|common|main|primary|secondary|underlying|"
    r"differential|diagnostic|therapeutic|preventive|preventative|clinical|"
    r"available|current|recommended|optimal|appropriate|effective|best|"
    r"standard|typical|usual|key|important|relevant|first[-\s]line|"
    r"second[-\s]line|the)"
)
# Aspect nouns that name the QUESTION facet rather than the clinical subject.
_STEM_NOUN = (
    r"(?:prognos\w*|diagnos\w*|treatments?|treating|manag\w*|therap(?:y|ies)|"
    r"prevent\w*|screen\w*|workups?|work[-\s]ups?|evaluations?|assessments?|"
    r"caus\w*|etiolog\w*|aetiolog\w*|pathophysiolog\w*|epidemiolog\w*|"
    r"incidence|prevalence|complications?|methods?|methodolog(?:y|ies)|"
    r"strateg(?:y|ies)|options?|approach(?:es)?|factors?|natural\s+history|"
    r"outcomes?|survival|mortality|role|risks?)"
)
# A leading descriptive stem = [<=3 adjectives] [1-3 aspect nouns] (of|for|...).
# The LLM often folds one of these into the population/condition ("prognosis of
# colon cancer", "Diagnostic methods for pulmonary embolism", "Potential causes
# of CKD"), producing an unmatchable anchor that caps the whole result pool to
# zero. Stripping it exposes the bare clinical subject.
_QUESTION_STEM_RE = re.compile(
    r"^(?:" + _STEM_ADJ + r"\s+){0,3}"
    r"(?:" + _STEM_NOUN + r"\s+){1,3}"
    r"(?:of|for|to|in|with|regarding|about)\s+",
    re.IGNORECASE,
)
# Optional leading population framing to peel off before the stem check so a
# stem buried after it ("patients with prognosis of X") is still exposed.
_POP_LEAD_RE = re.compile(
    r"^(?:in\s+)?(?:patients?|adults?|children|people|persons?|individuals?|"
    r"subjects?)\s+with\s+",
    re.IGNORECASE,
)


def _strip_question_stem(text: str) -> str:
    """Remove a leading interrogative stem ("prognosis of", "risk of", ...) that
    was mistakenly folded into a condition/population string. Idempotent — a
    no-op when the text is already a bare clinical subject. Preserves an
    optional "patients with " framing."""
    t = (text or "").strip()
    prefix = ""
    lead = _POP_LEAD_RE.match(t)
    if lead:
        prefix = t[:lead.end()]
        t = t[lead.end():].strip()
    for _ in range(2):  # handle a rare double stem
        m = _QUESTION_STEM_RE.match(t)
        if not m:
            break
        t = t[m.end():].strip()
    return (prefix + t).strip() or (text or "").strip()


class ClinicalQuestionParserService:
    async def parse(self, sanitized_question: str) -> dict[str, Any]:
        q_raw = (sanitized_question or "").strip()
        # Phase 5A — deterministic spelling/spacing normalization runs FIRST.
        # PHI guard has already executed in the route handler. The original
        # question is preserved by the caller; we only work on the normalized
        # copy.
        norm_result = spelling_normalizer.normalize(q_raw)
        q = norm_result.normalized
        spelling_corrections = list(norm_result.corrections)
        # Phase 5F — clinical shorthand normalization runs AFTER spelling
        # (so ``cholesterole → cholesterol`` already happened) and BEFORE
        # the LLM PICOTT extractor. The substitutions list is preserved
        # on the parsed output so the transparency panel can render
        # ``Interpreted 'glp1s' as 'GLP-1 receptor agonists'.``.
        try:
            from services.clinical_shorthand_normalizer import (  # noqa: WPS433
                clinical_shorthand_normalizer,
            )
            shorthand_result = clinical_shorthand_normalizer.normalize(q)
            q = shorthand_result.normalized
            shorthand_substitutions = list(shorthand_result.substitutions)
        except Exception:  # noqa: BLE001
            shorthand_substitutions = []
        normalized = re.sub(r"\s+", " ", q)
        # The picott_extractor expects the *raw* sanitized question (it will
        # re-normalize internally so it can record the spelling corrections).
        # IMPORTANT: ``q_raw`` MUST be carried via the call stack (not via
        # ``self``). The parser is a module-level singleton and ``parse()``
        # awaits an LLM call below — storing per-call state on ``self`` would
        # let a concurrent ``parse()`` call clobber it, producing a PICOTT
        # built from a DIFFERENT session's question. See
        # ``tests/test_parser_concurrency.py`` for the regression.

        # ---- LLM path (optional enrichment) ----
        if llm_provider.is_configured():
            result = await llm_provider.complete_json(_LLM_SYSTEM, q, _LLM_SCHEMA_HINT)
            validated = self._validate_llm_output(result)
            if validated is not None:
                validated["clinical_question_type"] = _canonicalize_qtype(
                    validated.get("clinical_question_type", "other")
                )
                # The LLM occasionally folds the question stem into the
                # population/intervention (e.g. "prognosis of colon cancer",
                # "treatment options for stage 4 prostate cancer"); strip it so
                # the primary anchor is the bare subject and actually matches.
                validated["population"] = _strip_question_stem(
                    validated.get("population") or ""
                )
                validated["intervention_or_exposure"] = _strip_question_stem(
                    validated.get("intervention_or_exposure") or ""
                )
                # A "treatment for X" question has no *specific* intervention —
                # if the stripped intervention just restates the condition, it
                # is not a real drug/exposure. Clear it so it can't become a
                # bogus ``drug_or_drug_class`` anchor that no article matches
                # (the condition anchor then drives retrieval).
                _iv = (validated.get("intervention_or_exposure") or "").strip().lower()
                _pop = (validated.get("population") or "").strip().lower()
                if _iv and _pop and (_iv == _pop or _iv in _pop or _pop in _iv):
                    validated["intervention_or_exposure"] = ""
                validated["rationale"] = (validated.get("rationale") or "") + " | source=llm"
                # Final Pydantic gate
                try:
                    legacy_out = ParsedPICO(**validated).model_dump()
                    return self._attach_picott(legacy_out, q, spelling_corrections, q_raw, shorthand_substitutions)
                except Exception:  # noqa: BLE001
                    pass  # fall through to deterministic

        # ---- Deterministic fallback (canonical) ----
        qtype = _canonicalize_qtype(_classify_type(q))
        specialty = _infer_specialty(q)
        key_concepts = _extract_key_concepts(q)
        synonyms = _expand_synonyms(key_concepts)
        drugs = _detect_drugs(q)

        # Population
        population, _ = _extract_population(q)
        population = _strip_question_stem(population)

        # Diagnostic accuracy short-circuit
        intervention = ""
        comparator = ""
        outcomes: list[str] = []
        rationale_extras: list[str] = []
        m_dx = _DX_ACC.search(q)
        if m_dx:
            intervention = _orig_slice(q, m_dx.start(1), m_dx.end(1))
            cond = _orig_slice(q, m_dx.start(2), m_dx.end(2))
            if m_dx.group(3):
                population = _orig_slice(q, m_dx.start(3), m_dx.end(3)) or population
            if cond and "for " not in cond:
                if cond not in outcomes:
                    outcomes.append(f"diagnostic accuracy for {cond}".strip())
                else:
                    outcomes.append("diagnostic accuracy")
            else:
                outcomes.append("diagnostic accuracy")
            comparator = ""  # diagnostic studies — reference standard implicit; leave empty.
            rationale_extras.append("dx_accuracy")
        else:
            # Harm pattern
            m_harm = _HARM.search(q)
            if m_harm:
                hout = _orig_slice(q, m_harm.start(1), m_harm.end(1))
                if hout:
                    outcomes.extend(_split_outcomes(hout))
                comparator = _orig_slice(q, m_harm.start(2), m_harm.end(2))
                # Try "in patients on X" exposure for intervention
                m_exp = _EXPOSURE.search(q)
                if m_exp:
                    intervention = _orig_slice(q, m_exp.start(1), m_exp.end(1))
                rationale_extras.append("harm_pattern")
            else:
                # Generic comparison patterns
                cmp_out = _extract_comparison(q)
                intervention = cmp_out.get("intervention") or ""
                comparator = cmp_out.get("comparator") or ""
                pattern_outcomes = cmp_out.get("outcomes") or []
                for o in pattern_outcomes:
                    if o not in outcomes:
                        outcomes.append(o)
                if intervention or comparator:
                    rationale_extras.append("comparison_pattern")

        # Common outcome phrases (light supplement, not authoritative)
        for w in [
            "mortality", "all-cause mortality", "cardiovascular mortality",
            "hospitalization", "rehospitalization", "stroke", "bleeding",
            "major bleeding", "stroke prevention", "remission", "response",
            "recurrence", "progression", "quality of life", "hba1c",
            "blood pressure", "ldl", "adverse events",
            "gastrointestinal bleeding", "major gastrointestinal bleeding",
        ]:
            if re.search(rf"\b{re.escape(w)}\b", q, re.IGNORECASE):
                # case-preserved from the question
                m = re.search(rf"\b{re.escape(w)}\b", q, re.IGNORECASE)
                if m and m.group(0) not in outcomes:
                    outcomes.append(m.group(0))

        # Setting
        setting = ""
        if re.search(r"\b(icu|intensive care)\b", q, re.IGNORECASE):
            setting = "icu"
        elif re.search(r"\b(outpatient|primary care|clinic)\b", q, re.IGNORECASE):
            setting = "outpatient"
        elif re.search(r"\b(emergency|ed|er)\b", q, re.IGNORECASE):
            setting = "emergency"
        elif re.search(r"\b(inpatient|hospitaliz)\b", q, re.IGNORECASE):
            setting = "inpatient"

        # Time horizon
        time_horizon = ""
        m3 = re.search(r"\b(\d{1,2})\s*(week|weeks|month|months|year|years)\b", q, re.IGNORECASE)
        if m3:
            time_horizon = f"{m3.group(1)} {m3.group(2)}"

        # Apply caps + case normalization
        intervention = _cap(intervention)
        comparator = _cap(comparator)
        population = _cap(population)

        # Confidence note in rationale
        notes: list[str] = ["source=deterministic_fallback"]
        if rationale_extras:
            notes.append("extractors=" + ",".join(rationale_extras))
        if not comparator:
            notes.append("comparator not confidently extracted")
        if not intervention:
            notes.append("intervention not confidently extracted")
        rationale = " | ".join(notes)

        result_dict: dict[str, Any] = {
            "normalized_question": normalized,
            "clinical_question_type": qtype,
            "specialty": specialty,
            "population": population,
            "intervention_or_exposure": intervention,
            "comparator": comparator,
            "outcomes": outcomes,
            "setting": setting,
            "time_horizon": time_horizon,
            "key_concepts": key_concepts,
            "mesh_terms": [],
            "synonyms": synonyms,
            "drug_names_and_classes": drugs,
            "exclusions": [],
            "rationale": rationale,
        }
        # Final validation through Pydantic — guarantees Literal enum.
        legacy_out = ParsedPICO(**result_dict).model_dump()
        return self._attach_picott(legacy_out, q, spelling_corrections, q_raw, shorthand_substitutions)

    def _attach_picott(
        self, legacy_out: dict[str, Any], normalized_q: str,
        spelling_corrections: list[dict[str, Any]],
        raw_q: str,
        shorthand_substitutions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Append Phase 5A ``picott_json`` + ``spelling_corrections`` +
        Phase 5F ``shorthand_substitutions`` to the legacy parser output
        without breaking BC. Existing callers that accept the ParsedPICO
        shape ignore the extra keys.

        ``raw_q`` MUST be passed in by the caller — the parser is a
        module-level singleton and storing this on ``self`` would race
        across concurrent ``parse()`` calls (P0 corruption bug fixed in
        the same change that introduced this parameter).
        """
        try:
            picott = picott_extractor.build(raw_q or normalized_q, legacy_out)
        except Exception:  # noqa: BLE001
            picott = None
        merged = dict(legacy_out)
        if picott:
            merged["picott_json"] = picott
        if spelling_corrections:
            merged["spelling_corrections"] = list(spelling_corrections)
        if shorthand_substitutions:
            # Surfaces on the session and the transparency panel.
            merged["shorthand_substitutions"] = list(shorthand_substitutions)
        return merged

    # Canonical ParsedPICO field partitioning, used to normalize LLM output.
    _STR_FIELDS = (
        "normalized_question", "clinical_question_type", "specialty",
        "population", "intervention_or_exposure", "comparator",
        "setting", "time_horizon", "rationale",
    )
    _LIST_FIELDS = (
        "outcomes", "key_concepts", "mesh_terms", "synonyms",
        "drug_names_and_classes", "exclusions",
    )
    # Sub-keys a nested ``PICO`` object commonly uses, mapped to our flat keys.
    _NESTED_ALIASES = {
        "population": ("population", "patient", "patients", "problem", "p", "P"),
        "intervention_or_exposure": (
            "intervention_or_exposure", "intervention", "exposure",
            "index_test", "i", "I", "e", "E",
        ),
        "comparator": ("comparator", "comparison", "control", "c", "C"),
        "outcomes": ("outcomes", "outcome", "o", "O"),
        "setting": ("setting", "context", "s", "S"),
        "time_horizon": ("time_horizon", "time", "timing", "timeframe", "t", "T"),
    }

    @staticmethod
    def _validate_llm_output(d: Any) -> dict[str, Any] | None:
        """Normalize the LLM's JSON into the flat ParsedPICO shape.

        The model frequently freelances a nested ``{"PICO": {...}, "question":
        ..., "question_type": ...}`` shape or single-key aliases. Rather than
        discard any answer that isn't already perfectly flat (which forced the
        deterministic fallback 100% of the time), salvage what we can: flatten a
        nested PICO object, resolve common aliases, coerce list/string types,
        and backfill missing keys with safe defaults. ``ParsedPICO(extra=ignore)``
        is the final gate, so an over-permissive result here still can't produce
        an invalid model.
        """
        if not isinstance(d, dict):
            return None
        cls = ClinicalQuestionParserService
        d = dict(d)  # shallow copy — never mutate the caller's object

        # --- Flatten a nested PICO container, if present. ---
        nested: dict[str, Any] | None = None
        for nk in ("PICO", "pico", "pico_elements", "PICOTT", "picott", "elements"):
            if isinstance(d.get(nk), dict):
                nested = d.pop(nk)
                break
        if nested is not None:
            for canon, aliases in cls._NESTED_ALIASES.items():
                if d.get(canon) not in (None, "", []):
                    continue
                for a in aliases:
                    if a in nested and nested[a] not in (None, "", []):
                        d[canon] = nested[a]
                        break

        # --- Resolve top-level aliases the model uses for our keys. ---
        _top_aliases = {
            "normalized_question": ("question", "normalized", "restated_question", "restated"),
            "clinical_question_type": ("question_type", "type_of_question", "type"),
            "intervention_or_exposure": ("intervention", "exposure", "index_test"),
            "outcomes": ("outcome",),
            "drug_names_and_classes": ("drugs", "drug_names", "medications"),
        }
        for canon, aliases in _top_aliases.items():
            if d.get(canon) not in (None, "", []):
                continue
            for a in aliases:
                if d.get(a) not in (None, "", []):
                    d[canon] = d[a]
                    break

        # --- Coerce string fields (a list -> comma-joined; anything -> str). ---
        for k in cls._STR_FIELDS:
            v = d.get(k)
            if v is None:
                d[k] = ""
            elif isinstance(v, str):
                d[k] = v.strip()
            elif isinstance(v, list):
                d[k] = ", ".join(str(x).strip() for x in v if str(x).strip())
            else:
                d[k] = str(v)

        # --- Coerce list fields (a string -> single-item list). ---
        for k in cls._LIST_FIELDS:
            v = d.get(k)
            if v is None:
                d[k] = []
            elif isinstance(v, str):
                d[k] = [v.strip()] if v.strip() else []
            elif isinstance(v, list):
                d[k] = [str(x).strip() for x in v if str(x).strip()]
            else:
                return None

        # --- Require a usable core; else let the deterministic parser win. ---
        if not (d.get("normalized_question")
                or d.get("population")
                or d.get("intervention_or_exposure")
                or d.get("outcomes")):
            return None
        return d


clinical_question_parser = ClinicalQuestionParserService()
