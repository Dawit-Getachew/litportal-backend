"""Phase 5A — Concept expansion dictionaries (deterministic, contextual).

Lookup tables that expand a normalized concept into a list of safe synonyms
+ MeSH-style terms. Only includes:

- Closed-set acronym↔long-form pairs (SAH↔subarachnoid hemorrhage).
- Spelling variants (UK↔US already handled by spelling_normalizer; these
  are clinical multi-word equivalents).
- Drug generic↔class equivalents (curated subset, NOT blind expansion).
- Disease subtype variants (HFpEF↔heart failure with preserved ejection
  fraction).
- Specialty packs: an opening pack for neurocritical care (SAH/vasospasm/
  DCI/milrinone) used by the milrinone-SAH benchmark case.

Each public function returns a *new* deduped list and never mutates inputs.
"""
from __future__ import annotations

import re
from typing import Iterable


# ---------------------------------------------------------------------------
# Core acronym/synonym pairs.
# Symmetric: both forward and reverse are produced via ``_reverse_map``.
# ---------------------------------------------------------------------------
ACRONYM_PAIRS: dict[str, list[str]] = {
    # Cardiology
    "atrial fibrillation": ["AF", "AFib", "A-fib", "NVAF", "nonvalvular atrial fibrillation"],
    "myocardial infarction": ["MI", "STEMI", "NSTEMI", "heart attack"],
    "heart failure": ["HF", "CHF", "congestive heart failure"],
    "heart failure with preserved ejection fraction": ["HFpEF"],
    "heart failure with reduced ejection fraction": ["HFrEF"],
    "hypertension": ["HTN", "high blood pressure"],
    # Pulmonary / vascular thrombosis
    "pulmonary embolism": ["PE"],
    "deep vein thrombosis": ["DVT"],
    "venous thromboembolism": ["VTE"],
    "chronic obstructive pulmonary disease": ["COPD"],
    # Renal
    "chronic kidney disease": ["CKD"],
    "acute kidney injury": ["AKI"],
    "end-stage renal disease": ["ESRD"],
    # Endocrine
    "type 2 diabetes": ["T2DM", "type 2 diabetes mellitus", "NIDDM"],
    "type 1 diabetes": ["T1DM", "type 1 diabetes mellitus", "IDDM"],
    # Neuro / neurocritical care
    "subarachnoid hemorrhage": [
        "SAH", "aSAH",
        "aneurysmal subarachnoid hemorrhage", "aneurysmal SAH",
    ],
    "intracerebral hemorrhage": ["ICH"],
    "ischemic stroke": ["acute ischemic stroke", "AIS"],
    "stroke": ["cerebrovascular accident", "CVA"],
    "transient ischemic attack": ["TIA"],
    "delayed cerebral ischemia": ["DCI", "delayed ischemic neurological deficit", "DIND"],
    "cerebral vasospasm": ["vasospasm"],
    # Infectious
    "covid-19": ["COVID", "SARS-CoV-2", "coronavirus disease"],
    "human immunodeficiency virus": ["HIV"],
    "tuberculosis": ["TB"],
    # GI
    "inflammatory bowel disease": ["IBD"],
    "crohn's disease": ["crohns disease"],
    "ulcerative colitis": ["UC"],
    "gastroesophageal reflux disease": ["GERD"],
    # Phase 6F — dermatology / psoriasis.
    "psoriasis": [
        "plaque psoriasis", "chronic plaque psoriasis",
        "psoriasis vulgaris", "psoriatic disease",
        "moderate-to-severe psoriasis", "moderate to severe psoriasis",
        "severe psoriasis",
    ],
    "plaque psoriasis": ["psoriasis", "psoriasis vulgaris"],
}


def _reverse_map(forward: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for canonical, variants in forward.items():
        for v in variants:
            out.setdefault(v.lower(), []).append(canonical)
        # canonical itself can be expanded into its variants too
        out.setdefault(canonical.lower(), []).extend(variants)
    return out


_REVERSE = _reverse_map(ACRONYM_PAIRS)


# ---------------------------------------------------------------------------
# Drug generic + class equivalence (subset — curated, NOT all-drugs).
# ---------------------------------------------------------------------------
DRUG_GENERIC_TO_CLASS: dict[str, list[str]] = {
    "apixaban": ["DOAC", "direct oral anticoagulant", "factor Xa inhibitor"],
    "rivaroxaban": ["DOAC", "direct oral anticoagulant", "factor Xa inhibitor"],
    "edoxaban": ["DOAC", "direct oral anticoagulant", "factor Xa inhibitor"],
    "dabigatran": ["DOAC", "direct oral anticoagulant", "direct thrombin inhibitor"],
    "warfarin": ["VKA", "vitamin K antagonist"],
    "milrinone": ["phosphodiesterase 3 inhibitor", "PDE3 inhibitor", "inotrope"],
    "ibuprofen": ["NSAID"],
    "naproxen": ["NSAID"],
    "celecoxib": ["NSAID", "COX-2 inhibitor"],
    "diclofenac": ["NSAID"],
    "indomethacin": ["NSAID"],
    "atorvastatin": ["statin", "HMG-CoA reductase inhibitor"],
    "rosuvastatin": ["statin"],
    "simvastatin": ["statin"],
    "empagliflozin": ["SGLT2 inhibitor"],
    "dapagliflozin": ["SGLT2 inhibitor"],
    "semaglutide": ["GLP-1 agonist", "GLP-1 receptor agonist"],
    "liraglutide": ["GLP-1 agonist"],
    "metformin": ["biguanide"],
    "sacubitril/valsartan": ["ARNI"],
    "spironolactone": ["MRA", "mineralocorticoid receptor antagonist"],
    "eplerenone": ["MRA", "mineralocorticoid receptor antagonist"],
    "finerenone": ["nonsteroidal MRA"],
    # Phase 6F — atypical antipsychotic drug synonyms. Symmetric: the
    # _reverse_map will register Seroquel → quetiapine and brand variants.
    "quetiapine": [
        "Seroquel", "Seroquel XR", "quetiapine fumarate",
        "atypical antipsychotic", "second-generation antipsychotic",
    ],
    "olanzapine": ["Zyprexa", "atypical antipsychotic"],
    "risperidone": ["Risperdal", "atypical antipsychotic"],
    "aripiprazole": ["Abilify", "atypical antipsychotic"],
    "clozapine": ["Clozaril", "atypical antipsychotic"],
    "ziprasidone": ["Geodon", "atypical antipsychotic"],
    "lurasidone": ["Latuda", "atypical antipsychotic"],
    # Phase 6F — psoriasis biologics + systemic agents.
    "methotrexate": ["MTX"],
    "cyclosporine": ["cyclosporin", "ciclosporin"],
    "apremilast": ["Otezla", "PDE4 inhibitor"],
    "adalimumab": ["Humira", "TNF inhibitor"],
    "etanercept": ["Enbrel", "TNF inhibitor"],
    "infliximab": ["Remicade", "TNF inhibitor"],
    "secukinumab": ["Cosentyx", "IL-17 inhibitor"],
    "ixekizumab": ["Taltz", "IL-17 inhibitor"],
    "brodalumab": ["Siliq", "IL-17 inhibitor"],
    "guselkumab": ["Tremfya", "IL-23 inhibitor"],
    "risankizumab": ["Skyrizi", "IL-23 inhibitor"],
    "tildrakizumab": ["Ilumya", "IL-23 inhibitor"],
    "ustekinumab": ["Stelara", "IL-12/23 inhibitor"],
}


# ---------------------------------------------------------------------------
# Specialty packs — concept clusters with explicit context terms.
# Each pack contributes ranking concepts (boost-only, NOT must-have).
# ---------------------------------------------------------------------------

# Neurocritical care: SAH / vasospasm / DCI / milrinone
NEUROCRIT_PACK = {
    "concepts": [
        "subarachnoid hemorrhage", "SAH", "aSAH",
        "aneurysmal subarachnoid hemorrhage",
        "cerebral vasospasm", "vasospasm",
        "delayed cerebral ischemia", "DCI",
        "delayed ischemic neurological deficit", "DIND",
        "milrinone",
    ],
    "milrinone_routes": [
        "intravenous milrinone", "IV milrinone",
        "intra-arterial milrinone", "IA milrinone",
        "intra-arterial infusion",
        "intrathecal milrinone", "cisternal milrinone",
        "intracisternal milrinone",
    ],
    "dosing_terms": [
        "dose", "dosing", "infusion", "bolus", "continuous infusion",
        "loading dose", "maintenance dose", "regimen", "protocol",
        "mcg/kg/min", "microgram/kg/min", "mg/kg/min",
        "ug/kg/min", "ng/kg/min", "mg/kg",
    ],
}


def _dedupe_preserve_order(seq: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in seq:
        if not s:
            continue
        k = s.lower().strip()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def expand_concept(concept: str) -> list[str]:
    """Return ``[concept, ...synonyms]`` for a single concept. Conservative.

    Returns the input concept plus its known equivalents. Returns an empty
    list when called with an empty string.
    """
    c = (concept or "").strip()
    if not c:
        return []
    cl = c.lower()
    out: list[str] = [c]
    if cl in ACRONYM_PAIRS:
        out.extend(ACRONYM_PAIRS[cl])
    if cl in _REVERSE:
        out.extend(_REVERSE[cl])
    if cl in DRUG_GENERIC_TO_CLASS:
        out.extend(DRUG_GENERIC_TO_CLASS[cl])
    return _dedupe_preserve_order(out)


def expand_concepts(concepts: Iterable[str]) -> list[str]:
    out: list[str] = []
    for c in concepts:
        out.extend(expand_concept(c))
    return _dedupe_preserve_order(out)


# ---------------------------------------------------------------------------
# Specialty-pack matching — only fires when the question already mentions a
# trigger concept (e.g., "milrinone" + "SAH"/"subarachnoid"/"vasospasm").
# This prevents the neurocrit pack from polluting unrelated questions.
# ---------------------------------------------------------------------------

_NEUROCRIT_TRIGGERS = re.compile(
    r"\b(subarachnoid|sah|asah|vasospasm|dci|"
    r"delayed cerebral ischemia|delayed ischemic neurological deficit|"
    r"milrinone|nimodipine|aneurysmal)\b",
    re.IGNORECASE,
)


def applicable_specialty_packs(normalized_question: str) -> list[str]:
    """Return the names of specialty packs whose trigger fires on the
    normalized question text. Today only ``neurocritical_care`` is shipped."""
    out: list[str] = []
    if _NEUROCRIT_TRIGGERS.search(normalized_question or ""):
        out.append("neurocritical_care")
    return out


def neurocrit_ranking_terms() -> list[str]:
    """All ranking terms contributed by the neurocrit pack (concepts +
    dosing context). Routes are included only as concepts the ranker can
    boost on — they're not must-have."""
    return _dedupe_preserve_order(
        NEUROCRIT_PACK["concepts"]
        + NEUROCRIT_PACK["milrinone_routes"]
        + NEUROCRIT_PACK["dosing_terms"]
    )


def neurocrit_dosing_context_terms() -> list[str]:
    return list(NEUROCRIT_PACK["dosing_terms"])


# ---------------------------------------------------------------------------
# Imaging / diagnostic decoys for the milrinone-SAH negative-concept logic.
# The ranker uses these to penalize imaging-only papers UNLESS the article
# also mentions the milrinone drug name.
# ---------------------------------------------------------------------------
IMAGING_DECOY_TERMS = [
    "CT angiography", "CTA",
    "computed tomographic angiography",
    "bone-subtraction CTA", "bone subtraction",
    "digital subtraction angiography", "DSA",
    "transcranial Doppler",  # TCD is a vasospasm decision tool, but treatment
                              # questions about milrinone shouldn't be dominated
                              # by TCD-only papers.
    "MRA", "MR angiography",
    "perfusion imaging",
]
