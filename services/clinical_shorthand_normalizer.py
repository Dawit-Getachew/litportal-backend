"""Phase 5F — Clinical Shorthand Normalizer.

Sits between the spelling normalizer (Phase 5A) and the PICOTT extractor /
ontology mapper. Expands closed-set clinical shorthand (glp1s, ldl, a1c, …)
into the canonical concept names used by the rest of the pipeline.

Hard rules:
  - Runs AFTER the PHI guard (PHI guard runs first; this module never sees
    raw PHI text — its caller does).
  - The persisted ``original_question`` is NEVER mutated. Only the
    normalized copy is rewritten.
  - Context-disambiguation: tokens like ``MI`` are only expanded to
    ``myocardial infarction`` when no neuro context (SAH, subarachnoid,
    aSAH, …) is nearby in the same question.
  - Returns a list of ``ShorthandSubstitution`` records so the UI can
    render ``"Interpreted 'glp1s' as 'GLP-1 receptor agonists'."``.
  - Pure / deterministic — no LLM call, no external API, no I/O.

Output shape per substitution:
  {
    "original": "glp1s",
    "normalized": "GLP-1 receptor agonists",
    "confidence": 1.0,
    "reason": "drug_class_shorthand" | "lab_shorthand" | "acronym_expansion"
              | "procedure_acronym" | "context_disambiguated",
  }
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Shorthand → canonical mapping tables.
# Each entry is closed-set. Drugs / drug classes only.
# ---------------------------------------------------------------------------

# Drug classes (closed-set shorthand → canonical class name).
# Lowercased keys; matched against the word-boundary normalized question.
# Phase 5F: glp1s/glp1/glp-1ra/etc. all canonicalize to "GLP-1 receptor
# agonists" so the ontology mapper + parser can hit the cardiometabolic
# specialty pack.
DRUG_CLASS_SHORTHAND: dict[str, str] = {
    "glp1s": "GLP-1 receptor agonists",
    "glp-1s": "GLP-1 receptor agonists",
    "glp1": "GLP-1",
    "glp-1": "GLP-1",
    "glp1ra": "GLP-1 receptor agonist",
    "glp-1ra": "GLP-1 receptor agonist",
    "glp1-ra": "GLP-1 receptor agonist",
    "glp1 ra": "GLP-1 receptor agonist",
    "glp-1 ra": "GLP-1 receptor agonist",
    "sglt2i": "SGLT2 inhibitor",
    "sglt2is": "SGLT2 inhibitors",
    "sglt2": "SGLT2",
    "dpp4i": "DPP-4 inhibitor",
    "dpp-4i": "DPP-4 inhibitor",
    "dpp4": "DPP-4",
    "pcsk9i": "PCSK9 inhibitor",
    "doac": "DOAC",
    "doacs": "DOACs",
    "nsaid": "NSAID",
    "nsaids": "NSAIDs",
    "ace inhibitor": "ACE inhibitor",
    "ace inhibitors": "ACE inhibitors",
    "acei": "ACE inhibitor",
    "aceis": "ACE inhibitors",
    "arb": "ARB",
    "arbs": "ARBs",
    "arni": "ARNI",
    "arnis": "ARNIs",
    "ccb": "calcium channel blocker",
    "ccbs": "calcium channel blockers",
    "mra": "mineralocorticoid receptor antagonist",
    "mras": "mineralocorticoid receptor antagonists",
    "ppi": "proton pump inhibitor",
    "ppis": "proton pump inhibitors",
}

# Biomarker / lab shorthand.
LAB_SHORTHAND: dict[str, str] = {
    "ldl": "LDL-C / low-density lipoprotein cholesterol",
    "ldl-c": "LDL-C",
    "hdl": "HDL-C / high-density lipoprotein cholesterol",
    "hdl-c": "HDL-C",
    "tg": "triglycerides",
    "tgs": "triglycerides",
    "a1c": "hemoglobin A1c / HbA1c",
    "hba1c": "HbA1c",
    "hgba1c": "HbA1c",
    "lp(a)": "lipoprotein(a)",
    "lpa": "lipoprotein(a)",
    "apob": "apolipoprotein B",
    "apo-b": "apolipoprotein B",
    "tc": "total cholesterol",
    "fpg": "fasting plasma glucose",
    "fbg": "fasting blood glucose",
    "bmi": "body mass index",
    "egfr": "estimated glomerular filtration rate",
    "crp": "C-reactive protein",
    "hscrp": "high-sensitivity C-reactive protein",
    "hs-crp": "high-sensitivity C-reactive protein",
    "bp": "blood pressure",
    "sbp": "systolic blood pressure",
    "dbp": "diastolic blood pressure",
}

# Procedure / disease acronyms with little/no ambiguity.
PROCEDURE_ACRONYMS: dict[str, str] = {
    "cabg": "coronary artery bypass graft",
    "pci": "percutaneous coronary intervention",
    "tavr": "transcatheter aortic valve replacement",
    "tavi": "transcatheter aortic valve implantation",
    "icd": "implantable cardioverter-defibrillator",
    "crt": "cardiac resynchronization therapy",
    "ablation": "catheter ablation",
    "ards": "acute respiratory distress syndrome",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "vte": "venous thromboembolism",
    "afib": "atrial fibrillation",
    "a-fib": "atrial fibrillation",
    "af": "atrial fibrillation",
    "nvaf": "nonvalvular atrial fibrillation",
    "hfpef": "heart failure with preserved ejection fraction",
    "hfref": "heart failure with reduced ejection fraction",
    "hfmref": "heart failure with mildly reduced ejection fraction",
    "stemi": "ST-elevation myocardial infarction",
    "nstemi": "non-ST-elevation myocardial infarction",
    "acs": "acute coronary syndrome",
    "copd": "chronic obstructive pulmonary disease",
    "ckd": "chronic kidney disease",
    "aki": "acute kidney injury",
    "esrd": "end-stage renal disease",
    "t1dm": "type 1 diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "t1d": "type 1 diabetes",
    "t2d": "type 2 diabetes",
    "ibd": "inflammatory bowel disease",
    "uc": "ulcerative colitis",
    "gerd": "gastroesophageal reflux disease",
    "ich": "intracerebral hemorrhage",
    "tia": "transient ischemic attack",
    "dci": "delayed cerebral ischemia",
}

# Context-disambiguated acronyms — expansion depends on what else is in
# the question. Each entry maps (token, default_canonical) but the
# expansion is DROPPED when one of ``skip_if_context_matches`` fires.
CONTEXT_DISAMBIGUATED: dict[str, dict] = {
    # ``MI`` only expands to "myocardial infarction" when no neuro/SAH
    # context is nearby — otherwise ``MI`` may mean "mitral insufficiency"
    # or part of "MILRINONE" etc.
    "mi": {
        "canonical": "myocardial infarction",
        "skip_if_context_matches": [
            r"\bsubarachnoid\b", r"\bsah\b", r"\basah\b",
            r"\bvasospasm\b", r"\bmilrinone\b", r"\bdci\b",
            # mitral insufficiency
            r"\bmitral\s+insufficien",
            # "MI" used as Michigan / measurement unit / acronym for
            # institutional context.
            r"\bmichigan\b", r"\b\d+\s*mi\b",
        ],
        "reason": "context_disambiguated",
    },
    # ``SAH`` is unambiguously "subarachnoid hemorrhage" in clinical
    # phrasing.
    "sah": {
        "canonical": "subarachnoid hemorrhage",
        "skip_if_context_matches": [],
        "reason": "context_disambiguated",
    },
    # ``CR`` could mean controlled-release or complete response — DO
    # NOT auto-expand. Listed here only to document the decision.
}

# Spelling corrections in the shorthand layer (kept separate from the
# Phase 5A spelling table so we can add clinical-domain spellings
# without polluting the UK↔US table).
CLINICAL_SPELLING_PAIRS: dict[str, str] = {
    "cholesterole": "cholesterol",
    "hemorrage": "hemorrhage",
    "hemorrages": "hemorrhages",
    "haemorrage": "hemorrhage",
    "pneumonea": "pneumonia",
    "pneumona": "pneumonia",
    "myocardal": "myocardial",
    "anitcoagulant": "anticoagulant",
    "antiplatlet": "antiplatelet",
    "athrosclerosis": "atherosclerosis",
    "diabetus": "diabetes",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class ShorthandNormalizationResult:
    """Pure-data normalization result."""
    normalized: str
    substitutions: list[dict]  # closed-set ShorthandSubstitution dicts


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _word_boundary_pattern(token: str) -> re.Pattern[str]:
    """Build a case-insensitive word-boundary pattern that tolerates
    hyphen / dash / parenthesis variants in the user's typed token.
    For tokens that already contain a hyphen, the hyphen is required
    (we don't want ``glp 1`` to match ``glp1``)."""
    # Escape but allow internal hyphen to match hyphen (so "GLP-1" matches
    # "glp-1" only). Also escape parentheses for "lp(a)".
    return re.compile(rf"(?<![\w-]){re.escape(token)}(?![\w-])",
                      re.IGNORECASE)


def _apply_map(
    text: str,
    table: dict[str, str],
    *,
    reason: str,
    substitutions: list[dict],
    seen: set[str],
) -> str:
    out = text
    # Apply longest tokens first to avoid partial overlap (e.g. ``glp-1ra``
    # before ``glp-1``).
    for variant in sorted(table.keys(), key=len, reverse=True):
        canonical = table[variant]
        patt = _word_boundary_pattern(variant)
        if not patt.search(out):
            continue
        if variant.lower() not in seen:
            seen.add(variant.lower())
            substitutions.append({
                "original": variant,
                "normalized": canonical,
                "confidence": 1.0,
                "reason": reason,
            })
        out = patt.sub(canonical, out)
    return out


def _apply_context_disambiguated(
    text: str,
    *,
    substitutions: list[dict],
    seen: set[str],
) -> str:
    out = text
    for variant, cfg in CONTEXT_DISAMBIGUATED.items():
        patt = _word_boundary_pattern(variant)
        if not patt.search(out):
            continue
        if any(re.search(skip_re, out, re.IGNORECASE)
               for skip_re in cfg["skip_if_context_matches"]):
            # Context says don't expand — skip.
            continue
        canonical = cfg["canonical"]
        if variant.lower() not in seen:
            seen.add(variant.lower())
            substitutions.append({
                "original": variant,
                "normalized": canonical,
                "confidence": 0.95,
                "reason": cfg.get("reason", "context_disambiguated"),
            })
        out = patt.sub(canonical, out)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_shorthand(text: str) -> ShorthandNormalizationResult:
    """Expand clinical shorthand in ``text``. Returns the rewritten text
    and the list of substitutions actually applied. Pure / sync."""
    if not text:
        return ShorthandNormalizationResult(normalized="", substitutions=[])
    s = text
    substitutions: list[dict] = []
    seen: set[str] = set()
    # Order: clinical spelling first (so ``cholesterole`` is fixed before
    # any downstream layer sees it), then drug-class shorthand, then lab
    # shorthand, then procedure acronyms, then context-disambiguated.
    s = _apply_map(s, CLINICAL_SPELLING_PAIRS,
                   reason="clinical_spelling_correction",
                   substitutions=substitutions, seen=seen)
    s = _apply_map(s, DRUG_CLASS_SHORTHAND,
                   reason="drug_class_shorthand",
                   substitutions=substitutions, seen=seen)
    s = _apply_map(s, LAB_SHORTHAND,
                   reason="lab_shorthand",
                   substitutions=substitutions, seen=seen)
    s = _apply_map(s, PROCEDURE_ACRONYMS,
                   reason="procedure_acronym",
                   substitutions=substitutions, seen=seen)
    s = _apply_context_disambiguated(s, substitutions=substitutions, seen=seen)
    return ShorthandNormalizationResult(normalized=s, substitutions=substitutions)


class ClinicalShorthandNormalizerService:
    """Stateless service wrapper. ``normalize`` is pure."""

    def normalize(self, text: str) -> ShorthandNormalizationResult:
        return normalize_shorthand(text)


clinical_shorthand_normalizer = ClinicalShorthandNormalizerService()


__all__ = [
    "ShorthandNormalizationResult",
    "ClinicalShorthandNormalizerService",
    "clinical_shorthand_normalizer",
    "normalize_shorthand",
    "DRUG_CLASS_SHORTHAND",
    "LAB_SHORTHAND",
    "PROCEDURE_ACRONYMS",
    "CLINICAL_SPELLING_PAIRS",
    "CONTEXT_DISAMBIGUATED",
]
