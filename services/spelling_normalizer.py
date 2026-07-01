"""Phase 5A — Deterministic medical spelling + spacing normalization.

Runs AFTER PHI guard, BEFORE clinical question parsing or search planning.
Records a list of ``SpellingCorrection`` so the UI can show
``Interpreted "subarchnoid" as "subarachnoid."``.

Rules:
- Never modifies the persisted ``original_question``. Only the normalized
  copy is rewritten.
- Conservative: high-confidence variant pairs (UK↔US, common single-typo
  pathology names) are auto-corrected. Drug names are NOT auto-corrected
  unless the typo has only one plausible normalization in our curated map.
- Token-aware: case-preserving and word-boundary aware. Multi-token
  variants (e.g. "non valvular" → "nonvalvular") are matched as phrases.
- The function is pure and synchronous — no LLM call, no external API.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Variant pair tables — single source of truth.
# Each entry: variant → canonical, with explicit (confidence, reason).
# ---------------------------------------------------------------------------

# UK↔US spelling pairs (we normalize to US spelling because PubMed/MeSH and
# most clinical content use US spelling). Confidence = 1.0 — these are
# closed-set lexical pairs and never ambiguous.
UK_US_PAIRS: dict[str, str] = {
    "haemorrhage": "hemorrhage",
    "haemorrhages": "hemorrhages",
    "haemorrhagic": "hemorrhagic",
    "haemodynamic": "hemodynamic",
    "haematology": "hematology",
    "haematological": "hematological",
    "haematocrit": "hematocrit",
    "anaemia": "anemia",
    "anaemic": "anemic",
    "ischaemia": "ischemia",
    "ischaemic": "ischemic",
    "ischaemias": "ischemias",
    "oesophagus": "esophagus",
    "oesophageal": "esophageal",
    "paediatric": "pediatric",
    "paediatrics": "pediatrics",
    "paediatrician": "pediatrician",
    "tumour": "tumor",
    "tumours": "tumors",
    "fibre": "fiber",
    "fibres": "fibers",
    "behaviour": "behavior",
    "behavioural": "behavioral",
    "oedema": "edema",
    "oedematous": "edematous",
    "leukaemia": "leukemia",
    "leukaemic": "leukemic",
    "lymphoedema": "lymphedema",
    "diarrhoea": "diarrhea",
    "diarrhoeal": "diarrheal",
    "gastroenteritis": "gastroenteritis",  # identity (kept for completeness)
    "sulphate": "sulfate",
    "sulphur": "sulfur",
    "caesarean": "cesarean",
    "foetus": "fetus",
    "foetal": "fetal",
    "orthopaedic": "orthopedic",
    "orthopaedics": "orthopedics",
    "anaesthesia": "anesthesia",
    "anaesthetic": "anesthetic",
    "anaesthesiology": "anesthesiology",
    "homoeopathic": "homeopathic",
    "favour": "favor",
    "favourable": "favorable",
    "centre": "center",
    "litre": "liter",
    "millilitre": "milliliter",
    "millimetre": "millimeter",
    "metre": "meter",
}

# Single-typo pathology / anatomy variants. Confidence 1.0 because each one
# has only one plausible canonical (we deliberately do NOT include "ambien"
# or other shorter ambiguous strings).
PATHOLOGY_TYPO_PAIRS: dict[str, str] = {
    "subarchnoid": "subarachnoid",
    "subarachoid": "subarachnoid",
    "subarachnid": "subarachnoid",
    "subaracnoid": "subarachnoid",
    "subarchnoids": "subarachnoids",
    "alzheimers": "alzheimer's",
    "parkinsons": "parkinson's",
    "diabeties": "diabetes",
    "diabetis": "diabetes",
    "pnuemonia": "pneumonia",
    "pneumoniae": "pneumonia",  # genitive often misspelled
    "myocadial": "myocardial",
    "myocaridal": "myocardial",
    "myocaridial": "myocardial",
    "athersclerosis": "atherosclerosis",
    "atheroscelosis": "atherosclerosis",
    "atrial fibrilation": "atrial fibrillation",
    "atrial fibrilliation": "atrial fibrillation",
    "atrial fibrillaton": "atrial fibrillation",
    "perferation": "perforation",
    "thrombosys": "thrombosis",
    "trombosis": "thrombosis",
    "asma": "asthma",
    "asthsma": "asthma",
    "tubercolosis": "tuberculosis",
    "intracranal": "intracranial",
    "intracerebal": "intracerebral",
    "vasocpasm": "vasospasm",
    "vasopasm": "vasospasm",
    "vasospams": "vasospasm",
    "aneurismal": "aneurysmal",
    "anuerysmal": "aneurysmal",
    "anuerysm": "aneurysm",
    "anurysm": "aneurysm",
    "embolus": "embolism",
    # Phase J5 — closed-set additive typo pairs for high-confidence
    # clinical lexical errors. Each entry has a single plausible
    # canonical (no ambiguity, no broad-English autocorrect). New
    # entries here flow through the same word-boundary-aware,
    # case-preserving replacement helper as the rest of the table.
    "repalcement": "replacement",
    "repalce": "replace",
    "repalces": "replaces",
    "repalcing": "replacing",
    "replacment": "replacement",
    "replcement": "replacement",
    "suplement": "supplement",
    "supplemnt": "supplement",
    "suplementation": "supplementation",
    "supplemntation": "supplementation",
    "magneisum": "magnesium",
    "magnesim": "magnesium",
    "potasium": "potassium",
    "calicum": "calcium",
    "anticoagualtion": "anticoagulation",
    "anticoaglation": "anticoagulation",
    "anticoagulaiton": "anticoagulation",
    "perioperative": "perioperative",  # identity (UK variant insurance)
    "tracheostomy": "tracheostomy",  # identity (anchors lexicon)
    "tracheotomy": "tracheostomy",  # canonical synonym
    "pacemkaer": "pacemaker",
    "pacemeker": "pacemaker",
    "pacmeaker": "pacemaker",
    "pacmaker": "pacemaker",
    "defribillator": "defibrillator",
    "defibrilator": "defibrillator",
    "defribrillator": "defibrillator",
    "bronchscopy": "bronchoscopy",
    "bronchospy": "bronchoscopy",
}

# Common multi-word normalizations to single-token canonical forms used in
# MeSH/keyword searches.
PHRASE_NORMALIZATIONS: dict[str, str] = {
    "non valvular": "nonvalvular",
    "non-valvular": "nonvalvular",
    "non valvular atrial fibrillation": "nonvalvular atrial fibrillation",
    "deep venous thrombosis": "deep vein thrombosis",
    "high blood pressure": "hypertension",
}

# Token spacing — collapse multiple spaces, normalize curly quotes, etc.
_WHITESPACE_RE = re.compile(r"\s+")
_SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'",
    "\u2013": "-", "\u2014": "-",
}


@dataclass
class SpellingNormalizationResult:
    """Pure-data result. Each correction is recorded with the original ↔
    normalized pair, confidence, and a one-word reason code for the UI.
    """

    normalized: str
    corrections: list[dict]  # list of SpellingCorrection-shaped dicts


def _normalize_quotes_and_spacing(s: str) -> str:
    for k, v in _SMART_QUOTES.items():
        s = s.replace(k, v)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def _apply_single_token_pairs(
    text: str,
    pairs: dict[str, str],
    *,
    reason: str,
    confidence: float,
    seen: set[str],
    corrections: list[dict],
) -> str:
    """Replace word-boundary token matches case-insensitively. Records each
    distinct correction once per call."""
    out = text
    for variant, canonical in pairs.items():
        # Word-boundary aware replace, case-insensitive.
        patt = re.compile(rf"\b{re.escape(variant)}\b", re.IGNORECASE)
        match = patt.search(out)
        if not match:
            continue
        key = variant.lower()
        if key not in seen:
            seen.add(key)
            corrections.append({
                "original": variant,
                "normalized": canonical,
                "confidence": confidence,
                "reason": reason,
            })
        # Replace using canonical, preserving the leading-cap if the original
        # had it (so "Subarchnoid" → "Subarachnoid", not "subarachnoid").
        def _repl(m: re.Match) -> str:
            tok = m.group(0)
            if tok[:1].isupper() and not tok.isupper():
                return canonical[:1].upper() + canonical[1:]
            if tok.isupper():
                return canonical.upper()
            return canonical
        out = patt.sub(_repl, out)
    return out


def _apply_phrase_pairs(
    text: str,
    pairs: dict[str, str],
    *,
    seen: set[str],
    corrections: list[dict],
) -> str:
    out = text
    # Match longest phrases first to avoid partial overlaps.
    for variant in sorted(pairs.keys(), key=len, reverse=True):
        canonical = pairs[variant]
        patt = re.compile(rf"\b{re.escape(variant)}\b", re.IGNORECASE)
        if patt.search(out):
            key = variant.lower()
            if key not in seen:
                seen.add(key)
                corrections.append({
                    "original": variant,
                    "normalized": canonical,
                    "confidence": 1.0,
                    "reason": "phrase_normalization",
                })
            out = patt.sub(canonical, out)
    return out


def normalize(text: str) -> SpellingNormalizationResult:
    """Normalize ``text`` deterministically. Returns the normalized string
    plus the list of corrections actually applied (most-significant first).

    The function is intentionally conservative — only canonical pairs from
    the curated tables above are applied. No fuzzy matching, no Levenshtein,
    no LLM call.
    """
    if not text:
        return SpellingNormalizationResult(normalized="", corrections=[])
    s = _normalize_quotes_and_spacing(text)
    corrections: list[dict] = []
    seen: set[str] = set()
    # Phrase normalizations first (so "non valvular" → "nonvalvular" before
    # token rules run).
    s = _apply_phrase_pairs(s, PHRASE_NORMALIZATIONS, seen=seen, corrections=corrections)
    # Pathology typos (most-specific clinical errors).
    s = _apply_single_token_pairs(
        s, PATHOLOGY_TYPO_PAIRS,
        reason="spelling_correction", confidence=1.0,
        seen=seen, corrections=corrections,
    )
    # UK↔US spelling.
    s = _apply_single_token_pairs(
        s, UK_US_PAIRS,
        reason="uk_to_us", confidence=1.0,
        seen=seen, corrections=corrections,
    )
    return SpellingNormalizationResult(normalized=s, corrections=corrections)


class SpellingNormalizerService:
    """Stateless service wrapper so the rest of the codebase has a single
    injection point. ``normalize`` is a pure function — exposed here so
    callers can mock the service in unit tests."""

    def normalize(self, text: str) -> SpellingNormalizationResult:
        return normalize(text)


spelling_normalizer = SpellingNormalizerService()
