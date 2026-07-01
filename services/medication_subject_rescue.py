"""Phase MedSafety-1 — deterministic medication subject rescue.

DETERMINISTIC. NO LLM. NO network. NO dosing recommendation.

The pipeline's primary-clinical-anchor derivation sometimes returns
``no_primary_clinical_entity`` for a clearly drug-centric dosing/safety
question when the drug is not in the ontology (e.g. dexmedetomidine /
Precedex). This module provides a *closed-set, lexicon-driven* rescue that
runs ONLY on that failure path: if the (normalized) question contains a
recognized medication (generic / brand / safe typo), it builds a
deterministic ``drug_or_drug_class`` primary anchor so the question can be
routed to evidence retrieval instead of ``awaiting_clinical_subject``.

IMPORTANT: this is evidence-retrieval support ONLY. It does NOT generate
any dosing recommendation, does not call an LLM, and does not bypass the
PHI guard (which runs FIRST, before the pipeline ever reaches here).

Sources scanned:
  1. ``services/data/medication_lexicon.json`` (curated gap set: ICU
     sedation / analgesia / antipsychotics / vasoactives + brands/typos).
  2. ``services/data/rxnorm_curated.json`` (existing single drugs + class
     members + their brand synonyms).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("clinevidence.medication_subject_rescue")

_HERE = Path(__file__).resolve().parent
_LEXICON_PATH = _HERE / "data" / "medication_lexicon.json"
_RXNORM_PATH = _HERE / "data" / "rxnorm_curated.json"

# Closed-set dosing / safety intent signals (presence routes to drug_safety).
_DOSING_SAFETY_RE = re.compile(
    r"\b(?:dose|doses|dosing|dosage|how\s+much|titrat\w*|infusion\s+rate|"
    r"loading\s+dose|maintenance\s+dose|mg\b|mcg\b|microgram\w*|"
    r"adverse\s+effects?|side\s+effects?|safety|safe\b|tolerability|"
    r"toxicity|overdose|contraindicat\w*|interaction\w*|"
    r"sedation|sedat\w*|analgesi\w*)\b",
    re.IGNORECASE,
)


class _MedEntry:
    __slots__ = ("generic", "brands", "drug_class", "source", "surface_forms")

    def __init__(self, generic: str, brands: list[str], drug_class: str,
                 source: str, surface_forms: list[str]):
        self.generic = generic
        self.brands = brands
        self.drug_class = drug_class
        self.source = source
        self.surface_forms = surface_forms


def _safe_load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.exception("medsafety1 lexicon load failed: %s", path.name)
        return {}


def _build_index() -> tuple[list[_MedEntry], list[tuple[str, _MedEntry]]]:
    """Return (entries, sorted_surface_index). The surface index maps each
    lower-cased surface form -> entry, sorted by length DESC so multi-word
    brands ("seroquel xr") match before single tokens."""
    entries: list[_MedEntry] = []

    lex = _safe_load(_LEXICON_PATH)
    for m in (lex.get("medications") or []):
        generic = str(m.get("generic") or "").strip()
        if not generic:
            continue
        brands = [str(b).strip() for b in (m.get("brands") or []) if str(b).strip()]
        typos = [str(t).strip() for t in (m.get("typos") or []) if str(t).strip()]
        surfaces = [generic, *brands, *typos]
        entries.append(_MedEntry(
            generic=generic, brands=brands,
            drug_class=str(m.get("drug_class") or ""),
            source="medication_lexicon",
            surface_forms=surfaces,
        ))

    rx = _safe_load(_RXNORM_PATH)
    def _add_rx(concept: str, synonyms: list[str], drug_class: str):
        concept = str(concept or "").strip()
        if not concept:
            return
        syns = [str(s).strip() for s in (synonyms or []) if str(s).strip()]
        entries.append(_MedEntry(
            generic=concept, brands=syns, drug_class=str(drug_class or ""),
            source="rxnorm_curated", surface_forms=[concept, *syns],
        ))
    for single in (rx.get("single_drug_entries") or []):
        _add_rx(single.get("concept"), single.get("synonyms") or [],
                single.get("class_concept") or "")
    for cls in (rx.get("drug_class_mappings") or []):
        for member in (cls.get("members") or []):
            _add_rx(member.get("concept"), member.get("synonyms") or [],
                    cls.get("class_concept") or "")

    # Build surface index (lower-cased), longest-first, de-duplicated on
    # first occurrence (lexicon entries take precedence over rxnorm).
    seen: set[str] = set()
    index: list[tuple[str, _MedEntry]] = []
    for e in entries:
        for sf in e.surface_forms:
            key = sf.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            index.append((key, e))
    index.sort(key=lambda kv: len(kv[0]), reverse=True)
    return entries, index


_ENTRIES, _SURFACE_INDEX = _build_index()

# Pre-compile a word-boundary regex per surface form (longest first).
_COMPILED: list[tuple[re.Pattern[str], _MedEntry, str]] = [
    (re.compile(rf"\b{re.escape(sf)}\b", re.IGNORECASE), entry, sf)
    for sf, entry in _SURFACE_INDEX
]


def detect_medication(question: str) -> Optional[dict[str, Any]]:
    """Return the first recognized medication in ``question`` (longest
    surface form wins), or ``None``. Deterministic, pure.

    Result shape:
      {"generic", "matched_surface", "is_brand_match", "brands",
       "synonyms", "drug_class", "source"}
    """
    if not question:
        return None
    for patt, entry, sf in _COMPILED:
        if patt.search(question):
            is_brand = sf.lower() != entry.generic.lower()
            # Synonyms surfaced for the query = brand names + (when matched
            # by a brand) the canonical generic is the concept itself.
            synonyms = list(dict.fromkeys(entry.brands))
            return {
                "generic": entry.generic,
                "matched_surface": sf,
                "is_brand_match": bool(is_brand),
                "brands": list(entry.brands),
                "synonyms": synonyms,
                "drug_class": entry.drug_class,
                "source": entry.source,
            }
    return None


def is_dosing_safety_question(question: str) -> bool:
    """True when the question carries a dosing / safety / sedation signal."""
    if not question:
        return False
    return bool(_DOSING_SAFETY_RE.search(question))


def build_medication_anchor(match: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic ``drug_or_drug_class`` primary anchor in the
    exact shape ``PrimaryAnchor.to_dict()`` produces, so it slots straight
    into ``primary_clinical_anchors_json['primary_anchors']``."""
    generic = str(match.get("generic") or "").strip()
    synonyms = list(match.get("synonyms") or [])
    return {
        "text": str(match.get("matched_surface") or generic),
        "canonical_concept": generic,
        "entity_type": "drug_or_drug_class",
        "role": "intervention",
        "synonyms": synonyms,
        "source": "drug_pack",
        "confidence": 0.9,
        "required_for_search": True,
        "missing_relevance_cap": 20,
    }


__all__ = [
    "detect_medication",
    "is_dosing_safety_question",
    "build_medication_anchor",
]
