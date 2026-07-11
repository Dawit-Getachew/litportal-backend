"""Deterministic topic subject rescue (supplements / nutrients / "benefit of X").

DETERMINISTIC. NO LLM. NO network.

Sibling of :mod:`services.medication_subject_rescue`. The primary-clinical-anchor
derivation returns ``no_primary_clinical_entity`` for perfectly answerable
questions whose clinical subject is an intervention or exposure rather than a
disease -- e.g. "What is the benefit of creatine supplementation in adult
males?". The medication rescue only covers drugs in the curated lexicon, so a
supplement / nutrient / general-intervention question falls through to the
bounded LLM Clinical Context Resolver, which frequently rejects it
(``confidence_below_threshold``) and the session dead-ends at
``awaiting_clinical_subject`` with zero articles.

This module runs ONLY on that same failure path. It extracts a concrete,
searchable topic from two closed-set signals and builds an intervention /
exposure anchor so the question runs a real search:

  1. "benefit / effect / efficacy / role / use / value / impact / evidence
     (and misspellings) of <TOPIC> [in <population>]" framings.
  2. A named supplement / nutrient from a curated closed set.

Generic topics ("treatment", "first-line therapy", ...) are rejected via the
shared ``GENERIC_MODIFIERS`` guard, so genuinely vague questions still clarify.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from services.primary_clinical_anchor import (
    GENERIC_MODIFIERS,
    _QUESTION_OPENER_TOKENS,
)

logger = logging.getLogger("clinevidence.topic_subject_rescue")

# "<modifier> of <topic>" — the answerable benefit/effect/role family only.
# Deliberately EXCLUDES risk / harm / complication / adverse: those have their
# own downstream routing (complication_risk / drug_safety) and are handled by
# the medication rescue when a drug is named. The ``\w*`` suffix absorbs common
# spelling variants ("benefite", "benefits", "effects", "efficacy").
_MODIFIER_OF_RE = re.compile(
    r"\b(?:benefit|advantage|effect|efficac|effectiven|role|use|uses|"
    r"utilit|value|impact|importance|evidence)\w*\s+"
    r"(?:and\s+\w+\s+)?"          # "benefits and value of"
    r"(?:of|for|from)\s+"
    r"(?P<topic>[a-z0-9][a-z0-9\-'\s]{2,}?)\s*$",
    re.IGNORECASE,
)

# First boundary preposition ends the intervention phrase and drops the
# population / outcome / setting tail: "creatine supplementation in adult
# males" -> "creatine supplementation"; "vitamin d for bone health" ->
# "vitamin d"; "creatine on muscle mass" -> "creatine".
_TOPIC_BOUNDARY_RE = re.compile(
    r"\s+(?:in|on|for|among|amongst|within|versus|vs|to|with|during|"
    r"after|before|regarding|concerning|by)\s+.*$",
    re.IGNORECASE,
)

# Curated closed set of common supplements / nutrients. Additions want a test.
_SUPPLEMENTS: frozenset[str] = frozenset({
    "creatine", "whey protein", "whey", "collagen", "omega-3", "omega 3",
    "fish oil", "cod liver oil", "vitamin d", "vitamin d3", "vitamin c",
    "vitamin b12", "vitamin b6", "vitamin e", "vitamin a", "vitamin k",
    "multivitamin", "folate", "folic acid", "magnesium", "zinc", "iron",
    "calcium", "melatonin", "probiotics", "prebiotics", "caffeine",
    "beta-alanine", "beta alanine", "glutamine", "bcaa", "l-carnitine",
    "carnitine", "ashwagandha", "turmeric", "curcumin", "coenzyme q10",
    "coq10", "glucosamine", "chondroitin", "psyllium", "niacin", "biotin",
    "selenium", "potassium", "taurine", "arginine", "citrulline", "hmb",
    "beetroot juice", "nitrate", "spirulina", "ginseng", "ginkgo",
    "st john's wort", "st johns wort", "saw palmetto", "milk thistle",
    "elderberry", "echinacea", "resveratrol", "quercetin", "berberine",
    "melatonin", "l-theanine", "theanine", "dhea", "l-arginine",
})

_SUPP_SORTED = sorted(_SUPPLEMENTS, key=len, reverse=True)
_SUPP_COMPILED = [
    (re.compile(rf"\b{re.escape(s)}\b", re.IGNORECASE), s) for s in _SUPP_SORTED
]

_DRUGISH_RE = re.compile(
    r"\b(?:supplement\w*|vitamin\w*|mineral\w*|protein|amino\s+acid|acid|"
    r"creatine|omega|probiotic\w*|prebiotic\w*|caffeine|drug|therapy|"
    r"agent|extract|oil|juice|nitrate|multivitamin)\b",
    re.IGNORECASE,
)


def _topic_is_generic(topic: str) -> bool:
    """True when, after stripping closed-set generic modifiers, nothing
    concrete remains (only question-openers / function words)."""
    s = (topic or "").lower().strip()
    if not s:
        return True
    for mod in sorted(GENERIC_MODIFIERS, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(mod)}\b", " ", s)
    leftover = [t for t in re.split(r"[^a-z0-9]+", s) if t]
    if not leftover:
        return True
    return all(t in _QUESTION_OPENER_TOKENS for t in leftover)


def _clean_topic(topic: str) -> str:
    t = (topic or "").strip().strip("?.!,;:").strip()
    t = _TOPIC_BOUNDARY_RE.sub("", t).strip()
    return t


def detect_topic_subject(question: str) -> Optional[str]:
    """Return a concrete searchable topic string, or ``None``.

    Pure / deterministic. Never raises on normal input.
    """
    if not question:
        return None
    q = re.sub(r"[?.!]+\s*$", "", question.strip()).strip()

    # 1. "<benefit/effect/role/...> of <topic> [in <population>]".
    m = _MODIFIER_OF_RE.search(q)
    if m:
        topic = _clean_topic(m.group("topic") or "")
        if topic and not _topic_is_generic(topic) and len(topic.split()) <= 6:
            return topic

    # 2. A named supplement / nutrient anywhere in the question. Attach a
    #    trailing "supplement(ation)" / "intake" when present so the query is
    #    "creatine supplementation" rather than the bare token.
    low = q.lower()
    for patt, surface in _SUPP_COMPILED:
        mm = patt.search(low)
        if not mm:
            continue
        tail = low[mm.end():].lstrip()
        for suffix in ("supplementation", "supplement", "intake"):
            if tail.startswith(suffix):
                return f"{surface} {suffix}"
        return surface
    return None


def build_topic_anchor(topic: str) -> dict[str, Any]:
    """Build a deterministic primary anchor in the exact
    ``PrimaryAnchor.to_dict()`` shape so it slots straight into
    ``primary_clinical_anchors_json['primary_anchors']``."""
    t = (topic or "").strip()
    drugish = bool(_DRUGISH_RE.search(t))
    entity_type = "drug_or_drug_class" if drugish else "exposure"
    role = "intervention" if drugish else "exposure"
    synonyms: list[str] = [t]
    # "creatine supplementation" -> also search bare "creatine".
    base = re.sub(r"\s+(?:supplementation|supplement|intake)$", "", t,
                  flags=re.IGNORECASE).strip()
    if base and base.lower() != t.lower():
        synonyms.append(base)
    return {
        "text": t,
        "canonical_concept": t,
        "entity_type": entity_type,
        "role": role,
        "synonyms": list(dict.fromkeys(synonyms)),
        "source": "rule",
        "confidence": 0.6,
        "required_for_search": True,
        "missing_relevance_cap": 20 if drugish else 30,
    }


__all__ = ["detect_topic_subject", "build_topic_anchor"]
