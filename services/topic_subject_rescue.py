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

# Disease / condition morphology — types a salvaged noun phrase as
# ``disease_or_problem`` (so ranking uses the disease cap) instead of a generic
# exposure. Covers named conditions plus common medical suffixes.
_DISEASEISH_RE = re.compile(
    r"(ha?emorrhage|infarct\w*|ischa?emi\w*|stroke|seps[ie]s|pneumonia|"
    r"failure|injury|syndrome|disease|disorder|deficiency|hypertension|"
    r"diabetes|asthma|copd|arrhythmia|fibrillation|embolism|thrombosis|"
    r"infection|cancer|carcinoma|tumou?r|sarcoma|lymphoma|leuka?emia|"
    r"fracture|[ae]dema|o[ae]dema|encephalopathy|nephropathy|neuropathy|"
    r"\w{3,}itis\b|\w{3,}osis\b|\w{3,}[ae]mia\b|\w{3,}oma\b|\w{3,}pathy\b|"
    r"\w{3,}algia\b)",
    re.IGNORECASE,
)

# Words that describe WHAT is being asked (question intent) — never the clinical
# subject. Stripped when salvaging the residual clinical noun phrase.
_QUESTION_TYPE_WORDS: frozenset[str] = frozenset({
    "prognosis", "prognostic", "prognoses", "survival", "mortality", "morbidity",
    "outcome", "outcomes", "complication", "complications", "sequelae",
    "epidemiology", "incidence", "prevalence", "etiology", "aetiology", "cause",
    "causes", "causing", "pathophysiology", "pathogenesis", "mechanism",
    "mechanisms", "presentation", "symptom", "symptoms", "sign", "signs",
    "feature", "features", "workup", "evaluation", "assessment", "screening",
    "prevention", "prophylaxis", "monitoring", "surveillance", "followup",
    "recurrence", "relapse", "diagnosis", "diagnostic", "diagnose", "treatment",
    "treat", "therapy", "management", "manage", "benefit", "benefits", "risk",
    "risks", "effect", "effects", "efficacy", "effectiveness", "safety", "role",
    "use", "uses", "value", "impact", "importance", "indication", "indications",
    "dose", "doses", "dosing", "dosage", "guideline", "guidelines",
    "recommendation", "recommendations", "definition", "overview", "differential",
})

# Population / setting / temporal glue — never the clinical subject on its own.
_FILLER_WORDS: frozenset[str] = frozenset({
    "patient", "patients", "adult", "adults", "male", "males", "female",
    "females", "men", "women", "man", "woman", "boy", "boys", "girl", "girls",
    "child", "children", "infant", "infants", "neonate", "neonates", "newborn",
    "newborns", "adolescent", "adolescents", "elderly", "senior", "seniors",
    "geriatric", "pediatric", "paediatric", "people", "person", "persons",
    "individual", "individuals", "subject", "subjects", "case", "cases",
    "healthy", "young", "old", "older", "population", "populations",
    "after", "before", "following", "during", "post", "pre", "with", "without",
    "in", "on", "for", "of", "to", "from", "among", "amongst", "within",
    "between", "and", "or", "vs", "versus", "regarding", "about", "due",
    "related", "having", "who", "that", "which", "undergoing", "presenting",
    "diagnosed", "known", "suspected",
    # treatment-sequencing / ranking modifiers (never the subject on their own)
    "first", "second", "third", "line", "first-line", "second-line",
    "third-line", "initial", "next", "best", "standard", "optimal",
    "appropriate", "indicated", "ideal", "preferred", "recommended",
})

_GM_SINGLE: frozenset[str] = frozenset(m for m in GENERIC_MODIFIERS if " " not in m)

# Everything that is NOT a candidate clinical-subject token.
_SALVAGE_STOP: frozenset[str] = (
    _QUESTION_OPENER_TOKENS | _FILLER_WORDS | _QUESTION_TYPE_WORDS | _GM_SINGLE
)


def _longest_clinical_span(question: str) -> str:
    """Return the longest contiguous run of non-stopword tokens — the residual
    clinical noun phrase after stripping question-intent / filler / generic
    words. E.g. "prognosis of a patient after cerebellar hemorrhage" ->
    "cerebellar hemorrhage". Returns "" when nothing concrete remains, so
    genuinely vague questions still clarify."""
    toks = [t for t in re.split(r"[^a-z0-9\-]+", (question or "").lower()) if t]
    best: list[str] = []
    cur: list[str] = []
    for t in toks:
        if t in _SALVAGE_STOP or (len(t) == 1 and not t.isdigit()):
            if len(cur) > len(best):
                best = cur
            cur = []
        else:
            cur.append(t)
    if len(cur) > len(best):
        best = cur
    return " ".join(best).strip()


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

    # 3. General clinical-subject salvage: the longest run of non-stopword
    #    tokens is the most likely clinical subject. Handles prognosis /
    #    diagnosis / etiology / complication / "X after Y" questions and LLM
    #    mis-parses (e.g. "cerebellar hemorrhage" from "prognosis of a patient
    #    after cerebellar hemorrhage"). Vague questions leave an empty / generic
    #    residual and correctly return None so the pipeline still clarifies.
    span = _longest_clinical_span(q)
    if span and not _topic_is_generic(span):
        words = span.split()
        if 1 <= len(words) <= 6 and any(len(w) >= 3 and w.isalpha() for w in words):
            return span
    return None


# Structural / generic words that must never become a lone OR synonym — as a
# single token they would match almost everything. The salvaged-token
# broadening below drops these.
_TOPIC_TOKEN_STOP = frozenset({
    "disease", "diseases", "syndrome", "syndromes", "disorder", "disorders",
    "chronic", "acute", "severe", "mild", "patient", "patients", "adult",
    "adults", "child", "children", "elderly", "therapy", "treatment",
    "management", "complication", "complications", "risk", "risks", "effect",
    "effects", "outcome", "outcomes", "prognosis", "diagnosis", "symptom",
    "symptoms", "condition", "conditions",
})


def build_topic_anchor(topic: str) -> dict[str, Any]:
    """Build a deterministic primary anchor in the exact
    ``PrimaryAnchor.to_dict()`` shape so it slots straight into
    ``primary_clinical_anchors_json['primary_anchors']``."""
    t = (topic or "").strip()
    if _DISEASEISH_RE.search(t):
        entity_type, role, cap = "disease_or_problem", "population", 20
    elif _DRUGISH_RE.search(t):
        entity_type, role, cap = "drug_or_drug_class", "intervention", 20
    else:
        entity_type, role, cap = "exposure", "exposure", 30
    synonyms: list[str] = [t]
    # "creatine supplementation" -> also search bare "creatine".
    base = re.sub(r"\s+(?:supplementation|supplement|intake)$", "", t,
                  flags=re.IGNORECASE).strip()
    if base and base.lower() != t.lower():
        synonyms.append(base)
    # A salvaged multi-word topic only exists here because normal anchor
    # detection already failed; the exact compound phrase frequently matches ~0
    # records. Add its individual significant tokens as OR synonyms so the
    # Boolean compiler broadens to them (e.g. "migraine cancer" also searches
    # "migraine" / "cancer") instead of dead-ending at zero results. This runs
    # ONLY on the last-resort rescue path, so it can never widen a query that
    # already anchors and works.
    _toks = [
        w for w in re.split(r"\s+", t)
        if len(w) >= 4 and w.isalpha() and w.lower() not in _TOPIC_TOKEN_STOP
    ]
    if len(_toks) >= 2:
        synonyms.extend(_toks)
    return {
        "text": t,
        "canonical_concept": t,
        "entity_type": entity_type,
        "role": role,
        "synonyms": list(dict.fromkeys(synonyms)),
        "source": "rule",
        "confidence": 0.6,
        "required_for_search": True,
        "missing_relevance_cap": cap,
    }


__all__ = ["detect_topic_subject", "build_topic_anchor"]
