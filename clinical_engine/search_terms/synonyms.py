"""Alternative wordings for each approved concept, classified by meaning.

The model proposes terms and classifies the MeSH entry terms; its output is
untrusted: strict schema, closed kinds, terms must be plain words (no operators),
never another concept of the question, and a head-preserving shortening of the
concept ("blood transfusion" for "whole blood transfusion") is always *broader*
whatever the model says. Only equivalent and narrower terms are switched on.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from clinical_engine.search_terms.mesh import norm

KINDS = ("equivalent", "narrower", "broader", "related")
_CAPS = {"equivalent": 8, "narrower": 5, "broader": 3, "related": 3}
MAX_WORDS = 6
MAX_CHARS = 80

SYSTEM_PROMPT = """You are a medical librarian building a literature search from a clinician's approved question.
For each concept, list the words authors use for THAT concept in titles and abstracts, and classify each term:
  equivalent — exactly the same concept: spelling variants, hyphenation, word order, abbreviations and
    acronyms, alternative names, the generic and brand names of the same single drug.
  narrower — one specific type or form OF the concept; every paper about it is also about the concept.
  broader — a more general concept that also covers other things.
  related — anything else: a sibling, alternative, component, comparator, cause or consequence.
  exclude — only for a MeSH candidate that is a different condition or irrelevant to this question.
Example, concept "non-invasive ventilation": equivalent "noninvasive ventilation", "NIV",
  "non-invasive positive pressure ventilation"; narrower "CPAP", "BiPAP"; broader "mechanical ventilation",
  "respiratory support"; related "high-flow nasal oxygen", "invasive ventilation".
Example, concept "dexamethasone": equivalent "Decadron"; broader "corticosteroids", "glucocorticoids";
  related "hydrocortisone".
When unsure whether a term is the same concept, classify it as related — a wrong equivalent or narrower
term floods the search with papers about something else.
Classify every MeSH candidate you are given, and add your own terms.
Return ONLY JSON: {"concepts": [{"id": "<concept id>", "terms": [{"text": "<term>", "kind": "<kind>"}]}]}
Rules:
- 1 to 6 words per term, written as in an abstract. No Boolean operators, quotes, brackets, field tags,
  wildcards or explanations.
- Never repeat the concept itself, and never use another concept of the question as a term of this one.
- Up to 8 equivalent, 5 narrower, 3 broader and 3 related terms per concept; fewer is fine. Never invent
  a term that is not used in the medical literature.
JSON only."""


class _ModelTerm(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = Field(min_length=1, max_length=200)
    kind: str


class _ModelConcept(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    terms: list[_ModelTerm] = Field(default_factory=list, max_length=60)


class _ModelOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    concepts: list[_ModelConcept] = Field(default_factory=list, max_length=40)


@dataclass
class ConceptInput:
    id: str
    role: str
    term: str
    mesh_heading: Optional[str] = None
    candidates: list[str] = field(default_factory=list)  # MeSH entry terms to classify


@dataclass
class ProposedTerm:
    text: str
    kind: str
    source: str  # mesh | model


@dataclass
class SynonymReading:
    terms: dict[str, list[ProposedTerm]]
    rejected: list[str] = field(default_factory=list)


_OPERATOR = re.compile(r"[\"\[\]()*?{}<>:;=|\\]|\b(?:AND|OR|NOT|NEAR)\b|\[(?:tiab|mh|mesh)")
_CITATIONISH = re.compile(r"(https?://|www\.|\bdoi\b|10\.\d{4,}/|\bpmid\b|\bet al\b)", re.I)


def clean_term(text: str) -> Optional[str]:
    """A plain search phrase, or None when *text* is not one."""
    t = " ".join((text or "").split()).strip(" .,;:-")
    if len(t) < 2 or len(t) > MAX_CHARS or len(t.split()) > MAX_WORDS:
        return None
    if _OPERATOR.search(t) or _CITATIONISH.search(t) or ", " in t:  # "shock, septic" is a heading, not prose
        return None
    if not re.search(r"[A-Za-z]", t):
        return None
    return t


def _tokens(text: str) -> list[str]:
    return norm(text).split()


_STOP = frozenset({"of", "the", "a", "an", "in", "with", "for", "and", "or", "to", "on", "by", "due", "patients",
                   "patient", "people", "adults", "adult", "use", "therapy", "treatment", "administration",
                   "stage"})


def _stem(word: str) -> str:
    w = word.lower().replace("haem", "hem").replace("oe", "e").replace("ae", "e")
    return w[:6]


# TNM: clinical (cT3) and pathological (pT3) T3 are both "T3" to a question that does not say which.
_TNM_PREFIX = re.compile(r"^[cp](t[0-4][a-c]?)$")


def _content(text: str) -> set[str]:
    return {_stem(_TNM_PREFIX.sub(r"\1", w)) for w in _tokens(text.replace("-", " ")) if w not in _STOP}


def _is_acronym(text: str) -> bool:
    t = text.replace("-", "").replace(" ", "")
    return 2 <= len(t) <= 10 and " " not in text.strip() and sum(ch.isupper() for ch in t) >= 2 \
        and any(ch.isalpha() for ch in t)


def _initials(text: str) -> str:
    """Letters an acronym would use: an all-caps word counts whole ("quick SOFA" → "qsofa")."""
    out = ""
    for w in re.split(r"[\s\-/]+", text.strip()):
        w = re.sub(r"[^A-Za-z0-9]", "", w)
        if not w or w.lower() in _STOP:
            continue
        out += w.lower() if (w.isupper() and len(w) > 1) else w[0].lower()
    return out


def _acronym_matches(acronym: str, expansion: str) -> bool:
    a = re.sub(r"[^a-z0-9]", "", acronym.lower())
    words = [w for w in _tokens(expansion) if w not in _STOP]
    if len(words) >= 2:
        return _initials(expansion) == a
    # single-word concept ("procalcitonin" → "PCT"): same first letter is all we can check
    return bool(words) and bool(a) and a[0] == words[0][0]


def plausibly_same(candidate: str, concept: str) -> bool:
    """Lexical sanity check for an "equivalent" label: every word of the concept
    is in the candidate (re-ordered, hyphenated, UK/US spelling, filler words:
    "transfusion of whole blood"), or it is the acronym / expansion of it.
    A sibling that swaps a word — "red blood cell transfusion" for "whole blood
    transfusion" — fails, as does an unrelated acronym like "RBC". True
    equivalents this misses stay listed, switched off, for the clinician."""
    if _is_acronym(candidate):
        return _acronym_matches(candidate, concept)
    if _is_acronym(concept):
        return _acronym_matches(concept, candidate)
    c, t = _content(candidate), _content(concept)
    return bool(t) and t <= c


# "whole blood transfusion" is about giving whole blood: a form of the product
# ("low titer O whole blood") is a form of the concept.
_ACTION_HEADS = frozenset({
    "transf", "infusi", "therap", "admini", "supple", "inject", "intake", "prophy", "manage", "use", "treatm",
})


def contains_concept(candidate: str, concept: str) -> bool:
    """A lexically certain narrowing: the whole concept plus a modifier
    ("traumatic hemorrhagic shock" for "hemorrhagic shock")."""
    c, t = _content(candidate), _content(concept)
    if not t:
        return False
    if t <= c and len(c) > len(t):
        return True
    words = _tokens(concept.replace("-", " "))
    head = _stem(words[-1]) if words else ""
    core = t - {head}
    return head in _ACTION_HEADS and len(core) >= 1 and core <= c and len(c) > len(core)


def is_bare_product(candidate: str, concept: str) -> bool:
    """"whole blood" for "whole blood transfusion": the product without the act of
    giving it also names lab specimens, assays and so on — not the same concept."""
    words = _tokens(concept.replace("-", " "))
    if len(words) < 2 or _stem(words[-1]) not in _ACTION_HEADS:
        return False
    if any(_stem(w) in _ACTION_HEADS for w in _tokens(candidate.replace("-", " "))):
        return False  # it names an act of giving too ("steroid therapy")
    return _content(candidate) == _content(concept) - {_stem(words[-1])}


def default_on(kind: str, text: str, concept: str, source: str) -> bool:
    """Switched on by default only when the meaning is safe for precision."""
    if kind == "equivalent":
        return plausibly_same(text, concept) and not is_bare_product(text, concept)
    if kind == "narrower":
        return contains_concept(text, concept)
    return False


def is_head_preserving_shortening(candidate: str, concept: str) -> bool:
    """"blood transfusion" / "transfusion" for "whole blood transfusion": the same
    head noun with modifiers dropped names a broader concept."""
    c, t = _tokens(candidate), _tokens(concept)
    return bool(c) and len(c) < len(t) and set(c) < set(t) and c[-1] == t[-1]


def validate_output(raw: Any, concepts: list[ConceptInput]) -> SynonymReading:
    """Raw model JSON → classified terms per concept id. Never raises."""
    rejected: list[str] = []
    out: dict[str, list[ProposedTerm]] = {c.id: [] for c in concepts}
    if not isinstance(raw, dict):
        return SynonymReading(out, ["schema_invalid"])
    try:
        parsed = _ModelOutput.model_validate(raw)
    except ValidationError:
        return SynonymReading(out, ["schema_invalid"])
    by_id = {c.id: c for c in concepts}
    # A term of one concept must never be another concept of the question.
    concept_keys = {norm(c.term): c.id for c in concepts}
    for item in parsed.concepts:
        concept = by_id.get(item.id)
        if concept is None:
            rejected.append("unknown_concept")
            continue
        seen = {norm(concept.term)}
        counts = {k: 0 for k in KINDS}
        candidates = {norm(x) for x in concept.candidates}
        for term in item.terms:
            kind = term.kind.strip().lower()
            text = clean_term(term.text)
            if text is None:
                rejected.append("not_a_plain_term")
                continue
            key = norm(text)
            if kind == "exclude" or key in seen:
                continue
            if kind not in KINDS:
                rejected.append("unknown_kind")
                continue
            owner = concept_keys.get(key)
            if owner is not None and owner != concept.id:
                rejected.append("is_another_concept")
                continue
            if kind in ("equivalent", "narrower") and is_head_preserving_shortening(text, concept.term):
                kind = "broader"
            if counts[kind] >= _CAPS[kind]:
                continue
            counts[kind] += 1
            seen.add(key)
            out[concept.id].append(ProposedTerm(text, kind, "mesh" if key in candidates else "model"))
    return SynonymReading(out, rejected)


def fallback_terms(concepts: list[ConceptInput]) -> dict[str, list[ProposedTerm]]:
    """No model: offer the MeSH entry terms, switched off (see builder), unclassified."""
    out: dict[str, list[ProposedTerm]] = {}
    for c in concepts:
        seen = {norm(c.term)}
        terms: list[ProposedTerm] = []
        for cand in c.candidates:
            text = clean_term(cand)
            if text is None or norm(text) in seen:
                continue
            seen.add(norm(text))
            kind = "broader" if is_head_preserving_shortening(text, c.term) else "equivalent"
            terms.append(ProposedTerm(text, kind, "mesh"))
        out[c.id] = terms[:_CAPS["equivalent"]]
    return out


def user_prompt(question: str, concepts: list[ConceptInput]) -> str:
    return json.dumps({
        "question": question,
        "concepts": [{"id": c.id, "role": c.role, "term": c.term, "mesh_heading": c.mesh_heading,
                      "candidates": c.candidates} for c in concepts],
    }, ensure_ascii=False)
