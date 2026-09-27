"""Build the reviewable search terms for an approved interpretation, and apply edits."""
from __future__ import annotations

import re
import time
from typing import Optional

import httpx

from clinical_engine.config import settings
from clinical_engine.contracts.question import SEARCH_ROLES, ClinicalQuestionMap
from clinical_engine.frameworks import role_label
from clinical_engine.gateway import ModelGateway
from clinical_engine.interpretation.present import EngineInputError
from clinical_engine.search_terms import mesh
from clinical_engine.search_terms.contracts import (
    DEFAULT_ON_KINDS,
    MAX_TERMS_PER_CONCEPT,
    MappingMeta,
    MeshHeading,
    SearchTerms,
    Term,
    TermConcept,
    TermEdit,
)
from clinical_engine.search_terms.synonyms import (
    SYSTEM_PROMPT,
    ConceptInput,
    clean_term,
    default_on,
    fallback_terms,
    is_head_preserving_shortening,
    user_prompt,
    validate_output,
)

# Concepts AND-ed into every query by default. Comparators and outcomes rank
# rather than restrict (AND-ing them loses relevant trials) — except where the
# outcome IS the subject: the condition being prevented, or the complication.
_REQUIRED_ROLES = frozenset({
    "condition", "presentation", "population", "intervention", "exposure", "index_test",
    "prognostic_factor", "phenomenon", "concept",
})
_OUTCOME_IS_SUBJECT = frozenset({"prevention", "complications"})
_MAX_CONCEPTS = 10


def _terms_gateways() -> list[ModelGateway]:
    """Classifying terms as same / narrower / broader / related is where a small
    model drifts (it called "massive transfusion" a form of whole blood
    transfusion), so this ~1k-token call uses a stronger model by default:
    CLINICAL_ENGINE_TERMS_MODEL, else GPT-4.1 on the configured provider."""
    base = ModelGateway()
    chosen = settings.TERMS_MODEL or {"openai": "gpt-4.1", "openrouter": "openai/gpt-4.1"}.get(base.provider, "")
    if not chosen or chosen == base.model:
        return [base]
    return [ModelGateway(model=chosen), base]


def _required(role: str, question_type: str) -> bool:
    return role in _REQUIRED_ROLES or (role == "outcome" and question_type in _OUTCOME_IS_SUBJECT)


_PEOPLE = r"(?:adults?|adult patients|children|child|kids|infants?|neonates?|newborns?|elderly|older adults?|" \
          r"older people|patients?|people|persons?|individuals?|subjects?|population|cases|men|women)"
_LEADING_PEOPLE = re.compile(rf"(?i)^(?:the\s+|all\s+)?{_PEOPLE}\s+(?:with|having|who have|undergoing|receiving|on|"
                             rf"admitted with|presenting with|diagnosed with)\s+")
_TRAILING_PEOPLE = re.compile(r"(?i)\s+(?:adults?|patients?|people|persons?|individuals?|subjects?|population|cases)$")
_ONLY_PEOPLE = re.compile(rf"(?i)^(?:{_PEOPLE}|critically|seriously|acutely)(?:\s+{_PEOPLE})*$")


_JOINED = r"(?:with|having|who have|undergoing|receiving|on|after|following|admitted with|presenting with|" \
          r"diagnosed with|for)"


def _population_term(value: str, others: list[str]) -> str | None:
    """A population descriptor as a search concept: age and people words are not
    searched ("critically ill adults" → "critically ill"; the age is the age
    focus), another concept inside it is removed ("hospitalized patients with
    severe COVID-19" → "hospitalized"), and a descriptor that only restates
    another concept ("adults with suspected infection") is dropped."""
    v = value.strip()
    for other in sorted(others, key=len, reverse=True):
        if other and other.lower() != v.lower():
            v = re.sub(rf"(?i)\s*\b{_JOINED}\s+{re.escape(other)}\b", "", v)
            v = re.sub(rf"(?i)^{re.escape(other)}\s+(?={_PEOPLE}\b)", "", v)
    v = _LEADING_PEOPLE.sub("", v.strip())
    v = _TRAILING_PEOPLE.sub("", v).strip(" ,")
    if not v or _ONLY_PEOPLE.match(v):
        return None
    key = mesh.norm(v)
    if any(key == mesh.norm(o) or f" {key} " in f" {mesh.norm(o)} " for o in others):
        return None
    return v


def concepts_from_map(cqm: ClinicalQuestionMap) -> list[TermConcept]:
    out: list[TermConcept] = []
    seen: set[tuple[str, str]] = set()
    non_population = [e.value for r in SEARCH_ROLES if r != "population" for e in cqm.active(r)]
    # Population descriptors ("trauma", "critically ill", "hospitalized") narrow who
    # the patients are; AND-ing them as well as the condition loses relevant papers,
    # so they rank rather than restrict — unless they are the only patient concept.
    has_condition = bool(cqm.active("condition") or cqm.active("presentation"))
    for role in SEARCH_ROLES:
        for e in cqm.active(role):
            term = _population_term(e.value, non_population) if role == "population" else e.value
            if not term:
                continue
            key = (role, mesh.norm(term))
            if key in seen or not key[1]:
                continue
            seen.add(key)
            required = _required(role, cqm.question_type) and not (role == "population" and has_condition)
            out.append(TermConcept(
                element_id=e.id, role=role, label=role_label(role, cqm.framework, cqm.question_type),
                term=term, required=required,
            ))
    return out[:_MAX_CONCEPTS]


# Qualitative questions: the phenomenon is found through how qualitative studies
# describe themselves, not one phrase (a standard qualitative search filter).
_QUALITATIVE_WORDS = (
    "experience", "experiences", "lived experience", "perception", "perceptions", "perspective",
    "perspectives", "attitudes", "qualitative", "interview", "interviews", "focus group", "focus groups",
    "thematic analysis",
)
_EXPERIENCE = re.compile(r"(?i)\b(experiences?|perceptions?|perspectives?|views?|attitudes?|beliefs?|lived)\b")
# Severity / certainty modifiers: "severe COVID-19" has no heading of its own, "COVID-19" does.
_MODIFIERS = re.compile(
    r"(?i)^(?:(?:severe|mild|moderate|moderate-to-severe|critical|acute|chronic|early|late|advanced|"
    r"recurrent|refractory|suspected|confirmed|presumed|hospitali[sz]ed|symptomatic|asymptomatic|"
    r"uncomplicated|complicated)\s+)+"
)


async def build_search_terms(
    cqm: ClinicalQuestionMap, *, question_version: int, gateway: Optional[ModelGateway] = None,
    allow_model: bool = True, mesh_transport: Optional[httpx.AsyncBaseTransport] = None,
) -> SearchTerms:
    started = time.perf_counter()
    concepts = concepts_from_map(cqm)
    meta = MappingMeta()
    warnings: list[str] = []

    lookup = await mesh.lookup_many([c.term for c in concepts], timeout_s=settings.MESH_TIMEOUT_S,
                                    transport=mesh_transport) if concepts else mesh.LookupResult(status="not_needed")
    meta.mesh = "unavailable" if lookup.status == "unavailable" else ("ncbi" if concepts else "not_needed")
    if lookup.status == "unavailable":
        warnings.append("MeSH could not be reached — concepts are searched as words only. Try again later "
                        "to add MeSH headings.")
    # No exact heading: offer the heading of the unmodified concept, switched off
    # ("severe COVID-19" → COVID-19 would also find non-severe cases).
    broader_forms = {c.term: _MODIFIERS.sub("", c.term).strip() for c in concepts
                     if lookup.records.get(c.term) is None}
    broader_forms = {k: v for k, v in broader_forms.items() if v and v.lower() != k.lower()}
    broader = (await mesh.lookup_many(sorted(set(broader_forms.values())), timeout_s=settings.MESH_TIMEOUT_S,
                                      transport=mesh_transport)
               if broader_forms and lookup.status != "unavailable" else mesh.LookupResult())
    inputs: list[ConceptInput] = []
    for c in concepts:
        record = lookup.records.get(c.term)
        if record is not None:
            c.mesh = MeshHeading(descriptor=record.descriptor, ui=record.ui or None, scope_note=record.scope_note)
        elif (wider := broader.records.get(broader_forms.get(c.term, ""))) is not None:
            c.mesh = MeshHeading(descriptor=wider.descriptor, ui=wider.ui or None, scope_note=wider.scope_note,
                                 match="broader", enabled=False)
        inputs.append(ConceptInput(
            id=c.id, role=c.role, term=c.term, mesh_heading=record.descriptor if record else None,
            candidates=mesh.entry_terms_as_words(record, c.term, limit=10) if record else [],
        ))

    gateways = [gateway] if gateway is not None else _terms_gateways()
    proposed = None
    if not concepts:
        meta.synonyms = "rules_only"
    elif not allow_model:
        meta.fallback_reason = "model_budget_exhausted"
    elif not gateways[0].is_configured():
        meta.fallback_reason = "model_not_configured"
    else:
        for gw in gateways:  # the terms model, then the interpreter model if that call fails
            result = await gw.complete_json(
                SYSTEM_PROMPT, user_prompt(cqm.question_text, inputs),
                max_tokens=settings.TERMS_MAX_TOKENS, timeout_s=settings.TERMS_TIMEOUT_S,
            )
            meta.model = result.model
            meta.tokens_in += result.tokens_in
            meta.tokens_out += result.tokens_out
            if result.data is not None:
                break
        if result.data is None:
            meta.fallback_reason = f"model_{result.error}"
        else:
            reading = validate_output(result.data, inputs)
            meta.rejected = reading.rejected[:20]
            proposed = reading.terms
    if proposed is None:
        meta.synonyms = "rules_only"
        proposed = fallback_terms(inputs)
        if concepts:
            warnings.append("Alternative wordings could not be checked automatically — MeSH entry terms are "
                            "listed switched off. Turn on the ones you want.")

    for c in concepts:
        for p in proposed.get(c.id, []):
            enabled = meta.synonyms == "model" and p.kind in DEFAULT_ON_KINDS and default_on(
                p.kind, p.text, c.term, p.source)
            c.terms.append(Term(text=p.text, kind=p.kind, source=p.source, enabled=enabled))  # type: ignore[arg-type]
        if c.role == "phenomenon" and cqm.question_type == "qualitative" and _EXPERIENCE.search(c.term):
            have = {mesh.norm(t.text) for t in c.terms} | {mesh.norm(c.term)}
            c.terms += [Term(text=w, kind="equivalent", source="vocabulary", enabled=True)
                        for w in _QUALITATIVE_WORDS if mesh.norm(w) not in have]
        c.terms = c.terms[:MAX_TERMS_PER_CONCEPT]
    meta.latency_ms = int((time.perf_counter() - started) * 1000)
    return SearchTerms(question_version=question_version, concepts=concepts, mapping=meta, warnings=warnings)


# ── Edits ────────────────────────────────────────────────────────────────────


def _concept(terms: SearchTerms, edit: TermEdit) -> TermConcept:
    c = terms.concept(edit.concept_id or "")
    if c is None:
        raise EngineInputError("unknown_concept", "That concept no longer exists — refresh and try again.")
    return c


def apply_term_edits(terms: SearchTerms, edits: list[TermEdit]) -> SearchTerms:
    if not edits:
        raise EngineInputError("no_edits", "No edits were provided.")
    if len(edits) > 40:
        raise EngineInputError("too_many_edits", "Too many edits in one request.")
    new = terms.model_copy(deep=True)
    for edit in edits:
        if edit.op == "set_use_synonyms":
            new.use_synonyms = bool(edit.enabled)
        elif edit.op == "set_publication_years":
            new.filters.publication_years = None if not edit.years else edit.years  # type: ignore[assignment]
        elif edit.op == "set_exclude_animal_only":
            new.filters.exclude_animal_only = bool(edit.enabled)
        elif edit.op == "set_required":
            _concept(new, edit).required = bool(edit.enabled)
        elif edit.op == "set_mesh_enabled":
            c = _concept(new, edit)
            if c.mesh is None:
                raise EngineInputError("no_mesh", "This concept has no MeSH heading.")
            c.mesh.enabled = bool(edit.enabled)
        elif edit.op == "set_term_enabled":
            c = _concept(new, edit)
            term = next((t for t in c.terms if t.id == edit.term_id), None)
            if term is None:
                raise EngineInputError("unknown_term", "That term no longer exists — refresh and try again.")
            term.enabled = bool(edit.enabled)
        elif edit.op == "remove_term":
            c = _concept(new, edit)
            c.terms = [t for t in c.terms if t.id != edit.term_id]
        elif edit.op == "add_term":
            c = _concept(new, edit)
            text = clean_term(edit.text or "")
            if text is None:
                raise EngineInputError(
                    "invalid_term", "Use plain words (up to 6) — no quotes, brackets or AND/OR/NOT.",
                )
            key = mesh.norm(text)
            if key == mesh.norm(c.term):
                continue
            existing = next((t for t in c.terms if mesh.norm(t.text) == key), None)
            if existing is not None:  # already listed (perhaps switched off): the clinician wants it in
                existing.enabled = True
                continue
            if len(c.terms) >= MAX_TERMS_PER_CONCEPT:
                raise EngineInputError("too_many_terms", "This concept already has the maximum number of terms.")
            kind = "broader" if is_head_preserving_shortening(text, c.term) else "equivalent"
            c.terms.append(Term(text=text, kind=kind, source="user", enabled=True))  # type: ignore[arg-type]
    new.version = terms.version + 1
    from models.common import utcnow

    new.updated_at = utcnow()
    return new


def blocking_problems(terms: SearchTerms) -> list[str]:
    if not terms.concepts:
        return ["No searchable concept in the approved question — go back and add what the question is about."]
    if not any(c.required for c in terms.concepts):
        return ["Keep at least one concept marked “must appear”, or the search would match almost anything."]
    return []
