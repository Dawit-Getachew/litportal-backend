"""Phase 6G — PICOTT term mapping projection service.

Pure projection over an already-persisted ``search_sessions`` document
that yields the canonical ``picott_term_mapping_json`` payload defined
in ``models/picott_term_mapping.py``.

This module does NOT re-run the primary clinical anchor service, the
PICOTT parser, the clinical context engine, or the boolean query
compiler. It only re-shapes what those upstream services already
persisted on the session document.

Determinism + safety:
- No LLM, no external API calls.
- No raw user text outside of canonical concepts that upstream services
  already sanitized.
- No URLs with query strings in the rendered ``query_block`` labels.
- Pydantic ``extra="forbid"`` enforcement on the output guards against
  accidental key drift.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from models.picott_term_mapping import (
    CompletenessBlock,
    PICOTT_ROLES,
    PicottTermMapping,
    TermMapEntry,
)
from services.primary_clinical_anchor import (
    CLARIFICATION_MESSAGE,
    GENERIC_MODIFIERS,
)

logger = logging.getLogger("clinevidence.picott_term_mapping")


# ---------------------------------------------------------------------
# Closed-set routing tables
# ---------------------------------------------------------------------

# entity_type → default PICOTT role when no more-specific signal exists.
_ENTITY_TYPE_TO_ROLE: dict[str, str] = {
    "disease_or_problem": "population_problem",
    "drug_or_drug_class": "intervention_exposure_index_test",
    "vaccine": "intervention_exposure_index_test",
    "diagnostic_test": "intervention_exposure_index_test",
    "procedure": "intervention_exposure_index_test",
    # Phase 6G.4P — procedure_or_surgery anchors map to
    # ``population_problem`` (the procedure is what the question is
    # ABOUT — the patient population). management_intervention,
    # device_or_implant, perioperative_context, and
    # postprocedure_context map to intervention slot.
    "procedure_or_surgery": "population_problem",
    "device_or_implant": "intervention_exposure_index_test",
    "perioperative_context": "intervention_exposure_index_test",
    "postprocedure_context": "intervention_exposure_index_test",
    "management_intervention": "intervention_exposure_index_test",
    "exposure": "intervention_exposure_index_test",
    "biomarker_or_lab": "outcomes_target_diagnosis",
    "population_group": "population_problem",
    "clinical_action": "clinical_action",
    "unknown": "population_problem",
}

# PICOTT concept role → PICOTT mapping field.
_CONCEPT_ROLE_TO_PICOTT_ROLE: dict[str, str] = {
    "intervention": "intervention_exposure_index_test",
    "index_test": "intervention_exposure_index_test",
    "comparator": "comparator_reference_standard",
    "reference_standard": "comparator_reference_standard",
    "condition": "population_problem",
    "population": "population_problem",
    "patient_group": "population_problem",
    "outcome": "outcomes_target_diagnosis",
    "target_diagnosis": "outcomes_target_diagnosis",
    "study_design": "clinical_action",
    "context": "clinical_action",
    "setting": "setting",
    "time_horizon": "time_horizon",
    "other": "clinical_action",
}

# Lexical hints for generic-modifier role routing (Task 2 spec).
_OUTCOME_MODIFIER_HINTS: tuple[str, ...] = (
    "side effect", "side effects",
    "adverse effect", "adverse effects",
    "adverse reaction", "adverse reactions",
    "adverse event", "adverse events",
    "drug safety", "safety", "tolerability", "toxicity",
    "outcome", "outcomes", "effectiveness", "efficacy",
    "benefit", "benefits",
)
_ACTION_MODIFIER_HINTS: tuple[str, ...] = (
    "first-line treatment", "first line treatment",
    "second-line treatment", "second line treatment",
    "first-line therapy", "first line therapy",
    "second-line therapy", "second line therapy",
    "initial treatment",
    "next treatment",
    "treatment algorithm", "management algorithm",
    "recommended treatment", "standard treatment",
    "treatment options",
    "therapy sequence",
    "after failure of first-line",
    "dose", "doses", "dosing",
    "first dose", "second dose",
    "timing",
    "recommended age", "vaccination age",
    "primary series", "booster", "catch-up",
    "two-dose schedule", "2-dose schedule",
    "treatment", "therapy", "management",
    "guideline", "guidelines",
    "diagnosis", "accuracy",
)

# Question subtype → list of required PICOTT roles. Used by the
# completeness guard. Empty list ⇒ a single primary anchor satisfies.
_REQUIRED_ROLES_BY_SUBTYPE: dict[str, list[str]] = {
    "drug_safety": ["intervention_exposure_index_test",
                    "outcomes_target_diagnosis"],
    "adverse_effects": ["intervention_exposure_index_test",
                        "outcomes_target_diagnosis"],
    "treatment_sequencing": ["population_problem", "clinical_action"],
    "first_line_treatment": ["population_problem", "clinical_action"],
    "first-line treatment": ["population_problem", "clinical_action"],
    "dosing_protocol": ["population_problem",
                        "intervention_exposure_index_test",
                        "clinical_action"],
    "dose_timing": ["population_problem",
                    "intervention_exposure_index_test",
                    "clinical_action"],
    "diagnosis": ["population_problem",
                  "intervention_exposure_index_test"],
    "diagnostic_accuracy": ["population_problem",
                            "intervention_exposure_index_test"],
    "vaccine_schedule": ["intervention_exposure_index_test",
                         "clinical_action"],
    "vaccine_benefit": ["intervention_exposure_index_test",
                        "outcomes_target_diagnosis"],
    "benefit": ["intervention_exposure_index_test",
                "outcomes_target_diagnosis"],
    # Phase 6G.1 — background disease / drug definition. A primary
    # anchor IS required; clinical_action ("definition / overview") is
    # informative but NOT required for completeness.
    "disease_overview": ["population_problem"],
    "definition": ["population_problem"],
    "drug_overview": ["intervention_exposure_index_test"],
    # Phase 6G.4P — post-procedure management requires BOTH a
    # population/problem anchor (the procedure) AND an intervention
    # anchor (the management).
    "post_procedure_management": [
        "population_problem", "intervention_exposure_index_test",
    ],
}


# Phase 6G.4P — closed-set refinement-note dictionary indexed by
# ``subdomain``. NEVER generated by an LLM — only these closed-set
# strings can surface in ``PicottTermMapping.refinement_note``.
_REFINEMENT_NOTES_BY_SUBDOMAIN: dict[str, str] = {
    "post_procedure_antithrombotic_management": (
        "Recommendations vary by mechanical vs bioprosthetic valve, "
        "surgical vs transcatheter intervention, and atrial fibrillation status."
    ),
    "post_procedure_vte_prophylaxis": (
        "Recommendations vary by procedure type, patient bleeding risk, "
        "and renal function."
    ),
    "post_procedure_antibiotic_management": (
        "Recommendations vary by procedure type, patient risk factors, "
        "and local antibiogram."
    ),
    "post_procedure_followup_surveillance": (
        "Surveillance interval and modality vary by initial pathology "
        "and risk stratification."
    ),
    "perioperative_medication_management": (
        "Restart timing varies by procedure bleeding risk, indication, "
        "and patient thromboembolic risk."
    ),
    "post_transplant_immunosuppression": (
        "Regimens vary by organ, induction protocol, and rejection history."
    ),
    "post_procedure_pain_management": (
        "Recommendations vary by procedure type, patient opioid tolerance, "
        "and comorbidities."
    ),
    # Phase H2 — post-transplant medication management (anti-CD20 /
    # rituximab maintenance / post-transplant consolidation regimens).
    "post_transplant_medication_management": (
        "Regimens vary by indication (lymphoma, multiple myeloma), "
        "transplant type (autologous vs allogeneic), and prior therapy."
    ),
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _empty_fields() -> dict[str, list[TermMapEntry]]:
    return {role: [] for role in PICOTT_ROLES}


_URL_QS_RE = re.compile(r"https?://\S+\?\S+")


def _safe_query_block_label(text: str) -> Optional[str]:
    """Return ``text`` if it does NOT contain a full URL with query
    string; else return ``None``. Used to keep ``query_block`` labels
    PHI-/secret-free.
    """
    s = _clean_str(text)
    if not s or _URL_QS_RE.search(s):
        return None
    return s


def _anchor_to_role(anchor: dict, *, subtype: str | None) -> str:
    """Map a primary-anchor dict (from PrimaryClinicalEntityAnchorService)
    to a PICOTT role per the Task 2 rule table.

    Rules (in priority order):
      1. drug_or_drug_class with subtype in {treatment_sequencing,
         drug_safety, adverse_effects, dosing_protocol} → intervention
      2. diagnostic_test with subtype in {diagnosis, accuracy} →
         intervention (as index test)
      3. vaccine with subtype in {vaccine_schedule, schedule, benefit} →
         intervention
      4. biomarker_or_lab with anchor.role="outcome" → outcomes
      5. Phase K4A.4.1 — when subtype is the closed-set epidemiology /
         incidence shape, the ``disease_or_problem`` anchor IS the
         outcome (incidence OF disease X). Route it to the outcomes
         bucket so the clinician panel shows it as "Outcome" rather
         than "Population". The exposure / risk-context anchor stays
         in its native ``intervention_exposure_index_test`` bucket.
      6. Otherwise: lookup by entity_type.
    """
    et = (anchor.get("entity_type") or "").strip().lower()
    role = (anchor.get("role") or "").strip().lower()
    st = (subtype or "").strip().lower()
    if et == "drug_or_drug_class" and st in {
        "treatment_sequencing", "drug_safety", "adverse_effects",
        "dosing_protocol", "first_line_treatment", "first-line treatment",
    }:
        return "intervention_exposure_index_test"
    if et == "diagnostic_test" and st in {
        "diagnosis", "accuracy", "diagnostic_accuracy",
    }:
        return "intervention_exposure_index_test"
    if et == "vaccine" and st in {
        "vaccine_schedule", "schedule", "benefit", "vaccine_benefit",
    }:
        return "intervention_exposure_index_test"
    if et == "biomarker_or_lab" and role == "outcome":
        return "outcomes_target_diagnosis"
    # Phase K4A.4.1 — epidemiology / incidence routing override.
    if st == "epidemiology_incidence_complication_risk" and et == "disease_or_problem":
        return "outcomes_target_diagnosis"
    return _ENTITY_TYPE_TO_ROLE.get(et, "population_problem")


def _modifier_to_role(text: str) -> str:
    """Generic-modifier-only fallback: outcome vs clinical_action based
    on lexical pattern. Per Task 2 spec:
      - "side effects" / "adverse" → outcome
      - "first-line treatment" / "dose" → clinical_action

    Phase K4A.4.1 — return the sentinel ``"_drop_"`` for purely
    generic population labels (``patients``, ``adults`` …). The
    ``build_picott_term_mapping`` bucketing skips entries with this
    role so the UI doesn't surface them in the "Clinical action"
    panel.
    """
    t = _clean_str(text).lower()
    if not t:
        return "clinical_action"
    if t in _DISPLAY_ONLY_GENERIC_TERMS:
        return "_drop_"
    if any(t == h or h in t for h in _OUTCOME_MODIFIER_HINTS):
        return "outcomes_target_diagnosis"
    if any(t == h or h in t for h in _ACTION_MODIFIER_HINTS):
        return "clinical_action"
    return "clinical_action"


# Phase K4A.4.1 — closed-set of purely generic population labels that
# must NEVER be surfaced as a clinical action / intent. They survive
# as ``primary_clinical_anchors_json.generic_modifiers`` for
# transparency but the projection drops them so the clinician panel
# isn't polluted with "patients" / "adults" as clinical actions.
_DISPLAY_ONLY_GENERIC_TERMS: frozenset[str] = frozenset({
    "patient", "patients",
    "adult", "adults",
    "child", "children",
    "infant", "infants",
    "person", "people", "individual", "individuals",
    "subject", "subjects",
    "case", "cases",
    "population", "populations",
})


def _anchor_synonyms(anchor: dict) -> list[str]:
    syns = anchor.get("synonyms")
    if not isinstance(syns, list):
        return []
    cleaned: list[str] = []
    for s in syns:
        v = _clean_str(s)
        if v and v not in cleaned:
            cleaned.append(v)
    return cleaned


def _anchor_to_entry(anchor: dict, subtype: str | None) -> TermMapEntry:
    role = _anchor_to_role(anchor, subtype=subtype)
    canonical = _clean_str(anchor.get("canonical_concept")) or _clean_str(
        anchor.get("text")
    )
    original = _clean_str(anchor.get("text")) or canonical
    return TermMapEntry(
        original_phrase=original,
        canonical_concept=canonical,
        role=role,  # type: ignore[arg-type]
        synonyms_used=_anchor_synonyms(anchor),
        mesh_terms=[],
        source="primary_anchor",
        required_for_high_relevance=bool(anchor.get("required_for_search")),
        query_block=None,
        confidence=float(anchor.get("confidence") or 0.0),
        rationale=f"primary_anchor:{anchor.get('entity_type', 'unknown')}",
    )


def _must_have_to_entry(must: dict) -> Optional[TermMapEntry]:
    concept = _clean_str(must.get("concept"))
    if not concept:
        return None
    role_in = (must.get("role") or "").strip().lower()
    picott_role = _CONCEPT_ROLE_TO_PICOTT_ROLE.get(role_in, "population_problem")
    syns: list[str] = []
    for s in must.get("synonyms") or []:
        v = _clean_str(s)
        if v and v not in syns:
            syns.append(v)
    return TermMapEntry(
        original_phrase=concept,
        canonical_concept=concept,
        role=picott_role,  # type: ignore[arg-type]
        synonyms_used=syns,
        mesh_terms=[],
        source="picott_parser",
        required_for_high_relevance=bool(
            must.get("required_for_high_relevance", True)
        ),
        query_block=None,
        confidence=0.7,
        rationale=f"must_have:{role_in or 'unspecified'}",
    )


def _translation_to_entry(trans: dict) -> Optional[TermMapEntry]:
    """Map a clinical_concept_translation entry to a TermMapEntry."""
    original = _clean_str(trans.get("original_phrase"))
    canonical = _clean_str(trans.get("canonical_concept"))
    if not canonical:
        return None
    # Skip internal subtype identifiers (snake_case canonicals are
    # parser tags, not clinical concepts).
    if re.match(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$", canonical):
        return None
    return TermMapEntry(
        original_phrase=original or canonical,
        canonical_concept=canonical,
        role="outcomes_target_diagnosis",  # translations are usually outcomes
        synonyms_used=[],
        mesh_terms=[],
        source="concept_translation",
        required_for_high_relevance=False,
        query_block=None,
        confidence=float(trans.get("confidence") or 0.6),
        rationale="concept_translation",
    )


def _role_repair_entries(
    repairs: Any, must_have_concepts: list[dict],
) -> list[TermMapEntry]:
    """Convert role_repairs into TermMapEntries when the repair points
    at an existing must_have concept. Adds the entry with
    source="role_repair" so the clinician can see why a concept moved
    between PICO slots.
    """
    if not isinstance(repairs, list):
        return []
    out: list[TermMapEntry] = []
    by_concept = {
        _clean_str(m.get("concept")).lower(): m for m in (must_have_concepts or [])
    }
    for r in repairs:
        if not isinstance(r, dict):
            continue
        concept = _clean_str(r.get("concept")) or _clean_str(r.get("term"))
        if not concept:
            continue
        new_role = (r.get("new_role") or r.get("to_role") or "").strip().lower()
        picott_role = _CONCEPT_ROLE_TO_PICOTT_ROLE.get(new_role, "")
        if not picott_role:
            continue
        m = by_concept.get(concept.lower(), {})
        out.append(TermMapEntry(
            original_phrase=concept,
            canonical_concept=concept,
            role=picott_role,  # type: ignore[arg-type]
            synonyms_used=list(m.get("synonyms") or []),
            mesh_terms=[],
            source="role_repair",
            required_for_high_relevance=bool(
                m.get("required_for_high_relevance", False)
            ),
            query_block=None,
            confidence=0.6,
            rationale=f"role_repair→{new_role}",
        ))
    return out


def _ontology_lift_entries(lifts: Any) -> list[TermMapEntry]:
    if not isinstance(lifts, list):
        return []
    out: list[TermMapEntry] = []
    for li in lifts:
        if not isinstance(li, dict):
            continue
        canonical = _clean_str(li.get("canonical_concept") or li.get("concept"))
        original = _clean_str(li.get("original_phrase") or li.get("term"))
        if not canonical and not original:
            continue
        role_in = (li.get("role") or "").strip().lower()
        picott_role = _CONCEPT_ROLE_TO_PICOTT_ROLE.get(
            role_in, "population_problem",
        )
        out.append(TermMapEntry(
            original_phrase=original or canonical,
            canonical_concept=canonical or original,
            role=picott_role,  # type: ignore[arg-type]
            synonyms_used=[
                _clean_str(s) for s in (li.get("synonyms") or [])
                if _clean_str(s)
            ],
            mesh_terms=[
                _clean_str(m) for m in (li.get("mesh_terms") or [])
                if _clean_str(m)
            ],
            source="ontology_lift",
            required_for_high_relevance=bool(
                li.get("required_for_high_relevance", False)
            ),
            query_block=None,
            confidence=float(li.get("confidence") or 0.65),
            rationale="ontology_lift",
        ))
    return out


def _dedupe_entries(entries: list[TermMapEntry]) -> list[TermMapEntry]:
    """Deduplicate by (role, canonical_concept.lower()). The first
    entry wins so primary_anchor entries (added first) are preserved.
    """
    seen: set[tuple[str, str]] = set()
    out: list[TermMapEntry] = []
    for e in entries:
        key = (e.role, e.canonical_concept.lower())
        if not e.canonical_concept:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _resolve_required_roles(
    subtype: str | None, type_of_question: str | None,
) -> list[str]:
    """Return the list of PICOTT roles required to consider the
    mapping search-safe. Lookups by question_subtype first, falls back
    to ``type_of_question`` family inference.
    """
    st = (subtype or "").strip().lower()
    if st in _REQUIRED_ROLES_BY_SUBTYPE:
        return list(_REQUIRED_ROLES_BY_SUBTYPE[st])
    toq = (type_of_question or "").strip().lower()
    if toq == "therapy":
        return ["population_problem", "intervention_exposure_index_test"]
    if toq == "diagnosis":
        return ["population_problem", "intervention_exposure_index_test"]
    if toq in {"etiology_harm", "prevention"}:
        return ["intervention_exposure_index_test",
                "outcomes_target_diagnosis"]
    if toq == "dosing_protocol":
        return ["population_problem",
                "intervention_exposure_index_test",
                "clinical_action"]
    return []


def _query_blocks_from_queries(queries: list[dict]) -> list[dict]:
    """Project the persisted ``search_queries.concept_blocks_used_json``
    into a closed-set list (one entry per unique block name) so the
    panel can show which concept families fed the boolean compiler.
    NO full URLs with query strings — only the block label + the
    closed-set source identifier.
    """
    if not isinstance(queries, list):
        return []
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for q in queries:
        if not isinstance(q, dict):
            continue
        source = _clean_str(q.get("source"))
        blocks = q.get("concept_blocks_used_json") or {}
        if not isinstance(blocks, dict):
            continue
        for block_name, terms in blocks.items():
            label = _safe_query_block_label(_clean_str(block_name))
            if not label:
                continue
            key = (source, label)
            if key in seen:
                continue
            seen.add(key)
            cleaned_terms: list[str] = []
            if isinstance(terms, list):
                for t in terms[:8]:
                    v = _safe_query_block_label(_clean_str(t))
                    if v:
                        cleaned_terms.append(v)
            out.append({
                "source": source or "unknown",
                "block_name": label,
                "terms": cleaned_terms,
            })
    return out


def build_picott_term_mapping(
    session_doc: dict,
    *,
    queries: Optional[list[dict]] = None,
) -> PicottTermMapping:
    """Compose ``picott_term_mapping_json`` from an already-persisted
    session document.

    Pure projection — does NOT re-run any upstream service.
    """
    if not isinstance(session_doc, dict):
        return PicottTermMapping(fields=_empty_fields())

    picott = session_doc.get("picott_json") or {}
    anchors_payload = session_doc.get("primary_clinical_anchors_json") or {}
    primary_anchors = anchors_payload.get("primary_anchors") or []
    generic_modifiers = anchors_payload.get("generic_modifiers") or []
    anchor_failure_reason = anchors_payload.get("anchor_failure_reason")
    role_repairs = session_doc.get("role_repairs_json") or []
    ontology_lifts = session_doc.get("ontology_lifts_json") or []
    multi_intent = session_doc.get("multi_intent_json") or {}

    # Question-level fields
    type_of_question = _clean_str(picott.get("type_of_question"))
    question_subtype = _clean_str(picott.get("question_subtype")) or None
    specialty = _clean_str(picott.get("specialty")) or None
    question_scope = _clean_str(picott.get("question_scope")) or "foreground"

    # Compose all term entries from each source. Order matters for
    # dedup priority — primary_anchor wins over ontology/picott/repair.
    all_entries: list[TermMapEntry] = []

    for a in primary_anchors:
        if not isinstance(a, dict):
            continue
        e = _anchor_to_entry(a, question_subtype)
        if e.canonical_concept:
            all_entries.append(e)

    for must in (picott.get("must_have_concepts") or []):
        if not isinstance(must, dict):
            continue
        e = _must_have_to_entry(must)
        if e is not None:
            all_entries.append(e)

    all_entries.extend(_role_repair_entries(
        role_repairs, picott.get("must_have_concepts") or [],
    ))
    all_entries.extend(_ontology_lift_entries(ontology_lifts))

    for trans in (picott.get("concept_translations") or []):
        if not isinstance(trans, dict):
            continue
        e = _translation_to_entry(trans)
        if e is not None:
            all_entries.append(e)

    # Generic modifiers — ALWAYS routed into a role bucket (clinical_action
    # vs outcomes_target_diagnosis based on lexical pattern). When NO
    # primary anchor exists, this is the only signal we surface and we
    # also route into the same buckets (the clarification banner is what
    # gates retrieval in that case).
    for mod in generic_modifiers:
        text = _clean_str(mod) if isinstance(mod, str) else _clean_str(
            mod.get("text") if isinstance(mod, dict) else None
        )
        if not text:
            continue
        role = _modifier_to_role(text)
        # Phase K4A.4.1 — the ``_drop_`` sentinel comes back for purely
        # generic population labels (``patients``, ``adults`` …) that
        # must never be surfaced as a clinical_action / intent entry.
        # Skip BEFORE constructing the TermMapEntry because the
        # Pydantic schema enforces a closed-set role enum.
        if role == "_drop_":
            continue
        all_entries.append(TermMapEntry(
            original_phrase=text,
            canonical_concept=text,
            role=role,  # type: ignore[arg-type]
            synonyms_used=[],
            mesh_terms=[],
            source="picott_parser",
            required_for_high_relevance=False,
            query_block=None,
            confidence=0.4,
            rationale=("generic_modifier_only" if not primary_anchors
                       else "generic_modifier_alongside_anchor"),
        ))

    # Dedupe per (role, canonical_concept.lower()).
    deduped = _dedupe_entries(all_entries)

    # Assign each entry to its field bucket.
    fields = _empty_fields()
    for e in deduped:
        if e.role in fields:
            fields[e.role].append(e)

    # Phase 6G.1 — synthesize a "definition / disease overview" entry
    # under clinical_action when the question is a background
    # disease / drug overview AND no clinical_action entry is present.
    # This makes the clinician panel show the question intent without
    # changing any retrieval behavior.
    if (question_subtype or "").lower() in {
        "disease_overview", "definition", "drug_overview",
    } and not fields["clinical_action"]:
        fields["clinical_action"].append(TermMapEntry(
            original_phrase="what is",
            canonical_concept="definition / disease overview",
            role="clinical_action",
            synonyms_used=["definition", "overview"],
            mesh_terms=[],
            source="primary_anchor",
            required_for_high_relevance=False,
            query_block=None,
            confidence=0.85,
            rationale=f"background_subtype:{question_subtype}",
        ))

    # ----- completeness -----
    has_primary_anchor = bool(primary_anchors) and any(
        float(a.get("confidence") or 0.0) >= 0.5
        for a in primary_anchors if isinstance(a, dict)
    )
    required_roles = _resolve_required_roles(
        question_subtype, type_of_question,
    )
    missing_required_roles: list[str] = []
    for r in required_roles:
        if not fields.get(r):
            missing_required_roles.append(r)
    has_required_role_mapping = (
        bool(primary_anchors) and not missing_required_roles
    )
    safe_to_search = bool(has_primary_anchor) and not missing_required_roles
    clarification_needed = (
        anchor_failure_reason == "no_primary_clinical_entity"
        and not has_primary_anchor
    )
    clarification_message = (
        CLARIFICATION_MESSAGE if clarification_needed else None
    )

    completeness = CompletenessBlock(
        has_primary_anchor=has_primary_anchor,
        has_required_role_mapping=has_required_role_mapping,
        missing_required_roles=missing_required_roles,
        safe_to_search=safe_to_search,
        clarification_needed=clarification_needed,
        clarification_message=clarification_message,
    )

    # ----- primary / generic / must / ranking / negatives -----
    primary_clinical_anchors: list[dict] = []
    for a in primary_anchors:
        if not isinstance(a, dict):
            continue
        primary_clinical_anchors.append({
            "text": _clean_str(a.get("text")),
            "canonical_concept": _clean_str(a.get("canonical_concept")),
            "entity_type": _clean_str(a.get("entity_type")),
            "role": _clean_str(a.get("role")),
            "confidence": float(a.get("confidence") or 0.0),
            "required_for_search": bool(a.get("required_for_search")),
        })

    generic_modifier_objs: list[dict] = []
    for mod in generic_modifiers:
        text = _clean_str(mod) if isinstance(mod, str) else _clean_str(
            (mod or {}).get("text") if isinstance(mod, dict) else None
        )
        if text:
            generic_modifier_objs.append({
                "text": text,
                "routed_role": _modifier_to_role(text),
            })

    must_have_out: list[dict] = []
    for must in (picott.get("must_have_concepts") or []):
        if not isinstance(must, dict):
            continue
        c = _clean_str(must.get("concept"))
        if not c:
            continue
        must_have_out.append({
            "concept": c,
            "role": (must.get("role") or "").strip().lower(),
            "required_for_high_relevance": bool(
                must.get("required_for_high_relevance", True)
            ),
            "synonyms": [
                _clean_str(s) for s in (must.get("synonyms") or [])
                if _clean_str(s)
            ],
        })

    ranking_concepts_out: list[str] = []
    for rc in (picott.get("ranking_concepts") or []):
        if isinstance(rc, dict):
            v = _clean_str(rc.get("concept"))
            if v and v not in ranking_concepts_out:
                ranking_concepts_out.append(v)
        else:
            v = _clean_str(rc)
            if v and v not in ranking_concepts_out:
                ranking_concepts_out.append(v)

    negative_concepts_out: list[dict] = []
    for neg in (picott.get("negative_or_penalty_concepts") or []):
        if not isinstance(neg, dict):
            continue
        c = _clean_str(neg.get("concept"))
        if not c:
            continue
        negative_concepts_out.append({
            "concept": c,
            "penalty_reason": _clean_str(neg.get("penalty_reason")),
            "unless_concepts_present": [
                _clean_str(s) for s in (neg.get("unless_concepts_present") or [])
                if _clean_str(s)
            ],
        })

    query_blocks_out = _query_blocks_from_queries(queries or [])

    return PicottTermMapping(
        question_scope=question_scope,
        type_of_question=type_of_question or "other",
        question_subtype=question_subtype,
        specialty=specialty,
        fields={role: list(items) for role, items in fields.items()},
        primary_clinical_anchors=primary_clinical_anchors,
        generic_modifiers=generic_modifier_objs,
        must_have_concepts=must_have_out,
        ranking_concepts=ranking_concepts_out,
        negative_concepts=negative_concepts_out,
        query_blocks=query_blocks_out,
        completeness=completeness,
        # Phase 6G.4P — post-procedure routing surfaces.
        subdomain=_resolve_subdomain(session_doc, picott),
        refinement_note=_resolve_refinement_note(session_doc, picott),
        post_procedure_routing=_resolve_post_procedure_routing(
            session_doc, anchors_payload,
        ),
    )


def _resolve_subdomain(
    session_doc: dict, picott: dict,
) -> Optional[str]:
    """Return the closed-set post-procedure subdomain when present on
    the session document or on the PICOTT payload. ``None`` for
    non-post-procedure sessions.
    """
    # Phase 6G.4P — the pipeline persists ``subdomain`` under
    # ``primary_clinical_anchors_json.post_procedure_routing`` AND/OR
    # ``picott_json.subdomain`` (depending on session age). Read both
    # for forward + backward compatibility.
    anchors_payload = session_doc.get("primary_clinical_anchors_json") or {}
    pp = anchors_payload.get("post_procedure_routing") or {}
    if isinstance(pp, dict) and pp.get("subdomain"):
        return _clean_str(pp.get("subdomain")) or None
    s = _clean_str(picott.get("subdomain")) if isinstance(picott, dict) else ""
    return s or None


def _resolve_refinement_note(
    session_doc: dict, picott: dict,
) -> Optional[str]:
    """Return a closed-set, server-generated refinement note when the
    session is routed under a post-procedure subdomain. NEVER LLM-
    generated.
    """
    subdomain = _resolve_subdomain(session_doc, picott)
    if not subdomain:
        return None
    return _REFINEMENT_NOTES_BY_SUBDOMAIN.get(subdomain)


def _resolve_post_procedure_routing(
    session_doc: dict, anchors_payload: dict,
) -> dict:
    """Surface the closed-set post-procedure routing payload (or
    ``{}`` when not applicable). Strips any unexpected keys so the
    transparency panel only sees the contract documented in the
    anchor service.
    """
    pp = (anchors_payload or {}).get("post_procedure_routing") or {}
    if not isinstance(pp, dict) or not pp:
        return {}
    allowed = {
        "matched", "subdomain", "intent",
        "procedure_canonical", "procedure_family",
        "specialty", "related_but_distinct",
        "management_canonical", "management_entity_type",
        "management_domains",
    }
    out: dict[str, Any] = {}
    for k in allowed:
        v = pp.get(k)
        if v is None:
            continue
        out[k] = v
    return out


# Forbidden display labels (Task 5). When the new mapping says
# foreground but the legacy display still says one of these, the
# read-time self-heal replaces them.
_FORBIDDEN_FOREGROUND_LABELS: frozenset[str] = frozenset({
    "primary_care",
    "Background",
    "background",
})

_FOREGROUND_SUBTYPES: frozenset[str] = frozenset({
    "drug_safety", "adverse_effects",
    "treatment_sequencing", "first_line_treatment",
    "first-line treatment",
    "dosing_protocol", "dose_timing",
    "diagnosis", "diagnostic_accuracy",
    "vaccine_schedule", "vaccine_benefit", "benefit",
})


def is_foreground_subtype(subtype: str | None) -> bool:
    s = _clean_str(subtype).lower()
    return s in {x.lower() for x in _FOREGROUND_SUBTYPES}


def needs_display_self_heal(
    display: dict | None, mapping: PicottTermMapping,
) -> bool:
    """Decide whether the cached ``interpreted_question_display_json``
    should be regenerated from the new term mapping. Triggers when:

      1. mapping says foreground (has primary anchor) AND
      2. display carries a forbidden label ("primary_care", "Background")
         OR the display's role fields are blank while the mapping has
         non-empty role buckets.
    """
    if not isinstance(display, dict):
        return False
    if not mapping.completeness.has_primary_anchor:
        return False
    # Foreground subtype required for the self-heal trigger.
    if not is_foreground_subtype(mapping.question_subtype):
        # Still allow the heal when type_of_question is concrete (non-
        # other) since the question is foreground in those cases too.
        toq = (mapping.type_of_question or "").lower()
        if toq in {"other", ""}:
            return False
    # Forbidden labels test.
    for key in ("specialty", "type_of_question", "question_scope"):
        v = _clean_str(display.get(key))
        if v in _FORBIDDEN_FOREGROUND_LABELS:
            return True
    # Blank PICO fields test.
    blank_fields = 0
    for key in ("population", "intervention_or_exposure_or_index_test",
                "outcomes_or_target_diagnosis"):
        v = display.get(key)
        if not v or v == ["Not specified"]:
            blank_fields += 1
    if blank_fields >= 2 and any(
        mapping.fields.get(role) for role in (
            "population_problem",
            "intervention_exposure_index_test",
            "outcomes_target_diagnosis",
        )
    ):
        return True
    return False


def derive_display_from_mapping(
    base_display: dict | None,
    mapping: PicottTermMapping,
) -> dict:
    """Return a refreshed display dict by overlaying mapping-derived
    PICO fields on top of ``base_display``. Does NOT touch
    ``original_question`` / ``sanitized_question``.
    """
    out = dict(base_display or {})

    # Replace forbidden labels.
    if _clean_str(out.get("specialty")) in _FORBIDDEN_FOREGROUND_LABELS:
        out["specialty"] = (
            _clean_str(mapping.specialty) or "Not specified"
        )
    if _clean_str(out.get("type_of_question")) in _FORBIDDEN_FOREGROUND_LABELS:
        # Prefer the subtype label; fall back to type_of_question.
        out["type_of_question"] = (
            _clean_str(mapping.question_subtype)
            or _clean_str(mapping.type_of_question)
            or "Not specified"
        )
    if _clean_str(out.get("question_scope")) in _FORBIDDEN_FOREGROUND_LABELS:
        out["question_scope"] = "Foreground"

    # Fill blank PICO fields from the mapping.
    pop_entries = mapping.fields.get("population_problem") or []
    int_entries = mapping.fields.get(
        "intervention_exposure_index_test"
    ) or []
    out_entries = mapping.fields.get("outcomes_target_diagnosis") or []

    def _fill(key: str, source: list[TermMapEntry]) -> None:
        current = out.get(key)
        if (not current) or current == ["Not specified"]:
            vals = [e.canonical_concept for e in source if e.canonical_concept]
            if vals:
                out[key] = vals[:5]

    _fill("population", pop_entries)
    _fill("intervention_or_exposure_or_index_test", int_entries)
    _fill("outcomes_or_target_diagnosis", out_entries)

    return out



# ---------------------------------------------------------------------
# Phase M4.1.6 — M4 read-side projection for the PICOTT term-mapping card.
#
# When the bounded-LLM M4 active repair was APPLIED for a degraded
# baseline, the clinician "PICOTT TERM MAPPING" card MUST reflect the M4
# structured interpretation, NOT the degraded baseline fused literal.
# This builder re-shapes the SAME ``PicottTermMapping`` schema from the
# validated M4 candidate. Specialty / question_scope / query_blocks /
# post-procedure surfaces are preserved from the baseline mapping
# (query_blocks already reflect the M4-swapped queries). Display-only;
# no retrieval / ranking / selection logic is touched. Production keeps
# M4 hard-off, so this never runs there.
# ---------------------------------------------------------------------


def _m4_terms(candidate: dict, *keys: str) -> list[str]:
    """Collect + de-duplicate (case-insensitive, order-preserving) the
    string members of one-or-more M4 candidate list fields."""
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        v = candidate.get(k)
        if not isinstance(v, list):
            continue
        for item in v:
            s = _clean_str(item)
            if not s:
                continue
            kl = s.lower()
            if kl in seen:
                continue
            seen.add(kl)
            out.append(s)
    return out


def _m4_field_entries(
    terms: list[str],
    role: str,
    *,
    source: str = "picott_parser",
    confidence: float = 0.9,
) -> list[TermMapEntry]:
    return [
        TermMapEntry(
            original_phrase=t,
            canonical_concept=t,
            role=role,  # type: ignore[arg-type]
            synonyms_used=[],
            mesh_terms=[],
            source=source,  # type: ignore[arg-type]
            required_for_high_relevance=(role == "population_problem"),
            query_block=None,
            confidence=confidence,
            rationale="m4_full_question_interpreter",
        )
        for t in terms
    ]


def build_m4_picott_term_mapping(
    m4_candidate: dict,
    base_mapping: Optional[dict] = None,
    *,
    queries: Optional[list[dict]] = None,
) -> PicottTermMapping:
    """Build a ``PicottTermMapping`` from an APPLIED M4 candidate.

    Interpretation-bearing fields (type_of_question, PICOTT field rows,
    primary anchors, must-have concepts) come from the M4 candidate.
    Specialty / question_scope / query_blocks / post-procedure surfaces
    are preserved from ``base_mapping`` (the baseline projection) so the
    two clinician cards stay mutually consistent.
    """
    c = m4_candidate if isinstance(m4_candidate, dict) else {}
    base = base_mapping if isinstance(base_mapping, dict) else {}

    population = _m4_terms(c, "population") or _m4_terms(c, "clinical_problem")
    intervention = _m4_terms(
        c, "intervention", "exposure", "index_test", "medication", "procedure",
    )
    comparator = _m4_terms(c, "comparator")
    outcomes = _m4_terms(c, "outcome")
    time_horizon = _m4_terms(c, "timeframe")
    setting = _m4_terms(c, "setting")
    required = _m4_terms(c, "required_anchors")
    synonyms = _m4_terms(c, "candidate_synonyms_unvalidated")
    ranking = _m4_terms(c, "optional_or_ranking_only_terms")

    fields = _empty_fields()
    fields["population_problem"] = _m4_field_entries(
        population, "population_problem", source="primary_anchor",
    )
    fields["intervention_exposure_index_test"] = _m4_field_entries(
        intervention, "intervention_exposure_index_test",
    )
    fields["comparator_reference_standard"] = _m4_field_entries(
        comparator, "comparator_reference_standard",
    )
    fields["outcomes_target_diagnosis"] = _m4_field_entries(
        outcomes, "outcomes_target_diagnosis",
    )
    fields["time_horizon"] = _m4_field_entries(time_horizon, "time_horizon")
    fields["setting"] = _m4_field_entries(setting, "setting")
    # ``clinical_action`` is intentionally left empty — the M4 intent is
    # surfaced via ``type_of_question`` (never the legacy
    # "definition / disease overview" placeholder).

    raw_conf = c.get("confidence")
    conf = float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.9

    primary_clinical_anchors = [
        {
            "text": a,
            "canonical_concept": a,
            "entity_type": "disease_or_problem",
            "role": "population",
            "confidence": conf,
            "required_for_search": True,
        }
        for a in required
    ]

    must_have_out = [
        {
            "concept": a,
            "role": "condition",
            "required_for_high_relevance": True,
            "synonyms": list(synonyms),
        }
        for a in required
    ]

    secondary = c.get("secondary_intents") or []
    subtype = (
        _clean_str(secondary[0])
        if isinstance(secondary, list) and secondary else None
    )

    query_blocks_out = (
        base.get("query_blocks")
        if isinstance(base.get("query_blocks"), list)
        else None
    )
    if query_blocks_out is None:
        query_blocks_out = _query_blocks_from_queries(queries or [])

    has_primary = bool(required)
    completeness = CompletenessBlock(
        has_primary_anchor=has_primary,
        has_required_role_mapping=has_primary,
        missing_required_roles=[],
        safe_to_search=has_primary,
        clarification_needed=False,
        clarification_message=None,
    )

    return PicottTermMapping(
        question_scope=_clean_str(base.get("question_scope")) or "foreground",
        type_of_question=_clean_str(c.get("primary_intent")) or "other",
        question_subtype=subtype,
        specialty=_clean_str(base.get("specialty")) or None,
        fields={role: list(items) for role, items in fields.items()},
        primary_clinical_anchors=primary_clinical_anchors,
        generic_modifiers=[],
        must_have_concepts=must_have_out,
        ranking_concepts=ranking,
        negative_concepts=[],
        query_blocks=query_blocks_out,
        completeness=completeness,
        subdomain=base.get("subdomain"),
        refinement_note=base.get("refinement_note"),
        post_procedure_routing=(
            base.get("post_procedure_routing")
            if isinstance(base.get("post_procedure_routing"), dict) else {}
        ),
    )
