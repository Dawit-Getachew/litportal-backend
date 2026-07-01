"""Phase 5F Task 5 — ClinicalRoleRepairService.

Repairs PICOTT role assignments that the LLM extractor missed, using
ontology mappings as the source of truth. NEVER invents concepts — only
adjusts the ``role`` of concepts already extracted by the parser.

Pipeline position:
  PHI → spelling → shorthand → ontology mapper → PICOTT extractor →
  >> ClinicalRoleRepairService <<  →  concept translation → query compiler

Rules (closed-set):
  R1. term maps to ``drug`` / ``drug_class`` AND question contains
      effect / impact / dose / treatment / effective / lower / raise →
      assign role=``intervention``
  R2. term maps to ``biomarker`` / ``lab`` AND question contains
      impact / effect / change / level / lower / reduce / increase →
      assign role=``outcome``
  R3. term maps to ``disease`` / ``syndrome`` AND question asks
      treatment / dose / benefit / management → assign role=``population``
  R4. term maps to ``diagnostic_test`` AND question asks
      accuracy / diagnosis / rule out → assign role=``index_test``
  R5. term maps to ``vaccine`` AND question asks age / schedule →
      assign role=``intervention`` + subtype=``vaccine_schedule``
  R6. If a concept's role is missing/other AND the ontology mapper
      returned a high-confidence (>=0.8) ``role_hint`` → apply that hint
      with rule code ``ontology_role_hint``.

Output: ``RoleRepairResult`` with ``repaired_picott`` and
``repair_provenance`` (list of repair records).

Provenance record shape:
  {
    "repaired_field": "must_have_concepts[2].role"
                      | "pico.population"
                      | "pico.intervention_or_exposure_or_index_test"
                      | "pico.outcomes_or_target_diagnosis"
                      | "ranking_concepts[3].role",
    "concept": "GLP-1 receptor agonists",
    "previous_value": "context",
    "new_value": "intervention",
    "rule": "drug_class_effect_question" | "biomarker_effect_question"
            | "disease_treatment_question" | "diagnostic_test_question"
            | "vaccine_schedule_question" | "ontology_role_hint",
    "ontology_source": "curated_mesh" | "curated_rxnorm" |
                        "specialty_pack" | "clinical_shorthand" |
                        "acronym_dict",
    "confidence": float,
    "rationale": str,
  }

The repair service NEVER:
  - adds new concepts
  - removes concepts
  - rewrites the original / sanitized / normalized question text
  - alters the spelling_corrections or concept_translations lists
"""
from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from services.clinical_ontology_mapper import (
    OntologyMapping,
    clinical_ontology_mapper,
)

logger = logging.getLogger("clinevidence.role_repair")


# ---------------------------------------------------------------------------
# Question-phrase classifiers (closed-set regexes).
# ---------------------------------------------------------------------------
_EFFECT_PHRASES = re.compile(
    r"\b(?:effect|impact|affect|influence|lower|raise|reduce|increase|"
    r"improve|change|alter)\w*\b",
    re.IGNORECASE,
)
_TREATMENT_PHRASES = re.compile(
    r"\b(?:treat|treatment|manage|management|therapy|cure|dosing|dose|"
    r"effective|effectiveness|benefit|efficacy|administer)\w*\b",
    re.IGNORECASE,
)
_BIOMARKER_VERBS = re.compile(
    r"\b(?:level|levels|concentration|lower|reduce|change|improve|increase|"
    r"raise|decrease|impact|effect|affect)\w*\b",
    re.IGNORECASE,
)
_DIAGNOSTIC_PHRASES = re.compile(
    r"\b(?:accuracy|diagnos|rule out|sensitivity|specificity|"
    r"positive predictive|negative predictive|likelihood ratio)\w*\b",
    re.IGNORECASE,
)
_VACCINE_PHRASES = re.compile(
    r"\b(?:age|schedule|interval|booster|primary series|catch-up)\b",
    re.IGNORECASE,
)


# Closed-set rule codes (also surfaced in the UI).
RULES = {
    "drug_class_effect_question",
    "biomarker_effect_question",
    "disease_treatment_question",
    "diagnostic_test_question",
    "vaccine_schedule_question",
    "ontology_role_hint",
}


@dataclass
class RoleRepairResult:
    repaired_picott: dict[str, Any]
    repair_provenance: list[dict[str, Any]] = field(default_factory=list)


def _question_text(picott: dict[str, Any]) -> str:
    return (picott.get("normalized_question")
            or picott.get("original_question")
            or "")


def _classify_repair(
    mapping: OntologyMapping, question: str,
) -> tuple[str | None, str | None]:
    """Return ``(new_role, rule_code)`` for the given concept mapping
    given the question text. Returns ``(None, None)`` when no rule
    applies."""
    if not mapping or mapping.confidence < 0.65:
        return None, None
    ct = (mapping.concept_type or "").lower()
    # R1 — drug / drug_class.
    if ct in ("drug", "drug_class"):
        if _EFFECT_PHRASES.search(question) or _TREATMENT_PHRASES.search(question):
            return "intervention", "drug_class_effect_question"
    # R2 — biomarker / lab.
    if ct in ("biomarker", "lab"):
        if _BIOMARKER_VERBS.search(question) or _EFFECT_PHRASES.search(question):
            return "outcome", "biomarker_effect_question"
    # R3 — disease / syndrome.
    if ct in ("disease", "syndrome"):
        if _TREATMENT_PHRASES.search(question):
            return "population", "disease_treatment_question"
    # R4 — diagnostic test.
    if ct == "diagnostic_test":
        if _DIAGNOSTIC_PHRASES.search(question):
            return "index_test", "diagnostic_test_question"
    # R5 — vaccine.
    if ct == "vaccine":
        if _VACCINE_PHRASES.search(question):
            return "intervention", "vaccine_schedule_question"
    # R6 — high-confidence ontology role hint (last-resort).
    if mapping.confidence >= 0.8 and mapping.role_hint in {
        "intervention", "exposure", "index_test", "comparator",
        "reference_standard", "outcome", "target_diagnosis",
        "population", "condition", "patient_group",
    }:
        # Normalize legacy "condition" / "patient_group" → "population".
        new_role = mapping.role_hint
        if new_role in {"condition", "patient_group"}:
            new_role = "population"
        return new_role, "ontology_role_hint"
    return None, None


def _push_provenance(
    provenance: list[dict[str, Any]], *, field: str, concept: str,
    previous: str, new: str, rule: str,
    ontology_source: str, confidence: float, rationale: str,
) -> None:
    provenance.append({
        "repaired_field": field,
        "concept": concept,
        "previous_value": previous,
        "new_value": new,
        "rule": rule,
        "ontology_source": ontology_source,
        "confidence": round(float(confidence), 3),
        "rationale": rationale,
    })


def _pico_block_for_role(role: str) -> str | None:
    """Map a canonical role to the PICOBlocks key it belongs in."""
    mapping = {
        "population": "population",
        "condition": "population",
        "patient_group": "population",
        "intervention": "intervention_or_exposure_or_index_test",
        "exposure": "intervention_or_exposure_or_index_test",
        "index_test": "intervention_or_exposure_or_index_test",
        "comparator": "comparator_or_reference_standard",
        "reference_standard": "comparator_or_reference_standard",
        "outcome": "outcomes_or_target_diagnosis",
        "target_diagnosis": "outcomes_or_target_diagnosis",
    }
    return mapping.get(role)


def _ensure_in_pico_block(
    picott: dict[str, Any], block: str, concept: str,
) -> bool:
    """Add ``concept`` to ``picott.pico[block]`` if not already there.
    Returns True if added."""
    pico = picott.setdefault("pico", {})
    arr = pico.setdefault(block, [])
    existing = {(s or "").lower() for s in arr}
    if concept.lower() in existing:
        return False
    arr.append(concept)
    return True


class ClinicalRoleRepairService:
    """Pure / sync. Returns a new picott dict (the caller decides
    whether to overwrite the session payload)."""

    def __init__(
        self, mapper=clinical_ontology_mapper,
    ):
        self._mapper = mapper

    def repair(
        self, picott: dict[str, Any] | None,
        ontology_mappings: list[dict[str, Any]] | None = None,
    ) -> RoleRepairResult:
        if not picott or not isinstance(picott, dict):
            return RoleRepairResult(repaired_picott=picott or {},
                                     repair_provenance=[])
        out = deepcopy(picott)
        provenance: list[dict[str, Any]] = []
        question = _question_text(out)

        # Build a quick lookup of ontology mappings by concept (lower).
        mapping_index: dict[str, dict[str, Any]] = {}
        for m in (ontology_mappings or []):
            if not isinstance(m, dict):
                continue
            key = (m.get("normalized_term") or m.get("original_term") or "").lower()
            if key:
                mapping_index[key] = m

        # If no ontology mappings were passed, derive them lazily from
        # the must_have / ranking concepts via the mapper.
        if not mapping_index:
            terms_seen: set[str] = set()
            terms_to_map: list[str] = []
            for mh in (out.get("must_have_concepts") or []):
                t = (mh.get("concept") or "").strip()
                if t and t.lower() not in terms_seen:
                    terms_seen.add(t.lower())
                    terms_to_map.append(t)
            for rc in (out.get("ranking_concepts") or []):
                t = (rc.get("concept") or "").strip()
                if t and t.lower() not in terms_seen:
                    terms_seen.add(t.lower())
                    terms_to_map.append(t)
            for d in self._mapper.map_many(terms_to_map):
                mapping_index[(d.get("normalized_term")
                                or d.get("original_term") or "").lower()] = d

        # Walk must_have_concepts and apply repair rules.
        for idx, mh in enumerate(out.get("must_have_concepts") or []):
            concept = (mh.get("concept") or "").strip()
            if not concept:
                continue
            md = mapping_index.get(concept.lower())
            if not md:
                continue
            mapping = OntologyMapping(**{
                k: v for k, v in md.items()
                if k in OntologyMapping.__annotations__
            })
            new_role, rule = _classify_repair(mapping, question)
            if not new_role or not rule:
                continue
            previous = (mh.get("role") or "other").lower()
            if previous == new_role:
                # Already correct — still add to the pico block (idempotent).
                pico_block = _pico_block_for_role(new_role)
                if pico_block:
                    _ensure_in_pico_block(out, pico_block, concept)
                continue
            mh["role"] = new_role
            _push_provenance(
                provenance,
                field=f"must_have_concepts[{idx}].role",
                concept=concept, previous=previous, new=new_role, rule=rule,
                ontology_source=mapping.source,
                confidence=mapping.confidence,
                rationale=(
                    f"{rule}: ontology mapped '{concept}' as "
                    f"{mapping.concept_type}; question phrasing supports "
                    f"role={new_role}."
                ),
            )
            # Mirror into the pico block.
            pico_block = _pico_block_for_role(new_role)
            if pico_block and _ensure_in_pico_block(out, pico_block, concept):
                _push_provenance(
                    provenance,
                    field=f"pico.{pico_block}",
                    concept=concept,
                    previous="", new=concept,
                    rule=rule,
                    ontology_source=mapping.source,
                    confidence=mapping.confidence,
                    rationale=(
                        f"mirrored repaired must-have into pico.{pico_block}."
                    ),
                )

        # Walk ranking_concepts and apply rules (lower confidence is fine
        # for ranking-only role assignments).
        for idx, rc in enumerate(out.get("ranking_concepts") or []):
            concept = (rc.get("concept") or "").strip()
            if not concept:
                continue
            md = mapping_index.get(concept.lower())
            if not md:
                continue
            mapping = OntologyMapping(**{
                k: v for k, v in md.items()
                if k in OntologyMapping.__annotations__
            })
            new_role, rule = _classify_repair(mapping, question)
            if not new_role or not rule:
                continue
            previous = (rc.get("role") or "other").lower()
            if previous == new_role:
                continue
            rc["role"] = new_role
            _push_provenance(
                provenance,
                field=f"ranking_concepts[{idx}].role",
                concept=concept, previous=previous, new=new_role, rule=rule,
                ontology_source=mapping.source,
                confidence=mapping.confidence,
                rationale=(
                    f"{rule}: ranking concept role updated to {new_role}."
                ),
            )
            pico_block = _pico_block_for_role(new_role)
            if pico_block and _ensure_in_pico_block(out, pico_block, concept):
                _push_provenance(
                    provenance,
                    field=f"pico.{pico_block}",
                    concept=concept,
                    previous="", new=concept,
                    rule=rule,
                    ontology_source=mapping.source,
                    confidence=mapping.confidence,
                    rationale="mirrored repaired ranking concept into pico block.",
                )

        return RoleRepairResult(repaired_picott=out,
                                 repair_provenance=provenance)


clinical_role_repair = ClinicalRoleRepairService()


__all__ = [
    "RoleRepairResult",
    "ClinicalRoleRepairService",
    "clinical_role_repair",
    "RULES",
]
