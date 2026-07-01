"""Phase 6F — PrimaryClinicalEntityAnchorService.

Runs AFTER PHI guard + clinical_question_parser + clinical_context_engine,
and BEFORE the Boolean query compiler. Walks the PICOTT must-have
concepts + ontology mappings and emits a closed-set anchor schema.

Output schema (closed-set; persisted on the session document under
``primary_clinical_anchors_json``):

    {
      "primary_anchors": [
        {
          "text": str,
          "canonical_concept": str,
          "entity_type": <one of ENTITY_TYPES>,
          "role": <one of ANCHOR_ROLES>,
          "synonyms": [str],
          "source": <one of ANCHOR_SOURCES>,
          "confidence": float in [0, 1],
          "required_for_search": bool,
          "missing_relevance_cap": int
        },
        ...
      ],
      "generic_modifiers": [str],
      "anchor_repair_applied": bool,
      "anchor_failure_reason": str | null
    }

Hard rules (mirror the Phase 6F directive):
  - Generic modifiers ("side effects", "first-line treatment", etc.)
    are NEVER valid sole anchors. They are captured separately in
    ``generic_modifiers``.
  - If only generic modifiers are present, the service attempts
    repair using the ontology mappings already produced by
    ``clinical_context_engine``. If repair fails, the service returns
    ``anchor_failure_reason="no_primary_clinical_entity"`` and the
    pipeline will surface a clarification message to the user.
  - Deterministic. NO LLM calls. NO external API calls.
  - PHI guard runs FIRST in the route handler — this service never
    sees raw PHI text.
  - Pure / sync.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("clinevidence.primary_clinical_anchor")


# Closed-set entity types (mirror the spec).
# Phase 6G.4P — added ``procedure_or_surgery``, ``device_or_implant``,
# ``perioperative_context``, ``postprocedure_context``, and
# ``management_intervention``. The legacy ``procedure`` is retained for
# backwards-compatibility with existing PICOTT outputs (treated as a
# synonym of ``procedure_or_surgery`` by downstream consumers).
ENTITY_TYPES: frozenset[str] = frozenset({
    "disease_or_problem",
    "drug_or_drug_class",
    "vaccine",
    "diagnostic_test",
    "procedure",
    "procedure_or_surgery",
    "device_or_implant",
    "perioperative_context",
    "postprocedure_context",
    "management_intervention",
    "exposure",
    "biomarker_or_lab",
    "population_group",
    "clinical_action",
    "unknown",
})

# Closed-set anchor roles.
ANCHOR_ROLES: frozenset[str] = frozenset({
    "population", "intervention", "exposure", "index_test",
    "outcome", "clinical_action", "context",
})

# Closed-set source provenance.
# Phase H1 — added ``llm_resolver_fallback`` for anchors synthesized
# by the bounded ``ClinicalContextResolverService``. Resolver-sourced
# anchors are still subject to mandatory-anchor caps (Phase 6F.1) and
# do NOT receive the +35 ``anchor_match_full`` boost simply because
# the resolver named them — the article must also match the anchor.
ANCHOR_SOURCES: frozenset[str] = frozenset({
    "ontology_pack", "drug_pack", "mesh", "rxnorm_style", "rule",
    "llm_repair", "llm_resolver_fallback",
})

# Generic modifiers — closed-set, lower-cased. A concept whose normalized
# form appears here can NEVER be the SOLE clinical anchor.
GENERIC_MODIFIERS: frozenset[str] = frozenset({
    # Adverse-effect modifiers (Q2 family)
    "side effect", "side effects",
    "adverse effect", "adverse effects",
    "adverse reaction", "adverse reactions",
    "adverse event", "adverse events",
    "drug safety", "safety", "tolerability", "toxicity",
    # Dosing modifiers
    "dose", "doses", "dosing",
    "first dose", "second dose",
    "timing",
    # Treatment-sequencing modifiers (Q1 family)
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
    # Vaccine modifiers
    "benefit", "benefits",
    "recommended age", "vaccination age",
    "primary series", "booster", "catch-up",
    "two-dose schedule", "2-dose schedule",
    # Generic foreground modifiers
    "treatment", "therapy", "management",
    "guideline", "guidelines",
    "diagnosis", "accuracy",
    "diagnostic accuracy", "test accuracy",
    # Outcome-only words
    "outcome", "outcomes", "effectiveness", "efficacy",
    # Phase K4A.2.1 — severity-marker generic-token denylist.
    # These single tokens MUST NEVER be the sole primary clinical
    # anchor. They may still be modifiers / ranking / expansion-family
    # concepts. Multi-word disease names that CONTAIN these tokens are
    # NOT blocked because the check is on the EXACT normalized concept
    # text (e.g., "Crohn's disease" ≠ "disease", "sickle cell disease"
    # ≠ "disease"). Closed-set, deterministic, NO LLM.
    "illness", "illnesses",
    "disease", "diseases",
    "condition", "conditions",
    "problem", "problems",
    "severity",
    "marker", "markers",
    "predictor", "predictors", "prediction",
    "risk", "risks",
    "patient", "patients",
    # Common ASR/spelling-corruption regressions for the severity-
    # marker family (closed-set typo denylist — also enforced in the
    # query-compiler pre-dispatch QA via ``_MALFORMED_ANCHOR_TOKENS``).
    "iilness", "iillness", "illnes",
})

# Map ontology / PICOTT concept_type → ENTITY_TYPES.
_CONCEPT_TYPE_TO_ENTITY: dict[str, str] = {
    "drug": "drug_or_drug_class",
    "drug_class": "drug_or_drug_class",
    "disease": "disease_or_problem",
    "syndrome": "disease_or_problem",
    "biomarker": "biomarker_or_lab",
    "lab": "biomarker_or_lab",
    "diagnostic_test": "diagnostic_test",
    "vaccine": "vaccine",
    "procedure": "procedure",
    "population": "population_group",
    "exposure": "exposure",
    "outcome": "unknown",
    "other": "unknown",
}

# Map ontology / PICOTT role → ANCHOR_ROLES.
_ROLE_TO_ANCHOR_ROLE: dict[str, str] = {
    "intervention": "intervention",
    "exposure": "exposure",
    "comparator": "intervention",
    "index_test": "index_test",
    "reference_standard": "index_test",
    "outcome": "outcome",
    "target_diagnosis": "outcome",
    "population": "population",
    "condition": "population",
    "patient_group": "population",
    "context": "context",
    "other": "context",
}

# Anchor-type → cap when this anchor is REQUIRED for search but the
# article doesn't mention it. Mirrors the Phase 6F directive (Task 8).
_ENTITY_MISSING_CAP: dict[str, int] = {
    "disease_or_problem": 20,
    "drug_or_drug_class": 20,
    "vaccine": 20,
    "diagnostic_test": 30,
    "procedure": 30,
    # Phase 6G.4P — same cap as ``procedure``. Procedure_or_surgery is
    # the explicit first-class entity type; ``procedure`` is the legacy
    # alias.
    "procedure_or_surgery": 30,
    "device_or_implant": 30,
    "perioperative_context": 30,
    "postprocedure_context": 30,
    "management_intervention": 30,
    "exposure": 30,
    "biomarker_or_lab": 50,
    "population_group": 50,
    "clinical_action": 50,
    "unknown": 100,
}

# Roles eligible to be the PRIMARY anchor (i.e. drives every query).
# Outcomes / context / clinical actions can be REQUIRED but they cannot
# substitute for a disease/drug/test/procedure/vaccine.
#
# Phase 6G.4P — ``procedure_or_surgery``, ``device_or_implant``,
# ``perioperative_context``, ``postprocedure_context``, and
# ``management_intervention`` are now first-class primary-eligible
# entity types so a post-procedure question can carry TWO primary
# anchors (procedure + management).
_PRIMARY_ELIGIBLE_ENTITIES: frozenset[str] = frozenset({
    "disease_or_problem",
    "drug_or_drug_class",
    "vaccine",
    "diagnostic_test",
    "procedure",
    "procedure_or_surgery",
    "device_or_implant",
    "perioperative_context",
    "postprocedure_context",
    "management_intervention",
    "exposure",
})

# Closed-set clarification message returned to the user when no
# primary clinical entity could be identified.
CLARIFICATION_MESSAGE: str = (
    "I could not confidently identify the clinical subject of the "
    "question. Please specify the drug, condition, test, vaccine, or "
    "procedure."
)


@dataclass
class PrimaryAnchor:
    text: str
    canonical_concept: str
    entity_type: str
    role: str
    synonyms: list[str] = field(default_factory=list)
    source: str = "rule"
    confidence: float = 0.85
    required_for_search: bool = True
    missing_relevance_cap: int = 20

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "canonical_concept": self.canonical_concept,
            "entity_type": self.entity_type,
            "role": self.role,
            "synonyms": list(self.synonyms),
            "source": self.source,
            "confidence": round(float(self.confidence), 3),
            "required_for_search": bool(self.required_for_search),
            "missing_relevance_cap": int(self.missing_relevance_cap),
        }


# Generic population labels — when the PICOTT role is "population" but
# the concept text matches one of these, the entity_type stays
# ``population_group``. Anything else with role="population" + no
# ontology hit is treated as a disease/problem anchor (closed-set
# heuristic — captures "subarachnoid hemorrhage", "psoriasis",
# "AMI patients", etc.).
_GENERIC_POPULATION_LABELS: frozenset[str] = frozenset({
    "adults", "adult", "patients", "patient", "children", "child",
    "men", "women", "male", "female",
    "elderly", "older adults", "older adult",
    "pediatric", "paediatric", "pediatrics",
    "neonates", "neonate", "infants", "infant",
    "general population",
    "icu patients", "ed patients",
    "outpatient", "outpatients", "inpatient", "inpatients",
})


def _is_generic_modifier(text: str) -> bool:
    if not text:
        return False
    return text.strip().lower() in GENERIC_MODIFIERS


def _detect_generic_modifiers(question: str) -> list[str]:
    """Return the list of generic modifier phrases that appear in the
    question text. Used for transparency only — these are never sole
    anchors.
    """
    if not question:
        return []
    out: list[str] = []
    q = question.lower()
    for mod in GENERIC_MODIFIERS:
        if re.search(rf"\b{re.escape(mod)}\b", q):
            if mod not in out:
                out.append(mod)
    return out


def _entity_type_for(role: str, concept_type: str | None, *, concept_text: str = "") -> str:
    """Resolve a PICOTT role + ontology concept_type into a closed-set
    entity_type. Concept_type wins when present; role disambiguates.

    Phase 6F — when role="population" but concept_type is missing AND
    the concept text is NOT a generic population label
    (``_GENERIC_POPULATION_LABELS``), default to ``disease_or_problem``.
    This handles cases where role_repair has rewritten role="condition"
    → role="population" for therapy/treatment questions (PICOTT places
    the target disease in the population slot), and the disease has no
    specialty-pack ontology hit (e.g., "subarachnoid hemorrhage"
    without a neurocrit pack).
    """
    ct = (concept_type or "").lower().strip()
    # Phase 6F — only let concept_type override when it's a SPECIFIC
    # closed-set category. "other" / "unknown" / "" should fall through
    # to the role-based dispatch (e.g., MeSH returned "Subarachnoid
    # Hemorrhage" with concept_type="other" because the descriptor
    # category-to-clinical-concept-type table doesn't cover it — but
    # the PICOTT role of "condition" is more authoritative here).
    if ct and ct not in {"other", "unknown"} and ct in _CONCEPT_TYPE_TO_ENTITY:
        return _CONCEPT_TYPE_TO_ENTITY[ct]
    r = (role or "").lower().strip()
    if r in {"condition", "population", "patient_group"}:
        if r == "condition":
            return "disease_or_problem"
        # role == "population" or "patient_group" — disambiguate via
        # the closed-set generic-population label list.
        text = (concept_text or "").strip().lower()
        if text and text not in _GENERIC_POPULATION_LABELS:
            return "disease_or_problem"
        return "population_group"
    if r in {"intervention", "comparator"}:
        return "drug_or_drug_class"
    if r in {"exposure"}:
        return "exposure"
    if r in {"index_test", "reference_standard"}:
        return "diagnostic_test"
    if r in {"outcome", "target_diagnosis"}:
        return "biomarker_or_lab"
    return "unknown"


def _ontology_index(ontology_mappings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a lower-cased lookup over (original_term, normalized_term,
    canonical_concept, synonyms) → ontology mapping dict.
    """
    out: dict[str, dict[str, Any]] = {}
    for m in (ontology_mappings or []):
        if not isinstance(m, dict):
            continue
        for k in (m.get("original_term"), m.get("normalized_term"),
                  m.get("canonical_concept")):
            if k and isinstance(k, str):
                out.setdefault(k.lower(), m)
        for s in (m.get("synonyms") or []):
            if s and isinstance(s, str):
                out.setdefault(s.lower(), m)
    return out


def _build_anchor_from_mh(
    mh: dict[str, Any],
    ontology_index: dict[str, dict[str, Any]],
) -> PrimaryAnchor | None:
    """Build a PrimaryAnchor from a single must_have_concept entry.
    Returns None for entries whose concept is empty or is a generic
    modifier with no ontology rescue.
    """
    concept = (mh.get("concept") or "").strip()
    if not concept:
        return None
    role = (mh.get("role") or "other").lower()
    syns = list(mh.get("synonyms") or [])

    # Look up the ontology mapping for this concept (canonical or syn).
    ont = ontology_index.get(concept.lower())
    if ont is None:
        for s in syns:
            ont = ontology_index.get((s or "").lower())
            if ont:
                break

    concept_type = (ont or {}).get("concept_type") or ""
    confidence = float((ont or {}).get("confidence") or 0.0)
    canonical_concept = (ont or {}).get("canonical_concept") or concept
    source = "ontology_pack" if ont else "rule"

    # Generic-modifier rescue: if the concept text is itself a generic
    # modifier AND we have no ontology mapping, skip it as an anchor.
    if _is_generic_modifier(concept) and not ont:
        return None

    entity_type = _entity_type_for(role, concept_type, concept_text=canonical_concept)
    anchor_role = _ROLE_TO_ANCHOR_ROLE.get(role, "context")
    # missing-cap is closed-set per entity_type.
    cap = _ENTITY_MISSING_CAP.get(entity_type, 20)
    # required_for_search: primary clinical entities + intervention/
    # exposure/index_test are required; outcomes/context are not.
    is_primary_eligible = entity_type in _PRIMARY_ELIGIBLE_ENTITIES
    required = bool(
        is_primary_eligible
        or anchor_role in {"intervention", "exposure", "index_test"}
        or bool(mh.get("required_for_high_relevance"))
    )
    return PrimaryAnchor(
        text=concept,
        canonical_concept=canonical_concept,
        entity_type=entity_type,
        role=anchor_role,
        synonyms=[s for s in syns if s],
        source=source,
        confidence=(confidence if ont else 0.85),
        required_for_search=required,
        missing_relevance_cap=cap,
    )


def _attempt_repair_from_ontology(
    ontology_mappings: list[dict[str, Any]],
) -> list[PrimaryAnchor]:
    """When the LLM PICOTT returned only generic modifiers, look for
    high-confidence primary-eligible ontology hits and lift them into
    anchors. Closed-set rules — only ``disease/drug/drug_class/vaccine/
    procedure/diagnostic_test`` concept_types are eligible.
    """
    out: list[PrimaryAnchor] = []
    for m in (ontology_mappings or []):
        if not isinstance(m, dict):
            continue
        ct = (m.get("concept_type") or "").lower()
        if ct not in {"drug", "drug_class", "disease", "syndrome",
                       "vaccine", "procedure", "diagnostic_test"}:
            continue
        confidence = float(m.get("confidence") or 0.0)
        if confidence < 0.85:
            continue
        canonical = (m.get("canonical_concept") or "").strip()
        if not canonical:
            continue
        role_hint = (m.get("role_hint") or "other").lower()
        entity_type = _entity_type_for(role_hint, ct, concept_text=canonical)
        anchor_role = _ROLE_TO_ANCHOR_ROLE.get(role_hint, "context")
        cap = _ENTITY_MISSING_CAP.get(entity_type, 20)
        out.append(PrimaryAnchor(
            text=canonical,
            canonical_concept=canonical,
            entity_type=entity_type,
            role=anchor_role,
            synonyms=list(m.get("synonyms") or []),
            source="llm_repair",
            confidence=confidence,
            required_for_search=True,
            missing_relevance_cap=cap,
        ))
    return out


# Phase 6F.2 (clean-up) — pre-PICOTT anchor pre-flight allowlist.
# Tokens that count as question-opener / function words. If, after
# stripping every closed-set ``GENERIC_MODIFIERS`` substring from the
# (already spelling+shorthand-normalized) question, every leftover
# token is in this allowlist, the question carries NO candidate
# clinical noun and the LLM PICOTT extractor SHOULD NOT fire. Closed
# set — additions require a regression test.
_QUESTION_OPENER_TOKENS: frozenset[str] = frozenset({
    # Wh-words + auxiliaries
    "what", "which", "when", "how", "where", "why", "who", "whom",
    "is", "are", "was", "were", "am", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "can", "could", "should", "would", "will", "shall", "may", "might",
    # Determiners / articles / pronouns
    "the", "a", "an",
    "this", "that", "these", "those",
    "my", "your", "his", "her", "their", "our",
    # Common prepositions / conjunctions / particles
    "of", "for", "in", "on", "to", "with", "without",
    "and", "or", "but", "vs", "versus",
    "any", "all", "some",
    # Misc question fragments
    "best", "good", "most", "least",
    "by", "as", "at",
})


class PrimaryClinicalEntityAnchorService:
    """Phase 6F — derive primary clinical entity anchors.

    Stateless / pure. Call ``derive(picott, ontology_mappings,
    normalized_question)`` and persist the result on the session.

    Phase 6F.2 clean-up — exposes a deterministic ``is_generic_only``
    pre-flight that the pipeline orchestrator calls BEFORE the LLM
    PICOTT extractor. The pre-flight uses ONLY the closed-set
    ``GENERIC_MODIFIERS`` + ``_QUESTION_OPENER_TOKENS`` allowlist —
    no LLM, no ontology lookup. When it returns ``True`` the pipeline
    short-circuits to ``awaiting_clinical_subject`` and the PICOTT
    LLM call is skipped entirely.
    """

    def is_generic_only(self, question: str) -> bool:
        """Deterministic anchor pre-flight.

        Returns ``True`` iff the (already spelling+shorthand-normalized)
        question consists EXCLUSIVELY of closed-set generic modifiers
        plus question-opener / function words. No clinical noun
        candidate is present → safe to skip the PICOTT LLM extractor.

        Closed-set. Never raises. Never makes an LLM call.

        Phase 6G.4P — when the post-procedure management router
        confidently matches the question (≥1 detected procedure +
        management phrase pair), the question carries TWO primary
        anchors. We MUST NOT short-circuit to clarification in that
        case — return ``False`` so the pipeline runs the deterministic
        post-procedure anchor injection in :py:meth:`derive`. Closed-
        set, deterministic, no LLM, no network.
        """
        # Phase 6G.4P — post-procedure pattern detection runs FIRST.
        try:
            from services.post_procedure_router import post_procedure_router
            from services.procedure_ontology_mapper import (
                procedure_ontology_mapper,
            )
            ppm = post_procedure_router.detect(question or "")
            if ppm.get("matched") and ppm.get("procedure_phrase"):
                resolved = procedure_ontology_mapper.resolve_procedure(
                    ppm["procedure_phrase"],
                )
                if resolved is not None:
                    return False
        except Exception:  # noqa: BLE001
            # Defensive: never let the post-procedure path break the
            # closed-set pre-flight. Log at debug-level only — this
            # path is hit on every clarification-bound question.
            logger.debug("phase6g4p post-procedure pre-flight failed",
                         exc_info=True)

        # Phase 6G.4P — closed-set "management without procedure
        # context" clarification trigger. Catches the Task 9.F
        # regression cases ("what are the recommendations?",
        # "what anticoagulation is recommended?", "what is
        # postoperative management?") deterministically WITHOUT
        # affecting the 5 Phase 6F.2 closed-set fast-path questions
        # (which still hit the legacy GENERIC_MODIFIERS path below).
        if _matches_management_without_context(question or ""):
            return True

        q = (question or "").lower().strip().rstrip("?.!")
        # Empty / whitespace-only — no clinical signal → safe to skip
        # PICOTT (the router would normally reject empty input, but
        # the function stays graceful).
        if not q:
            return True
        # Remove every multi-word generic modifier first (longest first
        # so "side effects" is stripped before "effects").
        for mod in sorted(GENERIC_MODIFIERS, key=len, reverse=True):
            q = re.sub(rf"\b{re.escape(mod)}\b", " ", q)
        # Tokenize leftover. If anything remains that is NOT a
        # closed-set opener / function word, the question carries a
        # candidate clinical noun → PICOTT must run.
        leftover = [t for t in re.split(r"[^a-z0-9]+", q) if t]
        if not leftover:
            return True
        for tok in leftover:
            if tok not in _QUESTION_OPENER_TOKENS:
                return False
        return True

    def derive(
        self,
        *,
        picott: dict[str, Any] | None,
        ontology_mappings: list[dict[str, Any]] | None = None,
        normalized_question: str = "",
    ) -> dict[str, Any]:
        picott = picott or {}
        ontology_mappings = list(ontology_mappings or [])
        ontology_idx = _ontology_index(ontology_mappings)

        # 1. Build anchors from must_have_concepts.
        anchors: list[PrimaryAnchor] = []
        seen_keys: set[str] = set()
        for mh in (picott.get("must_have_concepts") or []):
            a = _build_anchor_from_mh(mh, ontology_idx)
            if a is None:
                continue
            key = a.canonical_concept.lower()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            anchors.append(a)

        # 2. Detect generic modifiers in the question text (for the
        #    transparency panel — these are NEVER anchors).
        generic_modifiers = _detect_generic_modifiers(normalized_question)

        # 3. Check whether at least one primary-eligible anchor exists.
        has_primary_eligible = any(
            a.entity_type in _PRIMARY_ELIGIBLE_ENTITIES for a in anchors
        )
        anchor_repair_applied = False
        anchor_failure_reason: str | None = None
        picott_overrides: dict[str, Any] = {}

        # Phase 6G.4P — post-procedure management routing.
        # Runs BEFORE the legacy ontology / background-definition
        # repair passes so a confident post-procedure pattern match
        # always wins over the conservative noun-phrase fallback. The
        # detector is deterministic + closed-set; when it fires it
        # injects TWO primary anchors (procedure + management) and
        # the closed-set picott_overrides (question_subtype =
        # post_procedure_management, subdomain, specialty, intent).
        post_procedure_payload: dict[str, Any] = {}
        if normalized_question:
            pp = _detect_post_procedure_anchors(normalized_question)
            if pp is not None:
                inject_anchors, pp_overrides, pp_payload = pp
                for a in inject_anchors:
                    k = a.canonical_concept.lower()
                    if k in seen_keys:
                        # Upgrade entity_type on the already-present
                        # anchor so the UI / ranker / closed-set lane
                        # logic sees the Phase 6G.4P first-class type
                        # (procedure_or_surgery / management_intervention)
                        # rather than the legacy ``procedure`` tag the
                        # LLM extractor may have produced.
                        for existing in anchors:
                            if existing.canonical_concept.lower() == k:
                                if a.entity_type in (
                                    "procedure_or_surgery",
                                    "management_intervention",
                                    "device_or_implant",
                                ):
                                    existing.entity_type = a.entity_type
                                    existing.required_for_search = True
                                break
                        continue
                    seen_keys.add(k)
                    anchors.append(a)
                if pp_overrides:
                    picott_overrides = dict(pp_overrides)
                post_procedure_payload = dict(pp_payload)
                has_primary_eligible = any(
                    a.entity_type in _PRIMARY_ELIGIBLE_ENTITIES
                    for a in anchors
                )
                anchor_repair_applied = True

        # 4. Attempt repair from ontology mappings when no primary-
        # eligible anchor is present.
        if not has_primary_eligible:
            repaired = _attempt_repair_from_ontology(ontology_mappings)
            for a in repaired:
                k = a.canonical_concept.lower()
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                anchors.append(a)
            has_primary_eligible = any(
                a.entity_type in _PRIMARY_ELIGIBLE_ENTITIES for a in anchors
            )
            if repaired and has_primary_eligible:
                anchor_repair_applied = True

        # 5. Phase 6G.1 — background-definition repair. When the
        # PICOTT extractor produced no must_have concepts AND the
        # ontology repair also failed AND the question matches a
        # "What is [X]?" / "Define [X]" / etc. pattern, resolve ``[X]``
        # against the closed-set specialty packs / acronym table /
        # conservative noun-phrase rule. Generic-only subjects
        # ("treatment", "side effects", etc.) are explicitly excluded
        # by ``_subject_is_generic_only`` so the existing fast-path
        # still wins for true generic-only questions.
        if not has_primary_eligible and normalized_question:
            subject = _extract_definition_subject(normalized_question)
            if subject and not _subject_is_generic_only(subject):
                resolved = _resolve_background_anchor(subject)
                if resolved is not None:
                    bg_anchor, overrides = resolved
                    k = bg_anchor.canonical_concept.lower()
                    if k not in seen_keys:
                        seen_keys.add(k)
                        anchors.append(bg_anchor)
                    has_primary_eligible = any(
                        a.entity_type in _PRIMARY_ELIGIBLE_ENTITIES
                        for a in anchors
                    )
                    if has_primary_eligible:
                        anchor_repair_applied = True
                        # Phase 6G.3 — when the question is a "What
                        # is X?" definition AND no treatment trigger
                        # phrase is present, FORCE
                        # ``question_subtype=disease_overview``
                        # regardless of pack hint.
                        if not _question_has_treatment_trigger(
                            normalized_question,
                        ):
                            overrides = dict(overrides)
                            overrides["question_subtype"] = "disease_overview"
                        picott_overrides = dict(overrides)

        # 5b. Phase 6G.1 — even when step 1 / step 4 already produced
        # a primary-eligible anchor, propagate the
        # background-definition picott_overrides (specialty,
        # question_subtype, question_scope) when the question matches
        # a "What is [X]?" pattern AND the resolved subject matches
        # one of the existing anchors. This is what corrects the
        # "specialty=primary_care" leak for legacy paths that already
        # had the ontology pack supplying the anchor.
        if has_primary_eligible and not picott_overrides and normalized_question:
            subject = _extract_definition_subject(normalized_question)
            if subject and not _subject_is_generic_only(subject):
                resolved = _resolve_background_anchor(subject)
                if resolved is not None:
                    _bg_anchor, overrides = resolved
                    bg_canon = _bg_anchor.canonical_concept.lower()
                    if any(
                        a.canonical_concept.lower() == bg_canon
                        for a in anchors
                    ):
                        # Phase 6G.3 — same disease_overview override
                        # as in step 5: when the question carries no
                        # treatment trigger phrase, FORCE the subtype
                        # to ``disease_overview`` even if the pack
                        # hint says treatment_sequencing /
                        # drug_safety / etc. This is the canonical
                        # fix for the "What is psoriasis?" →
                        # treatment_sequencing label leak.
                        if not _question_has_treatment_trigger(
                            normalized_question,
                        ):
                            overrides = dict(overrides)
                            overrides["question_subtype"] = "disease_overview"
                        picott_overrides = dict(overrides)

        if not has_primary_eligible:
            anchor_failure_reason = "no_primary_clinical_entity"

        return {
            "primary_anchors": [a.to_dict() for a in anchors],
            "generic_modifiers": list(generic_modifiers),
            "anchor_repair_applied": bool(anchor_repair_applied),
            "anchor_failure_reason": anchor_failure_reason,
            "picott_overrides": dict(picott_overrides),
            # Phase 6G.4P — closed-set post-procedure routing payload.
            # Empty dict when no post-procedure pattern matched. Used
            # by the pipeline to persist subdomain + lane configuration
            # on the session document for downstream lane labeling.
            "post_procedure_routing": dict(post_procedure_payload),
        }


# ---------------------------------------------------------------------
# Phase 6G.1 — background-definition anchor repair.
# ---------------------------------------------------------------------
#
# When the PICOTT LLM extractor classifies a question as ``background``
# it routinely emits an empty ``must_have_concepts`` list (it treats
# definition / overview questions as classification-only). The anchor
# service then finds no primary-eligible anchor and the pipeline
# clarifies — which is WRONG for a question like "What is acne
# vulgaris?" where the disease name IS the clinical subject.
#
# The repair below detects "What is [X]?" / "Define [X]" / etc. before
# the final anchor-failure check fires, extracts ``[X]``, and resolves
# it against the closed-set specialty packs and a curated acronym map.
# A conservative noun-phrase fallback (no LLM) covers diseases not yet
# in a pack. The generic-only guard still wins when ``[X]`` resolves
# to a closed-set generic modifier.

_BACKGROUND_DEFINITION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?:what\s+(?:is|are|do|does)|tell\s+me\s+about|explain|"
        r"describe|overview\s+of|definition\s+of|define|"
        r"what(?:'s|s)?)\s+(?:the\s+|a\s+|an\s+)?(?P<x>.+?)\??$",
        re.IGNORECASE,
    ),
)

# Phase 6G.3 — closed-set treatment-trigger phrases. When ANY of these
# is present in the normalized question, the question is NOT a pure
# disease-definition query and we MUST NOT force ``subtype=disease_overview``
# even when the surface "What is" pattern matches.
_TREATMENT_TRIGGER_PHRASES: tuple[str, ...] = (
    "first-line treatment", "first line treatment",
    "second-line treatment", "second line treatment",
    "treatment", "therapy", "management",
    "treatment algorithm", "treatment sequence",
    "recommended treatment", "standard treatment",
    "sequencing", "dose", "doses", "dosing",
    "regimen", "protocol",
    "drug safety", "adverse effects", "side effects",
    "vaccine schedule", "schedule",
)


def _question_has_treatment_trigger(question: str) -> bool:
    """Return True iff ``question`` contains any closed-set treatment
    trigger phrase. Used by step 5b to refuse a ``treatment_sequencing``
    pack-hint override when the question is actually "What is X?" without
    any treatment language."""
    if not question:
        return False
    q = question.lower()
    return any(p in q for p in _TREATMENT_TRIGGER_PHRASES)


# Phase 6G.3 — closed-set comparison / association triggers. Drive
# the query compiler decision to AND two disease anchors together
# (only when one of these is present in the question).
_COMPARISON_TRIGGER_PHRASES: tuple[str, ...] = (
    "compare", "comparison", " vs ", " versus ",
    "differential", "association between", "associated with",
    "comorbid", "comorbidity",
    "overlap between", "relationship between",
    "difference between", "similar to",
)


def question_has_comparison_trigger(question: str) -> bool:
    """Return True iff ``question`` contains any closed-set comparison /
    association trigger phrase. Comparison/association mode is the
    ONLY route in which the boolean compiler is allowed to AND two
    independent disease anchors together."""
    if not question:
        return False
    q = question.lower()
    return any(p in q for p in _COMPARISON_TRIGGER_PHRASES)


# Phase 6G.3 — closed-set conjoined-pattern detector. Matches things
# like "What is X and what is Y?" / "Define X and define Y" — the
# user wants two independent background-definition searches, NOT a
# joint AND query.
_CONJOINED_BACKGROUND_RE = re.compile(
    r"^\s*"
    r"(?:what\s+(?:is|are)|define|tell\s+me\s+about|explain|overview\s+of)\s+"
    r"(?P<a>.+?)"
    r"\s+and\s+"
    r"(?:what\s+(?:is|are)|define|tell\s+me\s+about|explain|overview\s+of)\s+"
    r"(?P<b>.+?)\??\s*$",
    re.IGNORECASE,
)


def detect_conjoined_background_questions(question: str) -> list[str] | None:
    """If ``question`` matches "What is A and what is B?" pattern,
    return ``[A, B]`` (each as a standalone background subquestion).
    Returns ``None`` for non-conjoined questions."""
    if not question:
        return None
    m = _CONJOINED_BACKGROUND_RE.match(question.strip().strip("."))
    if not m:
        return None
    a = (m.group("a") or "").strip().strip("?.,").strip()
    b = (m.group("b") or "").strip().strip("?.,").strip()
    if not a or not b:
        return None
    return [a, b]

# Conservative-fallback stopwords. If the extracted ``[X]`` consists
# exclusively of these tokens (plus generic modifiers), no anchor is
# emitted and the generic-only path still wins.
_DEFINITION_FALLBACK_STOPWORDS: frozenset[str] = frozenset({
    "thing", "things", "term", "terms", "concept", "concepts",
    "subject", "topic", "topics", "matter", "issue", "issues",
    "definition", "meaning", "overview", "explanation",
    "stuff", "way", "ways", "type", "types", "kind", "kinds",
    "best", "worst", "good", "bad", "common", "rare",
})


# Phase J2 — closed-set clinical-setting tokens. When the captured
# subject of a "What is the [X]?" framing contains only generic
# modifiers + setting tokens (e.g., "treatment in ICU",
# "management in ICU patients", "care in the ICU"), the subject is
# insufficient by itself — clinical_setting alone CANNOT satisfy
# the clinical-subject requirement (brief J2 rule 3). These cases
# fall through to ``anchor_failure_reason=no_primary_clinical_entity``
# so the bounded LLM resolver is given a chance to extract the
# missing intervention / problem.
_CLINICAL_SETTING_ONLY_TOKENS: frozenset[str] = frozenset({
    "icu",
    "intensive", "care", "unit",  # "intensive care unit"
    "critical", "critically", "ill",  # "critically ill"
    "ed", "emergency", "department",  # "emergency department"
    "ward", "wards",
    "outpatient", "outpatients", "inpatient", "inpatients",
    "primary",  # "primary care" — also primary as adjective
    "perioperative", "preoperative", "postoperative",
    "intraoperative",
    "ambulatory", "clinic",
})


# Phase J1 — closed-set "modifier + subject" framing prefixes. When a
# captured definition subject starts with one of these, it is a
# foreground clinical question (risk/complication/safety/benefits)
# wrapped in a "What is the ..." phrasing — NOT a background-
# definition question. The conservative noun-phrase fallback in
# ``_resolve_background_anchor`` returns None for these so the
# pipeline falls through to ``anchor_failure_reason=
# no_primary_clinical_entity``, which is the existing trigger for the
# bounded LLM Clinical Context Resolver. NO exact-question
# hardcoding — pure prefix match.
_MODIFIER_PREFIX_PATTERNS: tuple[str, ...] = (
    "risk of ",
    "risks of ",
    "benefits and risks of ",
    "benefits of ",
    "complication of ",
    "complications of ",
    "complications after ",
    "complications following ",
    "side effects of ",
    "side effect of ",
    "adverse effects of ",
    "adverse events of ",
    "harm of ",
    "harms of ",
    "outcomes of ",
    "outcome of ",
    "evidence for ",
    "evidence of ",
    "efficacy of ",
    "effectiveness of ",
    "safety of ",
)


def _extract_definition_subject(question: str) -> str | None:
    """Return the candidate subject ``[X]`` if ``question`` matches a
    background-definition pattern, else ``None``. Returns ``None`` for
    empty / whitespace-only candidates."""
    if not question:
        return None
    q = question.strip().strip(".!").strip()
    for pat in _BACKGROUND_DEFINITION_PATTERNS:
        m = pat.match(q)
        if not m:
            continue
        x = (m.group("x") or "").strip().strip("?.,").strip()
        # Strip a leading generic article that survived the optional
        # group (defensive).
        x = re.sub(r"^(?:the|a|an)\s+", "", x, flags=re.IGNORECASE).strip()
        if not x:
            return None
        return x
    return None


def _subject_is_generic_only(subject: str) -> bool:
    """A candidate ``[X]`` is generic-only when, after stripping any
    multi-word ``GENERIC_MODIFIERS``, the leftover is empty OR all
    leftover tokens are either question-openers,
    ``_DEFINITION_FALLBACK_STOPWORDS``, or — Phase J2 — closed-set
    clinical-setting tokens (clinical_setting alone is insufficient
    to satisfy the clinical-subject requirement)."""
    s = (subject or "").lower().strip()
    if not s:
        return True
    # Strip every multi-word generic modifier first (longest first).
    for mod in sorted(GENERIC_MODIFIERS, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(mod)}\b", " ", s)
    leftover = [t for t in re.split(r"[^a-z0-9\-]+", s) if t]
    if not leftover:
        return True
    for tok in leftover:
        if tok in _QUESTION_OPENER_TOKENS:
            continue
        if tok in _DEFINITION_FALLBACK_STOPWORDS:
            continue
        # Phase J2 — clinical-setting tokens alone are insufficient.
        if tok in _CLINICAL_SETTING_ONLY_TOKENS:
            continue
        return False
    return True


# Acronym / shorthand expansion table for background anchors. Closed
# set; ANY addition requires a regression test. The shape mirrors
# ``concept_expansion.ACRONYM_PAIRS`` deliberately so we stay
# auditable.
_BACKGROUND_ACRONYMS: dict[str, str] = {
    "sah": "subarachnoid hemorrhage",
    "ami": "acute myocardial infarction",
    "ards": "acute respiratory distress syndrome",
    "copd": "chronic obstructive pulmonary disease",
    "ckd": "chronic kidney disease",
    "aki": "acute kidney injury",
    "hf": "heart failure",
    "chf": "congestive heart failure",
    "hfpef": "heart failure with preserved ejection fraction",
    "hfref": "heart failure with reduced ejection fraction",
    "af": "atrial fibrillation",
    "afib": "atrial fibrillation",
    "vte": "venous thromboembolism",
    "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis",
    "tia": "transient ischemic attack",
    "t2dm": "type 2 diabetes",
    "t1dm": "type 1 diabetes",
    "ibd": "inflammatory bowel disease",
    "uc": "ulcerative colitis",
    "gerd": "gastroesophageal reflux disease",
    "hiv": "human immunodeficiency virus",
    "tb": "tuberculosis",
    "ich": "intracerebral hemorrhage",
    "ais": "ischemic stroke",
    "dci": "delayed cerebral ischemia",
    "esrd": "end-stage renal disease",
}

# Drug brand → generic for the most common background-question targets.
# Closed-set, NOT a fuzzy match. The full RxNorm-style lookup lives
# elsewhere; this is the minimum surface needed for "What is
# Seroquel?" type background questions.
_BACKGROUND_BRAND_TO_GENERIC: dict[str, tuple[str, list[str]]] = {
    "seroquel": ("quetiapine", ["Seroquel"]),
    "abilify": ("aripiprazole", ["Abilify"]),
    "lipitor": ("atorvastatin", ["Lipitor"]),
    "crestor": ("rosuvastatin", ["Crestor"]),
    "humira": ("adalimumab", ["Humira"]),
    "enbrel": ("etanercept", ["Enbrel"]),
    "stelara": ("ustekinumab", ["Stelara"]),
    "ozempic": ("semaglutide", ["Ozempic", "Wegovy"]),
    "wegovy": ("semaglutide", ["Wegovy", "Ozempic"]),
    "rybelsus": ("semaglutide", ["Rybelsus"]),
    "trulicity": ("dulaglutide", ["Trulicity"]),
}


def _resolve_background_anchor(
    subject: str,
) -> tuple[PrimaryAnchor, dict[str, Any]] | None:
    """Resolve a background-question subject ``[X]`` to a closed-set
    anchor. Returns ``(anchor, picott_overrides)`` on success, ``None``
    when the subject cannot be confidently resolved.

    Resolution order:
      1. Specialty-pack concept lookup (e.g., acne vulgaris, psoriasis,
         atrial fibrillation).
      2. Acronym / brand-name expansion (closed set above).
      3. Conservative noun-phrase fallback — only when the subject is
         not a generic modifier, looks like a biomedical noun phrase
         (alphanumeric + space + hyphen), length ≥ 3, and the tokens
         are NOT all question-openers / stopwords.

    NEVER calls an LLM. NEVER hits the network.
    """
    raw = (subject or "").strip()
    if not raw:
        return None
    low = raw.lower()

    # Quick reject — pure generic modifier OR stopword-only.
    if _is_generic_modifier(low) or _subject_is_generic_only(low):
        return None

    # --- 1. Specialty pack lookup -----------------------------------
    try:
        from services.specialty_packs import all_packs  # local import
        packs = all_packs()
    except Exception:  # noqa: BLE001 — pack loader must not crash
        packs = {}
    for pack_name, pack in (packs or {}).items():
        groups = (pack or {}).get("concept_groups") or {}
        for _gname, group in groups.items():
            if not isinstance(group, dict):
                continue
            related = bool(group.get("related_but_distinct"))
            for entry in (group.get("concepts") or []):
                if not isinstance(entry, dict):
                    continue
                concept = (entry.get("concept") or "").strip()
                if not concept:
                    continue
                syns = [
                    s for s in (entry.get("synonyms") or [])
                    if isinstance(s, str) and s
                ]
                hit = (
                    low == concept.lower()
                    or low in {s.lower() for s in syns}
                )
                if not hit or related:
                    continue
                concept_type = (group.get("concept_type") or "disease").lower()
                role_hint = (group.get("role_hint") or "condition").lower()
                # Disease/syndrome packs always emit
                # entity_type=disease_or_problem with role=population.
                # Drug packs emit drug_or_drug_class. etc.
                ent_type = _CONCEPT_TYPE_TO_ENTITY.get(concept_type, "")
                if not ent_type:
                    ent_type = "disease_or_problem"
                role = _ROLE_TO_ANCHOR_ROLE.get(role_hint, "population")
                if ent_type not in _PRIMARY_ELIGIBLE_ENTITIES:
                    continue
                anchor = PrimaryAnchor(
                    text=raw,
                    canonical_concept=concept,
                    entity_type=ent_type,
                    role=role,
                    synonyms=[concept] + [s for s in syns if s != concept],
                    source="ontology_pack",
                    confidence=float(
                        entry.get("ontology_confidence", 0.9)
                    ),
                    required_for_search=True,
                    missing_relevance_cap=_ENTITY_MISSING_CAP.get(
                        ent_type, 30,
                    ),
                )
                subtype_hint = (
                    group.get("question_subtype_hint")
                    or "disease_overview"
                )
                specialty = (pack or {}).get("specialty") or None
                return anchor, {
                    "question_subtype": subtype_hint,
                    "type_of_question": "background",
                    "question_scope": "background",
                    "specialty": specialty,
                }

    # --- 2. Acronym / brand expansion ------------------------------
    if low in _BACKGROUND_ACRONYMS:
        canonical = _BACKGROUND_ACRONYMS[low]
        anchor = PrimaryAnchor(
            text=raw,
            canonical_concept=canonical,
            entity_type="disease_or_problem",
            role="population",
            synonyms=[canonical, raw.upper()],
            source="rule",
            confidence=0.88,
            required_for_search=True,
            missing_relevance_cap=_ENTITY_MISSING_CAP[
                "disease_or_problem"
            ],
        )
        return anchor, {
            "question_subtype": "disease_overview",
            "type_of_question": "background",
            "question_scope": "background",
            "specialty": None,
        }
    if low in _BACKGROUND_BRAND_TO_GENERIC:
        canonical, syns = _BACKGROUND_BRAND_TO_GENERIC[low]
        anchor = PrimaryAnchor(
            text=raw,
            canonical_concept=canonical,
            entity_type="drug_or_drug_class",
            role="intervention",
            synonyms=[canonical] + syns,
            source="rule",
            confidence=0.88,
            required_for_search=True,
            missing_relevance_cap=_ENTITY_MISSING_CAP[
                "drug_or_drug_class"
            ],
        )
        return anchor, {
            "question_subtype": "drug_overview",
            "type_of_question": "background",
            "question_scope": "background",
            "specialty": None,
        }

    # --- 3. Conservative noun-phrase fallback -----------------------
    # Only accept short biomedical-looking phrases: alphanumeric +
    # space + hyphen, length ≥ 3, ≤ 6 tokens, no token is a question
    # opener / stopword in isolation.
    if not re.match(r"^[a-z0-9][a-z0-9\-\s]{2,}$", low):
        return None
    tokens = [t for t in re.split(r"\s+", low) if t]
    if not tokens or len(tokens) > 6:
        return None
    # Phase J1 — reject "modifier + subject" framings (risk of X /
    # complications of X / side effects of X / benefits and risks of X
    # / complications after X). These are NOT background-definition
    # questions; they are foreground clinical questions that the
    # resolver should handle (typically question_type=
    # device_complication_risk, procedure_complication_risk,
    # therapy_safety, or drug_safety). Returning None here drops the
    # query into ``anchor_failure_reason=no_primary_clinical_entity``,
    # which is the existing trigger for the bounded LLM resolver. NO
    # exact-question hardcoding — pattern-based prefix match on a
    # closed-set of generic-modifier framings.
    for prefix in _MODIFIER_PREFIX_PATTERNS:
        if low.startswith(prefix):
            return None
    if all(
        t in _QUESTION_OPENER_TOKENS
        or t in _DEFINITION_FALLBACK_STOPWORDS
        or t in {m for m in GENERIC_MODIFIERS if " " not in m}
        for t in tokens
    ):
        return None
    anchor = PrimaryAnchor(
        text=raw,
        canonical_concept=low,
        entity_type="disease_or_problem",
        role="population",
        synonyms=[low],
        source="rule",
        confidence=0.6,
        required_for_search=True,
        missing_relevance_cap=_ENTITY_MISSING_CAP["disease_or_problem"],
    )
    return anchor, {
        "question_subtype": "disease_overview",
        "type_of_question": "background",
        "question_scope": "background",
        "specialty": None,
    }


primary_clinical_anchor_service = PrimaryClinicalEntityAnchorService()


# ---------------------------------------------------------------------
# Phase 6G.4P — closed-set "management without procedure context"
# clarification triggers. Used by ``is_generic_only`` to catch
# regression cases like:
#   "what are the recommendations?"
#   "what anticoagulation is recommended?"
#   "what is postoperative management?"
# These questions request a recommendation / generic management
# decision WITHOUT supplying a procedure, disease, or other primary
# anchor — the deterministic clarification path is correct.
#
# Patterns are deliberately narrow:
#   * Triple-token "management" / "recommendations" sentences with no
#     clinical noun.
#   * "what [mgmt-noun] is recommended/used?" with no procedure
#     phrase tail (the post-procedure router would have caught the
#     ones that DO carry a procedure phrase).
# ---------------------------------------------------------------------

_MGMT_NOUN_TOKENS: tuple[str, ...] = (
    "anticoagulation", "antithrombotic therapy", "antithrombotic",
    "antiplatelet", "antiplatelet therapy",
    "vte prophylaxis", "venous thromboembolism prophylaxis",
    "thromboprophylaxis",
    "antibiotic prophylaxis", "immunosuppression",
    "pain control", "pain management",
)


_GENERIC_CONTEXT_PHRASES: tuple[str, ...] = (
    "postoperative management", "post-operative management",
    "perioperative management",
    "post procedural management", "post-procedural management",
    "the recommendations", "recommendations",
    "the recommendation", "recommendation",
)


# "what [mgmt-noun] is recommended/used/preferred?" / "should [mgmt-noun]
# be used/given?" — closed-set pattern. Requires the question to NOT
# contain any post-procedure procedure-lexical token (those are caught
# earlier by the post-procedure router).
_MGMT_NO_CONTEXT_RE = re.compile(
    r"^\s*(?:what|which)\s+(?P<noun>[a-z0-9\s\-]{3,40})\s+"
    r"(?:is|are|should\s+be)\s+(?:recommended|used|preferred|appropriate|"
    r"indicated|advised|given|prescribed)\??\s*$",
    re.IGNORECASE,
)


def _matches_management_without_context(question: str) -> bool:
    """Closed-set check. Returns True ONLY when the question is a
    request for management / recommendations with NO clinical subject
    (no procedure, no disease, no drug name).
    """
    if not question:
        return False
    q = question.strip().strip(".!").strip()
    if not q:
        return False
    low = q.lower()

    # Pattern 1 — bare "what is/are [generic context phrase]" sentence.
    for ph in _GENERIC_CONTEXT_PHRASES:
        # "what is postoperative management?" / "what are the
        # recommendations?" — bare construction with no extra clinical
        # noun.
        if re.match(
            rf"^\s*(?:what|which)\s+(?:is|are)\s+(?:the\s+)?{re.escape(ph)}\s*\??\s*$",
            low,
        ):
            return True

    # Pattern 2 — "what [mgmt-noun] is recommended/used?" without
    # a procedure phrase tail. The post-procedure router would have
    # already short-circuited the "after [procedure]" variant.
    m = _MGMT_NO_CONTEXT_RE.match(q)
    if m:
        noun = (m.group("noun") or "").strip().lower()
        # Reject if the noun phrase carries a procedure / disease /
        # drug-name token — that would be a different clarification
        # category (or a real anchor).
        if any(t in noun for t in _MGMT_NOUN_TOKENS):
            # And there is NO procedure phrase tail (the regex anchors
            # to end of string with ``$``).
            return True
    return False


# ---------------------------------------------------------------------
# Phase 6G.4P — Post-procedure management anchor detection.
# ---------------------------------------------------------------------
#
# When the question matches a closed-set post-procedure pattern (see
# ``services.post_procedure_router``) AND the procedure phrase resolves
# against the procedure ontology mapper, we inject TWO primary anchors:
#
#   1) procedure_or_surgery anchor — canonical procedure (e.g.,
#      "aortic valve replacement"), specialty + synonyms + related-but-
#      distinct concepts come from the matched procedure pack entry.
#   2) management anchor — when the management phrase matches a closed-
#      set drug/drug-class signal we emit ``drug_or_drug_class``; when
#      it matches a higher-level management domain (VTE prophylaxis,
#      immunosuppression, surveillance) we emit
#      ``management_intervention``.
#
# Closed-set picott_overrides are emitted alongside so the downstream
# query compiler + ranker + UI receive the routing context:
#   question_subtype = "post_procedure_management"
#   subdomain        = one of post_procedure_router.SUBDOMAINS
#   type_of_question = "guideline" | "therapy" | "surveillance" | "management"
#   question_scope   = "foreground"
#   specialty        = first specialty token from the procedure pack
#
# The router output ALSO flows back to the pipeline via
# ``post_procedure_routing`` (returned by ``derive``) so the
# pipeline can persist it on the session document for downstream
# lane labelling.

# Closed-set drug / drug-class tokens that, when present in the
# management phrase, route the management anchor as
# ``drug_or_drug_class``. Anything else routes as
# ``management_intervention``.
_MGMT_DRUG_TOKENS: frozenset[str] = frozenset({
    "warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban",
    "aspirin", "clopidogrel", "ticagrelor", "prasugrel",
    "heparin", "enoxaparin",
    "tacrolimus", "cyclosporine", "ciclosporin",
    "mycophenolate", "mycophenolate mofetil",
    "azathioprine", "sirolimus", "everolimus",
    # Phase H2 — anti-CD20 / B-cell-depleting agents commonly named in
    # post-transplant medication management queries.
    "rituximab", "obinutuzumab", "ofatumumab",
})

# Closed-set canonical names per subdomain — used to pick the
# management anchor canonical concept when the user phrase is generic
# ("anticoagulation", "VTE prophylaxis", etc.) but the subdomain is
# precise.
_SUBDOMAIN_TO_MGMT_CANONICAL: dict[str, str] = {
    "post_procedure_antithrombotic_management": "antithrombotic therapy",
    "post_procedure_antibiotic_management": "antibiotic prophylaxis",
    "post_procedure_vte_prophylaxis": "VTE prophylaxis",
    "post_procedure_followup_surveillance": "post-procedure surveillance",
    "perioperative_medication_management": "perioperative medication management",
    "post_transplant_immunosuppression": "immunosuppression",
    "post_procedure_pain_management": "post-procedure pain management",
    # Phase H2 — post-transplant medication management.
    "post_transplant_medication_management": "post-transplant medication management",
}

# Closed-set management-anchor synonym packs keyed by subdomain. NEVER
# hallucinated — these are the canonical, conservative synonym sets
# clinicians use interchangeably.
_SUBDOMAIN_TO_MGMT_SYNONYMS: dict[str, list[str]] = {
    "post_procedure_antithrombotic_management": [
        "antithrombotic therapy", "anticoagulation", "anticoagulant therapy",
        "antiplatelet therapy", "DAPT", "dual antiplatelet therapy",
        "warfarin", "DOAC", "direct oral anticoagulant",
    ],
    "post_procedure_antibiotic_management": [
        "antibiotic prophylaxis", "perioperative antibiotic",
        "surgical site infection prophylaxis", "endocarditis prophylaxis",
    ],
    "post_procedure_vte_prophylaxis": [
        "VTE prophylaxis", "venous thromboembolism prophylaxis",
        "thromboprophylaxis", "DVT prophylaxis",
    ],
    "post_procedure_followup_surveillance": [
        "surveillance", "follow-up", "follow up",
        "surveillance interval", "post-treatment surveillance",
    ],
    "perioperative_medication_management": [
        "perioperative medication management", "bridging anticoagulation",
        "restarting anticoagulation", "medication restart",
        # Phase H2-tail Task C — inherited cardiac arrhythmia
        # perioperative phrases (Brugada / LQTS / CPVT). Closed-set
        # synonyms only; no free text.
        "perioperative management", "perioperative care",
        "anesthesia", "anaesthesia",
        "anesthesia management", "anaesthesia management",
        "anesthetic management", "anaesthetic management",
        "drug avoidance", "drugs to avoid",
        "medication avoidance", "medications to avoid",
        "sodium channel blocker avoidance",
        "QT-prolonging drug avoidance",
        "fever management",
    ],
    "post_transplant_immunosuppression": [
        "immunosuppression", "immunosuppressive therapy",
        "induction immunosuppression", "maintenance immunosuppression",
        "tacrolimus", "mycophenolate",
    ],
    "post_procedure_pain_management": [
        "pain control", "pain management", "postoperative analgesia",
        "multimodal analgesia",
    ],
    # Phase H2 — closed-set synonym pack for post-transplant
    # medication management (rituximab / anti-CD20 maintenance,
    # post-transplant consolidation, daratumumab / lenalidomide
    # maintenance in myeloma settings).
    "post_transplant_medication_management": [
        "post-transplant medication management",
        "rituximab maintenance",
        "post-transplant consolidation",
        "post-transplant maintenance",
        "anti-CD20 maintenance",
        "consolidation therapy",
        "maintenance therapy",
    ],
}


def _management_anchor(
    therapy_phrase: str | None, subdomain: str,
) -> PrimaryAnchor:
    """Build the management primary anchor for a post-procedure match.

    Returns a ``drug_or_drug_class`` anchor when the therapy_phrase
    contains a closed-set drug token, else a ``management_intervention``
    anchor. The canonical concept and synonym set come from the
    subdomain table (closed-set) — NEVER from free text.
    """
    canonical = _SUBDOMAIN_TO_MGMT_CANONICAL.get(
        subdomain, "post-procedure management",
    )
    syns = list(_SUBDOMAIN_TO_MGMT_SYNONYMS.get(subdomain, []))
    drug_match = ""
    if therapy_phrase:
        low = therapy_phrase.lower()
        for tok in _MGMT_DRUG_TOKENS:
            if tok in low:
                drug_match = tok
                break
    if drug_match:
        return PrimaryAnchor(
            text=therapy_phrase or canonical,
            canonical_concept=drug_match,
            entity_type="drug_or_drug_class",
            role="intervention",
            synonyms=[drug_match] + syns,
            source="ontology_pack",
            confidence=0.92,
            required_for_search=True,
            missing_relevance_cap=_ENTITY_MISSING_CAP["drug_or_drug_class"],
        )
    return PrimaryAnchor(
        text=therapy_phrase or canonical,
        canonical_concept=canonical,
        entity_type="management_intervention",
        role="intervention",
        synonyms=syns,
        source="ontology_pack",
        confidence=0.9,
        required_for_search=True,
        missing_relevance_cap=_ENTITY_MISSING_CAP["management_intervention"],
    )


# Closed-set subdomain → type_of_question label. The compiler reads
# ``type_of_question`` for clinical-queries category routing; we route
# post-procedure to ``therapy`` by default and ``prevention`` for
# antibiotic / VTE prophylaxis paths.
_SUBDOMAIN_TO_TOQ: dict[str, str] = {
    "post_procedure_antithrombotic_management": "therapy",
    "post_procedure_antibiotic_management": "prevention",
    "post_procedure_vte_prophylaxis": "prevention",
    "post_procedure_followup_surveillance": "prognosis",
    "perioperative_medication_management": "therapy",
    "post_transplant_immunosuppression": "therapy",
    "post_procedure_pain_management": "therapy",
    # Phase H2 — post-transplant medication management (rituximab
    # maintenance / consolidation). Type-of-question is ``therapy``
    # (clinician asking about a dosing / maintenance regimen).
    "post_transplant_medication_management": "therapy",
}


# Phase H2-tail Task C — closed-set list of canonical concepts that
# are inherited cardiac arrhythmia CONDITIONS (not procedures). When
# the procedure_ontology_mapper resolves to one of these via the
# cardiology_inherited_arrhythmias.json pack, the resulting first
# anchor is built with ``entity_type=disease_or_problem`` (and the
# conservative disease cap) instead of the legacy
# ``procedure_or_surgery``. Mirrors the canonical_concept strings
# emitted by ``cardiology_inherited_arrhythmias.json`` — must be
# kept in sync if the pack canonicals change.
_INHERITED_ARRHYTHMIA_CONDITION_CANONICALS: frozenset[str] = frozenset({
    "brugada syndrome",
    "long qt syndrome",
    "catecholaminergic polymorphic ventricular tachycardia",
})


def _detect_post_procedure_anchors(
    question: str,
) -> tuple[list[PrimaryAnchor], dict[str, Any], dict[str, Any]] | None:
    """Closed-set, deterministic detection of a post-procedure
    management question. Returns ``(anchors, picott_overrides,
    post_procedure_payload)`` on a confident match, ``None`` otherwise.

    The function NEVER calls an LLM or network source. Lookups go
    through the procedure ontology mapper (curated procedure packs +
    existing MeSH resolver). No fabrication.

    Phase H2-tail Task C — when the resolved canonical is in the
    closed-set ``_INHERITED_ARRHYTHMIA_CONDITION_CANONICALS`` list,
    the first anchor is built with
    ``entity_type=disease_or_problem`` (Brugada / LQTS / CPVT are
    conditions, not procedures) and the conservative disease cap.
    All other paths keep the legacy ``procedure_or_surgery``
    entity_type. No new entity_type tokens introduced.
    """
    if not question:
        return None
    try:
        from services.post_procedure_router import post_procedure_router
        from services.procedure_ontology_mapper import (
            procedure_ontology_mapper,
        )
    except Exception:  # noqa: BLE001
        return None
    routing = post_procedure_router.detect(question)
    if not routing.get("matched"):
        return None
    procedure_phrase = routing.get("procedure_phrase") or ""
    therapy_phrase = routing.get("therapy_phrase")
    subdomain = routing.get("subdomain") or "post_procedure_antithrombotic_management"
    resolved = procedure_ontology_mapper.resolve_procedure(procedure_phrase)
    if resolved is None:
        return None
    # Phase H2-tail Task C — inherited cardiac arrhythmia conditions
    # (Brugada / LQTS / CPVT) are CONDITIONS, not procedures, so the
    # first anchor must be built with ``entity_type=disease_or_problem``
    # and the conservative disease cap. Everything else continues with
    # the legacy ``procedure_or_surgery`` path.
    canonical_lower = (resolved.canonical_concept or "").lower()
    if canonical_lower in _INHERITED_ARRHYTHMIA_CONDITION_CANONICALS:
        proc_entity_type = "disease_or_problem"
        proc_role = "population"
    else:
        proc_entity_type = "procedure_or_surgery"
        proc_role = "population"
    proc_anchor = PrimaryAnchor(
        text=procedure_phrase,
        canonical_concept=resolved.canonical_concept,
        entity_type=proc_entity_type,
        role=proc_role,
        synonyms=[resolved.canonical_concept] + list(resolved.synonyms),
        source="ontology_pack",
        confidence=float(resolved.confidence),
        required_for_search=True,
        missing_relevance_cap=_ENTITY_MISSING_CAP[proc_entity_type],
    )
    mgmt_anchor = _management_anchor(therapy_phrase, subdomain)
    specialty = ""
    if resolved.specialty:
        specialty = resolved.specialty[0]
    type_of_question = _SUBDOMAIN_TO_TOQ.get(subdomain, "therapy")
    overrides: dict[str, Any] = {
        "question_subtype": "post_procedure_management",
        "type_of_question": type_of_question,
        "question_scope": "foreground",
        "specialty": specialty or None,
        # Closed-set transport key for the term-mapping projection.
        "subdomain": subdomain,
    }
    payload: dict[str, Any] = {
        "matched": True,
        "subdomain": subdomain,
        "intent": routing.get("intent") or "therapy",
        "procedure_canonical": resolved.canonical_concept,
        "procedure_family": resolved.procedure_family,
        "specialty": list(resolved.specialty),
        "related_but_distinct": list(resolved.related_but_distinct),
        "management_canonical": mgmt_anchor.canonical_concept,
        "management_entity_type": mgmt_anchor.entity_type,
        "management_domains": list(resolved.management_domains),
        # NEVER persist the raw user phrase — only the canonical /
        # closed-set fields above.
    }
    return [proc_anchor, mgmt_anchor], overrides, payload


__all__ = [
    "ENTITY_TYPES",
    "ANCHOR_ROLES",
    "ANCHOR_SOURCES",
    "GENERIC_MODIFIERS",
    "CLARIFICATION_MESSAGE",
    "PrimaryAnchor",
    "PrimaryClinicalEntityAnchorService",
    "primary_clinical_anchor_service",
]