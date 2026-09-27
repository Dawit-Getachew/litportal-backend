"""Phase 5F — Clinical Context Engine orchestrator.

Single sync entry point that runs the cascade:

  parsed dict (from ``clinical_question_parser.parse``)
   → ontology mapping        (clinical_ontology_mapper)
   → role repair             (clinical_role_repair)
   → role weighting          (role_weighting)
   → multi-intent detection  (deterministic, closed-set connectors)

Returns a ``ClinicalContext`` dict-like result that pipeline.py persists
under the session document. The result intentionally never raises — any
sub-step that fails returns an empty list and a soft warning so the
pipeline can degrade gracefully.

Output schema (closed set; reads by pipeline.py + transparency panel):
{
  "shorthand_substitutions": [ShorthandSubstitution, ...],
  "ontology_mappings":       [OntologyMapping dict, ...],
  "role_repairs":            [RepairProvenance, ...],
  "role_weights": {
    "type_of_question": str,
    "question_subtype": str | None,
    "weights": {role: int, ...},      # closed-set ROLE_WEIGHTS copy
  },
  "multi_intent": {
    "is_multi_intent": bool,
    "sub_questions": [
      {"question": str, "subtype": str, "role_weights": {...}},
      ...
    ],
    "rationale": str,
  },
  "repaired_picott": dict | None,     # the PICOTT to be PERSISTED.
}
"""
from __future__ import annotations

import logging
import re
from typing import Any

from services.clinical_ontology_mapper import clinical_ontology_mapper
from services.clinical_role_repair import clinical_role_repair
from services.role_weighting import ROLE_WEIGHTS, role_weighting

logger = logging.getLogger("clinevidence.context_engine")


# Closed-set multi-intent connectors. Conservative — only fires when
# question contains BOTH (a) a connector AND (b) we can identify two
# distinct sub-question types after splitting.
_MULTI_INTENT_CONNECTORS = (
    "? what is", "? what are", "? what's",
    " and what is", " and what are", " and what's",
    " and how ", " and how does", " and the benefit",
    " and what benefit", " and what's the benefit",
)


# Closed-set subtype heuristic — pure lexical fallback. Pipelines can
# augment this with LLM-detected subtypes when the parser supplies them.
_SUBTYPE_HEURISTICS: list[tuple[str, str]] = [
    # (regex over normalized text → subtype)
    (r"\b(?:recommended age|vaccination age|age at vaccination|"
      r"primary series|first dose|second dose|booster|catch-up|"
      r"two-dose schedule|2-dose schedule)\b",
     "immunization_schedule"),
    (r"\b(?:benefit|benefits|effectiveness|efficacy|prevent|"
      r"protect|reduces|hospitalization|herd immunity|mortality)\b",
     "vaccine_effectiveness_benefit"),
    (r"\b(?:timing|dose|dosing|when to start|how long|"
      r"duration|tapering|weaning|early vs late)\b",
     "timing_dose_protocol"),
    (r"\b(?:impact|effect|level|biomarker|LDL|HDL|HbA1c|"
      r"triglycerides|cholesterol)\b",
     "metabolic_biomarker_effect"),
]


def _detect_subtype_for(text: str) -> str | None:
    t = (text or "").lower()
    for pattern, subtype in _SUBTYPE_HEURISTICS:
        if re.search(pattern, t):
            return subtype
    return None


def _split_question_on_connector(q: str) -> list[str]:
    """Split a question into sub-questions based on closed-set connectors
    plus '?' boundaries. Returns at least [q] if no split fires."""
    if not q:
        return []
    text = q.strip()
    parts: list[str] = []
    # First, split on '? ' boundaries (preserve the '?').
    chunks = re.split(r"\?\s*", text)
    chunks = [c.strip() for c in chunks if c and c.strip()]
    # Then split each chunk on " and what " / " and how " connectors.
    for chunk in chunks:
        sub = re.split(
            r"\s+and\s+(?:what(?:'s)?|how(?:\s+does)?|"
            r"the\s+benefit|what\s+benefit)\s+",
            chunk,
            flags=re.IGNORECASE,
        )
        for s in sub:
            s = s.strip(" .;:,")
            if s:
                parts.append(s)
    return parts if parts else [text]


def _detect_multi_intent(
    question: str, parser_subtype: str | None = None,
) -> dict[str, Any]:
    """Conservative trigger: multi_intent=true only when BOTH
    (a) the question contains a closed-set connector AND
    (b) splitting yields ≥2 sub-questions that map to DIFFERENT
        ``question_subtype`` values.
    """
    if not question:
        return {"is_multi_intent": False, "sub_questions": [], "rationale": ""}
    q = question.lower()
    if not any(c in q for c in _MULTI_INTENT_CONNECTORS):
        return {
            "is_multi_intent": False, "sub_questions": [],
            "rationale": "no closed-set connector matched",
        }
    parts = _split_question_on_connector(question)
    if len(parts) < 2:
        return {
            "is_multi_intent": False, "sub_questions": [],
            "rationale": "connector matched but only one sub-question after split",
        }
    sub_specs: list[dict[str, Any]] = []
    subtypes_seen: set[str] = set()
    for part in parts[:4]:  # cap at 4 sub-questions
        subtype = _detect_subtype_for(part) or parser_subtype or "other"
        sub_specs.append({"question": part, "subtype": subtype})
        subtypes_seen.add(subtype)
    if len(subtypes_seen) < 2:
        return {
            "is_multi_intent": False, "sub_questions": [],
            "rationale": (
                "split yielded sub-questions but they share the same "
                "question_subtype — staying single-intent."
            ),
        }
    # Attach role weights per sub-question.
    for spec in sub_specs:
        spec["role_weights"] = role_weighting.weights_for(spec["subtype"])
    return {
        "is_multi_intent": True,
        "sub_questions": sub_specs,
        "rationale": (
            f"closed-set connector + {len(subtypes_seen)} distinct subtypes "
            f"{sorted(subtypes_seen)}"
        ),
    }


def _picott_terms_for_mapping(picott: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for mh in (picott.get("must_have_concepts") or []):
        t = (mh.get("concept") or "").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    for rc in (picott.get("ranking_concepts") or []):
        t = (rc.get("concept") or "").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    # Also map the PICO block contents.
    pico = picott.get("pico") or {}
    for k in (
        "population", "intervention_or_exposure_or_index_test",
        "comparator_or_reference_standard", "outcomes_or_target_diagnosis",
    ):
        for s in (pico.get(k) or []):
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
    return out


# Role-weight threshold for synonym expansion (Phase 5F refinement #6).
# Weights in ROLE_WEIGHTS are integers 0-100; the 0.7 fraction maps to 25.
# Roles with weight >= 25 get synonym expansion from ontology hits;
# lower-weight roles keep their canonical term only (avoids query bloat).
_SYN_EXPANSION_THRESHOLD = 25


# Phase 5F.1 — closed-set PICOTT roles that the ontology lift may
# populate. ``context``, ``schedule_terms``, ``dose_terms``, etc. are
# NOT real PICOTT roles and are skipped (they're modifiers / hints).
_LIFTABLE_PICO_ROLES: frozenset[str] = frozenset({
    "intervention", "exposure", "index_test",
    "comparator", "reference_standard",
    "outcome", "target_diagnosis",
    "population", "condition", "patient_group",
})


def _pico_block_for_role_lift(role: str) -> str | None:
    """Map a canonical role to the PICOBlocks key it belongs in."""
    role = (role or "").lower()
    if role in {"intervention", "exposure", "index_test"}:
        return "intervention_or_exposure_or_index_test"
    if role in {"comparator", "reference_standard"}:
        return "comparator_or_reference_standard"
    if role in {"outcome", "target_diagnosis"}:
        return "outcomes_or_target_diagnosis"
    if role in {"population", "condition", "patient_group"}:
        return "population"
    return None


# Words that join two noun phrases. A concept containing one is a glued
# phrase ("X in patients with Y") whose parts may be separate concepts; a
# concept without one is a single noun phrase.
_PHRASE_JOINERS: frozenset[str] = frozenset({
    "in", "with", "for", "of", "and", "or", "versus", "vs", "after", "before", "among",
    "on", "to", "from", "by", "during", "without", "compared", "than", "at", "following",
})


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _existing_phrases_by_block(picott: dict[str, Any]) -> list[tuple[list[str], str | None]]:
    """(tokens, PICO block) of every concept phrase the PICOTT already has."""
    out: list[tuple[list[str], str | None]] = []
    for key in ("must_have_concepts", "should_have_concepts", "ranking_concepts"):
        for c in (picott.get(key) or []):
            if isinstance(c, dict) and c.get("concept"):
                out.append((_tokens(c["concept"]), _pico_block_for_role_lift(c.get("role") or "")))
    for block, values in (picott.get("pico") or {}).items():
        for v in (values or []) if isinstance(values, list) else []:
            if isinstance(v, str) and v:
                out.append((_tokens(v), block))
    return out


def _is_fragment_of_existing(term: str, role: str, existing: list[tuple[list[str], str | None]]) -> bool:
    """True when *term* is a contiguous part of a longer single-noun-phrase
    concept already in the PICOTT, in the same PICO block — e.g. "blood
    transfusion" inside "whole blood transfusion". Lifting it would add a
    broader concept (whose pack synonyms may even name a sibling, such as
    red cell transfusion) as a separate required anchor."""
    t = _tokens(term)
    block = _pico_block_for_role_lift(role)
    if not t or block is None:
        return False
    n = len(t)
    for phrase, phrase_block in existing:
        if phrase_block != block or len(phrase) <= n or _PHRASE_JOINERS.intersection(phrase):
            continue
        if any(phrase[i:i + n] == t for i in range(len(phrase) - n + 1)):
            return True
    return False


def _lift_ontology_into_picott(
    picott: dict[str, Any],
    ontology_mappings: list[dict[str, Any]],
    *,
    role_weights_dict: dict[str, int],
    min_confidence: float = 0.85,
) -> list[dict[str, Any]]:
    """Phase 5F.1 FIX 1 — when the LLM omitted a concept that the
    ontology mapper found with high confidence and a PICOTT role hint,
    CREATE the missing ``must_have_concept`` and mirror it into the
    corresponding ``picott.pico.<block>`` array. Returns the list of
    lift events for ``ontology_lifts_json``.

    Closed-set behaviour. NEVER invents a canonical concept absent from
    the loaded curated packs (the mapper enforces that). NEVER removes
    or overrides an LLM-provided concept.
    """
    lifts: list[dict[str, Any]] = []
    if not picott or not ontology_mappings:
        return lifts

    # Build the set of existing concept names + synonyms (case-insensitive).
    existing_keys: set[str] = set()
    for mh in (picott.get("must_have_concepts") or []):
        for k in [mh.get("concept")] + list(mh.get("synonyms") or []):
            if k:
                existing_keys.add(k.lower())
    for rc in (picott.get("ranking_concepts") or []):
        for k in [rc.get("concept")] + list(rc.get("synonyms") or []):
            if k:
                existing_keys.add(k.lower())

    from services.role_weighting import _canonical_role  # noqa: WPS433

    existing_phrases = _existing_phrases_by_block(picott)
    must_have_list = picott.setdefault("must_have_concepts", [])
    pico_blocks = picott.setdefault("pico", {})

    for m in ontology_mappings:
        if not isinstance(m, dict):
            continue
        canonical = (m.get("canonical_concept") or "").strip()
        if not canonical:
            continue
        confidence = float(m.get("confidence") or 0.0)
        if confidence < min_confidence:
            continue
        role_hint = (m.get("role_hint") or "").lower()
        if role_hint not in _LIFTABLE_PICO_ROLES:
            continue
        if canonical.lower() in existing_keys:
            continue
        # Don't lift if a synonym of an existing concept already covers it.
        if any(s in existing_keys for s in (s.lower() for s in (m.get("synonyms") or []) if s)):
            continue
        # Don't lift a fragment of a concept the parser already has.
        matched_text = m.get("original_term") or m.get("normalized_term") or ""
        if _is_fragment_of_existing(matched_text, role_hint, existing_phrases):
            continue

        # Determine PICOTT block + canonical role.
        block = _pico_block_for_role_lift(role_hint)
        if not block:
            continue
        canonical_role_for_weight = _canonical_role(role_hint)
        weight = int(role_weights_dict.get(canonical_role_for_weight, 0))
        required = (
            role_hint in {"intervention", "outcome", "population"}
            and weight >= _SYN_EXPANSION_THRESHOLD
        )

        # Build closed-set synonym list (canonical synonyms + acronyms +
        # MeSH descriptor names + RxNorm NAMES; numeric RxCUIs skipped).
        syns: list[str] = []
        seen_syn: set[str] = set()
        for s in (m.get("synonyms") or []) + (m.get("acronyms") or []):
            sl = (s or "").strip()
            if sl and sl.lower() not in seen_syn:
                seen_syn.add(sl.lower())
                syns.append(sl)
        for s in (m.get("mesh_terms") or []):
            sl = (s or "").strip()
            if sl and sl.lower() not in seen_syn:
                seen_syn.add(sl.lower())
                syns.append(sl)
        for s in (m.get("rxnorm_terms") or []):
            sl = (s or "").strip()
            if sl and not sl.isdigit() and sl.lower() not in seen_syn:
                seen_syn.add(sl.lower())
                syns.append(sl)

        new_concept = {
            "concept": canonical,
            "role": role_hint,
            "synonyms": syns,
            "required_for_high_relevance": required,
            "source": "ontology_lift",
            "provenance": {
                "ontology_source": m.get("source") or "specialty_pack",
                "canonical_term": canonical,
                "confidence": confidence,
            },
        }
        must_have_list.append(new_concept)
        existing_keys.add(canonical.lower())
        for s in syns:
            existing_keys.add(s.lower())

        # Mirror into the PICO block (case-insensitive dedupe).
        arr = pico_blocks.setdefault(block, [])
        if not any((c or "").lower() == canonical.lower() for c in arr):
            arr.append(canonical)

        lifts.append({
            "canonical_term": canonical,
            "role": role_hint,
            "source": m.get("source") or "specialty_pack",
            "confidence": confidence,
            "reason": "LLM omitted concept; lifted from ontology hit",
        })

    return lifts


def _enrich_picott_with_ontology(
    picott: dict[str, Any],
    ontology_mappings: list[dict[str, Any]],
    *,
    role_weights_dict: dict[str, int],
) -> None:
    """In-place enrichment of ``picott["must_have_concepts"][*]["synonyms"]``
    using ontology mapping hits. Only fires for roles whose weight
    >= ``_SYN_EXPANSION_THRESHOLD`` (Phase 5F refinement #6).

    Closed-set behaviour:
      - canonical synonyms from the OntologyMapping ``synonyms`` list
      - MeSH descriptor names from ``mesh_terms``
      - RxNorm-derived names from ``rxnorm_terms`` are *names* (e.g.
        ``"semaglutide"``) — RxCUIs (numeric) are NOT added to queries
        (they would be invalid free-text tokens). If a numeric RxCUI is
        the only token we have, we soft-skip it.
      - existing STOP_PHRASES filtering on the compiler side still
        applies — we don't strip anything here.

    Never raises; never mutates the original_question.
    """
    if not picott or not ontology_mappings:
        return
    # Build a lookup: lower-cased term → mapping dict.
    mapping_index: dict[str, dict[str, Any]] = {}
    for m in ontology_mappings:
        if not isinstance(m, dict):
            continue
        for key in (m.get("normalized_term"), m.get("original_term"),
                    m.get("canonical_concept")):
            if key:
                mapping_index[key.lower()] = m
        for s in (m.get("synonyms") or []):
            if s:
                mapping_index.setdefault(s.lower(), m)
    # Walk must_have_concepts and append ontology synonyms when role
    # weight >= threshold.
    for mh in (picott.get("must_have_concepts") or []):
        concept = (mh.get("concept") or "").strip()
        role = (mh.get("role") or "other").lower()
        # Normalize role for weight lookup.
        from services.role_weighting import _canonical_role  # noqa: WPS433
        canonical_role = _canonical_role(role)
        weight = int(role_weights_dict.get(canonical_role, 0))
        if weight < _SYN_EXPANSION_THRESHOLD:
            continue
        md = mapping_index.get(concept.lower())
        if not md:
            continue
        existing_syns = {(s or "").lower() for s in (mh.get("synonyms") or [])}
        existing_syns.add(concept.lower())
        added: list[str] = []
        # Add canonical synonyms.
        for s in (md.get("synonyms") or []):
            sl = (s or "").strip()
            if sl and sl.lower() not in existing_syns:
                added.append(sl)
                existing_syns.add(sl.lower())
        # Add MeSH descriptor names (e.g. ``Cholesterol, LDL``).
        for s in (md.get("mesh_terms") or []):
            sl = (s or "").strip()
            if sl and sl.lower() not in existing_syns:
                added.append(sl)
                existing_syns.add(sl.lower())
        # Add RxNorm-derived NAMES only (skip purely numeric RxCUIs).
        for s in (md.get("rxnorm_terms") or []):
            sl = (s or "").strip()
            if not sl:
                continue
            if sl.isdigit():
                continue  # soft-skip RxCUI numerics
            if sl.lower() not in existing_syns:
                added.append(sl)
                existing_syns.add(sl.lower())
        if added:
            mh["synonyms"] = list(mh.get("synonyms") or []) + added
            # Annotate so the transparency panel + ranker know this was
            # ontology-enriched.
            mh["ontology_enrichment_applied"] = True


def build_clinical_context(parsed: dict[str, Any]) -> dict[str, Any]:
    """Run the Phase 5F context cascade. Never raises."""
    shorthand_subs = list(parsed.get("shorthand_substitutions") or [])
    picott = dict(parsed.get("picott_json") or {})
    out: dict[str, Any] = {
        "shorthand_substitutions": shorthand_subs,
        "ontology_mappings": [],
        "ontology_lifts": [],
        "role_repairs": [],
        "role_weights": {
            "type_of_question": "other",
            "question_subtype": None,
            "weights": dict(ROLE_WEIGHTS["other"]),
        },
        "multi_intent": {
            "is_multi_intent": False, "sub_questions": [], "rationale": "",
        },
        "repaired_picott": picott or None,
    }
    # 1. Ontology mapping over every concept that the picott carries.
    if picott:
        try:
            terms = _picott_terms_for_mapping(picott)
            out["ontology_mappings"] = clinical_ontology_mapper.map_many(terms)
        except Exception:  # noqa: BLE001
            logger.exception("Phase 5F ontology mapper failed (non-fatal)")
    # 1b. Phase 5F deterministic concept extraction. Runs UNCONDITIONALLY
    #     (was: only when must_haves empty) so it can recover concepts
    #     the LLM omitted even when the LLM provided a partial PICOTT.
    #     The lift step (FIX 1) below dedupes against existing concepts
    #     case-insensitively.
    #
    # Phase 5F.1 FIX — prefer the post-spelling+shorthand normalized
    # question (held on ``parsed["normalized_question"]``) over
    # ``picott.normalized_question``. The picott_extractor builds picott
    # from the legacy parser output, which stores the raw input verbatim;
    # only ``parsed["normalized_question"]`` carries the shorthand
    # expansions (``glp1s → GLP-1 receptor agonists``). Without this
    # ordering fix the extractor scans the RAW text and misses GLP-1.
    synth_norm = (
        parsed.get("normalized_question")
        or (picott or {}).get("normalized_question")
        or parsed.get("original_question")
        or ""
    )
    # Defense in depth — if shorthand substitutions exist on parsed but
    # the question text still contains the raw forms, apply the
    # substitutions one more time. This is a no-op when the parser
    # already did its job.
    sh_subs = parsed.get("shorthand_substitutions") or []
    if sh_subs and synth_norm:
        synth_lower = synth_norm.lower()
        for s in sh_subs:
            orig = (s.get("original") or "").lower()
            if orig and orig in synth_lower:
                try:
                    from services.clinical_shorthand_normalizer import (  # noqa: WPS433
                        clinical_shorthand_normalizer,
                    )
                    synth_norm = clinical_shorthand_normalizer.normalize(synth_norm).normalized
                    break
                except Exception:  # noqa: BLE001
                    pass
    if picott and synth_norm:
        try:
            synth_hits = clinical_ontology_mapper.extract_from_text(
                synth_norm, max_terms=8, min_confidence=0.85,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Phase 5F text extraction failed (non-fatal)")
            synth_hits = []
        # Merge into the persisted ontology_mappings (dedupe).
        if synth_hits:
            seen_terms = {
                (m.get("normalized_term") or m.get("original_term") or "").lower()
                for m in out["ontology_mappings"]
            }
            for h in synth_hits:
                k = (h.get("normalized_term") or h.get("original_term") or "").lower()
                if k and k not in seen_terms:
                    seen_terms.add(k)
                    out["ontology_mappings"].append(h)
        # When LLM left must_haves entirely empty, ALSO synthesize them
        # (existing behaviour — kept for sessions where the LLM emits
        # zero anchors). The lift step further down picks up any other
        # missed concepts.
        needs_synth = not (picott.get("must_have_concepts") or [])
        if needs_synth and synth_hits:
            existing_keys = set()
            synth_must: list[dict[str, Any]] = []
            for h in synth_hits:
                canonical = h.get("canonical_concept") or ""
                if not canonical or canonical.lower() in existing_keys:
                    continue
                existing_keys.add(canonical.lower())
                synth_must.append({
                    "concept": canonical,
                    # Role placeholder; the next step (role repair) sets
                    # the correct role from ontology evidence.
                    "role": h.get("role_hint") or "context",
                    "synonyms": list(h.get("synonyms") or []),
                    "ontology_source": h.get("source"),
                    "synthesized_by": "ontology_extract_from_text",
                })
            if synth_must:
                picott["must_have_concepts"] = synth_must
                # Merge synth hits into the persisted ontology_mappings.
                seen_terms = {
                    (m.get("normalized_term") or m.get("original_term") or "").lower()
                    for m in out["ontology_mappings"]
                }
                for h in synth_hits:
                    k = (h.get("normalized_term") or h.get("original_term") or "").lower()
                    if k and k not in seen_terms:
                        seen_terms.add(k)
                        out["ontology_mappings"].append(h)
                # Phase 5F.1 — record synth as ontology_lifts entries
                # too, so the audit trail covers both "LLM omitted
                # everything" (synth path) and "LLM omitted a subset"
                # (FIX 1 lift path) consistently.
                for c in synth_must:
                    canonical = (c.get("concept") or "").strip()
                    role = (c.get("role") or "").lower()
                    if not canonical or role not in _LIFTABLE_PICO_ROLES:
                        continue
                    # Find the matching ontology mapping for confidence.
                    conf = 0.85
                    src = "specialty_pack"
                    for h in synth_hits:
                        if (h.get("canonical_concept") or "").lower() == canonical.lower():
                            conf = float(h.get("confidence") or 0.85)
                            src = h.get("source") or src
                            break
                    out["ontology_lifts"].append({
                        "canonical_term": canonical,
                        "role": role,
                        "source": src,
                        "confidence": conf,
                        "reason": (
                            "LLM returned empty must_have_concepts; "
                            "lifted from ontology hit"
                        ),
                    })
                    # Mirror into the pico block for the display.
                    block = _pico_block_for_role_lift(role)
                    if block:
                        pico_blocks = picott.setdefault("pico", {})
                        arr = pico_blocks.setdefault(block, [])
                        if not any((s or "").lower() == canonical.lower() for s in arr):
                            arr.append(canonical)
    # 2. Role repair using ontology hits.
    if picott:
        try:
            repair_res = clinical_role_repair.repair(
                picott, ontology_mappings=out["ontology_mappings"],
            )
            out["repaired_picott"] = repair_res.repaired_picott
            out["role_repairs"] = list(repair_res.repair_provenance)
            picott = repair_res.repaired_picott
        except Exception:  # noqa: BLE001
            logger.exception("Phase 5F role repair failed (non-fatal)")
    # 2a. Phase 5F.1 FIX 1 — lift ontology hits the LLM omitted into
    #     must_have_concepts + picott.pico arrays. Runs AFTER role
    #     repair and BEFORE role_weighting / synonym enrichment so the
    #     weights account for the lifted concepts and the compiler sees
    #     them on must_have_concepts. NEVER overrides LLM concepts.
    if picott and out["ontology_mappings"]:
        try:
            qtype_for_lift = (
                (picott.get("question_subtype") or "").lower()
                or (picott.get("type_of_question") or "other").lower()
            )
            weights_for_lift = role_weighting.weights_for(qtype_for_lift)
            lifts = _lift_ontology_into_picott(
                picott, out["ontology_mappings"],
                role_weights_dict=weights_for_lift,
                min_confidence=0.85,
            )
            if lifts:
                out["ontology_lifts"] = lifts
                out["repaired_picott"] = picott
        except Exception:  # noqa: BLE001
            logger.exception("Phase 5F.1 ontology lift failed (non-fatal)")
    # 2b. Phase 5F refinement #6 — enrich must_have_concepts[].synonyms
    #     with ontology-derived synonyms + mesh descriptors + rxnorm
    #     member names for roles whose weight >= 0.7. The compiler picks
    #     these up automatically via the existing _mandatory_anchor_terms
    #     / _build_blocks helpers. Only top-weighted roles get expansion;
    #     low-weight roles keep their canonical term to avoid query bloat.
    if picott and out["ontology_mappings"]:
        try:
            _enrich_picott_with_ontology(
                picott, out["ontology_mappings"],
                role_weights_dict=role_weighting.weights_for(
                    ((picott or {}).get("question_subtype")
                      or (picott or {}).get("type_of_question")
                      or "other"),
                ),
            )
            out["repaired_picott"] = picott
        except Exception:  # noqa: BLE001
            logger.exception("Phase 5F synonym enrichment failed (non-fatal)")
    # 2c. Phase 6B.3 — apply ``question_type_hint`` / ``question_subtype_hint``
    #     repair when the LLM PICOTT returned ``type_of_question="other"``
    #     and a high-confidence ontology mapping has a curated hint.
    #     This is the "AMI transfusion → therapy / transfusion_threshold_
    #     comparison" closer for the documented 6B.2 limitation.
    if picott and (out.get("ontology_mappings") or []):
        try:
            current_toq = (picott.get("type_of_question") or "other").lower()
            current_subtype = (picott.get("question_subtype") or "").strip()
            # Find the first high-confidence mapping carrying a hint.
            hint_source = None
            chosen_type = None
            chosen_subtype = None
            for m in out["ontology_mappings"]:
                qt_hint = (m.get("question_type_hint") or "").strip().lower()
                qs_hint = (m.get("question_subtype_hint") or "").strip()
                conf = float(m.get("confidence") or 0.0)
                if conf < 0.85 or not qt_hint:
                    continue
                # Repair the type ONLY when it's currently "other".
                # Repair the subtype if missing.
                if current_toq == "other" and qt_hint and qt_hint != "other":
                    chosen_type = qt_hint
                if not current_subtype and qs_hint:
                    chosen_subtype = qs_hint
                if chosen_type or chosen_subtype:
                    hint_source = (
                        f"ontology_question_type_hint:"
                        f"{m.get('pack_name') or m.get('source')}:"
                        f"{m.get('canonical_concept')}"
                    )
                    break
            if chosen_type:
                picott["type_of_question"] = chosen_type
            if chosen_subtype:
                picott["question_subtype"] = chosen_subtype
            if hint_source and (chosen_type or chosen_subtype):
                # Surface in role_repairs_json so the audit trail is
                # explicit and the UI can show it.
                out["role_repairs"].append({
                    "repaired_field": "picott.type_of_question"
                                       if chosen_type else "picott.question_subtype",
                    "concept": "(question)",
                    "previous_value": (current_toq if chosen_type
                                        else (current_subtype or "")),
                    "new_value": chosen_type or chosen_subtype or "",
                    "rule": "ontology_question_type_hint",
                    "ontology_source": "specialty_pack",
                    "confidence": 0.92,
                    "rationale": hint_source,
                })
                out["repaired_picott"] = picott
        except Exception:  # noqa: BLE001
            logger.exception(
                "Phase 6B.3 ontology question_type_hint repair failed (non-fatal)",
            )
    # 3. Role weighting conditioned on the type_of_question + subtype.
    toq = ((picott or {}).get("type_of_question") or "other").lower()
    subtype = ((picott or {}).get("question_subtype") or "") or None
    # If the parser left subtype empty, fall back to a heuristic over the
    # normalized question text (still closed-set).
    q_norm = (parsed.get("normalized_question")
              or (picott or {}).get("normalized_question")
              or "")
    detected_subtype = _detect_subtype_for(q_norm) if not subtype else None
    effective_subtype_key = subtype or detected_subtype or toq
    weights = role_weighting.weights_for(effective_subtype_key) \
              if effective_subtype_key in ROLE_WEIGHTS \
              else role_weighting.weights_for(toq)
    out["role_weights"] = {
        "type_of_question": toq,
        "question_subtype": subtype or detected_subtype,
        "weights": weights,
    }
    # 4. Multi-intent detection.
    out["multi_intent"] = _detect_multi_intent(q_norm, parser_subtype=subtype)
    return out


__all__ = ["build_clinical_context"]
