"""Phase 5D.1 — ``interpreted_question_display_json`` builder.

Single source of truth for the clinician-facing PICOTT card. Built on
read (not pre-computed) so existing sessions don't need a migration.

Source-of-truth priority (highest wins, by spec):
  1. PICOTT object (Phase 5A+)
  2. ClinicalConceptTranslationService output (Phase 5B)
  3. BooleanQueryCompiler concept blocks (Phase 5D)
  4. Legacy ``pico_json`` (only when ALL of the above are absent)

Invariants:
  - Legacy ``clinical_question_type`` MUST NOT override PICOTT
    ``type_of_question``.
  - Legacy ``specialty`` MUST NOT override PICOTT ``specialty``.
  - Legacy fragment ``key_concepts`` like ``"difference between"`` /
    ``"between high"`` MUST NOT be displayed when PICOTT
    ``must_have_concepts`` exist.
  - Empty / null / missing fields render as the literal string
    ``"Not specified"`` — never a blank dash.

The builder is pure (no DB writes). The router caches the result on
``search_sessions.interpreted_question_display_json`` after first
compute.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("clinevidence.interpreted_question_display")

NOT_SPECIFIED = "Not specified"

# Bump whenever the SHAPE of the returned display dict changes. The
# read path in ``routers/search_sessions.py`` compares this against the
# cached ``interpreted_question_display_json`` and rebuilds on mismatch
# — without it, every pre-existing session keeps serving the old card
# forever, because the cache is only ever invalidated by the narrow
# ``Other (...)`` heal.
#   1 — implicit baseline (pre-versioning)
#   2 — adds question_framework / question_frameworks / clinical_intent
DISPLAY_SCHEMA_VERSION = 2

# Closed set of legacy specialty values that must NEVER override a
# PICOTT specialty. ``primary_care`` is the most-frequent culprit on
# narrow specialty questions (cardiology, neurology, etc.).
_LEGACY_SPECIALTY_BLOCKLIST = {"primary_care", "general_practice", ""}

# ``ClinicalIntent`` / framework tokens are snake_case or acronyms.
# Acronyms (PICOTT, PIRD, CoCoPop, CHARMS...) are already display-ready
# and MUST NOT be title-cased; only the descriptive snake_case members
# (``concept_mechanism``, ``problem_context_differential``,
# ``guideline_stabilization``) need humanizing.
def _humanize_framework(framework: Any) -> str:
    """Render a framework token for the clinician card."""
    s = str(framework or "").strip()
    if not s:
        return NOT_SPECIFIED
    if "_" not in s:
        # Acronym form — preserve the canonical casing exactly.
        return s
    return s.replace("_", " ").strip().capitalize()


def _humanize_clinical_intent(intent: Any) -> str:
    """Render a ClinicalIntent token for the clinician card."""
    s = str(intent or "").strip()
    if not s:
        return NOT_SPECIFIED
    return s.replace("_", " ").strip().capitalize()


def framework_display_fields(session_doc: dict) -> dict[str, Any]:
    """Framework / intent fields from the persisted ClinicalQuestionMap.

    The CQM has been computed live on every search since Phase K1 and
    persisted to ``search_sessions.clinical_question_map_json``, but no
    clinician-facing surface ever read it — the card rendered a fixed
    PICO grid for all 14 frameworks, which is why a prevalence question
    (CoCoPop) and a diagnostic-accuracy question (PIRD) looked
    identical. Read-only; never affects retrieval or ranking.
    """
    cqm = session_doc.get("clinical_question_map_json") or {}
    if not isinstance(cqm, dict):
        return {
            "question_framework": NOT_SPECIFIED,
            "question_framework_raw": None,
            "question_frameworks": [],
            "clinical_intent": NOT_SPECIFIED,
            "clinical_intent_raw": None,
            "secondary_clinical_intents": [],
        }
    frameworks = [
        str(f).strip() for f in (cqm.get("frameworks") or [])
        if isinstance(f, str) and str(f).strip()
    ]
    primary_framework = frameworks[0] if frameworks else None
    primary_intent = cqm.get("primary_intent")
    secondary = [
        str(i).strip() for i in (cqm.get("secondary_intents") or [])
        if isinstance(i, str) and str(i).strip()
    ]
    return {
        "question_framework": _humanize_framework(primary_framework),
        "question_framework_raw": primary_framework,
        "question_frameworks": [_humanize_framework(f) for f in frameworks],
        "clinical_intent": _humanize_clinical_intent(primary_intent),
        "clinical_intent_raw": (
            str(primary_intent).strip() if primary_intent else None
        ),
        "secondary_clinical_intents": [
            _humanize_clinical_intent(i) for i in secondary
        ],
    }


def _str_or_not_specified(value: Any) -> str:
    """Render a string field — empty / null / whitespace → 'Not specified'."""
    if value is None:
        return NOT_SPECIFIED
    s = str(value).strip()
    return s if s else NOT_SPECIFIED


def _list_or_not_specified(value: Any) -> list[str]:
    """Render a list field — empty/None → ['Not specified']. Otherwise
    return a cleaned list of strings."""
    if not value:
        return [NOT_SPECIFIED]
    if not isinstance(value, list):
        return [NOT_SPECIFIED]
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            # Concept dicts: prefer "concept", fall back to "term".
            concept = item.get("concept") or item.get("term") or ""
            concept = str(concept).strip()
            if concept:
                out.append(concept)
        else:
            s = str(item).strip()
            if s:
                out.append(s)
    return out or [NOT_SPECIFIED]


def _picott_field(picott: dict, *path: str, default: Any = None) -> Any:
    """Walk a dotted path into the picott dict, returning ``default``
    on miss."""
    cur: Any = picott
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
        if cur is None:
            return default
    return cur


def _humanize_type_of_question(toq: str | None, subtype: str | None,
                                  role_weights: dict | None = None) -> str:
    """Render the type-of-question header using the same cascade as
    ``services.interpretation_canonical.canonical_label_for_rank_explanation``
    so the session-level display NEVER drifts from what the per-article
    ``rank_explanation`` says.

    Cascade (Phase 6B.3.1):
      1) PICOTT.type_of_question if concrete (non-"other")
      2) ``role_weights.type_of_question`` if concrete (non-"other")
      3) PICOTT.question_subtype mapped to a parent type via the
         canonical helper's ``_SUBTYPE_PARENT_TYPE`` table — produces
         e.g. "Pharmacology / metabolic biomarker effect".
      4) Subtype-only with "/" separator (never "(<subtype>)" parens).
      5) "Not specified".
    """
    from services.interpretation_canonical import (  # noqa: WPS433
        canonical_label_for_rank_explanation,
    )
    picott_payload = {
        "type_of_question": toq or "",
        "question_subtype": subtype or "",
    }
    label = canonical_label_for_rank_explanation(
        picott=picott_payload,
        role_weights=role_weights or {},
    )
    if label == "other":
        return NOT_SPECIFIED
    return label


def _humanize_question_scope(scope: str | None) -> str:
    if not scope:
        return NOT_SPECIFIED
    s = scope.strip().lower()
    if s == "foreground":
        return "Foreground"
    if s == "background":
        return "Background"
    if s == "hybrid":
        return "Hybrid"
    if s == "unclear":
        return "Unclear"
    return s.capitalize()


def _humanize_specialty(spec: str | None) -> str:
    if not spec or not str(spec).strip():
        return NOT_SPECIFIED
    s = str(spec).strip()
    # Normalize underscores → spaces, Title-Case for display.
    return s.replace("_", " ").title()



def _role_weights_for_qtype(toq: str | None) -> dict[str, int]:
    """Phase 5F — return the closed-set ROLE_WEIGHTS dict for the given
    question type so the clinician panel can render
    ``"Intervention · 40% weight (pharmacology)"`` per PICO row. Falls
    back to ``"other"`` weights if the type is unknown. Returns a fresh
    copy (caller may mutate)."""
    try:
        pass  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return {}


# Phase 5F.1 FIX 2 — closed-set pack → human specialty label map.
_PACK_SPECIALTY_LABELS = {
    "cardiometabolic_lipids": "Cardiometabolic / Lipidology",
    "critical_care_pulmonary": "Critical Care / Pulmonary",
    "immunization_vaccines": "Preventive Medicine / Infectious Disease",
    "cardiology_transfusion": "Cardiology",
}


def _derive_specialty_from_ontology(
    ontology_mappings: list[dict] | None,
) -> str | None:
    """Phase 5F.1 — majority-pack specialty derivation from ontology hits.

    Looks at each confident mapping's rationale for a pack name, and
    additionally does a reverse lookup of canonical_concept against the
    loaded packs (covers mappings where ``source=curated_mesh`` and the
    rationale doesn't carry the pack name explicitly).
    """
    if not ontology_mappings:
        return None
    pack_canonicals = _load_pack_canonical_index()
    counts: dict[str, int] = {}
    for m in ontology_mappings:
        if not isinstance(m, dict):
            continue
        if (m.get("confidence") or 0) < 0.85:
            continue
        # Try rationale-based lookup first.
        rationale = (m.get("rationale") or "").lower()
        pack_hit = None
        for pack_name in _PACK_SPECIALTY_LABELS:
            if pack_name in rationale:
                pack_hit = pack_name
                break
        # Reverse lookup by canonical_concept / synonyms.
        if not pack_hit:
            keys: list[str] = []
            c = m.get("canonical_concept") or ""
            if c:
                keys.append(c.lower())
            for s in (m.get("synonyms") or []):
                if s:
                    keys.append(s.lower())
            for k in keys:
                if k in pack_canonicals:
                    pack_hit = pack_canonicals[k]
                    break
        if pack_hit and pack_hit in _PACK_SPECIALTY_LABELS:
            counts[pack_hit] = counts.get(pack_hit, 0) + 1
    if not counts:
        return None
    winner = max(counts.items(), key=lambda x: x[1])[0]
    return _PACK_SPECIALTY_LABELS.get(winner)


# Phase 5F.1 — lazily-built reverse-lookup index of every concept
# canonical / synonym in the loaded specialty packs → pack name.
_PACK_CANONICAL_INDEX_CACHE: dict[str, str] | None = None


def _load_pack_canonical_index() -> dict[str, str]:
    global _PACK_CANONICAL_INDEX_CACHE  # noqa: PLW0603
    if _PACK_CANONICAL_INDEX_CACHE is not None:
        return _PACK_CANONICAL_INDEX_CACHE
    from pathlib import Path  # noqa: WPS433
    import json as _json  # noqa: WPS433
    packs_dir = Path(__file__).resolve().parent / "specialty_packs"
    idx: dict[str, str] = {}
    if not packs_dir.exists():
        _PACK_CANONICAL_INDEX_CACHE = idx
        return idx
    for p in sorted(packs_dir.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as fp:
                data = _json.load(fp)
        except Exception:  # noqa: BLE001
            continue
        pack_name = data.get("name") or p.stem
        groups = data.get("concept_groups") or {}
        for _g, group in groups.items():
            if not isinstance(group, dict):
                continue
            cls = group.get("class_name")
            if cls:
                idx[cls.lower()] = pack_name
            for s in (group.get("synonyms") or []):
                if s:
                    idx[s.lower()] = pack_name
            for m in (group.get("members") or group.get("concepts") or []):
                if not isinstance(m, dict):
                    continue
                name = (m.get("concept") or "").strip()
                if name:
                    idx[name.lower()] = pack_name
                for s in (m.get("synonyms") or []):
                    if s:
                        idx.setdefault(s.lower(), pack_name)
                md = m.get("mesh_descriptor")
                if md:
                    idx.setdefault(md.lower(), pack_name)
    _PACK_CANONICAL_INDEX_CACHE = idx
    return idx


def _ontology_fallback_for_block(
    resolved: list[str],
    ontology_mappings: list[dict] | None,
    accepted_roles: set[str],
) -> list[str]:
    """Phase 5F.1 FIX 2 — when the resolved block is empty / 'Not specified',
    fall back to ontology hits with a role_hint in ``accepted_roles``."""
    if resolved and resolved != [NOT_SPECIFIED]:
        return resolved
    if not ontology_mappings:
        return resolved
    seen: set[str] = set()
    out: list[str] = []
    for m in ontology_mappings:
        if not isinstance(m, dict):
            continue
        if (m.get("confidence") or 0) < 0.85:
            continue
        role = (m.get("role_hint") or "").lower()
        if role not in accepted_roles:
            continue
        canonical = (m.get("canonical_concept") or "").strip()
        if not canonical or canonical.lower() in seen:
            continue
        seen.add(canonical.lower())
        out.append(canonical)
    return out if out else resolved


def _derive_question_scope(
    must_have_concepts: list[dict] | None,
    raw_scope: str | None,
) -> str:
    """Phase 5F.1 FIX 2 — derive scope from the roles present on
    ``must_have_concepts``. PICOTT-declared scope wins when set; otherwise
    fall back to role-based inference. ``Foreground`` when any role in
    {intervention, comparator, outcome, exposure, index_test,
    target_diagnosis} is present; ``Background`` when only population
    roles are present; ``Unclear`` otherwise."""
    if raw_scope and str(raw_scope).strip().lower() not in ("", "unclear"):
        return raw_scope
    if not must_have_concepts:
        return raw_scope or "unclear"
    foreground_roles = {
        "intervention", "exposure", "index_test",
        "comparator", "reference_standard",
        "outcome", "target_diagnosis",
    }
    population_roles = {"population", "condition", "patient_group"}
    seen_fg = False
    seen_pop = False
    for mh in must_have_concepts:
        r = ((mh or {}).get("role") or "").lower()
        if r in foreground_roles:
            seen_fg = True
        elif r in population_roles:
            seen_pop = True
    if seen_fg:
        return "foreground"
    if seen_pop:
        return "background"
    return raw_scope or "unclear"


def _role_weights_for_qtype(toq: str | None) -> dict[str, int]:
    """Phase 5F — return the closed-set ROLE_WEIGHTS dict for the given
    question type so the clinician panel can render
    ``"Intervention · 40% weight (pharmacology)"`` per PICO row. Falls
    back to ``"other"`` weights if the type is unknown. Returns a fresh
    copy (caller may mutate)."""
    try:
        from services.role_weighting import ROLE_WEIGHTS  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return {}
    qt = (toq or "").strip().lower()
    return dict(ROLE_WEIGHTS.get(qt) or ROLE_WEIGHTS.get("other") or {})


def _concept_dicts_to_chips(concepts: list[Any]) -> list[dict[str, Any]]:
    """Turn a list of must_have/should_have/ranking concept dicts into
    a list of chip-shaped dicts for the UI. Each chip has ``concept``,
    ``role``, ``synonyms`` (list), and ``required`` (bool).
    """
    out: list[dict[str, Any]] = []
    for c in concepts or []:
        if not isinstance(c, dict):
            continue
        concept = (c.get("concept") or "").strip()
        if not concept:
            continue
        out.append({
            "concept": concept,
            "role": c.get("role") or "other",
            "synonyms": [s for s in (c.get("synonyms") or []) if s],
            "required_for_high_relevance": bool(
                c.get("required_for_high_relevance", True)
            ),
        })
    return out


# Phase 5D.1 Fix 1 — derive PICO blocks from must_have_concepts[].role
# when ``picott_json.pico.<field>`` is empty. The PICOTT extractor
# sometimes populates must_have_concepts but leaves the structured
# pico sub-dict empty; we recover the values from the role tags here
# so the clinician card doesn't render ``"Not specified"`` for fields
# the parser actually identified.
#
# Role mapping (closed-set; mirrors models/picott.py::RankingConcept):
#   population:              ``role`` in {"population", "condition", "patient_group"}
#   intervention_or_exposure_or_index_test:
#                            ``role`` in {"intervention", "exposure", "index_test"}
#   comparator_or_reference_standard:
#                            ``role`` in {"comparator", "reference_standard"}
#   outcomes_or_target_diagnosis:
#                            ``role`` in {"outcome", "target_diagnosis"}
_ROLE_TO_PICO_FIELD = {
    "population": ("population", "condition", "patient_group"),
    "intervention_or_exposure_or_index_test":
        ("intervention", "exposure", "index_test"),
    "comparator_or_reference_standard":
        ("comparator", "reference_standard"),
    "outcomes_or_target_diagnosis":
        ("outcome", "target_diagnosis"),
}


def _derive_pico_from_must_haves(
    must_have_concepts: list[Any],
    role_targets: tuple[str, ...],
) -> list[str]:
    """Filter must_have_concepts by role, returning concept strings."""
    out: list[str] = []
    seen: set[str] = set()
    for c in must_have_concepts or []:
        if not isinstance(c, dict):
            continue
        role = (c.get("role") or "").lower().strip()
        if role not in role_targets:
            continue
        concept = (c.get("concept") or "").strip()
        if not concept:
            continue
        key = concept.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(concept)
    return out


def _derive_pico_from_translations(
    concept_translations: list[Any],
    role_targets: tuple[str, ...],
) -> list[str]:
    """Filter ClinicalConceptTranslation outputs by role attribute,
    returning the canonical_concept strings."""
    out: list[str] = []
    seen: set[str] = set()
    for t in concept_translations or []:
        if not isinstance(t, dict):
            continue
        role = (t.get("role") or "").lower().strip()
        if role not in role_targets:
            continue
        canonical = (
            t.get("canonical_concept")
            or t.get("canonical")
            or t.get("term")
            or ""
        )
        canonical = str(canonical).strip()
        if not canonical:
            continue
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(canonical)
    return out


def _resolve_pico_block(
    pico_blocks: dict | None,
    field_name: str,
    must_have_concepts: list[Any],
    concept_translations: list[Any],
    ranking_concepts: list[Any] | None = None,
) -> list[str]:
    """Phase 5D.1 Fix 1 — resolve a PICO block value with the cascade:

      1. ``picott_json.pico.<field_name>``  (if non-empty)
      2. ``must_have_concepts`` filtered by ``role`` matching the field
      3. ``concept_translations`` filtered by ``role`` matching the field
      4. ``ranking_concepts`` filtered by ``role`` matching the field
         (used as 3rd-source fallback for ``outcomes_or_target_diagnosis``
         only — outcomes often live on ranking concepts when the parser
         doesn't promote them to must-have)
      5. ``[NOT_SPECIFIED]`` literal — only if all four sources are empty

    Returns a list of strings (never a list of ``"Not specified"`` mixed
    with real terms; ``"Not specified"`` is only used when there are NO
    real terms across any source).
    """
    # Source 1 — structured pico block.
    if isinstance(pico_blocks, dict):
        raw = pico_blocks.get(field_name)
        cleaned = _list_or_not_specified(raw)
        # _list_or_not_specified returns [NOT_SPECIFIED] for empty input;
        # only accept it if the real source had content.
        if cleaned and cleaned != [NOT_SPECIFIED]:
            return cleaned

    role_targets = _ROLE_TO_PICO_FIELD.get(field_name) or ()
    if not role_targets:
        return [NOT_SPECIFIED]

    # Source 2 — must_have_concepts by role.
    from_must = _derive_pico_from_must_haves(must_have_concepts, role_targets)
    # Source 3 — concept_translations by role.
    from_trans = _derive_pico_from_translations(
        concept_translations, role_targets,
    )

    combined: list[str] = []
    seen: set[str] = set()
    for src in (from_must, from_trans):
        for v in src:
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            combined.append(v)

    # Source 4 — ranking_concepts (outcomes-only fallback).
    if not combined and field_name == "outcomes_or_target_diagnosis":
        from_rank = _derive_pico_from_must_haves(
            ranking_concepts, role_targets,
        )
        for v in from_rank:
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            combined.append(v)

    return combined if combined else [NOT_SPECIFIED]


def _has_picott(picott: dict | None) -> bool:
    """A PICOTT 'exists' when at least one of must_have_concepts or
    the pico blocks have content — i.e., the parser actually produced
    a structured interpretation. An empty dict / None / all-empty
    fields fall through to legacy."""
    if not picott or not isinstance(picott, dict):
        return False
    if picott.get("must_have_concepts"):
        return True
    pico = picott.get("pico") or {}
    if isinstance(pico, dict):
        for k in (
            "population",
            "intervention_or_exposure_or_index_test",
            "comparator_or_reference_standard",
            "outcomes_or_target_diagnosis",
        ):
            if pico.get(k):
                return True
    return False


def build_interpreted_question_display(session_doc: dict) -> dict[str, Any]:
    """Build the canonical PICOTT display JSON from a session document.

    See module docstring for the source-of-truth cascade. Returns a
    JSON-serializable dict — safe to embed in any HTTP response.
    """
    picott = session_doc.get("picott_json") or {}
    legacy_pico = session_doc.get("pico_json") or {}
    legacy_qtype = session_doc.get("clinical_question_type") or ""

    has_picott = _has_picott(picott)

    if has_picott:
        # ---- Phase 5A+ path — PICOTT is authoritative. -----------------
        toq = _picott_field(picott, "type_of_question")
        subtype = _picott_field(picott, "question_subtype")
        scope = _picott_field(picott, "question_scope")
        specialty = _picott_field(picott, "specialty")

        # Phase 5F.1 FIX 2 — cascade hydration sources. Read context-engine
        # outputs the pipeline persists on the session.
        role_weights_doc = session_doc.get("role_weights_json") or {}
        rw_qtype = (role_weights_doc.get("type_of_question") or "").strip().lower()
        rw_subtype = (role_weights_doc.get("question_subtype") or "")
        ontology_mappings_doc = session_doc.get("ontology_mappings_json") or []
        multi_intent_doc = session_doc.get("multi_intent_json") or {}

        # Cascade for ``question_subtype``:
        #   1) picott.question_subtype if non-null AND non-"other"
        #   2) role_weights_json.question_subtype
        #   3) multi_intent.sub_questions[0].subtype
        #   4) "Not specified"
        subtype_cascaded = subtype if (
            subtype and str(subtype).strip()
            and str(subtype).strip().lower() != "other"
        ) else None
        if not subtype_cascaded and rw_subtype:
            subtype_cascaded = rw_subtype
        if not subtype_cascaded:
            subs = multi_intent_doc.get("sub_questions") or []
            if subs:
                first = subs[0] or {}
                fs = first.get("subtype") or first.get("question_subtype")
                if fs:
                    subtype_cascaded = fs

        # Cascade for ``type_of_question``:
        #   1) picott.type_of_question if non-"other"
        #   2) role_weights_json.type_of_question if non-"other"
        #   3) keep raw (which may be "other")
        toq_cascaded = toq if (
            toq and str(toq).strip().lower() != "other"
        ) else None
        if not toq_cascaded and rw_qtype and rw_qtype != "other":
            toq_cascaded = rw_qtype

        # Cascade for ``specialty``:
        #   1) picott.specialty if non-blocklisted (not primary_care)
        #   2) majority specialty_tag from ontology_mappings_json
        #   3) "Primary Care"
        if specialty and str(specialty).strip().lower() not in _LEGACY_SPECIALTY_BLOCKLIST:
            specialty_display = _humanize_specialty(specialty)
        else:
            specialty_from_ontology = _derive_specialty_from_ontology(
                ontology_mappings_doc,
            )
            if specialty_from_ontology:
                specialty_display = specialty_from_ontology
            elif specialty and str(specialty).strip():
                # Blocklisted but at least non-empty — render legacy.
                specialty_display = _humanize_specialty(specialty)
            else:
                specialty_display = NOT_SPECIFIED

        pico_blocks = picott.get("pico") or {}
        # Phase 5B — translation outputs (already inside picott_json).
        translations = picott.get("concept_translations") or []
        spelling_corrections = picott.get("spelling_corrections") or []
        ideal_study_types = picott.get("ideal_study_types") or []
        latent_concepts = picott.get("latent_concepts") or []
        translated_question = picott.get("translated_question") or ""
        translation_warnings = picott.get("translation_warnings") or []
        interpretation_for_user = picott.get("interpretation_for_user") or ""
        rationale = picott.get("rationale") or ""
        confidence = picott.get("confidence")

        # Phase 5D.1 Fix 1 — when ``picott_json.pico.<field>`` is empty
        # (which happens when the extractor populates only
        # ``must_have_concepts`` with role tags), derive the PICO block
        # values from must_have_concepts → concept_translations →
        # ranking_concepts in that order. Only ``"Not specified"`` if
        # ALL three sources are empty for that role.
        must_have_raw = picott.get("must_have_concepts") or []
        ranking_raw = picott.get("ranking_concepts") or []
        population_resolved = _resolve_pico_block(
            pico_blocks, "population",
            must_have_raw, translations, ranking_raw,
        )
        intervention_resolved = _resolve_pico_block(
            pico_blocks, "intervention_or_exposure_or_index_test",
            must_have_raw, translations, ranking_raw,
        )
        comparator_resolved = _resolve_pico_block(
            pico_blocks, "comparator_or_reference_standard",
            must_have_raw, translations, ranking_raw,
        )
        outcomes_resolved = _resolve_pico_block(
            pico_blocks, "outcomes_or_target_diagnosis",
            must_have_raw, translations, ranking_raw,
        )

        # Phase 5F.1 FIX 2 — final fallback for the must-have arrays:
        # ontology hits with matching role_hint.
        population_resolved = _ontology_fallback_for_block(
            population_resolved, ontology_mappings_doc,
            {"population", "condition", "patient_group"},
        )
        intervention_resolved = _ontology_fallback_for_block(
            intervention_resolved, ontology_mappings_doc,
            {"intervention", "exposure", "index_test"},
        )
        comparator_resolved = _ontology_fallback_for_block(
            comparator_resolved, ontology_mappings_doc,
            {"comparator", "reference_standard"},
        )
        outcomes_resolved = _ontology_fallback_for_block(
            outcomes_resolved, ontology_mappings_doc,
            {"outcome", "target_diagnosis"},
        )

        # Phase 5F.1 FIX 2 — question_scope derived from the resolved
        # roles on must_have_concepts.
        scope_cascaded = _derive_question_scope(must_have_raw, scope)

        return {
            "source": "picott",
            "has_picott": True,
            "display_schema_version": DISPLAY_SCHEMA_VERSION,
            # Framework / intent badges from the ClinicalQuestionMap.
            **framework_display_fields(session_doc),
            # Top-line badges (Phase 5F.1 cascade)
            "question_scope": _humanize_question_scope(scope_cascaded),
            "type_of_question": _humanize_type_of_question(
                toq_cascaded or toq, subtype_cascaded,
                role_weights=role_weights_doc or {},
            ),
            "type_of_question_raw": (toq_cascaded or toq or "other"),
            "question_subtype": _str_or_not_specified(subtype_cascaded),
            "specialty": specialty_display,
            # PICO blocks → chip-list rows. Phase 5D.1 Fix 1 — use the
            # resolved values (pico → must_have_concepts → translations →
            # ranking) so empty pico.<field> arrays are recovered from
            # the role-tagged must-have / translation / ranking sources.
            "population": population_resolved,
            "intervention_or_exposure_or_index_test": intervention_resolved,
            "comparator_or_reference_standard": comparator_resolved,
            "outcomes_or_target_diagnosis": outcomes_resolved,
            "time_horizon": _list_or_not_specified(
                pico_blocks.get("time_horizon")
                if isinstance(pico_blocks, dict) else None
            ),
            "setting": _list_or_not_specified(
                pico_blocks.get("setting")
                if isinstance(pico_blocks, dict) else None
            ),
            "ideal_study_types": (
                list(ideal_study_types) if ideal_study_types
                else [NOT_SPECIFIED]
            ),
            # Concept lists
            "must_have_concepts": _concept_dicts_to_chips(
                picott.get("must_have_concepts") or []
            ),
            "should_have_concepts": _concept_dicts_to_chips(
                picott.get("should_have_concepts") or []
            ),
            "ranking_concepts": _concept_dicts_to_chips(
                picott.get("ranking_concepts") or []
            ),
            # Phase 5B — translation outputs
            "concept_translations": list(translations),
            "translated_question": _str_or_not_specified(translated_question),
            "latent_concepts": list(latent_concepts),
            "translation_warnings": list(translation_warnings),
            # Free-text strings
            "spelling_corrections": list(spelling_corrections),
            "interpretation_for_user": _str_or_not_specified(
                interpretation_for_user
            ),
            "rationale": _str_or_not_specified(rationale),
            "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
            # Phase 5F — shorthand substitutions + role-weight + ontology
            # mappings + role-repair provenance. The clinician transparency
            # panel reads these directly. Backward-compatible: when
            # missing, the panel renders nothing.
            "shorthand_substitutions": list(
                session_doc.get("shorthand_substitutions_json")
                or picott.get("shorthand_substitutions") or []
            ),
            "role_weights": _role_weights_for_qtype(toq_cascaded or toq),
            "ontology_mappings": list(
                session_doc.get("ontology_mappings_json")
                or picott.get("ontology_mappings") or []
            ),
            "ontology_lifts": list(
                session_doc.get("ontology_lifts_json") or []
            ),
            "role_repairs": list(
                session_doc.get("role_repairs_json")
                or picott.get("role_repairs") or []
            ),
            "multi_intent": dict(
                session_doc.get("multi_intent_json") or {}
            ),
            "question_subtype_raw": (subtype_cascaded or "") or None,
            # Demoted legacy fields (kept for the "Advanced legacy" debug
            # toggle on the UI; never displayed on the main card).
            "legacy_clinical_question_type": legacy_qtype or None,
            "legacy_pico_json": legacy_pico or {},
        }

    # ---- Legacy-only fallback path ---------------------------------
    # PICOTT absent / empty → display the legacy pico_json directly. We
    # still wrap it in the same schema so the frontend can use a single
    # render path.
    legacy = legacy_pico or {}
    return {
        "source": "legacy_pico_json",
        "has_picott": False,
        "display_schema_version": DISPLAY_SCHEMA_VERSION,
        # The CQM is built independently of PICOTT, so a legacy-only
        # session can still carry a valid framework.
        **framework_display_fields(session_doc),
        "question_scope": NOT_SPECIFIED,
        "type_of_question": _humanize_type_of_question(legacy_qtype, None),
        "type_of_question_raw": legacy_qtype or "other",
        "question_subtype": NOT_SPECIFIED,
        "specialty": NOT_SPECIFIED,
        # Best-effort: legacy population is a single string, not a list.
        "population": _list_or_not_specified(
            [legacy.get("population")] if legacy.get("population") else []
        ),
        "intervention_or_exposure_or_index_test": _list_or_not_specified(
            [legacy.get("intervention_or_exposure")]
            if legacy.get("intervention_or_exposure") else []
        ),
        "comparator_or_reference_standard": _list_or_not_specified(
            [legacy.get("comparator")] if legacy.get("comparator") else []
        ),
        "outcomes_or_target_diagnosis": _list_or_not_specified(
            legacy.get("outcomes")
        ),
        "time_horizon": [NOT_SPECIFIED],
        "setting": [NOT_SPECIFIED],
        "ideal_study_types": [NOT_SPECIFIED],
        "must_have_concepts": [],
        "should_have_concepts": [],
        "ranking_concepts": [],
        "concept_translations": [],
        "translated_question": NOT_SPECIFIED,
        "latent_concepts": [],
        "translation_warnings": [],
        "spelling_corrections": [],
        "interpretation_for_user": NOT_SPECIFIED,
        "rationale": NOT_SPECIFIED,
        "confidence": None,
        "legacy_clinical_question_type": legacy_qtype or None,
        "legacy_pico_json": legacy,
    }


# ---------------------------------------------------------------------
# Phase M4.1.6 — M4 read-side display projection.
#
# When the bounded-LLM M4 active repair was APPLIED for a degraded
# baseline, the clinician PICOTT card MUST reflect the M4 structured
# interpretation, NOT the degraded baseline fused literal. This builder
# maps an ACCEPTED + APPLIED ``M4Candidate`` (``model_dump``) onto the
# SAME display schema produced by ``build_interpreted_question_display``
# so the frontend render path is unchanged. Display-only; no retrieval /
# ranking / default-selection logic is touched.
# ---------------------------------------------------------------------

# Closed-set human labels for M4 ``primary_intent`` /
# ``secondary_intents`` enum values (mirrors
# ``services.semantic_anchor_repair.ALLOWED_PRIMARY_INTENTS``). Unknown
# values fall back to a Title-cased token.
_M4_INTENT_LABELS = {
    "prognosis_natural_history": "Prognosis / natural history",
    "prognosis": "Prognosis",
    "prognostic_factor": "Prognostic factor",
    "therapy": "Therapy",
    "therapy_safety": "Therapy / safety",
    "diagnosis": "Diagnosis",
    "diagnostic_accuracy": "Diagnostic accuracy",
    "rule_out_test_utility": "Rule-out test utility",
    "screening": "Screening",
    "etiology": "Etiology",
    "risk_etiology": "Risk / etiology",
    "risk_etiology_harm": "Risk / etiology / harm",
    "complication_risk": "Complication risk",
    "complication_after_procedure": "Complication after procedure",
    "mechanism": "Mechanism",
    "pathophysiology": "Pathophysiology",
    "epidemiology_prevalence_incidence": "Epidemiology / prevalence / incidence",
    "drug_overview": "Drug overview",
    "drug_safety": "Drug safety",
    "disease_overview": "Disease overview",
    "background_overview": "Background overview",
    "definition": "Definition",
    "severity_marker_overview": "Severity marker overview",
    "other": NOT_SPECIFIED,
}


def _humanize_m4_intent(intent: Any) -> str:
    """Render an M4 ``primary_intent`` / ``secondary_intent`` enum value
    as a clinician-facing label. Closed-set map first; unknown values
    Title-case the underscored token. Empty/``other`` → 'Not specified'."""
    if not intent:
        return NOT_SPECIFIED
    key = str(intent).strip().lower()
    if not key:
        return NOT_SPECIFIED
    if key in _M4_INTENT_LABELS:
        return _M4_INTENT_LABELS[key]
    return key.replace("_", " ").capitalize()


def _m4_list(candidate: dict, *keys: str) -> list[str]:
    """Collect + de-duplicate (case-insensitive, order-preserving) the
    string members of one-or-more M4 candidate list fields."""
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        v = candidate.get(k)
        if not isinstance(v, list):
            continue
        for item in v:
            s = str(item).strip()
            if not s:
                continue
            kl = s.lower()
            if kl in seen:
                continue
            seen.add(kl)
            out.append(s)
    return out


def build_m4_interpreted_display(
    m4_candidate: dict, base_display: dict | None = None,
) -> dict[str, Any]:
    """Build the canonical PICOTT display JSON from an APPLIED M4
    candidate. Starts from ``base_display`` (so badges the M4 candidate
    does not carry — specialty, question_scope, ideal_study_types — are
    preserved) and overrides the interpretation-bearing fields with the
    M4 structured values. Sets ``source`` / ``interpretation_source`` to
    ``"m4"``.
    """
    c = m4_candidate if isinstance(m4_candidate, dict) else {}
    base = dict(base_display) if isinstance(base_display, dict) else {}

    population = _m4_list(c, "population") or _m4_list(c, "clinical_problem")
    intervention = _m4_list(
        c, "intervention", "exposure", "index_test", "medication", "procedure",
    )
    comparator = _m4_list(c, "comparator")
    outcomes = _m4_list(c, "outcome")
    time_horizon = _m4_list(c, "timeframe")
    setting = _m4_list(c, "setting")
    required = _m4_list(c, "required_anchors")

    secondary = c.get("secondary_intents") or []
    subtype = (
        _humanize_m4_intent(secondary[0])
        if isinstance(secondary, list) and secondary else NOT_SPECIFIED
    )

    base.update({
        "source": "m4",
        "interpretation_source": "m4",
        "has_picott": True,
        "type_of_question": _humanize_m4_intent(c.get("primary_intent")),
        "type_of_question_raw": (c.get("primary_intent") or "other"),
        "question_subtype": subtype,
        "population": population or [NOT_SPECIFIED],
        "intervention_or_exposure_or_index_test": (
            intervention or [NOT_SPECIFIED]
        ),
        "comparator_or_reference_standard": comparator or [NOT_SPECIFIED],
        "outcomes_or_target_diagnosis": outcomes or [NOT_SPECIFIED],
        "time_horizon": time_horizon or [NOT_SPECIFIED],
        "setting": setting or [NOT_SPECIFIED],
        "must_have_concepts": [
            {
                "concept": a,
                "role": "condition",
                "synonyms": [],
                "required_for_high_relevance": True,
            }
            for a in required
        ],
    })
    return base


__all__ = [
    "NOT_SPECIFIED",
    "build_interpreted_question_display",
    "build_m4_interpreted_display",
]
