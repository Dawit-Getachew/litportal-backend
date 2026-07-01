"""Phase 5F — ClinicalOntologyMapperService.

Maps a free-text term to a canonical concept with MeSH / RxNorm / UMLS
provenance. The mapper is the single source of truth for concept
canonicalization downstream of the PICOTT extractor — both the role
repair service and the Boolean query compiler read its output to
decide how to render a concept.

Hard rules:
  - PHI guard runs FIRST (in the route handler). The mapper never sees
    raw PHI text.
  - Curated tables only. ``umls_cui`` is optional and may be null when
    not curated. We NEVER fabricate MeSH descriptors or RxCUIs.
  - Source order (highest-confidence wins):
      1. Curated MeSH descriptors (``services/mesh/mesh_descriptors.json``)
      2. Curated RxNorm class mappings (``services/data/rxnorm_curated.json``)
      3. Specialty pack concepts (``services/specialty_packs/*.json``)
      4. Clinical shorthand expansions
      5. Acronym expansions from ``services/concept_expansion.py``
  - If no confident match is found: returns confidence=0.0 with
    ``canonical_concept`` empty, keeping the original term.
  - Pure / sync.

Output shape (``OntologyMapping``):
  {
    "original_term": str,
    "normalized_term": str,
    "canonical_concept": str,
    "concept_type": "drug" | "drug_class" | "disease" | "syndrome" |
                    "biomarker" | "lab" | "diagnostic_test" | "vaccine" |
                    "procedure" | "population" | "outcome" | "other",
    "role_hint": "intervention" | "exposure" | "index_test" |
                 "comparator" | "reference_standard" | "outcome" |
                 "target_diagnosis" | "population" | "condition" |
                 "patient_group" | "other",
    "synonyms": [str, ...],
    "acronyms": [str, ...],
    "spelling_variants": [str, ...],
    "mesh_terms": [str, ...],
    "rxnorm_terms": [str, ...],
    "umls_cui": str | null,
    "confidence": float in [0, 1],
    "source": "curated_mesh" | "curated_rxnorm" |
              "curated_umls_synonyms" | "specialty_pack" |
              "clinical_shorthand" | "acronym_dict" | "none",
    "rationale": str,
  }
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.clinical_shorthand_normalizer import (
    CLINICAL_SPELLING_PAIRS,
    DRUG_CLASS_SHORTHAND,
    LAB_SHORTHAND,
    PROCEDURE_ACRONYMS,
)
from services.concept_expansion import (
    ACRONYM_PAIRS,
    DRUG_GENERIC_TO_CLASS,
    expand_concept,
)
from services.mesh import mesh_resolver

logger = logging.getLogger("clinevidence.ontology_mapper")


# Phase 5F — closed-set stop tokens for ``extract_from_text``. Mirrors
# the ATM stop-list shape but is independent of the compiler's
# STOP_PHRASES (which strips phrases inside already-formed queries).
_EXTRACT_STOP_TOKENS: frozenset[str] = frozenset({
    "what", "what's", "when", "how", "why", "where", "who",
    "the", "a", "an", "of", "to", "for", "with", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done",
    "and", "or", "but", "not", "no",
    "right", "best", "good", "better",
    "patient", "patients", "adult", "adults", "child", "children",
    "get", "got", "have", "has", "had",
    "its", "their", "his", "her",
    "this", "that", "these", "those",
    "by", "from", "as", "vs", "versus",
    "dose", "timing", "administer", "impact", "effect",
    "recommended", "benefit", "benefits",
})


# Closed-set enums for the OntologyMapping output.
CONCEPT_TYPES = {
    "drug", "drug_class", "disease", "syndrome", "biomarker", "lab",
    "diagnostic_test", "vaccine", "procedure", "population",
    "outcome", "other",
}
ROLE_HINTS = {
    "intervention", "exposure", "index_test", "comparator",
    "reference_standard", "outcome", "target_diagnosis", "population",
    "condition", "patient_group", "other",
}
MAPPING_SOURCES = {
    "curated_mesh", "curated_rxnorm", "curated_umls_synonyms",
    "specialty_pack", "clinical_shorthand", "acronym_dict", "none",
}


# ---------------------------------------------------------------------------
# Specialty-pack loader
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PACKS_DIR = _HERE / "specialty_packs"


def _load_specialty_packs() -> list[dict[str, Any]]:
    """Load all JSON files under ``specialty_packs/``. Returns a list of
    pack dicts (one per file)."""
    packs: list[dict[str, Any]] = []
    if not _PACKS_DIR.exists():
        return packs
    for p in sorted(_PACKS_DIR.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as fp:
                packs.append(json.load(fp))
        except Exception:  # noqa: BLE001
            logger.exception("failed to load specialty pack %s", p)
    return packs


def _load_rxnorm_table() -> dict[str, Any]:
    p = _HERE / "data" / "rxnorm_curated.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:  # noqa: BLE001
        logger.exception("failed to load RxNorm curated table")
        return {}


@dataclass
class OntologyMapping:
    original_term: str
    normalized_term: str = ""
    canonical_concept: str = ""
    concept_type: str = "other"
    role_hint: str = "other"
    synonyms: list[str] = field(default_factory=list)
    acronyms: list[str] = field(default_factory=list)
    spelling_variants: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    rxnorm_terms: list[str] = field(default_factory=list)
    umls_cui: str | None = None
    confidence: float = 0.0
    source: str = "none"
    rationale: str = ""
    # Phase 6B.3 — closed-set hints lifted from specialty packs. Used
    # by clinical_role_repair to seed canonical question type when the
    # LLM PICOTT extractor returned ``other``.
    question_type_hint: str | None = None
    question_subtype_hint: str | None = None
    pack_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_term": self.original_term,
            "normalized_term": self.normalized_term,
            "canonical_concept": self.canonical_concept,
            "concept_type": self.concept_type,
            "role_hint": self.role_hint,
            "synonyms": list(self.synonyms),
            "acronyms": list(self.acronyms),
            "spelling_variants": list(self.spelling_variants),
            "mesh_terms": list(self.mesh_terms),
            "rxnorm_terms": list(self.rxnorm_terms),
            "umls_cui": self.umls_cui,
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
            "rationale": self.rationale,
            # Phase 6B.3.
            "question_type_hint": self.question_type_hint,
            "question_subtype_hint": self.question_subtype_hint,
            "pack_name": self.pack_name,
        }


class ClinicalOntologyMapperService:
    """Maps free-text terms to canonical concepts using curated tables.

    The mapper is initialized once at import time; the JSON tables are
    re-loaded on each instance creation, which keeps tests fast.
    """

    def __init__(self):
        self._packs = _load_specialty_packs()
        self._rxnorm = _load_rxnorm_table()
        # Build a lower-case lookup index for specialty-pack concepts.
        self._pack_index: dict[str, dict[str, Any]] = {}
        self._build_pack_index()
        # RxNorm lookup: maps lower-cased synonym/concept → (entry, class_concept).
        self._rxnorm_index: dict[str, tuple[dict[str, Any], str | None]] = {}
        self._build_rxnorm_index()

    # ------------------------------------------------------------------
    # Phase 6B.3 — Morphology variants (curated stem rules ONLY).
    # ------------------------------------------------------------------
    @staticmethod
    def _morphology_variants(term: str) -> list[str]:
        """Generate the closed-set of conservative morphology variants
        for a curated term. No broad fuzzy matching; rules are scoped
        to vocabulary actually seen in clinical literature.
        """
        if not term:
            return []
        t = term.strip()
        if not t:
            return []
        low = t.lower()
        out: set[str] = {low}

        # ---- 1. Plural / singular pivot. ----
        # singular → +s, +es
        out.add(low + "s")
        if low.endswith("y") and len(low) > 2 and low[-2] not in "aeiou":
            out.add(low[:-1] + "ies")  # therapy → therapies
        else:
            out.add(low + "es")
        # plural → trim
        if low.endswith("ies") and len(low) > 3:
            out.add(low[:-3] + "y")
        if low.endswith("es") and len(low) > 3:
            out.add(low[:-2])
        if low.endswith("s") and len(low) > 2 and not low.endswith("ss"):
            out.add(low[:-1])

        # ---- 2. vaccine ↔ vaccination ↔ immunization family. ----
        vacc_families: tuple[tuple[str, ...], ...] = (
            ("vaccine", "vaccines", "vaccination", "vaccinations",
             "immunization", "immunizations", "immunisation", "immunisations"),
        )
        for family in vacc_families:
            for root in family:
                if root in low:
                    for alt in family:
                        out.add(low.replace(root, alt))

        # ---- 3. Corticosteroid family. ----
        steroid_family: tuple[str, ...] = (
            "steroid", "steroids", "corticosteroid", "corticosteroids",
            "glucocorticoid", "glucocorticoids", "systemic corticosteroid",
            "systemic corticosteroids",
        )
        if any(root in low for root in steroid_family):
            for root in steroid_family:
                if root in low:
                    for alt in steroid_family:
                        out.add(low.replace(root, alt))

        # ---- 4. Hyphen / space / no-hyphen variants. ----
        if "-" in low:
            out.add(low.replace("-", ""))
            out.add(low.replace("-", " "))
        if " " in low:
            out.add(low.replace(" ", ""))
            out.add(low.replace(" ", "-"))
        # Numeric-suffix hyphenation: ``glp1`` ↔ ``glp-1`` ↔ ``glp 1``.
        for num in ("1", "2", "3", "4"):
            if low.endswith(num) and len(low) > len(num):
                stem = low[: -len(num)].rstrip("-").rstrip(" ")
                out.add(stem + num)
                out.add(stem + "-" + num)
                out.add(stem + " " + num)

        # ---- 5. -RA / -ra receptor-agonist suffix expansion. ----
        if low.endswith(" ra") or low.endswith("-ra") or low.endswith("ra"):
            stem = low.rstrip("-ra ").rstrip(" ")
            if stem and len(stem) >= 2:
                out.add(stem + " receptor agonist")
                out.add(stem + " receptor agonists")

        # ---- 6. LDL-C variants. ----
        if low in {"ldl", "ldl-c", "ldlc", "ldl c", "ldl cholesterol",
                   "ldl-cholesterol"}:
            out.update({"ldl", "ldl-c", "ldlc", "ldl c", "ldl cholesterol",
                        "ldl-cholesterol", "low-density lipoprotein",
                        "low density lipoprotein", "low-density lipoprotein cholesterol"})

        # ---- 7. British / American spelling pairs. ----
        spelling_pairs: tuple[tuple[str, str], ...] = (
            ("aetiology", "etiology"),
            ("haemoglobin", "hemoglobin"),
            ("anaemia", "anemia"),
            ("orthopaedic", "orthopedic"),
            ("oedema", "edema"),
            ("oesophag", "esophag"),
            ("cholesterole", "cholesterol"),  # common typo
        )
        for a, b in spelling_pairs:
            if a in low:
                out.add(low.replace(a, b))
            if b in low:
                out.add(low.replace(b, a))

        return sorted({v for v in out if v})

    @staticmethod
    def _index_term(index: dict[str, Any], term: str, entry: Any,
                     *, overwrite: bool = False) -> None:
        """Register all morphology variants of ``term`` in ``index``.
        Uses setdefault so the first (highest-confidence) entry wins
        when variants collide — unless ``overwrite`` is True.
        """
        for v in ClinicalOntologyMapperService._morphology_variants(term):
            if overwrite:
                index[v] = entry
            else:
                index.setdefault(v, entry)

    # ------------------------------------------------------------------
    # Index builders
    # ------------------------------------------------------------------
    def _build_pack_index(self) -> None:
        for pack in self._packs:
            pack_name = pack.get("name") or ""
            groups = pack.get("concept_groups") or {}
            for group_name, group in groups.items():
                if not isinstance(group, dict):
                    continue
                concept_type = group.get("concept_type") or "other"
                role_hint = group.get("role_hint") or "other"
                # Phase 6B.3 — pack-level question_type / question_subtype
                # hints. When the LLM picott returned ``other`` and a
                # mapped concept carries these hints, the role-repair
                # service uses them to seed the canonical type.
                group_q_type_hint = group.get("question_type_hint")
                group_q_subtype_hint = group.get("question_subtype_hint")
                # Top-level "class" entry (e.g. GLP-1 receptor agonists)
                class_name = group.get("class_name")
                if class_name:
                    syns = list(group.get("synonyms") or [])
                    mesh_terms = list(group.get("mesh_class_descriptors") or [])
                    conf = float(group.get("ontology_confidence") or 0.85)
                    entry = {
                        "canonical": class_name,
                        "concept_type": "drug_class",
                        "role_hint": role_hint,
                        "synonyms": syns,
                        "mesh_terms": mesh_terms,
                        "rxnorm_terms": [m["concept"] for m in (group.get("members") or []) if m.get("concept")],
                        "umls_cui": None,
                        "confidence": conf,
                        "source": "specialty_pack",
                        "pack_name": pack_name,
                        "group_name": group_name,
                        "question_type_hint": group_q_type_hint,
                        "question_subtype_hint": group_q_subtype_hint,
                    }
                    # Phase 6B.3 — register the canonical + all synonyms
                    # via the morphology helper so plural / hyphenation /
                    # vaccine-vaccination / etc. variants resolve.
                    self._index_term(self._pack_index, class_name, entry)
                    for s in syns:
                        self._index_term(self._pack_index, s, entry)
                # Member-level concepts (e.g. semaglutide, liraglutide, ...
                # OR lipid_outcomes.concepts).
                members = group.get("members") or group.get("concepts") or []
                for m in members:
                    if not isinstance(m, dict):
                        continue
                    name = (m.get("concept") or "").strip()
                    if not name:
                        continue
                    m_syns = list(m.get("synonyms") or [])
                    m_mesh = ([m["mesh_descriptor"]]
                              if m.get("mesh_descriptor") else [])
                    m_conf = float(m.get("ontology_confidence") or 0.90)
                    rx = m.get("rxnorm_rxcui") or m.get("rxcui")
                    entry = {
                        "canonical": name,
                        "concept_type": (
                            "drug" if m.get("rxnorm_rxcui") or m.get("rxcui")
                            else concept_type
                        ),
                        "role_hint": role_hint,
                        "synonyms": m_syns,
                        "mesh_terms": m_mesh,
                        "rxnorm_terms": [rx] if rx else [],
                        "umls_cui": m.get("umls_cui"),
                        "confidence": m_conf,
                        "source": "specialty_pack",
                        "pack_name": pack_name,
                        "group_name": group_name,
                        "is_glp1_ra": bool(m.get("is_glp1_ra")),
                        "related_cap_reason": m.get("related_cap_reason"),
                        # Phase 6B.3 — per-member overrides; fall back
                        # to the group-level hint when not set.
                        "question_type_hint": (
                            m.get("question_type_hint")
                            or group_q_type_hint
                        ),
                        "question_subtype_hint": (
                            m.get("question_subtype_hint")
                            or group_q_subtype_hint
                        ),
                    }
                    self._index_term(self._pack_index, name, entry)
                    for s in m_syns:
                        self._index_term(self._pack_index, s, entry)

    def _build_rxnorm_index(self) -> None:
        rx = self._rxnorm or {}
        for cls in (rx.get("drug_class_mappings") or []):
            class_concept = cls.get("class_concept")
            cls_syns = list(cls.get("class_synonyms") or [])
            cls_conf = float(cls.get("ontology_confidence") or 0.90)
            if class_concept:
                entry = {
                    "canonical": class_concept,
                    "concept_type": "drug_class",
                    "role_hint": "intervention",
                    "synonyms": cls_syns,
                    "rxnorm_terms": [m["concept"]
                                      for m in (cls.get("members") or [])
                                      if m.get("concept")],
                    "confidence": cls_conf,
                    "source": "curated_rxnorm",
                    "class_concept": class_concept,
                    "is_glp1_ra": bool(cls.get("is_glp1_ra", True)),
                }
                # Phase 6B.3 — register morphology variants for the
                # class concept + all class synonyms so "glp1s" /
                # "GLP-1 RA" / "GLP 1 receptor agonists" all resolve.
                for v in self._morphology_variants(class_concept):
                    self._rxnorm_index.setdefault(v, (entry, class_concept))
                for s in cls_syns:
                    for v in self._morphology_variants(s):
                        self._rxnorm_index.setdefault(v, (entry, class_concept))
            for m in (cls.get("members") or []):
                name = (m.get("concept") or "").strip()
                if not name:
                    continue
                m_syns = list(m.get("synonyms") or [])
                rx_id = m.get("rxcui")
                m_conf = float(m.get("confidence") or 0.90)
                entry = {
                    "canonical": name,
                    "concept_type": "drug",
                    "role_hint": "intervention",
                    "synonyms": m_syns,
                    "rxnorm_terms": [rx_id] if rx_id else [],
                    "confidence": m_conf,
                    "source": "curated_rxnorm",
                    "class_concept": class_concept,
                    "is_glp1_ra": bool(cls.get("is_glp1_ra", True)),
                }
                for v in self._morphology_variants(name):
                    self._rxnorm_index.setdefault(v, (entry, class_concept))
                for s in m_syns:
                    for v in self._morphology_variants(s):
                        self._rxnorm_index.setdefault(v, (entry, class_concept))
        for single in (rx.get("single_drug_entries") or []):
            name = (single.get("concept") or "").strip()
            if not name:
                continue
            entry = {
                "canonical": name,
                "concept_type": "drug",
                "role_hint": "intervention",
                "synonyms": list(single.get("synonyms") or []),
                "rxnorm_terms": [single["rxcui"]] if single.get("rxcui") else [],
                "confidence": float(single.get("confidence") or 0.95),
                "source": "curated_rxnorm",
                "class_concept": single.get("class_concept"),
            }
            for v in self._morphology_variants(name):
                self._rxnorm_index.setdefault(v, (entry, single.get("class_concept")))
            for s in (single.get("synonyms") or []):
                for v in self._morphology_variants(s):
                    self._rxnorm_index.setdefault(v, (entry, single.get("class_concept")))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def map_term(
        self, term: str, question_context: dict[str, Any] | None = None,
    ) -> OntologyMapping:
        """Map ``term`` to the highest-confidence ``OntologyMapping``.

        ``question_context`` is reserved for future hinting (e.g. specialty,
        type_of_question). Today it's unused but accepted so callers can
        already pass it without breaking when we wire the hinting in.
        """
        _ = question_context  # reserved for future hinting
        original = term or ""
        norm = original.strip()
        if not norm:
            return OntologyMapping(original_term=original, rationale="empty input")
        key = norm.lower()

        # Tier A — curated MeSH descriptor (highest confidence).
        mesh = mesh_resolver.resolve(norm)
        if mesh:
            # Also try RxNorm and pack to enrich synonyms.
            extra_rx = self._rxnorm_index.get(key, (None, None))[0]
            extra_pack = self._pack_index.get(key)
            # If a specialty-pack entry exists for the same key, prefer
            # the pack's clinical canonical (e.g. "GLP-1 receptor
            # agonists") over the raw MeSH descriptor name
            # (e.g. "Glucagon-Like Peptide-1 Receptor"). The MeSH
            # descriptor still surfaces in ``mesh_terms`` for the query
            # compiler to use.
            canonical_display = (
                extra_pack["canonical"] if extra_pack
                else (extra_rx["canonical"] if extra_rx
                       else mesh.mesh_descriptor)
            )
            return self._merge_match(
                original=original, norm=norm,
                primary_source="curated_mesh",
                canonical=canonical_display,
                concept_type=self._guess_concept_type(
                    extra_pack, extra_rx, default="other",
                ),
                role_hint=self._guess_role_hint(extra_pack, extra_rx),
                mesh_terms=[mesh.mesh_descriptor],
                rxnorm_terms=list(extra_rx.get("rxnorm_terms") or []) if extra_rx else [],
                synonyms=self._collect_synonyms(extra_pack, extra_rx, norm),
                umls_cui=(extra_pack.get("umls_cui") if extra_pack else None),
                confidence=max(float(mesh.confidence), 0.85),
                rationale=(
                    f"curated MeSH descriptor '{mesh.mesh_descriptor}' "
                    f"(confidence {mesh.confidence:.2f}); "
                    "enriched with RxNorm / specialty-pack synonyms."
                ),
            )

        # Tier B — curated RxNorm class / member.
        rx = self._rxnorm_index.get(key)
        if rx:
            entry, class_concept = rx
            extra_pack = self._pack_index.get(key)
            return self._merge_match(
                original=original, norm=norm,
                primary_source="curated_rxnorm",
                canonical=entry["canonical"],
                concept_type=entry["concept_type"],
                role_hint=entry["role_hint"],
                mesh_terms=list(extra_pack.get("mesh_terms") or []) if extra_pack else [],
                rxnorm_terms=list(entry.get("rxnorm_terms") or []),
                synonyms=self._collect_synonyms(extra_pack, entry, norm),
                umls_cui=(extra_pack.get("umls_cui") if extra_pack else None),
                confidence=float(entry["confidence"]),
                rationale=(
                    f"curated RxNorm entry (class={class_concept!r}); "
                    "canonical name + class-level RxCUIs available."
                ),
                extra={
                    "class_concept": class_concept,
                    "is_glp1_ra": entry.get("is_glp1_ra"),
                },
            )

        # Tier C — specialty pack (concepts not in MeSH / RxNorm, e.g.
        # ``cardiometabolic risk``, ``LDL-C`` synonyms).
        pack = self._pack_index.get(key)
        if pack:
            return self._merge_match(
                original=original, norm=norm,
                primary_source="specialty_pack",
                canonical=pack["canonical"],
                concept_type=pack.get("concept_type", "other"),
                role_hint=pack.get("role_hint", "other"),
                mesh_terms=list(pack.get("mesh_terms") or []),
                rxnorm_terms=list(pack.get("rxnorm_terms") or []),
                synonyms=list(pack.get("synonyms") or []),
                umls_cui=pack.get("umls_cui"),
                confidence=float(pack.get("confidence", 0.85)),
                rationale=(
                    f"specialty pack {pack.get('pack_name')!r} / group "
                    f"{pack.get('group_name')!r}."
                ),
                extra={
                    "is_glp1_ra": pack.get("is_glp1_ra"),
                    "related_cap_reason": pack.get("related_cap_reason"),
                },
                # Phase 6B.3 — surface curated hints to OntologyMapping.
                question_type_hint=pack.get("question_type_hint"),
                question_subtype_hint=pack.get("question_subtype_hint"),
                pack_name=pack.get("pack_name"),
            )

        # Tier D — clinical shorthand. The shorthand normalizer already
        # rewrites the question text; here we surface the substitution
        # as a low-confidence mapping so the role-repair service can
        # still see it.
        if key in DRUG_CLASS_SHORTHAND:
            return OntologyMapping(
                original_term=original,
                normalized_term=norm,
                canonical_concept=DRUG_CLASS_SHORTHAND[key],
                concept_type="drug_class",
                role_hint="intervention",
                synonyms=[],
                confidence=0.70,
                source="clinical_shorthand",
                rationale="drug-class shorthand expansion",
            )
        if key in LAB_SHORTHAND:
            return OntologyMapping(
                original_term=original,
                normalized_term=norm,
                canonical_concept=LAB_SHORTHAND[key],
                concept_type="biomarker",
                role_hint="outcome",
                synonyms=[],
                confidence=0.70,
                source="clinical_shorthand",
                rationale="lab biomarker shorthand expansion",
            )
        if key in PROCEDURE_ACRONYMS:
            return OntologyMapping(
                original_term=original,
                normalized_term=norm,
                canonical_concept=PROCEDURE_ACRONYMS[key],
                concept_type="procedure",
                role_hint="intervention",
                synonyms=[],
                confidence=0.70,
                source="clinical_shorthand",
                rationale="procedure / disease acronym expansion",
            )
        if key in CLINICAL_SPELLING_PAIRS:
            return OntologyMapping(
                original_term=original,
                normalized_term=CLINICAL_SPELLING_PAIRS[key],
                canonical_concept="",
                concept_type="other",
                role_hint="other",
                spelling_variants=[norm],
                confidence=0.0,
                source="clinical_shorthand",
                rationale="spelling correction only (no canonical concept)",
            )

        # Tier E — acronym dictionary fallback.
        if key in ACRONYM_PAIRS:
            return OntologyMapping(
                original_term=original,
                normalized_term=norm,
                canonical_concept=norm,
                concept_type="disease",
                role_hint="condition",
                synonyms=list(ACRONYM_PAIRS[key]),
                confidence=0.65,
                source="acronym_dict",
                rationale="general acronym dictionary match",
            )
        if key in DRUG_GENERIC_TO_CLASS:
            class_list = DRUG_GENERIC_TO_CLASS[key]
            return OntologyMapping(
                original_term=original,
                normalized_term=norm,
                canonical_concept=norm,
                concept_type="drug",
                role_hint="intervention",
                synonyms=list(class_list),
                confidence=0.65,
                source="acronym_dict",
                rationale="drug generic→class fallback",
            )

        # Tier F — no confident match.
        return OntologyMapping(
            original_term=original,
            normalized_term=norm,
            canonical_concept="",
            confidence=0.0,
            source="none",
            rationale="no curated entry found",
        )

    def map_many(
        self, terms: list[str], question_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Convenience: map a list of terms; returns serialized dicts."""
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for t in terms or []:
            if not t:
                continue
            k = (t.strip() if isinstance(t, str) else "").lower()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(self.map_term(t, question_context).to_dict())
        return out

    def extract_from_text(
        self,
        text: str,
        *,
        max_terms: int = 8,
        min_confidence: float = 0.85,
    ) -> list[dict[str, Any]]:
        """Phase 5F — deterministic concept extraction from raw question
        text. Scans every n-gram (1..4 words) against the curated lookup
        indexes; returns up to ``max_terms`` distinct, high-confidence
        ontology mappings ordered by token length (longest matches win).

        Used by the context engine to synthesize ``must_have_concepts``
        when the LLM PICOTT extractor returns an empty list. Pure /
        sync. Closed-set behaviour.
        """
        if not text:
            return []
        words = [w for w in re.split(r"\s+", text) if w]
        if not words:
            return []
        candidates: list[tuple[int, str]] = []
        for n in (4, 3, 2, 1):
            if n > len(words):
                continue
            for i in range(0, len(words) - n + 1):
                chunk = " ".join(words[i:i + n]).strip(" .,;:!?")
                if not chunk:
                    continue
                candidates.append((n, chunk))
        seen_keys: set[str] = set()
        hits: list[dict[str, Any]] = []
        for _n, token in candidates:
            key = token.lower()
            if key in seen_keys:
                continue
            if len(token) < 2 or token.lower() in _EXTRACT_STOP_TOKENS:
                continue
            m = self.map_term(token)
            if m.confidence < min_confidence or not m.canonical_concept:
                continue
            seen_keys.add(key)
            hits.append(m.to_dict())
            if len(hits) >= max_terms:
                break
        return hits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _guess_concept_type(pack_entry, rx_entry, *, default: str = "other") -> str:
        if pack_entry and pack_entry.get("concept_type") in CONCEPT_TYPES:
            return pack_entry["concept_type"]
        if rx_entry and rx_entry.get("concept_type") in CONCEPT_TYPES:
            return rx_entry["concept_type"]
        return default

    @staticmethod
    def _guess_role_hint(pack_entry, rx_entry) -> str:
        if pack_entry and pack_entry.get("role_hint") in ROLE_HINTS:
            return pack_entry["role_hint"]
        if rx_entry and rx_entry.get("role_hint") in ROLE_HINTS:
            return rx_entry["role_hint"]
        return "other"

    @staticmethod
    def _collect_synonyms(pack_entry, rx_entry, norm: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for src in (pack_entry, rx_entry):
            if not src:
                continue
            for s in (src.get("synonyms") or []):
                k = (s or "").lower()
                if k and k not in seen and k != norm.lower():
                    seen.add(k)
                    out.append(s)
        # Also pull in acronym pairs (closed-set).
        for s in expand_concept(norm):
            k = s.lower()
            if k and k not in seen and k != norm.lower():
                seen.add(k)
                out.append(s)
        return out

    def _merge_match(
        self, *, original: str, norm: str, primary_source: str,
        canonical: str, concept_type: str, role_hint: str,
        mesh_terms: list[str], rxnorm_terms: list[str],
        synonyms: list[str], umls_cui: str | None,
        confidence: float, rationale: str,
        extra: dict[str, Any] | None = None,
        question_type_hint: str | None = None,
        question_subtype_hint: str | None = None,
        pack_name: str | None = None,
    ) -> OntologyMapping:
        m = OntologyMapping(
            original_term=original,
            normalized_term=norm,
            canonical_concept=canonical,
            concept_type=concept_type,
            role_hint=role_hint,
            synonyms=list(synonyms),
            mesh_terms=list(mesh_terms),
            rxnorm_terms=list(rxnorm_terms),
            umls_cui=umls_cui,
            confidence=confidence,
            source=primary_source,
            rationale=rationale,
            question_type_hint=question_type_hint,
            question_subtype_hint=question_subtype_hint,
            pack_name=pack_name,
        )
        # Carry the extra dict as a private attribute (for callers that
        # need is_glp1_ra / related_cap_reason / class_concept). The
        # to_dict() output is intentionally minimal.
        if extra:
            m.__dict__["extra"] = dict(extra)
        return m


clinical_ontology_mapper = ClinicalOntologyMapperService()


__all__ = [
    "OntologyMapping",
    "ClinicalOntologyMapperService",
    "clinical_ontology_mapper",
    "CONCEPT_TYPES",
    "ROLE_HINTS",
    "MAPPING_SOURCES",
]
