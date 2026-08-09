"""Phase 5F Group A — ClinicalShorthandNormalizer + ClinicalOntologyMapper.

Covers:
- Shorthand: glp1s → GLP-1 receptor agonists, ldl → LDL-C, a1c → HbA1c,
  cholesterole → cholesterol, mi context-disambiguation.
- Ontology mapper: GLP-1 RAs / semaglutide / liraglutide map to canonical
  GLP-1 RA class; LDL → LDL-C with MeSH descriptor; misses return
  confidence 0 with empty canonical (NEVER hallucinated).
- The pipeline cascade (PHI guard → spelling → shorthand → ontology) is
  unit-tested at the shorthand + mapper level here; the parser-level
  integration is verified in Group F (benchmark).
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.clinical_ontology_mapper import (  # noqa: E402
    clinical_ontology_mapper,
)
from services.clinical_shorthand_normalizer import (  # noqa: E402
    normalize_shorthand,
)


# =============================================================================
# Shorthand normalization
# =============================================================================

class TestShorthandNormalization:

    def test_glp1s_expands_to_glp1_receptor_agonists(self):
        r = normalize_shorthand("how do glp1s impact ldl cholesterole?")
        assert "GLP-1 receptor agonists" in r.normalized, r.normalized
        # cholesterole correction stays in normalized text.
        assert "cholesterol" in r.normalized
        # LDL canonicalized.
        joined = " | ".join(s["original"] for s in r.substitutions)
        assert "glp1s" in joined
        assert "ldl" in joined
        assert "cholesterole" in joined
        # Drug-class shorthand reason recorded.
        glp1_sub = next(s for s in r.substitutions if s["original"] == "glp1s")
        assert glp1_sub["reason"] == "drug_class_shorthand"
        assert glp1_sub["normalized"] == "GLP-1 receptor agonists"
        assert glp1_sub["confidence"] == 1.0

    def test_ldl_expands_with_full_form(self):
        r = normalize_shorthand("does empagliflozin lower ldl in patients?")
        assert "LDL-C / low-density lipoprotein cholesterol" in r.normalized
        ldl = next(s for s in r.substitutions if s["original"] == "ldl")
        assert ldl["reason"] == "lab_shorthand"

    def test_a1c_expands_to_hba1c(self):
        r = normalize_shorthand("does metformin reduce a1c after 12 weeks?")
        assert "hemoglobin A1c / HbA1c" in r.normalized
        sub = next(s for s in r.substitutions if s["original"] == "a1c")
        assert sub["reason"] == "lab_shorthand"

    def test_cholesterole_spelling_correction(self):
        r = normalize_shorthand("how do glp1s impact ldl cholesterole?")
        sub = next(s for s in r.substitutions if s["original"] == "cholesterole")
        assert sub["normalized"] == "cholesterol"
        assert sub["reason"] == "clinical_spelling_correction"

    def test_mi_context_disambiguation_skipped_for_sah(self):
        """MI should NOT expand to 'myocardial infarction' when SAH /
        subarachnoid / vasospasm / milrinone is in the same question."""
        r = normalize_shorthand("what is the dose of milrinone in SAH MI?")
        # MI substitution must NOT appear because milrinone+SAH context.
        for sub in r.substitutions:
            assert sub["original"] != "mi", (
                f"MI was wrongly expanded in SAH context: {sub!r}"
            )

    def test_mi_context_expansion_for_cardiac_question(self):
        """MI should expand for cardiology questions without neuro context."""
        r = normalize_shorthand("aspirin after MI in elderly patients?")
        mi_subs = [s for s in r.substitutions if s["original"] == "mi"]
        assert mi_subs, f"MI should expand for cardiac context; got {r.substitutions!r}"
        assert mi_subs[0]["normalized"] == "myocardial infarction"

    def test_cabg_pci_acronyms(self):
        r = normalize_shorthand("CABG vs PCI in patients with multivessel disease")
        assert "coronary artery bypass graft" in r.normalized
        assert "percutaneous coronary intervention" in r.normalized

    def test_ards_expansion(self):
        r = normalize_shorthand("prone positioning in ARDS")
        assert "acute respiratory distress syndrome" in r.normalized

    def test_nsaid_shorthand_expands(self):
        r = normalize_shorthand("nsaid use and bleeding risk")
        assert "NSAID" in r.normalized

    def test_persisted_original_question_untouched(self):
        """The persisted question is the caller's responsibility — the
        normalizer never mutates an external sequence. Test that the
        returned ``normalized`` is a *different* string when corrections
        fire."""
        orig = "glp1s and ldl"
        r = normalize_shorthand(orig)
        assert r.normalized != orig
        # No side effects on the input.
        assert orig == "glp1s and ldl"

    def test_empty_input_returns_empty(self):
        r = normalize_shorthand("")
        assert r.normalized == ""
        assert r.substitutions == []

    def test_no_shorthand_no_substitutions(self):
        r = normalize_shorthand("does aspirin reduce stroke?")
        assert r.substitutions == []
        assert r.normalized == "does aspirin reduce stroke?"


# =============================================================================
# Ontology mapping
# =============================================================================

class TestOntologyMapper_GLP1Class:
    """GLP-1 receptor agonists + members → canonical class via specialty
    pack OR curated RxNorm. MeSH descriptor present when curated."""

    def test_class_term_maps_to_canonical(self):
        m = clinical_ontology_mapper.map_term("GLP-1 receptor agonists")
        assert m.canonical_concept == "GLP-1 receptor agonists"
        assert m.concept_type == "drug_class"
        assert m.role_hint == "intervention"
        assert m.confidence >= 0.85
        # MeSH descriptor curated.
        assert any("Glucagon-Like" in t for t in m.mesh_terms), m.mesh_terms

    def test_member_semaglutide_maps_to_canonical_drug(self):
        m = clinical_ontology_mapper.map_term("semaglutide")
        assert m.canonical_concept == "Semaglutide" or m.canonical_concept == "semaglutide"
        assert m.concept_type in ("drug", "drug_class")
        assert m.role_hint == "intervention"
        # RxNorm RxCUI surfaces.
        assert any(rx == "1991302" for rx in m.rxnorm_terms), m.rxnorm_terms

    def test_member_liraglutide_maps_to_canonical(self):
        m = clinical_ontology_mapper.map_term("liraglutide")
        assert m.canonical_concept in ("Liraglutide", "liraglutide")
        assert m.role_hint == "intervention"
        assert any(rx == "475968" for rx in m.rxnorm_terms)

    def test_dulaglutide_exenatide_lixisenatide_albiglutide_all_map(self):
        for drug in ("dulaglutide", "exenatide", "lixisenatide", "albiglutide"):
            m = clinical_ontology_mapper.map_term(drug)
            assert m.confidence > 0.0, f"{drug} returned no mapping"
            assert m.role_hint == "intervention", drug
            assert m.concept_type in ("drug", "drug_class"), drug

    def test_tirzepatide_is_related_but_not_glp1_ra(self):
        m = clinical_ontology_mapper.map_term("tirzepatide")
        assert m.canonical_concept.lower() == "tirzepatide"
        # Specialty-pack carries is_glp1_ra=False on extra.
        extra = m.__dict__.get("extra") or {}
        # When loaded via curated RxNorm path, the class_concept is
        # "dual GIP/GLP-1 receptor agonists".
        if "class_concept" in extra:
            assert "GIP" in (extra["class_concept"] or "")
        # is_glp1_ra MUST NOT be True.
        assert extra.get("is_glp1_ra") is not True

    def test_glp1ra_shorthand_routes_to_class(self):
        """The shorthand normalizer rewrites ``glp1ra`` →
        ``GLP-1 receptor agonist`` before the mapper sees it; the
        mapper's standalone fallback for the raw shorthand should also
        produce a confident mapping."""
        m = clinical_ontology_mapper.map_term("glp1ra")
        # Either the shorthand-fallback path OR specialty-pack picks it up.
        assert m.canonical_concept != ""
        assert "GLP-1" in m.canonical_concept


class TestOntologyMapper_LipidOutcomes:

    def test_ldl_canonical_lipid_outcome(self):
        m = clinical_ontology_mapper.map_term("LDL-C")
        # Mapper resolves via MeSH ('Cholesterol, LDL') OR pack canonical.
        assert m.confidence > 0.0
        assert m.role_hint == "outcome"
        assert "Cholesterol, LDL" in m.mesh_terms or m.canonical_concept == "LDL-C"

    def test_low_density_lipoprotein_cholesterol_maps(self):
        m = clinical_ontology_mapper.map_term("low-density lipoprotein cholesterol")
        assert m.confidence > 0.0
        assert any("LDL" in t or "Cholesterol" in t
                   for t in (m.mesh_terms + [m.canonical_concept]))

    def test_triglycerides_maps_to_canonical(self):
        m = clinical_ontology_mapper.map_term("triglycerides")
        assert m.confidence > 0.0
        assert m.role_hint == "outcome"
        assert "Triglycerides" in m.mesh_terms or m.canonical_concept.lower() == "triglycerides"

    def test_apob_maps_to_apolipoprotein_b(self):
        m = clinical_ontology_mapper.map_term("apolipoprotein B")
        assert m.confidence > 0.0
        assert m.role_hint == "outcome"

    def test_dyslipidemia_maps_to_canonical(self):
        m = clinical_ontology_mapper.map_term("dyslipidemia")
        assert m.confidence > 0.0
        assert "Dyslipidemias" in m.mesh_terms or m.canonical_concept.lower() == "dyslipidemia"


class TestOntologyMapper_Populations:

    def test_t2dm_maps_to_type2_diabetes(self):
        m = clinical_ontology_mapper.map_term("type 2 diabetes")
        assert m.confidence > 0.0
        assert m.role_hint == "population"
        assert "Diabetes Mellitus, Type 2" in m.mesh_terms or m.canonical_concept.lower() == "type 2 diabetes"

    def test_obesity_maps_with_role_population(self):
        m = clinical_ontology_mapper.map_term("obesity")
        assert m.role_hint == "population"
        assert m.confidence > 0.0

    def test_atherosclerosis_maps_with_role_population(self):
        m = clinical_ontology_mapper.map_term("atherosclerosis")
        assert m.role_hint == "population"


class TestOntologyMapper_NoHallucination:
    """Curated only: a concept NOT in MeSH / RxNorm / packs / acronyms
    MUST return confidence=0.0 with an empty canonical."""

    def test_unknown_concept_returns_zero_confidence(self):
        m = clinical_ontology_mapper.map_term("blarflarx-9 receptor antagonist")
        assert m.confidence == 0.0
        assert m.canonical_concept == ""
        assert m.source == "none"

    def test_misspelled_concept_returns_zero_confidence(self):
        # Phase 6B.3 — the morphology layer now resolves the common
        # ``cholesterole`` typo to the curated ``cholesterol`` /
        # ``total cholesterol`` pack entry. This is a curated stem rule,
        # NOT broad fuzzy matching (drug names without an explicit
        # spelling pair still fail closed). Update the assertion to
        # confirm a high-confidence pack hit AND record that the source
        # is the cardiometabolic pack (not ``none``).
        m = clinical_ontology_mapper.map_term("cholesterole")
        assert m.source == "specialty_pack"
        assert (m.canonical_concept or "").lower() in (
            "cholesterol", "total cholesterol",
        )
        # Random unrelated misspellings still return zero confidence.
        m2 = clinical_ontology_mapper.map_term("blarflarx receptor xygblr")
        assert m2.confidence == 0.0
        assert m2.canonical_concept == ""

    def test_empty_term_returns_zero(self):
        m = clinical_ontology_mapper.map_term("")
        assert m.confidence == 0.0
        assert m.canonical_concept == ""

    def test_mesh_threshold_respected(self):
        """A concept whose MeSH confidence < 0.85 returns no MeSH
        descriptor (curated keyword_only_concepts return None)."""
        m = clinical_ontology_mapper.map_term("liberal vs restrictive transfusion strategy")
        # No MeSH match: confidence stays low or falls through to none.
        assert "Blood Transfusion" not in m.mesh_terms


class TestOntologyMapper_ApiContract:
    """The OntologyMapping output shape is closed-set — frontend relies
    on it."""

    _required_keys = {
        "original_term", "normalized_term", "canonical_concept",
        "concept_type", "role_hint", "synonyms", "acronyms",
        "spelling_variants", "mesh_terms", "rxnorm_terms",
        "umls_cui", "confidence", "source", "rationale",
    }

    def test_glp1_response_has_all_keys(self):
        m = clinical_ontology_mapper.map_term("GLP-1 receptor agonists").to_dict()
        for k in self._required_keys:
            assert k in m, f"missing key {k}"

    def test_unknown_response_has_all_keys(self):
        m = clinical_ontology_mapper.map_term("zzz unknown").to_dict()
        for k in self._required_keys:
            assert k in m, f"missing key {k}"

    def test_concept_type_in_closed_set(self):
        from services.clinical_ontology_mapper import CONCEPT_TYPES
        for term in ("semaglutide", "LDL-C", "type 2 diabetes",
                      "cabg", "unknown-foo"):
            m = clinical_ontology_mapper.map_term(term)
            assert m.concept_type in CONCEPT_TYPES, (term, m.concept_type)

    def test_role_hint_in_closed_set(self):
        from services.clinical_ontology_mapper import ROLE_HINTS
        for term in ("semaglutide", "LDL-C", "obesity"):
            m = clinical_ontology_mapper.map_term(term)
            assert m.role_hint in ROLE_HINTS, (term, m.role_hint)

    def test_source_in_closed_set(self):
        from services.clinical_ontology_mapper import MAPPING_SOURCES
        for term in ("semaglutide", "LDL-C", "unknown-foo", ""):
            m = clinical_ontology_mapper.map_term(term)
            assert m.source in MAPPING_SOURCES, (term, m.source)


class TestOntologyMapper_MapMany:

    def test_map_many_dedupes_and_returns_dicts(self):
        out = clinical_ontology_mapper.map_many([
            "GLP-1 receptor agonists", "glp-1 receptor agonists",  # dedupe
            "LDL-C", "type 2 diabetes",
        ])
        assert isinstance(out, list)
        assert all(isinstance(o, dict) for o in out)
        # Dedupe by lower-case key.
        canonicals = [o["canonical_concept"].lower() for o in out]
        assert len(canonicals) == len(set(canonicals)) or any(
            c == "" for c in canonicals
        )

    def test_map_many_ignores_empty(self):
        out = clinical_ontology_mapper.map_many(["", None, "  ", "semaglutide"])
        # Only the semaglutide term survives.
        assert len(out) == 1
        assert out[0]["canonical_concept"].lower() == "semaglutide"
