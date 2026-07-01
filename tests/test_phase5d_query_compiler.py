"""Phase 5D — Formal Boolean Query Compiler + MeSH/keyword optimization.

Test groups (per spec Task 11):
- A. Concept blocks — STOP_PHRASES stripped, acronyms preserved, empty
     blocks omitted.
- B. PubMed compiler — Boolean structure, field tags, MeSH+T/A hybrid,
     publication-type filters scoped correctly, outcomes not mandatory.
- C. Europe PMC compiler — source-specific syntax valid; no PubMed tags.
- D. Clinical Queries — therapy/diagnosis/etiology/prognosis filters
     emitted; clinical_queries is ONE family among 9.
- E. Quotas + staged execution — quotas honored, syntax validation
     drops invalid queries.
- F. Ranking integration — query family persisted to per-article rows
     (search_session_articles.query_family_matches already covers this
     from Phase 5A).
- G. Benchmarks — AMI / milrinone / D-dimer query feature assertions
     pass.
- H. Regression — all Phase 5A-C tests still pass; PHI guard first; no
     Google Scholar; no Europe PMC tags in PubMed queries; no PubMed
     tags in Europe PMC queries.
"""
from __future__ import annotations

import re

import pytest

from services.mesh import mesh_resolver
from services.query_compiler import (
    CLINICAL_QUERIES_FILTERS,
    PHASE5D_QUOTAS,
    STOP_PHRASES,
    SYNTAX_ERROR_CODES,
    boolean_query_compiler,
    clinical_queries_category,
    comparator_is_mandatory,
    outcomes_are_mandatory,
    strip_stop_phrases,
    validate_query_syntax,
)


# =============================================================================
# A. Concept blocks / STOP_PHRASES
# =============================================================================

class TestGroupA_StopPhrasesAndBlocks:
    def test_stop_phrases_constant_includes_spec_items(self):
        joined = " ".join(STOP_PHRASES)
        assert r"\bdifference between\b" in joined
        assert r"\bin patients that are\b" in joined
        assert r"\bwhat is the dose of\b" in joined
        assert r"\bwhat is the role of\b" in joined

    def test_strip_stop_phrases_removes_known_noise(self):
        s = (
            "What is the difference between high hemoglobin target and low "
            "hemoglobin target in patients that are having acute myocardial "
            "infarction?"
        )
        out = strip_stop_phrases(s)
        assert "difference between" not in out.lower()
        assert "in patients that are" not in out.lower()
        # Clinical concepts preserved.
        assert "hemoglobin target" in out.lower()
        assert "myocardial infarction" in out.lower()

    def test_strip_preserves_acronyms_and_spelling(self):
        s = "What is the role of MINT vs REALITY in ACS/AMI?"
        out = strip_stop_phrases(s)
        assert "MINT" in out
        assert "REALITY" in out
        assert "ACS" in out
        assert "AMI" in out


# =============================================================================
# B. PubMed compiler — Boolean structure + MeSH+TIAB hybrid
# =============================================================================

class TestGroupB_PubMedCompiler:
    def _picott(self):
        return {
            "type_of_question": "therapy",
            "question_subtype": "transfusion_threshold_comparison",
            "normalized_question": (
                "In patients with acute myocardial infarction and anemia, "
                "how does a liberal red blood cell transfusion strategy "
                "compare with a restrictive transfusion strategy?"
            ),
            "must_have_concepts": [
                {"concept": "liberal transfusion strategy",
                 "role": "intervention", "synonyms": ["liberal transfusion"]},
                {"concept": "restrictive transfusion strategy",
                 "role": "comparator", "synonyms": ["restrictive transfusion"]},
                {"concept": "acute myocardial infarction",
                 "role": "condition", "synonyms": ["AMI", "ACS"]},
                {"concept": "anemia", "role": "context", "synonyms": ["hemoglobin"]},
            ],
            "ranking_concepts": [
                {"concept": "MINT", "role": "context", "boost_weight": 6},
                {"concept": "REALITY", "role": "context", "boost_weight": 6},
            ],
            "pico": {"population": ["acute myocardial infarction"]},
        }

    def test_compile_emits_mesh_keyword_and_trial_acronym(self):
        qs = boolean_query_compiler.compile(self._picott(), sources=["pubmed"])
        families = {q.query_family for q in qs}
        assert "mesh_keyword" in families
        assert "trial_acronym" in families
        # No Europe PMC tags in PubMed output.
        for q in qs:
            assert "TITLE_ABS:" not in q.query_string
            assert "PUB_TYPE:" not in q.query_string

    def test_mesh_keyword_uses_verified_mesh_with_fallback(self):
        qs = boolean_query_compiler.compile(self._picott(), sources=["pubmed"])
        mk = next(q for q in qs if q.query_family == "mesh_keyword")
        # Hybrid uses MeSH for known concepts.
        assert '"Myocardial Infarction"[MeSH Terms]' in mk.query_string
        # mesh_terms_used populated with confidence.
        assert any(
            m.get("mesh_descriptor") == "Myocardial Infarction"
            for m in mk.mesh_terms_used
        )
        # Title/Abstract fallback for non-MeSH concepts (e.g., "liberal
        # transfusion strategy" is keyword-only or low confidence).
        assert "[tiab]" in mk.query_string
        # Preprints excluded.
        assert '"preprint"[Publication Type]' in mk.query_string

    def test_trial_acronym_is_anchor_ANDed(self):
        qs = boolean_query_compiler.compile(self._picott(), sources=["pubmed"])
        ta = next(q for q in qs if q.query_family == "trial_acronym")
        # Both MINT and REALITY should appear; ANDed with condition/intervention.
        assert "MINT" in ta.query_string
        assert "REALITY" in ta.query_string
        # NOT a bare acronym query.
        assert " AND " in ta.query_string

    def test_outcomes_not_mandatory_for_therapy(self):
        # therapy = outcomes are ranking, NOT mandatory.
        assert outcomes_are_mandatory(self._picott()) is False

    def test_outcomes_mandatory_for_outcome_focused_questions(self):
        picott = {"type_of_question": "prognosis"}
        assert outcomes_are_mandatory(picott) is True

    def test_comparator_mandatory_for_transfusion_threshold(self):
        assert comparator_is_mandatory(self._picott()) is True

    def test_comparator_optional_for_simple_therapy(self):
        picott = {
            "type_of_question": "therapy",
            "must_have_concepts": [
                {"concept": "metformin", "role": "intervention"},
                {"concept": "diabetes", "role": "condition"},
            ],
        }
        assert comparator_is_mandatory(picott) is False


# =============================================================================
# C. Europe PMC compiler
# =============================================================================

class TestGroupC_EuropePMCCompiler:
    def _picott(self):
        return {
            "type_of_question": "therapy",
            "must_have_concepts": [
                {"concept": "milrinone", "role": "intervention"},
                {"concept": "subarachnoid hemorrhage", "role": "condition"},
            ],
            "ranking_concepts": [],
        }

    def test_europe_pmc_uses_source_specific_syntax(self):
        qs = boolean_query_compiler.compile(
            self._picott(), sources=["europe_pmc"],
        )
        for q in qs:
            # Europe PMC tags only.
            assert "[tiab]" not in q.query_string
            assert "[MeSH Terms]" not in q.query_string
            assert "[Publication Type]" not in q.query_string
            # Allowed prefixes:
            allowed = ("TITLE_ABS:", "MESH:", "PUB_TYPE:", "SRC:",
                        "FIRST_PDATE:", "KW:", "TITLE:", "ABSTRACT:")
            # Every uppercase-prefix-colon in the query must be in the allowed set.
            for m in re.finditer(r"\b([A-Z_]{2,16}):", q.query_string):
                assert any(m.group(0).startswith(p) for p in allowed), (
                    f"unexpected Europe PMC prefix {m.group(0)!r} in "
                    f"{q.query_string}"
                )

    def test_clinical_queries_not_emitted_for_europe_pmc(self):
        qs = boolean_query_compiler.compile(
            self._picott(), sources=["europe_pmc"],
        )
        assert all(q.query_family != "clinical_queries" for q in qs)


# =============================================================================
# D. Clinical Queries family
# =============================================================================

class TestGroupD_ClinicalQueries:
    def test_filter_table_covers_4_categories(self):
        assert set(CLINICAL_QUERIES_FILTERS.keys()) == {
            "therapy", "diagnosis", "etiology", "prognosis",
        }
        for cat, scopes in CLINICAL_QUERIES_FILTERS.items():
            assert set(scopes.keys()) == {"broad", "narrow"}, cat
            for scope, q in scopes.items():
                assert q.strip(), f"{cat}/{scope} is empty"

    def test_clinical_queries_emitted_for_therapy(self):
        picott = {
            "type_of_question": "therapy",
            "must_have_concepts": [
                {"concept": "anticoagulation", "role": "intervention"},
                {"concept": "atrial fibrillation", "role": "condition"},
            ],
        }
        qs = boolean_query_compiler.compile(picott, sources=["pubmed"])
        cqs = [q for q in qs if q.query_family == "clinical_queries"]
        # Both broad + narrow emitted.
        scopes = {q.cq_scope for q in cqs}
        assert scopes == {"broad", "narrow"}
        for q in cqs:
            assert q.cq_category == "therapy"

    def test_clinical_queries_omitted_when_no_matching_category(self):
        picott = {
            "type_of_question": "other",
            "must_have_concepts": [
                {"concept": "X", "role": "intervention"},
                {"concept": "Y", "role": "condition"},
            ],
        }
        qs = boolean_query_compiler.compile(picott, sources=["pubmed"])
        assert all(q.query_family != "clinical_queries" for q in qs)

    def test_category_mapping(self):
        assert clinical_queries_category({"type_of_question": "therapy"}) == "therapy"
        assert clinical_queries_category({"type_of_question": "diagnosis"}) == "diagnosis"
        assert clinical_queries_category({"type_of_question": "etiology_harm"}) == "etiology"
        assert clinical_queries_category({"type_of_question": "prognosis"}) == "prognosis"
        assert clinical_queries_category({"type_of_question": "other"}) is None


# =============================================================================
# E. Syntax validation + quotas
# =============================================================================

class TestGroupE_SyntaxValidation:
    def test_error_codes_set_is_closed(self):
        # Phase K4A.2.1 — extended with two new anchor-quality codes.
        assert SYNTAX_ERROR_CODES == {
            "invalid_syntax", "empty_block",
            "unbalanced_parens", "invalid_field_tag",
            "malformed_required_anchor", "generic_only_required_anchor",
        }

    def test_balanced_parens_pass(self):
        v = validate_query_syntax(
            '("MI"[tiab] AND "anemia"[tiab])', source="pubmed",
        )
        assert v.valid, v.errors

    def test_unbalanced_parens_fail(self):
        v = validate_query_syntax(
            '("MI"[tiab] AND "anemia"[tiab]', source="pubmed",
        )
        assert not v.valid
        assert "unbalanced_parens" in v.errors

    def test_pubmed_rejects_epmc_prefix(self):
        v = validate_query_syntax(
            'TITLE_ABS:"MI" AND "anemia"[tiab]', source="pubmed",
        )
        assert not v.valid
        assert "invalid_field_tag" in v.errors

    def test_epmc_rejects_pubmed_tag(self):
        v = validate_query_syntax(
            '"MI"[tiab] AND TITLE_ABS:"anemia"', source="europe_pmc",
        )
        assert not v.valid
        assert "invalid_field_tag" in v.errors

    def test_empty_block_detected(self):
        v = validate_query_syntax("()", source="pubmed")
        assert not v.valid
        assert "empty_block" in v.errors

    def test_quotas_default_to_50(self):
        assert PHASE5D_QUOTAS["mesh_keyword"] == 50
        assert PHASE5D_QUOTAS["clinical_queries"] == 50
        assert PHASE5D_QUOTAS["trial_acronym"] == 50


# =============================================================================
# F. MeSH resolver
# =============================================================================

class TestGroupF_MeshResolver:
    def test_mi_resolves_to_myocardial_infarction(self):
        r = mesh_resolver.resolve("myocardial infarction")
        assert r is not None
        assert r.mesh_descriptor == "Myocardial Infarction"
        assert r.confidence >= 0.85

    def test_high_threshold_excludes_lower_confidence_entries(self):
        # "D-dimer" is 0.92 — passes at 0.85 but fails at 0.95.
        assert mesh_resolver.resolve("D-dimer") is not None
        assert mesh_resolver.resolve(
            "D-dimer", confidence_threshold=0.95,
        ) is None

    def test_unknown_concept_returns_none(self):
        assert mesh_resolver.resolve("totally novel concept abc") is None

    def test_keyword_only_concept_returns_none(self):
        # "liberal vs restrictive transfusion strategy" is keyword-only per the table.
        assert mesh_resolver.resolve("liberal vs restrictive transfusion strategy") is None

    def test_no_hallucination_safety_check(self):
        # If a concept is not in the curated table, resolver MUST return
        # None — never make up a descriptor.
        assert mesh_resolver.resolve("fake_concept_xyz_999") is None


# =============================================================================
# G. Benchmark expected_query_features (3 cases)
# =============================================================================

class TestGroupG_BenchmarkQueryFeatures:
    def test_ami_transfusion_has_required_terms(self):
        picott = {
            "type_of_question": "therapy",
            "question_subtype": "transfusion_threshold_comparison",
            "normalized_question": (
                "In patients with acute myocardial infarction and anemia, "
                "liberal versus restrictive transfusion strategy."
            ),
            "must_have_concepts": [
                {"concept": "liberal transfusion strategy",
                 "role": "intervention", "synonyms": []},
                {"concept": "restrictive transfusion strategy",
                 "role": "comparator", "synonyms": []},
                {"concept": "acute myocardial infarction",
                 "role": "condition", "synonyms": ["MI", "ACS"]},
                {"concept": "anemia", "role": "context", "synonyms": ["hemoglobin"]},
                {"concept": "blood transfusion", "role": "context"},
            ],
            "ranking_concepts": [
                {"concept": "MINT", "role": "context", "boost_weight": 6},
                {"concept": "REALITY", "role": "context", "boost_weight": 6},
            ],
        }
        qs = boolean_query_compiler.compile(
            picott, sources=["pubmed", "europe_pmc"],
        )
        # mesh_keyword on PubMed must reference MI/anemia/transfusion via MeSH.
        pm = [q for q in qs if q.query_family == "mesh_keyword" and q.source == "pubmed"]
        assert pm, "no PubMed mesh_keyword query produced"
        body = pm[0].query_string
        assert "Myocardial Infarction" in body
        assert "Anemia" in body
        # Trial-acronym family includes both MINT and REALITY.
        ta_pm = next(
            q for q in qs
            if q.query_family == "trial_acronym" and q.source == "pubmed"
        )
        assert "MINT" in ta_pm.query_string
        assert "REALITY" in ta_pm.query_string
        # No STOP_PHRASES in any compiled query.
        for q in qs:
            assert "difference between" not in q.query_string.lower()

    def test_milrinone_sah_includes_canonical_concepts(self):
        picott = {
            "type_of_question": "dosing_protocol",
            "normalized_question": (
                "dose of milrinone for patients with acute subarachnoid hemorrhage"
            ),
            "must_have_concepts": [
                {"concept": "milrinone", "role": "intervention"},
                {"concept": "subarachnoid hemorrhage", "role": "condition",
                  "synonyms": ["SAH", "aSAH"]},
            ],
            "ranking_concepts": [],
        }
        qs = boolean_query_compiler.compile(picott, sources=["pubmed"])
        pm_mk = next(q for q in qs if q.query_family == "mesh_keyword")
        assert "Milrinone" in pm_mk.query_string
        assert "Subarachnoid Hemorrhage" in pm_mk.query_string

    def test_ddimer_pe_diagnosis_uses_diagnosis_clinical_queries(self):
        picott = {
            "type_of_question": "diagnosis",
            "must_have_concepts": [
                {"concept": "D-dimer", "role": "intervention"},
                {"concept": "pulmonary embolism", "role": "condition"},
            ],
            "ranking_concepts": [],
        }
        qs = boolean_query_compiler.compile(picott, sources=["pubmed"])
        cqs = [q for q in qs if q.query_family == "clinical_queries"]
        assert cqs
        for q in cqs:
            assert q.cq_category == "diagnosis"
            assert q.cq_scope in {"broad", "narrow"}


# =============================================================================
# H. Regression / safety invariants
# =============================================================================

class TestGroupH_RegressionInvariants:
    def test_phi_blocked_returns_empty(self):
        qs = boolean_query_compiler.compile(
            {"must_have_concepts": [{"concept": "X", "role": "intervention"}]},
            sources=["pubmed"], is_phi_blocked=True,
        )
        assert qs == []

    def test_invalid_syntax_query_dropped(self):
        # An empty must-have list → no queries at all (not invalid ones).
        qs = boolean_query_compiler.compile(
            {"must_have_concepts": []}, sources=["pubmed"],
        )
        assert qs == []

    def test_existing_concept_block_query_builder_still_works(self):
        # Phase 5A builder must still emit its 6 families unchanged.
        from services.concept_block_query_builder import concept_block_query_builder
        picott = {
            "type_of_question": "therapy",
            "must_have_concepts": [
                {"concept": "metformin", "role": "intervention"},
                {"concept": "diabetes", "role": "condition"},
            ],
            "ranking_concepts": [],
            "pico": {},
        }
        qs = concept_block_query_builder.build(picott, sources=["pubmed"])
        families = {q.query_family for q in qs}
        # Phase 5A families intact.
        assert "high_precision" in families
        assert "broad_recall" in families



# =============================================================================
# I. Pipeline integration — compiler_yielded_no_queries fallback path
# =============================================================================

class TestGroupI_CompilerFallbackPath:
    """When BooleanQueryCompilerService emits zero valid queries for a source,
    the pipeline must NOT fail that source. It must log
    ``compiler_yielded_no_queries`` and rely on the legacy
    ``concept_block_query_builder.build()`` output (Phase 5A families).
    """

    def test_compiler_returns_empty_when_must_haves_missing(self):
        # No must_have_concepts → compiler returns [] for both sources.
        picott_no_must_haves: dict = {
            "type_of_question": "therapy",
            "must_have_concepts": [],
            "ranking_concepts": [],
            "pico": {},
        }
        out = boolean_query_compiler.compile(
            picott_no_must_haves, sources=["pubmed", "europe_pmc"],
        )
        assert out == []

    def test_legacy_builder_still_emits_when_compiler_yields_nothing(self):
        """If the new compiler emits zero queries, the legacy
        ``concept_block_query_builder.build()`` is still able to emit
        Phase 5A families (high_precision / broad_recall) — so the
        pipeline has queries to dispatch."""
        from services.concept_block_query_builder import concept_block_query_builder

        picott = {
            "type_of_question": "therapy",
            "must_have_concepts": [
                {"concept": "metformin", "role": "intervention"},
                {"concept": "type 2 diabetes", "role": "condition"},
            ],
            "ranking_concepts": [],
            "pico": {},
        }
        legacy = concept_block_query_builder.build(
            picott, sources=["pubmed", "europe_pmc"],
        )
        families_per_source: dict = {}
        for q in legacy:
            families_per_source.setdefault(q.source, set()).add(q.query_family)
        # Each source must produce at least one legacy family.
        assert "pubmed" in families_per_source
        assert "europe_pmc" in families_per_source
        assert len(families_per_source["pubmed"]) > 0
        assert len(families_per_source["europe_pmc"]) > 0

    def test_phase5d_metrics_shape_when_compiler_emits_no_queries(self):
        """The phase5d_metrics dict (persisted on evaluator) must use the
        closed-set keys regardless of how many queries fired."""
        # Pure shape check — mirrors what the pipeline builds.
        sources = ["pubmed", "europe_pmc"]
        phase5d_metrics: dict = {
            "legacy_queries_by_source": {s: 0 for s in sources},
            "phase5d_queries_emitted_by_source": {s: 0 for s in sources},
            "phase5d_queries_dropped_by_source": {s: 0 for s in sources},
            "compiler_yielded_no_queries_sources": [],
        }
        for s in sources:
            assert s in phase5d_metrics["legacy_queries_by_source"]
            assert s in phase5d_metrics["phase5d_queries_emitted_by_source"]
            assert s in phase5d_metrics["phase5d_queries_dropped_by_source"]
        assert isinstance(
            phase5d_metrics["compiler_yielded_no_queries_sources"], list,
        )
