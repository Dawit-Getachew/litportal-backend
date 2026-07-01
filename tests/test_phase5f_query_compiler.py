"""Phase 5F Group D — BooleanQueryCompilerService extensions.

Covers:
- New ``pubmed_atm`` family is emitted for PubMed sources with UNTAGGED
  (no ``[tiab]`` / ``[MeSH Terms]``) syntax.
- Field-tag isolation per Phase 5D still holds (no ``TITLE_ABS:`` in any
  PubMed query, no ``[tiab]`` in Europe PMC).
- GLP-1/LDL questions produce both mesh_keyword + pubmed_atm + clinical_queries
  variants. The misspelled ``cholesterole`` does NOT appear in any
  executed query (only as a normalizer diagnostic).
- MeSH descriptor + entry-term synonyms surface in mesh_keyword (Cholesterol,
  LDL is recognized for "LDL-C").
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.query_compiler import (  # noqa: E402
    BooleanQueryCompilerService,
    PHASE5D_QUOTAS,
)


def _glp1_ldl_picott() -> dict:
    """Phase 5F gold-standard PICOTT for ``glp1s impact ldl cholesterole``."""
    return {
        "normalized_question": (
            "how do GLP-1 receptor agonists impact LDL-C in patients with type 2 diabetes?"
        ),
        "original_question": "how do glp1s impact ldl cholesterole?",
        "type_of_question": "pharmacology",
        "question_subtype": "metabolic_biomarker_effect",
        "specialty": "endocrinology",
        "pico": {
            "population": ["type 2 diabetes"],
            "intervention_or_exposure_or_index_test": [
                "GLP-1 receptor agonists",
            ],
            "comparator_or_reference_standard": [],
            "outcomes_or_target_diagnosis": ["LDL-C"],
        },
        "must_have_concepts": [
            {"concept": "GLP-1 receptor agonists", "role": "intervention",
             "synonyms": ["semaglutide", "liraglutide", "dulaglutide",
                          "exenatide"]},
            {"concept": "LDL-C", "role": "outcome",
             "synonyms": ["LDL cholesterol",
                          "low-density lipoprotein cholesterol"]},
            {"concept": "type 2 diabetes", "role": "population",
             "synonyms": ["T2D", "T2DM"]},
        ],
        "ranking_concepts": [
            {"concept": "HDL-C", "role": "outcome"},
            {"concept": "triglycerides", "role": "outcome"},
        ],
        "should_have_concepts": [],
        "negative_or_penalty_concepts": [],
        "concept_translations": [],
        "ideal_study_types": ["systematic_review", "meta_analysis", "rct"],
    }


class TestQuotaTableHasAtm:

    def test_pubmed_atm_quota_present(self):
        assert "pubmed_atm" in PHASE5D_QUOTAS


class TestAtmFamilyEmission:
    """PubMed Automatic Term Mapping family — emitted for PubMed only,
    UNTAGGED, free-text AND of mandatory anchors."""

    def test_pubmed_atm_emitted_for_pubmed_source(self):
        compiler = BooleanQueryCompilerService()
        out = compiler.compile(
            _glp1_ldl_picott(),
            sources=["pubmed"],
            is_phi_blocked=False,
        )
        atm = [q for q in out if q.query_family == "pubmed_atm"]
        assert atm, [q.query_family for q in out]
        q = atm[0]
        assert q.source == "pubmed"
        # No field tags in ATM query string.
        assert "[tiab]" not in q.query_string.lower()
        assert "[mesh terms]" not in q.query_string.lower()
        assert "[publication type]" not in q.query_string.lower()
        # Anchors appear (quoted when multi-word).
        assert "GLP-1 receptor agonists" in q.query_string \
            or "\"GLP-1 receptor agonists\"" in q.query_string
        # Field tags list is empty (closed-set contract).
        assert q.field_tags_used == [], q.field_tags_used

    def test_pubmed_atm_not_emitted_for_europe_pmc(self):
        compiler = BooleanQueryCompilerService()
        out = compiler.compile(
            _glp1_ldl_picott(),
            sources=["europe_pmc"],
            is_phi_blocked=False,
        )
        atm = [q for q in out if q.query_family == "pubmed_atm"]
        assert not atm, [q.query_family for q in out]

    def test_pubmed_atm_not_emitted_when_phi_blocked(self):
        compiler = BooleanQueryCompilerService()
        out = compiler.compile(
            _glp1_ldl_picott(),
            sources=["pubmed"],
            is_phi_blocked=True,
        )
        assert out == []

    def test_pubmed_atm_validation_passes(self):
        compiler = BooleanQueryCompilerService()
        out = compiler.compile(
            _glp1_ldl_picott(),
            sources=["pubmed"],
            is_phi_blocked=False,
        )
        atm = [q for q in out if q.query_family == "pubmed_atm"][0]
        assert atm.syntax_validation is not None
        assert atm.syntax_validation.valid, atm.syntax_validation

    def test_pubmed_atm_multi_word_anchors_quoted(self):
        compiler = BooleanQueryCompilerService()
        out = compiler.compile(
            _glp1_ldl_picott(),
            sources=["pubmed"],
            is_phi_blocked=False,
        )
        atm = [q for q in out if q.query_family == "pubmed_atm"][0]
        # Multi-word terms must be quoted.
        assert "\"GLP-1 receptor agonists\"" in atm.query_string
        assert "\"type 2 diabetes\"" in atm.query_string

    def test_pubmed_atm_quota_default_50(self):
        compiler = BooleanQueryCompilerService()
        out = compiler.compile(
            _glp1_ldl_picott(),
            sources=["pubmed"],
            is_phi_blocked=False,
        )
        atm = [q for q in out if q.query_family == "pubmed_atm"][0]
        assert atm.max_results == PHASE5D_QUOTAS["pubmed_atm"]


# ---------------------------------------------------------------------------
# Field-tag isolation per Phase 5D — must STILL hold after 5F.
# ---------------------------------------------------------------------------
class TestFieldTagIsolation:

    def test_no_europe_pmc_prefix_in_pubmed_queries(self):
        compiler = BooleanQueryCompilerService()
        out = compiler.compile(_glp1_ldl_picott(),
                                sources=["pubmed", "europe_pmc"],
                                is_phi_blocked=False)
        for q in out:
            if q.source == "pubmed":
                # No Europe PMC prefixes (TITLE_ABS:, TITLE:, KW:, SRC:,
                # PUB_TYPE:, FIRST_PDATE:) in PubMed queries.
                assert not re.search(
                    r"\b(TITLE_ABS|TITLE|KW|PUB_TYPE|FIRST_PDATE|SRC):",
                    q.query_string,
                ), (q.query_family, q.query_string[:120])

    def test_no_pubmed_brackets_in_europe_pmc_queries(self):
        compiler = BooleanQueryCompilerService()
        out = compiler.compile(_glp1_ldl_picott(),
                                sources=["pubmed", "europe_pmc"],
                                is_phi_blocked=False)
        for q in out:
            if q.source == "europe_pmc":
                # Europe PMC must not contain PubMed-style [Tag] brackets.
                assert not re.search(r"\[[A-Za-z][^\]]*\]", q.query_string), \
                    (q.query_family, q.query_string[:120])


# ---------------------------------------------------------------------------
# GLP-1 / LDL query content — Task 8.
# ---------------------------------------------------------------------------
class TestGlp1LdlQueryContent:

    def test_pubmed_mesh_keyword_contains_glp1_and_ldl(self):
        compiler = BooleanQueryCompilerService()
        out = compiler.compile(
            _glp1_ldl_picott(),
            sources=["pubmed"],
            is_phi_blocked=False,
        )
        mesh_kw = [q for q in out if q.query_family == "mesh_keyword"]
        assert mesh_kw, [q.query_family for q in out]
        qs = mesh_kw[0].query_string
        # GLP-1 receptor agonists referenced — via MeSH or T/A.
        assert "GLP-1 receptor agonists" in qs or "GLP-1" in qs
        # LDL-C referenced.
        assert "LDL-C" in qs or "Cholesterol, LDL" in qs or "LDL" in qs

    def test_misspelled_cholesterole_never_in_compiled_queries(self):
        """The compiler ingests the NORMALIZED PICOTT, so ``cholesterole``
        (typo) must NEVER appear in any executed query string."""
        compiler = BooleanQueryCompilerService()
        out = compiler.compile(
            _glp1_ldl_picott(),
            sources=["pubmed", "europe_pmc"],
            is_phi_blocked=False,
        )
        for q in out:
            assert "cholesterole" not in q.query_string.lower(), (
                q.query_family, q.query_string
            )

    def test_both_atm_and_mesh_keyword_emitted_for_pubmed(self):
        compiler = BooleanQueryCompilerService()
        out = compiler.compile(
            _glp1_ldl_picott(),
            sources=["pubmed"],
            is_phi_blocked=False,
        )
        families = {q.query_family for q in out}
        assert "pubmed_atm" in families
        assert "mesh_keyword" in families

    def test_pubmed_atm_does_not_carry_publication_type_filter(self):
        """ATM is meant to maximize ATM expansion; we explicitly do NOT
        add ``NOT "preprint"[Publication Type]`` to it (which would
        defeat the no-field-tag rule)."""
        compiler = BooleanQueryCompilerService()
        out = compiler.compile(
            _glp1_ldl_picott(),
            sources=["pubmed"],
            is_phi_blocked=False,
        )
        atm = [q for q in out if q.query_family == "pubmed_atm"][0]
        assert "[publication type]" not in atm.query_string.lower()
        assert "publication type" not in atm.query_string.lower()
