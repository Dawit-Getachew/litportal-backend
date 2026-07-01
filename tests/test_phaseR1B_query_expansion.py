"""Phase R1B — query expansion focused unit tests.

Pure tests against ``services/r1b_query_expansion``. No I/O.
No LLM. No connector hits.
"""
from __future__ import annotations

import pytest

from services.r1b_query_expansion import (
    CASE_ID_TO_EXPANSION_MAP,
    CURATED_EXPANSION_MAPS,
    GENERIC_PROCEDURE_ANCHOR_VOCAB,
    MAX_QUERY_STRING_LEN,
    MECHANISM_CONCEPT_TERMS,
    PROCEDURE_COMPLICATION_OUTCOME_VOCAB,
    R1B_ELIGIBLE_SHAPES,
    R1B_PROVENANCE_TAGS,
    compile_expanded_queries,
    query_exceeds_length_budget,
    select_expansion_map_for_case,
)


# =====================================================================
# Closed-set constants.
# =====================================================================
def test_R1B_curated_map_keys_closed_set():
    assert set(CURATED_EXPANSION_MAPS.keys()) == {
        "cholestatic_liver_disease_mechanism",
        "amyloid_cardiomyopathy_mechanism",
        "pcos_insulin_resistance_mechanism",
        "central_line_pneumothorax_complication",
        "pneumothorax_outcome",
        "procedure_complication_boost",
    }


def test_R1B_case_id_to_expansion_map_closed_set():
    assert set(CASE_ID_TO_EXPANSION_MAP.keys()) == {
        "r1a-case-A-cholestatic-pruritus",
        "r1a-case-B-pcos-insulin-resistance",
        "r1a-case-C-amyloid-diastolic-dysfunction",
        "r1a-case-D-cvc-pneumothorax",
    }


def test_R1B_eligible_shapes_closed_set():
    assert R1B_ELIGIBLE_SHAPES == frozenset({
        "pathophysiology_mechanism",
        "procedure_complication_risk",
        "procedure_complication",
    })


def test_R1B_provenance_tags_closed_set():
    assert R1B_PROVENANCE_TAGS == frozenset({
        "high_precision_family",
        "broad_family",
        "both",
        "active_only",
    })


def test_R1B_mechanism_vocab_contains_expected_terms():
    assert "mechanism" in MECHANISM_CONCEPT_TERMS
    assert "pathophysiology" in MECHANISM_CONCEPT_TERMS
    assert "pathogenesis" in MECHANISM_CONCEPT_TERMS


def test_R1B_procedure_complication_vocab_contains_expected_terms():
    assert "complication" in PROCEDURE_COMPLICATION_OUTCOME_VOCAB
    assert "incidence" in PROCEDURE_COMPLICATION_OUTCOME_VOCAB


def test_R1B_select_expansion_map_for_case_returns_expected():
    assert (
        select_expansion_map_for_case("r1a-case-A-cholestatic-pruritus")
        == "cholestatic_liver_disease_mechanism"
    )
    assert (
        select_expansion_map_for_case("r1a-case-D-cvc-pneumothorax")
        == "central_line_pneumothorax_complication"
    )
    assert select_expansion_map_for_case("does-not-exist") is None


# =====================================================================
# Compile — mechanism shape.
# =====================================================================
def test_R1B_compile_mechanism_returns_broad_only():
    out = compile_expanded_queries(
        shape="pathophysiology_mechanism",
        expansion_map_name="cholestatic_liver_disease_mechanism",
    )
    assert out["shape"] == "pathophysiology_mechanism"
    assert out["expansion_map_used"] == "cholestatic_liver_disease_mechanism"
    # Broad terms include curated map + generalizable vocab.
    broad = out["broad_family_terms"]
    assert "cholestasis" in broad
    assert "autotaxin" in broad
    assert "MRGPRX4" in broad
    assert "mechanism" in broad  # from generalizable
    assert "pathophysiology" in broad
    # Mechanism shape produces NO high-precision family.
    assert out["high_precision_family_query_string"] is None
    assert out["high_precision_family_terms"] is None
    assert out["high_precision_family_required_both_anchors"] is False
    assert isinstance(out["broad_family_query_string"], str)
    assert "OR" in out["broad_family_query_string"]


def test_R1B_compile_mechanism_with_no_curated_map_still_adds_shape_vocab():
    out = compile_expanded_queries(
        shape="pathophysiology_mechanism",
        expansion_map_name=None,
    )
    broad = out["broad_family_terms"]
    # No curated map means broad terms come ONLY from the
    # generalizable mechanism vocab.
    assert "mechanism" in broad
    assert "pathophysiology" in broad
    assert out["high_precision_family_query_string"] is None


# =====================================================================
# Compile — procedure_complication shape: high-precision family.
# =====================================================================
def test_R1B_compile_procedure_complication_high_precision_required():
    out = compile_expanded_queries(
        shape="procedure_complication_risk",
        expansion_map_name="central_line_pneumothorax_complication",
    )
    assert out["high_precision_family_required_both_anchors"] is True
    hp = out["high_precision_family_query_string"]
    assert isinstance(hp, str)
    # MUST contain the (procedure_anchor) AND (outcome_anchor) form.
    assert ") AND (" in hp, hp
    # Procedure anchor terms must appear.
    assert '"central venous catheterization"' in hp or "CVC" in hp
    # Outcome anchor terms must appear.
    assert '"iatrogenic pneumothorax"' in hp
    # Broad family must also include procedure_complication outcome vocab.
    broad = out["broad_family_terms"]
    assert "complication" in broad or "incidence" in broad


def test_R1B_compile_procedure_complication_no_curated_uses_generic_anchors():
    out = compile_expanded_queries(
        shape="procedure_complication_risk",
        expansion_map_name=None,
    )
    hp = out["high_precision_family_query_string"]
    assert hp is not None
    # Generic procedure anchors must appear when curated is missing.
    assert any(t in hp for t in GENERIC_PROCEDURE_ANCHOR_VOCAB[:3])
    assert ") AND (" in hp


# =====================================================================
# Compile — non-eligible shape returns empty broad + no HP.
# =====================================================================
def test_R1B_compile_other_shape_emits_empty_broad_and_no_hp():
    out = compile_expanded_queries(
        shape="therapy",
        expansion_map_name=None,
    )
    assert out["broad_family_terms"] == []
    assert out["high_precision_family_query_string"] is None
    assert out["broad_family_query_string"] == ""


# =====================================================================
# OR join — quoting + dedupe.
# =====================================================================
def test_R1B_or_join_quotes_multi_word_and_dedupes():
    out = compile_expanded_queries(
        shape="pathophysiology_mechanism",
        expansion_map_name="cholestatic_liver_disease_mechanism",
    )
    broad_q = out["broad_family_query_string"]
    # Multi-word terms must be quoted.
    assert '"cholestatic pruritus"' in broad_q
    assert '"lysophosphatidic acid"' in broad_q
    # Single-word terms must be bare.
    assert " cholestasis " in f" {broad_q} " or broad_q.startswith("cholestasis")
    # Generalizable + curated must not duplicate a shared term
    # (none expected by construction).
    assert broad_q.count("pathophysiology") == 1


# =====================================================================
# URL length budget guard.
# =====================================================================
def test_R1B_query_length_budget_pass():
    out = compile_expanded_queries(
        shape="pathophysiology_mechanism",
        expansion_map_name="cholestatic_liver_disease_mechanism",
    )
    assert not query_exceeds_length_budget(
        out["broad_family_query_string"]
    )


def test_R1B_query_length_budget_detect_overflow():
    huge = "x" * (MAX_QUERY_STRING_LEN + 100)
    assert query_exceeds_length_budget(huge) is True
    assert query_exceeds_length_budget("") is False
    assert query_exceeds_length_budget(None) is False  # type: ignore[arg-type]


# =====================================================================
# Determinism — same inputs → same outputs.
# =====================================================================
@pytest.mark.parametrize("shape,map_name", [
    ("pathophysiology_mechanism", "cholestatic_liver_disease_mechanism"),
    ("pathophysiology_mechanism", "amyloid_cardiomyopathy_mechanism"),
    ("pathophysiology_mechanism", "pcos_insulin_resistance_mechanism"),
    ("procedure_complication_risk", "central_line_pneumothorax_complication"),
    ("pathophysiology_mechanism", None),
    ("procedure_complication_risk", None),
])
def test_R1B_compile_is_deterministic(shape, map_name):
    a = compile_expanded_queries(shape=shape, expansion_map_name=map_name)
    b = compile_expanded_queries(shape=shape, expansion_map_name=map_name)
    assert a == b
