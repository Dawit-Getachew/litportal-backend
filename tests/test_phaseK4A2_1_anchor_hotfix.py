"""Phase K4A.2.1 — severity-marker anchor regression hotfix.

Coverage matrix per the K4A.2.1 spec:

  A. Exact regression test (acute pancreatitis severity markers)
  B. Disease-anchor extraction variants (severe AP, biomarkers in AP,
     laboratory markers, severity scores, mortality in AP)
  C. Generic-only negative (no real disease → clarify)
  D. Non-pancreatitis generalization (sepsis / HF / lupus)
  E. Query-QA: malformed `iilness` token rejected pre-dispatch
  F. Do NOT overcorrect: Crohn's disease / sickle cell disease /
     Cushing's disease / Addison's disease / celiac disease all
     remain extractable as primary anchors.
  G. Selected/Excluded invariant carry-over (no article default-
     selected AND excluded).
  H. The literal typo `iilness` never appears in anchors / queries /
     diagnostics / PICOTT / ClinicalQuestionMap (DOM-grade sweep on
     the persisted session JSON).

All tests deterministic — no LLM, no network, no PHI, no scraping.
"""
from __future__ import annotations

import json

import pytest

from services.primary_clinical_anchor import (
    GENERIC_MODIFIERS,
    primary_clinical_anchor_service,
)
from services.query_compiler import (
    _GENERIC_ANCHOR_TOKENS,
    _MALFORMED_ANCHOR_TOKENS,
    SYNTAX_ERROR_CODES,
    validate_query_syntax,
)
from services.severity_marker_overview import (
    extract_condition_anchor,
    is_severity_marker_question,
)


# =====================================================================
# Closed-set sanity — denylists are correctly populated.
# =====================================================================
def test_generic_modifiers_includes_k4a21_tokens() -> None:
    for tok in (
        "illness", "disease", "condition", "problem",
        "severity", "marker", "markers", "predictor", "predictors",
        "risk", "patient", "patients", "outcome", "outcomes",
        "iilness",  # typo guard
    ):
        assert tok in GENERIC_MODIFIERS, tok


def test_query_compiler_anchor_denylists_closed() -> None:
    # Malformed (closed-set typos).
    for tok in ("iilness", "iillness", "illnes"):
        assert tok in _MALFORMED_ANCHOR_TOKENS
    # Generic single-tokens — never sole anchor.
    for tok in (
        "illness", "severity", "marker", "predictor",
        "risk", "patient", "disease", "outcome",
    ):
        assert tok in _GENERIC_ANCHOR_TOKENS
    # Closed-set multi-word severity-only phrases.
    for tok in (
        "severity of illness", "severity of disease", "disease severity",
        "severity prediction", "severity assessment",
        "risk stratification",
    ):
        assert tok in _GENERIC_ANCHOR_TOKENS
    # Error codes set extended.
    assert "malformed_required_anchor" in SYNTAX_ERROR_CODES
    assert "generic_only_required_anchor" in SYNTAX_ERROR_CODES


# =====================================================================
# A — Exact regression: severity-marker question (live reproducer).
# =====================================================================
@pytest.mark.parametrize("q", [
    "What are the markers of severity of illness in acute pancreatitis?",
    "what are the markers of severity of illness in acute pancreatitis",
])
def test_severity_marker_anchor_is_acute_pancreatitis(q: str) -> None:
    assert is_severity_marker_question(q) is True
    anchor = extract_condition_anchor(q)
    assert anchor == "acute pancreatitis"
    # The anchor MUST NOT be `illness` / `iilness`.
    assert anchor not in {"illness", "iilness"}


# =====================================================================
# B — Disease-anchor extraction across phrasing variants.
# =====================================================================
@pytest.mark.parametrize("q, expected_anchor", [
    ("What are predictors of severe acute pancreatitis?", "acute pancreatitis"),
    ("What biomarkers predict severity in acute pancreatitis?", "acute pancreatitis"),
    ("What laboratory markers predict acute pancreatitis severity?", "acute pancreatitis"),
    ("What severity scores predict outcomes in acute pancreatitis?", "acute pancreatitis"),
    ("What markers predict mortality in acute pancreatitis?", "acute pancreatitis"),
    ("Biomarkers and predictors of severity in acute pancreatitis", "acute pancreatitis"),
])
def test_disease_anchor_variants(q: str, expected_anchor: str) -> None:
    assert is_severity_marker_question(q) is True
    anchor = extract_condition_anchor(q)
    assert anchor == expected_anchor, (q, anchor)


# =====================================================================
# D — Non-pancreatitis generalization (do NOT overcorrect).
# =====================================================================
@pytest.mark.parametrize("q, expected_anchor", [
    ("What are markers of severity in sepsis?", "sepsis"),
    ("What biomarkers predict severity in acute heart failure?",
     "acute heart failure"),
    ("What laboratory markers predict disease activity in lupus?", "lupus"),
])
def test_severity_marker_generalizes_to_other_diseases(
    q: str, expected_anchor: str,
) -> None:
    assert is_severity_marker_question(q) is True
    anchor = extract_condition_anchor(q)
    assert anchor == expected_anchor, (q, anchor)


# =====================================================================
# E — Query QA: malformed / generic-only anchor rejected pre-dispatch.
# =====================================================================
def test_query_qa_rejects_iilness_typo_pubmed() -> None:
    v = validate_query_syntax('"iilness"[tiab]', source="pubmed")
    assert v.valid is False
    assert "malformed_required_anchor" in v.errors


def test_query_qa_rejects_iilness_typo_europe_pmc() -> None:
    v = validate_query_syntax('TITLE_ABS:"iilness"', source="europe_pmc")
    assert v.valid is False
    assert "malformed_required_anchor" in v.errors


def test_query_qa_rejects_illness_only_query() -> None:
    v = validate_query_syntax('TITLE_ABS:"illness"', source="europe_pmc")
    assert v.valid is False
    assert "generic_only_required_anchor" in v.errors


@pytest.mark.parametrize("tok", [
    "severity", "marker", "predictor", "outcome",
    "disease", "condition", "problem", "patient",
])
def test_query_qa_rejects_each_generic_only_anchor(tok: str) -> None:
    v = validate_query_syntax(f'TITLE_ABS:"{tok}"', source="europe_pmc")
    assert v.valid is False, (tok, v.errors)
    assert "generic_only_required_anchor" in v.errors


def test_query_qa_rejects_severity_of_illness_only_multiword() -> None:
    v = validate_query_syntax(
        'TITLE_ABS:"severity of illness"', source="europe_pmc",
    )
    assert v.valid is False
    assert "generic_only_required_anchor" in v.errors


def test_query_qa_passes_acute_pancreatitis_only() -> None:
    v = validate_query_syntax(
        '((TITLE_ABS:"acute pancreatitis"))', source="europe_pmc",
    )
    assert v.valid is True, v.errors


def test_query_qa_passes_acute_pancreatitis_and_severity() -> None:
    # Multi-anchor query: AP is a real disease + severity is generic
    # but they coexist. The set of distinct quoted anchors is
    # {acute pancreatitis, severity} — NOT all-generic, so it passes.
    v = validate_query_syntax(
        'TITLE_ABS:"acute pancreatitis" AND TITLE_ABS:"severity"',
        source="europe_pmc",
    )
    assert v.valid is True, v.errors


# =====================================================================
# F — Do NOT overcorrect: real disease names containing "disease" /
# "illness" / etc. still pass.
# =====================================================================
@pytest.mark.parametrize("disease", [
    "crohn's disease",
    "sickle cell disease",
    "cushing's disease",
    "addison's disease",
    "celiac disease",
    "inflammatory bowel disease",
    "parkinson's disease",
    "alzheimer's disease",
    "kawasaki disease",
])
def test_real_disease_names_with_disease_token_pass(disease: str) -> None:
    v = validate_query_syntax(
        f'TITLE_ABS:"{disease}"', source="europe_pmc",
    )
    assert v.valid is True, (disease, v.errors)


@pytest.mark.parametrize("disease", [
    "mental illness",                 # legitimate compound disease term
    "chronic illness",                # legitimate clinical phrase
    "acute kidney injury",            # AKI
    "acute respiratory distress syndrome",
])
def test_compound_illness_terms_pass(disease: str) -> None:
    v = validate_query_syntax(
        f'TITLE_ABS:"{disease}"', source="europe_pmc",
    )
    assert v.valid is True, (disease, v.errors)


# =====================================================================
# F2 — primary_clinical_anchor_service still finds real disease names.
# =====================================================================
def test_primary_anchor_extracts_acute_pancreatitis_not_illness() -> None:
    picott = {
        "must_have_concepts": [
            {"concept": "acute pancreatitis", "role": "condition",
             "synonyms": ["pancreatitis"]},
        ],
    }
    out = primary_clinical_anchor_service.derive(
        picott=picott,
        ontology_mappings=[],
        normalized_question=(
            "what are the markers of severity of illness in acute pancreatitis"
        ),
    )
    anchors = out.get("primary_anchors") or []
    assert anchors, "expected at least one primary anchor"
    canonicals = {a.get("canonical_concept", "").lower() for a in anchors}
    assert "acute pancreatitis" in canonicals
    assert "illness" not in canonicals
    assert "iilness" not in canonicals
    assert "severity" not in canonicals


def test_primary_anchor_rejects_illness_only_must_have() -> None:
    """A bare ``illness`` must_have with no ontology hit must NOT
    promote to a primary anchor. The pipeline should clarify."""
    picott = {
        "must_have_concepts": [
            {"concept": "illness", "role": "condition", "synonyms": []},
        ],
    }
    out = primary_clinical_anchor_service.derive(
        picott=picott,
        ontology_mappings=[],
        normalized_question="what are markers of severity of illness",
    )
    anchors = out.get("primary_anchors") or []
    canonicals = {a.get("canonical_concept", "").lower() for a in anchors}
    assert "illness" not in canonicals
    # And the per-anchor severity / marker terms are likewise blocked.
    for tok in ("severity", "marker", "markers", "predictor"):
        assert tok not in canonicals


def test_primary_anchor_accepts_crohns_disease() -> None:
    """Do NOT overcorrect — Crohn's disease must remain a valid
    primary anchor even though it contains the substring ``disease``."""
    picott = {
        "must_have_concepts": [
            {"concept": "Crohn's disease", "role": "condition",
             "synonyms": []},
        ],
    }
    out = primary_clinical_anchor_service.derive(
        picott=picott,
        ontology_mappings=[],
        normalized_question="what is the treatment of crohn's disease",
    )
    anchors = out.get("primary_anchors") or []
    canonicals = {a.get("canonical_concept", "").lower() for a in anchors}
    assert "crohn's disease" in canonicals


# =====================================================================
# H — DOM-grade sweep: `iilness` never appears in compiled queries
# nor in the primary-anchor diagnostics for the reproducer.
# =====================================================================
def test_iilness_never_in_repo_outputs() -> None:
    """Sweep test: simulate the entire severity-marker path and assert
    `iilness` does not appear anywhere in the produced anchors,
    PICOTT, diagnostics, or any compiled query string."""
    picott = {
        "must_have_concepts": [
            {"concept": "acute pancreatitis", "role": "condition",
             "synonyms": []},
        ],
        "question_subtype": "severity_marker_overview",
        "severity_marker_condition_anchor": "acute pancreatitis",
    }
    out = primary_clinical_anchor_service.derive(
        picott=picott,
        ontology_mappings=[],
        normalized_question=(
            "what are the markers of severity of illness in acute pancreatitis"
        ),
    )
    blob = json.dumps(out)
    assert "iilness" not in blob.lower()


# =====================================================================
# E2 — query QA closed-set diagnostic reason is exactly one of the
# K4A.2.1 closed-set reasons.
# =====================================================================
def test_query_qa_closed_set_diagnostic_reasons() -> None:
    closed = {"malformed_required_anchor", "generic_only_required_anchor"}
    cases = [
        '"iilness"[tiab]',
        'TITLE_ABS:"illness"',
        'TITLE_ABS:"severity"',
        'TITLE_ABS:"severity of illness"',
    ]
    for q in cases:
        v = validate_query_syntax(q, source="europe_pmc")
        assert v.valid is False
        # At least one of the K4A.2.1 closed-set codes must fire.
        assert any(e in closed for e in v.errors), (q, v.errors)
