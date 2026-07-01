"""Phase M3-preflight — would-be compiler-input structural projection
tests.

DETERMINISTIC. NO LLM. NO RETRIEVAL. NO SOURCE DISPATCH.

The projection module produces a closed-set dict mapping the
validator-accepted candidate to what the deterministic Boolean
query compiler WOULD receive as inputs if M3 were active. Each
test asserts:

  * Schema of the returned dict is closed-set.
  * ``active_retrieval_changed`` is always False (M3-preflight
    invariant).
  * ``compiler_dry_run_status`` is the closed-set
    ``"deferred_to_m3_active"`` token.
  * Diff fields correctly classify added / removed / changed.
"""
from __future__ import annotations

from services.semantic_repair_active_projection import (
    COMPILER_DRY_RUN_STATUSES,
    project_would_be_compiler_input,
)


def _good_candidate() -> dict:
    return {
        "primary_intent": "risk_etiology",
        "secondary_intents": [],
        "frameworks": ["risk_etiology"],
        "branches": [],
        "slots": {
            "population": ["young adults"],
            "condition": ["schizophrenia"],
            "intervention_or_exposure": [],
            "comparator": [],
            "outcome": [],
            "index_test": [],
            "target_condition": [],
            "procedure_or_device": [],
            "timeframe": [],
            "setting": [],
            "mechanism": [],
        },
        "slot_policies": {"condition": "required_for_retrieval"},
        "required_anchors": ["schizophrenia"],
        "optional_or_ranking_only_concepts": [],
        "inferred_concepts": [],
        "candidate_keywords_unvalidated": [],
        "candidate_mesh_terms_unvalidated": ["Schizophrenia"],
        "semantic_repair_reason": "missing_risk_etiology_framework",
        "confidence": 0.85,
        "provenance": "bounded_semantic_repair_shadow",
    }


def test_M3_projection_returns_closed_set_keys() -> None:
    out = project_would_be_compiler_input(
        candidate_dict=_good_candidate(),
        current_primary_intent="background",
        current_frameworks=[],
        current_required_anchors=["causes of schizophrenia in young adults"],
        current_branch_count=0,
    )
    assert set(out.keys()) == {
        "would_be_primary_intent",
        "would_be_frameworks",
        "would_be_required_anchors",
        "would_be_branch_count",
        "would_be_semantic_repair_reason",
        "would_be_diff_against_current",
        "compiler_dry_run_status",
        "active_retrieval_changed",
    }


def test_M3_projection_active_retrieval_never_changes_in_preflight() -> None:
    out = project_would_be_compiler_input(
        candidate_dict=_good_candidate(),
        current_primary_intent="background",
        current_frameworks=[],
        current_required_anchors=["causes of schizophrenia in young adults"],
        current_branch_count=0,
    )
    assert out["active_retrieval_changed"] is False


def test_M3_projection_compiler_dry_run_status_is_deferred() -> None:
    out = project_would_be_compiler_input(
        candidate_dict=_good_candidate(),
        current_primary_intent="background",
        current_frameworks=[],
        current_required_anchors=[],
        current_branch_count=0,
    )
    assert out["compiler_dry_run_status"] == "deferred_to_m3_active"
    assert out["compiler_dry_run_status"] in COMPILER_DRY_RUN_STATUSES


def test_M3_projection_intent_diff_detects_change() -> None:
    out = project_would_be_compiler_input(
        candidate_dict=_good_candidate(),
        current_primary_intent="background",
        current_frameworks=[],
        current_required_anchors=[],
        current_branch_count=0,
    )
    diff = out["would_be_diff_against_current"]
    assert diff["intent"]["current"] == "background"
    assert diff["intent"]["would_be"] == "risk_etiology"
    assert diff["intent"]["changed"] is True


def test_M3_projection_intent_diff_no_change_when_equivalent_casing() -> None:
    out = project_would_be_compiler_input(
        candidate_dict=_good_candidate(),
        current_primary_intent="RISK_ETIOLOGY",  # casing variance
        current_frameworks=[],
        current_required_anchors=[],
        current_branch_count=0,
    )
    diff = out["would_be_diff_against_current"]
    assert diff["intent"]["changed"] is False


def test_M3_projection_required_anchors_diff_detects_added_and_removed() -> None:
    out = project_would_be_compiler_input(
        candidate_dict=_good_candidate(),
        current_primary_intent="risk_etiology",
        current_frameworks=["risk_etiology"],
        current_required_anchors=["causes of schizophrenia in young adults"],
        current_branch_count=0,
    )
    diff = out["would_be_diff_against_current"]["required_anchors"]
    assert "schizophrenia" in diff["added"]
    assert "causes of schizophrenia in young adults" in diff["removed"]
    assert diff["changed"] is True


def test_M3_projection_frameworks_diff_detects_added() -> None:
    out = project_would_be_compiler_input(
        candidate_dict=_good_candidate(),
        current_primary_intent="risk_etiology",
        current_frameworks=[],
        current_required_anchors=["schizophrenia"],
        current_branch_count=0,
    )
    diff = out["would_be_diff_against_current"]["frameworks"]
    assert "risk_etiology" in diff["added"]
    assert diff["removed"] == []
    assert diff["changed"] is True


def test_M3_projection_branch_count_diff() -> None:
    cand = _good_candidate()
    cand["branches"] = [
        {"branch_label": "x", "branch_intent": "risk_etiology",
         "anchors": ["schizophrenia"]},
        {"branch_label": "y", "branch_intent": "risk_etiology",
         "anchors": ["young adults"]},
    ]
    out = project_would_be_compiler_input(
        candidate_dict=cand,
        current_primary_intent="risk_etiology",
        current_frameworks=["risk_etiology"],
        current_required_anchors=["schizophrenia"],
        current_branch_count=0,
    )
    bc = out["would_be_diff_against_current"]["branch_count"]
    assert bc["current"] == 0
    assert bc["would_be"] == 2
    assert bc["changed"] is True


def test_M3_projection_handles_empty_candidate() -> None:
    out = project_would_be_compiler_input(
        candidate_dict=None,
        current_primary_intent="risk_etiology",
        current_frameworks=[],
        current_required_anchors=[],
        current_branch_count=0,
    )
    assert out["would_be_primary_intent"] is None
    assert out["would_be_required_anchors"] == []
    assert out["active_retrieval_changed"] is False
    assert out["compiler_dry_run_status"] == "deferred_to_m3_active"


def test_M3_projection_does_not_invoke_compiler() -> None:
    """The projection module is pure-data: it derives structural
    fields from the candidate and the current map, but it does NOT
    import or invoke ``services.search_planning`` /
    ``services.literature_retrieval`` / anything in the query
    compilation / source dispatch path."""
    import services.semantic_repair_active_projection as mod
    # Inspect dir() for closed-set of forbidden module names.
    forbidden = (
        "search_planning",
        "literature_retrieval",
        "evidence_ranker",
        "summarizer",
        "citation_verifier",
        "query_compiler",
    )
    for name in dir(mod):
        for f in forbidden:
            assert f not in str(name).lower(), (
                f"projection module references {f!r} via attr {name!r}"
            )
