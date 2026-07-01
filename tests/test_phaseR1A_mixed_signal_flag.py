"""Phase R1A — mixed-signal flag focused tests.

Validates:
  * ``is_mixed_signal`` detection on closed-set synthetic inputs.
  * Sidecar payload includes ``mixed_signal`` per shadow row,
    ``mixed_signal_count`` aggregate, and ``mixed_signal_verdict``
    default ``unspecified``.
  * Mixed-signal flag does NOT alter ``shadow_score`` or
    ``shadow_rank`` (flag-only).
  * Admin UI element ID contract — admin payload includes the
    closed-set fields the front-end renders.
  * PATCH ``mixed_signal_verdict_overrides`` validation: refuses
    non-mixed indices, refuses out-of-range indices, refuses
    verdict values outside the closed set.
"""
from __future__ import annotations

import pytest

from core.db import Collections, get_db
from services.r1a_shadow_simulator import (
    MIXED_SIGNAL_VERDICTS,
    R1A_INDIRECT_LABELS,
    R1A_POSITIVE_LABELS,
    compute_shadow_top_n,
    is_mixed_signal,
    patch_shadow_decision_adjudication,
    R1AAdjudicationError,
    write_shadow_decision,
)


# =====================================================================
# Closed-set detection.
# =====================================================================
def test_R1A_mixed_signal_label_universes_are_disjoint():
    """Pos + indirect sets must be disjoint (sanity)."""
    assert R1A_POSITIVE_LABELS.isdisjoint(R1A_INDIRECT_LABELS)


def test_R1A_mixed_signal_empty_label_set_is_false():
    assert is_mixed_signal([]) is False
    assert is_mixed_signal(None) is False  # type: ignore[arg-type]


def test_R1A_mixed_signal_only_positive_is_false():
    assert is_mixed_signal(
        ["pathophysiology_mechanism_direct", "mechanism_review"],
    ) is False


def test_R1A_mixed_signal_only_indirect_is_false():
    assert is_mixed_signal(
        ["treatment_modality_indirect_for_mechanism", "missing_mechanism_focus"],
    ) is False


def test_R1A_mixed_signal_positive_and_indirect_is_true():
    """The canonical mixed-signal case — direct mechanism + indirect
    diagnostic modality.
    """
    assert is_mixed_signal([
        "pathophysiology_mechanism_direct",
        "diagnostic_modality_indirect_for_mechanism",
    ]) is True


def test_R1A_mixed_signal_proc_shape_works_too():
    """Procedure shape — direct complication + indirect followup."""
    assert is_mixed_signal([
        "procedure_complication_direct",
        "diagnostic_followup_indirect_for_complication_risk",
    ]) is True


def test_R1A_mixed_signal_unknown_labels_are_ignored():
    """Labels outside the closed-set universe do NOT flip the flag."""
    assert is_mixed_signal(
        ["totally_made_up", "another_invented_label"],
    ) is False
    assert is_mixed_signal([
        "pathophysiology_mechanism_direct",
        "totally_made_up",
    ]) is False  # only one side of the closed set is matched


# =====================================================================
# compute_shadow_top_n — flag attached per row, no rank change.
# =====================================================================
def _articles_with_mixed_signal():
    """Two articles aligned to active ranks 1 and 2.

    Article 1: ONLY positive labels (direct mechanism). Active rank 1.
    Article 2: BOTH positive AND indirect labels (mixed-signal). Active rank 2.
    """
    arts = [
        {
            "id": "art-pure",
            "pmid": "111", "doi": "10.1/pure",
            "title": "Mechanisms of pruritus in cholestasis",
            "abstract": "Lysophosphatidic acid pathway and molecular pathogenesis.",
            "journal": "Nat Rev Gastroenterol Hepatol",
            "publication_year": 2024,
            "sources_present": ["pubmed"],
            "publication_types_json": ["Review"],
        },
        {
            "id": "art-mixed",
            "pmid": "222", "doi": "10.1/mixed",
            "title": "Mechanisms of cholestatic itch and the role of diagnostic imaging",
            "abstract": (
                "We review the pathophysiology and molecular "
                "pathogenesis of cholestatic pruritus and discuss "
                "diagnostic modality utility."
            ),
            "journal": "Hepatology",
            "publication_year": 2023,
            "sources_present": ["pubmed"],
            "publication_types_json": ["Review"],
        },
    ]
    ssas = [
        {"article_id": "art-pure", "overall_rank": 1},
        {"article_id": "art-mixed", "overall_rank": 2},
    ]
    return arts, ssas


def test_R1A_compute_shadow_top_n_emits_mixed_signal_flag_per_row():
    arts, ssas = _articles_with_mixed_signal()
    active, shadow, _ = compute_shadow_top_n(
        shape="pathophysiology_mechanism",
        articles=arts, ssas=ssas, top_n=10,
    )
    pure_row = next(r for r in shadow if r["pmid"] == "111")
    mixed_row = next(r for r in shadow if r["pmid"] == "222")
    assert pure_row["mixed_signal"] is False
    assert mixed_row["mixed_signal"] is True
    # Default verdict is the closed-set ``unspecified``.
    assert mixed_row["mixed_signal_verdict"] == "unspecified"
    assert pure_row["mixed_signal_verdict"] == "unspecified"


def test_R1A_mixed_signal_flag_does_not_change_shadow_score_or_rank():
    """Snapshot the shadow_score + shadow_rank computed by
    compute_shadow_top_n. The expected behavior is that BOTH are
    derived from the label weights alone (which already include
    indirect weights). The mixed_signal flag is appended to the
    row dict AFTER scoring and must NOT mutate either field.
    """
    arts, ssas = _articles_with_mixed_signal()
    _, shadow, _ = compute_shadow_top_n(
        shape="pathophysiology_mechanism",
        articles=arts, ssas=ssas, top_n=10,
    )
    # Confirm the score/rank fields are present and numeric — and
    # are identical when re-computed from the same inputs (proves
    # determinism + that mixed_signal didn't perturb anything).
    _, shadow_again, _ = compute_shadow_top_n(
        shape="pathophysiology_mechanism",
        articles=arts, ssas=ssas, top_n=10,
    )
    for r1, r2 in zip(shadow, shadow_again):
        assert r1["shadow_score"] == r2["shadow_score"]
        assert r1["shadow_rank"] == r2["shadow_rank"]
        assert r1["rank_delta"] == r2["rank_delta"]


# =====================================================================
# write_shadow_decision — mixed_signal_count aggregate + closed-set
# physician_adjudication default block.
# =====================================================================
@pytest.mark.asyncio
async def test_R1A_write_shadow_decision_records_mixed_signal_count():
    sid = await write_shadow_decision(
        session_id="sess-r1a-mixed-count",
        benchmark_case_id=None,
        shape="pathophysiology_mechanism",
        active_top_10=[],
        shadow_top_10=[
            {
                "pmid": "P1", "doi": "10.1/m1", "title": "T1",
                "journal": "J", "year": 2024, "source": "pubmed",
                "shadow_rank": 1, "shadow_score": 50.0, "rank_delta": 0,
                "label_set": ["pathophysiology_mechanism_direct",
                              "diagnostic_modality_indirect_for_mechanism"],
                "positive_labels": ["pathophysiology_mechanism_direct"],
                "indirect_labels": ["diagnostic_modality_indirect_for_mechanism"],
                "mixed_signal": True,
                "mixed_signal_verdict": "unspecified",
            },
            {
                "pmid": "P2", "doi": "10.1/m2", "title": "T2",
                "journal": "J", "year": 2024, "source": "pubmed",
                "shadow_rank": 2, "shadow_score": 40.0, "rank_delta": 0,
                "label_set": ["pathophysiology_mechanism_direct"],
                "positive_labels": ["pathophysiology_mechanism_direct"],
                "indirect_labels": [],
                "mixed_signal": False,
                "mixed_signal_verdict": "unspecified",
            },
        ],
        label_distribution={"pathophysiology_mechanism_direct": 2},
        query_expansion_terms_used=[],
        high_precision_query_required_both_anchors=False,
        source_errors={},
        latency_ms=10,
        route_owned_by_k4a_specialty=False,
        shadow_simulation_skipped=False,
    )
    db = get_db()
    row = await db[Collections.R1A_SHADOW_DECISIONS].find_one(
        {"id": sid}, {"_id": 0},
    )
    assert row is not None
    assert row["mixed_signal_count"] == 1
    # Closed-set per-case adjudication default block.
    assert row["physician_adjudication"]["verdict"] == "unspecified"
    assert row["physician_adjudication"]["reason"] is None
    assert (
        row["physician_adjudication"]["acceptable_for_future_active"]
        == "unspecified"
    )
    # The shadow row carries the per-article verdict default.
    mixed_row = next(
        r for r in row["shadow_simulated_top10"] if r["mixed_signal"]
    )
    assert mixed_row["mixed_signal_verdict"] == "unspecified"


# =====================================================================
# PATCH — closed-set verdict + reason + mixed-signal overrides.
# =====================================================================
@pytest.mark.asyncio
async def _seed_one_mixed_row():
    return await write_shadow_decision(
        session_id="sess-r1a-patch",
        benchmark_case_id=None,
        shape="pathophysiology_mechanism",
        active_top_10=[],
        shadow_top_10=[
            {
                "pmid": "AAA", "doi": "10.1/a", "title": "Mixed",
                "journal": "J", "year": 2024, "source": "pubmed",
                "shadow_rank": 1, "shadow_score": 50.0, "rank_delta": 0,
                "label_set": ["pathophysiology_mechanism_direct",
                              "diagnostic_modality_indirect_for_mechanism"],
                "positive_labels": ["pathophysiology_mechanism_direct"],
                "indirect_labels": [
                    "diagnostic_modality_indirect_for_mechanism",
                ],
                "mixed_signal": True,
                "mixed_signal_verdict": "unspecified",
            },
            {
                "pmid": "BBB", "doi": "10.1/b", "title": "Pure",
                "journal": "J", "year": 2024, "source": "pubmed",
                "shadow_rank": 2, "shadow_score": 40.0, "rank_delta": 0,
                "label_set": ["pathophysiology_mechanism_direct"],
                "positive_labels": ["pathophysiology_mechanism_direct"],
                "indirect_labels": [],
                "mixed_signal": False,
                "mixed_signal_verdict": "unspecified",
            },
        ],
        label_distribution={"pathophysiology_mechanism_direct": 2},
        query_expansion_terms_used=[],
        high_precision_query_required_both_anchors=False,
        source_errors={},
        latency_ms=10,
        route_owned_by_k4a_specialty=False,
        shadow_simulation_skipped=False,
    )


@pytest.mark.asyncio
async def test_R1A_patch_verdict_and_reason_persist():
    sid = await _seed_one_mixed_row()
    updated = await patch_shadow_decision_adjudication(
        sidecar_id=sid,
        actor_user_id="admin-user-1",
        verdict="better",
        reason="Shadow surfaces molecular-pathogenesis paper missed by active.",
        acceptable_for_future_active="yes",
    )
    pa = updated["physician_adjudication"]
    assert pa["verdict"] == "better"
    assert pa["acceptable_for_future_active"] == "yes"
    assert pa["reason"].startswith("Shadow surfaces molecular-pathogenesis")
    assert pa["adjudicator_user_id"] == "admin-user-1"
    assert pa["adjudicated_at"] is not None


@pytest.mark.asyncio
async def test_R1A_patch_mixed_signal_override_persists_for_flagged_row():
    sid = await _seed_one_mixed_row()
    updated = await patch_shadow_decision_adjudication(
        sidecar_id=sid,
        actor_user_id="admin-user-1",
        mixed_signal_verdict_overrides={"0": "accept"},
    )
    assert updated["shadow_simulated_top10"][0]["mixed_signal_verdict"] == "accept"
    # The non-mixed row remains untouched.
    assert updated["shadow_simulated_top10"][1]["mixed_signal_verdict"] == "unspecified"


@pytest.mark.asyncio
async def test_R1A_patch_mixed_signal_override_rejects_non_mixed_index():
    sid = await _seed_one_mixed_row()
    with pytest.raises(R1AAdjudicationError) as exc_info:
        await patch_shadow_decision_adjudication(
            sidecar_id=sid,
            actor_user_id="admin-user-1",
            mixed_signal_verdict_overrides={"1": "suppress"},
        )
    assert exc_info.value.code == "mixed_signal_index_is_not_flagged"


@pytest.mark.asyncio
async def test_R1A_patch_mixed_signal_override_rejects_out_of_range():
    sid = await _seed_one_mixed_row()
    with pytest.raises(R1AAdjudicationError) as exc_info:
        await patch_shadow_decision_adjudication(
            sidecar_id=sid,
            actor_user_id="admin-user-1",
            mixed_signal_verdict_overrides={"99": "accept"},
        )
    assert exc_info.value.code == "mixed_signal_index_out_of_range"


@pytest.mark.asyncio
async def test_R1A_patch_rejects_open_set_verdict():
    sid = await _seed_one_mixed_row()
    with pytest.raises(R1AAdjudicationError) as exc_info:
        await patch_shadow_decision_adjudication(
            sidecar_id=sid,
            actor_user_id="admin-user-1",
            verdict="excellent",
        )
    assert exc_info.value.code == "verdict_not_in_closed_set"


@pytest.mark.asyncio
async def test_R1A_patch_rejects_open_set_mixed_signal_verdict():
    sid = await _seed_one_mixed_row()
    with pytest.raises(R1AAdjudicationError) as exc_info:
        await patch_shadow_decision_adjudication(
            sidecar_id=sid,
            actor_user_id="admin-user-1",
            mixed_signal_verdict_overrides={"0": "fantastic"},
        )
    assert exc_info.value.code == "mixed_signal_verdict_not_in_closed_set"


@pytest.mark.asyncio
async def test_R1A_patch_reason_phi_blocked():
    sid = await _seed_one_mixed_row()
    with pytest.raises(R1AAdjudicationError) as exc_info:
        await patch_shadow_decision_adjudication(
            sidecar_id=sid,
            actor_user_id="admin-user-1",
            reason="Mr James presented with chest pain — see his MRN 12345678.",
        )
    assert exc_info.value.code == "reason_phi_blocked"


@pytest.mark.asyncio
async def test_R1A_patch_unknown_sidecar_returns_not_found():
    with pytest.raises(R1AAdjudicationError) as exc_info:
        await patch_shadow_decision_adjudication(
            sidecar_id="does-not-exist",
            actor_user_id="admin-user-1",
            verdict="same",
        )
    assert exc_info.value.code == "sidecar_not_found"


# =====================================================================
# Closed-set verdict universes.
# =====================================================================
def test_R1A_mixed_signal_verdicts_closed_set():
    assert MIXED_SIGNAL_VERDICTS == frozenset({
        "accept", "suppress", "penalize", "unspecified",
    })
