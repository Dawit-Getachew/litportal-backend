"""Phase R1C.1 — shape-conditional recency cap + PubMed eSummary UID fix.

Shadow-only, no clinician path. Tests cover:
  1. ``compute_recency_base_score`` per-shape behaviour (cap for
     mechanism, linear for others) and edge cases (None, str, neg).
  2. ``SHADOW_LABEL_WEIGHTS`` byte-identical snapshot (R1A weights
     UNCHANGED).
  3. Case A end-to-end mock: Beuers 2023 surfaces into top-10 against
     newer noisy mechanism papers.
  4. Case B PCOS safety regression — treatment overrank counter does
     not worsen vs R1C baseline.
  5. Case D anchor specificity preservation — outcome_exact count
     >= 5 in top-10 with the cap applied (cap should not bite on the
     procedure shape).
  6. Case C source-coverage marker — closed-set miss reason
     ``not_indexed_in_pubmed_or_epmc`` when Crossref hits but PubMed/EPMC
     don't surface the article.
  7. PubMed eSummary UID iteration — empty / single / multi / malformed.
  8. R1A label weights snapshot (in addition to import-time assert).
  9. K4A invariance (shape gate refuses non-eligible shapes).
 10. M3.2 dormancy unchanged.
 11. Leak-scan on persisted sidecar artifact.
"""
from __future__ import annotations

import pytest

from core.db import Collections, get_db
from services.r1a_directness_labels import (
    MECHANISM_RECENCY_HARD_CAP,
    SHADOW_LABEL_WEIGHTS,
    compute_recency_base_score,
)
from services.r1a_gold_comparator_cases import seed_r1a_gold_comparator_cases


# =====================================================================
# 1. compute_recency_base_score — shape-conditional behaviour.
# =====================================================================
def test_R1C_1_mechanism_shape_caps_recency_at_hard_cap():
    # 2026 raw would be 36; cap = 2.0 → cap should bite.
    assert MECHANISM_RECENCY_HARD_CAP == 2.0
    out = compute_recency_base_score(
        year=2026, shape="pathophysiology_mechanism",
    )
    assert out == MECHANISM_RECENCY_HARD_CAP


def test_R1C_1_mechanism_shape_old_article_unaffected_by_cap():
    # 1991 raw = 1 < 2.0 → cap doesn't bite.
    out = compute_recency_base_score(
        year=1991, shape="pathophysiology_mechanism",
    )
    assert out == 1.0


def test_R1C_1_procedure_shape_recency_unchanged():
    # 2026 raw = 36; procedure shape preserves linear signal.
    out = compute_recency_base_score(
        year=2026, shape="procedure_complication_risk",
    )
    assert out == 36.0


def test_R1C_1_unknown_shape_preserves_linear_signal():
    out = compute_recency_base_score(year=2024, shape="diagnostic_yield")
    assert out == 34.0


def test_R1C_1_empty_shape_preserves_linear_signal():
    out = compute_recency_base_score(year=2024, shape="")
    assert out == 34.0


def test_R1C_1_year_none_returns_zero():
    out = compute_recency_base_score(
        year=None, shape="pathophysiology_mechanism",
    )
    # None coerces to 1990 → max(0, 1990 - 1990) = 0.
    assert out == 0.0


def test_R1C_1_year_string_parses_cleanly():
    out = compute_recency_base_score(
        year="2025", shape="pathophysiology_mechanism",
    )
    # 2025-1990=35 capped at 2.0 for mechanism.
    assert out == 2.0


def test_R1C_1_year_garbage_falls_back_to_zero():
    out = compute_recency_base_score(year="not-a-year", shape="")
    assert out == 0.0


def test_R1C_1_year_pre_1990_clamped_to_zero():
    # Negative recency is impossible — must be clamped.
    out = compute_recency_base_score(year=1980, shape="")
    assert out == 0.0


def test_R1C_1_mechanism_case_sensitivity_normalized():
    # Shape gate is case-insensitive (defensive).
    out = compute_recency_base_score(
        year=2025, shape="PATHOPHYSIOLOGY_MECHANISM",
    )
    assert out == 2.0


# =====================================================================
# 2. R1A SHADOW_LABEL_WEIGHTS snapshot — byte-identical.
# =====================================================================
def test_R1C_1_R1A_SHADOW_LABEL_WEIGHTS_byte_identical_snapshot():
    """The R1A directness label weights are operator-deferred for
    retuning. R1C.1 MUST NOT touch them. Snapshot here so any
    accidental change blows up loudly.
    """
    EXPECTED = {
        # pathophysiology_mechanism positive
        "pathophysiology_mechanism_direct": +1.0,
        "mechanism_review": +1.0,
        "clinical_mechanism_review": +1.0,
        "human_translational_mechanism": +0.8,
        "molecular_pathogenesis": +0.7,
        "disease_pathogenesis_review": +0.9,
        "mechanism_with_named_process": +1.2,
        # pathophysiology_mechanism negative
        "treatment_modality_indirect_for_mechanism": -0.6,
        "diagnostic_modality_indirect_for_mechanism": -0.4,
        "management_guideline_indirect_for_mechanism": -0.5,
        "broad_disease_overview_indirect_for_mechanism": -0.5,
        "animal_basic_science_lower_priority": -0.3,
        "missing_mechanism_focus": -0.8,
        # procedure_complication_risk positive
        "procedure_complication_direct": +1.0,
        "procedure_complication_incidence": +1.1,
        "procedure_complication_risk_factor": +1.0,
        "procedure_complication_cohort": +0.9,
        "procedure_complication_registry": +1.0,
        "procedure_complication_systematic_review": +1.2,
        "mechanical_complication_rate": +1.0,
        # procedure_complication_risk negative
        "diagnostic_followup_indirect_for_complication_risk": -0.6,
        "routine_imaging_utility_indirect": -0.5,
        "training_or_detection_method_indirect": -0.6,
        "procedure_general_overview_indirect": -0.4,
        "complication_missing_outcome": -0.7,
        "procedure_missing_exposure": -0.7,
    }
    assert SHADOW_LABEL_WEIGHTS == EXPECTED


# =====================================================================
# 3. Case A end-to-end mock — Beuers surfaces.
# =====================================================================
def _fetch_stub(records_for_broad, records_for_hp=None):
    async def _fn(*, query_string, search_session_id=None):  # noqa: ARG001
        if records_for_hp is not None and " AND " in (query_string or ""):
            return list(records_for_hp)
        return list(records_for_broad)
    return _fn


def _recovery_stub_no_op():
    async def _fn(**kwargs):  # noqa: ARG001
        return {
            "recovery_strategies_attempted": [],
            "recovery_strategy_that_surfaced_gold": None,
            "matched_records": [],
            "crossref_used": False,
            "pagination_pages_used": {"pubmed": 0, "europe_pmc": 0},
            "connector_recall_miss_reason": None,
            "pagination_ceiling_exceeded": False,
        }
    return _fn


@pytest.mark.asyncio
async def test_R1C_1_caseA_beuers_surfaces_into_top10_with_recency_cap():
    """Case A simulation: Beuers 2023 mechanism review competes
    against newer (2025-2026) noisy "mechanism" papers. With the
    +2.0 hard cap on mechanism-shape recency, Beuers' directness
    labels can dominate.
    """
    await seed_r1a_gold_comparator_cases()
    beuers = {
        "pmid": "BEUERS",
        "doi": "10.1038/s41575-022-00687-7",
        "title": "Mechanisms of pruritus in cholestasis: understanding and treating the itch",
        "abstract": (
            "Mechanistic review of cholestatic pruritus. Pathogenesis "
            "involves bile acids, lysophosphatidic acid, autotaxin, "
            "MRGPRX4, TGR5 and the itch pathway."
        ),
        "journal": "Nat Rev Gastroenterol Hepatol",
        "publication_year": 2023,
        "source": "pubmed",
        "publication_types_json": ["Review"],
    }
    newer_noise = []
    for i in range(8):
        newer_noise.append({
            "pmid": f"NOISE-{i}",
            "doi": f"10.0/noise{i}",
            "title": f"Mechanism of generic process in unrelated disease {i}",
            "abstract": "Molecular mechanism remodeling fibrosis pathway.",
            "journal": "J",
            "publication_year": 2025 + (i % 2),
            "source": "pubmed",
            "publication_types_json": ["Journal Article"],
        })
    candidates = newer_noise + [beuers]

    from services.r1c_shadow_simulator import replay_benchmark_case_r1c
    fetch = _fetch_stub(candidates)
    result = await replay_benchmark_case_r1c(
        case_id="r1a-case-A-cholestatic-pruritus",
        actor_user_id="admin-1",
        fetch_fn=fetch,
        recovery_fn=_recovery_stub_no_op(),
    )
    sidecar_id = result["sidecar_doc_id"]
    db = get_db()
    row = await db[Collections.R1C_SHADOW_DECISIONS].find_one(
        {"id": sidecar_id}, {"_id": 0},
    )
    pmids = [r["pmid"] for r in row["r1c_simulated_top_n_metadata"]]
    # Beuers MUST be in top-20 (top-10 stretch target).
    assert "BEUERS" in pmids, f"Beuers absent from top-20: {pmids}"
    rank = pmids.index("BEUERS") + 1
    assert rank <= 10, f"Beuers ranked {rank} — should be top-10 (cap applied)"


# =====================================================================
# 4. Case B PCOS safety regression — treatment overrank doesn't worsen.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_1_caseB_pcos_safety_treatment_overrank_does_not_worsen():
    """Closed-set R1C.1 invariant: applying the recency cap MUST NOT
    increase the treatment_overrank_count_r1c on the PCOS pathophysiology
    case relative to the R1B baseline.
    """
    await seed_r1a_gold_comparator_cases()
    trt1 = {
        "pmid": "TX1",
        "doi": "10.1/tx1",
        "title": "Randomized trial of metformin in PCOS insulin resistance",
        "abstract": "Treatment trial; metformin pharmacotherapy in PCOS.",
        "journal": "JCEM", "publication_year": 2024,
        "source": "pubmed",
        "publication_types_json": ["Randomized Controlled Trial"],
    }
    trt2 = {
        "pmid": "TX2",
        "doi": "10.1/tx2",
        "title": "Comparative trial of inositol for PCOS metabolic phenotype",
        "abstract": "Inositol therapy treatment trial in PCOS.",
        "journal": "JCEM", "publication_year": 2024,
        "source": "pubmed",
        "publication_types_json": ["Randomized Controlled Trial"],
    }
    mech = {
        "pmid": "MECHB",
        "doi": "10.1/mb",
        "title": "Insulin signaling in PCOS ovarian steroidogenesis",
        "abstract": "Insulin resistance pathophysiology in PCOS.",
        "journal": "JCEM", "publication_year": 2024,
        "source": "pubmed",
        "publication_types_json": ["Review"],
    }

    from services.r1c_shadow_simulator import replay_benchmark_case_r1c
    fetch = _fetch_stub([trt1, trt2, mech])
    result = await replay_benchmark_case_r1c(
        case_id="r1a-case-B-pcos-insulin-resistance",
        actor_user_id="admin-1",
        fetch_fn=fetch,
        recovery_fn=_recovery_stub_no_op(),
    )
    sidecar_id = result["sidecar_doc_id"]
    db = get_db()
    row = await db[Collections.R1C_SHADOW_DECISIONS].find_one(
        {"id": sidecar_id}, {"_id": 0},
    )
    # Stop-line: r1c_1 path's treatment_overrank_count_r1c MUST NOT
    # exceed the parallel r1b counter on the same candidate set.
    assert (
        row["treatment_overrank_count_r1c"]
        <= row["treatment_overrank_count_r1b"]
    ), row


# =====================================================================
# 5. Case D anchor specificity preservation under the cap.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_1_caseD_outcome_exact_count_holds_in_top10():
    """Procedure shape preserves the linear recency signal AND the
    procedure-complication anchor labels. outcome_exact count must
    remain ≥ 5 in top-10 with this candidate set.
    """
    await seed_r1a_gold_comparator_cases()
    arts: list[dict] = []
    for i in range(6):
        arts.append({
            "pmid": f"PNX-{i}",
            "doi": f"10.1/pnx{i}",
            "title": f"Incidence {i} of pneumothorax after central venous catheterization",
            "abstract": (
                "Prospective cohort of central venous catheter pneumothorax. "
                "Risk factors for procedure-related pneumothorax in ICU."
            ),
            "journal": "Crit Care Med",
            "publication_year": 2024,
            "source": "pubmed",
            "publication_types_json": ["Prospective Cohort Study"],
        })
    for i in range(2):
        arts.append({
            "pmid": f"WRONG-{i}",
            "doi": f"10.1/w{i}",
            "title": f"Deep vein thrombosis {i} after central venous catheter placement",
            "abstract": "DVT risk after CVC.",
            "journal": "J Vasc",
            "publication_year": 2024,
            "source": "pubmed",
            "publication_types_json": ["Review"],
        })

    from services.r1c_shadow_simulator import replay_benchmark_case_r1c
    fetch = _fetch_stub(records_for_broad=arts, records_for_hp=arts)
    result = await replay_benchmark_case_r1c(
        case_id="r1a-case-D-cvc-pneumothorax",
        actor_user_id="admin-1",
        fetch_fn=fetch,
        recovery_fn=_recovery_stub_no_op(),
    )
    sidecar_id = result["sidecar_doc_id"]
    db = get_db()
    row = await db[Collections.R1C_SHADOW_DECISIONS].find_one(
        {"id": sidecar_id}, {"_id": 0},
    )
    assert row["procedure_complication_outcome_exact_count"] >= 5, (
        f"outcome_exact dropped: {row['procedure_complication_outcome_exact_count']}"
    )


# =====================================================================
# 6. Case C source-coverage marker — closed-set miss reason.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_1_caseC_obokata_records_not_indexed_when_crossref_used():
    """When Crossref returns metadata but PubMed/EPMC don't surface
    the article via title, miss reason is the closed-set
    ``not_indexed_in_pubmed_or_epmc``.
    """
    from services.r1c_gold_recovery import attempt_recovery

    async def _empty(*args, **kw):  # noqa: ARG001
        return []

    async def _crossref_used_empty(*args, **kw):  # noqa: ARG001
        return True, []  # crossref_used=True but no matched records

    async def _pag(*args, **kw):  # noqa: ARG001
        return [], 1, 1, False

    result = await attempt_recovery(
        gold_dois=["10.1016/j.jcmg.2018.12.034"],
        gold_title="Obokata",
        broad_query_string="amyloid cardiomyopathy",
        expansion_map_name="amyloid_cardiomyopathy_mechanism",
        fetch_overrides={
            "exact_title": _empty,
            "doi_anchor_term": _empty,
            "crossref_metadata_lookup": _crossref_used_empty,
            "deeper_pagination": _pag,
            "synonym_expansion": _empty,
        },
    )
    assert result["crossref_used"] is True
    assert (
        result["connector_recall_miss_reason"]
        == "not_indexed_in_pubmed_or_epmc"
    )


# =====================================================================
# 7. PubMed eSummary UID iteration fix.
# =====================================================================
class _StubAsyncClient:
    def __init__(self, body):
        self._body = body

    async def get(self, *args, **kwargs):
        return _StubResponse(self._body)


class _StubResponse:
    def __init__(self, body):
        self._body = body
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


@pytest.mark.asyncio
async def test_R1C_1_pubmed_summary_strips_uids_list_entry():
    """PubMed eSummary returns ``result.uids`` (list of PMIDs) alongside
    one dict per PMID. The fix MUST filter the list-valued entries.
    """
    from services.r1c_gold_recovery import _pubmed_summary
    body = {
        "result": {
            "uids": ["123", "456"],
            "123": {"title": "T1", "articleids": [
                {"idtype": "doi", "value": "10.1/a"}
            ]},
            "456": {"title": "T2", "articleids": [
                {"idtype": "doi", "value": "10.1/b"}
            ]},
        }
    }
    client = _StubAsyncClient(body)
    out = await _pubmed_summary(client, ["123", "456"])
    assert set(out.keys()) == {"123", "456"}
    for v in out.values():
        assert isinstance(v, dict)


@pytest.mark.asyncio
async def test_R1C_1_pubmed_summary_empty_response():
    from services.r1c_gold_recovery import _pubmed_summary
    body = {"result": {}}
    client = _StubAsyncClient(body)
    out = await _pubmed_summary(client, ["123"])
    assert out == {}


@pytest.mark.asyncio
async def test_R1C_1_pubmed_summary_single_uid_only_list():
    """eSummary with only the uids list (no per-PMID dicts) yields
    an empty result dict — no crash.
    """
    from services.r1c_gold_recovery import _pubmed_summary
    body = {"result": {"uids": ["999"]}}
    client = _StubAsyncClient(body)
    out = await _pubmed_summary(client, ["999"])
    assert out == {}


@pytest.mark.asyncio
async def test_R1C_1_pubmed_summary_malformed_response():
    """Missing ``result`` key entirely → empty dict (defensive)."""
    from services.r1c_gold_recovery import _pubmed_summary
    body = {"some_other_key": "garbage"}
    client = _StubAsyncClient(body)
    out = await _pubmed_summary(client, ["1"])
    assert out == {}


@pytest.mark.asyncio
async def test_R1C_1_pubmed_summary_multi_with_mixed_value_types():
    """eSummary with mixed dict + non-dict entries → only dicts kept."""
    from services.r1c_gold_recovery import _pubmed_summary
    body = {
        "result": {
            "uids": ["1", "2", "3"],
            "1": {"title": "A"},
            "2": "unexpected-string",
            "3": {"title": "B"},
            "noise_list": [1, 2, 3],
        }
    }
    client = _StubAsyncClient(body)
    out = await _pubmed_summary(client, ["1", "2", "3"])
    assert set(out.keys()) == {"1", "3"}


# =====================================================================
# 8. End-to-end recovery — synonym_expansion no longer raises.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_1_recovery_end_to_end_no_uid_crash():
    """Integration: run attempt_recovery with a synonym_expansion that
    would previously have hit the UID list crash. Closed-set: it
    completes without raising and records the miss reason.
    """
    from services.r1c_gold_recovery import attempt_recovery

    async def _empty(*args, **kw):  # noqa: ARG001
        return []

    async def _crossref_empty(*args, **kw):  # noqa: ARG001
        return False, []

    async def _pag(*args, **kw):  # noqa: ARG001
        return [], 1, 1, False

    async def _synonym_ok(*args, **kw):  # noqa: ARG001
        return []

    result = await attempt_recovery(
        gold_dois=["10.1/x"], gold_title="y",
        broad_query_string="foo",
        expansion_map_name="cholestatic_liver_disease_mechanism",
        fetch_overrides={
            "exact_title": _empty,
            "doi_anchor_term": _empty,
            "crossref_metadata_lookup": _crossref_empty,
            "deeper_pagination": _pag,
            "synonym_expansion": _synonym_ok,
        },
    )
    assert "synonym_expansion" in result["recovery_strategies_attempted"]
    assert result["recovery_strategy_that_surfaced_gold"] is None
    assert (
        result["connector_recall_miss_reason"]
        == "connector_returned_empty_for_doi"
    )


# =====================================================================
# 9. K4A invariance — non-eligible shape refused.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_1_k4a_routes_still_refused():
    from services.r1c_shadow_simulator import (
        R1CReplayError,
        replay_benchmark_case_r1c,
    )
    db = get_db()
    K4A_CASE_ID = "test-k4a-synth-r1c1"
    await db[Collections.R1A_GOLD_COMPARATOR_CASES].update_one(
        {"case_id": K4A_CASE_ID},
        {"$set": {
            "case_id": K4A_CASE_ID,
            "question": "synthetic K4A question",
            "shape": "severity_marker_overview",
            "gold_article_dois": [],
            "gold_article_pmids": [],
        }},
        upsert=True,
    )
    try:
        async def _empty(*args, **kw):  # noqa: ARG001
            return []
        with pytest.raises(R1CReplayError) as exc_info:
            await replay_benchmark_case_r1c(
                case_id=K4A_CASE_ID,
                actor_user_id="admin-1",
                fetch_fn=_empty,
            )
        assert exc_info.value.code == "shape_not_eligible_for_r1c"
    finally:
        await db[Collections.R1A_GOLD_COMPARATOR_CASES].delete_one(
            {"case_id": K4A_CASE_ID},
        )


# =====================================================================
# 10. M3.2 dormancy unchanged after R1C.1 runs.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_1_does_not_flip_M3_2_kill_switch():
    from services.m3_2_feature_flags import read_kill_switch_state
    before = await read_kill_switch_state()
    # touch r1c.1 path
    _ = compute_recency_base_score(year=2024, shape="pathophysiology_mechanism")
    after = await read_kill_switch_state()
    assert before == after


# =====================================================================
# 11. Leak-scan on R1C.1 persisted sidecar.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_1_sidecar_does_not_leak_abstract_or_credentials():
    await seed_r1a_gold_comparator_cases()
    fetch = _fetch_stub([
        {
            "pmid": "LK1",
            "doi": "10.1/lk1",
            "title": "Mechanisms of pruritus in cholestasis",
            "abstract": "This abstract MUST NOT leak via r1c.1 sidecar; bile acids, autotaxin.",
            "journal": "J",
            "publication_year": 2023,
            "source": "pubmed",
            "publication_types_json": ["Review"],
        },
    ])
    from services.r1c_shadow_simulator import replay_benchmark_case_r1c
    result = await replay_benchmark_case_r1c(
        case_id="r1a-case-A-cholestatic-pruritus",
        actor_user_id="admin-1",
        fetch_fn=fetch,
        recovery_fn=_recovery_stub_no_op(),
    )
    db = get_db()
    row = await db[Collections.R1C_SHADOW_DECISIONS].find_one(
        {"id": result["sidecar_doc_id"]}, {"_id": 0},
    )
    flat = str(row).lower()
    assert "this abstract must not leak" not in flat
    for forbidden in ("system:", "assistant:", "sk-", "bearer ", "password", "x-api-key"):
        assert forbidden not in flat


# =====================================================================
# 12. R1B invariance — configuration unchanged.
# =====================================================================
def test_R1C_1_does_not_mutate_R1B_configuration():
    from services.r1b_shadow_simulator import (
        BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER,
        HIGH_PRECISION_SHAPES,
        R1B_SOURCES,
    )
    from services.r1b_query_expansion import (
        CURATED_EXPANSION_MAPS,
        R1B_ELIGIBLE_SHAPES,
    )
    assert BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER == 0.5
    assert HIGH_PRECISION_SHAPES == frozenset({
        "procedure_complication_risk", "procedure_complication",
    })
    assert R1B_SOURCES == ("europe_pmc", "pubmed")
    assert set(CURATED_EXPANSION_MAPS.keys()) == {
        "cholestatic_liver_disease_mechanism",
        "amyloid_cardiomyopathy_mechanism",
        "pcos_insulin_resistance_mechanism",
        "central_line_pneumothorax_complication",
        "pneumothorax_outcome",
        "procedure_complication_boost",
    }
    assert R1B_ELIGIBLE_SHAPES == frozenset({
        "pathophysiology_mechanism",
        "procedure_complication_risk",
        "procedure_complication",
    })
