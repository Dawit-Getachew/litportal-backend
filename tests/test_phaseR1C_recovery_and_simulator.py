"""Phase R1C — gold-recovery + shadow simulator + admin endpoint
integration tests.

Mocks the per-strategy fetch functions so tests are deterministic
and never hit PubMed / EPMC / Crossref.
"""
from __future__ import annotations

import pytest

from core.db import Collections, get_db
from services.r1a_gold_comparator_cases import seed_r1a_gold_comparator_cases
from services.r1c_gold_recovery import (
    MAX_PAGES,
    PAGE_SIZE,
    RECALL_MISS_REASONS,
    RECOVERY_STRATEGIES,
    attempt_recovery,
)
from services.r1c_shadow_simulator import (
    R1CReplayError,
    list_r1c_shadow_decisions,
    replay_benchmark_case_r1c,
)


# =====================================================================
# Closed-set constants.
# =====================================================================
def test_R1C_recovery_strategy_names_closed_set():
    assert RECOVERY_STRATEGIES == (
        "exact_title", "doi_anchor_term", "crossref_metadata_lookup",
        "deeper_pagination", "synonym_expansion",
    )


def test_R1C_recall_miss_reasons_closed_set():
    assert RECALL_MISS_REASONS == frozenset({
        "not_indexed_in_pubmed_or_epmc",
        "connector_pagination_ceiling_exceeded_500_results",
        "connector_returned_empty_for_doi",
        "unknown_classify_manually",
    })


def test_R1C_pagination_knobs_match_operator_decision_2a():
    # Operator confirmed pagesize=100, max_pages=5 (= 500 cap).
    assert PAGE_SIZE == 100
    assert MAX_PAGES == 5


# =====================================================================
# Recovery — strategy order + first-success short-circuit.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_recovery_exact_title_short_circuits():
    """When exact_title returns a match, the simulator does NOT
    proceed to subsequent strategies.
    """
    called = {k: 0 for k in RECOVERY_STRATEGIES}
    matched_doi = "10.1/found"

    async def _exact(*args, **kw):  # noqa: ARG001
        called["exact_title"] += 1
        return [{"pmid": "P1", "doi": matched_doi, "title": "x"}]

    async def _other(*args, **kw):  # noqa: ARG001
        # If called, this would be a bug.
        called.setdefault("OTHER", 0)
        called["OTHER"] += 1
        return []

    async def _crossref(*args, **kw):  # noqa: ARG001
        called["crossref_metadata_lookup"] += 1
        return False, []

    async def _pag(*args, **kw):  # noqa: ARG001
        called["deeper_pagination"] += 1
        return [], 0, 0, False

    result = await attempt_recovery(
        gold_dois=[matched_doi], gold_title="exact title",
        broad_query_string="foo", expansion_map_name=None,
        fetch_overrides={
            "exact_title": _exact,
            "doi_anchor_term": _other,
            "crossref_metadata_lookup": _crossref,
            "deeper_pagination": _pag,
            "synonym_expansion": _other,
        },
    )
    assert result["recovery_strategy_that_surfaced_gold"] == "exact_title"
    assert result["recovery_strategies_attempted"] == ["exact_title"]
    assert called["exact_title"] == 1
    # No other strategy executed.
    assert called.get("OTHER", 0) == 0
    assert called["crossref_metadata_lookup"] == 0
    assert called["deeper_pagination"] == 0


@pytest.mark.asyncio
async def test_R1C_recovery_falls_through_to_doi_anchor_term():
    matched_doi = "10.1/d"

    async def _exact(*args, **kw):  # noqa: ARG001
        return []

    async def _doi(*args, **kw):  # noqa: ARG001
        return [{"pmid": "PD", "doi": matched_doi, "title": "via doi"}]

    result = await attempt_recovery(
        gold_dois=[matched_doi], gold_title="",
        broad_query_string="foo", expansion_map_name=None,
        fetch_overrides={
            "exact_title": _exact,
            "doi_anchor_term": _doi,
        },
    )
    assert result["recovery_strategy_that_surfaced_gold"] == "doi_anchor_term"
    assert result["recovery_strategies_attempted"] == [
        "exact_title", "doi_anchor_term",
    ]


@pytest.mark.asyncio
async def test_R1C_recovery_records_pagination_ceiling_exceeded():
    """When every strategy fails AND deeper_pagination used the
    full 5-page allowance, the miss reason is closed-set
    ``connector_pagination_ceiling_exceeded_500_results``.
    """
    async def _empty(*args, **kw):  # noqa: ARG001
        return []

    async def _crossref_empty(*args, **kw):  # noqa: ARG001
        return False, []

    async def _pag_max(*args, **kw):  # noqa: ARG001
        return [], MAX_PAGES, MAX_PAGES, True  # ceiling exceeded

    result = await attempt_recovery(
        gold_dois=["10.1/x"], gold_title="t",
        broad_query_string="foo", expansion_map_name=None,
        fetch_overrides={
            "exact_title": _empty,
            "doi_anchor_term": _empty,
            "crossref_metadata_lookup": _crossref_empty,
            "deeper_pagination": _pag_max,
            "synonym_expansion": _empty,
        },
    )
    assert result["recovery_strategy_that_surfaced_gold"] is None
    assert (
        result["connector_recall_miss_reason"]
        == "connector_pagination_ceiling_exceeded_500_results"
    )
    assert result["pagination_pages_used"]["pubmed"] == MAX_PAGES
    assert result["pagination_pages_used"]["europe_pmc"] == MAX_PAGES


@pytest.mark.asyncio
async def test_R1C_recovery_records_connector_returned_empty_for_doi():
    """When Crossref short-circuits (mailto unset) AND every PubMed/EPMC
    strategy returned empty, the miss reason is
    ``connector_returned_empty_for_doi``.
    """
    async def _empty(*args, **kw):  # noqa: ARG001
        return []

    async def _crossref_empty(*args, **kw):  # noqa: ARG001
        return False, []  # crossref_used=False

    async def _pag(*args, **kw):  # noqa: ARG001
        return [], 1, 1, False  # not ceiling exceeded

    result = await attempt_recovery(
        gold_dois=["10.1/x"], gold_title="t",
        broad_query_string="foo", expansion_map_name=None,
        fetch_overrides={
            "exact_title": _empty,
            "doi_anchor_term": _empty,
            "crossref_metadata_lookup": _crossref_empty,
            "deeper_pagination": _pag,
            "synonym_expansion": _empty,
        },
    )
    assert result["recovery_strategy_that_surfaced_gold"] is None
    # Crossref was NOT used (mailto unset stub), and DOI was provided →
    # ``connector_returned_empty_for_doi``.
    assert (
        result["connector_recall_miss_reason"]
        == "connector_returned_empty_for_doi"
    )
    assert result["crossref_used"] is False


@pytest.mark.asyncio
async def test_R1C_recovery_records_not_indexed_when_crossref_used_but_no_match():
    """Crossref returned metadata (i.e., DOI exists in Crossref's
    index) but PubMed/EPMC still couldn't find the article → not
    indexed in PubMed/EPMC.
    """
    async def _empty(*args, **kw):  # noqa: ARG001
        return []

    async def _crossref_used_empty(*args, **kw):  # noqa: ARG001
        return True, []  # crossref_used=True but no PubMed/EPMC match

    async def _pag(*args, **kw):  # noqa: ARG001
        return [], 1, 1, False

    result = await attempt_recovery(
        gold_dois=["10.1/x"], gold_title="t",
        broad_query_string="foo", expansion_map_name=None,
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


@pytest.mark.asyncio
async def test_R1C_recovery_deeper_pagination_surfaces_beuers():
    """Case A simulation: deeper_pagination strategy returns Beuers
    after pages 1-2 — recovery records ``deeper_pagination`` as
    the strategy that surfaced the gold.
    """
    BEUERS = {
        "pmid": "BEUERS",
        "doi": "10.1038/s41575-022-00687-7",
        "title": "Pathogenesis and treatment of pruritus in cholestasis",
        "journal": "Nat Rev Gastroenterol Hepatol",
        "publication_year": "2022",
        "source": "pubmed",
        "publication_types_json": ["Review"],
    }

    async def _empty(*args, **kw):  # noqa: ARG001
        return []

    async def _empty_crossref(*args, **kw):  # noqa: ARG001
        return False, []

    async def _pag_finds(*args, **kw):  # noqa: ARG001
        return [BEUERS], 2, 0, False

    result = await attempt_recovery(
        gold_dois=["10.1038/s41575-022-00687-7"],
        gold_title=BEUERS["title"],
        broad_query_string="cholestasis OR pruritus",
        expansion_map_name="cholestatic_liver_disease_mechanism",
        fetch_overrides={
            "exact_title": _empty,
            "doi_anchor_term": _empty,
            "crossref_metadata_lookup": _empty_crossref,
            "deeper_pagination": _pag_finds,
        },
    )
    assert (
        result["recovery_strategy_that_surfaced_gold"] == "deeper_pagination"
    )
    assert result["matched_records"]
    assert result["pagination_pages_used"]["pubmed"] == 2


# =====================================================================
# No-fabrication test.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_recovery_does_not_fabricate_pmid_or_doi():
    """Recovery returns ONLY records that came back from the
    connector stubs. No synthetic identifiers are introduced.
    """
    seen: dict[str, str] = {}

    async def _exact(*args, **kw):  # noqa: ARG001
        rec = {
            "pmid": "STUB-PMID",
            "doi": "10.1/stub",
            "title": "stub title",
        }
        seen["stub"] = "from-fetch-fn"
        return [rec]

    result = await attempt_recovery(
        gold_dois=["10.1/stub"], gold_title="stub title",
        broad_query_string="foo", expansion_map_name=None,
        fetch_overrides={"exact_title": _exact},
    )
    pmids_returned = {r.get("pmid") for r in result["matched_records"]}
    assert pmids_returned == {"STUB-PMID"}  # came from fetch_fn, not fabricated
    # Sanity: the fetch fn was the only source of the identifier.
    assert seen["stub"] == "from-fetch-fn"


# =====================================================================
# Shadow simulator integration.
# =====================================================================
async def _ensure_gold_cases():
    await seed_r1a_gold_comparator_cases()


def _fetch_stub(records_for_broad, records_for_hp=None):
    async def _fn(*, query_string, search_session_id=None):  # noqa: ARG001
        if records_for_hp is not None and " AND " in (query_string or ""):
            return list(records_for_hp)
        return list(records_for_broad)
    return _fn


def _recovery_stub(*, found_records=None, strategy=None, miss_reason=None,
                  pages=None):
    async def _fn(**kwargs):  # noqa: ARG001
        return {
            "recovery_strategies_attempted": [strategy] if strategy else [],
            "recovery_strategy_that_surfaced_gold": strategy,
            "matched_records": list(found_records or []),
            "crossref_used": False,
            "pagination_pages_used": pages or {"pubmed": 0, "europe_pmc": 0},
            "connector_recall_miss_reason": miss_reason,
            "pagination_ceiling_exceeded": (
                miss_reason == "connector_pagination_ceiling_exceeded_500_results"
            ),
        }
    return _fn


@pytest.mark.asyncio
async def test_R1C_replay_unknown_case_raises():
    await _ensure_gold_cases()
    with pytest.raises(R1CReplayError) as exc_info:
        await replay_benchmark_case_r1c(
            case_id="does-not-exist",
            actor_user_id="admin-1",
            fetch_fn=_fetch_stub([]),
            recovery_fn=_recovery_stub(),
        )
    assert exc_info.value.code == "unknown_case_id"


@pytest.mark.asyncio
async def test_R1C_caseD_high_precision_wrong_outcome_demoted():
    """Case D: direct pneumothorax-after-CVC article outranks a
    DVT-after-CVC article (wrong outcome → 0.3× multiplier).
    """
    await _ensure_gold_cases()
    direct = {
        "pmid": "DIRECT",
        "doi": "10.1/d",
        "title": "Incidence of pneumothorax after central venous catheterization",
        "abstract": (
            "Prospective cohort of central venous catheter pneumothorax. "
            "Risk factors for procedure-related pneumothorax in ICU."
        ),
        "journal": "Crit Care Med",
        "publication_year": 2024,
        "source": "pubmed",
        "publication_types_json": ["Prospective Cohort Study"],
    }
    wrong = {
        "pmid": "WRONG",
        "doi": "10.1/w",
        "title": "Deep vein thrombosis after central venous catheter placement",
        "abstract": "DVT risk after CVC.",
        "journal": "J Vasc",
        "publication_year": 2024,
        "source": "pubmed",
        "publication_types_json": ["Review"],
    }
    fetch = _fetch_stub(records_for_broad=[direct, wrong], records_for_hp=[direct, wrong])
    result = await replay_benchmark_case_r1c(
        case_id="r1a-case-D-cvc-pneumothorax",
        actor_user_id="admin-1",
        fetch_fn=fetch,
        recovery_fn=_recovery_stub(),
    )
    sidecar_id = result["sidecar_doc_id"]
    db = get_db()
    row = await db[Collections.R1C_SHADOW_DECISIONS].find_one(
        {"id": sidecar_id}, {"_id": 0},
    )
    assert row is not None
    pmids = [r["pmid"] for r in row["r1c_simulated_top_n_metadata"]]
    assert pmids.index("DIRECT") < pmids.index("WRONG"), pmids
    # Closed-set counters.
    assert row["procedure_complication_outcome_exact_count"] >= 1
    assert row["procedure_complication_wrong_outcome_count"] >= 1


@pytest.mark.asyncio
async def test_R1C_caseA_mechanism_target_missing_demoted():
    """Case A: a mechanism-target-exact article outranks a
    disease-only-indirect article.
    """
    await _ensure_gold_cases()
    direct = {
        "pmid": "MECH",
        "doi": "10.1/m",
        "title": "Mechanism of pruritus in cholestatic liver disease",
        "abstract": "Autotaxin and lysophosphatidic acid mediate itch pathway.",
        "journal": "Nat Rev Gastroenterol Hepatol",
        "publication_year": 2024,
        "source": "pubmed",
        "publication_types_json": ["Review"],
    }
    disease_only = {
        "pmid": "DISEASE",
        "doi": "10.1/d",
        "title": "Approach to cholestatic liver disease",
        "abstract": "Overview of clinical features and diagnostic workup.",
        "journal": "J Hepatol",
        "publication_year": 2024,
        "source": "pubmed",
        "publication_types_json": ["Review"],
    }
    fetch = _fetch_stub([direct, disease_only])
    result = await replay_benchmark_case_r1c(
        case_id="r1a-case-A-cholestatic-pruritus",
        actor_user_id="admin-1",
        fetch_fn=fetch,
        recovery_fn=_recovery_stub(),
    )
    sidecar_id = result["sidecar_doc_id"]
    db = get_db()
    row = await db[Collections.R1C_SHADOW_DECISIONS].find_one(
        {"id": sidecar_id}, {"_id": 0},
    )
    pmids = [r["pmid"] for r in row["r1c_simulated_top_n_metadata"]]
    assert pmids.index("MECH") < pmids.index("DISEASE")
    assert row["mechanism_target_exact_count"] >= 1
    assert row["disease_only_indirect_for_mechanism_count"] >= 1


@pytest.mark.asyncio
async def test_R1C_caseB_safety_check_does_not_worsen_treatment_overrank():
    """PCOS Case B safety check — R1C treatment overrank counter
    must be ≤ R1B counter for the same candidate set.
    """
    await _ensure_gold_cases()
    # Mixed candidate set with two treatment trials + one mechanism review.
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
    fetch = _fetch_stub([trt1, trt2, mech])
    result = await replay_benchmark_case_r1c(
        case_id="r1a-case-B-pcos-insulin-resistance",
        actor_user_id="admin-1",
        fetch_fn=fetch,
        recovery_fn=_recovery_stub(),
    )
    sidecar_id = result["sidecar_doc_id"]
    db = get_db()
    row = await db[Collections.R1C_SHADOW_DECISIONS].find_one(
        {"id": sidecar_id}, {"_id": 0},
    )
    # Closed-set acceptance: R1C must NOT increase the treatment
    # overrank vs R1B for the same candidate set.
    assert (
        row["treatment_overrank_count_r1c"]
        <= row["treatment_overrank_count_r1b"]
    ), row


# =====================================================================
# Mixed-signal review preserved (no penalty).
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_mechanism_review_with_treatment_preserves_rank():
    """``mechanism_review_with_treatment_discussion`` is flag-only;
    multiplier is 1.0 so the article keeps its R1A-derived rank.
    """
    await _ensure_gold_cases()
    review = {
        "pmid": "REV",
        "doi": "10.1/rev",
        "title": "Review of cholestatic pruritus: mechanisms and itch pathway",
        "abstract": (
            "Comprehensive review of pathophysiology including treatment "
            "trials of IBAT inhibitors."
        ),
        "journal": "Nat Rev Gastroenterol Hepatol",
        "publication_year": 2024,
        "source": "pubmed",
        "publication_types_json": ["Review"],
    }
    fetch = _fetch_stub([review])
    result = await replay_benchmark_case_r1c(
        case_id="r1a-case-A-cholestatic-pruritus",
        actor_user_id="admin-1",
        fetch_fn=fetch,
        recovery_fn=_recovery_stub(),
    )
    sidecar_id = result["sidecar_doc_id"]
    db = get_db()
    row = await db[Collections.R1C_SHADOW_DECISIONS].find_one(
        {"id": sidecar_id}, {"_id": 0},
    )
    assert row["mechanism_review_with_treatment_discussion_count"] >= 1


# =====================================================================
# Leak-scan + closed-set field exclusion.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_sidecar_contains_no_abstract_field():
    await _ensure_gold_cases()
    fetch = _fetch_stub([
        {
            "pmid": "PA",
            "doi": "10.1/pa",
            "title": "Mechanism of pruritus in cholestatic liver disease",
            "abstract": "This abstract MUST NOT leak through the R1C sidecar.",
            "journal": "J",
            "publication_year": 2024,
            "source": "pubmed",
            "publication_types_json": ["Review"],
        },
    ])
    result = await replay_benchmark_case_r1c(
        case_id="r1a-case-A-cholestatic-pruritus",
        actor_user_id="admin-1",
        fetch_fn=fetch,
        recovery_fn=_recovery_stub(),
    )
    db = get_db()
    row = await db[Collections.R1C_SHADOW_DECISIONS].find_one(
        {"id": result["sidecar_doc_id"]}, {"_id": 0},
    )
    flat = str(row).lower()
    assert "this abstract must not leak" not in flat
    for forbidden in ("system:", "assistant:", "sk-", "bearer ", "password"):
        assert forbidden not in flat


# =====================================================================
# Listing.
# =====================================================================
@pytest.mark.asyncio
async def test_R1C_list_returns_most_recent_first():
    await _ensure_gold_cases()
    fetch = _fetch_stub([])
    a1 = await replay_benchmark_case_r1c(
        case_id="r1a-case-A-cholestatic-pruritus",
        actor_user_id="admin-1",
        fetch_fn=fetch,
        recovery_fn=_recovery_stub(),
    )
    a2 = await replay_benchmark_case_r1c(
        case_id="r1a-case-A-cholestatic-pruritus",
        actor_user_id="admin-1",
        fetch_fn=fetch,
        recovery_fn=_recovery_stub(),
    )
    rows = await list_r1c_shadow_decisions(limit=10)
    ids = [r["id"] for r in rows]
    assert a2["sidecar_doc_id"] in ids
    assert a1["sidecar_doc_id"] in ids
    assert ids.index(a2["sidecar_doc_id"]) < ids.index(a1["sidecar_doc_id"])
