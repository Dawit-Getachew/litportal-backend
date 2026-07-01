"""Phase R1B — shadow simulator + admin endpoint integration tests.

Mocks the literature retrieval ``fetch_fn`` so tests are
deterministic, do not hit PubMed/EPMC, and run fast.
"""
from __future__ import annotations

import pytest

from core.db import Collections, get_db
from services.r1a_gold_comparator_cases import seed_r1a_gold_comparator_cases
from services.r1b_shadow_simulator import (
    BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER,
    HIGH_PRECISION_SHAPES,
    R1B_PER_ARTICLE_FIELDS,
    R1BReplayError,
    list_r1b_shadow_decisions,
    replay_benchmark_case_r1b,
)


# =====================================================================
# Closed-set constants.
# =====================================================================
def test_R1B_high_precision_shapes_closed_set():
    assert HIGH_PRECISION_SHAPES == frozenset({
        "procedure_complication_risk",
        "procedure_complication",
    })


def test_R1B_per_article_fields_contract():
    assert set(R1B_PER_ARTICLE_FIELDS) == {
        "pmid", "doi", "title", "journal", "year", "source",
        "current_rank", "simulated_rank", "simulated_score",
        "candidate_provenance", "label_set",
    }


def test_R1B_broad_family_penalty_constant_is_lte_one():
    # The penalty is a closed-set multiplier ≤ 1 so it can only
    # deprioritize, not boost.
    assert 0.0 <= BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER <= 1.0
    assert BROAD_FAMILY_ONLY_PENALTY_MULTIPLIER < 1.0


# =====================================================================
# Helpers.
# =====================================================================
async def _ensure_gold_cases():
    await seed_r1a_gold_comparator_cases()


def _make_fetch(records_for_broad, records_for_hp=None):
    """Returns an async fetch_fn that maps the two query strings
    to the closed-set canned record lists.
    """
    counter = {"hits": 0}

    async def _fn(*, query_string: str, search_session_id=None):  # noqa: ARG001
        counter["hits"] += 1
        # The broad family runs FIRST in the simulator; the HP family
        # runs second when present. Each call gets the matching list.
        if records_for_hp is not None and " AND " in (query_string or ""):
            return list(records_for_hp)
        return list(records_for_broad)

    _fn.counter = counter  # type: ignore[attr-defined]
    return _fn


# =====================================================================
# Replay — unknown case + ineligible shape.
# =====================================================================
@pytest.mark.asyncio
async def test_R1B_replay_unknown_case_raises():
    await _ensure_gold_cases()
    with pytest.raises(R1BReplayError) as exc_info:
        await replay_benchmark_case_r1b(
            case_id="does-not-exist",
            actor_user_id="admin-1",
            fetch_fn=_make_fetch([]),
        )
    assert exc_info.value.code == "unknown_case_id"


# =====================================================================
# Case A — Beuers DOI surfaces in R1B top-N when expansion brings it
# in, but not in active.
# =====================================================================
@pytest.mark.asyncio
async def test_R1B_caseA_gold_enters_r1b_top_n_when_expansion_surfaces_it():
    await _ensure_gold_cases()
    beuers = {
        "pmid": "BEUERS",
        "doi": "10.1038/s41575-022-00687-7",
        "title": "Pathogenesis and treatment of pruritus in cholestasis",
        "abstract": (
            "Lysophosphatidic acid and autotaxin in the molecular "
            "pathogenesis of cholestatic pruritus."
        ),
        "journal": "Nat Rev Gastroenterol Hepatol",
        "publication_year": 2022,
        "publication_types_json": ["Review"],
        "source": "pubmed",
    }
    other = {
        "pmid": "OTHER",
        "doi": "10.1/other",
        "title": "Random cholestatic disease review",
        "abstract": "Overview content.",
        "journal": "Gen Hep",
        "publication_year": 2024,
        "publication_types_json": ["Review"],
        "source": "pubmed",
    }
    fetch_fn = _make_fetch([beuers, other])
    result = await replay_benchmark_case_r1b(
        case_id="r1a-case-A-cholestatic-pruritus",
        actor_user_id="admin-1",
        fetch_fn=fetch_fn,
    )
    assert result["sidecar_doc_id"]
    assert result["case_id"] == "r1a-case-A-cholestatic-pruritus"
    assert result["shape"] == "pathophysiology_mechanism"
    # Gold rank in R1B arm: Beuers carries
    # ``pathophysiology_mechanism_direct`` + ``molecular_pathogenesis``
    # so it should outrank the generic review.
    assert result["gold_comparator_rank_r1b"] == 1
    # Pull the sidecar and check the field contract.
    sidecar_id = result["sidecar_doc_id"]
    db = get_db()
    row = await db[Collections.R1B_SHADOW_DECISIONS].find_one(
        {"id": sidecar_id}, {"_id": 0},
    )
    assert row is not None
    assert row["benchmark_case_id"] == "r1a-case-A-cholestatic-pruritus"
    assert row["shape"] == "pathophysiology_mechanism"
    assert "expansion_concept_map_used" in row
    assert row["expansion_concept_map_used"] == "cholestatic_liver_disease_mechanism"
    # Beuers must be in r1b_simulated_top_n.
    found = [
        r for r in row["r1b_simulated_top_n_metadata"]
        if r.get("doi") == "10.1038/s41575-022-00687-7"
    ]
    assert len(found) == 1
    # No abstract / prompt / completion leakage.
    assert "abstract" not in str(row).lower() or "abstract_only" in str(row).lower()
    # Per-article contract.
    art_keys = set(found[0].keys())
    assert art_keys == set(R1B_PER_ARTICLE_FIELDS)


# =====================================================================
# Case C — Obokata DOI similar test.
# =====================================================================
@pytest.mark.asyncio
async def test_R1B_caseC_gold_enters_r1b_top_n_when_expansion_surfaces_it():
    await _ensure_gold_cases()
    obokata = {
        "pmid": "OBOKATA",
        "doi": "10.1016/j.jcmg.2018.12.034",
        "title": (
            "Mechanism of myocardial stiffness and impaired relaxation "
            "in HFpEF with cardiac amyloidosis"
        ),
        "abstract": (
            "Pathophysiology and molecular pathogenesis of diastolic "
            "dysfunction in transthyretin amyloid cardiomyopathy."
        ),
        "journal": "JACC Cardiovasc Imaging",
        "publication_year": 2018,
        "publication_types_json": ["Review"],
        "source": "pubmed",
    }
    irrelevant = {
        "pmid": "TX",
        "doi": "10.1/tx",
        "title": "Treatment trial of tafamidis in ATTR-CM",
        "abstract": "Double blind placebo controlled trial of a treatment.",
        "journal": "NEJM",
        "publication_year": 2024,
        "publication_types_json": ["Randomized Controlled Trial"],
        "source": "pubmed",
    }
    fetch_fn = _make_fetch([obokata, irrelevant])
    result = await replay_benchmark_case_r1b(
        case_id="r1a-case-C-amyloid-diastolic-dysfunction",
        actor_user_id="admin-1",
        fetch_fn=fetch_fn,
    )
    assert result["gold_comparator_rank_r1b"] == 1


# =====================================================================
# Case D — high-precision family activation + broad-family-only
# deprioritization.
# =====================================================================
@pytest.mark.asyncio
async def test_R1B_caseD_high_precision_family_activates_and_deprioritizes_broad_only():
    await _ensure_gold_cases()
    # Direct procedure-complication article — in both broad AND
    # high-precision streams.
    direct = {
        "pmid": "DIRECT",
        "doi": "10.1/d",
        "title": "Incidence of pneumothorax after central venous catheterization",
        "abstract": (
            "Prospective cohort of central venous catheter placement "
            "complications including pneumothorax in critical care."
        ),
        "journal": "Crit Care Med",
        "publication_year": 2023,
        "publication_types_json": ["Prospective Cohort Study"],
        "source": "pubmed",
    }
    # Chest x-ray utility paper — broad-family-only (does NOT match
    # high-precision (procedure AND outcome) constraint).
    chest_xray = {
        "pmid": "XRAY",
        "doi": "10.1/x",
        "title": "Routine chest radiograph after central line placement utility",
        "abstract": "Post-procedure imaging utility for line confirmation.",
        "journal": "Radiology",
        "publication_year": 2024,
        "publication_types_json": [],
        "source": "pubmed",
    }
    # Broad family returns BOTH; HP family returns only direct.
    fetch_fn = _make_fetch(
        records_for_broad=[direct, chest_xray],
        records_for_hp=[direct],
    )
    result = await replay_benchmark_case_r1b(
        case_id="r1a-case-D-cvc-pneumothorax",
        actor_user_id="admin-1",
        fetch_fn=fetch_fn,
    )
    sidecar_id = result["sidecar_doc_id"]
    db = get_db()
    row = await db[Collections.R1B_SHADOW_DECISIONS].find_one(
        {"id": sidecar_id}, {"_id": 0},
    )
    assert row is not None
    assert row["high_precision_family_required_both_anchors"] is True
    # Provenance counts.
    pc = row["candidate_provenance_counts"]
    assert pc["both"] == 1, pc
    assert pc["broad_family_only"] == 1, pc
    assert pc["high_precision_family_only"] == 0, pc
    # Direct should outrank chest_xray (broad-family-only is penalized).
    pmids = [
        r["pmid"] for r in row["r1b_simulated_top_n_metadata"]
    ]
    assert pmids.index("DIRECT") < pmids.index("XRAY"), pmids


# =====================================================================
# Case B — runs and writes a sidecar; verdict signal documented as
# OUT OF SCOPE for R1B uplift (PICO parse misread).
# =====================================================================
@pytest.mark.asyncio
async def test_R1B_caseB_expansion_runs_and_records_out_of_scope():
    await _ensure_gold_cases()
    fetch_fn = _make_fetch([
        {
            "pmid": "PCOSPATHO",
            "doi": "10.1/pcos",
            "title": "Insulin signaling and ovarian steroidogenesis in PCOS",
            "abstract": "Pathophysiology of insulin resistance and hyperandrogenism.",
            "journal": "JCEM",
            "publication_year": 2024,
            "publication_types_json": ["Review"],
            "source": "pubmed",
        },
    ])
    result = await replay_benchmark_case_r1b(
        case_id="r1a-case-B-pcos-insulin-resistance",
        actor_user_id="admin-1",
        fetch_fn=fetch_fn,
    )
    assert result["sidecar_doc_id"]
    assert result["case_id"] == "r1a-case-B-pcos-insulin-resistance"


# =====================================================================
# No-LLM invariant — assert the simulator does NOT touch LiteLLM.
# =====================================================================
@pytest.mark.asyncio
async def test_R1B_simulator_does_not_invoke_llm(monkeypatch):
    """If anything in the R1B path imported litellm, assert no calls."""
    import sys
    if "litellm" not in sys.modules:
        return  # nothing to track
    import litellm
    called = {"count": 0}

    def _stub(*args, **kwargs):  # noqa: ARG001
        called["count"] += 1
        raise RuntimeError("R1B should not call litellm")

    if hasattr(litellm, "completion"):
        monkeypatch.setattr(litellm, "completion", _stub)
    if hasattr(litellm, "acompletion"):
        monkeypatch.setattr(litellm, "acompletion", _stub)
    await _ensure_gold_cases()
    fetch_fn = _make_fetch([])
    await replay_benchmark_case_r1b(
        case_id="r1a-case-A-cholestatic-pruritus",
        actor_user_id="admin-1",
        fetch_fn=fetch_fn,
    )
    assert called["count"] == 0


# =====================================================================
# Leak-scan — no abstracts / no PHI in sidecar.
# =====================================================================
@pytest.mark.asyncio
async def test_R1B_sidecar_contains_no_abstract_field():
    await _ensure_gold_cases()
    fetch_fn = _make_fetch([
        {
            "pmid": "P1",
            "doi": "10.1/p",
            "title": "Mechanism review",
            "abstract": "This abstract MUST NOT leak through the sidecar.",
            "journal": "J",
            "publication_year": 2023,
            "publication_types_json": ["Review"],
            "source": "pubmed",
        },
    ])
    result = await replay_benchmark_case_r1b(
        case_id="r1a-case-A-cholestatic-pruritus",
        actor_user_id="admin-1",
        fetch_fn=fetch_fn,
    )
    db = get_db()
    row = await db[Collections.R1B_SHADOW_DECISIONS].find_one(
        {"id": result["sidecar_doc_id"]}, {"_id": 0},
    )
    flat = str(row).lower()
    assert "this abstract must not leak" not in flat
    for forbidden in (
        "system:", "assistant:", "sk-", "bearer ", "password",
    ):
        assert forbidden not in flat


# =====================================================================
# Listing endpoint helper.
# =====================================================================
@pytest.mark.asyncio
async def test_R1B_list_returns_most_recent_first():
    await _ensure_gold_cases()
    fetch_fn = _make_fetch([])
    a1 = await replay_benchmark_case_r1b(
        case_id="r1a-case-A-cholestatic-pruritus",
        actor_user_id="admin-1",
        fetch_fn=fetch_fn,
    )
    a2 = await replay_benchmark_case_r1b(
        case_id="r1a-case-A-cholestatic-pruritus",
        actor_user_id="admin-1",
        fetch_fn=fetch_fn,
    )
    rows = await list_r1b_shadow_decisions(limit=10)
    ids = [r["id"] for r in rows]
    # Both must be present; the second insert should appear earlier
    # in the list (most-recent first).
    assert a2["sidecar_doc_id"] in ids
    assert a1["sidecar_doc_id"] in ids
    assert ids.index(a2["sidecar_doc_id"]) < ids.index(a1["sidecar_doc_id"])
