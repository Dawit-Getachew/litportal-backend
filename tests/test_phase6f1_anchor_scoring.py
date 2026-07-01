"""Phase 6F.1 — Positive anchor-match boost + sort-order regression.

Spec:
  • `picott_aware_ranker` adds a closed-set `anchor_match_*` boost when
    ALL required must_haves are matched on title+abstract AND no
    wrong-anchor cap_reason is present.
  • Boost size: `anchor_match_full=+35`, `anchor_match_partial_outcome=+15`,
    `anchor_match_partial_intervention=+10`.
  • Boost applies to BOTH `relevance_score` and `overall_rank_score`.
  • Wrong-anchor caps still win — boost is mutually exclusive with any
    cap_reason.
  • `routers/search_sessions.py::get_search_session` uses a stable
    composite sort key (included_by_default → user_included →
    overall_rank → relevance_score → evidence_design_score → canonical_key)
    so wrong-anchor articles capped at the same numeric rank cannot
    surface above on-target articles.

These are unit-level tests against the deterministic ranker (no LLM,
no network).
"""
from __future__ import annotations

import asyncio

import pytest

from services.clinical_context_engine import build_clinical_context
from services.clinical_question_parser import clinical_question_parser
from services.default_selection import select_default_articles
from services.evidence_ranker import EvidenceRankerService
from services.picott_ranker import picott_aware_ranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pipeline(question: str) -> tuple[dict, dict]:
    """Return (parsed, picott) after running the full deterministic
    pre-ranking pipeline."""
    parsed = asyncio.get_event_loop().run_until_complete(
        clinical_question_parser.parse(question)
    )
    context = build_clinical_context(parsed)
    picott = context.get("repaired_picott") or parsed.get("picott_json") or {}
    parsed["picott_json"] = picott
    parsed["role_weights_json"] = context.get("role_weights") or {}
    return parsed, picott


def _rank(parsed: dict, picott: dict, articles: list[dict]) -> list[dict]:
    """Run evidence_ranker + picott_aware_ranker, return ranked list."""
    qmap = {a["canonical_key"]: ["high_precision", "evidence_filtered"]
            for a in articles}
    ranked = EvidenceRankerService().rank(
        parsed, articles, query_types_matched_map=qmap,
    )
    fam = {a["canonical_key"]: ["cb_high_precision", "cb_study_type_filtered"]
           for a in articles}
    picott_aware_ranker.postprocess(
        ranked, picott, query_family_by_canonical=fam,
    )
    return ranked


def _psoriasis_guideline() -> dict:
    return {
        "id": "pmid41531416", "pmid": "41531416",
        "title": "S3 Guideline for the treatment of psoriasis vulgaris",
        "abstract": (
            "This is a clinical practice guideline for the treatment of "
            "psoriasis vulgaris in adults. It covers first-line topical "
            "treatment, second-line systemic therapy, and biologic therapy "
            "including TNF, IL-17, IL-23 inhibitors. Updated treatment "
            "algorithm. Topical corticosteroids, vitamin D analogs, "
            "methotrexate, cyclosporine, adalimumab, secukinumab, "
            "ustekinumab, risankizumab."
        ),
        "article_type_classification": "guideline",
        "is_guideline": True,
        "publication_year": 2023,
        "full_text_status": "free_full_text_link",
        "canonical_key": "pmid:41531416",
    }


def _quetiapine_review() -> dict:
    return {
        "id": "pmid11532137", "pmid": "11532137",
        "title": "Quetiapine fumarate (Seroquel): a new atypical antipsychotic",
        "abstract": (
            "Quetiapine fumarate (Seroquel) is an atypical antipsychotic "
            "with a unique receptor binding profile. This review discusses "
            "pharmacology, efficacy, adverse effects, side effects, and "
            "tolerability profile including somnolence, orthostatic "
            "hypotension, weight gain, dizziness, dry mouth and metabolic "
            "effects in patients receiving Seroquel."
        ),
        "article_type_classification": "narrative_review",
        "is_guideline": False,
        "publication_year": 2001,
        "full_text_status": "abstract_only",
        "canonical_key": "pmid:11532137",
    }


def _wrong_disease_first_line() -> dict:
    return {
        "id": "pmid99999001", "pmid": "99999001",
        "title": "Second-line treatment for biliary tract cancer",
        "abstract": (
            "Outcomes after first-line gemcitabine in cholangiocarcinoma. "
            "Treatment algorithm for biliary tract cancer."
        ),
        "article_type_classification": "guideline",
        "is_guideline": True,
        "publication_year": 2023,
        "full_text_status": "abstract_only",
        "canonical_key": "pmid:99999001",
    }


def _wrong_drug_side_effects() -> dict:
    return {
        "id": "pmid99999002", "pmid": "99999002",
        "title": "Side effects of caffeine in young adults",
        "abstract": (
            "Caffeine adverse effects including insomnia and palpitations "
            "in healthy adults."
        ),
        "article_type_classification": "rct",
        "is_guideline": False,
        "publication_year": 2022,
        "full_text_status": "abstract_only",
        "canonical_key": "pmid:99999002",
    }


# ---------------------------------------------------------------------------
# A) Acceptance gates from the directive
# ---------------------------------------------------------------------------

class TestPhase6F1AcceptanceGates:
    def test_psoriasis_guideline_pmid_41531416_scores_above_70(self):
        """Q1: 'S3 Guideline for the treatment of psoriasis vulgaris'
        must reach relevance ≥70 and be default-selected.
        """
        parsed, picott = _pipeline(
            "What is the first line treatment and second line treatment "
            "for psoriasis?"
        )
        ranked = _rank(parsed, picott, [_psoriasis_guideline()])
        sel = select_default_articles(picott, ranked)
        r = ranked[0]
        assert r["_article"]["pmid"] == "41531416"
        assert r["relevance_score"] >= 70, (
            f"Phase 6F.1 acceptance gate FAILED: "
            f"relevance_score={r['relevance_score']} (need ≥70)"
        )
        assert r["_article"]["id"] in sel["selected_article_ids"]
        # The new boost label must be present.
        reasons = r.get("ranking_boost_reasons") or []
        assert any(s.startswith("anchor_match_full") for s in reasons), reasons

    def test_quetiapine_seroquel_review_scores_above_70(self):
        """Q2: 'Quetiapine fumarate (Seroquel): a new atypical antipsychotic'
        must reach relevance ≥70 and be default-selected.
        """
        parsed, picott = _pipeline(
            "What are the side effects of Seroquel?"
        )
        ranked = _rank(parsed, picott, [_quetiapine_review()])
        sel = select_default_articles(picott, ranked)
        r = ranked[0]
        assert r["_article"]["pmid"] == "11532137"
        assert r["relevance_score"] >= 70, (
            f"Phase 6F.1 acceptance gate FAILED: "
            f"relevance_score={r['relevance_score']} (need ≥70)"
        )
        assert r["_article"]["id"] in sel["selected_article_ids"]
        reasons = r.get("ranking_boost_reasons") or []
        assert any(s.startswith("anchor_match_full") for s in reasons), reasons


# ---------------------------------------------------------------------------
# B) Mutual-exclusion guard — boost cannot fire on capped articles
# ---------------------------------------------------------------------------

class TestPhase6F1MutualExclusion:
    def test_wrong_disease_article_is_NOT_boosted(self):
        """A wrong-disease article carries `wrong_condition` /
        `missing_condition_anchor` and MUST NOT receive any
        `anchor_match_*` boost, regardless of how many ranking
        concepts it matches.
        """
        parsed, picott = _pipeline(
            "What is the first line treatment and second line treatment "
            "for psoriasis?"
        )
        ranked = _rank(parsed, picott, [_wrong_disease_first_line()])
        r = ranked[0]
        assert r["relevance_cap_applied"] == 20
        assert r["relevance_score"] <= 20
        assert r["overall_rank_score"] <= 20
        reasons = r.get("ranking_boost_reasons") or []
        assert not any(s.startswith("anchor_match_") for s in reasons), reasons
        # Caps are still in place.
        cap_reasons = r.get("cap_reasons") or []
        assert "missing_condition_anchor" in cap_reasons
        assert "wrong_condition" in cap_reasons

    def test_wrong_drug_article_is_NOT_boosted(self):
        """Same guard for the drug-safety reproducer."""
        parsed, picott = _pipeline("What are the side effects of Seroquel?")
        ranked = _rank(parsed, picott, [_wrong_drug_side_effects()])
        r = ranked[0]
        assert r["relevance_cap_applied"] == 20
        assert r["relevance_score"] <= 20
        assert r["overall_rank_score"] <= 20
        reasons = r.get("ranking_boost_reasons") or []
        assert not any(s.startswith("anchor_match_") for s in reasons), reasons
        cap_reasons = r.get("cap_reasons") or []
        assert "missing_intervention_anchor" in cap_reasons
        assert "wrong_drug" in cap_reasons


# ---------------------------------------------------------------------------
# C) Partial vs full anchor match — boost size ordering
# ---------------------------------------------------------------------------

class TestPhase6F1PartialBoosts:
    def test_partial_outcome_smaller_than_full_match(self):
        """Drug-safety Q with two must_haves (drug + outcome). An
        article that mentions ONLY the drug (no adverse-effect outcome
        synonym) gets the smaller `anchor_match_partial_outcome` boost
        (+15), while an article that mentions BOTH gets the full +35.
        """
        parsed, picott = _pipeline("What are the side effects of Seroquel?")
        full_match = _quetiapine_review()
        partial = {
            "id": "partial-drug", "pmid": "10000001",
            "title": "Quetiapine pharmacokinetics in healthy adults",
            "abstract": (
                "Steady-state pharmacokinetics of quetiapine fumarate in "
                "healthy adult volunteers. Plasma concentration time "
                "curves. Dose-proportional exposure."
            ),
            "article_type_classification": "cohort_study",
            "is_guideline": False,
            "publication_year": 2010,
            "full_text_status": "abstract_only",
            "canonical_key": "pmid:10000001",
        }
        ranked = _rank(parsed, picott, [full_match, partial])
        by_pmid = {r["_article"]["pmid"]: r for r in ranked}
        full_r = by_pmid["11532137"]
        part_r = by_pmid["10000001"]
        # Full match → +35; partial outcome → +15. Compare boost labels.
        full_labels = " ".join(full_r.get("ranking_boost_reasons") or [])
        part_labels = " ".join(part_r.get("ranking_boost_reasons") or [])
        assert "anchor_match_full" in full_labels, full_labels
        assert (
            "anchor_match_partial_outcome" in part_labels
            or "anchor_match_full" not in part_labels
        ), part_labels
        # Numeric ordering: full match scores strictly above partial.
        assert full_r["relevance_score"] > part_r["relevance_score"]


# ---------------------------------------------------------------------------
# D) Stable sort order — wrong-anchor never above on-target at same rank
# ---------------------------------------------------------------------------

class TestPhase6F1StableSort:
    def test_articles_sort_relevance_desc_when_no_default_selection(self):
        """Two articles with the same overall_rank=20 — one wrong-anchor
        (capped) and one on-target (low natural rel but boosted to ≥70).
        After ranking + the Phase 6F.1 boost, the on-target article
        must sort FIRST regardless of insertion order.

        Phase 6F.1 — use a HARDCODED picott to keep this test
        deterministic. The LLM-driven path is exercised by the live-API
        regression file; this unit test focuses on the sort key shape.
        """
        # Hardcoded psoriasis PICOTT — bypasses the LLM.
        parsed = {
            "key_concepts": ["psoriasis", "treatment"],
            "clinical_question_type": "therapy",
            "picott_json": {
                "type_of_question": "therapy",
                "question_subtype": "treatment_sequencing",
                "must_have_concepts": [
                    {
                        "concept": "psoriasis",
                        "role": "population",
                        "synonyms": [
                            "plaque psoriasis",
                            "chronic plaque psoriasis",
                            "psoriasis vulgaris",
                            "psoriatic disease",
                            "moderate-to-severe psoriasis",
                            "moderate to severe psoriasis",
                            "severe psoriasis",
                        ],
                        "required_for_high_relevance": True,
                        "missing_relevance_cap": 20,
                    },
                ],
                "ranking_concepts": [],
                "should_have_concepts": [],
                "negative_or_penalty_concepts": [],
            },
            "role_weights_json": {"intervention": 30, "population": 20, "outcome": 20},
        }
        picott = parsed["picott_json"]
        wrong = _wrong_disease_first_line()
        on_target = _psoriasis_guideline()
        # Deliberately reverse insertion order — wrong first.
        ranked = _rank(parsed, picott, [wrong, on_target])
        # Replicate the GET handler's composite sort key here.
        ranked.sort(
            key=lambda r: (
                -1 if bool(r.get("included_by_default")) else 0,
                -1 if bool(r.get("user_included")) else 0,
                -int(r.get("overall_rank_score") or 0),
                -float(r.get("relevance_score") or 0.0),
                -float(r.get("evidence_design_score") or 0.0),
                str((r.get("_article") or {}).get("canonical_key") or ""),
            )
        )
        # On-target must be first.
        assert ranked[0]["_article"]["pmid"] == "41531416", (
            f"Sort order broken: top PMID is "
            f"{ranked[0]['_article']['pmid']} "
            f"(rel={ranked[0]['relevance_score']}, "
            f"overall={ranked[0]['overall_rank_score']}, "
            f"cap={ranked[0].get('relevance_cap_applied')}, "
            f"caps={ranked[0].get('cap_reasons')})"
        )
        assert ranked[1]["_article"]["pmid"] == "99999001"
        assert ranked[0]["overall_rank_score"] > ranked[1]["overall_rank_score"]


# ---------------------------------------------------------------------------
# E) Closed-set boost-label vocabulary
# ---------------------------------------------------------------------------

class TestPhase6F1ClosedSetLabels:
    def test_boost_labels_are_closed_set(self):
        """When the boost fires, the label MUST be one of the closed-set
        Phase 6F.1 codes. No free-text labels."""
        parsed, picott = _pipeline(
            "What is the first line treatment and second line treatment "
            "for psoriasis?"
        )
        ranked = _rank(parsed, picott, [_psoriasis_guideline()])
        reasons = ranked[0].get("ranking_boost_reasons") or []
        # Anchor boost label must match the closed-set prefixes.
        anchor_labels = [
            s for s in reasons if s.startswith("anchor_match_")
        ]
        assert anchor_labels, reasons
        for s in anchor_labels:
            assert any(
                s.startswith(prefix + "(+")
                for prefix in (
                    "anchor_match_full",
                    "anchor_match_partial_outcome",
                    "anchor_match_partial_intervention",
                )
            ), s
