"""Phase 5D round-3 — Fix 3, Fix 4, Fix 5 regression tests.

Round-2 of the tester reported these three contracts as still broken:
  - Fix 3: DEV_FORCE_ENABLE_PUBMED startup hook didn't flip the DB row
  - Fix 4: STOP_PHRASES only wired into Phase 5D, legacy builder still
           leaked single-token stopwords ("tell", "about", "me")
  - Fix 5: ranking_boost_total exceeded the +10 cap

These tests pin each contract.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.concept_block_query_builder import concept_block_query_builder  # noqa: E402
from services.picott_ranker import picott_aware_ranker  # noqa: E402
from services.query_compiler import boolean_query_compiler  # noqa: E402
from services.query_stop_phrases import (  # noqa: E402
    STOP_TOKENS,
    is_query_term_meaningful,
    strip_stop_phrases,
)


# =============================================================================
# Fix 4 — STOP_PHRASES + STOP_TOKENS wired into BOTH builders.
# =============================================================================

class TestFix4_StopTokensAcrossBothBuilders:
    """Single-token stopwords like 'tell', 'about', 'me' must NEVER
    appear as a search term, regardless of which builder produced it."""

    # The exact tokens reported by the tester: "tell", "about", "me" plus
    # representative members of the broader STOP_TOKENS set.
    _LEAK_TOKENS = ["tell", "about", "me", "what", "the", "is", "in", "of"]

    def _tell_me_about_picott(self) -> dict:
        """Synthetic PICOTT that mimics an LLM mis-extraction of
        ``tell me about diabetes`` — the parser hallucinates the
        connective tokens as concepts/synonyms."""
        return {
            "type_of_question": "background",
            "must_have_concepts": [
                {
                    "concept": "diabetes",
                    "role": "condition",
                    "synonyms": ["tell", "about", "me"],
                },
            ],
            "ranking_concepts": [
                {"concept": "tell me about", "role": "context"},
                {"concept": "about", "role": "context"},
                {"concept": "me", "role": "context"},
            ],
            "pico": {
                "population": ["tell", "me"],
                "outcomes_or_target_diagnosis": ["about"],
            },
        }

    def _ami_difference_picott(self) -> dict:
        """A 'difference between X and Y' AMI question."""
        return {
            "type_of_question": "therapy",
            "must_have_concepts": [
                {
                    "concept": "difference between liberal and restrictive transfusion",
                    "role": "intervention",
                    "synonyms": ["comparison of strategies"],
                },
                {"concept": "acute myocardial infarction", "role": "condition"},
            ],
            "ranking_concepts": [
                {"concept": "comparing strategies", "role": "context"},
            ],
            "pico": {},
        }

    def _assert_no_leak(self, queries, *, source_q: str):
        for q in queries:
            for tok in self._LEAK_TOKENS:
                # Look for the term as a quoted single-token search term.
                # PubMed: ``"tell"[tiab]`` / Europe PMC: ``TITLE_ABS:"tell"``
                # We also reject any quoted occurrence of these single tokens.
                leaked = (
                    f'"{tok}"[tiab]' in q.query_string
                    or f'TITLE_ABS:"{tok}"' in q.query_string
                    or f'"{tok}"[MeSH Terms]' in q.query_string
                )
                assert not leaked, (
                    f"STOP_TOKEN '{tok}' leaked into "
                    f"[{q.source}/{q.query_family}] for source_question="
                    f"{source_q!r}: {q.query_string!r}"
                )

    def test_legacy_builder_tell_me_about_no_leak(self):
        picott = self._tell_me_about_picott()
        legacy = concept_block_query_builder.build(
            picott, sources=["pubmed", "europe_pmc"],
        )
        # Legacy must still emit at least one query (with "diabetes").
        assert legacy, "legacy must emit at least one query"
        self._assert_no_leak(legacy, source_q="tell me about diabetes")
        # Every emitted query must reference the real anchor.
        diabetes_seen = any("diabetes" in q.query_string for q in legacy)
        assert diabetes_seen, "diabetes anchor must survive the filter"

    def test_phase5d_compiler_tell_me_about_no_leak(self):
        picott = self._tell_me_about_picott()
        queries = boolean_query_compiler.compile(
            picott, sources=["pubmed", "europe_pmc"],
        )
        self._assert_no_leak(queries, source_q="tell me about diabetes")

    def test_legacy_builder_ami_difference_no_leak(self):
        picott = self._ami_difference_picott()
        legacy = concept_block_query_builder.build(
            picott, sources=["pubmed", "europe_pmc"],
        )
        assert legacy, "legacy must emit at least one query"
        # No "difference between" phrase, no single-token leaks.
        for q in legacy:
            assert "difference between" not in q.query_string.lower()
            self._assert_no_leak([q], source_q="AMI difference question")

    def test_phase5d_compiler_ami_difference_no_leak(self):
        picott = self._ami_difference_picott()
        queries = boolean_query_compiler.compile(
            picott, sources=["pubmed", "europe_pmc"],
        )
        for q in queries:
            assert "difference between" not in q.query_string.lower()
            self._assert_no_leak([q], source_q="AMI difference question")

    def test_combined_builders_zero_stopword_emission(self):
        """Strong end-to-end assertion: across both builders, both
        sources, all 9 families, the noise PICOTT produces ZERO single-
        token stopword emissions."""
        picott = self._tell_me_about_picott()
        legacy = concept_block_query_builder.build(
            picott, sources=["pubmed", "europe_pmc"],
        )
        phase5d = boolean_query_compiler.compile(
            picott, sources=["pubmed", "europe_pmc"],
        )
        all_q = list(legacy) + list(phase5d)
        self._assert_no_leak(all_q, source_q="tell me about diabetes")

    def test_is_query_term_meaningful_unit_contract(self):
        """Single-token stopwords are not meaningful."""
        for tok in ["tell", "me", "about", "what", "the"]:
            assert not is_query_term_meaningful(tok), tok
        # Real medical terms are meaningful.
        for term in ["diabetes", "MINT", "milrinone", "D-dimer", "HFpEF"]:
            assert is_query_term_meaningful(term), term
        # Multi-token phrases that contain at least one real concept survive.
        assert is_query_term_meaningful("about diabetes")
        assert is_query_term_meaningful("tell me about diabetes")

    def test_stop_tokens_includes_tester_reported_words(self):
        """Closed-set contract: the exact tokens the tester saw leak
        must be in STOP_TOKENS."""
        for tok in ("tell", "me", "about"):
            assert tok in STOP_TOKENS, f"{tok!r} missing from STOP_TOKENS"


# =============================================================================
# Fix 5 — ranking_boost_total capped at +10 (uncapped reasons preserved).
# =============================================================================

class TestFix5_RankingBoostCappedAtTen:
    """Per-article boost MUST be clamped to +10. The reasons list keeps
    every fired reason (including the synthetic ``boost_total_capped_at_10``
    breadcrumb) so admins can see the un-clamped value for diagnostics."""

    def _theoretical_20_boost_input(self) -> tuple[dict, list, dict]:
        """Build an input that would, pre-fix, accumulate 20 boost points:
          +8 from 4 matched ranking concepts
          +5 from study-design (RCT design pathway is +10 → //2 = +5)
          +5 from high-signal family with satisfied must-haves
          = 18 (the closest reachable; theoretical 20 needs 5 ranking
          concepts; the exact total is fine as long as it's >10).
        """
        picott = {
            "type_of_question": "therapy",
            "must_have_concepts": [
                {"concept": "milrinone", "role": "intervention"},
                {"concept": "subarachnoid hemorrhage", "role": "condition"},
            ],
            "ranking_concepts": [
                {"concept": "cerebral vasospasm", "role": "outcome"},
                {"concept": "delayed cerebral ischemia", "role": "outcome"},
                {"concept": "mortality", "role": "outcome"},
                {"concept": "neurological outcome", "role": "outcome"},
            ],
            "pico": {},
        }
        # Article matches every ranking concept and is an RCT — pre-cap
        # total would be 8 + 5 + 5 = 18.
        article = {
            "canonical_key": "k1",
            "title": (
                "Milrinone for cerebral vasospasm and delayed cerebral "
                "ischemia after subarachnoid hemorrhage — RCT"
            ),
            "abstract": (
                "Randomized trial of milrinone in adults with subarachnoid "
                "hemorrhage. Outcomes: cerebral vasospasm, delayed cerebral "
                "ischemia, mortality, neurological outcome at 90 days."
            ),
            "article_type_classification": "rct",
        }
        ranked = [{
            "_article": article,
            "title": article["title"],
            "abstract": article["abstract"],
            "relevance_score": 80,
            "evidence_design_score": 95,
            "overall_rank_score": 80,
            "rank_explanation": "test",
            "query_types_matched": [
                "cb_high_precision", "cb_mesh_keyword", "cb_clinical_queries",
            ],
        }]
        qmap = {"k1": ["cb_high_precision", "cb_mesh_keyword",
                       "cb_clinical_queries"]}
        return picott, ranked, qmap

    def test_ranking_boost_total_capped_at_ten(self):
        picott, ranked, qmap = self._theoretical_20_boost_input()
        out = picott_aware_ranker.postprocess(
            ranked, picott, query_family_by_canonical=qmap,
        )
        r = out[0]
        # The cap is the contract — total must never exceed +10.
        assert r["ranking_boost_total"] <= 10, (
            f"boost_total exceeded cap: {r['ranking_boost_total']}"
        )

    def test_uncapped_reasons_preserved_when_clamped(self):
        """Even when the total is clamped, ALL reason codes that fired
        must remain in ``ranking_boost_reasons`` PLUS a synthetic
        ``boost_total_capped_at_10(was_N)`` breadcrumb."""
        picott, ranked, qmap = self._theoretical_20_boost_input()
        out = picott_aware_ranker.postprocess(
            ranked, picott, query_family_by_canonical=qmap,
        )
        r = out[0]
        reasons = r["ranking_boost_reasons"]
        # Multiple boost reasons must remain.
        assert len(reasons) >= 3, (
            f"expected ≥3 boost reasons before clamp; got {reasons!r}"
        )
        # And the cap breadcrumb must be present with the un-clamped value.
        capped_codes = [c for c in reasons if c.startswith("boost_total_capped_at_10")]
        assert capped_codes, (
            f"expected boost_total_capped_at_10 breadcrumb; got {reasons!r}"
        )
        # The breadcrumb must include the un-clamped integer.
        breadcrumb = capped_codes[0]
        assert "was_" in breadcrumb, breadcrumb

    def test_no_cap_breadcrumb_when_total_within_cap(self):
        """For an article whose un-clamped total is ≤10, no breadcrumb
        is added (clean diagnostics)."""
        picott = {
            "type_of_question": "therapy",
            "must_have_concepts": [
                {"concept": "metformin", "role": "intervention"},
                {"concept": "diabetes", "role": "condition"},
            ],
            # No ranking concepts → no concept-match boost.
            "ranking_concepts": [],
            "pico": {},
        }
        article = {
            "canonical_key": "k1",
            "title": "Metformin in T2D — observational cohort",
            # Article type with NO design boost (case_report → 0).
            "abstract": "Cohort study in adults with diabetes.",
            "article_type_classification": "cohort_study",
        }
        ranked = [{
            "_article": article,
            "title": article["title"],
            "abstract": article["abstract"],
            "relevance_score": 70,
            "evidence_design_score": 60,
            "overall_rank_score": 70,
            "rank_explanation": "test",
            "query_types_matched": ["cb_high_precision"],
        }]
        out = picott_aware_ranker.postprocess(
            ranked, picott, query_family_by_canonical={"k1": ["cb_high_precision"]},
        )
        r = out[0]
        assert r["ranking_boost_total"] <= 10
        # No "boost_total_capped_at_10" breadcrumb when within budget.
        assert not any(
            c.startswith("boost_total_capped_at_10")
            for c in r["ranking_boost_reasons"]
        ), r["ranking_boost_reasons"]

    def test_anchor_cap_still_applies_when_must_haves_missing(self):
        """Phase 5A invariant: a missing-intervention-anchor cap of 30
        is NOT lifted by the boost. (Boost is applied to overall_rank_score
        but capped via the lowest-wins anchor cap.)"""
        picott = {
            "type_of_question": "therapy",
            "must_have_concepts": [
                {"concept": "milrinone", "role": "intervention"},
                {"concept": "subarachnoid hemorrhage", "role": "condition"},
            ],
            "ranking_concepts": [
                {"concept": "cerebral vasospasm", "role": "outcome"},
            ],
            "pico": {},
        }
        # Article DOES NOT mention milrinone — missing intervention anchor.
        article = {
            "canonical_key": "k1",
            "title": "Cerebral vasospasm after subarachnoid hemorrhage",
            "abstract": (
                "Review of cerebral vasospasm following aneurysmal "
                "subarachnoid hemorrhage. No intervention discussed."
            ),
            "article_type_classification": "review",
        }
        ranked = [{
            "_article": article,
            "title": article["title"],
            "abstract": article["abstract"],
            "relevance_score": 90,
            "evidence_design_score": 60,
            "overall_rank_score": 90,
            "rank_explanation": "test",
            "query_types_matched": ["cb_broad_recall"],
        }]
        out = picott_aware_ranker.postprocess(
            ranked, picott, query_family_by_canonical={"k1": ["cb_broad_recall"]},
        )
        r = out[0]
        # Cap must be ≤ 30 since intervention anchor missing.
        assert r["overall_rank_score"] <= 30, (
            f"missing-must-have cap violated: {r['overall_rank_score']}"
        )


# =============================================================================
# Fix 3 — closed-set log lines for the DEV_FORCE_ENABLE_PUBMED hook.
# =============================================================================

class TestFix3_DevForceEnablePubmedHookLogs:
    """Contract: the hook MUST emit closed-set log lines on every
    startup. Verified by re-invoking ``_seed_source_configs`` directly
    (since lifespan startup already ran in the test process)."""

    @pytest.mark.asyncio
    async def test_hook_emits_attempted_and_succeeded_log_lines(self, caplog):
        """Direct call to the seeder verifies the log lines fire."""
        import logging
        from server import _seed_source_configs

        caplog.set_level(logging.INFO, logger="clinevidence")
        with patch(
            "server.settings.DEV_FORCE_ENABLE_PUBMED", True,
        ), patch(
            "server.settings.IS_PRODUCTION", False,
        ):
            await _seed_source_configs()

        msgs = [r.getMessage() for r in caplog.records]
        attempted = [m for m in msgs if "phase5d_dev_force_enable_pubmed_hook_attempted" in m]
        succeeded_or_failed = [
            m for m in msgs
            if "phase5d_dev_force_enable_pubmed_hook_succeeded" in m
            or "phase5d_dev_force_enable_pubmed_hook_failed_reason=" in m
        ]
        assert attempted, (
            f"expected phase5d_dev_force_enable_pubmed_hook_attempted log; "
            f"got {msgs!r}"
        )
        assert succeeded_or_failed, (
            f"expected phase5d_dev_force_enable_pubmed_hook_succeeded OR "
            f"_failed_reason log; got {msgs!r}"
        )

    @pytest.mark.asyncio
    async def test_hook_does_not_fire_in_production(self, caplog):
        """Closed-set contract: when ``IS_PRODUCTION=True``, the hook
        logs ``_failed_reason=is_production_env`` and does NOT modify
        the row."""
        import logging
        from server import _seed_source_configs

        caplog.set_level(logging.INFO, logger="clinevidence")
        with patch(
            "server.settings.DEV_FORCE_ENABLE_PUBMED", True,
        ), patch(
            "server.settings.IS_PRODUCTION", True,
        ):
            await _seed_source_configs()
        msgs = [r.getMessage() for r in caplog.records]
        failed = [
            m for m in msgs
            if "phase5d_dev_force_enable_pubmed_hook_failed_reason=is_production_env" in m
        ]
        assert failed, (
            f"expected production failed_reason log; got {msgs!r}"
        )

    @pytest.mark.asyncio
    async def test_pubmed_enabled_after_hook_in_dev(self):
        """End-to-end contract: after the dev hook runs, the pubmed row
        must have ``enabled=True``."""
        from core.db import get_db, Collections
        from server import _seed_source_configs

        with patch(
            "server.settings.DEV_FORCE_ENABLE_PUBMED", True,
        ), patch(
            "server.settings.IS_PRODUCTION", False,
        ):
            await _seed_source_configs()
        row = await get_db()[Collections.SOURCE_CONFIGS].find_one(
            {"source_name": "pubmed"}, {"_id": 0, "enabled": 1},
        )
        assert row is not None
        assert row["enabled"] is True, (
            f"pubmed row enabled state after hook: {row!r}"
        )
