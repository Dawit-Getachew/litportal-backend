"""Phase 5F Group E — SemanticReranker.article_answer_class + GLP-1/LDL
ranking integration.

Covers:
- direct: GLP-1 RA + lipid outcome in human study → ``direct``.
- related: GLP-1 + cardiovascular but no lipid section → ``related``
  (handled via wrong_outcome when intervention/outcome mismatch).
- mechanistic: in-vitro / receptor signaling content → ``mechanistic``.
- wrong_outcome: GLP-1 without lipid outcome.
- wrong_intervention: lipid article without GLP-1.
- closed-set enum membership.
- ranking integration: direct article ranks higher than mechanistic /
  wrong_outcome / wrong_intervention.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.semantic_reranker import (  # noqa: E402
    ARTICLE_ANSWER_CLASSES,
    classify_article_answer,
    semantic_reranker,
)


def _glp1_lipid_picott() -> dict:
    return {
        "normalized_question": (
            "how do GLP-1 receptor agonists impact LDL-C in patients with type 2 diabetes?"
        ),
        "type_of_question": "pharmacology",
        "question_subtype": "metabolic_biomarker_effect",
        "must_have_concepts": [
            {"concept": "GLP-1 receptor agonists", "role": "intervention",
             "synonyms": ["semaglutide", "liraglutide", "dulaglutide"]},
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
    }


# =============================================================================
# Group E1 — closed-set enum
# =============================================================================

class TestArticleAnswerClass_ClosedSet:

    def test_enum_members(self):
        assert ARTICLE_ANSWER_CLASSES == (
            "direct", "related", "mechanistic",
            "wrong_population", "wrong_intervention", "wrong_outcome",
            "background_only",
        )

    def test_classifier_always_returns_enum_member(self):
        picott = _glp1_lipid_picott()
        cases = [
            ("", ""),
            ("Semaglutide and LDL-C", "Semaglutide reduced LDL cholesterol."),
            ("Beta-cell signaling", "GLP-1 receptor signaling in mice."),
            ("Atorvastatin and LDL", "Statin reduced LDL."),
        ]
        for title, abstract in cases:
            cls = classify_article_answer(
                title=title, abstract=abstract, picott=picott,
            )
            assert cls in ARTICLE_ANSWER_CLASSES, cls


# =============================================================================
# Group E2 — GLP-1/LDL specialized classifier behaviour
# =============================================================================

class TestArticleAnswerClass_GLP1Lipid:

    def test_direct_glp1_plus_lipid_in_human_study(self):
        picott = _glp1_lipid_picott()
        cls = classify_article_answer(
            title="Effect of semaglutide on LDL-C in type 2 diabetes",
            abstract=(
                "Semaglutide reduced LDL cholesterol over 26 weeks in a "
                "randomized trial of adults with type 2 diabetes."
            ),
            picott=picott,
        )
        assert cls == "direct", cls

    def test_wrong_outcome_glp1_without_lipid(self):
        picott = _glp1_lipid_picott()
        cls = classify_article_answer(
            title="Semaglutide and HbA1c",
            abstract="Semaglutide reduced HbA1c in type 2 diabetes.",
            picott=picott,
        )
        # No lipid outcome term mentioned → wrong_outcome.
        assert cls == "wrong_outcome", cls

    def test_wrong_intervention_lipid_without_glp1(self):
        picott = _glp1_lipid_picott()
        cls = classify_article_answer(
            title="Atorvastatin lowers LDL-C",
            abstract=(
                "Atorvastatin reduced LDL cholesterol by 50% vs placebo "
                "in 200 patients."
            ),
            picott=picott,
        )
        assert cls == "wrong_intervention", cls

    def test_mechanistic_glp1_receptor_signaling(self):
        picott = _glp1_lipid_picott()
        cls = classify_article_answer(
            title="GLP-1 receptor signaling in pancreatic beta cells",
            abstract=(
                "In vitro experiments showed GLP-1 receptor activation "
                "in mouse beta cell lines."
            ),
            picott=picott,
        )
        assert cls == "mechanistic", cls

    def test_background_no_relevant_terms(self):
        picott = _glp1_lipid_picott()
        cls = classify_article_answer(
            title="A history of insulin discovery",
            abstract="The discovery of insulin in 1921 was a landmark.",
            picott=picott,
        )
        assert cls == "background_only", cls


# =============================================================================
# Group E3 — Integration with ranker output
# =============================================================================

class TestSemanticRerankerIntegration:

    def test_rerank_adds_article_answer_class_field(self):
        picott = _glp1_lipid_picott()
        ranked = [
            {"_article": {
                "title": "Semaglutide reduces LDL-C",
                "abstract": "Semaglutide lowered LDL cholesterol in T2D.",
            }, "relevance_score": 80},
            {"_article": {
                "title": "GLP-1 receptor in mouse islets",
                "abstract": "In vitro signaling of GLP-1 receptor.",
            }, "relevance_score": 60},
            {"_article": {
                "title": "Atorvastatin lowers LDL",
                "abstract": "Statin reduced LDL cholesterol.",
            }, "relevance_score": 50},
        ]
        out = semantic_reranker.rerank(ranked, picott)
        assert out[0]["article_answer_class"] == "direct"
        assert out[1]["article_answer_class"] in {"mechanistic",
                                                     "wrong_outcome"}
        assert out[2]["article_answer_class"] == "wrong_intervention"
        for r in out:
            assert r["article_answer_class"] in ARTICLE_ANSWER_CLASSES

    def test_rerank_when_phi_blocked_uses_background_default(self):
        ranked = [{"_article": {"title": "x", "abstract": "y"}}]
        out = semantic_reranker.rerank(ranked, _glp1_lipid_picott(),
                                        is_phi_blocked=True)
        assert out[0]["article_answer_class"] == "background_only"
        assert out[0]["semantic_relevance_score"] is None

    def test_rerank_no_picott_defaults_to_background(self):
        ranked = [{"_article": {"title": "x", "abstract": "y"}}]
        out = semantic_reranker.rerank(ranked, None)
        assert out[0]["article_answer_class"] == "background_only"

    def test_rerank_direct_outscores_mechanistic_semantically(self):
        """Semantic relevance score (independent of caps) should put
        direct > mechanistic for the same input."""
        picott = _glp1_lipid_picott()
        ranked = [
            {"_article": {
                "title": "Semaglutide reduces LDL-C in T2D",
                "abstract": "Randomized trial; LDL cholesterol fell by 0.4 mmol/L.",
            }, "relevance_score": 80},
            {"_article": {
                "title": "GLP-1 receptor expression in beta cells",
                "abstract": "In vitro and rodent islet studies of GLP-1 signaling.",
            }, "relevance_score": 80},
        ]
        out = semantic_reranker.rerank(ranked, picott)
        direct = next(r for r in out if r["article_answer_class"] == "direct")
        mech = next(r for r in out if r["article_answer_class"]
                    in {"mechanistic", "wrong_outcome"})
        assert direct["semantic_relevance_score"] > mech["semantic_relevance_score"]


# =============================================================================
# Group E4 — Ranker + caps integration (sanity)
# =============================================================================

class TestRankingCapsAlignWithAnswerClass:
    """Sanity: a tirzepatide-only article (capped at 70) and a
    glp1_biology_no_lipid article (capped at 50) and an ldl_without_glp1
    article (capped at 30) — confirm the cap labels align with what the
    classifier produces."""

    def test_tirzepatide_only_classified_as_wrong_intervention(self):
        picott = _glp1_lipid_picott()
        cls = classify_article_answer(
            title="Tirzepatide lowers LDL cholesterol",
            abstract="Tirzepatide reduced LDL-C in T2D patients.",
            picott=picott,
        )
        # Tirzepatide is NOT a GLP-1 RA; classifier sees no GLP-1 term in
        # the article → wrong_intervention. (The cap will additionally
        # apply tirzepatide_only_for_glp1_question = 70.)
        assert cls == "wrong_intervention", cls

    def test_glp1_biology_classified_as_mechanistic_or_wrong_outcome(self):
        picott = _glp1_lipid_picott()
        cls = classify_article_answer(
            title="Semaglutide and beta-cell function",
            abstract="Semaglutide improved beta cell function in vitro.",
            picott=picott,
        )
        assert cls in {"mechanistic", "wrong_outcome"}, cls

    def test_ldl_without_glp1_classified_as_wrong_intervention(self):
        picott = _glp1_lipid_picott()
        cls = classify_article_answer(
            title="Statins and LDL-C lowering",
            abstract="Atorvastatin reduced LDL cholesterol by 50%.",
            picott=picott,
        )
        assert cls == "wrong_intervention", cls
