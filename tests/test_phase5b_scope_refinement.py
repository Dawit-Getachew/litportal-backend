"""Phase 5B retest — post-translation question-scope refinement.

The classifier alone can't always tell a foreground comparison-frame
question that starts with "What is the difference between X and Y in
patients ..." from a genuinely hybrid one. After concept translation we
know roles + comparison-pattern signals + foreground type-of-question.
This module promotes ``hybrid`` → ``foreground`` when those signals
confirm a foreground question, and keeps ``hybrid`` for questions that
legitimately mix a "What is X?" definition with a decision frame.

The 4 tester-mandated cases:
1. AMI hemoglobin/transfusion           → foreground
2. milrinone/SAH dosing (regression)    → foreground
3. "What is septic shock?" (regression) → background
4. HF + sacubitril/valsartan comparison → hybrid
"""
from __future__ import annotations

import pytest

from services.clinical_question_parser import clinical_question_parser


@pytest.mark.asyncio
class TestPhase5BScopeRefinement:
    async def test_ami_hemoglobin_transfusion_is_foreground(self):
        q = (
            "What is the difference between high hemoglobin target and low "
            "hemoglobin target in patients that are having acute myocardial "
            "infarction?"
        )
        out = await clinical_question_parser.parse(q)
        p = out.get("picott_json") or {}
        assert p.get("question_scope") == "foreground", (
            "AMI hemoglobin/transfusion is a foreground comparison-frame "
            f"question; got scope={p.get('question_scope')!r}"
        )
        # Sanity: type/subtype must remain therapy/transfusion_threshold.
        assert p.get("type_of_question") == "therapy"
        assert p.get("question_subtype") == "transfusion_threshold_comparison"

    async def test_milrinone_sah_dosing_is_foreground(self):
        # Regression — dosing/pharmacology question with a population
        # anchor and a drug must classify as foreground, not hybrid.
        q = (
            "What is the dose of milrinone for patients with acute "
            "subarchnoid hemorrhage?"
        )
        out = await clinical_question_parser.parse(q)
        p = out.get("picott_json") or {}
        assert p.get("question_scope") == "foreground", (
            "milrinone/SAH dosing is a foreground question; "
            f"got scope={p.get('question_scope')!r}"
        )
        assert p.get("type_of_question") == "dosing_protocol"

    async def test_what_is_septic_shock_is_background(self):
        # Regression — "what is X" with X a bare concept noun stays
        # background. No decision frame and no foreground type.
        q = "What is septic shock?"
        out = await clinical_question_parser.parse(q)
        p = out.get("picott_json") or {}
        assert p.get("question_scope") == "background", (
            "'What is septic shock?' must remain background; "
            f"got scope={p.get('question_scope')!r}"
        )
        assert p.get("type_of_question") == "background"

    async def test_definitional_plus_decision_frame_is_hybrid(self):
        # Genuinely hybrid: definitional "What is X" + decision frame
        # "does Y compare with Z?".
        q = (
            "What is heart failure, and does sacubitril/valsartan reduce "
            "mortality compared with enalapril?"
        )
        out = await clinical_question_parser.parse(q)
        p = out.get("picott_json") or {}
        assert p.get("question_scope") == "hybrid", (
            "Mixed definitional + decision-frame question must stay "
            f"hybrid; got scope={p.get('question_scope')!r}"
        )


@pytest.mark.asyncio
class TestPhase5BScopeRefinementRegressions:
    """Spot-check that the refinement does not corrupt other classifier
    outputs."""

    async def test_apixaban_vs_warfarin_is_foreground(self):
        q = (
            "In adults with nonvalvular atrial fibrillation, how does "
            "apixaban compare to warfarin for stroke prevention?"
        )
        out = await clinical_question_parser.parse(q)
        assert (out.get("picott_json") or {}).get("question_scope") == "foreground"

    async def test_what_is_npm_unknown_concept_stays_background(self):
        # No decision frame anywhere — must stay background.
        q = "What is hyperkalemia?"
        out = await clinical_question_parser.parse(q)
        assert (out.get("picott_json") or {}).get("question_scope") == "background"

    async def test_pathophysiology_stays_background(self):
        # Pathophysiology questions are background regardless of
        # downstream decisions.
        q = "What is the pathophysiology of acute kidney injury?"
        out = await clinical_question_parser.parse(q)
        assert (out.get("picott_json") or {}).get("question_scope") == "background"
