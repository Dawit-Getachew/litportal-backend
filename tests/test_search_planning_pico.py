"""Phase 2A.1 — PICO-driven SearchPlanning tests."""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.search_planning import SearchPlanningService  # noqa: E402


def _parsed_apixaban():
    return {
        "normalized_question": "In adults with nonvalvular atrial fibrillation, how does apixaban compare with warfarin for stroke prevention and major bleeding?",
        "clinical_question_type": "treatment",
        "specialty": "cardiology",
        "population": "adults with nonvalvular atrial fibrillation",
        "intervention_or_exposure": "apixaban",
        "comparator": "warfarin",
        "outcomes": ["stroke prevention", "major bleeding"],
        "setting": "",
        "time_horizon": "",
        "key_concepts": ["atrial fibrillation", "apixaban", "warfarin", "stroke"],
        "mesh_terms": [],
        "synonyms": ["af", "afib", "nvaf"],
        "drug_names_and_classes": ["apixaban", "warfarin", "DOACs", "VKAs"],
        "exclusions": [],
        "rationale": "deterministic",
    }


def test_high_precision_includes_intervention_and_comparator():
    plans = SearchPlanningService().plan(_parsed_apixaban())
    hp = next(p for p in plans if p.query_type == "high_precision").query_string
    assert 'TITLE_ABS:"apixaban"' in hp
    assert 'TITLE_ABS:"warfarin"' in hp
    assert "NOT SRC:PPR" in hp


def test_high_precision_includes_population_condition():
    plans = SearchPlanningService().plan(_parsed_apixaban())
    hp = next(p for p in plans if p.query_type == "high_precision").query_string
    # Either the full short-pop or the condition fragment must be present.
    assert "nonvalvular atrial fibrillation" in hp.lower()


def test_evidence_filtered_keeps_pub_types_and_adds_pico():
    plans = SearchPlanningService().plan(_parsed_apixaban())
    ef = next(p for p in plans if p.query_type == "evidence_filtered").query_string
    assert 'TITLE_ABS:"apixaban"' in ef
    assert 'TITLE_ABS:"warfarin"' in ef
    assert "systematic review" in ef
    assert "randomized controlled trial" in ef
    assert "OPEN_ACCESS:Y" in ef


def test_guideline_focused_still_generated_with_pico():
    plans = SearchPlanningService().plan(_parsed_apixaban())
    gf = next(p for p in plans if p.query_type == "guideline_focused").query_string
    assert 'PUB_TYPE:"practice guideline"' in gf or 'PUB_TYPE:"guideline"' in gf
    # Should reference the topic (population or intervention)
    assert ("atrial fibrillation" in gf.lower()) or ('TITLE_ABS:"apixaban"' in gf)


def test_all_queries_still_contain_not_src_ppr():
    plans = SearchPlanningService().plan(_parsed_apixaban())
    assert len(plans) == 4
    for p in plans:
        assert "NOT SRC:PPR" in p.query_string
        assert 'NOT PUB_TYPE:"preprint"' in p.query_string


def test_fallback_when_pico_missing():
    parsed = {
        "normalized_question": "Best evidence for SGLT2 inhibitors",
        "clinical_question_type": "treatment",
        "specialty": "cardiology",
        "population": "", "intervention_or_exposure": "", "comparator": "",
        "outcomes": [], "setting": "", "time_horizon": "",
        "key_concepts": ["sglt2 inhibitors", "evidence"],
        "mesh_terms": [], "synonyms": [],
        "drug_names_and_classes": ["SGLT2 inhibitors"],
        "exclusions": [], "rationale": "",
    }
    plans = SearchPlanningService().plan(parsed)
    hp = next(p for p in plans if p.query_type == "high_precision").query_string
    # No empty TITLE_ABS:""
    assert 'TITLE_ABS:""' not in hp
    assert "NOT SRC:PPR" in hp
    # Should fall back to using key_concepts
    assert "sglt2 inhibitors" in hp.lower() or "sglt" in hp.lower()
