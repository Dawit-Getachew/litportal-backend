"""Tests for the redundant-anchor over-rejection fix.

Two parts:
  * anchor_dedup.concept_is_redundant — collapses a parent-molecule duplicate
    ("Glucagon-Like Peptide 1") into the drug-class anchor family, but never
    collapses genuinely distinct concepts.
  * picott_ranker._check_concepts — a missing role-"other" (supplementary)
    concept no longer hard-caps; genuine intervention/condition anchors still do.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services import anchor_dedup  # noqa: E402
from services import picott_ranker  # noqa: E402


# ---------------------------------------------------------------- anchor_dedup
def test_redundant_parent_molecule_collapses():
    existing = [[
        "GLP-1 receptor agonists",
        "glucagon-like peptide-1 receptor agonist",
        "GLP-1RA",
    ]]
    assert anchor_dedup.concept_is_redundant("Glucagon-Like Peptide 1", existing) is True


def test_distinct_concepts_not_collapsed():
    existing = [["GLP-1 receptor agonists", "GLP-1RA"]]
    assert anchor_dedup.concept_is_redundant("acute myocardial infarction", existing) is False
    assert anchor_dedup.concept_is_redundant("heart failure", existing) is False


def test_single_token_not_collapsed():
    existing = [["GLP-1 receptor agonists", "glucagon-like peptide-1 receptor agonist"]]
    # "GLP-1" has <2 significant tokens -> never collapsed on its own.
    assert anchor_dedup.concept_is_redundant("GLP-1", existing) is False


def test_dedup_flag_default_on(monkeypatch):
    monkeypatch.delenv("PICOTT_ANCHOR_DEDUP", raising=False)
    assert anchor_dedup.anchor_dedup_enabled() is True
    monkeypatch.setenv("PICOTT_ANCHOR_DEDUP", "false")
    assert anchor_dedup.anchor_dedup_enabled() is False


# --------------------------------------------------- picott_ranker soft anchor
def _picott_three_anchors():
    return {
        "type_of_question": "background",
        "must_have_concepts": [
            {"concept": "GLP-1 receptor agonists", "role": "intervention", "synonyms": []},
            {"concept": "acute myocardial infarction", "role": "condition", "synonyms": ["ACS"]},
            # redundant parent-molecule concept, role "other":
            {"concept": "Glucagon-Like Peptide 1", "role": "other", "synonyms": []},
        ],
    }


_TITLE = "GLP-1 receptor agonists in ACS"
_ABSTRACT = "A cohort of GLP-1 receptor agonists in acute coronary syndrome (ACS) patients."


def test_other_role_does_not_cap_by_default(monkeypatch):
    monkeypatch.delenv("PICOTT_SOFT_ANCHOR_NONCAPPING", raising=False)
    res = picott_ranker._check_concepts(_TITLE, _ABSTRACT, "", _picott_three_anchors())
    # The two genuine anchors are present; the missing redundant "other"
    # concept must NOT cap.
    assert res.cap == 100
    assert "Glucagon-Like Peptide 1" not in res.missing_must_have


def test_other_role_caps_when_flag_disabled(monkeypatch):
    monkeypatch.setenv("PICOTT_SOFT_ANCHOR_NONCAPPING", "false")
    res = picott_ranker._check_concepts(_TITLE, _ABSTRACT, "", _picott_three_anchors())
    # Old behavior restored: the redundant concept caps to 30 and shows missing.
    assert res.cap == 30
    assert "Glucagon-Like Peptide 1" in res.missing_must_have


def test_genuine_intervention_anchor_still_caps(monkeypatch):
    monkeypatch.delenv("PICOTT_SOFT_ANCHOR_NONCAPPING", raising=False)
    p = {
        "type_of_question": "background",
        "must_have_concepts": [
            {"concept": "GLP-1 receptor agonists", "role": "intervention", "synonyms": []},
            {"concept": "acute myocardial infarction", "role": "condition", "synonyms": ["ACS"]},
        ],
    }
    res = picott_ranker._check_concepts(
        "Aspirin in ACS", "Aspirin for acute coronary syndrome (ACS).", "", p
    )
    # Missing the real intervention anchor still caps hard.
    assert res.cap <= 30
    assert "missing_intervention_anchor" in res.cap_reasons
