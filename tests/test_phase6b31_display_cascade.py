"""Phase 6B.3.1 — Display-layer cascade alignment.

The session-level ``interpreted_question_display_json.type_of_question``
formatter MUST share the same cascade as the per-article
``rank_explanation`` so the user never sees inconsistent labels.
"""
from __future__ import annotations

import pytest

from services.interpretation_canonical import (
    canonical_label_for_rank_explanation,
    _SUBTYPE_PARENT_TYPE,
)
from services.interpreted_question_display import _humanize_type_of_question


# ---------------------------------------------------------------------------
# 1. GLP-1 case — the tester's specific failure.
# ---------------------------------------------------------------------------
def test_display_serializer_resolves_other_with_subtype_to_parent_type():
    out = _humanize_type_of_question("other", "metabolic_biomarker_effect")
    assert "Pharmacology" in out, out
    assert "metabolic biomarker effect" in out.lower(), out
    assert "Other (" not in out, out


# ---------------------------------------------------------------------------
# 2. The session-level label matches the article-level rank_explanation
#    header for the same (type, subtype) pair.
# ---------------------------------------------------------------------------
def test_display_serializer_matches_rank_explanation_label_for_glp1():
    session_label = _humanize_type_of_question(
        "other", "metabolic_biomarker_effect",
    )
    rx_label = canonical_label_for_rank_explanation(
        picott={"type_of_question": "other",
                 "question_subtype": "metabolic_biomarker_effect"},
    )
    # The session-level label may add " / <subtype>" suffix; assert
    # both contain the same parent-type token.
    assert "Pharmacology" in session_label
    assert "Pharmacology" in rx_label
    assert session_label == rx_label, (session_label, rx_label)


# ---------------------------------------------------------------------------
# 3. Every known entry in _SUBTYPE_PARENT_TYPE resolves to a "<Parent>
#    / <humanized subtype>" pair — no parens, no bare "Other".
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("subtype,parent", list(_SUBTYPE_PARENT_TYPE.items()))
def test_display_serializer_handles_known_subtypes_table(subtype, parent):
    out = _humanize_type_of_question("other", subtype)
    assert out and out != "Not specified", (subtype, out)
    assert "Other (" not in out, (subtype, out)
    # Parent-type token (Title Cased) appears in the output.
    parent_token = parent.split("_")[0].title()
    assert parent_token in out, (subtype, parent, out)


# ---------------------------------------------------------------------------
# 4. Subtype not in the table → "Other / <subtype humanized>" with the
#    "/" separator. NEVER "Other (<subtype>)" with parens.
# ---------------------------------------------------------------------------
def test_display_serializer_unknown_subtype_falls_back_gracefully():
    out = _humanize_type_of_question("other", "screening_threshold_xyz")
    # Either we resolve via canonical's fallback ("other / <subtype>"
    # via _humanize) OR an explicit "Not specified". Both are accepted
    # — what is NOT accepted is the legacy parens format.
    assert "Other (" not in out, out
    if out != "Not specified":
        assert "/" in out, out


# ---------------------------------------------------------------------------
# 5. PICOTT-level concrete type wins (no subtype downgrade).
# ---------------------------------------------------------------------------
def test_display_serializer_picott_concrete_type_wins():
    out = _humanize_type_of_question("therapy",
                                       "transfusion_threshold_comparison")
    assert "Therapy" in out
    assert "transfusion threshold comparison" in out.lower()


# ---------------------------------------------------------------------------
# 6. role_weights override path — when PICOTT is "other" but
#    role_weights carries a concrete type, the serializer picks it.
# ---------------------------------------------------------------------------
def test_display_serializer_uses_role_weights_when_picott_is_other():
    out = _humanize_type_of_question(
        "other", None, role_weights={"type_of_question": "pharmacology"},
    )
    assert "Pharmacology" in out
    assert "Other (" not in out
