"""Phase K3B Wave 1 Step 1.6 — Wave-1-strict verdict adjudicator.

Closed-set, Wave-1-only verdicts. EoC verdicts and EoC reason-codes
are NOT eligible for Wave 1 enforcement and are routed to the
defect-discovery branch (an EoC-only row should never have surfaced
as WBE=true; if it did, that's a detector bug, not an adjudication
question).

Closed-set verdict labels (Wave 1):
- ``true_positive_retracted``
- ``false_positive``
- ``unclear_requires_source_review``

Closed-set verdict bases (per Step 1.6 spec):
- ``pubmed_retracted_publication_type``
- ``pubmed_retraction_in_link``
- ``europepmc_comment_corrections``
- ``crossref_update_to_retraction``
- ``journal_official_retraction_notice_via_pubmed``
- ``metadata_disagreement_across_sources``
- ``source_metadata_missing_or_ambiguous``
- ``article_join_unstable_no_verdict_possible``

Closed-set defect labels (NOT verdicts — these are detector defects
flagged for ``fix_detector_before_proceeding``):
- ``detector_defect_wbe_on_unstable_join``
- ``detector_defect_eoc_only_wbe``
- ``detector_defect_non_wave1_reason_code``
- ``detector_defect_missing_retracted_reason_code``
- ``detector_defect_wbe_histogram_contains_non_retracted_key``

Pure function. No IO. No LLM. No scrape.
"""
from __future__ import annotations

from typing import Any, Literal

VerdictLabelWave1 = Literal[
    "true_positive_retracted",
    "false_positive",
    "unclear_requires_source_review",
]

VerdictBasisWave1 = Literal[
    "pubmed_retracted_publication_type",
    "pubmed_retraction_in_link",
    "europepmc_comment_corrections",
    "crossref_update_to_retraction",
    "journal_official_retraction_notice_via_pubmed",
    "metadata_disagreement_across_sources",
    "source_metadata_missing_or_ambiguous",
    "article_join_unstable_no_verdict_possible",
]

DefectLabelWave1 = Literal[
    "detector_defect_wbe_on_unstable_join",
    "detector_defect_eoc_only_wbe",
    "detector_defect_non_wave1_reason_code",
    "detector_defect_missing_retracted_reason_code",
    "detector_defect_wbe_histogram_contains_non_retracted_key",
]


# Wave-1 permanently-barred enforcement reasons. If any one of these
# is the SOLE reason on a WBE=true row, the row is a detector defect.
# (We test "barred-only" — if ``retracted`` is also present we still
# treat the row as Wave-1-eligible.)
_WAVE1_BARRED_REASONS: tuple[str, ...] = (
    "expression_of_concern",
    "missing_primary_anchor",
    "cross_intent_drift",
    "low_branch_alignment",
    "wrong_framework_lane",
    "animal_only_for_human_question",
    "preprint_unverified",
    "generic_modifier_only_match",
)

_WAVE1_ALLOWED_REASONS: tuple[str, ...] = ("retracted",)


_RETRACTED_TOKENS: tuple[str, ...] = (
    "retracted publication", "retraction of publication",
)


def _norm(pub_types: list[Any] | None) -> list[str]:
    return [
        str(p).strip().lower() for p in (pub_types or [])
        if isinstance(p, str)
    ]


def adjudicate_wave1(record: dict[str, Any]) -> dict[str, Any]:
    """Adjudicate a single WBE=true soak row against the Wave-1
    closed-set. ``record`` is a JSONL row from
    ``scripts/k3b_wave1_step1_soak_collector.py``.

    Returns a dict with::

        {
            "verdict": <VerdictLabelWave1>,
            "verdict_basis": <VerdictBasisWave1>,
            "signal": <human-readable string>,
            "detector_defect": <DefectLabelWave1 | None>,
        }

    When ``detector_defect`` is non-None the caller MUST treat the
    row as a detector-defect discovery, NOT as a false-positive.
    """
    # Closed-set null defect by default.
    defect: str | None = None

    join = str(record.get("article_join_status") or "stable")
    wbe_codes_raw = (
        record.get("would_have_been_excluded_reason_codes") or []
    )
    wbe_codes = [
        str(c).strip().lower() for c in wbe_codes_raw
        if isinstance(c, str)
    ]
    cap_codes_raw = record.get("cap_reason_codes") or []
    cap_codes = [
        str(c).strip().lower() for c in cap_codes_raw
        if isinstance(c, str)
    ]
    pub_types = _norm(record.get("publication_types"))
    is_retracted = bool(record.get("is_retracted"))
    has_eoc = bool(record.get("has_expression_of_concern"))
    is_retraction_notice = bool(record.get("is_retraction_notice"))

    # -----------------------------------------------------------------
    # Step 1: Defense-in-depth defect detection BEFORE any TP/FP
    # adjudication. Every one of these branches should NEVER fire
    # post-Step-1.5; if it does, the detector has regressed and we
    # must pause adjudication.
    # -----------------------------------------------------------------

    # 1a. WBE=true on an unstable join — should not happen after the
    # Step 1.5 back-stamp fix. The K3B gate does its own stability
    # check at fire time, so a WBE=true on an unstable join means the
    # in-flight diagnostic disagrees with the live join — a bug.
    if join != "stable":
        return {
            "verdict": "unclear_requires_source_review",
            "verdict_basis":
                "article_join_unstable_no_verdict_possible",
            "signal":
                f"article_join_status={join!r} on a WBE=true row — "
                "this should not happen post-Step-1.5",
            "detector_defect":
                "detector_defect_wbe_on_unstable_join",
        }

    # 1b. Wave-1 reason allow-list. If the WBE row's reason codes
    # contain anything outside ``retracted``, that's a detector
    # defect (the gate is supposed to be retracted-only in Wave 1).
    non_wave1 = [c for c in wbe_codes if c not in _WAVE1_ALLOWED_REASONS]
    if non_wave1:
        # Specifically: if the row is EoC-only that's a stronger
        # defect signal — EoC enforcement is explicitly uncertified
        # in Wave 1.
        if set(non_wave1) == {"expression_of_concern"} and not any(
            c == "retracted" for c in wbe_codes
        ):
            return {
                "verdict": "unclear_requires_source_review",
                "verdict_basis":
                    "metadata_disagreement_across_sources",
                "signal":
                    "EoC-only WBE=true row — Wave 1 must not enforce "
                    "EoC; this is a detector defect",
                "detector_defect":
                    "detector_defect_eoc_only_wbe",
            }
        return {
            "verdict": "unclear_requires_source_review",
            "verdict_basis": "metadata_disagreement_across_sources",
            "signal": (
                f"non-Wave-1 enforcement reason codes present: "
                f"{sorted(non_wave1)!r}"
            ),
            "detector_defect":
                "detector_defect_non_wave1_reason_code",
        }

    # 1c. The row must claim ``retracted`` in its WBE codes (we just
    # established no non-Wave-1 codes; the only Wave-1 code is
    # ``retracted`` so the list must contain it).
    if "retracted" not in wbe_codes:
        return {
            "verdict": "unclear_requires_source_review",
            "verdict_basis": "source_metadata_missing_or_ambiguous",
            "signal": (
                "WBE=true row missing 'retracted' enforcement reason "
                "code"
            ),
            "detector_defect":
                "detector_defect_missing_retracted_reason_code",
        }

    # 1d. ``cap_reason_codes`` must also contain ``retracted`` —
    # otherwise the upstream classifier and the K3B gate disagree.
    if "retracted" not in cap_codes:
        return {
            "verdict": "unclear_requires_source_review",
            "verdict_basis": "metadata_disagreement_across_sources",
            "signal": (
                "WBE codes contain 'retracted' but cap_reason_codes "
                "do not — classifier/gate disagreement"
            ),
            "detector_defect":
                "detector_defect_missing_retracted_reason_code",
        }

    # -----------------------------------------------------------------
    # Step 2: Wave-1 TP / FP adjudication. The row passes all
    # invariants; now verify a trusted structured retraction signal
    # exists on the article metadata.
    # -----------------------------------------------------------------

    pub_says_retracted = any(
        any(tok in p for tok in _RETRACTED_TOKENS) for p in pub_types
    )

    if pub_says_retracted:
        return {
            "verdict": "true_positive_retracted",
            "verdict_basis": "pubmed_retracted_publication_type",
            "signal":
                "pubmed.publication_types contains "
                "'Retracted Publication'",
            "detector_defect": None,
        }
    if is_retracted:
        return {
            "verdict": "true_positive_retracted",
            "verdict_basis":
                "journal_official_retraction_notice_via_pubmed",
            "signal": "article.is_retracted=true",
            "detector_defect": None,
        }
    if is_retraction_notice:
        return {
            "verdict": "true_positive_retracted",
            "verdict_basis": "pubmed_retraction_in_link",
            "signal":
                "article.is_retraction_notice=true (this record IS "
                "the retraction notice)",
            "detector_defect": None,
        }
    if has_eoc:
        # The row was flagged as retracted but the article has only
        # an EoC signal — classic metadata disagreement.
        return {
            "verdict": "unclear_requires_source_review",
            "verdict_basis": "metadata_disagreement_across_sources",
            "signal":
                "cap_reason=retracted but article only carries "
                "expression_of_concern signal",
            "detector_defect": None,
        }

    # No structured retraction signal — false positive.
    return {
        "verdict": "false_positive",
        "verdict_basis": "source_metadata_missing_or_ambiguous",
        "signal":
            "no retraction signal in pubmed.publication_types, "
            "article.is_retracted, or article.is_retraction_notice",
        "detector_defect": None,
    }


__all__ = [
    "adjudicate_wave1",
    "VerdictLabelWave1",
    "VerdictBasisWave1",
    "DefectLabelWave1",
]
