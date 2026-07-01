"""Phase K3A.3 — Verdict adjudicator v2 (expanded closed-set verdicts).

Adds the K3A.3 spec verdicts:

- ``true_positive_retracted`` (specific)
- ``true_positive_expression_of_concern`` (specific)
- ``true_positive_retracted_or_eoc`` (legacy alias for K3A.2 rows)
- ``article_join_unstable_no_verdict_possible`` (new basis — used
  whenever ``article_join_status != "stable"``)
- ``pubmed_expression_of_concern_publication_type``
- ``pubmed_expression_of_concern_in_link``

Same deterministic, closed-set logic as the K3A.2 adjudicator — pure
function, no LLM, no scrape. The K3A.2 adjudicator is preserved
unchanged for back-compat with the K3A.2 review artifacts.
"""
from __future__ import annotations

from typing import Any, Literal

VerdictLabelV2 = Literal[
    "true_positive_retracted_or_eoc",
    "true_positive_retracted",
    "true_positive_expression_of_concern",
    "false_positive",
    "unclear_requires_source_review",
]

VerdictBasisV2 = Literal[
    "pubmed_retracted_publication_type",
    "pubmed_retraction_in_link",
    "europepmc_comment_corrections",
    "crossref_update_to_retraction",
    "pubmed_expression_of_concern_publication_type",
    "pubmed_expression_of_concern_in_link",
    "journal_official_retraction_notice_via_pubmed",
    "journal_official_eoc_notice_via_pubmed",
    "metadata_disagreement_across_sources",
    "source_metadata_missing_or_ambiguous",
    "article_join_unstable_no_verdict_possible",
]

_RETRACTED_TOKENS: tuple[str, ...] = (
    "retracted publication", "retraction of publication",
)
_EOC_TOKENS: tuple[str, ...] = (
    "expression of concern", "expressionofconcernin",
)


def _norm(pub_types: list[Any] | None) -> list[str]:
    return [
        str(p).strip().lower() for p in (pub_types or [])
        if isinstance(p, str)
    ]


def adjudicate_v2(record: dict[str, Any]) -> dict[str, str]:
    """Closed-set verdict per the K3A.3 spec. Includes
    ``article_join_status`` short-circuit.
    """
    code = str(record.get("cap_reason_code") or "")
    join = str(record.get("article_join_status") or "stable")
    pub_types = _norm(record.get("publication_types"))
    is_retracted = bool(record.get("is_retracted"))
    has_eoc = bool(record.get("has_expression_of_concern"))
    is_retraction_notice = bool(record.get("is_retraction_notice"))

    # K3A.3 — if the join is unstable at review time, no verdict
    # is possible against current DB state. Route to unclear with
    # the new ``article_join_unstable_no_verdict_possible`` basis.
    if join != "stable":
        return {
            "verdict": "unclear_requires_source_review",
            "verdict_basis": "article_join_unstable_no_verdict_possible",
            "signal": f"article_join_status={join!r}",
        }

    metadata_blank = (
        not pub_types and not is_retracted and not has_eoc
        and not is_retraction_notice
        and not record.get("pmid")
        and not record.get("doi")
        and not record.get("title")
    )

    pub_says_retracted = any(
        any(tok in p for tok in _RETRACTED_TOKENS) for p in pub_types
    )
    pub_says_eoc = any(
        any(tok in p for tok in _EOC_TOKENS) for p in pub_types
    )

    if code in ("retracted", "expression_of_concern") and metadata_blank:
        return {
            "verdict": "unclear_requires_source_review",
            "verdict_basis": "source_metadata_missing_or_ambiguous",
            "signal": (
                "article metadata not joinable at adjudication time"
            ),
        }

    if code == "retracted":
        if pub_says_retracted:
            return {
                "verdict": "true_positive_retracted",
                "verdict_basis": "pubmed_retracted_publication_type",
                "signal": (
                    "pubmed.publication_types contains "
                    "'Retracted Publication'"
                ),
            }
        if is_retracted:
            return {
                "verdict": "true_positive_retracted",
                "verdict_basis":
                    "journal_official_retraction_notice_via_pubmed",
                "signal": "article.is_retracted=true",
            }
        if is_retraction_notice:
            return {
                "verdict": "true_positive_retracted",
                "verdict_basis": "pubmed_retraction_in_link",
                "signal": (
                    "article.is_retraction_notice=true (this record IS "
                    "the retraction notice)"
                ),
            }
        return {
            "verdict": "false_positive",
            "verdict_basis": "source_metadata_missing_or_ambiguous",
            "signal": (
                "no retraction signal in pubmed.publication_types, "
                "article.is_retracted, or article.is_retraction_notice"
            ),
        }

    if code == "expression_of_concern":
        if pub_says_eoc:
            return {
                "verdict": "true_positive_expression_of_concern",
                "verdict_basis":
                    "pubmed_expression_of_concern_publication_type",
                "signal": (
                    "pubmed.publication_types contains "
                    "'Expression of Concern'"
                ),
            }
        if has_eoc:
            return {
                "verdict": "true_positive_expression_of_concern",
                "verdict_basis":
                    "journal_official_eoc_notice_via_pubmed",
                "signal": "article.has_expression_of_concern=true",
            }
        if pub_says_retracted or is_retracted:
            return {
                "verdict": "unclear_requires_source_review",
                "verdict_basis":
                    "metadata_disagreement_across_sources",
                "signal": (
                    "cap_reason said EoC but PubMed says retracted"
                ),
            }
        return {
            "verdict": "false_positive",
            "verdict_basis": "source_metadata_missing_or_ambiguous",
            "signal": (
                "no EoC signal in pubmed.publication_types or "
                "article.has_expression_of_concern"
            ),
        }

    return {
        "verdict": "unclear_requires_source_review",
        "verdict_basis": "source_metadata_missing_or_ambiguous",
        "signal": f"unrecognized cap_reason_code={code!r}",
    }
