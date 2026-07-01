"""Phase K3A.2 — Verdict adjudicator for shadow soak candidates.

Deterministic, closed-set adjudicator that decides whether each
``retracted`` / ``expression_of_concern`` candidate flagged by the
K3A shadow classifier is a TRUE positive, FALSE positive, or UNCLEAR
based purely on metadata from approved upstream sources:

- PubMed (E-utilities — already in
  ``services.connectors.pubmed``): ``publication_types_json``
  containing ``"Retracted Publication"`` or ``"Expression of Concern"``
  is the gold signal.
- Article doc ``is_retracted`` / ``has_expression_of_concern`` flags
  populated upstream by the metadata normalizer.
- Optional Europe PMC / Crossref cross-checks via the article doc's
  ``source_ids_json`` and ``source_urls_json`` — these are read but
  treated as confirmatory only, never as the sole basis.

No LLM. No publisher scrape. No HTML parsing. No new external source.

The adjudicator is exposed as a pure function ``adjudicate(record)``
so the test suite can drive it with synthetic records, and a thin
async driver ``adjudicate_from_db(...)`` for the live soak path.
"""
from __future__ import annotations

from typing import Any, Literal

VerdictLabel = Literal[
    "true_positive_retracted_or_eoc",
    "false_positive",
    "unclear_requires_source_review",
]

VerdictBasis = Literal[
    "pubmed_retracted_publication_type",
    "pubmed_retraction_in_link",
    "europepmc_comment_corrections",
    "crossref_update_to_retraction",
    "journal_official_retraction_notice_via_pubmed",
    "journal_official_eoc_notice_via_pubmed",
    "metadata_disagreement_across_sources",
    "source_metadata_missing_or_ambiguous",
]

_RETRACTED_TOKENS: tuple[str, ...] = (
    "retracted publication",
    "retraction of publication",
)
_EOC_TOKENS: tuple[str, ...] = (
    "expression of concern",
    "expressionofconcernin",
)


def _norm(pub_types: list[Any] | None) -> list[str]:
    return [
        str(p).strip().lower() for p in (pub_types or [])
        if isinstance(p, str)
    ]


def adjudicate(record: dict[str, Any]) -> dict[str, str]:
    """Return ``{verdict, verdict_basis, signal}`` for one candidate.

    ``record`` shape (closed-set, all optional except ``cap_reason_code``):

    - ``cap_reason_code`` — one of ``"retracted"`` or
      ``"expression_of_concern"`` (input from the K3A classifier).
    - ``publication_types`` — list of PubMed-style pub-type strings.
    - ``is_retracted`` — bool from the article doc.
    - ``has_expression_of_concern`` — bool from the article doc.
    - ``is_retraction_notice`` — bool from the article doc.
    - ``source_ids`` — dict (read but not authoritative).
    - ``source_urls`` — list (read but not authoritative).

    Decision table:

    | cap_reason_code         | signal                          | verdict          | basis                                            |
    | ----------------------- | ------------------------------- | ---------------- | ------------------------------------------------ |
    | retracted               | pub_type contains retraction    | true_positive    | pubmed_retracted_publication_type                |
    | retracted               | article.is_retracted=true       | true_positive    | journal_official_retraction_notice_via_pubmed    |
    | retracted               | no retraction signals anywhere  | false_positive   | source_metadata_missing_or_ambiguous             |
    | retracted               | retraction-notice paper itself  | true_positive    | pubmed_retraction_in_link                        |
    | expression_of_concern   | pub_type contains EoC           | true_positive    | journal_official_eoc_notice_via_pubmed           |
    | expression_of_concern   | article.has_expression... true  | true_positive    | journal_official_eoc_notice_via_pubmed           |
    | expression_of_concern   | only retraction signal, no EoC  | unclear          | metadata_disagreement_across_sources             |
    | expression_of_concern   | no EoC/retraction signals       | false_positive   | source_metadata_missing_or_ambiguous             |
    | (other / missing)       | (any)                           | unclear          | source_metadata_missing_or_ambiguous             |
    """
    code = str(record.get("cap_reason_code") or "")
    pub_types = _norm(record.get("publication_types"))
    is_retracted_flag = bool(record.get("is_retracted"))
    has_eoc_flag = bool(record.get("has_expression_of_concern"))
    is_retraction_notice = bool(record.get("is_retraction_notice"))

    # K3A.2 — distinguish "metadata available but no retraction
    # signals" (true FP) from "metadata not joinable / orphaned"
    # (unclear; the classifier saw metadata we no longer can). When
    # every retrieval-time field is blank AND every flag is False,
    # the article doc is orphaned (deduplicated / cleaned-up /
    # missing) and we cannot deterministically verify the classifier.
    metadata_blank = (
        not pub_types
        and not is_retracted_flag
        and not has_eoc_flag
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

    # Metadata not joinable at adjudication time → classifier saw
    # metadata we no longer can; route to manual review rather than
    # call it a false positive. Hygiene finding documented separately.
    if code in ("retracted", "expression_of_concern") and metadata_blank:
        return {
            "verdict": "unclear_requires_source_review",
            "verdict_basis": "source_metadata_missing_or_ambiguous",
            "signal": (
                "article metadata not joinable at soak time "
                "(orphaned article doc — classifier saw retraction "
                "signal at retrieval time, post-hoc verification "
                "requires PubMed re-fetch by PMID)"
            ),
        }

    if code == "retracted":
        if pub_says_retracted:
            return {
                "verdict": "true_positive_retracted_or_eoc",
                "verdict_basis": "pubmed_retracted_publication_type",
                "signal": (
                    "pubmed.publication_types contains 'Retracted "
                    "Publication'"
                ),
            }
        if is_retracted_flag:
            return {
                "verdict": "true_positive_retracted_or_eoc",
                "verdict_basis":
                    "journal_official_retraction_notice_via_pubmed",
                "signal": "article.is_retracted=true",
            }
        if is_retraction_notice:
            return {
                "verdict": "true_positive_retracted_or_eoc",
                "verdict_basis": "pubmed_retraction_in_link",
                "signal": (
                    "article.is_retraction_notice=true (this record IS "
                    "the retraction notice)"
                ),
            }
        # No retraction signal anywhere → false positive.
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
                "verdict": "true_positive_retracted_or_eoc",
                "verdict_basis": "journal_official_eoc_notice_via_pubmed",
                "signal": (
                    "pubmed.publication_types contains 'Expression of "
                    "Concern'"
                ),
            }
        if has_eoc_flag:
            return {
                "verdict": "true_positive_retracted_or_eoc",
                "verdict_basis": "journal_official_eoc_notice_via_pubmed",
                "signal": "article.has_expression_of_concern=true",
            }
        if pub_says_retracted or is_retracted_flag:
            # The cap reason said EoC but PubMed says retracted →
            # metadata disagreement, route to manual review.
            return {
                "verdict": "unclear_requires_source_review",
                "verdict_basis":
                    "metadata_disagreement_across_sources",
                "signal": (
                    "cap_reason said EoC but PubMed says retracted "
                    "(escalate to manual source review)"
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

    # Unknown / missing cap_reason_code → can't adjudicate.
    return {
        "verdict": "unclear_requires_source_review",
        "verdict_basis": "source_metadata_missing_or_ambiguous",
        "signal": f"unrecognized cap_reason_code={code!r}",
    }
