"""DeduplicationService — cross-source canonical-key merging (Phase 2B).

For each record, the canonical key is already computed by the normalizer
(PubMed connector) or `metadata_normalizer.normalize()` (Europe PMC),
using the priority: PMID → DOI → PMCID → title|year|journal|firstauthor.

Phase 2B addition: after a first pass keyed by canonical_key, we run a
second pass that merges records sharing the same DOI/PMID/PMCID even if
they ended up under different canonical_keys (e.g. an OpenAlex-only record
with a DOI but no PMID could otherwise miss merging with a PubMed record
keyed by PMID — the OpenAlex enrichment step is the safer place to do that
catch-up, but here we still cross-check using all identifiers).

When merging, prefer the record with the longer abstract, union
publication_types, source_ids, source_urls, and ``sources_present``.
``query_types_matched`` carries forward — stored on search_session_articles.
"""
from __future__ import annotations

from typing import Any


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge b into a (a wins for scalars unless missing). Returns new dict."""
    out = dict(a)
    a_abs = (a.get("abstract") or "")
    b_abs = (b.get("abstract") or "")
    if len(b_abs) > len(a_abs):
        out["abstract"] = b.get("abstract")
    for k in (
        "title", "journal", "publication_year", "publication_date",
        "doi", "pmid", "pmcid", "article_type_classification",
        "full_text_status",
    ):
        if not out.get(k) and b.get(k):
            out[k] = b[k]
    for k in ("publication_types_json", "source_urls_json", "sources_present"):
        merged = list(out.get(k) or [])
        for item in (b.get(k) or []):
            if item not in merged:
                merged.append(item)
        out[k] = merged
    src_a = dict(out.get("source_ids_json") or {})
    src_b = dict(b.get("source_ids_json") or {})
    for k, v in src_b.items():
        if v and not src_a.get(k):
            src_a[k] = v
    out["source_ids_json"] = src_a
    for k in ("is_guideline", "is_preprint", "is_retracted",
              "is_retraction_notice", "has_expression_of_concern"):
        out[k] = bool(out.get(k)) or bool(b.get(k))
    if len(b.get("authors_json") or []) > len(out.get("authors_json") or []):
        out["authors_json"] = b.get("authors_json")
    return out


def _id_keys_for(art: dict[str, Any]) -> list[str]:
    """Return the strongest cross-source identifier keys for an article."""
    keys: list[str] = []
    pmid = (art.get("pmid") or "").strip()
    if pmid:
        keys.append(f"pmid:{pmid}")
    doi = (art.get("doi") or "").strip().lower()
    if doi:
        keys.append(f"doi:{doi}")
    pmcid = (art.get("pmcid") or "").strip()
    if pmcid:
        keys.append(f"pmcid:{pmcid}")
    return keys


class DeduplicationService:
    def dedupe(
        self,
        records: list[tuple[str, dict[str, Any]]],
    ) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
        """Input: list of (query_type, normalized_article_dict).
        Output: (unique_articles, canonical_key -> [query_types_matched]).
        """
        # Pass 1: bucket by canonical_key
        merged: dict[str, dict[str, Any]] = {}
        query_types_matched: dict[str, list[str]] = {}
        for qtype, art in records:
            key = art.get("canonical_key")
            if not key:
                continue
            if key in merged:
                merged[key] = _merge(merged[key], art)
            else:
                merged[key] = dict(art)
            qts = query_types_matched.setdefault(key, [])
            if qtype not in qts:
                qts.append(qtype)

        # Pass 2: merge across canonical_keys that share strong identifiers
        # (PMID/DOI/PMCID). This catches cross-source duplicates where one
        # record was keyed by PMID and another by DOI.
        id_to_ck: dict[str, str] = {}
        # Iterate over a stable copy of keys; merge later entries into earlier ones.
        for ck in list(merged.keys()):
            if ck not in merged:
                continue
            art = merged[ck]
            id_keys = _id_keys_for(art)
            target_ck: str | None = None
            for idk in id_keys:
                existing_ck = id_to_ck.get(idk)
                if existing_ck and existing_ck != ck and existing_ck in merged:
                    target_ck = existing_ck
                    break
            if target_ck is not None:
                # Merge ck into target_ck.
                merged[target_ck] = _merge(merged[target_ck], merged.pop(ck))
                qts_a = query_types_matched.pop(ck, [])
                for q in qts_a:
                    if q not in query_types_matched.get(target_ck, []):
                        query_types_matched.setdefault(target_ck, []).append(q)
                # Update id index to point all ids of ck and target to target_ck.
                for idk in _id_keys_for(merged[target_ck]):
                    id_to_ck[idk] = target_ck
            else:
                # Register all this record's ids under its own canonical key.
                for idk in id_keys:
                    id_to_ck.setdefault(idk, ck)

        return list(merged.values()), query_types_matched


deduplication = DeduplicationService()
