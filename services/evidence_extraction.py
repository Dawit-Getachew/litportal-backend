"""EvidenceExtractionService — Phase 2D + Phase 3.

Ensures `article_text_extracts` rows exist for each article that needs
downstream consideration. NEVER calls an LLM. NEVER scrapes. Conservatively
falls back to abstract-only when no user-uploaded full text is present.

Priority order:
  1. user_uploaded_full_text — already persisted by UserUploadedFullTextService
  2. oa_full_text — Phase 4 work (TODO; only honored if PMC text is already
     cached locally — no fetch at extraction time)
  3. abstract — always available; created here if missing.

Phase 3: `prepare_evidence_packet` also returns the summarization-readiness
view (Task 2) consumed by the Generate-Summary button and pipeline.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from core.db import Collections, get_db

logger = logging.getLogger("clinevidence.extraction")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _token_count(text: str) -> int:
    return (len(text.split()) * 4) // 3


class EvidenceExtractionService:
    """Prepare evidence_text_extracts for downstream summarization."""

    async def ensure_abstract_extract(
        self,
        *,
        article: dict[str, Any],
        search_session_id: str,
        user_id: str | None,
        expires_at: datetime | None,
    ) -> dict[str, Any] | None:
        """Create an abstract extract for this session+article if missing.

        Skips if a user_uploaded_full_text or abstract extract already exists
        for this article in this session. Returns the new row (or existing).
        """
        db = get_db()
        existing = await db[Collections.ARTICLE_TEXT_EXTRACTS].find_one(
            {
                "search_session_id": search_session_id,
                "article_id": article["id"],
            },
            {"_id": 0},
        )
        if existing:
            return existing
        abstract = (article.get("abstract") or "").strip()
        if not abstract:
            return None
        doc = {
            "id": uuid.uuid4().hex,
            "article_id": article["id"],
            "uploaded_full_text_asset_id": None,
            "search_session_id": search_session_id,
            "user_id": user_id,
            "text_source": "abstract",
            "evidence_text_level": "abstract_only",
            "section_name": "abstract",
            "section_text": abstract,
            "token_count": _token_count(abstract),
            "include_in_summary": True,
            "created_at": _now(),
            "expires_at": expires_at,
        }
        await db[Collections.ARTICLE_TEXT_EXTRACTS].insert_one(doc)
        return doc

    async def prepare_evidence_packet(
        self,
        *,
        session: dict[str, Any],
        user_id: str,
    ) -> dict[str, Any]:
        """Ensure each selected article has at least an abstract extract.

        Returns a readiness summary suitable for /prepare-evidence-packet.
        Does NOT call any LLM and does NOT fetch any external text.

        Phase 2D counter rules:
          - `with_user_uploaded_full_text` counts articles by SOURCE PRESENCE
            (any extract row with evidence_text_level=user_uploaded_full_text),
            regardless of include_in_summary on the underlying sections.
          - `with_oa_full_text` counts articles by evidence_text_level=oa_full_text.
          - `with_abstract_only` counts articles whose only text source is
            abstract (no user-uploaded / oa rows).
          - `ready_for_summary_sections_count` counts every extract row with
            include_in_summary=true across the selected articles.
          - `summary_eligible_articles_count` counts articles that have at
            least one include_in_summary=true extract.
          - `ready_for_future_summary = summary_eligible_articles_count >= 1`.
        """
        db = get_db()
        sid = session["id"]
        sa_rows = await (
            db[Collections.SEARCH_SESSION_ARTICLES]
            .find(
                {
                    "search_session_id": sid,
                    "user_included": True,
                    "exclusion_reason": None,
                },
                {"_id": 0, "article_id": 1},
            )
            .to_list(length=200)
        )
        article_ids = [r["article_id"] for r in sa_rows]
        articles_by_id: dict[str, dict] = {}
        if article_ids:
            async for art in db[Collections.ARTICLES].find(
                {"id": {"$in": article_ids}}, {"_id": 0}
            ):
                articles_by_id[art["id"]] = art

        with_abstract_only = 0
        with_user_full_text = 0
        with_oa_full_text = 0
        with_no_text = 0
        total_tokens = 0
        ready_for_summary_sections_count = 0
        summary_eligible_articles_count = 0
        expires_at = session.get("expires_at")
        # Phase 4A: inline OA fetch for selected articles that don't yet have
        # an OA extract. Errors are tolerated; each article that fails falls
        # back to abstract. Hard-capped by OA_MAX_FULL_TEXT_FETCHES_PER_SEARCH.
        failed_oa_fetch_count = 0
        oa_fetch_warnings: list[dict] = []
        if os.environ.get("OA_FETCHER_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}:
            try:
                from services.oa_fetcher import oa_fetcher as _oa
                from services.oa_fetcher.license_gate import is_article_eligible
                # Find articles that don't yet have OA extracts in this session.
                existing_oa = set(await db[Collections.ARTICLE_TEXT_EXTRACTS].distinct(
                    "article_id",
                    {"search_session_id": sid, "text_source": "oa_full_text"},
                ))
                to_fetch: list[str] = []
                for aid in article_ids:
                    if aid in existing_oa:
                        continue
                    art = articles_by_id.get(aid) or {}
                    eligible, _ = is_article_eligible(art)
                    if eligible:
                        to_fetch.append(aid)
                if to_fetch:
                    res = await _oa.fetch_for_articles(
                        article_ids=to_fetch, session_id=sid, user_id=user_id,
                    )
                    for r in res.get("results", []):
                        if r["status"] in ("failed", "skipped_license"):
                            failed_oa_fetch_count += 1
                            oa_fetch_warnings.append({
                                "article_id": r["article_id"],
                                "reason": r.get("error_code") or r["status"],
                            })
            except Exception:  # noqa: BLE001
                logger.exception("inline OA fetch failed (non-fatal)")
        for aid in article_ids:
            art = articles_by_id.get(aid)
            if not art:
                with_no_text += 1
                continue
            # Find ALL extracts for this article in this session. User uploads
            # are scoped to the requesting user.
            extracts = await db[Collections.ARTICLE_TEXT_EXTRACTS].find(
                {
                    "search_session_id": sid,
                    "article_id": aid,
                    "$or": [
                        {"text_source": "user_uploaded_full_text", "user_id": user_id},
                        {"text_source": "oa_full_text"},
                        {"text_source": "abstract"},
                    ],
                },
                {"_id": 0, "token_count": 1, "evidence_text_level": 1,
                 "text_source": 1, "include_in_summary": 1},
            ).to_list(length=500)

            has_user = any(e.get("evidence_text_level") == "user_uploaded_full_text" for e in extracts)
            has_oa = any(e.get("evidence_text_level") == "oa_full_text" for e in extracts)
            has_abstract = any(e.get("text_source") == "abstract" for e in extracts)

            if has_user:
                with_user_full_text += 1
            elif has_oa:
                with_oa_full_text += 1
            elif has_abstract:
                with_abstract_only += 1
            else:
                # Fall back: create an abstract extract from article.abstract.
                row = await self.ensure_abstract_extract(
                    article=art,
                    search_session_id=sid,
                    user_id=user_id,
                    expires_at=expires_at,
                )
                if row and (row.get("section_text") or "").strip():
                    with_abstract_only += 1
                    extracts.append({
                        "token_count": row.get("token_count") or 0,
                        "evidence_text_level": "abstract_only",
                        "text_source": "abstract",
                        "include_in_summary": bool(row.get("include_in_summary")),
                    })
                    has_abstract = True
                else:
                    with_no_text += 1

            # Token totals + summary-eligibility — counted per article.
            eligible_rows = [e for e in extracts if e.get("include_in_summary")]
            ready_for_summary_sections_count += len(eligible_rows)
            total_tokens += sum(int(e.get("token_count") or 0) for e in eligible_rows)
            if eligible_rows:
                summary_eligible_articles_count += 1

        ready = summary_eligible_articles_count >= 1
        return {
            "selected_articles_count": len(article_ids),
            "with_abstract_only": with_abstract_only,
            "with_user_uploaded_full_text": with_user_full_text,
            "with_oa_full_text": with_oa_full_text,
            "with_no_text": with_no_text,
            "ready_for_summary_sections_count": ready_for_summary_sections_count,
            "summary_eligible_articles_count": summary_eligible_articles_count,
            "total_tokens": total_tokens,
            "ready_for_future_summary": ready,
            # Phase 3 — Task 2 keys (richer view used by Generate Summary).
            "usable_articles_count": summary_eligible_articles_count,
            "abstract_only_count": with_abstract_only,
            "oa_full_text_count": with_oa_full_text,
            "user_uploaded_full_text_count": with_user_full_text,
            "metadata_only_count": with_no_text,
            "total_extract_sections": ready_for_summary_sections_count,
            "total_token_estimate": total_tokens,
            "ready_for_summary": ready and (session.get("status") in ("awaiting_selection", "complete")),
            "block_reason": (
                None if ready and session.get("status") == "awaiting_selection"
                else "session_expired" if session.get("deleted_at")
                else "phi_blocked_session" if session.get("status") == "blocked_phi"
                else "no_selected_articles" if len(article_ids) == 0
                else "no_usable_evidence" if not ready
                else None
            ),
            # Phase 4A — OA fetcher visibility on the packet panel.
            "failed_oa_fetch_count": failed_oa_fetch_count,
            "oa_fetch_warnings": oa_fetch_warnings,
            "note": (
                "Evidence packet prepared. Summarization will run after the "
                "summarization and citation verification phase."
            ),
        }


evidence_extraction = EvidenceExtractionService()
