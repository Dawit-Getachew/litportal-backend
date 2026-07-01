"""HTTP client for the Scienthesis LitHub Service.

Ported from litbridge-backend/src/clients/lithub_client.py and adapted to this
backend (stdlib logging, core.config settings). LitPortal mirrors library saves
into LitHub via the service-token internal API, keyed by the user's Identity
``sub``, so papers saved on LitPulse and LitPortal share one central library.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import httpx

from core.config import Settings, get_settings
from clients.service_token import mint_service_token

logger = logging.getLogger("litportal.clients.lithub")

_LITHUB_AUDIENCE = "scienthesis-lithub"


class LitHubClientError(Exception):
    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"LitHub {status_code}: {detail}")


class LitHubUpstreamError(Exception):
    """LitHub unreachable or 5xx."""


class LitHubClient:
    """Async client for LitHub internal (service-token) endpoints."""

    def __init__(self, http_client: httpx.AsyncClient, settings: Settings | None = None) -> None:
        self._client = http_client
        self._settings = settings or get_settings()
        # Tolerate an unset base at construction so the client can be created
        # without crashing an un-configured deploy; a real request with no base
        # raises at call time instead.
        self._base = (self._settings.LITHUB_BASE_URL or "").rstrip("/")

    def _require_base(self) -> str:
        if not self._base:
            raise LitHubUpstreamError("LITHUB_BASE_URL is not configured.")
        return self._base

    # ── Internal (service-token) endpoints ─────────────────────────

    async def internal_save_paper(
        self, user_id: UUID | str, item: dict[str, Any],
    ) -> dict[str, Any]:
        """Service-to-service save on behalf of a user (Identity sub).

        Returns the canonical ``paper_id`` / ``library_entry_id`` so the caller
        can persist the cross-service link. Idempotent on the LitHub side.
        """
        return await self._internal_post(
            "/api/v1/internal/library/save",
            {"user_id": str(user_id), "item": item},
        )

    async def internal_list_library(
        self, user_id: UUID | str, *, params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Read a user's central LitHub library (keyed by Identity sub)."""
        q: dict[str, Any] = {"user_id": str(user_id)}
        if params:
            q.update({k: v for k, v in params.items() if v is not None})
        return await self._internal_get("/api/v1/internal/library", params=q)

    async def internal_membership(
        self, user_id: UUID | str, *, pmid: str | None = None, doi: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"user_id": str(user_id)}
        if pmid:
            params["pmid"] = pmid
        if doi:
            params["doi"] = doi
        return await self._internal_get(
            "/api/v1/internal/library/membership", params=params,
        )

    async def internal_delete_library(
        self, user_id: UUID | str, *, pmid: str | None = None, doi: str | None = None,
    ) -> dict[str, Any]:
        """Idempotent service-to-service delete of a paper from a user's library."""
        params: dict[str, Any] = {"user_id": str(user_id)}
        if pmid:
            params["pmid"] = pmid
        if doi:
            params["doi"] = doi
        return await self._internal_delete("/api/v1/internal/library", params=params)

    async def internal_papers_bulk(
        self, paper_ids: list[UUID | str],
    ) -> list[dict[str, Any]]:
        """Fetch canonical paper metadata by id (service token)."""
        if not paper_ids:
            return []
        result = await self._internal_post(
            "/api/v1/internal/papers/bulk",
            {"paper_ids": [str(pid) for pid in paper_ids]},
        )
        return result.get("papers", []) if isinstance(result, dict) else []

    async def internal_bulk_import(
        self, user_id: UUID | str, items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return await self._internal_post(
            "/api/v1/internal/library/bulk-import",
            {"user_id": str(user_id), "items": items},
        )

    # ── Helpers ────────────────────────────────────────────────────

    async def _internal_get(
        self, path: str, params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        token = mint_service_token(_LITHUB_AUDIENCE)
        return await self._request(
            "GET", path, params=params, headers={"X-Service-Token": token},
        )

    async def _internal_post(
        self, path: str, body: dict[str, Any],
    ) -> dict[str, Any]:
        token = mint_service_token(_LITHUB_AUDIENCE)
        return await self._request(
            "POST", path, json=body, headers={"X-Service-Token": token},
        )

    async def _internal_delete(
        self, path: str, params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        token = mint_service_token(_LITHUB_AUDIENCE)
        return await self._request(
            "DELETE", path, params=params, headers={"X-Service-Token": token},
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._require_base()}{path}"
        try:
            resp = await self._client.request(
                method, url, json=json, params=params, headers=headers,
            )
        except httpx.HTTPError as exc:
            logger.error("lithub_request_transport_failed path=%s error=%s", path, exc)
            raise LitHubUpstreamError(
                f"Could not reach LitHub at {url}: {exc}",
            ) from exc

        if resp.status_code >= 500:
            raise LitHubUpstreamError(
                f"LitHub returned {resp.status_code}: {resp.text[:200]}",
            )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise LitHubClientError(status_code=resp.status_code, detail=detail)
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()
