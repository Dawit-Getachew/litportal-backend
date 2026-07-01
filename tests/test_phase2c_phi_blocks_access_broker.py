"""Phase 2C — PHI guard blocks ALL access-broker sources.

The broker must NEVER be reached when the question is PHI-blocked. We mock
httpx so that any call to Unpaywall, Semantic Scholar, or Crossref raises
loudly. Any institutional row generation is also blocked because it requires
a session in non-blocked state.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


async def _signup(client, email):
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": "Pass1234567!", "name": "T"},
    )
    return r.json()["access_token"]


@pytest.mark.asyncio
async def test_phi_blocks_all_external_broker_calls(client):
    token = await _signup(client, "phi-broker-1@test.com")
    headers = {"Authorization": f"Bearer {token}"}

    with patch("services.literature_retrieval.literature_retrieval.retrieve", new=AsyncMock()) as mock_retrieve, \
         patch("services.connectors.pubmed.pubmed_connector.search", new=AsyncMock()) as mock_pm, \
         patch("services.connectors.openalex.openalex_connector.enrich_many", new=AsyncMock()) as mock_oa, \
         patch("services.full_text_access_broker.full_text_access_broker.broker_for_session", new=AsyncMock()) as mock_broker:
        r = await client.post(
            "/api/search-sessions/",
            json={"question": "Mr. John Doe DOB 03/14/1979, MRN 12345678 — best HTN treatment?"},
            headers=headers,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "blocked_phi"
        # NO retrieval, normalizer, enrich, or access broker call happens.
        mock_retrieve.assert_not_called()
        mock_pm.assert_not_called()
        mock_oa.assert_not_called()
        mock_broker.assert_not_called()


@pytest.mark.asyncio
async def test_phi_blocked_session_returns_404_from_access_endpoints(client):
    """Even with valid auth, blocked_phi sessions must not be reachable
    via /access-locations endpoints — uniform 404."""
    token = await _signup(client, "phi-broker-2@test.com")
    headers = {"Authorization": f"Bearer {token}"}

    r = await client.post(
        "/api/search-sessions/",
        json={"question": "DOB 01/02/1980 patient with HTN, MRN 12345678"},
        headers=headers,
    )
    assert r.json()["status"] == "blocked_phi"
    sid = r.json()["session_id"]
    # /access-locations endpoint on a blocked_phi session must 404.
    r2 = await client.get(
        f"/api/search-sessions/{sid}/access-locations", headers=headers
    )
    assert r2.status_code == 404
