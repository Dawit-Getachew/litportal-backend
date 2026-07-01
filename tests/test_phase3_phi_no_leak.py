"""Phase 3 — PHI + no-LLM-leak regression (Task 11 / G)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tests._phase3_helpers import (  # noqa: E402
    signup, patch_llm_configured,
)


@pytest.mark.asyncio
async def test_phi_blocked_clinical_question_never_calls_summarizer_llm(client):
    t, _ = await signup(client, "phi1@test.com")
    h = {"Authorization": f"Bearer {t}"}
    with patch_llm_configured() as prov:
        mock = AsyncMock(return_value=None)
        prov.complete_json = mock  # type: ignore[assignment]
        r = await client.post("/api/search-sessions/", headers=h, json={
            "question": "45-year-old male, MRN 12345678, DOB 03/14/1979, contact patient@example.com — best tx?",
        })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "blocked_phi"
    # The LLM must NEVER be invoked for a PHI-blocked question.
    assert mock.call_count == 0


@pytest.mark.asyncio
async def test_phi_blocked_clinical_question_never_calls_verifier_llm(client):
    """Same guarantee, covering the verifier LLM path."""
    t, _ = await signup(client, "phi2@test.com")
    h = {"Authorization": f"Bearer {t}"}
    with patch_llm_configured() as prov:
        mock = AsyncMock(return_value=None)
        prov.complete_json = mock  # type: ignore[assignment]
        r = await client.post("/api/search-sessions/", headers=h, json={
            "question": "Patient Mr. John Smith — review evidence for SGLT2",
        })
    body = r.json()
    assert body["status"] == "blocked_phi"
    assert mock.call_count == 0


@pytest.mark.asyncio
async def test_upload_pipeline_still_never_calls_llm(client):
    """Regression: PDF upload extraction is deterministic and pure-Python."""
    from unittest.mock import AsyncMock
    from tests.test_phase2d_uploads import _make_session_with_article, _do_upload
    from tests.fixtures_pdf import make_apixaban_pdf
    t, _ = await signup(client, "phi3@test.com")
    h = {"Authorization": f"Bearer {t}"}
    sid, aid, _ = await _make_session_with_article(
        client, h, doi="10.1056/NEJMoa1107039", pmid="21870978", year=2011,
        title="Apixaban versus warfarin in patients with atrial fibrillation",
    )
    with patch_llm_configured() as prov:
        mock = AsyncMock(return_value=None)
        prov.complete_json = mock  # type: ignore[assignment]
        r = await _do_upload(client, h, sid, aid, make_apixaban_pdf())
    assert r.status_code == 200
    assert mock.call_count == 0
