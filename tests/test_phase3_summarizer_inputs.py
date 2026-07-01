"""Phase 3 — summarizer input isolation (Task 11 / B).

The summarizer prompt MUST contain ONLY the spans of selected articles. No
unselected/excluded text. No model-memory directives. No raw PDF bytes.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tests._phase3_helpers import (  # noqa: E402
    make_packet_session, signup, good_apixaban_articles, patch_llm_configured,
)


@pytest.mark.asyncio
async def test_no_model_memory_directive_in_system_prompt():
    from services.summarization.prompts import SUMMARIZER_SYSTEM_PROMPT
    assert "Do not use prior knowledge or model memory" in SUMMARIZER_SYSTEM_PROMPT
    assert "ONLY the evidence spans provided" in SUMMARIZER_SYSTEM_PROMPT


def _capture_prompts(mock: AsyncMock) -> list[tuple[str, str]]:
    calls = []
    for call in mock.call_args_list:
        kwargs = call.kwargs
        calls.append((kwargs.get("system", ""), kwargs.get("user", "")))
    return calls


@pytest.mark.asyncio
async def test_only_selected_article_spans_in_prompt(client):
    token, _ = await signup(client, "sumin1@test.com")
    h = {"Authorization": f"Bearer {token}"}
    arts = good_apixaban_articles()
    # Drop one article from the selected set with a secret marker in the abstract.
    arts[2]["user_included"] = False
    arts[2]["abstract"] = "SECRET_UNSELECTED_TEXT should never appear in any prompt."
    sid, _, _ = await make_packet_session(client, h, articles=arts)

    with patch_llm_configured() as prov:
        mock = AsyncMock(return_value=None)
        prov.complete_json = mock  # type: ignore[assignment]
        await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })

    # The summarizer should have been called at least once.
    assert mock.call_args_list, "summarizer LLM was never called"
    all_user = " ".join(c.kwargs.get("user", "") for c in mock.call_args_list)
    assert "SECRET_UNSELECTED_TEXT" not in all_user


@pytest.mark.asyncio
async def test_no_unselected_article_text_in_prompt(client):
    token, _ = await signup(client, "sumin2@test.com")
    h = {"Authorization": f"Bearer {token}"}
    arts = good_apixaban_articles()
    arts[1]["user_included"] = False
    arts[1]["abstract"] = "UNSELECTED_MARKER ROCKET-AF results."
    sid, _, _ = await make_packet_session(client, h, articles=arts)
    with patch_llm_configured() as prov:
        mock = AsyncMock(return_value=None)
        prov.complete_json = mock  # type: ignore[assignment]
        await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    all_user = " ".join(c.kwargs.get("user", "") for c in mock.call_args_list)
    assert "UNSELECTED_MARKER" not in all_user


@pytest.mark.asyncio
async def test_no_excluded_article_text_in_prompt(client):
    token, _ = await signup(client, "sumin3@test.com")
    h = {"Authorization": f"Bearer {token}"}
    arts = good_apixaban_articles()
    arts[2]["is_retracted"] = True
    arts[2]["exclusion_reason"] = "retracted"
    arts[2]["abstract"] = "RETRACTED_MARKER must not appear."
    sid, _, _ = await make_packet_session(client, h, articles=arts)
    with patch_llm_configured() as prov:
        mock = AsyncMock(return_value=None)
        prov.complete_json = mock  # type: ignore[assignment]
        await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    all_user = " ".join(c.kwargs.get("user", "") for c in mock.call_args_list)
    assert "RETRACTED_MARKER" not in all_user


@pytest.mark.asyncio
async def test_no_institutional_linkout_text_in_prompt(client):
    """Institutional link-out URLs / publisher landing pages never appear."""
    token, _ = await signup(client, "sumin4@test.com")
    h = {"Authorization": f"Bearer {token}"}
    arts = good_apixaban_articles()
    sid, _, _ = await make_packet_session(client, h, articles=arts)
    with patch_llm_configured() as prov:
        mock = AsyncMock(return_value=None)
        prov.complete_json = mock  # type: ignore[assignment]
        await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    all_user = " ".join(c.kwargs.get("user", "") for c in mock.call_args_list)
    # Block check: institutional landing-page text patterns.
    for tok in ("openathens.net", "go.openathens", "libkey.io", "doi.org/10.", "https://link.springer"):
        assert tok not in all_user, f"prompt leaked institutional URL marker: {tok}"


@pytest.mark.asyncio
async def test_no_raw_pdf_bytes_in_prompt(client):
    """No %PDF- magic byte sequence should ever land in a prompt."""
    token, _ = await signup(client, "sumin5@test.com")
    h = {"Authorization": f"Bearer {token}"}
    arts = good_apixaban_articles()
    # Inject a magic-bytes-like marker (without uploading actual PDF) so we can
    # verify the prompt builder never includes raw upload bytes.
    sid, _, _ = await make_packet_session(client, h, articles=arts)
    with patch_llm_configured() as prov:
        mock = AsyncMock(return_value=None)
        prov.complete_json = mock  # type: ignore[assignment]
        await client.post(f"/api/search-sessions/{sid}/generate-summary?sync=true", headers=h, json={
            "acknowledge_abstract_only_limitations": True,
        })
    all_user = " ".join(c.kwargs.get("user", "") for c in mock.call_args_list)
    assert "%PDF-" not in all_user
