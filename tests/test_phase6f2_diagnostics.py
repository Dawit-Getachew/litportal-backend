"""Phase 6F.2 — Diagnostics Instrumentation tests.

Verifies the closed-set contract for the three previously-null fields on
``search_sessions.session_diagnostics_json``:

  * ``llm_calls_count`` (int) + ``llm_calls_by_stage`` (dict, closed-set keys)
  * ``token_estimate`` ``{prompt_tokens, completion_tokens, total_tokens,
    estimate_method}`` where ``estimate_method`` ∈ closed-set
  * ``latency_by_stage_json`` ``{stage_name: ms_int}`` — closed-set keys

Safety invariants under test:

  * **No PHI, no prompts, no completions** ever land in the diagnostic
    fields. Only integers + closed-set strings.
  * Stage keys are restricted to ``STAGE_NAMES`` — ``"unknown"`` and any
    non-closed-set string never appear.
  * The PHI-blocked path stores zero LLM calls and only the
    ``phi_guard`` latency stage (the route handler pre-stamps this
    BEFORE creating the row; no pipeline runs).
  * The clean path runs the pipeline and stamps every closed-set
    stage that actually executes; ``llm_calls_count`` is an int even
    when no LLM is configured (it just stays 0).

These tests do NOT call PubMed or Europe PMC — they exercise the
diagnostics contract directly via the in-memory ``SessionDiagnostics``
class plus a thin integration check over the live FastAPI app using
``TestClient``.
"""
from __future__ import annotations

import pytest

from services.diagnostics import (
    ESTIMATE_METHODS,
    STAGE_NAMES,
    SessionDiagnostics,
    bind as diag_bind,
    reset as diag_reset,
)


pytestmark = pytest.mark.xdist_group(name="phase6f2_diagnostics")


# ---------------------------------------------------------------------------
# 1. SessionDiagnostics closed-set contract
# ---------------------------------------------------------------------------


class TestSessionDiagnosticsContract:
    def test_to_dict_has_closed_set_keys_only(self):
        d = SessionDiagnostics()
        out = d.to_dict()
        assert set(out.keys()) == {
            "llm_calls_count",
            "llm_calls_by_stage",
            "token_estimate",
            "latency_by_stage_json",
        }
        assert isinstance(out["llm_calls_count"], int)
        assert isinstance(out["llm_calls_by_stage"], dict)
        assert isinstance(out["token_estimate"], dict)
        assert isinstance(out["latency_by_stage_json"], dict)
        assert set(out["token_estimate"].keys()) == {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "estimate_method",
        }
        assert out["token_estimate"]["estimate_method"] in ESTIMATE_METHODS

    def test_unknown_stage_raises_on_stage_timer(self):
        d = SessionDiagnostics()
        with pytest.raises(ValueError):
            with d.stage("not_a_real_stage_name"):
                pass

    def test_llm_call_outside_stage_is_filtered_to_no_stage_bucket(self):
        # When no stage is active, the call counts toward
        # ``llm_calls_count`` but NOT toward ``llm_calls_by_stage``
        # (the ``unknown`` placeholder is intentionally NOT in
        # STAGE_NAMES so it gets dropped from the persisted dict).
        d = SessionDiagnostics()
        d.record_llm_call(prompt_tokens=10, completion_tokens=5,
                          method="provider_reported")
        out = d.to_dict()
        assert out["llm_calls_count"] == 1
        assert out["llm_calls_by_stage"] == {}
        # "unknown" must NEVER appear in the persisted closed-set keys.
        assert "unknown" not in out["llm_calls_by_stage"]
        assert "unknown" not in out["latency_by_stage_json"]

    def test_llm_call_inside_stage_attributes_to_that_stage(self):
        d = SessionDiagnostics()
        with d.stage("summarizer"):
            d.record_llm_call(prompt_tokens=100, completion_tokens=50,
                              method="provider_reported")
        with d.stage("citation_verifier"):
            d.record_llm_call(prompt_tokens=80, completion_tokens=20,
                              method="provider_reported")
            d.record_llm_call(prompt_tokens=70, completion_tokens=25,
                              method="provider_reported")
        out = d.to_dict()
        assert out["llm_calls_count"] == 3
        assert out["llm_calls_by_stage"] == {
            "summarizer": 1, "citation_verifier": 2,
        }
        # Every stage key in ``latency_by_stage_json`` is in STAGE_NAMES.
        for stage_key in out["latency_by_stage_json"].keys():
            assert stage_key in STAGE_NAMES
        # Every stage key in ``llm_calls_by_stage`` is in STAGE_NAMES.
        for stage_key in out["llm_calls_by_stage"].keys():
            assert stage_key in STAGE_NAMES

    def test_token_estimate_method_label_is_correct(self):
        d = SessionDiagnostics()
        with d.stage("summarizer"):
            d.record_llm_call(prompt_tokens=10, completion_tokens=5,
                              method="provider_reported")
        out = d.to_dict()
        assert out["token_estimate"]["estimate_method"] == "provider_reported"
        assert out["token_estimate"]["prompt_tokens"] == 10
        assert out["token_estimate"]["completion_tokens"] == 5
        assert out["token_estimate"]["total_tokens"] == 15

        d2 = SessionDiagnostics()
        with d2.stage("summarizer"):
            d2.record_llm_call(prompt_tokens=10, completion_tokens=5,
                               method="char_div_4_estimate")
        out2 = d2.to_dict()
        assert out2["token_estimate"]["estimate_method"] == "char_div_4_estimate"

        d3 = SessionDiagnostics()
        with d3.stage("summarizer"):
            d3.record_llm_call(prompt_tokens=10, completion_tokens=5,
                               method="provider_reported")
            d3.record_llm_call(prompt_tokens=10, completion_tokens=5,
                               method="char_div_4_estimate")
        out3 = d3.to_dict()
        assert out3["token_estimate"]["estimate_method"] == "mixed"

    def test_stage_latency_accumulates_across_invocations(self):
        d = SessionDiagnostics()
        with d.stage("pubmed_retrieval"):
            pass
        with d.stage("pubmed_retrieval"):
            pass
        out = d.to_dict()
        # Two invocations both land in pubmed_retrieval bucket.
        assert "pubmed_retrieval" in out["latency_by_stage_json"]
        assert isinstance(out["latency_by_stage_json"]["pubmed_retrieval"], int)
        assert out["latency_by_stage_json"]["pubmed_retrieval"] >= 0


# ---------------------------------------------------------------------------
# 2. PHI / prompt / completion redaction — strict safety invariant
# ---------------------------------------------------------------------------


class TestPhiAndPromptRedaction:
    def test_diagnostic_blob_never_contains_prompts_or_completions(self):
        # Even when we record a call with a giant prompt + completion,
        # the persisted dict must carry ONLY closed-set numerics +
        # closed-set strings. NEVER the raw text.
        from services.llm_provider import _record_diagnostics_call
        d = SessionDiagnostics()
        token = diag_bind(d)
        try:
            with d.stage("summarizer"):
                phi_like_prompt = (
                    "Patient John Doe MRN 12345678 DOB 03/14/1979 "
                    "email john@hospital.example treated with metformin."
                )
                completion = (
                    "{\"answer\": \"This text MUST NEVER leak into diagnostics.\"}"
                )
                _record_diagnostics_call(
                    prompt_text=phi_like_prompt,
                    completion_text=completion,
                    usage={"prompt_tokens": 42, "completion_tokens": 18},
                )
        finally:
            diag_reset(token)
        out = d.to_dict()
        # Serialize and grep — no PHI / prompt / completion text leaks.
        blob = repr(out).lower()
        for needle in [
            "john doe", "mrn", "12345678", "1979", "@hospital",
            "metformin", "never leak", "answer", "patient",
        ]:
            assert needle not in blob, f"diagnostics blob leaked {needle!r}"
        # And the structure is still correct.
        assert out["llm_calls_count"] == 1
        assert out["llm_calls_by_stage"]["summarizer"] == 1
        assert out["token_estimate"]["prompt_tokens"] == 42
        assert out["token_estimate"]["completion_tokens"] == 18

    def test_diagnostics_never_breaks_pipeline_on_bad_usage(self):
        # Bogus usage values must not raise — diagnostics must never
        # break the LLM provider or the pipeline.
        from services.llm_provider import _record_diagnostics_call
        d = SessionDiagnostics()
        token = diag_bind(d)
        try:
            with d.stage("summarizer"):
                _record_diagnostics_call(
                    prompt_text="hi", completion_text="world",
                    usage={"prompt_tokens": "garbage", "completion_tokens": None},
                )
        finally:
            diag_reset(token)
        out = d.to_dict()
        # Falls back to char/4 estimate when usage is unparseable.
        assert out["llm_calls_count"] == 1
        assert out["token_estimate"]["estimate_method"] == "char_div_4_estimate"


# ---------------------------------------------------------------------------
# 3. End-to-end via async TestClient: PHI-blocked path + clean draft path.
# ---------------------------------------------------------------------------


async def _login_or_signup(client, *, email: str, password: str, name: str = "Test User"):
    # Try signup first; if email already exists, login.
    r = await client.post(
        "/api/auth/signup",
        json={"email": email, "password": password, "name": name},
    )
    if r.status_code == 201:
        return r.json()["access_token"]
    r2 = await client.post(
        "/api/auth/login", json={"email": email, "password": password},
    )
    assert r2.status_code == 200, r2.text
    return r2.json()["access_token"]


# ---------------------------------------------------------------------------
# 4. Phase 6F.2 clean-up — pre-PICOTT anchor pre-flight regression.
# ---------------------------------------------------------------------------


class TestPrePicottAnchorPreflight:
    """Unit-level coverage for the deterministic ``is_generic_only``
    pre-flight that decides whether the LLM PICOTT extractor fires.
    Closed-set: every case below is locked against the closed-set
    ``GENERIC_MODIFIERS`` allowlist + the question-opener function
    tokens. Adding a new closed-set modifier MUST come with a case
    here.
    """

    @pytest.mark.parametrize("question", [
        "What are the side effects?",
        "What is first-line treatment?",
        "What is the dose?",
        "What is the recommended age?",
        "What is treatment options?",
        "side effects",
        "first-line treatment",
        "What is the safety?",
        "What is the efficacy?",
    ])
    def test_generic_only_questions_return_true(self, question):
        from services.primary_clinical_anchor import (
            primary_clinical_anchor_service,
        )
        assert primary_clinical_anchor_service.is_generic_only(question) is True

    @pytest.mark.parametrize("question", [
        "What are the side effects of quetiapine?",
        "What is the first-line treatment for psoriasis?",
        "What is the dose of metformin?",
        "Does SGLT2 inhibitor reduce HFpEF mortality?",
        "What is the recommended age for the HPV vaccine?",
        "What is the evidence for atorvastatin in primary prevention?",
    ])
    def test_non_generic_questions_return_false(self, question):
        from services.primary_clinical_anchor import (
            primary_clinical_anchor_service,
        )
        assert primary_clinical_anchor_service.is_generic_only(question) is False

    @pytest.mark.parametrize("question", ["", "   "])
    def test_empty_or_whitespace_returns_true_short_circuits_picott(self, question):
        # Empty / whitespace-only questions carry no clinical signal —
        # the pre-flight returns True so the LLM PICOTT extractor is
        # skipped (graceful degradation; user gets the clarification
        # banner).
        from services.primary_clinical_anchor import (
            primary_clinical_anchor_service,
        )
        assert primary_clinical_anchor_service.is_generic_only(question) is True


@pytest.mark.asyncio
class TestGenericOnlyAttribution:
    """Phase 6F.2 clean-up regression — for a generic-only question,
    the LLM PICOTT extractor MUST NOT fire. Diagnostics must show:
      * ``llm_calls_count == 0``
      * ``picott_parsing`` NOT in ``llm_calls_by_stage``
      * ``latency_by_stage_json`` contains ONLY closed-set stages from
        the pre-PICOTT path: a subset of
        ``{phi_guard, normalization, primary_clinical_anchor}``.
      * No retrieval / ranking / summarizer stages appear.
    """

    async def test_generic_only_question_skips_picott_llm_call(self, client):
        import asyncio
        import uuid
        email = f"phi-generic-{uuid.uuid4().hex[:8]}@test.com"
        token = await _login_or_signup(
            client, email=email, password="TestPass1234!",
        )
        r = await client.post(
            "/api/search-sessions/",
            headers={"Authorization": f"Bearer {token}"},
            json={"question": "What are the side effects?"},
        )
        assert r.status_code == 200
        sid = r.json()["id"]
        # Poll until pipeline lands at awaiting_clinical_subject.
        diag: dict = {}
        body: dict = {}
        for _ in range(20):
            r2 = await client.get(
                f"/api/search-sessions/{sid}",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert r2.status_code == 200
            body = r2.json()
            diag = body.get("session_diagnostics_json") or {}
            if (
                body.get("status") == "awaiting_clinical_subject"
                and isinstance(diag.get("llm_calls_count"), int)
            ):
                break
            await asyncio.sleep(0.5)
        assert body.get("status") == "awaiting_clinical_subject", (
            f"expected awaiting_clinical_subject, got {body.get('status')}"
        )
        # Hard contract: zero LLM calls.
        assert diag.get("llm_calls_count") == 0, (
            f"PICOTT extractor leaked: llm_calls_count={diag.get('llm_calls_count')}, "
            f"by_stage={diag.get('llm_calls_by_stage')}"
        )
        # picott_parsing MUST NOT appear in either dict.
        by_stage = diag.get("llm_calls_by_stage") or {}
        assert "picott_parsing" not in by_stage, (
            f"picott_parsing leaked into llm_calls_by_stage: {by_stage}"
        )
        lat = diag.get("latency_by_stage_json") or {}
        assert "picott_parsing" not in lat, (
            f"picott_parsing leaked into latency_by_stage_json: {lat}"
        )
        # Latency keys are a subset of the pre-PICOTT closed-set.
        allowed_pre_picott = {
            "phi_guard", "normalization", "primary_clinical_anchor",
        }
        assert set(lat.keys()).issubset(allowed_pre_picott), (
            f"unexpected stages leaked: {set(lat.keys()) - allowed_pre_picott}"
        )
        # No retrieval / ranking / summarizer stages.
        forbidden = {
            "pubmed_retrieval", "europe_pmc_retrieval", "openalex_retrieval",
            "boolean_query_compilation", "deduplication", "safety_filter",
            "ranking", "default_selection", "full_text_access_broker",
            "evidence_packet_preparation", "summarizer", "citation_verifier",
        }
        for stage in forbidden:
            assert stage not in lat, f"forbidden stage leaked: {stage}"
            assert stage not in by_stage, (
                f"forbidden stage leaked into by_stage: {stage}"
            )
        # Clarification banner is present.
        assert body.get("clarification_message"), (
            "clarification_message missing from awaiting_clinical_subject session"
        )
        # source_api_call_count is zero (no external retrieval ran).
        sd = body.get("session_diagnostics_json") or {}
        assert sd.get("source_api_call_count") == 0


@pytest.mark.asyncio
class TestSessionDiagnosticsEndToEnd:
    async def test_phi_blocked_session_has_zero_llm_calls_and_only_phi_guard_latency(
        self, client,
    ):
        # Phase 6F.2 contract for PHI-blocked sessions: 0 LLM calls,
        # 0 tokens, ``latency_by_stage_json`` contains ONLY the
        # ``phi_guard`` stage (no other stage runs).
        import uuid
        email = f"phi-block-{uuid.uuid4().hex[:8]}@test.com"
        token = await _login_or_signup(
            client, email=email, password="TestPass1234!",
        )
        # Question contains an explicit MRN + DOB + email — guaranteed
        # to trip the PHI guard.
        question = (
            "45M with MRN 12345678, DOB 03/14/1979, email patient@x.com — "
            "best tx for HTN?"
        )
        r = await client.post(
            "/api/search-sessions/",
            headers={"Authorization": f"Bearer {token}"},
            json={"question": question},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "blocked_phi"
        sid = body["session_id"]
        # Fetch the session to read session_diagnostics_json.
        r2 = await client.get(
            f"/api/search-sessions/{sid}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r2.status_code == 200
        diag = r2.json().get("session_diagnostics_json") or {}
        assert diag.get("llm_calls_count") == 0
        assert diag.get("llm_calls_by_stage") == {}
        te = diag.get("token_estimate") or {}
        assert te.get("prompt_tokens") == 0
        assert te.get("completion_tokens") == 0
        assert te.get("total_tokens") == 0
        assert te.get("estimate_method") in ESTIMATE_METHODS
        lat = diag.get("latency_by_stage_json") or {}
        # Closed-set: only phi_guard latency for blocked sessions.
        # No PubMed / Europe PMC / parser / anchor calls happened.
        assert set(lat.keys()) == {"phi_guard"}
        assert isinstance(lat["phi_guard"], int)
        assert lat["phi_guard"] >= 0
        # Every key in latency_by_stage_json is a closed-set STAGE name.
        for k in lat.keys():
            assert k in STAGE_NAMES
        # Question + sanitized question MUST be null per the PHI
        # contract (proves no PHI is persisted alongside diagnostics).
        assert r2.json().get("original_question") in (None, "")
        assert r2.json().get("sanitized_question") in (None, "")

    async def test_clean_session_diagnostics_shape_after_pipeline(self, client):
        # Closed-set shape after a full pipeline run on a clinical
        # question (quetiapine reliably anchors). The test asserts
        # the SHAPE of the diagnostic block, not the count of articles.
        import asyncio
        import uuid
        email = f"phi-clean-{uuid.uuid4().hex[:8]}@test.com"
        token = await _login_or_signup(
            client, email=email, password="TestPass1234!",
        )
        # A clean clinical question — quetiapine reliably produces a
        # primary anchor.
        r = await client.post(
            "/api/search-sessions/",
            headers={"Authorization": f"Bearer {token}"},
            json={"question": "What are the side effects of quetiapine?"},
        )
        assert r.status_code == 200
        sid = r.json()["id"]
        # Poll up to ~120s. We accept either a terminal state OR any
        # state where the diagnostics flush has clearly happened
        # (more than just the pre-stamped phi_guard stage). PubMed
        # rate-limiting can elongate this.
        diag: dict = {}
        body: dict = {}
        for _ in range(120):
            r2 = await client.get(
                f"/api/search-sessions/{sid}",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert r2.status_code == 200
            body = r2.json()
            diag = body.get("session_diagnostics_json") or {}
            lat = diag.get("latency_by_stage_json") or {}
            terminal = body.get("status") in (
                "awaiting_selection", "awaiting_clinical_subject",
                "complete", "error",
            )
            flushed = (
                isinstance(diag.get("llm_calls_count"), int)
                and len(lat) > 1  # more than just phi_guard pre-stamp
            )
            if terminal and flushed:
                break
            await asyncio.sleep(1.0)
        # Closed-set shape.
        assert isinstance(diag.get("llm_calls_count"), int), (
            f"status={body.get('status')} diag={diag}"
        )
        by_stage = diag.get("llm_calls_by_stage") or {}
        assert isinstance(by_stage, dict)
        for k in by_stage.keys():
            assert k in STAGE_NAMES, f"non-closed-set stage key {k!r}"
        te = diag.get("token_estimate") or {}
        assert isinstance(te, dict)
        assert set(te.keys()) == {
            "prompt_tokens", "completion_tokens",
            "total_tokens", "estimate_method",
        }
        assert te["estimate_method"] in ESTIMATE_METHODS
        assert isinstance(te["prompt_tokens"], int)
        assert isinstance(te["completion_tokens"], int)
        assert te["total_tokens"] == te["prompt_tokens"] + te["completion_tokens"]
        lat = diag.get("latency_by_stage_json") or {}
        assert isinstance(lat, dict)
        for k, v in lat.items():
            assert k in STAGE_NAMES, f"non-closed-set latency stage {k!r}"
            assert isinstance(v, int) and v >= 0
        # "unknown" stage must NEVER appear in either dict.
        assert "unknown" not in by_stage
        assert "unknown" not in lat
        # The question itself must NOT leak into the diagnostics block.
        # Phase K1.1 follow-up: ``clinical_question_map.framework_neutral_slots``
        # IS intended to surface the canonical clinical anchors
        # (closed-set ``canonical_concept`` + ``original_phrase``) so
        # the frontend transparency panel can render the slot chips.
        # Those tokens are explicitly approved by the orchestrator's
        # K1.1 follow-up scope. We scope the leak check to the
        # NON-cqm portion of the diagnostics block.
        diag_for_leak_check = {
            k: v for k, v in diag.items()
            if k != "clinical_question_map"
        }
        blob = repr(diag_for_leak_check).lower()
        for needle in ["quetiapine", "seroquel", "side effects"]:
            assert needle not in blob, (
                f"diagnostics blob (excluding clinical_question_map) "
                f"leaked question content {needle!r}"
            )
