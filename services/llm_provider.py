"""LLM provider abstraction — Phase 3.

`LLM_PROVIDER` selects the runtime path:
  - `none`       — disabled. `is_configured()` returns False; every call
                   returns None. The whole app must keep working in this
                   mode (Generate Summary is disabled with a readiness
                   message).
  - `openai`     — direct OpenAI Chat Completions (JSON mode). Requires
                   `OPENAI_API_KEY`.
  - `emergent`   — Emergent universal gateway via `emergentintegrations`.
                   Requires `EMERGENT_LLM_KEY`.

Both paths force `response_format={"type":"json_object"}` and `temperature=0`
so the output is deterministic JSON. Any non-JSON or parser-failed reply
returns None to the caller (the caller persists
`failed_reason_code="parser_validation_failed"`).

NEVER log keys. NEVER include keys in API responses. NEVER pass user-supplied
keys through the request body. The summarizer and verifier paths do NOT
include raw prompts in audit logs (caller's responsibility).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger("clinevidence.llm")


class LLMProvider:
    """Phase 3 provider. Reads env at construction time so tests can rebind."""

    def __init__(self) -> None:
        self.provider = os.environ.get("LLM_PROVIDER", "none").strip().lower()
        self._openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        self._anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        self._emergent_key = os.environ.get("EMERGENT_LLM_KEY", "").strip()
        # OpenRouter (OpenAI-compatible). Used when LLM_PROVIDER=openrouter.
        self._openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        self._openrouter_model = (
            os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
            or "openai/gpt-4o-mini"
        )
        # Model defaults — explicit envs win; OpenRouter defaults to its own model.
        # NOTE: the fallback must be a REAL model id. The Emergent original
        # shipped "gpt-5.4-mini" (a placeholder that 400s on the OpenAI API),
        # which silently broke the LLM whenever OPENAI_MODEL was unset. Default
        # to a known-good model so the LLM works out of the box.
        _default_model = (
            self._openrouter_model if self.provider == "openrouter"
            else os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        )
        self._summary_model = (
            os.environ.get("SUMMARY_MODEL") or _default_model or "gpt-4o-mini"
        ).strip()
        self._verifier_model = (
            os.environ.get("VERIFIER_MODEL") or self._summary_model
        ).strip() or self._summary_model
        # Timeout shared by both paths. Phase 3 defaults to 60s.
        try:
            self._timeout_s = float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))
        except ValueError:
            self._timeout_s = 60.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_configured(self) -> bool:
        if self.provider == "openai":
            return bool(self._openai_key)
        if self.provider == "anthropic":
            return bool(self._anthropic_key)
        if self.provider == "emergent":
            return bool(self._emergent_key)
        if self.provider == "openrouter":
            return bool(self._openrouter_key)
        return False

    @property
    def summary_model(self) -> Optional[str]:
        return self._summary_model if self.is_configured() else None

    @property
    def verifier_model(self) -> Optional[str]:
        return self._verifier_model if self.is_configured() else None

    def status(self) -> dict[str, Any]:
        """Safe-to-return-to-admin status snapshot. No keys ever included."""
        return {
            "configured": self.is_configured(),
            "provider": self.provider,
            "summary_model": self.summary_model,
            "verifier_model": self.verifier_model,
        }

    async def complete_json(
        self,
        system: str,
        user: str,
        schema: Optional[dict[str, Any]] = None,
        *,
        schema_hint: Optional[dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Return a JSON dict from the LLM, or None on any failure / not configured.

        The summarizer and verifier MUST handle None (treat as
        parser_validation_failed / llm_not_configured and persist the safe
        failure path).

        Back-compat: pre-Phase-3 callers (e.g. clinical_question_parser) pass
        ``schema`` positionally — accepted as a synonym for ``schema_hint``.
        """
        hint = schema_hint if schema_hint is not None else schema
        if not self.is_configured():
            return None
        try:
            if self.provider == "openai":
                return await self._openai_json(
                    system, user, hint,
                    model=model or self._summary_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            if self.provider == "emergent":
                return await self._emergent_json(
                    system, user, hint,
                    model=model or self._summary_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    session_id=session_id,
                )
            if self.provider == "openrouter":
                return await self._openrouter_json(
                    system, user, hint,
                    model=model or self._summary_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            logger.warning("LLM provider %r not implemented", self.provider)
            return None
        except Exception:  # noqa: BLE001 - never propagate to callers
            logger.exception("LLM call failed; returning None")
            return None

    # ------------------------------------------------------------------
    # OpenAI direct
    # ------------------------------------------------------------------
    async def _openai_json(
        self,
        system: str,
        user: str,
        schema_hint: Optional[dict[str, Any]],
        *,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> Optional[dict[str, Any]]:
        user_payload = user
        if schema_hint:
            user_payload += (
                "\n\nReturn JSON matching this shape (documentation only — "
                "the Pydantic schema is the source of truth): "
                + json.dumps(schema_hint, separators=(",", ":"))
            )
        body: dict[str, Any] = {
            "model": model,
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_payload},
            ],
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._openai_key}"},
                json=body,
            )
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            # Phase 6F.2 — record provider-reported token usage if any
            # session-scoped diagnostics object is bound. Never raises;
            # never leaks raw prompts or completions.
            _record_diagnostics_call(
                prompt_text=system + user_payload,
                completion_text=content,
                usage=(data.get("usage") or {}),
            )
            return _safe_load_json(content)

    # ------------------------------------------------------------------
    # OpenRouter (OpenAI-compatible Chat Completions)
    # ------------------------------------------------------------------
    async def _openrouter_json(
        self,
        system: str,
        user: str,
        schema_hint: Optional[dict[str, Any]],
        *,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> Optional[dict[str, Any]]:
        user_payload = user
        if schema_hint:
            user_payload += (
                "\n\nReturn JSON matching this shape (documentation only — "
                "the Pydantic schema is the source of truth): "
                + json.dumps(schema_hint, separators=(",", ":"))
            )
        body: dict[str, Any] = {
            "model": model,
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_payload},
            ],
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._openrouter_key}",
                    # Optional attribution headers recommended by OpenRouter.
                    "HTTP-Referer": "https://litportal.scienthesis.ai",
                    "X-Title": "LitPortal",
                },
                json=body,
            )
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            _record_diagnostics_call(
                prompt_text=system + user_payload,
                completion_text=content,
                usage=(data.get("usage") or {}),
            )
            return _safe_load_json(content)

    # ------------------------------------------------------------------
    # Emergent gateway via emergentintegrations
    # ------------------------------------------------------------------
    async def _emergent_json(
        self,
        system: str,
        user: str,
        schema_hint: Optional[dict[str, Any]],
        *,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        session_id: Optional[str],
    ) -> Optional[dict[str, Any]]:
        # Lazy import so the dep is only required when actually used.
        # `emergentintegrations` is an Emergent-internal package that is NOT
        # installed off-platform (it's commented out of requirements.txt). If
        # LLM_PROVIDER=emergent without it, fail soft with an actionable hint
        # instead of a noisy traceback — switch to LLM_PROVIDER=openai (or
        # openrouter) with a real key, which is the supported off-platform path.
        try:
            from emergentintegrations.llm.chat import LlmChat, UserMessage
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM_PROVIDER=emergent but 'emergentintegrations' is not installed "
                "(%s). Set LLM_PROVIDER=openai or openrouter with a real API key, "
                "or install the Emergent package. Returning no completion.", exc,
            )
            return None

        # Reinforce the JSON-only contract in the system prompt, since the
        # universal gateway may sit in front of providers that do not all
        # honor response_format=json_object identically.
        json_only_system = (
            system.rstrip()
            + "\n\nIMPORTANT: Return ONLY a valid JSON object. No prose. No markdown fences."
        )
        # Pick provider family from model name (default to openai).
        provider_family = "openai"
        lower = model.lower()
        if lower.startswith("claude-"):
            provider_family = "anthropic"
        elif lower.startswith("gemini-"):
            provider_family = "gemini"

        chat_session_id = session_id or "summarizer-session"
        chat = LlmChat(
            api_key=self._emergent_key,
            session_id=chat_session_id,
            system_message=json_only_system,
        ).with_model(provider_family, model)

        user_payload = user
        if schema_hint:
            user_payload += (
                "\n\nReturn JSON matching this shape (documentation only — "
                "the Pydantic schema is the source of truth): "
                + json.dumps(schema_hint, separators=(",", ":"))
            )

        # No streaming, no tools. UserMessage is the only message type used.
        msg = UserMessage(text=user_payload)
        # `send_message` is async per playbook. Some library versions return
        # a string, others an object — we handle both.
        try:
            response = await chat.send_message(msg)  # type: ignore[func-returns-value]
        except Exception:  # noqa: BLE001
            logger.exception("emergentintegrations send_message failed")
            return None
        text = response if isinstance(response, str) else getattr(response, "text", None) or str(response)
        # Phase 6F.2 — emergentintegrations does not surface token usage
        # in a stable shape. Fall back to a char/4 estimate. Never
        # raises; never leaks prompts/completions.
        _record_diagnostics_call(
            prompt_text=json_only_system + user_payload,
            completion_text=text,
            usage=None,
        )
        return _safe_load_json(text)


# ---- helpers ---------------------------------------------------------------


def _record_diagnostics_call(
    *,
    prompt_text: str,
    completion_text: str | None,
    usage: dict[str, Any] | None,
) -> None:
    """Phase 6F.2 — record an LLM call into the session-scoped
    diagnostics block, if one is bound to the current contextvar.

    Closed-set behavior:
      * Provider-reported usage (OpenAI ``usage.prompt_tokens`` +
        ``usage.completion_tokens``) wins when present and ≥ 0.
      * Otherwise fall back to a clearly-labeled char/4 estimate.
      * NEVER persist the prompt or completion text — only token counts.
      * Never raise. Diagnostics MUST NOT break the pipeline.
    """
    try:
        # Lazy import so a circular at module-import time can't break
        # the LLM provider.
        from services.diagnostics import get_current  # noqa: WPS433

        d = get_current()
        if d is None:
            return
        method = "char_div_4_estimate"
        prompt_tokens = 0
        completion_tokens = 0
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            if isinstance(pt, int) and pt >= 0 and isinstance(ct, int) and ct >= 0:
                prompt_tokens = pt
                completion_tokens = ct
                method = "provider_reported"
        if method == "char_div_4_estimate":
            # Char/4 estimate fallback. NO prompt/completion text is
            # stored — we only consume their length and discard.
            prompt_tokens = int(len(prompt_text or "") / 4)
            completion_tokens = int(len(completion_text or "") / 4)
        d.record_llm_call(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            method=method,
        )
    except Exception:  # noqa: BLE001
        # Diagnostics is best-effort. Never let it break an LLM call.
        logger.debug("phase6f2 diagnostics record skipped", exc_info=True)


def _safe_load_json(content: str) -> Optional[dict[str, Any]]:
    """Parse the LLM response as JSON; tolerate code-fence wrapping."""
    if not content:
        return None
    s = content.strip()
    # Strip optional ```json ... ``` fences defensively.
    if s.startswith("```"):
        s = s.strip("`")
        # remove leading "json\n" if present
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip("`").strip()
    # Find first { and last } to be tolerant of stray prose around JSON.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        lb, rb = s.find("{"), s.rfind("}")
        if lb >= 0 and rb > lb:
            try:
                return json.loads(s[lb : rb + 1])
            except json.JSONDecodeError:
                return None
    return None


# Module-level singleton — tests can rebuild.
llm_provider = LLMProvider()
