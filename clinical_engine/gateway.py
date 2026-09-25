"""Model gateway: typed JSON calls with usage, latency and closed-set errors.

Uses the same provider env as services/llm_provider.py (LLM_PROVIDER,
OPENAI_API_KEY / OPENROUTER_API_KEY, OPENAI_MODEL / OPENROUTER_MODEL) but, unlike
that wrapper, returns token usage and *why* a call failed — both are needed for
enforced budgets and honest fallbacks. The legacy wrapper is left untouched
(the summary worker depends on it).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from clinical_engine.config import settings

logger = logging.getLogger("litportal.clinical_engine.gateway")

_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


@dataclass
class ModelCallResult:
    data: Optional[dict[str, Any]]
    model: Optional[str]
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    # None | not_configured | timeout | http_<status> | invalid_json | transport_error
    error: Optional[str] = None

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class ModelGateway:
    def __init__(self, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self.provider = os.environ.get("LLM_PROVIDER", "none").strip().lower()
        self._transport = transport
        if self.provider == "openai":
            self._key = os.environ.get("OPENAI_API_KEY", "").strip()
            default_model = os.environ.get("OPENAI_MODEL", "").strip() or "gpt-4o-mini"
        elif self.provider == "openrouter":
            self._key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            default_model = os.environ.get("OPENROUTER_MODEL", "").strip() or "openai/gpt-4o-mini"
        else:
            self._key = ""
            default_model = ""
        self.model = (settings.INTERPRETER_MODEL or default_model) or None

    def is_configured(self) -> bool:
        return self.provider in _ENDPOINTS and bool(self._key) and bool(self.model)

    async def complete_json(
        self, system: str, user: str, *, max_tokens: int, timeout_s: float,
    ) -> ModelCallResult:
        if not self.is_configured():
            return ModelCallResult(data=None, model=None, error="not_configured")
        body = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        headers = {"Authorization": f"Bearer {self._key}"}
        if self.provider == "openrouter":
            headers.update({"HTTP-Referer": "https://litportal.scienthesis.ai", "X-Title": "LitPortal"})
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout_s, transport=self._transport) as client:
                resp = await client.post(_ENDPOINTS[self.provider], headers=headers, json=body)
        except httpx.TimeoutException:
            return ModelCallResult(None, self.model, latency_ms=_ms(started), error="timeout")
        except httpx.HTTPError as exc:
            logger.warning("model gateway transport error: %s", type(exc).__name__)
            return ModelCallResult(None, self.model, latency_ms=_ms(started), error="transport_error")
        latency = _ms(started)
        if resp.status_code >= 400:
            logger.warning("model gateway HTTP %s (model=%s)", resp.status_code, self.model)
            return ModelCallResult(None, self.model, latency_ms=latency, error=f"http_{resp.status_code}")
        try:
            payload = resp.json()
            content = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage") or {}
            tokens_in = int(usage.get("prompt_tokens") or 0) or (len(system) + len(user)) // 4
            tokens_out = int(usage.get("completion_tokens") or 0) or len(content or "") // 4
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("not an object")
        except (ValueError, KeyError, IndexError, TypeError):
            return ModelCallResult(None, self.model, latency_ms=latency, error="invalid_json")
        return ModelCallResult(data, self.model, tokens_in, tokens_out, latency)


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
