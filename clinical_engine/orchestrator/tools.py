"""Tool registry + gateway: the only way a task reaches anything outside itself.

Every call is checked, in order:
  1. the tool is registered;
  2. it is on the calling task's allowlist;
  3. evidence-source tools require an approved, still-current question version
     (defence in depth — runs are only created by the confirm endpoint);
  4. the run's call budget (atomic, multi-replica safe).
Blocked calls are recorded in the run's usage and trace; they never retry.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from clinical_engine import store
from clinical_engine.contracts.run import TraceDoc


class ToolBlocked(Exception):
    def __init__(self, tool: str, reason: str) -> None:
        super().__init__(f"{tool}: {reason}")
        self.tool = tool
        self.reason = reason


class BudgetExceeded(Exception):
    def __init__(self, budget: str) -> None:
        super().__init__(budget)
        self.budget = budget


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: Literal["evidence_source", "model", "internal"]
    description: str

    @property
    def requires_approval(self) -> bool:
        return self.category == "evidence_source"


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec for spec in (
        ToolSpec("legacy.search_session", "evidence_source",
                 "Start the existing LitPortal search pipeline for the approved question (Week-1 bridge)."),
        ToolSpec("mock.evidence_search", "evidence_source",
                 "Labelled fixture candidates for demos/tests. Never live evidence."),
        ToolSpec("pubmed.search", "evidence_source", "PubMed E-utilities (Week 2 lanes)."),
        ToolSpec("europepmc.search", "evidence_source", "Europe PMC REST (Week 2 lanes)."),
        ToolSpec("access.resolve", "evidence_source",
                 "Verified full-text routes for one identity-checked paper (Europe PMC, PMC Cloud, Unpaywall)."),
        ToolSpec("model.json", "model", "Typed JSON model call through the model gateway."),
    )
}


class ToolGateway:
    def __init__(self, run: dict, task: dict) -> None:
        self.run = run
        self.task = task

    async def _trace(self, event: str, **details: Any) -> None:
        await store.add_trace(TraceDoc(
            question_id=self.run["question_id"], run_id=self.run["id"], task_id=self.task["id"],
            event=event, details=details,
        ).model_dump())

    async def _block(self, tool: str, reason: str) -> None:
        await store.note_blocked_call(self.run["id"])
        await self._trace("tool_blocked", tool=tool, reason=reason)
        raise ToolBlocked(tool, reason)

    async def call(self, tool: str, fn: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        spec = TOOLS.get(tool)
        if spec is None:
            await self._block(tool, "unregistered_tool")
        if tool not in (self.task.get("tools") or []):
            await self._block(tool, "not_on_task_allowlist")
        if spec.requires_approval:
            version = await store.get_version_by_id(self.run["question_version_id"])
            if not version or not version.get("approval"):
                await self._block(tool, "question_not_approved")
            if version.get("status") != "current":
                await self._block(tool, "question_version_superseded")
        budget = self.run.get("budget") or {}
        if not await store.consume_budget(self.run["id"], "tool_calls", 1, int(budget.get("max_tool_calls", 0))):
            await self._trace("budget_exceeded", budget="tool_calls", tool=tool)
            raise BudgetExceeded("tool_calls")
        started = time.perf_counter()
        try:
            result = await fn(*args, **kwargs)
        except Exception as exc:
            await self._trace("tool_call", tool=tool, ok=False, error=type(exc).__name__,
                              duration_ms=int((time.perf_counter() - started) * 1000))
            raise
        await self._trace("tool_call", tool=tool, ok=True,
                          duration_ms=int((time.perf_counter() - started) * 1000))
        return result

    async def consume_tokens(self, tokens: int) -> None:
        budget = self.run.get("budget") or {}
        if not await store.consume_budget(self.run["id"], "llm_tokens", max(0, int(tokens)),
                                          int(budget.get("max_llm_tokens", 0))):
            await self._trace("budget_exceeded", budget="llm_tokens", tokens=tokens)
            raise BudgetExceeded("llm_tokens")
