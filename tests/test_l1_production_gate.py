"""Phase L1.0.1 — production visibility gate tests for the
``POST /search-sessions/{id}/evidence-lens`` endpoint.

Critical safety assertion: when the L1 UI is hidden (production default),
the preference endpoint refuses WITHOUT any mutation, so L1 can never
alter production ranking/display even via a direct API call.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.responses import JSONResponse

import routers.search_sessions as ss
from models.sessions import EvidenceLensUpdate


class _FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def __aiter__(self):
        self._it = iter(self._docs)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _FakeColl:
    def __init__(self, find_one_doc=None, find_docs=None):
        self._find_one_doc = find_one_doc
        self._find_docs = find_docs or []
        self.update_calls = []
        self.insert_calls = []

    async def find_one(self, *a, **k):
        return self._find_one_doc

    async def update_one(self, *a, **k):
        self.update_calls.append((a, k))

    async def insert_one(self, *a, **k):
        self.insert_calls.append((a, k))

    def find(self, *a, **k):
        return _FakeCursor(self._find_docs)


class _FakeDB:
    def __init__(self, colls):
        self._colls = colls

    def __getitem__(self, key):
        return self._colls[key]


def test_endpoint_refuses_without_mutation_when_ui_hidden(monkeypatch):
    sess_doc = {
        "id": "sess-1",
        "user_id": "u1",
        "status": "awaiting_selection",
        "deleted_at": None,
    }
    sessions = _FakeColl(find_one_doc=sess_doc)
    sa = _FakeColl()
    articles = _FakeColl()
    fake_db = _FakeDB({
        "search_sessions": sessions,
        "search_session_articles": sa,
        "articles": articles,
    })

    monkeypatch.setattr(ss, "get_db", lambda: fake_db)
    # Simulate production with the L1 UI hidden.
    monkeypatch.setattr(ss.evidence_lens, "l1_ui_visible", lambda: False)

    result = asyncio.run(
        ss.update_evidence_lens(
            "sess-1",
            EvidenceLensUpdate(evidence_lens="mechanism", population_focus="any"),
            current_user={"id": "u1"},
        )
    )

    assert isinstance(result, JSONResponse)
    assert result.status_code == 403
    # The decisive safety check: NOTHING was mutated.
    assert sessions.update_calls == []
    assert sa.update_calls == []
    assert sa.insert_calls == []


def test_endpoint_phi_blocked_returns_409(monkeypatch):
    sess_doc = {
        "id": "sess-2",
        "user_id": "u1",
        "status": "blocked_phi",
        "deleted_at": None,
    }
    sessions = _FakeColl(find_one_doc=sess_doc)
    fake_db = _FakeDB({
        "search_sessions": sessions,
        "search_session_articles": _FakeColl(),
        "articles": _FakeColl(),
    })
    monkeypatch.setattr(ss, "get_db", lambda: fake_db)
    monkeypatch.setattr(ss.evidence_lens, "l1_ui_visible", lambda: True)

    with pytest.raises(ss.HTTPException) as exc:
        asyncio.run(
            ss.update_evidence_lens(
                "sess-2",
                EvidenceLensUpdate(evidence_lens="mechanism"),
                current_user={"id": "u1"},
            )
        )
    assert exc.value.status_code == 409
    assert sessions.update_calls == []
