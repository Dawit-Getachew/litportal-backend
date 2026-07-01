"""Phase 6F.2 closeout — Atlas-resilient ``ensure_indexes`` tests.

Covers the closed-set of ``OperationFailure`` codes that must be
tolerated under concurrent worker boots against MongoDB Atlas:

* 26  NamespaceNotFound       (drop noop)
* 27  IndexNotFound           (drop noop)
* 68  CannotCreateIndex       (log warning + continue)
* 85  IndexOptionsConflict    (drop + recreate)
* 86  IndexKeySpecsConflict   (drop + recreate)
* 276 IndexBuildAborted       (log + retry with backoff)

Plus the safety invariants:

* Spec-match short-circuit — drop+create are SKIPPED when the
  existing index already has the desired keys + options.
* Per-process idempotency — calling ``ensure_indexes`` twice in the
  same process returns early on the second call.
* Unexpected codes (e.g. 13 ``Unauthorized``) are re-raised so
  misconfiguration stays visible.
"""
from __future__ import annotations

from typing import Any

import pytest
from pymongo.errors import OperationFailure

from core import db as db_mod
from core.db import (
    MAX_INDEX_BOOTSTRAP_ATTEMPTS,
    _index_spec_matches,
    _infer_index_name,
    _normalize_keys,
    _reset_indexes_ensured_for_test,
    _safe_create_index,
    _safe_drop_index,
    ensure_indexes,
)


pytestmark = pytest.mark.xdist_group(name="phase6f2_atlas_indexes")


# ---------------------------------------------------------------------------
# Test doubles — a minimal async fake collection that mimics Motor's surface
# for ``drop_index``, ``create_index``, ``index_information``.
# ---------------------------------------------------------------------------


class _FakeCollection:
    def __init__(
        self,
        name: str = "fake_coll",
        *,
        drop_error: BaseException | None = None,
        create_errors: list[BaseException] | None = None,
        info: dict[str, dict] | None = None,
    ):
        self.name = name
        self._drop_error = drop_error
        # create_errors is consumed left-to-right: each create_index
        # call pops the next entry. ``None`` slots mean "succeed".
        self._create_errors = list(create_errors or [])
        self._info = info or {}
        self.drop_calls: list[str] = []
        self.create_calls: list[tuple[Any, dict[str, Any]]] = []
        self.info_calls: int = 0

    async def drop_index(self, name: str) -> None:
        self.drop_calls.append(name)
        if self._drop_error is not None:
            err = self._drop_error
            # Drop-error fires once unless it's a one-shot raise list.
            self._drop_error = None
            raise err

    async def create_index(self, keys: Any, **opts: Any) -> str:
        self.create_calls.append((keys, dict(opts)))
        if self._create_errors:
            err = self._create_errors.pop(0)
            if err is not None:
                raise err
        return _infer_index_name(keys)

    async def index_information(self) -> dict:
        self.info_calls += 1
        return dict(self._info)


# ---------------------------------------------------------------------------
# 1. Closed-set OperationFailure tolerance
# ---------------------------------------------------------------------------


class TestSafeDropIndexTolerance:
    @pytest.mark.asyncio
    async def test_swallows_index_not_found(self, caplog):
        # Code 27 — index already absent. Drop is idempotent.
        coll = _FakeCollection(
            drop_error=OperationFailure("IndexNotFound", code=27),
        )
        cache: dict = {}
        await _safe_drop_index(coll, "missing_idx", cache=cache)
        # No raise; drop_calls captured the attempt.
        assert coll.drop_calls == ["missing_idx"]

    @pytest.mark.asyncio
    async def test_swallows_namespace_not_found(self, caplog):
        # Code 26 — collection doesn't exist yet.
        coll = _FakeCollection(
            drop_error=OperationFailure("NamespaceNotFound", code=26),
        )
        cache: dict = {}
        await _safe_drop_index(coll, "x_1", cache=cache)
        assert coll.drop_calls == ["x_1"]

    @pytest.mark.asyncio
    async def test_swallows_index_build_aborted(self, caplog):
        # Code 276 — concurrent build aborted.
        coll = _FakeCollection(
            drop_error=OperationFailure("IndexBuildAborted", code=276),
        )
        cache: dict = {}
        with caplog.at_level("WARNING"):
            await _safe_drop_index(coll, "x_1", cache=cache)
        assert coll.drop_calls == ["x_1"]
        assert any("IndexBuildAborted" in r.message or "code=276" in r.message
                   for r in caplog.records)

    @pytest.mark.asyncio
    async def test_unexpected_error_propagates(self):
        # Code 13 — Unauthorized. NOT in the tolerated set → propagates.
        coll = _FakeCollection(
            drop_error=OperationFailure("Unauthorized", code=13),
        )
        cache: dict = {}
        with pytest.raises(OperationFailure) as ei:
            await _safe_drop_index(coll, "x_1", cache=cache)
        assert ei.value.code == 13


class TestSafeCreateIndexTolerance:
    @pytest.mark.asyncio
    async def test_swallows_index_build_aborted_after_retries(self, caplog):
        # Code 276 fires on every attempt. After
        # MAX_INDEX_BOOTSTRAP_ATTEMPTS attempts, the call returns
        # False (no crash) and logs the exhaustion warning.
        errs = [
            OperationFailure("IndexBuildAborted", code=276)
            for _ in range(MAX_INDEX_BOOTSTRAP_ATTEMPTS)
        ]
        coll = _FakeCollection(create_errors=errs)
        cache: dict = {}
        with caplog.at_level("WARNING"):
            created = await _safe_create_index(coll, "email", cache=cache, unique=True)
        assert created is False
        assert len(coll.create_calls) == MAX_INDEX_BOOTSTRAP_ATTEMPTS
        assert any("IndexBuildAborted" in r.message or "code=276" in r.message
                   for r in caplog.records)

    @pytest.mark.asyncio
    async def test_index_build_aborted_retries_then_succeeds(self):
        # First attempt fails with 276, second attempt succeeds.
        coll = _FakeCollection(
            create_errors=[
                OperationFailure("IndexBuildAborted", code=276),
                None,
            ],
        )
        cache: dict = {}
        created = await _safe_create_index(coll, "user_id", cache=cache)
        assert created is True
        assert len(coll.create_calls) == 2

    @pytest.mark.asyncio
    async def test_handles_index_options_conflict(self):
        # Code 85 — existing index has same key/name but different
        # options. _safe_create_index drops then retries.
        coll = _FakeCollection(
            create_errors=[
                OperationFailure("IndexOptionsConflict", code=85),
                None,  # retry succeeds
            ],
        )
        cache: dict = {}
        created = await _safe_create_index(
            coll, "email", cache=cache, unique=True,
        )
        assert created is True
        assert coll.drop_calls == [_infer_index_name("email")]
        assert len(coll.create_calls) == 2

    @pytest.mark.asyncio
    async def test_handles_index_key_specs_conflict(self):
        # Code 86 — name collision with different keys.
        coll = _FakeCollection(
            create_errors=[
                OperationFailure("IndexKeySpecsConflict", code=86),
                None,
            ],
        )
        cache: dict = {}
        created = await _safe_create_index(
            coll, "email", cache=cache, name="email_unique", unique=True,
        )
        assert created is True
        assert coll.drop_calls == ["email_unique"]

    @pytest.mark.asyncio
    async def test_swallows_cannot_create_index(self, caplog):
        coll = _FakeCollection(
            create_errors=[
                OperationFailure("CannotCreateIndex", code=68),
            ],
        )
        cache: dict = {}
        with caplog.at_level("WARNING"):
            created = await _safe_create_index(coll, "x_field", cache=cache)
        assert created is False
        assert any("code=68" in r.message or "CannotCreateIndex" in r.message
                   for r in caplog.records)

    @pytest.mark.asyncio
    async def test_unexpected_error_propagates(self):
        # Code 13 — Unauthorized. Must propagate so misconfiguration
        # stays visible.
        coll = _FakeCollection(
            create_errors=[OperationFailure("Unauthorized", code=13)],
        )
        cache: dict = {}
        with pytest.raises(OperationFailure) as ei:
            await _safe_create_index(coll, "x", cache=cache)
        assert ei.value.code == 13


# ---------------------------------------------------------------------------
# 2. Spec-match short-circuit
# ---------------------------------------------------------------------------


class TestSpecMatchShortCircuit:
    @pytest.mark.asyncio
    async def test_short_circuits_when_spec_matches(self):
        # An identical index already exists → no drop, no create.
        info = {
            "email_1": {
                "v": 2,
                "key": [("email", 1)],
                "unique": True,
            }
        }
        coll = _FakeCollection(info=info)
        cache: dict = {}
        created = await _safe_create_index(
            coll, "email", cache=cache, unique=True,
        )
        assert created is False
        assert coll.drop_calls == []
        assert coll.create_calls == []

    @pytest.mark.asyncio
    async def test_short_circuits_compound_index_with_partial_filter(self):
        info = {
            "user_library_active_unique": {
                "v": 2,
                "key": [("user_id", 1), ("article_id", 1)],
                "unique": True,
                "partialFilterExpression": {"status": "active"},
            }
        }
        coll = _FakeCollection(info=info)
        cache: dict = {}
        created = await _safe_create_index(
            coll,
            [("user_id", 1), ("article_id", 1)],
            cache=cache,
            unique=True,
            name="user_library_active_unique",
            partialFilterExpression={"status": "active"},
        )
        assert created is False
        assert coll.drop_calls == []
        assert coll.create_calls == []

    @pytest.mark.asyncio
    async def test_drops_and_recreates_when_spec_differs(self):
        # Existing index has unique=False, we want unique=True → drop + create.
        info = {
            "email_1": {
                "v": 2,
                "key": [("email", 1)],
                "unique": False,
            }
        }
        coll = _FakeCollection(info=info)
        cache: dict = {}
        created = await _safe_create_index(
            coll, "email", cache=cache, unique=True,
        )
        assert created is True
        assert coll.drop_calls == ["email_1"]
        assert len(coll.create_calls) == 1

    def test_index_spec_matches_helper(self):
        existing = {
            "v": 2,
            "key": [("email", 1)],
            "unique": True,
        }
        # exact match
        assert _index_spec_matches(existing, "email", {"unique": True}) is True
        # unique differs
        assert _index_spec_matches(existing, "email", {"unique": False}) is False
        # key differs
        assert _index_spec_matches(existing, "name", {"unique": True}) is False
        # partial filter expression mismatch
        e2 = {"v": 2, "key": [("status", 1)],
              "partialFilterExpression": {"status": "active"}}
        assert _index_spec_matches(
            e2, "status", {"partialFilterExpression": {"status": "active"}},
        ) is True
        assert _index_spec_matches(
            e2, "status", {"partialFilterExpression": {"status": "archived"}},
        ) is False

    def test_normalize_keys_helper(self):
        assert _normalize_keys("email") == [("email", 1)]
        assert _normalize_keys([("a", 1), ("b", -1)]) == [("a", 1), ("b", -1)]
        assert _normalize_keys([["a", 1]]) == [("a", 1)]

    def test_infer_index_name_helper(self):
        assert _infer_index_name("email") == "email_1"
        assert _infer_index_name([("a", 1), ("b", -1)]) == "a_1_b_-1"


# ---------------------------------------------------------------------------
# 3. Per-process idempotency guard
# ---------------------------------------------------------------------------


class TestEnsureIndexesIdempotency:
    @pytest.mark.asyncio
    async def test_ensure_indexes_idempotent_within_process(self, monkeypatch):
        # Trip the idempotency guard. First call does real work
        # (mocked db). Second call short-circuits and never touches
        # the mock db.
        _reset_indexes_ensured_for_test()

        class _StubColl:
            name = "stub"
            calls = 0
            async def index_information(self):
                _StubColl.calls += 1
                return {}
            async def create_index(self, keys, **opts):
                return "ok"
            async def drop_index(self, name):
                return None

        class _StubDb:
            def __init__(self):
                self.access_count = 0
            def __getitem__(self, name):
                self.access_count += 1
                return _StubColl()

        stub_db = _StubDb()

        await ensure_indexes(stub_db)
        first_access = stub_db.access_count
        assert first_access > 0
        assert db_mod._INDEXES_ENSURED is True

        # Second call: idempotency guard returns immediately. No new
        # ``db[coll]`` accesses are recorded.
        await ensure_indexes(stub_db)
        assert stub_db.access_count == first_access

        # Reset for the next test.
        _reset_indexes_ensured_for_test()

    @pytest.mark.asyncio
    async def test_reset_helper_clears_guard(self):
        _reset_indexes_ensured_for_test()
        assert db_mod._INDEXES_ENSURED is False
        db_mod._INDEXES_ENSURED = True  # type: ignore[assignment]
        _reset_indexes_ensured_for_test()
        assert db_mod._INDEXES_ENSURED is False


# ---------------------------------------------------------------------------
# 4. ensure_indexes — end-to-end against the live test database. The first
# call did all the work via the session fixture; calling it again with the
# guard cleared MUST complete without raising (proves spec-match
# short-circuit + idempotent re-bootstrap on a real Mongo).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEnsureIndexesLiveSandbox:
    async def test_ensure_indexes_second_run_is_noop_on_live_db(self, client):
        # `client` fixture has already booted the lifespan once, so
        # all indexes exist. Reset the guard and call again — the
        # spec-match short-circuit should make this a fast no-op
        # with zero errors logged at ERROR level.
        _reset_indexes_ensured_for_test()
        # Re-run against the live test database. No exception means
        # the second boot is safe.
        await ensure_indexes()
        # Guard is set again after the second successful run.
        assert db_mod._INDEXES_ENSURED is True
