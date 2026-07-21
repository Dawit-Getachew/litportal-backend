"""Unit tests for thin-pool supplemental-retrieval decision helpers."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services import recall_supplement as rs  # noqa: E402


def test_should_supplement_thresholds():
    assert rs.should_supplement(3, enabled=True, min_pool=8) is True
    assert rs.should_supplement(8, enabled=True, min_pool=8) is False   # not < min
    assert rs.should_supplement(20, enabled=True, min_pool=8) is False
    assert rs.should_supplement(0, enabled=False, min_pool=8) is False  # disabled
    assert rs.should_supplement("bad", enabled=True, min_pool=8) is False  # safe


def test_sources_to_broaden_excludes_already_fallback():
    # Only sources NOT already broadened by the legacy fallback are re-run.
    assert rs.sources_to_broaden(["pubmed", "europe_pmc"], ["europe_pmc"]) == ["pubmed"]
    assert rs.sources_to_broaden(["pubmed", "europe_pmc"], []) == ["pubmed", "europe_pmc"]
    assert rs.sources_to_broaden(["pubmed", "europe_pmc"], ["pubmed", "europe_pmc"]) == []
    assert rs.sources_to_broaden([], ["pubmed"]) == []


def test_thin_pool_settings_default_on(monkeypatch):
    monkeypatch.delenv("THIN_POOL_SUPPLEMENT_ENABLED", raising=False)
    monkeypatch.delenv("THIN_POOL_SUPPLEMENT_MIN_POOL", raising=False)
    enabled, min_pool = rs.thin_pool_settings()
    assert enabled is True            # default ON, no env needed
    assert min_pool == rs.DEFAULT_MIN_POOL


def test_thin_pool_settings_can_disable(monkeypatch):
    monkeypatch.setenv("THIN_POOL_SUPPLEMENT_ENABLED", "false")
    monkeypatch.setenv("THIN_POOL_SUPPLEMENT_MIN_POOL", "12")
    enabled, min_pool = rs.thin_pool_settings()
    assert enabled is False
    assert min_pool == 12


def test_thin_pool_settings_bad_min_pool_falls_back(monkeypatch):
    monkeypatch.setenv("THIN_POOL_SUPPLEMENT_MIN_POOL", "notanumber")
    _, min_pool = rs.thin_pool_settings()
    assert min_pool == rs.DEFAULT_MIN_POOL
