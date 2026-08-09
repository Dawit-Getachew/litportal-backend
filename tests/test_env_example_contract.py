"""Guards the fresh-dev setup contract: `cp .env.example .env` must work.

The failure this prevents: a line written as ``VAR=<spaces># comment`` parses
as the literal value ``"# comment"``, not as empty. That silently handed a new
developer a JWT secret of ``"# [REQUIRED-PROD] random 64-char hex"`` and API
keys containing prose, producing 401s instead of the not-configured path.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = ROOT / ".env.example"


@pytest.fixture(scope="module")
def parsed() -> dict[str, str | None]:
    assert ENV_EXAMPLE.exists(), ".env.example must ship with the repo"
    return dotenv_values(ENV_EXAMPLE)


def test_no_value_is_actually_a_comment(parsed):
    """Empty variables must parse as empty, not as their trailing comment."""
    offenders = {k: v for k, v in parsed.items() if v and v.strip().startswith("#")}
    assert not offenders, (
        "these variables parse to comment text — move the comment to its own "
        f"line above the variable: {sorted(offenders)}"
    )


def test_required_variables_have_usable_defaults(parsed):
    """Copying the template must satisfy every import-time requirement."""
    for name in ("MONGO_URL", "DB_NAME"):
        value = (parsed.get(name) or "").strip()
        assert value, f"{name} is required at import time and must ship a dev default"
        assert not value.startswith("#")


def test_copying_the_template_boots_config():
    """End-to-end: the values in .env.example are enough to load Settings."""
    load_dotenv(ENV_EXAMPLE, override=True)
    import core.config as config

    importlib.reload(config)
    assert config.Settings.MONGO_URL
    assert config.Settings.DB_NAME
    # Secrets are generated when absent — never the template's comment text.
    assert not config.Settings.JWT_ACCESS_SECRET.startswith("#")
    assert len(config.Settings.JWT_ACCESS_SECRET) >= 32


def test_missing_required_var_raises_actionable_error(monkeypatch):
    """A bare KeyError tells a new developer nothing; the message must name the
    variable and the fix."""
    import core.config as config

    monkeypatch.delenv("MONGO_URL", raising=False)
    with pytest.raises(config.ConfigurationError) as exc:
        config._required("MONGO_URL", why="needed for the test")
    message = str(exc.value)
    assert "MONGO_URL" in message
    assert ".env.example" in message
