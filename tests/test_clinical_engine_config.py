"""Engine master switch: on by default, explicit false switches it off."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE = "from clinical_engine.config import settings; print(settings.ENABLED)"


def _enabled_with(value: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "CLINICAL_ENGINE_ENABLED"}
    if value is not None:
        env["CLINICAL_ENGINE_ENABLED"] = value
    run = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-1000:]
    return run.stdout.strip()


def test_engine_is_on_by_default():
    assert _enabled_with(None) == "True"
    assert _enabled_with("") == "True"


def test_engine_can_be_switched_off():
    for off in ("false", "0", "no", "off", "FALSE"):
        assert _enabled_with(off) == "False"
    assert _enabled_with("true") == "True"
