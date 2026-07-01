"""Phase Q1 — security & correctness cleanup regression tests.

Coverage:
  A. Circular import fix: ``BenchmarkCase`` is importable from both
     ``services.search_benchmark`` (re-export) and
     ``services.search_benchmark_common`` (canonical home), and both
     paths produce the SAME class. The lazy import is gone.
  B. eval() anti-regression sweep: the closed-set "previously reported"
     files MUST NOT contain a bare ``eval(`` call. A malicious-string
     synthetic test asserts that ``ast.literal_eval`` (the safe
     replacement we standardize on) raises ``ValueError`` on a payload
     that would be executed by ``eval()``.
  C. F821 anti-regression: ``ruff check --select F821 services/ scripts/``
     reports zero hits across the backend (the 2 previously reported
     ``os`` imports in ``scripts/k3a2_seed_search_corpus.py`` are fixed).
  D. Hardcoded-test-credential allowlist: the in-test passwords match
     the seeded values documented in ``/app/memory/test_credentials.md``
     so the testing agent + fork agents continue to authenticate.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT


# =====================================================================
# A — Circular import fix
# =====================================================================
def test_benchmark_case_importable_from_common() -> None:
    from services.search_benchmark_common import BenchmarkCase as BCCommon
    from services.search_benchmark import BenchmarkCase as BCReexport

    # Same class object — re-export is not a copy.
    assert BCCommon is BCReexport


def test_benchmark_phase5c_imports_from_common() -> None:
    src = (BACKEND_ROOT / "services" / "search_benchmark_phase5c.py").read_text()
    # Must import from the common module …
    assert "from services.search_benchmark_common import BenchmarkCase" in src
    # … and MUST NOT import BenchmarkCase from the legacy parent any more
    # (that was the circular edge we removed).
    assert "from services.search_benchmark import BenchmarkCase" not in src


def test_search_benchmark_has_no_lazy_phase5c_import() -> None:
    src = (BACKEND_ROOT / "services" / "search_benchmark.py").read_text()
    # The lazy ``def list_all -> from services.search_benchmark_phase5c import ...``
    # workaround is gone — the import is now top-level + clean.
    assert "noqa: WPS433" not in src
    # And the module-level import IS present.
    assert (
        "from services.search_benchmark_phase5c import PHASE5C_PROVISIONAL_CASES"
        in src
    )


def test_search_benchmark_service_list_all_works() -> None:
    from services.search_benchmark import search_benchmark_service

    all_cases = search_benchmark_service.list_all()
    # Legacy 7 + 21 provisional = 28 (stable headcount).
    assert len(all_cases) >= 28
    # Every entry is a BenchmarkCase instance.
    from services.search_benchmark_common import BenchmarkCase
    assert all(isinstance(c, BenchmarkCase) for c in all_cases)


# =====================================================================
# B — eval() anti-regression
# =====================================================================
_REPORTED_FILES = [
    "services/search_benchmark_phase5c.py",
    "services/benchmark_runs.py",
    "tests/test_phase6b3_morphology_multiintent.py",
    "tests/test_phase2b_live_api.py",
]

_EVAL_CALL_RE = re.compile(r"(?<!\w)eval\s*\(")


@pytest.mark.parametrize("relpath", _REPORTED_FILES)
def test_no_bare_eval_call_in_reported_file(relpath: str) -> None:
    p = BACKEND_ROOT / relpath
    if not p.exists():
        pytest.skip(f"{relpath} not present in this repo snapshot")
    src = p.read_text()
    matches = _EVAL_CALL_RE.findall(src)
    assert matches == [], (
        f"{relpath} contains a bare eval() call — security regression. "
        f"Use ast.literal_eval or json.loads instead."
    )


def test_ast_literal_eval_rejects_malicious_payload() -> None:
    """Synthetic anti-eval regression: a malicious string that would
    execute under ``eval()`` MUST raise under ``ast.literal_eval``."""
    malicious = "__import__('os').system('echo pwned')"
    with pytest.raises((ValueError, SyntaxError)):
        ast.literal_eval(malicious)


def test_ast_literal_eval_rejects_attr_chain() -> None:
    """A second malicious payload variant — attribute chain."""
    malicious = "().__class__.__bases__[0].__subclasses__()"
    with pytest.raises((ValueError, SyntaxError)):
        ast.literal_eval(malicious)


def test_ast_literal_eval_accepts_trusted_literal_shapes() -> None:
    # Sanity — common safe shapes still parse.
    assert ast.literal_eval("{'a': 1, 'b': [2, 3]}") == {"a": 1, "b": [2, 3]}
    assert ast.literal_eval("[1, 2, 3]") == [1, 2, 3]
    assert ast.literal_eval('"hello"') == "hello"


# =====================================================================
# C — F821 anti-regression (ruff)
# =====================================================================
def test_ruff_f821_clean_on_services_and_scripts() -> None:
    """Backend should report zero F821 across services/ + scripts/.

    The 2 previously-reported hits in ``scripts/k3a2_seed_search_corpus.py``
    (missing ``import os``) are fixed.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check",
         "--select", "F821", "--no-fix",
         "services", "scripts"],
        cwd=BACKEND_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # ruff exits 0 when clean.
    if proc.returncode != 0:
        pytest.fail(
            f"ruff F821 not clean: stdout=\n{proc.stdout}\n"
            f"stderr=\n{proc.stderr}",
        )


# =====================================================================
# D — Hardcoded test credentials match the documented seed values
# =====================================================================
def test_hardcoded_test_credentials_match_seeds() -> None:
    """The hardcoded test credentials in ``tests/test_auth.py`` MUST
    match the values documented in ``/app/memory/test_credentials.md``
    so the testing agent and fork agents can authenticate.

    Real production secrets must never be embedded in test files. We
    enforce that by asserting the values exactly match the documented
    DEV-ONLY test seeds + the dev test-user passwords.
    """
    creds_path = Path("/app/memory/test_credentials.md")
    if not creds_path.exists():
        pytest.skip("test_credentials.md not present")
    creds = creds_path.read_text()
    assert "clinician@test.com" in creds
    assert "ClinicianTest123!" in creds
    assert "admin@test.com" in creds
    assert "AdminTest123!" in creds

    auth_test_src = (BACKEND_ROOT / "tests" / "test_auth.py").read_text()
    # The two hardcoded test passwords in test_auth.py are documented
    # in test_credentials.md — they are test-only fixtures, not real
    # secrets.
    assert "ClinicianTest123!" in auth_test_src
    assert "AdminTest123!" in auth_test_src


def test_no_real_secret_shape_in_test_files() -> None:
    """Sanity sweep on the reported "hardcoded secret" files: no
    payload matches a closed-set "looks like a real prod secret"
    pattern (e.g., ``sk-`` OpenAI, ``AKIA`` AWS, ``ghp_`` GitHub PAT,
    ``-----BEGIN ... PRIVATE KEY-----``)."""
    candidate_files = [
        "tests/test_phase6f_live_api.py",
        "tests/test_phase6f2_diagnostics.py",
        "tests/test_phase6f1_live_acceptance.py",
        "tests/test_auth.py",
        "tests/fixtures_pdf.py",
    ]
    real_secret_signatures = [
        "sk-",                      # OpenAI / similar
        "AKIA",                     # AWS access key prefix
        "ghp_",                     # GitHub personal access token
        "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----",
        "xoxb-",                    # Slack bot token
        "xoxp-",                    # Slack user token
        "AIza",                     # Google API key prefix
    ]
    for rel in candidate_files:
        p = BACKEND_ROOT / rel
        if not p.exists():
            continue
        src = p.read_text()
        for sig in real_secret_signatures:
            assert sig not in src, (
                f"{rel} contains a string matching a known production-"
                f"secret prefix {sig!r}. STOP and report."
            )
