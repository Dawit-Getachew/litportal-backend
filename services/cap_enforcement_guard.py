"""Phase K3A.3 — Article-join status + cap-enforcement guard.

Closed-set, deterministic, stdlib + Motor only. No LLM, no network.

The K3A classifier emits per-article diagnostics referenced by
``article_id``. K3A.2 surfaced one case where, between classification
time and review time, the ``articles`` doc was no longer joinable
(suspected post-dedup canonical-key shift). K3B Wave 1 enforcement
cannot afford to hard-exclude an article it cannot locate; this
module is the defense-in-depth that K3B will lean on.

Two responsibilities:

1. **Compute** the join status of one diagnostic against the current
   DB state (``compute_article_join_status``). Pure function over
   already-fetched documents — easy to unit-test.

2. **Guard** any future enforcement decision
   (``should_enforce_cap_for_article``). Returns ``False`` unconditionally
   while ``CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE != "active"``, AND
   returns ``False`` whenever the join is anything other than
   ``"stable"``, regardless of mode. Defense-in-depth — even with a
   hypothetical ``"active"`` mode set, an unstable join NEVER produces
   an enforcement decision.
"""
from __future__ import annotations

import os
from typing import Any, Literal, Optional

JoinStatus = Literal[
    "stable",
    "article_doc_missing",
    "search_session_article_missing",
    "canonical_key_missing_or_changed",
    "article_removed_between_classification_and_review",
]


def compute_article_join_status(
    *,
    article_doc: Optional[dict[str, Any]],
    search_session_article_doc: Optional[dict[str, Any]],
    classification_canonical_key: Optional[str] = None,
) -> JoinStatus:
    """Compute the closed-set join status of one diagnostic against
    DB state captured at *call* time. Pure function — accepts the
    already-fetched ``articles`` and ``search_session_articles`` docs
    (or ``None`` when missing) plus the optional canonical key the
    classifier saw at classification time.

    Resolution order (first match wins):

    1. ``articles`` AND ``search_session_articles`` both missing →
       ``"article_removed_between_classification_and_review"`` (the
       loudest signal — strongly suggests post-classification deletion
       or dedup-shift).
    2. Only ``articles`` missing → ``"article_doc_missing"``.
    3. Only ``search_session_articles`` missing →
       ``"search_session_article_missing"``.
    4. Both present but ``canonical_key`` on the article doc is
       missing OR differs from the classification-time key →
       ``"canonical_key_missing_or_changed"``.
    5. All checks pass → ``"stable"``.
    """
    article_present = isinstance(article_doc, dict) and bool(
        article_doc.get("id")
    )
    ssa_present = (
        isinstance(search_session_article_doc, dict)
        and bool(search_session_article_doc.get("id"))
    )
    if not article_present and not ssa_present:
        return "article_removed_between_classification_and_review"
    if not article_present:
        return "article_doc_missing"
    if not ssa_present:
        return "search_session_article_missing"
    # Both present — check canonical key.
    if classification_canonical_key is not None:
        live_key = article_doc.get("canonical_key")
        if not live_key or live_key != classification_canonical_key:
            return "canonical_key_missing_or_changed"
    elif not article_doc.get("canonical_key"):
        return "canonical_key_missing_or_changed"
    return "stable"


def should_enforce_cap_for_article(
    *,
    article_join_status: Optional[str],
    cap_reason_codes: Optional[list[str]] = None,
    enforcement_mode: Optional[str] = None,
) -> bool:
    """K3B Wave 1 — Retracted-only shadow/active enforcement gate.

    Return ``True`` only when ALL of the following hold:

    - ``enforcement_mode`` is one of ``{"shadow", "active"}``
      (read from ``CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE`` if not
      passed explicitly).
    - ``article_join_status == "stable"``.
    - ``"retracted"`` is a member of ``cap_reason_codes``
      (membership, NOT exact equality — additional reason codes do
      not block).
    - All inputs are non-empty / non-``None``.

    K3B Wave 1 invariants (defense-in-depth, never bypassable):

    - While ``CAP_ENFORCEMENT_MODE == "disabled"``, this function
      ALWAYS returns ``False``. No exception.
    - On any unstable join, the function ALWAYS returns ``False``,
      regardless of mode. Even with a hypothetical ``"active"`` mode
      mocked in, an unstable join still yields ``False``.
    - On a Wave-1-out-of-scope reason set (e.g. ``[
      "expression_of_concern"]`` without ``retracted``,
      ``["missing_primary_anchor"]``, etc.), returns ``False``.
    - The function NEVER raises on missing / ``None`` inputs.

    K3B Wave 1 Step 1 keeps ``"active"`` mode hard-asserted-rejected
    at the env load layer (see ``services/pipeline.py``). The
    guard's logic stays correct for the future ``"active"`` flip
    so Step 2 inherits a tested gate.
    """
    if enforcement_mode is None:
        enforcement_mode = os.environ.get(
            "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE", "disabled",
        )
    try:
        mode = str(enforcement_mode or "").strip().lower()
    except Exception:  # noqa: BLE001
        return False
    if mode not in {"shadow", "active"}:
        return False
    if not article_join_status:
        return False
    if str(article_join_status).strip().lower() != "stable":
        return False
    # K3B Wave 1 — retracted-only reason-code membership check.
    if not cap_reason_codes:
        return False
    try:
        codes = [str(c).strip().lower() for c in cap_reason_codes]
    except Exception:  # noqa: BLE001
        return False
    return "retracted" in codes


# ---------------------------------------------------------------------
# Phase K3B Wave 1 Step 2.0.1 — Startup-fingerprint production-active
# guardrail.
#
# Background: an auto-deploy carrying the dev/staging ``.env`` value
# ``CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE="active"`` to production
# armed the K3B Wave 1 Step 2 gate on production without operator
# approval. No clinician triggered a retraction-rich query during
# the risk window so zero exclusions actually happened, but the
# class of incident must never recur. This guard is the
# belt-and-suspenders.
#
# Behaviour (closed-set):
#   * Fires at backend STARTUP only (called from
#     ``server.py::lifespan`` BEFORE any DB I/O, outside the try /
#     except so a RuntimeError propagates and crashes startup).
#   * No-op on dev / staging (``settings.IS_PRODUCTION == False``).
#   * No-op when the resolved mode is anything other than
#     ``"active"``.
#   * On production + active:
#       - Requires ``DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE=true`` OR
#         (alias) ``ALLOW_PRODUCTION_CAP_ACTIVE=true`` to be
#         explicitly set in the production env. Default unset → fail
#         closed.
#       - Requires ``ACTIVE_CAP_REASON_ALLOWLIST`` to be a non-empty,
#         comma-separated, case-insensitive list containing
#         ``"retracted"``. Empty / missing / missing-retracted →
#         fail closed.
#
# Operator runbook: see
# ``/app/memory/phaseK3B_startup_fingerprint_guardrail_*.md`` for
# the exact env-var values a future operator-approved production
# active rollout must set, AND the verification curl.
# ---------------------------------------------------------------------


def _resolved_cap_enforcement_mode(
    env: Optional[dict[str, str]] = None,
) -> str:
    """Read the same env var the rest of the code reads.

    Matches ``services/pipeline.py``, ``services/cap_observability.py``
    and the K3B Wave 1 Step 1 / Step 2 invariants — single source of
    truth for the resolved mode at any point in time. Tests pass a
    custom mapping; production calls use the live process env.
    """
    src = env if env is not None else os.environ
    return (
        src.get("CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE", "disabled")
        or ""
    ).strip().lower()


def verify_startup_fingerprint(
    *,
    is_production: Optional[bool] = None,
    resolved_mode: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    logger_obj: Optional[Any] = None,
) -> None:
    """Backend-startup fail-closed guardrail. See module-level
    docstring for the full behaviour spec.

    Arguments are all optional and default to the live runtime
    values; tests inject closed-set fixtures.

    Raises:
        RuntimeError: when production + active + (no fingerprint flag
        OR allowlist missing ``"retracted"``). The RuntimeError
        message is closed-set and matches the spec's exact wording.
    """
    import logging

    log = logger_obj if logger_obj is not None else logging.getLogger(
        "clinevidence",
    )

    if is_production is None:
        # Reuse the same primitive every other path uses
        # (settings.IS_PRODUCTION ← APP_ENV == "production").
        # Import locally so test fixtures can override before this
        # function is called.
        try:
            from core.config import settings  # type: ignore
            is_production = bool(settings.IS_PRODUCTION)
        except Exception:  # noqa: BLE001 - defensive only
            is_production = False
    if resolved_mode is None:
        resolved_mode = _resolved_cap_enforcement_mode(env=env)
    env_map = env if env is not None else os.environ
    env_label = "production" if is_production else "non_production"

    if not (is_production and resolved_mode == "active"):
        # Non-production, OR production with a non-active mode —
        # guard does not fire. Log the no-op so the tester / SRE
        # can confirm the check executed on every startup.
        log.info(
            "cap_enforcement_fingerprint_check passed "
            "mode=%s env=%s",
            resolved_mode, env_label,
        )
        return

    # Production + active. The two binding clauses apply.

    fp1 = (
        env_map.get("DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE", "") or ""
    ).strip().lower()
    fp2 = (
        env_map.get("ALLOW_PRODUCTION_CAP_ACTIVE", "") or ""
    ).strip().lower()
    fingerprint_ok = fp1 == "true" or fp2 == "true"

    if not fingerprint_ok:
        log.error(
            "cap_enforcement_fingerprint_check failed mode=active "
            "env=production reason=fingerprint_flag_missing"
        )
        raise RuntimeError(
            "Production active cap enforcement requires explicit "
            "DEPLOYMENT_FINGERPRINT_ALLOW_ACTIVE=true or "
            "ALLOW_PRODUCTION_CAP_ACTIVE=true"
        )

    allowlist_raw = env_map.get("ACTIVE_CAP_REASON_ALLOWLIST", "") or ""
    allowlist = [
        s.strip().lower()
        for s in allowlist_raw.split(",")
        if s.strip()
    ]
    if "retracted" not in allowlist:
        log.error(
            "cap_enforcement_fingerprint_check failed mode=active "
            "env=production reason=allowlist_missing_retracted "
            "allowlist=%r", allowlist_raw,
        )
        raise RuntimeError(
            "Production active cap enforcement requires explicit "
            "ACTIVE_CAP_REASON_ALLOWLIST containing 'retracted'"
        )

    log.info(
        "cap_enforcement_fingerprint_check passed mode=active "
        "env=production allowlist=%r",
        allowlist_raw,
    )


__all__ = [
    "JoinStatus",
    "compute_article_join_status",
    "should_enforce_cap_for_article",
    "verify_startup_fingerprint",
]
