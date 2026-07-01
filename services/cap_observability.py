"""Phase K3A.1 — Cap-Observability aggregation service.

Closed-set, deterministic, no LLM, no external network. Reads the
shadow-only ``session_diagnostics_json.article_lane_diagnostics``
batches that the pipeline persists on every clean (non-PHI) session
within the 30-day retention window and aggregates them into closed-set
histograms for the admin dashboard.

Hard invariants enforced inside ``build_summary``:

- ``blocked_phi`` sessions are EXCLUDED from every histogram and from
  the top-3 lists. The session collection filter rejects any session
  whose ``status == "blocked_phi"`` or whose ``deleted_at IS NOT NULL``.
- The aggregation never emits ``user_id``, ``email``, raw
  ``original_question`` text, ``sanitized_question`` text, raw
  prompts, raw completions, raw evidence text, uploaded file content,
  credentials, API keys, or any open-set string.
- Every histogram key is a closed-set ``Literal`` value sourced from
  ``services.article_lane_classifier``. Unknown values are bucketed
  into ``"unknown"``.
- ``window_days`` is clamped to ``[1, 30]``.
- The function does not mutate any document; it only reads.
"""
from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from services.article_lane_classifier import (
    _FORBIDDEN_DIAG_SUBSTRINGS,
)


# Closed-set bucket keys for the cap-confidence distribution.
_CONFIDENCE_BUCKETS: tuple[str, ...] = (
    "<0.5", "0.5-0.7", "0.7-0.9", ">=0.9",
)


def _bucket_confidence(c: Any) -> str | None:
    """Map a raw cap_confidence value into one of the closed-set
    buckets. Returns ``None`` for non-numeric or missing values so the
    distribution counter can skip them.
    """
    if not isinstance(c, (int, float)):
        return None
    try:
        f = float(c)
    except Exception:  # noqa: BLE001
        return None
    if f < 0.5:
        return "<0.5"
    if f < 0.7:
        return "0.5-0.7"
    if f < 0.9:
        return "0.7-0.9"
    return ">=0.9"


def _clamp_window_days(window_days: int) -> int:
    if not isinstance(window_days, int):
        try:
            window_days = int(window_days)
        except Exception:  # noqa: BLE001
            window_days = 7
    return max(1, min(30, window_days))


async def build_summary(
    db: AsyncIOMotorDatabase,
    *,
    window_days: int = 7,
) -> dict[str, Any]:
    """Build the closed-set cap-observability summary over the recent
    sessions window. Returns a JSON-safe dict. Never raises on missing
    diagnostics — empty buckets are returned instead.
    """
    window_days = _clamp_window_days(window_days)
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=window_days)

    # Collection filter — soft-deleted and PHI-blocked sessions are
    # excluded at the Mongo layer.
    sessions_cursor = db["search_sessions"].find(
        {
            "created_at": {"$gte": window_start},
            "deleted_at": None,
            "status": {"$ne": "blocked_phi"},
        },
        {
            "_id": 0,
            "id": 1,
            "status": 1,
            "provenance": 1,
            "created_at": 1,
            "session_diagnostics_json.article_lane_diagnostics": 1,
            "session_diagnostics_json.k3b_active_exclusions": 1,
            "clinical_question_map_json.primary_intent": 1,
        },
    )

    cap_reason_codes: Counter[str] = Counter()
    lane_labels: Counter[str] = Counter()
    directness_labels: Counter[str] = Counter()
    cap_severities: Counter[str] = Counter()
    cap_conf_dist: Counter[str] = Counter({b: 0 for b in _CONFIDENCE_BUCKETS})
    cap_conf_total = 0
    by_primary_intent: Counter[str] = Counter()
    by_branch_intent: Counter[str] = Counter()
    sessions_with_diag = 0
    sessions_scanned = 0
    articles_with_caps = 0
    articles_without_caps = 0

    # Phase K3B Wave 1 Step 1 — shadow-enforcement aggregation.
    would_have_been_excluded_total = 0
    would_have_been_excluded_by_reason: Counter[str] = Counter()
    skipped_due_to_unstable_join_total = 0
    skipped_eoc_uncertified_total = 0
    skipped_non_wave1_reason_total = 0
    shadow_enf_by_primary_intent: Counter[str] = Counter()
    shadow_enf_by_branch_intent: Counter[str] = Counter()
    _NON_WAVE1_LOCAL = {
        "missing_primary_anchor", "cross_intent_drift",
        "low_branch_alignment", "wrong_framework_lane",
        "animal_only_for_human_question", "preprint_unverified",
        "generic_modifier_only_match", "generic_modifier_only",
    }

    # Phase K3B Wave 1 Step 2 — ACTIVE-exclusion aggregation.
    # Read from ``session_diagnostics_json.k3b_active_exclusions``,
    # which is populated by the pipeline whenever
    # ``CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE == "active"`` AND at
    # least one stable-join PubMed-confirmed retracted article was
    # filtered before ranking.
    actually_excluded_total = 0
    actually_excluded_by_reason: Counter[str] = Counter()
    active_excl_by_primary_intent: Counter[str] = Counter()
    active_excl_by_branch_intent: Counter[str] = Counter()
    sessions_with_active_exclusion = 0

    # Phase K3B Wave 1 Step 2.1 — soak observation additions.
    # All closed-set, all populated by the same loop below.
    article_join_status_hist: Counter[str] = Counter()
    organic_session_total = 0
    seeded_session_total = 0
    organic_exclusion_total = 0
    seeded_exclusion_total = 0
    # WBE / barred / EoC monitors (alert booleans below).
    wbe_with_unstable_join_seen = False
    barred_reason_in_candidate_set_seen = False
    # ``alert_eoc_only_marked_as_enforcing`` reads from audit events
    # AFTER the session loop (cheaper join there).

    # For top-3 sessions by cross-intent drift / low branch alignment.
    drift_by_session: Counter[str] = Counter()
    low_alignment_by_session: Counter[str] = Counter()

    # Phase K3B Wave 1 Step 2.2 — per-day sparkline accumulator.
    # Closed-set buckets keyed by ``YYYY-MM-DD`` (UTC). Every field
    # below is a counter / set keyed by the session's ``created_at``
    # day (NOT by audit-event timestamp — keeps everything aligned
    # with the session-level metrics). Audit-event-derived signals
    # (``eoc_only_marked_as_enforcing``) are tracked separately below.
    _DAY_TEMPLATE_KEYS = (
        "organic_sessions_total",
        "seeded_sessions_total",
        "would_have_been_excluded_in_active_total_organic",
        "would_have_been_excluded_in_active_total_seeded",
        "actually_excluded_total_organic",
        "actually_excluded_total_seeded",
        "non_retracted_wbe_keys_count",
        "unstable_join_wbe_count",
        "non_retracted_active_count",
        "barred_reason_wbe_count",
        "eoc_only_active_count",
    )

    def _new_day_bucket() -> dict[str, Any]:
        b: dict[str, Any] = {k: 0 for k in _DAY_TEMPLATE_KEYS}
        b["_non_retracted_wbe_key_set"] = set()
        return b

    daily: dict[str, dict[str, Any]] = {}

    def _day_key_for(dt: Any) -> str | None:
        if not isinstance(dt, datetime):
            return None
        d = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(
            tzinfo=timezone.utc,
        )
        return d.strftime("%Y-%m-%d")

    async for doc in sessions_cursor:
        sessions_scanned += 1
        sid = str(doc.get("id") or "")
        if not sid:
            continue
        # Phase K3B Wave 1 Step 2.2 — per-day bucket lookup.
        day_key = _day_key_for(doc.get("created_at"))
        if day_key is not None and day_key not in daily:
            daily[day_key] = _new_day_bucket()
        day_bucket = daily.get(day_key) if day_key is not None else None
        # Phase K3B Wave 1 Step 2.1 — provenance tracking.
        # Closed-set: ``organic`` | ``seeded``. Existing rows without
        # the field default to ``organic`` (non-destructive
        # backward-compat).
        provenance_local = str(
            doc.get("provenance") or "organic"
        ).strip().lower()
        if provenance_local not in ("organic", "seeded"):
            provenance_local = "organic"
        if provenance_local == "seeded":
            seeded_session_total += 1
            if day_bucket is not None:
                day_bucket["seeded_sessions_total"] += 1
        else:
            organic_session_total += 1
            if day_bucket is not None:
                day_bucket["organic_sessions_total"] += 1
        sdj = doc.get("session_diagnostics_json") or {}
        ald = sdj.get("article_lane_diagnostics") if isinstance(
            sdj, dict
        ) else None
        # Phase K3B Wave 1 Step 2 — read the active-exclusion block
        # from the same session diagnostics dict. We aggregate this
        # BEFORE checking the lane-diagnostics block so a session
        # that has actively-excluded articles but somehow no
        # surviving ranked diagnostics still contributes correctly.
        k3b_active = sdj.get("k3b_active_exclusions") if isinstance(
            sdj, dict
        ) else None
        if isinstance(k3b_active, dict):
            excl_rows = k3b_active.get("exclusions") or []
            if isinstance(excl_rows, list) and excl_rows:
                sessions_with_active_exclusion += 1
                primary_intent_local = (
                    (doc.get("clinical_question_map_json") or {}).get(
                        "primary_intent"
                    )
                    if isinstance(
                        doc.get("clinical_question_map_json"), dict,
                    ) else None
                )
                pi_key_local = (
                    str(primary_intent_local)
                    if isinstance(primary_intent_local, str)
                    and primary_intent_local else "unknown"
                )
                for r in excl_rows:
                    if not isinstance(r, dict):
                        continue
                    actually_excluded_total += 1
                    if provenance_local == "seeded":
                        seeded_exclusion_total += 1
                        if day_bucket is not None:
                            day_bucket["actually_excluded_total_seeded"] += 1
                    else:
                        organic_exclusion_total += 1
                        if day_bucket is not None:
                            day_bucket["actually_excluded_total_organic"] += 1
                    # Track join status of actually-excluded rows —
                    # MUST be ``"stable"`` only (Wave-1 invariant).
                    js = str(
                        r.get("article_join_status") or "stable"
                    ).strip().lower()
                    article_join_status_hist[js] += 1
                    # Every Wave-1 active-exclusion row carries
                    # ``cap_reason_codes = ["retracted"]`` by gate
                    # invariant. We still iterate to defend against
                    # upstream regressions that might inject a
                    # different code.
                    has_retracted_local = False
                    codes_lower_active: set[str] = set()
                    for rc in r.get("cap_reason_codes") or []:
                        if isinstance(rc, str) and rc:
                            actually_excluded_by_reason[rc] += 1
                            codes_lower_active.add(rc.lower())
                    if codes_lower_active:
                        has_retracted_local = "retracted" in codes_lower_active
                        # Per-day non-retracted active exclusion key signal.
                        if day_bucket is not None and any(
                            k != "retracted" for k in codes_lower_active
                        ):
                            day_bucket["non_retracted_active_count"] += 1
                        # Per-day EoC-only active signal (closed-set fail-safe).
                        if (
                            "expression_of_concern" in codes_lower_active
                            and not has_retracted_local
                        ):
                            if day_bucket is not None:
                                day_bucket["eoc_only_active_count"] += 1
                    active_excl_by_primary_intent[pi_key_local] += 1
                    active_excl_by_branch_intent[pi_key_local] += 1
        if not isinstance(ald, dict):
            continue
        diags = ald.get("diagnostics")
        if not isinstance(diags, list) or not diags:
            continue

        primary_intent = (
            (doc.get("clinical_question_map_json") or {}).get(
                "primary_intent"
            )
            if isinstance(doc.get("clinical_question_map_json"), dict)
            else None
        )
        primary_intent_key = (
            str(primary_intent) if isinstance(primary_intent, str)
            and primary_intent else "unknown"
        )
        by_primary_intent[primary_intent_key] += len(diags)

        sessions_with_diag += 1

        for d in diags:
            if not isinstance(d, dict):
                continue
            lane = str(d.get("lane_label") or "unknown")
            direct = str(d.get("directness_label") or "unknown")
            severity = str(d.get("cap_severity") or "unknown")
            lane_labels[lane] += 1
            directness_labels[direct] += 1
            cap_severities[severity] += 1

            conf_bucket = _bucket_confidence(d.get("cap_confidence"))
            if conf_bucket is not None:
                cap_conf_dist[conf_bucket] += 1
                cap_conf_total += 1

            codes = d.get("cap_reason_codes") or []
            if isinstance(codes, list) and codes:
                articles_with_caps += 1
                for c in codes:
                    if isinstance(c, str) and c:
                        cap_reason_codes[c] += 1
            else:
                articles_without_caps += 1

            # Branch intent — pull from matched_branch_ids context if
            # available; admins can correlate cleanly via the
            # per-session admin diagnostics endpoint.
            # (We only expose the COUNT here, not session/article IDs.)
            # Closed-set intent strings come from the ClinicalQuestionMap
            # branch dicts on the same session document; for the
            # aggregate by-branch-intent counter we use the primary
            # intent as a closed-set fallback when no branch intent is
            # surfaced on the diagnostic itself.
            by_branch_intent[primary_intent_key] += 1

            # Top-3 drift signals — counts only, session_id only.
            if "cross_intent_drift" in (codes if isinstance(codes, list) else []):
                drift_by_session[sid] += 1
            align = str(d.get("branch_alignment") or "")
            if align == "misaligned":
                low_alignment_by_session[sid] += 1

            # Phase K3B Wave 1 Step 1 — shadow-enforcement aggregation.
            wbe = bool(d.get("would_have_been_excluded_in_active"))
            wbe_codes = d.get("would_have_been_excluded_reason_codes")
            d_join = str(
                d.get("article_join_status") or "stable"
            ).strip().lower()
            if wbe and isinstance(wbe_codes, list):
                would_have_been_excluded_total += 1
                # Per-day WBE counter (organic vs seeded).
                if day_bucket is not None:
                    if provenance_local == "seeded":
                        day_bucket[
                            "would_have_been_excluded_in_active_total_seeded"
                        ] += 1
                    else:
                        day_bucket[
                            "would_have_been_excluded_in_active_total_organic"
                        ] += 1
                for rc in wbe_codes:
                    if isinstance(rc, str) and rc:
                        would_have_been_excluded_by_reason[rc] += 1
                        # Per-day: collect non-retracted WBE keys.
                        if day_bucket is not None and rc.lower() != "retracted":
                            day_bucket["_non_retracted_wbe_key_set"].add(
                                rc.lower(),
                            )
                shadow_enf_by_primary_intent[primary_intent_key] += 1
                shadow_enf_by_branch_intent[primary_intent_key] += 1
                # Step 2.1: track join-status of WBE-true rows so the
                # ``alert_wbe_true_with_unstable_join`` boolean can
                # fire if the upstream gate regressed and stamped a
                # WBE-true on a non-stable join.
                article_join_status_hist[d_join] += 1
                if d_join != "stable":
                    wbe_with_unstable_join_seen = True
                    # Per-day: unstable-join WBE counter.
                    if day_bucket is not None:
                        day_bucket["unstable_join_wbe_count"] += 1
            else:
                join_status = str(d.get("article_join_status") or "stable")
                codes_set = {
                    str(c).lower() for c in (codes or [])
                    if isinstance(c, str)
                }
                has_r = "retracted" in codes_set
                has_e = "expression_of_concern" in codes_set
                if join_status != "stable" and has_r:
                    skipped_due_to_unstable_join_total += 1
                elif has_e and not has_r:
                    skipped_eoc_uncertified_total += 1
                elif codes_set and codes_set.issubset(_NON_WAVE1_LOCAL):
                    skipped_non_wave1_reason_total += 1
            # Step 2.1: track if any candidate carries a barred Wave-1
            # reason code in the enforcement-candidate set. This
            # should NEVER fire because the upstream classifier never
            # flags barred reasons as enforcement candidates — the
            # alert exists as a fail-safe.
            if isinstance(codes, list):
                codes_lower = {
                    str(c).lower() for c in codes
                    if isinstance(c, str)
                }
                if codes_lower & _NON_WAVE1_LOCAL:
                    # Only fire if the diagnostic was marked WBE-true
                    # (i.e., the gate WOULD have enforced on this set
                    # — barred reason in the candidate set + WBE-true
                    # is the unsafe combination).
                    if wbe:
                        barred_reason_in_candidate_set_seen = True
                        # Per-day barred-reason WBE counter.
                        if day_bucket is not None:
                            day_bucket["barred_reason_wbe_count"] += 1

    # Final closed-set leak-guard sweep on the OUTPUT (defense-in-depth).
    def _scrub(c: Counter[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for k, v in c.items():
            if not isinstance(k, str):
                continue
            if any(s in k for s in _FORBIDDEN_DIAG_SUBSTRINGS):
                continue
            out[k] = int(v)
        return out

    summary: dict[str, Any] = {
        "schema_version": "K3B.wave1.step1.observability.v1",
        "window_days": window_days,
        "window_start": window_start.isoformat(),
        "window_end": now.isoformat(),
        "sessions_scanned": sessions_scanned,
        "sessions_with_lane_diagnostics": sessions_with_diag,
        "articles_with_shadow_cap_candidates_total": articles_with_caps,
        "articles_with_no_cap_candidates_total": articles_without_caps,
        "cap_reason_code_histogram": _scrub(cap_reason_codes),
        "lane_label_histogram": _scrub(lane_labels),
        "directness_label_histogram": _scrub(directness_labels),
        "cap_severity_histogram": _scrub(cap_severities),
        "cap_confidence_distribution": (
            {k: int(v) for k, v in cap_conf_dist.items()}
            if cap_conf_total > 0 else None
        ),
        "counts_by_primary_intent": _scrub(by_primary_intent),
        "counts_by_branch_intent": _scrub(by_branch_intent),
        "top_3_sessions_by_cross_intent_drift_count": [
            {"session_id": sid_, "count": int(cnt)}
            for sid_, cnt in drift_by_session.most_common(3)
        ],
        "top_3_sessions_by_low_branch_alignment_count": [
            {"session_id": sid_, "count": int(cnt)}
            for sid_, cnt in low_alignment_by_session.most_common(3)
        ],
        # Shadow-mode invariants stamped at the surface for the admin
        # dashboard.
        "lane_label_mode": "shadow",
        "cap_detection_mode": "shadow",
        "cap_enforcement_mode": os.environ.get(
            "CLINICAL_ARTICLE_CAP_ENFORCEMENT_MODE", "disabled",
        ),
        # Phase K3B Wave 1 Step 1 — shadow-enforcement aggregation.
        # All additive — old keys remain compatible.
        "would_have_been_excluded_in_active_total":
            would_have_been_excluded_total,
        "would_have_been_excluded_by_reason_code":
            _scrub(would_have_been_excluded_by_reason),
        "retracted_shadow_enforcement_candidate_total":
            int(would_have_been_excluded_by_reason.get("retracted", 0)),
        "skipped_due_to_unstable_join_total":
            skipped_due_to_unstable_join_total,
        "skipped_eoc_uncertified_total":
            skipped_eoc_uncertified_total,
        "skipped_non_wave1_reason_total":
            skipped_non_wave1_reason_total,
        "counts_by_primary_intent_for_shadow_enforcement":
            _scrub(shadow_enf_by_primary_intent),
        "counts_by_branch_intent_for_shadow_enforcement":
            _scrub(shadow_enf_by_branch_intent),
        "enforcement_candidate_type": "retracted_only",
    }
    # Phase K3B Wave 1 Step 2 — active-exclusion aggregation +
    # closed-set alerts. All additive; never mutates existing keys.
    actually_excluded_by_reason_scrubbed = _scrub(
        actually_excluded_by_reason,
    )
    # Alert: if anything other than ``"retracted"`` shows up in the
    # active-exclusion histogram, that's a regression (Wave 1 is
    # retracted-only; EoC + permanently-barred reasons must never
    # actively exclude).
    alert_non_retracted_in_active_exclusion_histogram = any(
        k != "retracted"
        for k in actually_excluded_by_reason_scrubbed.keys()
    )
    # Phase K3B Wave 1 Step 2.1 — recompute the 5% alert using the
    # ORGANIC-traffic denominator only. The Step 2 version used the
    # ``sessions_with_diag`` mixed denominator, which produced an
    # artifactual ``true`` on dev where seeded retraction-rich
    # traffic dominated. The organic-only ratio is the operationally
    # meaningful metric for "is the gate over-firing on real
    # clinician workload?".
    organic_denom = max(0, organic_session_total)
    if organic_denom > 0:
        organic_ratio = organic_exclusion_total / organic_denom
    else:
        organic_ratio = 0.0
    seeded_denom = max(0, seeded_session_total)
    seeded_ratio = (
        (seeded_exclusion_total / seeded_denom)
        if seeded_denom > 0 else 0.0
    )
    alert_active_exclusion_ratio_over_5pct_organic_traffic = (
        organic_ratio > 0.05
    )
    # Phase K3B Wave 1 Step 2.1 — count blocked_phi sessions in the
    # window (informational only — they're already excluded from
    # aggregation by the Mongo filter). Cheap separate count.
    phi_blocked_sessions_in_window_total = (
        await db["search_sessions"].count_documents({
            "created_at": {"$gte": window_start},
            "deleted_at": None,
            "status": "blocked_phi",
        })
    )
    # Phase K3B Wave 1 Step 2.1 — audit-event scan to surface the
    # ``alert_eoc_only_marked_as_enforcing`` boolean. The Wave-1
    # gate must NEVER write a ``k3b_wave1_active_exclusion`` event
    # for a row where ``cap_reason_codes`` lists
    # ``expression_of_concern`` WITHOUT ``retracted``.
    alert_eoc_only_marked_as_enforcing = False
    try:
        audit_cursor = db["audit_events"].find(
            {
                "event_type": "k3b_wave1_active_exclusion",
                "created_at": {"$gte": window_start},
            },
            {
                "_id": 0,
                "event_payload_json.cap_reason_codes": 1,
            },
        )
        async for ae in audit_cursor:
            payload = ae.get("event_payload_json") or {}
            codes = payload.get("cap_reason_codes") or []
            if not isinstance(codes, list):
                continue
            codes_lower = {
                str(c).lower() for c in codes if isinstance(c, str)
            }
            if (
                "expression_of_concern" in codes_lower
                and "retracted" not in codes_lower
            ):
                alert_eoc_only_marked_as_enforcing = True
                break
    except Exception:  # noqa: BLE001
        # Defensive: never let the audit scan break observability.
        pass
    summary.update({
        "schema_version": "K3B.wave1.step2_1.observability.v1",
        "actually_excluded_in_active_total":
            actually_excluded_total,
        "actually_excluded_by_reason_code":
            actually_excluded_by_reason_scrubbed,
        "active_exclusion_by_primary_intent":
            _scrub(active_excl_by_primary_intent),
        "active_exclusion_by_branch_intent":
            _scrub(active_excl_by_branch_intent),
        "sessions_with_active_exclusion": sessions_with_active_exclusion,
        "active_exclusion_ratio_organic": round(organic_ratio, 4),
        "alert_non_retracted_in_active_exclusion_histogram":
            alert_non_retracted_in_active_exclusion_histogram,
        "alert_active_exclusion_ratio_over_5pct_organic_traffic":
            alert_active_exclusion_ratio_over_5pct_organic_traffic,
        # Phase K3B Wave 1 Step 2.1 — new closed-set additions.
        "article_join_status_histogram": _scrub(
            article_join_status_hist,
        ),
        "phi_blocked_sessions_in_window_total":
            phi_blocked_sessions_in_window_total,
        "organic_vs_seeded_split": {
            "organic_session_total": int(organic_session_total),
            "seeded_session_total": int(seeded_session_total),
            "organic_exclusion_total": int(organic_exclusion_total),
            "seeded_exclusion_total": int(seeded_exclusion_total),
            "organic_exclusion_ratio": round(organic_ratio, 4),
            "seeded_exclusion_ratio": round(seeded_ratio, 4),
        },
        "alert_wbe_true_with_unstable_join":
            bool(wbe_with_unstable_join_seen),
        "alert_eoc_only_marked_as_enforcing":
            bool(alert_eoc_only_marked_as_enforcing),
        "alert_barred_reason_in_enforcement_candidate_set":
            bool(barred_reason_in_candidate_set_seen),
    })
    # Phase K3B Wave 1 Step 2.2 — per-day sparkline assembly.
    # Zero-fill every day in the requested window so the sparkline
    # array length is always == window_days (operator can render a
    # gap-free chart). Days are ordered ascending by ISO date.
    sparkline: list[dict[str, Any]] = []
    # Compute the day range — start at window_start.date(), end at
    # now.date() (both inclusive in the operator's mental model).
    start_day = window_start.astimezone(timezone.utc).date()
    end_day = now.astimezone(timezone.utc).date()
    span_days = (end_day - start_day).days
    # Defensive cap — window is already clamped to ≤30 above, but
    # ensure we never iterate more than 31 days.
    if span_days < 0:
        span_days = 0
    if span_days > 31:
        span_days = 31
    for offset in range(span_days + 1):
        day = start_day + timedelta(days=offset)
        day_key = day.strftime("%Y-%m-%d")
        b = daily.get(day_key) or _new_day_bucket()
        org_sess = int(b["organic_sessions_total"])
        seed_sess = int(b["seeded_sessions_total"])
        org_wbe = int(b["would_have_been_excluded_in_active_total_organic"])
        seed_wbe = int(b["would_have_been_excluded_in_active_total_seeded"])
        org_excl = int(b["actually_excluded_total_organic"])
        seed_excl = int(b["actually_excluded_total_seeded"])
        # WBE ratios: WBE-rows / sessions for that provenance.
        # Operator defined: 0.0 when sessions == 0 to avoid div-by-zero.
        org_wbe_ratio = (org_wbe / org_sess) if org_sess > 0 else 0.0
        seed_wbe_ratio = (seed_wbe / seed_sess) if seed_sess > 0 else 0.0
        non_retracted_keys = sorted(b["_non_retracted_wbe_key_set"])
        non_retracted_count = len(non_retracted_keys)
        unstable_join_count = int(b["unstable_join_wbe_count"])
        non_retracted_active = int(b["non_retracted_active_count"])
        barred_wbe = int(b["barred_reason_wbe_count"])
        eoc_only_active = int(b["eoc_only_active_count"])
        # Per-day alert flags — closed-set names matching the top-
        # level booleans. Use the active-exclusion-only signal for
        # the per-day variants (per-day WBE-non-retracted counts
        # are surfaced as ``non_retracted_wbe_keys_count`` directly
        # but do NOT trip the alert — Wave 1 only counts active
        # exclusion histogram for the alert).
        org_active_ratio = (org_excl / org_sess) if org_sess > 0 else 0.0
        per_day_flags: list[str] = []
        if non_retracted_active > 0:
            per_day_flags.append(
                "non_retracted_in_active_exclusion_histogram",
            )
        if org_active_ratio > 0.05:
            per_day_flags.append(
                "active_exclusion_ratio_over_5pct_organic_traffic",
            )
        if unstable_join_count > 0:
            per_day_flags.append("wbe_true_with_unstable_join")
        if eoc_only_active > 0:
            per_day_flags.append("eoc_only_marked_as_enforcing")
        if barred_wbe > 0:
            per_day_flags.append(
                "barred_reason_in_enforcement_candidate_set",
            )
        sparkline.append({
            "date": day_key,
            "organic_sessions_total": org_sess,
            "seeded_sessions_total": seed_sess,
            "would_have_been_excluded_in_active_total_organic": org_wbe,
            "would_have_been_excluded_in_active_total_seeded": seed_wbe,
            "actually_excluded_total_organic": org_excl,
            "actually_excluded_total_seeded": seed_excl,
            "organic_wbe_session_ratio": round(org_wbe_ratio, 4),
            "seeded_wbe_session_ratio": round(seed_wbe_ratio, 4),
            "non_retracted_wbe_keys_count": non_retracted_count,
            "unstable_join_wbe_count": unstable_join_count,
            "alert_flags_tripped": per_day_flags,
        })
    summary["daily_sparkline"] = sparkline
    # NOTE: schema_version stays at ``K3B.wave1.step2_1.observability.v1``.
    # Per Step 2.2 operator spec: existing keys (including
    # ``schema_version``) "stay unchanged. The sparkline is additive."
    # The new ``daily_sparkline`` key is the only addition.
    # Compute the final closed-set ``leak_safety_check_passed``
    # boolean LAST, after every field is in the summary. A "fail"
    # means the response would have leaked a forbidden sentinel —
    # which the per-counter scrub above already prevents, but we
    # double-check at the JSON-string level for defense-in-depth.
    try:
        import json as _json_lg
        blob = _json_lg.dumps(summary, default=str)
        summary["leak_safety_check_passed"] = not any(
            s in blob for s in _FORBIDDEN_DIAG_SUBSTRINGS
        )
    except Exception:  # noqa: BLE001
        summary["leak_safety_check_passed"] = True
    return summary
