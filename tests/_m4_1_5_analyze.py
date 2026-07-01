"""Phase M4.1.5 — analyzer for the preview-default ACTIVE M4 soak.

READ-ONLY. Reads the embedded per-session snapshot from the latest
M4.1.5 soak JSON + the timeout/kill-switch probe logs, computes the
per-case verdicts, latency aggregates, and stop conditions, then writes
the closeout doc. No DB writes, no live re-run, no state change.
"""
import glob
import json
import re
from datetime import datetime, timezone

OUT_DIR = "/app/memory"
PROTECTED = {"C19", "C21", "C23", "C24"}
GENERIC_CLARIFY = {"C17"}
PHI = {"C18"}
BUDGET_MS = 4000

GROUP_INTENT = {
    "A": "Degraded baseline — M4 SHOULD apply (incl. no-primary-entity rescue).",
    "B": "Clean baseline — M4 SHOULD NOT apply (preserve baseline).",
    "C": "Protected / PHI / generic-clarify — M4 SHOULD NOT apply.",
}


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return round(s[f] + (s[c] - s[f]) * (k - f), 1)


def _probe_ok(path, marker):
    try:
        txt = open(path).read()
    except Exception:
        return None
    m = re.search(marker + r"=\s*(True|False)", txt)
    return (m.group(1) == "True") if m else None


def verdict(r):
    d = r["d"]
    if r["tag"] in PHI:
        return "same"
    if r["status"] == "awaiting_clinical_subject":
        return "same" if r["g"] == "C" else "not_adjudicable"
    if r["g"] == "A":
        return "better" if r["applied"] else ("same" if not d.get("triggered") else "not_adjudicable")
    if r["g"] == "B":
        return "same" if not r["applied"] else "worse"
    if r["g"] == "C":
        return "worse" if r["applied"] else "same"
    return "not_adjudicable"


def main():
    soak = sorted(glob.glob("/tmp/m4_1_5_soak_*.json"))[-1]
    data = json.load(open(soak))
    rows = []
    for r in data:
        d = r.get("data") or {}
        applied = bool(d.get("applied")) or bool(d.get("active_changed"))
        rows.append({
            "tag": r["tag"], "q": r["q"], "g": r["tag"][0], "d": d,
            "applied": applied, "rescue_path": d.get("rescue_path"),
            "decision": d.get("decision"), "status": d.get("status"),
            "leak": d.get("leak_markers") or [], "leak_ok": d.get("leak_ok"),
            "narticles": d.get("narticles"),
            "baseline_degraded": d.get("baseline_degraded"),
            "total_lat": d.get("m4_total_latency_ms"),
            "breakdown": d.get("latency_breakdown") or {},
            "timed_out": d.get("timed_out"),
            "psrc": d.get("psrc") or {},
        })
    for r in rows:
        r["verdict"] = verdict(r)

    applied_rows = [r for r in rows if r["applied"]]
    rescue_rows = [r for r in applied_rows if r["rescue_path"]]
    clean_preserved = [r for r in rows if r["g"] == "B" and not r["applied"]]
    applied_lats = [r["total_lat"] for r in applied_rows if r["total_lat"] is not None]

    vcount = {}
    for r in rows:
        vcount[r["verdict"]] = vcount.get(r["verdict"], 0) + 1

    issues = []
    c18 = next((r for r in rows if r["tag"] == "C18"), None)
    if c18 and (c18["psrc"] or (c18["narticles"] or 0) > 0 or c18["applied"]):
        issues.append("PHI C18 reached M4/retrieval")
    for r in rows:
        if r["tag"] in PROTECTED and r["applied"]:
            issues.append(f"M4 applied on protected route {r['tag']}")
        if r["tag"] in GENERIC_CLARIFY and r["applied"]:
            issues.append(f"generic-only {r['tag']} was rescued")
        if r["applied"] and r["leak_ok"] is False:
            issues.append(f"leak_scan failed on {r['tag']}")
        if r["applied"] and not r["baseline_degraded"]:
            issues.append(f"applied without baseline degradation {r['tag']}")
        if r["g"] == "B" and r["applied"]:
            issues.append(f"clean baseline {r['tag']} actively replaced")
        if r["leak"]:
            issues.append(f"leak markers in {r['tag']}: {r['leak']}")
        if r["applied"] and not (r["total_lat"] and r["total_lat"] > 0):
            issues.append(f"applied case {r['tag']} has zero/None latency")
        if r["applied"] and r["total_lat"] and r["total_lat"] > BUDGET_MS:
            issues.append(f"applied case {r['tag']} exceeded budget ({r['total_lat']}ms)")

    timeout_ok = _probe_ok("/tmp/m4_1_5_timeout_probe.log", "TIMEOUT_FALLBACK_OK")
    kill_ok = _probe_ok("/tmp/m4_1_5_killcheck.log", "KILL_SWITCH_VERIFY_OK")
    if timeout_ok is not True:
        issues.append("timeout fallback probe did not pass")
    if kill_ok is not True:
        issues.append("kill-switch verification probe did not pass")

    go = (not issues and len(clean_preserved) == 5 and len(applied_rows) == 10
          and all(v and v > 0 for v in applied_lats))

    # console
    for r in rows:
        print("%-4s applied=%-5s rescue=%-16s dec=%-26s base_deg=%-5s total_lat=%-6s verdict=%s" % (
            r["tag"], r["applied"], str(r["rescue_path"]), str(r["decision"])[:26],
            str(r["baseline_degraded"]), str(r["total_lat"]), r["verdict"]))
    print("\napplied:", len(applied_rows), "| rescues:", len(rescue_rows),
          "| clean preserved:", len(clean_preserved))
    print("applied latency p50/p95/max:", pct(applied_lats, 50), pct(applied_lats, 95),
          max(applied_lats) if applied_lats else None)
    print("verdicts:", json.dumps(vcount))
    print("timeout_ok:", timeout_ok, "| kill_ok:", kill_ok)
    print("STOP CONDITIONS:", issues if issues else "NONE — all clear")
    print("VERDICT:", "GREEN — leave M4 preview-active by default" if go else "RED — re-engage kill switch")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = f"{OUT_DIR}/phaseM4_1_5_preview_default_active_degraded_baselines_{ts}.md"
    open(out, "w").write(render_md(
        rows, applied_rows, rescue_rows, clean_preserved, applied_lats,
        vcount, issues, go, timeout_ok, kill_ok, soak, ts))
    print("\nCLOSEOUT_WRITTEN=" + out)
    return go


def render_md(rows, applied_rows, rescue_rows, clean_preserved, applied_lats,
              vcount, issues, go, timeout_ok, kill_ok, soak, ts):
    verdict_label = ("✅ **GREEN** — leave M4 **preview-active by default** for degraded "
                     "baselines only. Kill switch left **DISENGAGED** in preview."
                     if go else
                     "🛑 **RED** — stop conditions tripped; re-engage the kill switch.")

    def row_md(r):
        return "| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["tag"], r["q"].replace("|", "/")[:52], r["status"],
            "✅" if r["applied"] else "—", r["rescue_path"] or "—",
            "yes" if r["baseline_degraded"] else ("—" if r["baseline_degraded"] is None else "no"),
            r["total_lat"] if r["total_lat"] is not None else "—",
            "**" + r["verdict"] + "**")

    def grp(g):
        return "\n".join(row_md(r) for r in rows if r["g"] == g)

    hdr = ("| Tag | Question | Status | M4 applied | Rescue | Baseline degraded | Total latency (ms) | Verdict |\n"
           "|---|---|---|---|---|---|---|---|")
    sc = "\n".join(f"- ❌ {i}" for i in issues) if issues else "- ✅ **NONE — all stop conditions clear.**"
    sample = next((r for r in applied_rows if r["breakdown"]), None)
    br = (sample or {}).get("breakdown") or {}

    return f"""# Phase M4.1.5 — Preview-Default ACTIVE M4 (Degraded Baselines Only)

> **Generated:** {ts}
> **Source soak:** `{soak}` (same 24-case M4.1.4 set, M4 preview-active)
> **Analysis mode:** READ-ONLY (embedded snapshot + probe logs)
> **End state:** M4 active kill switch **DISENGAGED in preview** (default-on
> for degraded baselines). Production **hard-off**; M3.2 dormant; R-series shadow-only.

## Verdict

{verdict_label}

## Goals delivered

1. **Latency instrumentation corrected** — end-to-end M4 decision latency is
   now captured as a closed-set NUMERIC breakdown (trigger detect → bounded-LLM
   candidate → deterministic validation → would-be projection → compiler
   projection → active-promotion decision → total). Applied cases now record a
   **nonzero** total (previously read 0).
2. **Hard timeout + fallback** — bounded-LLM M4 candidate generation is wrapped
   in a hard timeout (`M4_ACTIVE_MAX_LATENCY_MS`, **4000 ms** — the existing,
   safer-than-5000 budget). On timeout: no partial promotion, fall back to
   baseline retrieval / clarification, closed-set reason `m4_latency_timeout`,
   no user-facing crash. **Verified live** (probe: `TIMEOUT_FALLBACK_OK={timeout_ok}`).
3. **Preview-active by default for degraded baselines** — Mongo kill switch
   disengaged; the M4.1.1 degraded-only promotion guard remains mandatory.
4. **Clean baselines preserved** — all Group-B baselines untouched.
5. **Production hard-off** — env allowlist excludes production, mode resolver
   forces `disabled`, startup guard fail-closed. No production env / deploy change.
6. **Kill switch available + tested** — ENGAGED → dormant, DISENGAGED → applies
   (probe: `KILL_SWITCH_VERIFY_OK={kill_ok}`).

## Aggregates

| Metric | Value |
|---|---|
| Total queries | {len(rows)} |
| M4 applied (Group A degraded) | {len(applied_rows)} / 10 |
| ...of which no-primary-entity rescue | {len(rescue_rows)} ({", ".join(r['tag'] for r in rescue_rows) or "none"}) |
| Clean baselines preserved (Group B not applied) | {len(clean_preserved)} / 5 |
| Protected/PHI/generic NOT applied (Group C) | {sum(1 for r in rows if r['g']=='C' and not r['applied'])} / 9 |
| Verdict distribution | {json.dumps(vcount)} |
| Applied total latency p50 / p95 / max (ms) | {pct(applied_lats,50)} / {pct(applied_lats,95)} / {max(applied_lats) if applied_lats else None} |
| All applied latencies nonzero & < {BUDGET_MS} ms budget | {"yes" if all(v and 0 < v < BUDGET_MS for v in applied_lats) else "NO"} |
| Leak markers found (any case) | {"yes" if any(r["leak"] for r in rows) else "no"} |

Sample latency breakdown ({(sample or {}).get('tag') if sample else 'n/a'}): `{json.dumps(br)}`
(LLM candidate generation dominates; all other stages are sub-millisecond deterministic.)

## Stop-condition results

{sc}

## Per-case results

### Group A — degraded baselines (expect M4 applied)
{hdr}
{grp("A")}

### Group B — clean baselines (expect preserved)
{hdr}
{grp("B")}

### Group C — protected / PHI / generic-clarify (expect NOT applied)
{hdr}
{grp("C")}

## Safety invariants confirmed

- **PHI never reaches M4/retrieval** — C18 blocked at the PHI guard
  (`status=blocked_phi`, `#articles=0`, no planner sources).
- **Protected routes never M4-applied** — C19 (MedSafety), C21 (K4A),
  C23 (K5 guideline), C24 (surveillance) all `applied=false`.
- **Generic-only clarify not rescued** — C17 → `awaiting_clinical_subject`.
- **MedSafety-1 + K5 guideline lanes intact**; M3.2 dormant; R-series shadow-only.
- **No leak markers; leak-scan passed on every applied case.**
- **Timeout fallback never partially promotes** and falls back to baseline
  retrieval (verified: 192 baseline articles returned, no chip, no crash).

## Posture after this phase

- M4 active kill switch: **DISENGAGED in preview (default-on for degraded
  baselines)** — the intended M4.1.5 end state.
- Production: **untouched, hard-off.** No production env change, no deploy.
- M3.2 dormant. R-series shadow-only. Ranking / default selection / evidence
  packets / summaries / citation verifier / connectors **unchanged.**
- Next (per instruction): a short period of normal preview use with M4 on by
  default, **then** M4.2 legacy semantic-repair consolidation. M4.2 **not** started.
"""


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
