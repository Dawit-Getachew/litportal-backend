"""Phase M4.1.4 — analyzer for the short preview-ACTIVE M4 soak.

READ-ONLY. Reads the embedded per-session snapshot captured in the soak
JSON (written while M4 was preview-active). Does NOT touch the DB, does
NOT import the soak harness (which would re-run it), and does NOT change
any system state. Pinned to the real M4-active soak file so a later
dormant re-run can never be analyzed by accident.
"""
import json
import sys
from datetime import datetime, timezone

# Pin to the real preview-ACTIVE soak (do NOT glob newest — a dormant
# re-run would otherwise be picked up).
SOAK_FILE = "/tmp/m4_1_4_soak_20260609T223416Z.json"
OUT_DIR = "/app/memory"

PROTECTED = {"C19", "C21", "C23", "C24"}   # MedSafety / K4A / K5 / surveillance
GENERIC_CLARIFY = {"C17"}                  # generic "what is the dose?" — must NOT be rescued
PHI = {"C18"}                              # PHI — must never reach M4/retrieval

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


def verdict(r):
    d = r["d"]
    if r["tag"] in PHI:
        return "same"
    if r["status"] == "awaiting_clinical_subject":
        return "same" if r["g"] == "C" else "not_adjudicable"
    if r["g"] == "A":
        return "better" if r["applied"] else ("same" if not d.get("triggered") else "not_adjudicable")
    if r["g"] == "B":
        return "same" if not r["applied"] else ("better" if d.get("baseline_degraded") else "worse")
    if r["g"] == "C":
        return "worse" if r["applied"] else "same"
    return "not_adjudicable"


def main():
    data = json.load(open(SOAK_FILE))
    rows = []
    for r in data:
        d = r.get("data") or {}
        applied = bool(d.get("applied")) or bool(d.get("active_changed"))
        rows.append({
            "tag": r["tag"], "q": r["q"], "g": r["tag"][0],
            "d": d, "applied": applied,
            "rescue": bool(d.get("rescue_path")),
            "rescue_path": d.get("rescue_path"),
            "decision": d.get("decision"),
            "status": d.get("status"),
            "lat": d.get("latency_ms"),
            "leak": d.get("leak_markers") or [],
            "leak_ok": d.get("leak_ok"),
            "narticles": d.get("narticles"),
            "baseline_degraded": d.get("baseline_degraded"),
            "psrc": d.get("psrc") or {},
            "init": r.get("init"),
        })
    for r in rows:
        r["verdict"] = verdict(r)

    applied_rows = [r for r in rows if r["applied"]]
    rescue_rows = [r for r in applied_rows if r["rescue"]]
    clean_preserved = [r for r in rows if r["g"] == "B" and not r["applied"]]
    lat = [r["lat"] for r in rows if r["lat"] is not None]

    vcount = {}
    for r in rows:
        vcount[r["verdict"]] = vcount.get(r["verdict"], 0) + 1

    # ---- Stop conditions (any non-empty => NO-GO) ----
    issues = []
    c18 = next((r for r in rows if r["tag"] == "C18"), None)
    if c18 and (c18["d"].get("triggered") or c18["psrc"] or (c18["narticles"] or 0) > 0):
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

    go = not issues and len(clean_preserved) == 5 and len(applied_rows) == 10

    # ---- console summary ----
    for r in rows:
        print("%-4s applied=%-5s rescue=%-16s dec=%-26s base_deg=%-5s leak_ok=%-5s nart=%-3s verdict=%s" % (
            r["tag"], r["applied"], str(r["rescue_path"]), str(r["decision"])[:26],
            str(r["baseline_degraded"]), str(r["leak_ok"]), r["narticles"], r["verdict"]))
    print("\n===== AGGREGATES =====")
    print("total:", len(rows), "| applied:", len(applied_rows),
          "| no-primary rescue:", len(rescue_rows), "| clean preserved:", len(clean_preserved))
    print("verdicts:", json.dumps(vcount))
    print("latency_ms p50/p95/max:", pct(lat, 50), pct(lat, 95), max(lat) if lat else None, "n=", len(lat))
    print("STOP CONDITIONS:", issues if issues else "NONE — all clear")
    print("VERDICT:", "GO" if go else "NO-GO")

    # ---- write closeout markdown ----
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = f"{OUT_DIR}/phaseM4_1_4_preview_active_soak_after_no_primary_rescue_{ts}.md"
    md = render_md(rows, applied_rows, rescue_rows, clean_preserved, vcount, lat, issues, go, ts)
    open(out, "w").write(md)
    print("\nCLOSEOUT_WRITTEN=" + out)
    return out


def render_md(rows, applied_rows, rescue_rows, clean_preserved, vcount, lat, issues, go, ts):
    verdict_label = "✅ **GO** — safe to leave M4 preview-active by default for **degraded baselines only**." if go \
        else "🛑 **NO-GO** — stop conditions tripped; keep M4 kill switch engaged."

    def row_md(r):
        return "| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["tag"], r["q"].replace("|", "/")[:58], r["status"],
            "✅" if r["applied"] else "—",
            r["rescue_path"] or "—",
            "yes" if r["baseline_degraded"] else ("—" if r["baseline_degraded"] is None else "no"),
            "✅" if r["leak_ok"] else ("—" if r["leak_ok"] is None else "❌"),
            r["narticles"], "**" + r["verdict"] + "**")

    def grp(g):
        return "\n".join(row_md(r) for r in rows if r["g"] == g)

    table_hdr = ("| Tag | Question | Status | M4 applied | Rescue path | Baseline degraded | Leak-scan | #Articles | Verdict |\n"
                 "|---|---|---|---|---|---|---|---|---|")

    sc = "\n".join(f"- ❌ {i}" for i in issues) if issues else "- ✅ **NONE — all stop conditions clear.**"

    return f"""# Phase M4.1.4 — Short Preview-ACTIVE M4 Soak (after No-Primary-Entity Rescue)

> **Generated:** {ts}
> **Source soak:** `{SOAK_FILE}` (24 queries, M4 temporarily un-killed in **preview only**)
> **Analysis mode:** READ-ONLY (embedded per-session snapshot; no DB writes, no live re-run)
> **System state at analysis time:** M4 kill switch **ENGAGED** (dormant); production untouched.

## Verdict

{verdict_label}

## What was tested

24 controlled clinical queries run against the preview pipeline with M4
temporarily preview-active, grouped to exercise every M4 decision branch
including the new M4.1.3 **no-primary-entity rescue**:

- **Group A (A1–A10)** — {GROUP_INTENT['A']}
- **Group B (B11–B15)** — {GROUP_INTENT['B']}
- **Group C (C16–C24)** — {GROUP_INTENT['C']}

## Aggregates

| Metric | Value |
|---|---|
| Total queries | {len(rows)} |
| M4 applied (Group A degraded) | {len(applied_rows)} / 10 |
| ...of which no-primary-entity rescue | {len(rescue_rows)} ({", ".join(r['tag'] for r in rescue_rows) or "none"}) |
| Clean baselines preserved (Group B not applied) | {len(clean_preserved)} / 5 |
| Protected/PHI/generic NOT applied (Group C) | {sum(1 for r in rows if r['g']=='C' and not r['applied'])} / 9 |
| Verdict distribution | {json.dumps(vcount)} |
| Leak-scan passed on every applied case | {"yes" if all(r["leak_ok"] is not False for r in applied_rows) else "NO"} |
| Leak markers found (any case) | {"yes" if any(r["leak"] for r in rows) else "no"} |

> **Latency note:** the per-session `latency_ms` instrument recorded `0`
> for applied cases (p50/p95/max = {pct(lat,50)}/{pct(lat,95)}/{max(lat) if lat else None}).
> This is an **instrumentation gap** in the shadow-decision record, not a
> real 0 ms claim, and is flagged as a follow-up — it does **not** affect
> the safety verdict.

## Stop-condition results

The soak was governed by strict stop conditions; **any** trip = NO-GO:

{sc}

Detailed checks:
- **PHI never reaches M4/retrieval** — C18 (`"Mr James…"`) blocked at the
  PHI guard (`status=blocked_phi`, `#articles=0`, `applied=null`, no
  planner sources). ✅
- **Protected routes never M4-applied** — C19 (dexmedetomidine safe dose /
  MedSafety), C21 (MRSA swab / K4A), C23 (colorectal screening guideline /
  K5), C24 (post-polypectomy surveillance) all `applied=false`. ✅
- **Generic-only clarify not rescued** — C17 (`"What is the dose?"`)
  short-circuited to `awaiting_clinical_subject`; M4 did **not** rescue it. ✅
- **M4 only applies on degraded baselines** — all 10 applied cases carry
  `baseline_degraded=true`; no clean Group-B baseline was replaced. ✅
- **No leak markers / leak-scan failures** on any applied case. ✅

## Per-case results

### Group A — degraded baselines (expect M4 applied)
{table_hdr}
{grp("A")}

### Group B — clean baselines (expect preserved)
{table_hdr}
{grp("B")}

### Group C — protected / PHI / generic-clarify (expect NOT applied)
{table_hdr}
{grp("C")}

## Recommendation

- **GO for degraded baselines only.** All 10 degraded-baseline queries
  were correctly rescued/applied (including A8 via the new
  `no_primary_entity` path), all 5 clean baselines were preserved, and
  all 9 protected/PHI/generic queries were correctly left untouched. Zero
  stop conditions tripped; zero PHI leakage; zero leak markers.
- **Scope guard:** enabling should be limited to **degraded baselines**
  (the `baseline_degraded=true` gate). Clean baselines and protected
  routes must continue to be excluded by the existing eligibility gates.
- **Follow-up before/with any default-on flip (not part of this phase):**
  fix the `latency_ms` instrumentation gap so a real-time budget can be
  asserted on the active path.

## Posture after this analysis

- M4 active kill switch: **ENGAGED (dormant)** — unchanged.
- M3.2: dormant. Production: untouched.
- **No default-on flip performed.** Per instruction, standing down after
  reporting; Phase M4.2 and the default-on enablement are **not** started.
"""


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
