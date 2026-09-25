"""Interpretation baseline over the frozen benchmark (W1.5).

    python -m clinical_engine.benchmark.run_interpretation_baseline [--model]

Scores question-type agreement against the cases' provisional engine labels
(not physician-adjudicated) and checks no case gets an age group it did not
state. Writes a versioned JSON result; never touches any database.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from clinical_engine import ENGINE_VERSION
from clinical_engine.benchmark.loader import FROZEN_SHA256, load
from clinical_engine.gateway import ModelGateway
from clinical_engine.interpretation.service import interpret


async def main(use_model: bool, out: Path) -> dict:
    bench, digest = load()
    gateway = ModelGateway() if use_model else None
    rows = []
    for case in bench.cases:
        result = await interpret(case.question, gateway=gateway, allow_model=use_model)
        cqm = result.map
        stated_age = any(e.role == "age_group" and e.source == "user_text" for e in cqm.elements)
        rows.append({
            "id": case.id, "split": case.split, "expected_type": case.engine_question_type,
            "got_type": cqm.question_type, "type_match": cqm.question_type == case.engine_question_type,
            "age_group": cqm.age_group(), "age_applied_without_statement": cqm.age_group() != "not_stated" and not stated_age,
            "blocking_conflicts": [c.code for c in result.conflicts if c.severity == "blocking"],
            "method": result.meta.method, "fallback_reason": result.meta.fallback_reason,
            "tokens": result.meta.tokens_in + result.meta.tokens_out, "latency_ms": result.meta.latency_ms,
            "summary": cqm.summary,
        })
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    report = {
        "benchmark_version": bench.benchmark_version, "benchmark_sha256": digest,
        "frozen_hash_ok": digest == FROZEN_SHA256, "engine_version": ENGINE_VERSION, "commit": commit,
        "run_at": datetime.now(timezone.utc).isoformat(), "mode": "model" if use_model else "rules_only",
        "evidence_label": "live" if use_model else "fixture",
        "labels_note": "expected types are provisional engine labels, not physician-adjudicated",
        "cases": len(rows),
        "type_agreement": sum(r["type_match"] for r in rows),
        "silent_age_inferences": sum(r["age_applied_without_statement"] for r in rows),
        "total_tokens": sum(r["tokens"] for r in rows),
        "rows": rows,
    }
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    target = Path(args.out or f"baseline_interpretation_{'model' if args.model else 'rules'}.json")
    r = asyncio.run(main(args.model, target))
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))
