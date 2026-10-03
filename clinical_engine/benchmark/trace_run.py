"""Stage trace of a development benchmark case's references in one engine run (W2.2).

    python -m clinical_engine.benchmark.trace_run --case W15-07 --run <run_id> [--json out.json]

Every reference ends in the evidence list (main / other / more) or gets the
stage it was lost at, with the evidence (see clinical_engine.retrieval.trace).
Development cases only: held-out references are kept by the physician and are
never traced or labelled by the developer.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from clinical_engine import store
from clinical_engine.benchmark.loader import load
from clinical_engine.retrieval import trace
from services import engine_concept_plan as ecp


async def _main(case_id: str, run_id: str, out: str | None) -> int:
    from clinical_engine.contracts.question import ClinicalQuestionMap
    from clinical_engine.orchestrator.tasks import approved_concept_plan

    bench, _ = load()
    case = next((c for c in bench.cases if c.id == case_id), None)
    if case is None:
        print(f"unknown case {case_id}", file=sys.stderr)
        return 2
    if case.split != "dev":
        print(f"{case_id} is held out: its references are not traced by the developer.", file=sys.stderr)
        return 2
    run = await store.get_run(run_id)
    if run is None or run.get("retrieval_adapter") != "engine":
        print(f"run {run_id} is not an engine run", file=sys.stderr)
        return 2
    version = await store.get_version_by_id(run["question_version_id"])
    raw = approved_concept_plan(version, ClinicalQuestionMap.model_validate(version["map"]))
    plan = ecp.validate_plan(raw) if raw else None
    refs = [r.model_dump() for r in case.must_find]
    traces = await trace.trace(run_id, refs, plan=plan)
    print(f"{case.id} — {case.question}\nrun {run_id} ({run['status']})")
    for t in traces:
        where = (f"{t['section']} #{t['position']} ({t['state']})" if t.get("stage") == "listed"
                 else f"LOST at {t['lost_at']}: {t.get('detail', '')}")
        print(f"  {t['pmid']:>9}  {where}\n             {str(t.get('title'))[:100]}")
    summary = trace.summary(traces)
    print(json.dumps(summary))
    if out:
        with open(out, "w") as fh:
            json.dump({"case": case.id, "run_id": run_id, "summary": summary, "traces": traces}, fh, indent=2,
                      default=str)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    sys.exit(asyncio.run(_main(a.case, a.run, a.json)))


if __name__ == "__main__":
    main()
