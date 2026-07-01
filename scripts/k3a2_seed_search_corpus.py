"""Phase K3A.2 — Deterministic seed harness for retraction-rich topics.

When organic preview traffic produces too few ``retracted`` /
``expression_of_concern`` candidates to meet the K3A.2 sample-size
floor (≥20 combined), this harness submits a small list of clinical
evidence questions that target well-documented retraction clusters
(HCQ/COVID Surgisphere, MMR/autism Wakefield, ivermectin/COVID,
stem-cell heart, etc).

The harness does NOT mock the pipeline, fabricate retraction status,
or inject articles directly. It only submits non-PHI clinical
evidence questions through the normal authed API surface and lets the
existing pipeline (PHI guard → CQM → K2 shadow plan → retrieval →
metadata normalizer → K3A classifier) do its work. The PubMed
connector is the source of truth for ``Retracted Publication`` /
``Expression of Concern`` pub-type stamping, the metadata normalizer
propagates ``is_retracted`` / ``has_expression_of_concern`` on the
article doc, and the K3A classifier deterministically stamps the
``retracted`` / ``expression_of_concern`` cap candidates downstream.

Seed questions are intentionally generic clinical evidence prompts
(no patient details, no PHI). The list itself is deterministic and
auditable — see ``SEED_QUESTIONS`` below.

Usage::

    python -m scripts.k3a2_seed_search_corpus \\
        --api-url https://med-evidence-qa.preview.emergentagent.com \\
        --email   clinician@test.com \\
        --password ClinicianTest123! \\
        --wait-secs 90

The harness prints one line per submitted question with the resulting
session_id and final status, and waits for each session to reach
``awaiting_selection`` (or hit the wait timeout) before moving to the
next question. Concurrency is intentionally 1 to avoid hammering
PubMed E-utilities.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Clinical evidence questions on topics with KNOWN retraction /
# expression-of-concern clusters indexed in PubMed.
SEED_QUESTIONS: tuple[str, ...] = (
    # HCQ + COVID — Surgisphere/Mehra retractions, NEJM + Lancet.
    "What is the evidence for hydroxychloroquine in COVID-19 treatment?",
    # MMR + autism — Wakefield 1998 Lancet retraction.
    "What is the evidence linking MMR vaccine and autism in children?",
    # Ivermectin + COVID — Elgazzar, Niaee, Carvallo retractions.
    "What is the evidence for ivermectin in COVID-19 treatment?",
    # Reuben anesthesia / analgesia fraud cluster.
    "What is the evidence for multimodal analgesia in orthopedic surgery?",
    # Anversa cardiac stem-cell retractions.
    "What is the evidence for stem cell therapy in heart failure?",
    # Macchiarini trachea-transplant retractions and EoCs.
    "What are the outcomes of synthetic trachea transplantation in adults?",
    # Das resveratrol retractions.
    "What is the evidence for resveratrol in cardiovascular disease?",
    # STAP cells / induced pluripotency — Obokata retractions.
    "What is the evidence for stimulus-triggered acquisition of pluripotency in cells?",
    # Schon / molecular electronics — outside biomedical, drop.
    # Andrew Wakefield co-author / autism — second-order retractions.
    "What is the evidence on thimerosal in vaccines and developmental outcomes?",
    # COVID convalescent plasma — multiple corrections / EoCs.
    "What is the evidence for convalescent plasma in COVID-19?",
)


def _post(url: str, body: dict, token: str | None = None) -> dict:
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "clinevidence-k3a2-seed/1.0",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return json.loads(resp.read())


def _get(url: str, token: str) -> dict:
    req = urllib.request.Request(
        url, headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "clinevidence-k3a2-seed/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return json.loads(resp.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", required=True)
    ap.add_argument("--email", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--wait-secs", type=int, default=120)
    ap.add_argument("--poll-secs", type=int, default=3)
    args = ap.parse_args()

    api = args.api_url.rstrip("/")

    login = _post(
        f"{api}/api/auth/login",
        {"email": args.email, "password": args.password},
    )
    token = login["access_token"]
    print(f"logged in as {args.email}; token length={len(token)}")

    submitted: list[dict] = []
    for q in SEED_QUESTIONS:
        try:
            r = _post(
                f"{api}/api/search-sessions/",
                {"question": q}, token=token,
            )
        except Exception as e:  # noqa: BLE001
            print(f"submit failed: {e!r} q={q[:60]!r}")
            continue
        sid = r.get("id") or ""
        status = r.get("status") or ""
        print(f"submitted sid={sid} status={status} q={q[:70]!r}")
        if not sid:
            continue
        # Poll until awaiting_selection or wait_secs elapses.
        t0 = time.time()
        final_status = status
        while time.time() - t0 < args.wait_secs:
            try:
                s = _get(
                    f"{api}/api/search-sessions/{sid}/status",
                    token=token,
                )
            except Exception:  # noqa: BLE001
                break
            final_status = s.get("status") or ""
            if final_status in ("awaiting_selection", "complete", "error"):
                break
            time.sleep(args.poll_secs)
        print(f"  → final status={final_status} elapsed={time.time()-t0:.1f}s")
        submitted.append({"sid": sid, "status": final_status, "q": q})

    # Phase K3B Wave 1 Step 2.1 — stamp ``provenance="seeded"`` on
    # every submitted session via a direct Mongo update so the
    # observability `organic_vs_seeded_split` reflects this run.
    try:
        from pymongo import MongoClient
        mongo_url = os.environ.get("MONGO_URL")
        db_name = os.environ.get("DB_NAME")
        if mongo_url and db_name and submitted:
            sids = [s["sid"] for s in submitted if s.get("sid")]
            if sids:
                pmc = MongoClient(mongo_url)
                pmc[db_name].search_sessions.update_many(
                    {"id": {"$in": sids}},
                    {"$set": {"provenance": "seeded"}},
                )
                pmc.close()
                print(
                    f"stamped provenance=seeded on {len(sids)} session(s)"
                )
    except Exception as e:  # noqa: BLE001
        # Non-fatal — the seed script's primary job is corpus seeding.
        print(f"provenance stamping skipped: {e!r}")

    n_ready = sum(1 for s in submitted if s["status"] == "awaiting_selection")
    print(f"\n{n_ready}/{len(submitted)} sessions reached awaiting_selection")


if __name__ == "__main__":
    main()
