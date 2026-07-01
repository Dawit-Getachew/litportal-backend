"""Phase 6G — fresh-query validation harness.

Submits the 7 spec questions to the running dev backend, polls for
completion, and prints the picott_term_mapping_json block + populated
roles + completeness + source_api_call_count + llm_calls_count per
the Task 7 deliverable.

Designed to run against the local pod (REACT_APP_BACKEND_URL), NOT
the production deployment.
"""
import asyncio
import os
import sys
import time

import httpx

EMAIL = os.environ.get("PHASE6G_EMAIL", "clinician@test.com")
PASSWORD = os.environ.get("PHASE6G_PASSWORD", "ClinicianTest123!")

QUERIES = [
    ("Q1", "What are the side effects of Seroquel?"),
    ("Q2", "What is the first line treatment and second line treatment for psoriasis?"),
    ("Q3", "how do glp1s impact ldl cholesterole?"),
    ("Q4", "What is the right timing and dose to administer steroids for patients with ARDS?"),
    ("Q5", "What is the recommended age to get varicella vaccination? What is its benefit?"),
    ("Q6", "What are the side effects?"),
    ("Q7", "What is first-line treatment?"),
]


def _base_url() -> str:
    # The pod's frontend .env points at the externally-exposed URL.
    env_path = "/app/frontend/.env"
    with open(env_path) as f:
        for line in f:
            if line.startswith("REACT_APP_BACKEND_URL="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("REACT_APP_BACKEND_URL not found in /app/frontend/.env")


async def _login(client: httpx.AsyncClient, base: str) -> str:
    r = await client.post(
        f"{base}/api/auth/login",
        json={"email": EMAIL, "password": PASSWORD},
        timeout=15.0,
    )
    r.raise_for_status()
    return r.json()["access_token"]


async def _submit_and_poll(
    client: httpx.AsyncClient, base: str, token: str, question: str,
) -> dict:
    h = {"Authorization": f"Bearer {token}"}
    r = await client.post(
        f"{base}/api/search-sessions/",
        json={"question": question},
        headers=h,
        timeout=20.0,
    )
    r.raise_for_status()
    body = r.json()
    sid = body.get("id") or body.get("session_id")
    status = body.get("status")
    # Generic-only / blocked branches are terminal — return immediately.
    if status in {"awaiting_clinical_subject", "blocked_phi"}:
        time.sleep(1)
        rr = await client.get(
            f"{base}/api/search-sessions/{sid}",
            headers=h, timeout=20.0,
        )
        rr.raise_for_status()
        return rr.json()
    # Otherwise poll for ``awaiting_selection``.
    for _ in range(60):
        await asyncio.sleep(2)
        rr = await client.get(
            f"{base}/api/search-sessions/{sid}",
            headers=h, timeout=20.0,
        )
        rr.raise_for_status()
        j = rr.json()
        if j.get("status") in {
            "awaiting_selection", "error", "awaiting_clinical_subject",
            "blocked_phi", "complete",
        }:
            return j
    raise RuntimeError(f"timeout waiting for session {sid}")


def _summarize(label: str, question: str, sess: dict) -> str:
    sid = sess.get("id")
    status = sess.get("status")
    m = sess.get("picott_term_mapping_json") or {}
    comp = (m.get("completeness") or {})
    populated_roles = [
        r for r, v in (m.get("fields") or {}).items() if v
    ]
    primary_anchors = m.get("primary_clinical_anchors") or []
    sdg = sess.get("session_diagnostics_json") or {}
    src_count = sdg.get("source_api_call_count")
    llm_count = sdg.get("llm_calls_count")
    return (
        f"=== {label}: {question}\n"
        f"  session_id: {sid}\n"
        f"  status: {status}\n"
        f"  completeness: safe_to_search={comp.get('safe_to_search')} "
        f"clarification_needed={comp.get('clarification_needed')} "
        f"missing_required_roles={comp.get('missing_required_roles')}\n"
        f"  populated_roles: {populated_roles}\n"
        f"  primary_anchors_count: {len(primary_anchors)} "
        f"texts={[a.get('canonical_concept') or a.get('text') for a in primary_anchors]}\n"
        f"  source_api_call_count: {src_count}\n"
        f"  llm_calls_count: {llm_count}\n"
    )


async def main() -> int:
    base = _base_url()
    print(f"[base url]  {base}")
    async with httpx.AsyncClient() as client:
        token = await _login(client, base)
        for label, q in QUERIES:
            try:
                sess = await _submit_and_poll(client, base, token, q)
                print(_summarize(label, q, sess))
            except Exception as e:  # noqa: BLE001
                print(f"=== {label} FAILED: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
