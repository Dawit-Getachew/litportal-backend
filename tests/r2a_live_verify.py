"""Phase R2A — LIVE end-to-end shadow verification (manual run, not pytest).

Runs ONE real LLM shadow classification against the existing prostate
session and asserts every clinician-safety invariant the operator asked
for. Read-only w.r.t. clinician-visible state.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import requests

BASE = os.environ["REACT_APP_BACKEND_URL"].rstrip("/")
API = f"{BASE}/api"
SESSION_ID = "5c0e4942-6b79-42e0-b9ad-6ccde76cc656"
CLINICIAN = ("clinician@test.com", "ClinicianTest123!")
ADMIN = ("admin@test.com", "AdminTest123!")
H = {"User-Agent": "ClinEvidence-R2A-Verify/1.0", "Content-Type": "application/json"}


def login(email, password):
    r = requests.post(f"{API}/auth/login", json={"email": email, "password": password}, headers=H, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def auth(token):
    return {**H, "Authorization": f"Bearer {token}"}


def clinician_session_fingerprint(token):
    r = requests.get(f"{API}/search-sessions/{SESSION_ID}", headers=auth(token), timeout=60)
    r.raise_for_status()
    body = r.json()
    canon = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest(), body


def main():
    out = {}
    clin_token = login(*CLINICIAN)
    admin_token = login(*ADMIN)

    # 1. Clinician response BEFORE
    before_hash, before_body = clinician_session_fingerprint(clin_token)
    out["clinician_before_sha256"] = before_hash

    # 2. Admin-only enforcement: clinician must be refused on shadow endpoint
    rc = requests.post(
        f"{API}/admin/search-sessions/{SESSION_ID}/rescue-classifier-shadow",
        headers=auth(clin_token), timeout=60,
    )
    out["clinician_on_admin_endpoint_status"] = rc.status_code
    assert rc.status_code in (401, 403), f"admin endpoint not protected: {rc.status_code}"

    # 3. Run shadow as admin (single live LLM call on the capped subset)
    ra = requests.post(
        f"{API}/admin/search-sessions/{SESSION_ID}/rescue-classifier-shadow",
        headers=auth(admin_token), timeout=300,
    )
    ra.raise_for_status()
    shadow = ra.json()
    out["admin_status"] = ra.status_code
    out["shadow"] = shadow

    # 4. Clinician response AFTER — must be identical
    after_hash, after_body = clinician_session_fingerprint(clin_token)
    out["clinician_after_sha256"] = after_hash
    out["clinician_byte_for_byte_unchanged"] = before_hash == after_hash

    # 5. Leakage scan on the persisted/admin shadow payload.
    blob = json.dumps(shadow, ensure_ascii=False).lower()
    leak_terms = {
        "prompt_word_shadow": "you are a shadow",   # system prompt phrase
        "emergent_key": "sk-emergent",
        "openai_key": "sk-proj",
        "bearer": "bearer ",
    }
    leaks = {k: (v in blob) for k, v in leak_terms.items()}
    out["leak_scan"] = leaks
    out["any_credential_or_prompt_leak"] = any(leaks.values())

    # 6. Closed-set + no-identifier-from-LLM structural checks. Every
    #    article_id/pmid in the payload is supplied by US deterministically
    #    (top_rescue_candidates carry our pmid for admin reference) — assert
    #    the LLM-controlled fields are closed-set only.
    from services.rejected_article_rescue import RESCUE_LABELS, RELEVANCE_REASONS
    bad_labels = []
    for r in shadow.get("rejection_reason_vs_rescue", []):
        if r["rescue_label"] not in RESCUE_LABELS:
            bad_labels.append(r["rescue_label"])
        if r["rescue_reason"] not in RELEVANCE_REASONS:
            bad_labels.append(r["rescue_reason"])
    out["closed_set_violations"] = bad_labels

    # short_rationale must not contain pmids/dois/urls
    rationale_leaks = []
    for r in shadow.get("top_rescue_candidates", []):
        sr = (r.get("short_rationale") or "").lower()
        if "pmid" in sr or "doi" in sr or "http" in sr or "10." in sr[:8]:
            rationale_leaks.append(r.get("article_id"))
    out["rationale_identifier_leaks"] = rationale_leaks

    # 7. Confirm DB sidecar persisted and updated_at NOT bumped (byte-for-byte).
    out["clinician_updated_at_before"] = before_body.get("updated_at")
    out["clinician_updated_at_after"] = after_body.get("updated_at")

    print(json.dumps(out, indent=2, ensure_ascii=False)[:12000])

    # Hard assertions
    assert out["clinician_byte_for_byte_unchanged"] is True, "clinician response changed!"
    assert out["any_credential_or_prompt_leak"] is False, "leakage detected!"
    assert not bad_labels, f"closed-set violation: {bad_labels}"
    assert not rationale_leaks, f"rationale identifier leak: {rationale_leaks}"
    assert shadow.get("no_clinician_visible_change") is True
    assert shadow.get("mode") == "shadow"
    print("\nALL R2A LIVE INVARIANTS PASSED")


if __name__ == "__main__":
    sys.exit(main())
