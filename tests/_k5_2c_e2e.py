"""K5.2C-0 preview e2e — real search → confirm Cochrane lane on article rows."""
import sys
import time
import json
import httpx

sys.path.insert(0, "/app/backend")
BASE = "http://localhost:8001/api"
EMAIL, PW = "clinician@test.com", "ClinicianTest123!"
Q = ("What is the most effective anticoagulation strategy for stroke "
     "prevention in patients with atrial fibrillation?")


def main():
    with httpx.Client(timeout=60) as cli:
        tok = cli.post(f"{BASE}/auth/login",
                       json={"email": EMAIL, "password": PW}).json()
        token = tok.get("access_token") or tok.get("token")
        assert token, f"login failed: {tok}"
        h = {"Authorization": f"Bearer {token}"}
        r = cli.post(f"{BASE}/search-sessions/", json={"question": Q}, headers=h).json()
        sid = r.get("session_id") or r.get("id")
        print("session:", sid, "status:", r.get("status"))
        assert sid

        status = r.get("status")
        deadline = time.time() + 150
        doc = {}
        while time.time() < deadline:
            doc = cli.get(f"{BASE}/search-sessions/{sid}", headers=h).json()
            status = doc.get("status")
            if status in ("awaiting_selection", "complete", "ready", "error",
                          "awaiting_clinical_subject"):
                break
            time.sleep(5)
        print("final status:", status)

        arts = doc.get("articles") or []
        print("included articles:", len(arts))
        coch = [a for a in arts if a.get("cochrane_review_lane")]
        print("cochrane-labeled (included):", len(coch))
        # Also scan excluded for labeling coverage.
        exc = doc.get("excluded_articles") or []
        coch_exc = [a for a in exc if a.get("cochrane_review_lane")]
        print("cochrane-labeled (excluded):", len(coch_exc))

        for a in (coch[:4] or coch_exc[:4]):
            art = a.get("article") or {}
            print("  -", json.dumps({
                "title": (art.get("title") or "")[:60],
                "journal": art.get("journal"),
                "cochrane_review_lane": a.get("cochrane_review_lane"),
                "cochrane_review_provenance": a.get("cochrane_review_provenance"),
                "cochrane_review_shadow_boost": a.get("cochrane_review_shadow_boost"),
                "cap_reasons": a.get("cap_reasons"),
                "overall_rank": a.get("overall_rank"),
                "boost_reasons": [b for b in (a.get("ranking_boost_reasons") or [])
                                  if "cochrane" in b],
            }))
        total_coch = len(coch) + len(coch_exc)
        if total_coch:
            print("E2E RESULT: PASS — Cochrane reviews detected + labeled on rows")
        else:
            print("E2E RESULT: NO_COCHRANE_RETRIEVED (status=%s) — see note" % status)


if __name__ == "__main__":
    main()
