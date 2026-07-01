"""Phase K3B Wave 1 Step 1.6 — Seeded retraction-rich corpus for the
WBE soak.

Submits ~8 additional clinical evidence questions that target
well-documented, PubMed-indexed retraction clusters. Topics span:

- Surgisphere / HCQ COVID follow-ons (Mehra 2020 Lancet retraction,
  Mehra 2020 NEJM retraction).
- Rosiglitazone / Avandia cardiovascular safety retractions
  (Nissen et al. follow-ons, Avandia-related observational follow-ons).
- Ivermectin COVID retractions (Elgazzar, Niaee, Bryant meta-analysis
  follow-ons).
- Wakefield / MMR-autism second-order retractions (Brian Deer,
  Lancet retraction notice).
- Surgisphere ivermectin combined with doxycycline follow-on.

All questions are non-PHI clinical evidence prompts. The harness
reuses the K3A.2 base machinery; only the topic list is overridden.

Usage::

    python -m scripts.k3b_wave1_step1_6_seed_corpus \\
        --api-url https://med-evidence-qa.preview.emergentagent.com \\
        --email clinician@test.com \\
        --password ClinicianTest123! \\
        --wait-secs 120
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.k3a2_seed_search_corpus import main as _seed_main  # noqa: E402
import scripts.k3a2_seed_search_corpus as _seed_mod  # noqa: E402


# K3B Wave 1 Step 1.6 seeded questions — 8 retraction-rich clinical
# evidence prompts. No PHI; each topic anchored to a published
# retraction cluster verifiable via PubMed E-utilities.
_seed_mod.SEED_QUESTIONS = (
    # ---- Surgisphere / HCQ COVID — Mehra Lancet retraction (PMID
    # 32450107 / 32511207 cluster) — also drags in NEJM Surgisphere
    # retraction (PMID 32379955).
    "What is the cardiovascular safety of hydroxychloroquine in "
    "hospitalized COVID-19 patients?",
    # ---- Rosiglitazone CV safety — Nissen 2007 / 2010 follow-on
    # retractions and corrections in the rosiglitazone CV-risk
    # literature.
    "What is the myocardial infarction risk associated with "
    "rosiglitazone in type 2 diabetes mellitus?",
    # ---- Ivermectin COVID — Elgazzar 2020 retraction, Niaee 2021
    # retraction, Bryant et al. meta-analysis follow-ons.
    "What is the evidence for ivermectin in non-hospitalized COVID-19 "
    "outpatients?",
    # ---- Wakefield MMR — Lancet 1998 retraction follow-ons,
    # PMID 9500320 retraction-of-publication notice.
    "What is the evidence for measles-mumps-rubella vaccine causation "
    "of autism in pediatric populations?",
    # ---- Surgisphere ivermectin-doxycycline follow-on retractions.
    "What is the evidence for ivermectin combined with doxycycline "
    "in early COVID-19 treatment?",
    # ---- HCQ + azithromycin combination retractions (Gautret /
    # Marseille follow-on).
    "What is the evidence for hydroxychloroquine plus azithromycin "
    "in mild-to-moderate COVID-19?",
    # ---- Ivermectin meta-analysis — Kory / Marik follow-on
    # retractions in IDSA-adjacent literature.
    "What is the evidence for ivermectin meta-analyses in COVID-19 "
    "mortality reduction?",
    # ---- Avandia / thiazolidinedione cardiovascular outcomes —
    # Avandia label-warning era follow-on retractions.
    "What is the evidence for thiazolidinedione therapy on major "
    "adverse cardiovascular events?",
)


if __name__ == "__main__":
    _seed_main()
