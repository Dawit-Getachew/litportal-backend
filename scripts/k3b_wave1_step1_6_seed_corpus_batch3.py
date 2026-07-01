"""Phase K3B Wave 1 Step 1.6 — Batch 3 final top-up seed.

Targeted queries to push reviewed sample to ≥ 20. Each query uses
specific terminology from known retraction-rich clusters indexed
in the local ``articles`` collection.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.k3a2_seed_search_corpus import main as _seed_main  # noqa: E402
import scripts.k3a2_seed_search_corpus as _seed_mod  # noqa: E402


_seed_mod.SEED_QUESTIONS = (
    # ---- Cinnamaldehyde / cerebral vasospasm (PMID 31444631)
    "What is the neuroprotective evidence for cinnamaldehyde in "
    "subarachnoid hemorrhage and cerebral vasospasm?",
    # ---- Inflammatory skin disease / breast cancer Mendelian
    # randomization (PMID 38937884).
    "What is the Mendelian randomization evidence linking "
    "inflammatory skin diseases to breast cancer risk?",
    # ---- Compassionate-use HCQ deaths (PMID 38171239).
    "What is the mortality evidence for compassionate-use "
    "hydroxychloroquine in early COVID-19 hospitalizations?",
    # ---- Asymptomatic ivermectin RCT (PMID 34073401).
    "What is the evidence for single-dose ivermectin in "
    "asymptomatic SARS-CoV-2 infection?",
    # ---- Multidrug ivermectin in hypoxic outpatients
    # (PMID 35135310).
    "What is the evidence for ivermectin-based multidrug therapy in "
    "hypoxic ambulatory COVID-19?",
    # ---- Bioengineered tissue viability (PMID 23473965).
    "What is the evidence for verifying cell viability in "
    "bioengineered tissues prior to clinical transplantation?",
)


if __name__ == "__main__":
    _seed_main()
