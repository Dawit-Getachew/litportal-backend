"""Phase K3B Wave 1 Step 1.6 — Batch 4 final top-up to clear ≥ 20.

Targets retracted clusters that the prior batches did not surface
in their top-K. Queries here are constructed to look like real
clinical evidence questions while being phrased close to the
retracted papers' title language to maximize the chance the
PubMed retrieval surfaces them.
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
    # Surgisphere / NEJM HCQ outpatient retraction wave.
    "What is the in-hospital mortality of "
    "hydroxychloroquine treatment in patients with COVID-19?",
    # Mehra Lancet retraction wave — ACE inhibitors / ARBs and
    # COVID-19 outcomes.
    "What is the evidence that angiotensin-converting enzyme "
    "inhibitor therapy affects mortality in hospitalized "
    "COVID-19 patients?",
    # Niaee 2021 ivermectin (retracted) — moderate/severe inpatient
    # ivermectin RCT.
    "What is the evidence for ivermectin in moderate to severe "
    "hospitalized COVID-19?",
    # Elgazzar 2020 ivermectin (retracted) — prophylaxis and
    # treatment.
    "What is the evidence for ivermectin prophylaxis against "
    "COVID-19 in healthcare workers?",
    # Mehra Lancet retraction wave — outcomes in cardiac
    # arrhythmia.
    "What is the evidence that hydroxychloroquine causes "
    "ventricular arrhythmia in COVID-19 inpatients?",
    # Mehra NEJM Surgisphere retraction wave —
    # cardiovascular disease in COVID-19 outpatients.
    "What is the evidence for cardiovascular comorbidities and "
    "in-hospital mortality in COVID-19?",
    # Rosiglitazone (Avandia) MI / death meta-analysis wave.
    "What is the evidence for rosiglitazone increasing risk of "
    "death from cardiovascular causes in type 2 diabetes?",
)


if __name__ == "__main__":
    _seed_main()
