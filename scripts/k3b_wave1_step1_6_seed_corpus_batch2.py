"""Phase K3B Wave 1 Step 1.6 — Batch 2 seed (top-up to push reviewed
sample ≥ 20).

These queries target retraction clusters already indexed in the
local ``articles`` collection (verified post-Batch-1 inventory).
No PHI, no patient identifiers. All public PubMed-indexed
retraction notices.
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
    # ---- Vitamin C in sepsis (PMID 33094466 retracted).
    "What is the evidence for high-dose vitamin C in sepsis "
    "management?",
    # ---- DPP-4 vs sulfonylurea CV outcomes (PMID 40323973
    # retracted).
    "What is the cardiovascular outcome of DPP-4 inhibitors versus "
    "sulfonylureas in type 2 diabetes?",
    # ---- Stem cell therapy diabetic foot ulcer
    # (PMID 38156706 retracted).
    "What is the efficacy of stem cell therapy for diabetic foot "
    "ulcers?",
    # ---- Core decompression for femoral head osteonecrosis
    # (PMID 34912470 retracted).
    "What is the evidence for core decompression in osteonecrosis "
    "of the femoral head?",
    # ---- Celecoxib in multimodal post-operative analgesia
    # (PMID 17578978 retracted).
    "What is the analgesic efficacy of celecoxib as part of "
    "multimodal post-operative pain management?",
    # ---- Favipiravir vs HCQ in COVID-19 (PMID 33790308 retracted).
    "What is the comparative efficacy of favipiravir versus "
    "hydroxychloroquine in COVID-19?",
    # ---- Cardiac findings in a randomized placebo trial
    # (PMID 39702045 retracted, distinct from HCQ cluster).
    "What is the cardiac safety of combination chemotherapy "
    "regimens in oncology trials?",
)


if __name__ == "__main__":
    _seed_main()
