"""Phase K3A.3 — Top-up seed harness to push reviewed sample ≥ 22.

Designed to retrieve specific PubMed-indexed
``Expression of Concern[Publication Type]`` notices that the K3A.3
first-pass seeding missed, plus a few extra retraction-rich topics
as filler.

Anchor PMIDs (closed-set; verified to exist in PubMed with the
``Expression of Concern`` pub-type at the time of seeding):

- 41365576 — ovarian cancer immunotherapy EoC (Sem Cancer Biol)
- 40211391 — CAR-T multiple myeloma EoC (Stem Cell Res Ther)
- 36687664 — ivermectin microbiome / COVID EoC (Front Microbiol)
- 35142703 — Kory / Marik ivermectin RCT EoC (Am J Ther)
- 34410284 — ivermectin SARS-CoV-2 meta-analysis EoC (Open Forum ID)

Seeds carry NO PHI. They are pure clinical evidence questions.
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
    # ---- EoC anchors ----
    "What is the evidence for T cell exhaustion and senescence in ovarian cancer immunotherapy?",
    "What is the evidence for CAR-T cell therapy in multiple myeloma?",
    "What is the evidence on ivermectin and the gut microbiome in COVID-19?",
    "What is the evidence for ivermectin meta-analysis in SARS-CoV-2 mortality?",
    "What is the evidence for ivermectin in COVID-19 outcomes from randomized trials?",
    # ---- Retraction-rich top-ups (filler) ----
    "What is the evidence for thiopental in cardiac arrest neurologic recovery?",
    "What is the evidence for high-flow nasal cannula in COVID-19 acute respiratory failure?",
)


if __name__ == "__main__":
    _seed_main()
