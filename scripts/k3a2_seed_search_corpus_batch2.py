"""Phase K3A.2 — Second seed batch (top-up to reach ≥20 candidates).

Run only if the first seed batch produces fewer than 20 combined
``retracted`` + ``expression_of_concern`` candidates. Targets
additional retraction-rich topics from independent biomedical
clusters (Marik vitamin-C sepsis, Anversa-adjacent stem cell,
Avandia/rosiglitazone, COVID convalescent plasma EoCs, Wakefield
follow-ups, etc.) so the soak corpus stays diverse.

Same PHI / no-LLM / no-scrape posture as the first batch.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.k3a2_seed_search_corpus import (  # noqa: E402
    main as _seed_main, SEED_QUESTIONS as _BASE,
)
# Monkey-patch in a second wave of topics, then call the existing
# main() so the harness is reused.
import scripts.k3a2_seed_search_corpus as _seed_mod  # noqa: E402

_seed_mod.SEED_QUESTIONS = (
    # Marik vitamin C / hydrocortisone / thiamine sepsis cluster.
    "What is the evidence for high-dose vitamin C in sepsis treatment?",
    # Famotidine in COVID-19 — Janowitz cluster.
    "What is the evidence for famotidine in COVID-19?",
    # Avandia / rosiglitazone retraction cluster.
    "What is the evidence for rosiglitazone in type 2 diabetes?",
    # Tocilizumab COVID — some EoCs.
    "What is the evidence for tocilizumab in severe COVID-19?",
    # COVID convalescent plasma — additional EoCs.
    "What is the safety and efficacy of convalescent plasma in COVID-19 pneumonia?",
    # Cardiac stem-cell follow-on cluster.
    "What is the evidence for bone marrow stem cell transplantation in myocardial infarction?",
    # COVID lopinavir/ritonavir cluster.
    "What is the evidence for lopinavir-ritonavir in COVID-19 treatment?",
    # COVID umifenovir cluster.
    "What is the evidence for umifenovir (arbidol) in COVID-19?",
)


if __name__ == "__main__":
    _seed_main()
