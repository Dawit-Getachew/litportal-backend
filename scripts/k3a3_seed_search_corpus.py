"""Phase K3A.3 — Extended seed harness (top-up to reach ≥20 candidates).

Builds on the K3A.2 seed harness but expands the topic surface to:

- Additional COVID retraction/EoC clusters (Mehra Lancet HCQ EoC,
  ivermectin EoCs, lopinavir-ritonavir, dexamethasone derivatives).
- Wakefield 2010/2011 follow-on retractions (Lancet retraction
  notices, Brian Deer-related).
- Rosiglitazone / Avandia cardiovascular safety retraction cluster.
- EoC-targeted seed questions designed to retrieve the EoC notice
  records themselves (e.g. "Expression of concern on hydroxychloroquine
  treatment for COVID-19").
- Surgical and obstetric retraction clusters (Macchiarini,
  vibroacoustic stimulation, episiotomy controversies).

All seed questions are non-PHI clinical evidence prompts. The
harness reuses the K3A.2 base machinery; only the topic list is
overridden. Same User-Agent header for preview ingress.
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
    # ---- EoC-targeted (specifically the EoC notice records) ----
    "Expression of concern on hydroxychloroquine treatment for COVID-19",
    "Withdrawal of recommendations on chloroquine in COVID-19 treatment",
    "Concerns regarding ivermectin clinical trials in COVID-19",
    "Retraction notices regarding remdesivir in COVID-19 treatment",
    # ---- COVID retraction follow-on clusters ----
    "What is the evidence for chloroquine in COVID-19 treatment?",
    "What is the evidence for azithromycin combined with hydroxychloroquine in COVID-19?",
    "What is the evidence for dexamethasone in COVID-19 acute respiratory distress?",
    # ---- Wakefield / vaccine-autism follow-on retractions ----
    "What is the evidence on autism risk after measles vaccination?",
    "What is the evidence linking childhood vaccines and autism spectrum disorder?",
    # ---- Rosiglitazone / Avandia ----
    "What is the cardiovascular safety profile of rosiglitazone in type 2 diabetes?",
    "What is the evidence for thiazolidinediones in glycemic control?",
    # ---- Cardiac stem-cell + Anversa ----
    "What is the evidence for c-kit positive cardiac stem cells in heart regeneration?",
    "What is the evidence for autologous mesenchymal stem cells in heart failure?",
    # ---- Macchiarini / synthetic trachea ----
    "What are the outcomes of bioengineered trachea grafts in patients?",
    # ---- Surgical / obstetric retractions ----
    "What is the evidence for episiotomy in routine vaginal delivery?",
    "What is the evidence for vibroacoustic stimulation in fetal monitoring?",
    # ---- Reuben analgesia cluster top-up ----
    "What is the evidence for pregabalin in postoperative pain management?",
    # ---- Vitamin D mega-dose claims ----
    "What is the evidence for high-dose vitamin D in COVID-19 outcomes?",
    "What is the evidence for vitamin D supplementation in upper respiratory infection prevention?",
    # ---- Surgisphere follow-on ----
    "What is the evidence for ivermectin combined with doxycycline in COVID-19?",
)


if __name__ == "__main__":
    _seed_main()
