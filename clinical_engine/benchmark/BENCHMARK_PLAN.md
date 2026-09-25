# W1.5 — Benchmark freeze and acceptance plan

**Status: DRAFT, frozen for development** (`benchmark.v1-draft`, 29 cases, SHA-256 pinned in
`loader.py`). Nothing here gates a release until the physician owner signs off the decisions in §6.
Any change to `cases_v1.json` must bump the version and the pinned hash together
(`tests/test_clinical_engine_benchmark.py` fails otherwise) — no silent edits during tuning.

## 1. What is frozen

| | Count | Notes |
|---|---|---|
| Cases | 29 | 13 development (10 families), 16 held-out candidates (15 families) |
| Critical care / neurocritical care | 19 | Scope: critical-care-led initial validation |
| Question types | treatment 12, harm 4, prognosis 3, prevention 3, diagnosis 3, background 2, screening 1, other 1 | Engine labels are provisional (not physician-adjudicated) |
| References | every stored PMID verified against PubMed esummary (2026-09-25) | identity only, not clinical adjudication |

Sources: the client's 9.10.26 report (verbatim steroid question), repository cases
(`services/search_benchmark*.py`, `r1a_gold_comparator_cases.py`, `scripts/phase6g…`),
historical session records, and proposals marked "needs physician adjudication".

## 2. Split discipline

- A **family** is condition × intervention/test/exposure. A family is either all development or all
  held-out (enforced by `discipline_problems`).
- Anything with legacy case-specific code or tuning history is development (steroids/sepsis,
  milrinone/SAH, AMI transfusion, CVC pneumothorax, GLP-1, varicella, colon-cancer prognosis).
- **Held-out references are not stored in any repository.** The proposed candidates are in
  `DrSeife/week1-deliverables/W1.5_HELDOUT_REFERENCES_PRIVATE.json` for the physician; because the
  developer has seen them, the physician should re-curate and keep the final lists private.
- The developer never labels held-out cases.

## 3. Labels

- `must_find` — recall-gate references only (currently the ones the repo already designates).
- Graded relevance for everything else: direct, broad-answering, supportive, indirect, uncertain,
  excluded (W2.4 states).
- Roles recorded per reference: main trial, separate report (subgroup / post hoc / follow-up — must not
  merge with the main trial), current / superseded / focused-update guideline, wrong-destination
  identity, hard negative.
- **Scoring is at the work level**: one guideline co-published in two journals counts once; a main
  trial and its subgroup report count separately (matches "8 works / 9 records").
- Evidence labels: `historical` (past UAT/audit observations), `fixture` (repository-defined cases or
  synthetic corpora), `live` (only runs recorded with commit, flags, date and source snapshot).
  Old "live" benchmark runs in the legacy harness are relabelled `fixture` (the harness used fixtures).

## 4. Metrics and thresholds (scope §7 — owner to confirm before tuning)

| Metric | Target | Definition |
|---|---|---|
| Must-find recall@15 | ≥ 90 % | Pooled over held-out must-find works in the first 15 results the clinician sees; report per case too. **Decide pooled vs per-case now** (with 8 must-find works, 7/8 = 87.5 % fails a per-case gate). |
| Wrong-condition rate | < 1 % | Share of top-15 results about a different condition. |
| nDCG@5 / @10, Precision@5 / @10, Success@5 | report; P@5 ≥ 80 % proposed | On answerable held-out cases. |
| Currency | report | Current-guideline and update capture; redundant evidence. |
| Interpretation (engine-specific, Week 1) | report | Question-type agreement; silent age inference must be 0. |

Report raw counts, per-case and per-type breakdowns, and exclusions; never change denominators after
results are seen.

## 5. Versioned baseline (2026-09-25, `feat/week1-clinical-engine` @ `b3b1825`)

Interpretation baseline over all 29 cases (`python -m clinical_engine.benchmark.run_interpretation_baseline [--model]`):

| Mode | Type agreement (provisional labels) | Silent age inferences | Tokens (29 cases) | p50 latency | Label |
|---|---|---|---|---|---|
| Rules only | 18 / 29 | 0 | 0 | — | fixture |
| Model (gpt-4o-mini), run 1 | 22 / 29 | 0 | 21,773 | 1.65 s | live |
| Model (gpt-4o-mini), run 2 | 22 / 29 | 0 | 21,782 | 1.41 s | live |

The only case that blocks confirmation is W15-13 "What are the side effects?" (underspecified by
design — the engine asks for the missing exposure instead of searching). Most remaining type
disagreements are defensible readings (e.g. qSOFA mortality prediction read as diagnostic accuracy,
dosing questions read as management) — adjudication will settle the labels. Result files:
`DrSeife/week1-deliverables/baseline_interpretation_*.json`.

Retrieval baselines (legacy pipeline vs Best Match vs new lanes) are W4.1 work; they need the
must-find lists from §6.

## 6. Owner decisions required before tuning

1. The client's eight-work / nine-record steroid reference list (identities are not in the workspace).
   Hypotheses to confirm: main trials ADRENAL (PMID 29347874) and APROCCHSS (29490185); SSC 2026
   guideline co-published as 41869847 / 41870560 (one work, two records).
2. The second steroid question (candidate: W15-02, ARDS timing and dose).
3. Family definition (condition × intervention, as frozen, or condition-only — which shrinks held-out
   from 16 to about 9 cases).
4. Held-out minimum (proposed ≥ 20 families: add ≥ 5 private physician families) and the P@5 target.
5. Reviewer plan: physician adjudicator + second reviewer double-labelling held-out cases.
6. Source scope: PubMed + Europe PMC + vetted guideline sources; mark out-of-scope works at freeze time.

## 7. Path to 50–100 cases (W4.1)

Add ~20 held-out families from the physician's private set (recent-evidence slice 2024–26: SSC 2026,
REVISE, 2024 steroid focused update), expand development families with non-treatment questions
(diagnosis, prognosis, harm), and add identity fixtures already found while verifying: adjacent PMIDs
that are different papers (26903335–38; 37952131 vs 37952133; 29485925 vs 29485926), guideline
co-publications, and main-trial vs subgroup pairs (ADRENAL post hoc 31651531, APROCCHSS CAP subgroup
38310918, REALITY HF subgroup 38408702, MINT 6-month letter 39221566).

## 8. Legacy benchmark defects (why old numbers are not a baseline)

Persisted recall/MRR is always 0.0 (result key `case_id` vs `id`); only 7 legacy cases ever run (module
list rebound after the service captured it); `execution_mode="live"` still uses fixtures; placeholder
"PMIDs" (e.g. 91001, 55001, 71001) are real unrelated 1975–79 papers; fixture titles/abstracts are
synthetic; `tests/test_phase5f_pipeline_wiring.py:112` labels PMID 30872119 as an ARDS steroid review,
but it is an insect-physiology paper.
