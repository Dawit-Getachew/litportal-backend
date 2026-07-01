"""LitPortal AI agents (2.5-week scope).

Three DISTINCT, advisory-only LLM sidecars, each in its own module:

  1. query_categorizer    — pre-parser enrich/categorize (fixes keyword-gated
                            misclassification before the deterministic parser).
  2. question_interpreter — question understanding / ContextAugment (B3);
                            diagnostics only, never alters retrieval.
  3. evidence_curator     — evidence surfacing + ranker optimization (B4/B4a);
                            surfaces genuinely-relevant excluded papers onto a
                            SEPARATE advisory shelf.

Governance (enforced here): agents NEVER call sources, select/exclude/reorder
deterministic results, generate PMIDs/DOIs/citations, or bypass permission
checks. Every agent is feature-flagged (default OFF), fail-closed (any failure
is a no-op), and operates only on PHI-screened input.
"""
