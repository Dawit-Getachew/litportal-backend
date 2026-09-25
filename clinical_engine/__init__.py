"""LitPortal clinical evidence engine — Week-1 foundation.

Isolated package: nothing outside ``clinical_engine`` imports it except the
router registration in ``server.py``. The AWS worker images never copy it.
On by default; ``CLINICAL_ENGINE_ENABLED=false`` switches it off.
"""
ENGINE_VERSION = "engine.w1"
