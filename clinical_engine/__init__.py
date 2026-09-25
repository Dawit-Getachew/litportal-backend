"""LitPortal clinical evidence engine — Week-1 foundation.

Isolated package: nothing outside ``clinical_engine`` imports it except the
router registration in ``server.py``. The AWS worker images never copy it.
Enabled only with ``CLINICAL_ENGINE_ENABLED=true`` (default off).
"""
ENGINE_VERSION = "engine.w1"
