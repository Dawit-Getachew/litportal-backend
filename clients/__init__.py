"""Scienthesis platform clients (Identity + LitHub) for LitPortal.

Framework-agnostic httpx + PyJWT clients ported from litbridge-backend's
``src/clients`` and adapted to this backend's ``core.config`` settings. They
are only exercised when the integration feature flags are enabled
(``LITPORTAL_USE_IDENTITY`` / ``LITPORTAL_DUAL_WRITE_LITHUB``).
"""
