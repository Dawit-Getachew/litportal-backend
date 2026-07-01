"""Scaffolded guideline connectors (Phase 2B). No HTTP calls. No fake data.

Each returns a uniform `ConnectorResult` shape so the pipeline + admin UI can
display source coverage and configuration status. When the corresponding env
var is missing → `not_configured` response. Never scrapes websites.
"""
from __future__ import annotations

import os
from typing import Any


def _result(source: str, status: str, message: str) -> dict[str, Any]:
    return {
        "source": source,
        "status": status,
        "results": [],
        "message": message,
    }


async def _search_scaffolded(source: str, env_key: str) -> dict[str, Any]:
    if not os.environ.get(env_key, "").strip():
        return _result(source, "not_configured", "Connector scaffolded; API/access not configured.")
    # No HTTP call yet — explicitly do not invoke external services.
    return _result(source, "not_configured", "Connector scaffolded; live retrieval not implemented in Phase 2B.")


class USPSTFConnector:
    async def search(self, parsed_pico: dict[str, Any]) -> dict[str, Any]:
        return await _search_scaffolded("uspstf", "USPSTF_API_KEY")


class NICEConnector:
    async def search(self, parsed_pico: dict[str, Any]) -> dict[str, Any]:
        return await _search_scaffolded("nice", "NICE_API_KEY")


class WHOConnector:
    async def search(self, parsed_pico: dict[str, Any]) -> dict[str, Any]:
        return await _search_scaffolded("who", "WHO_API_KEY")


class CDCConnector:
    async def search(self, parsed_pico: dict[str, Any]) -> dict[str, Any]:
        return await _search_scaffolded("cdc", "CDC_API_KEY")


class SpecialtySocietyConnector:
    async def search(self, parsed_pico: dict[str, Any]) -> dict[str, Any]:
        return await _search_scaffolded("specialty_society", "SPECIALTY_SOCIETY_API_KEY")


uspstf_connector = USPSTFConnector()
nice_connector = NICEConnector()
who_connector = WHOConnector()
cdc_connector = CDCConnector()
specialty_society_connector = SpecialtySocietyConnector()
