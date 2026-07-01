"""Phase 5B — specialty pack loaders (cardiology/transfusion, etc.).

Each pack is a JSON file in this directory. Packs are loaded on import and
exposed through ``get_pack(name)`` and ``all_packs()`` helpers. Keep them
auditable JSON (no executable code in packs).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent

_PACK_CACHE: dict[str, dict[str, Any]] = {}


def _load() -> None:
    if _PACK_CACHE:
        return
    for path in sorted(_HERE.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            name = data.get("name") or path.stem
            _PACK_CACHE[name] = data
        except Exception:  # noqa: BLE001 — bad pack must not crash startup
            continue


def get_pack(name: str) -> dict[str, Any] | None:
    _load()
    return _PACK_CACHE.get(name)


def all_packs() -> dict[str, dict[str, Any]]:
    _load()
    return dict(_PACK_CACHE)
