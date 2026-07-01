"""Phase 5D — MeSH Resolver Service.

Curated, deterministic lookup of MeSH descriptors. Ships a JSON table
(``mesh_descriptors.json``) of manually-verified NLM MeSH descriptors;
the resolver returns the descriptor only when the confidence meets
the caller's threshold. The compiler falls back to Title/Abstract
keywords when no confident MeSH match is found.

HARD RULE: never hallucinate MeSH. If a concept is not in the table or
falls below the threshold, return None — the compiler MUST use
Title/Abstract keywords for that concept.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("clinevidence.mesh")

_HERE = Path(__file__).resolve().parent
_TABLE_PATH = _HERE / "mesh_descriptors.json"


@dataclass(frozen=True)
class MeshResolution:
    concept: str
    mesh_descriptor: str
    confidence: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "mesh_descriptor": self.mesh_descriptor,
            "confidence": self.confidence,
            "source": self.source,
        }


def _default_threshold() -> float:
    try:
        return max(0.0, min(1.0, float(
            os.environ.get("MESH_RESOLVER_CONFIDENCE_THRESHOLD", "0.85"),
        )))
    except (TypeError, ValueError):
        return 0.85


class MeshResolverService:
    """Looks up MeSH descriptors from the curated table."""

    def __init__(self, table_path: Path = _TABLE_PATH):
        self._by_concept: dict[str, MeshResolution] = {}
        self._keyword_only: set[str] = set()
        self._version: str = ""
        self._load(table_path)

    def _load(self, table_path: Path) -> None:
        if not table_path.exists():
            logger.warning("mesh table missing at %s", table_path)
            return
        try:
            with open(table_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:  # noqa: BLE001
            logger.exception("failed to load mesh table %s", table_path)
            return
        self._version = data.get("version") or ""
        for entry in (data.get("descriptors") or []):
            c = (entry.get("concept") or "").strip().lower()
            md = (entry.get("mesh_descriptor") or "").strip()
            if not c or not md:
                continue
            self._by_concept[c] = MeshResolution(
                concept=c,
                mesh_descriptor=md,
                confidence=float(entry.get("confidence") or 0.0),
                source=str(entry.get("source") or ""),
            )
        for entry in (data.get("keyword_only_concepts") or []):
            c = (entry.get("concept") or "").strip().lower()
            if c:
                self._keyword_only.add(c)

    def resolve(
        self, concept: str, *,
        confidence_threshold: float | None = None,
    ) -> MeshResolution | None:
        if not concept:
            return None
        threshold = (
            confidence_threshold if confidence_threshold is not None
            else _default_threshold()
        )
        key = concept.strip().lower()
        if key in self._keyword_only:
            return None
        res = self._by_concept.get(key)
        if res is None:
            return None
        if res.confidence < threshold:
            return None
        return res

    def is_keyword_only(self, concept: str) -> bool:
        return (concept or "").strip().lower() in self._keyword_only

    @property
    def version(self) -> str:
        return self._version


mesh_resolver = MeshResolverService()
