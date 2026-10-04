"""Exact MeSH heading lookup (NCBI E-utilities, db=mesh).

A concept maps to a descriptor only when one of that descriptor's entry terms
*is* the concept (after spelling / plural normalisation). There is no "closest
heading" fallback: that is how "whole blood transfusion" would silently turn
into the much broader "Blood Transfusion" (which explodes to red cell, platelet
and plasma transfusion). No match simply means the concept is searched as words.

Vocabulary lookups retrieve no evidence; they run after the clinician approved
the interpretation, to show the mapping for review.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger("litportal.clinical_engine.mesh")

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_CACHE_MAX = 4000
_CACHE_TTL_S = 24 * 3600
_REQUEST_TIMEOUT_S = 6.0


@dataclass(frozen=True)
class MeshRecord:
    ui: str
    descriptor: str
    entry_terms: tuple[str, ...]
    scope_note: Optional[str] = None


@dataclass
class LookupResult:
    records: dict[str, Optional[MeshRecord]] = field(default_factory=dict)
    status: str = "ncbi"  # ncbi | unavailable


# ── Normalisation ───────────────────────────────────────────────────────────

_UK_US = (
    ("haem", "hem"), ("anaem", "anem"), ("oesoph", "esoph"), ("oedem", "edem"), ("paediatr", "pediatr"),
    ("tumour", "tumor"), ("caesarean", "cesarean"), ("leukaem", "leukem"), ("diarrhoea", "diarrhea"),
    ("oestrogen", "estrogen"), ("foetal", "fetal"), ("orthopaed", "orthoped"), ("anaesth", "anesth"),
    ("ischaem", "ischem"), ("haemorrh", "hemorrh"),
)
_GENERIC_HEADS = re.compile(
    r"\s+(?:patients?|people|persons?|individuals?|subjects?|population|populations|adults?|cases)$", re.I,
)
# The act of giving something is not part of its heading: "steroid administration"
# is indexed as Steroids (with /administration & dosage). Tried last, so a heading
# for the whole phrase ("Fluid Therapy") always wins.
_ACTION_TAIL = re.compile(r"\s+(?:administration|administering|therapy|therapies|treatment|treatments)$", re.I)
# A clause about the patients is not part of the heading either ("septic shock who continue to
# require vasopressors…" is indexed as Shock, Septic). Also tried last.
_PATIENT_CLAUSE = re.compile(r"\s+(?:who|whose|in whom|requiring|receiving|undergoing|treated with|despite)\b.*$",
                             re.I)


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def _us(text: str) -> str:
    out = text.lower()
    for uk, us in _UK_US:
        out = out.replace(uk, us)
    return out


def _singular(text: str) -> str:
    words = text.split()
    if words and len(words[-1]) > 4 and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def variants(term: str) -> list[str]:
    """Lookup forms, most faithful first. Dropping a trailing generic noun
    ("trauma patients" → "trauma") names the same concept, not a broader one."""
    base = " ".join((term or "").split()).strip(" .,;:")
    out: list[str] = []
    for v in (base, _us(base), _singular(_us(base)), _GENERIC_HEADS.sub("", _us(base)),
              _ACTION_TAIL.sub("", _us(base)), _PATIENT_CLAUSE.sub("", _us(base))):
        v = v.strip()
        if v and len(v) >= 3 and v.lower() not in {o.lower() for o in out}:
            out.append(v)
    return out


def entry_matches(term_variant: str, record: MeshRecord) -> bool:
    target = {norm(term_variant), norm(_singular(term_variant))}
    for entry in record.entry_terms:
        n = norm(entry)
        if n in target or norm(_singular(entry)) in target:
            return True
    return False


# ── Rate limit + cache (process-wide) ───────────────────────────────────────

_cache: "OrderedDict[str, tuple[float, Optional[MeshRecord]]]" = OrderedDict()
_limiters: dict[int, asyncio.Lock] = {}
_next_slot = 0.0


def _limiter() -> asyncio.Lock:
    # One lock per event loop (a lock is bound to the loop that first uses it).
    loop_id = id(asyncio.get_running_loop())
    lock = _limiters.get(loop_id)
    if lock is None:
        _limiters.clear()
        lock = _limiters[loop_id] = asyncio.Lock()
    return lock


def _cache_get(key: str) -> tuple[bool, Optional[MeshRecord]]:
    hit = _cache.get(key)
    if hit is None:
        return False, None
    stored_at, record = hit
    if time.monotonic() - stored_at > _CACHE_TTL_S:
        _cache.pop(key, None)
        return False, None
    _cache.move_to_end(key)
    return True, record


def _cache_put(key: str, record: Optional[MeshRecord]) -> None:
    _cache[key] = (time.monotonic(), record)
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


def clear_cache() -> None:
    _cache.clear()


def _params(**extra: str) -> dict[str, str]:
    p = {"db": "mesh", "retmode": "json", "tool": os.environ.get("NCBI_TOOL", "LitPortal") or "LitPortal"}
    if os.environ.get("NCBI_EMAIL", "").strip():
        p["email"] = os.environ["NCBI_EMAIL"].strip()
    if os.environ.get("NCBI_API_KEY", "").strip():
        p["api_key"] = os.environ["NCBI_API_KEY"].strip()
    p.update(extra)
    return p


async def _get(client: httpx.AsyncClient, url: str, params: dict[str, str]) -> dict:
    """NCBI allows 3 requests/s without a key, 10 with one; space requests accordingly."""
    global _next_slot
    interval = 0.12 if "api_key" in params else 0.4
    for attempt in range(3):
        async with _limiter():
            wait = _next_slot - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            _next_slot = time.monotonic() + interval
        resp = await client.get(url, params=params)
        # 429 = over NCBI's per-IP rate (shared with the search pipeline); back off and retry.
        if resp.status_code in (429, 502, 503) and attempt < 2:
            await asyncio.sleep(1.0 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    raise httpx.HTTPError("unreachable")


async def _descriptor_ids(client: httpx.AsyncClient, variant: str) -> list[str]:
    phrase = variant.replace('"', "")
    data = await _get(client, ESEARCH, _params(term=f'"{phrase}"[mh]', retmax="5"))
    # Qualifiers (e.g. "injuries") and supplementary records are dropped after
    # esummary, which reports each record's type.
    return [str(i) for i in (data.get("esearchresult") or {}).get("idlist") or []]


async def _summaries(client: httpx.AsyncClient, ids: list[str]) -> dict[str, MeshRecord]:
    if not ids:
        return {}
    data = await _get(client, ESUMMARY, _params(id=",".join(ids)))
    result = data.get("result") or {}
    out: dict[str, MeshRecord] = {}
    for uid in ids:
        doc = result.get(uid) or {}
        terms = [t for t in (doc.get("ds_meshterms") or []) if isinstance(t, str) and t.strip()]
        if doc.get("ds_recordtype") != "descriptor" or not terms:
            continue
        out[uid] = MeshRecord(
            ui=str(doc.get("ds_meshui") or ""), descriptor=terms[0], entry_terms=tuple(terms),
            scope_note=(doc.get("ds_scopenote") or "").strip()[:400] or None,
        )
    return out


async def _first_hit(client: httpx.AsyncClient, term: str, skip: int) -> Optional[tuple[str, list[str], int]]:
    """(variant, ids, index) for the first lookup form from *skip* on that NCBI knows."""
    forms = variants(term)
    for idx in range(skip, len(forms)):
        ids = await _descriptor_ids(client, forms[idx])
        if ids:
            return forms[idx], ids, idx
    return None


async def _lookup_uncached(client: httpx.AsyncClient, terms: list[str]) -> dict[str, Optional[MeshRecord]]:
    """One esearch per concept (more only when a spelling variant is needed) and
    one batched esummary per pass — a handful of requests per question."""
    out: dict[str, Optional[MeshRecord]] = {t: None for t in terms}
    summaries: dict[str, MeshRecord] = {}
    pending = {t: 0 for t in terms}  # term -> next variant index to try
    for _pass in range(2):
        if not pending:
            break
        hits = await asyncio.gather(*(_first_hit(client, t, i) for t, i in pending.items()))
        new_ids = sorted({i for h in hits if h for i in h[1] if i not in summaries})
        summaries.update(await _summaries(client, new_ids))
        retry: dict[str, int] = {}
        for term, hit in zip(list(pending), hits):
            if hit is None:
                continue
            variant, ids, idx = hit
            match = next((summaries[i] for i in ids if i in summaries and entry_matches(variant, summaries[i])), None)
            if match is not None:
                out[term] = match
            else:  # e.g. only a qualifier matched: try the next spelling
                retry[term] = idx + 1
        pending = retry
    return out


async def lookup_many(
    terms: list[str], *, timeout_s: float = 10.0, transport: Optional[httpx.AsyncBaseTransport] = None,
) -> LookupResult:
    """Exact MeSH descriptor per term (None = no exact heading). Never raises."""
    result = LookupResult()
    todo: list[str] = []
    for term in terms:
        hit, record = _cache_get(norm(term))
        if hit:
            result.records[term] = record
        elif term not in todo:
            todo.append(term)
    if not todo:
        return result
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S, transport=transport) as client:
            found = await asyncio.wait_for(_lookup_uncached(client, todo), timeout=timeout_s)
    except (httpx.HTTPError, asyncio.TimeoutError, ValueError) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        logger.warning("mesh lookup unavailable: %s %s", type(exc).__name__, status or "")
        result.status = "unavailable"
        for term in todo:
            result.records[term] = None
        return result
    for term, record in found.items():
        _cache_put(norm(term), record)
        result.records[term] = record
    return result


# Entry terms that are codes, inverted headings or very long are no use as text words.
_CODE_LIKE = re.compile(r"\b[A-Z]{1,5}[- ]?\d{2,}\b|\d{3,}")


def entry_terms_as_words(record: MeshRecord, concept: str, limit: int = 6) -> list[str]:
    """Natural-language entry terms to search as title/abstract words."""
    seen = {norm(concept), norm(_singular(concept))}
    out: list[str] = []
    for entry in record.entry_terms:
        if ", " in entry or _CODE_LIKE.search(entry) or len(entry.split()) > 5 or len(entry) < 3:
            continue
        key = norm(_singular(entry))
        if key in seen or norm(entry) in seen:
            continue
        seen.update({key, norm(entry)})
        out.append(entry)
        if len(out) >= limit:
            break
    return out
