"""Wikidata search for entity linking in the annotation UI."""

from __future__ import annotations

import time
from typing import Any

import httpx

USER_AGENT = "QuotegraphAnnotation/0.1 (https://github.com/au-clan/quotegraph)"
SEARCH_URL = "https://www.wikidata.org/w/api.php"
_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
CACHE_TTL = 300.0
HUMAN = "Q5"
FICTIONAL_HUMAN = "Q15632617"


def _instance_ids(entity: dict[str, Any]) -> set[str]:
    claims = (entity.get("claims") or {}).get("P31") or []
    ids: set[str] = set()
    for claim in claims:
        try:
            ids.add(claim["mainsnak"]["datavalue"]["value"]["id"])
        except (KeyError, TypeError):
            continue
    return ids


def _occupation_labels(entity: dict[str, Any], extras: dict[str, Any]) -> str:
    claims = (entity.get("claims") or {}).get("P106") or []
    labels: list[str] = []
    for claim in claims[:3]:
        try:
            qid = claim["mainsnak"]["datavalue"]["value"]["id"]
        except (KeyError, TypeError):
            continue
        label = ((extras.get(qid) or {}).get("labels") or {}).get("en") or {}
        if label.get("value"):
            labels.append(label["value"])
    return ", ".join(labels)


def search_entities(query: str, *, limit: int = 8) -> list[dict[str, Any]]:
    query = query.strip()
    if len(query) < 2:
        return []
    key = query.lower()
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1]

    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": "en",
        "type": "item",
        "limit": str(max(limit, 12)),
        "format": "json",
    }
    with httpx.Client(timeout=12.0, headers={"User-Agent": USER_AGENT}) as client:
        search = client.get(SEARCH_URL, params=params)
        search.raise_for_status()
        hits = search.json().get("search") or []
        ids = [row["id"] for row in hits if row.get("id")]
        if not ids:
            _cache[key] = (now, [])
            return []
        entities_resp = client.get(
            SEARCH_URL,
            params={
                "action": "wbgetentities",
                "ids": "|".join(ids),
                "props": "labels|descriptions|claims",
                "languages": "en",
                "format": "json",
            },
        )
        entities_resp.raise_for_status()
        entities = entities_resp.json().get("entities") or {}

        occupation_ids: list[str] = []
        for entity in entities.values():
            for claim in (entity.get("claims") or {}).get("P106") or []:
                try:
                    occupation_ids.append(claim["mainsnak"]["datavalue"]["value"]["id"])
                except (KeyError, TypeError):
                    continue
        extras: dict[str, Any] = {}
        if occupation_ids:
            extra_resp = client.get(
                SEARCH_URL,
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(dict.fromkeys(occupation_ids)),
                    "props": "labels",
                    "languages": "en",
                    "format": "json",
                },
            )
            extra_resp.raise_for_status()
            extras = extra_resp.json().get("entities") or {}

    rows: list[dict[str, Any]] = []
    for row in hits:
        entity = entities.get(row["id"]) or {}
        types = _instance_ids(entity)
        rows.append(
            {
                "qid": row["id"],
                "label": row.get("label") or ((entity.get("labels") or {}).get("en") or {}).get("value") or row["id"],
                "description": row.get("description")
                or ((entity.get("descriptions") or {}).get("en") or {}).get("value")
                or "",
                "occupation": _occupation_labels(entity, extras),
                "is_human": HUMAN in types or FICTIONAL_HUMAN in types,
                "fictional": FICTIONAL_HUMAN in types,
            }
        )

    rows = rows[:limit]
    _cache[key] = (now, rows)
    return rows
