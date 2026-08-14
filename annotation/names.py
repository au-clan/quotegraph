"""Proper-name spans used as Wikidata search chips in the annotator."""

from __future__ import annotations

import re
from typing import Any

from annotation.schema import PRONOUNS

NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:[ '-][A-Z][a-z]+){0,3}\b")

STOP = frozenset(
    {
        "The",
        "A",
        "An",
        "And",
        "But",
        "Or",
        "For",
        "In",
        "On",
        "At",
        "To",
        "From",
        "With",
        "This",
        "That",
        "These",
        "Those",
        "There",
        "Then",
        "After",
        "Before",
        "During",
        "While",
        "When",
        "Where",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "President",
        "Prime",
        "Minister",
        "Senator",
        "Representative",
        "Secretary",
        "House",
        "White",
        "New",
        "United",
        "States",
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "Later",
        "According",
        "Washington",
    }
)


def find_name_spans(text: str) -> list[dict[str, Any]]:
    """Capitalized name-like spans. Titles and calendar words are dropped."""
    found: list[dict[str, Any]] = []
    for match in NAME_RE.finditer(text):
        surface = match.group(0)
        if surface in STOP or surface.lower() in PRONOUNS:
            continue
        if all(part in STOP for part in surface.replace("-", " ").split()):
            continue
        found.append({"start": match.start(), "end": match.end(), "surface": surface})
    return found


def nearest_name_candidates(
    text: str,
    *,
    center: int,
    limit: int = 10,
    window: int = 600,
) -> list[dict[str, Any]]:
    """Deduplicated names near ``center``, closest first."""
    left, right = max(0, center - window), min(len(text), center + window)
    ranked: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()
    for span in find_name_spans(text):
        if span["end"] < left or span["start"] > right:
            continue
        key = span["surface"].lower()
        if key in seen:
            continue
        seen.add(key)
        ranked.append((abs(span["start"] - center), span))
    ranked.sort(key=lambda item: (item[0], item[1]["start"]))
    return [span for _, span in ranked[:limit]]
