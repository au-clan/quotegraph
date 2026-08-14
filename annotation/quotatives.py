"""Nearby reporting-verb / cue candidates for quotative annotation."""

from __future__ import annotations

from typing import Any

# Frozen news cues. Not Hu's full inventory — enough to suggest spans.
# Annotators can still select any span, including verbs missing here.
QUOTATIVE_CUES: tuple[str, ...] = tuple(
    sorted(
        {
            "according to",
            "acknowledged",
            "added",
            "adds",
            "admitted",
            "alleged",
            "announced",
            "argued",
            "argues",
            "asked",
            "asks",
            "asserted",
            "claimed",
            "claims",
            "commented",
            "conceded",
            "confirmed",
            "continued",
            "declared",
            "denied",
            "described",
            "estimated",
            "explained",
            "insisted",
            "joked",
            "noted",
            "notes",
            "predicted",
            "promised",
            "recalled",
            "replied",
            "reported",
            "responded",
            "said",
            "says",
            "saying",
            "stated",
            "stating",
            "suggested",
            "testified",
            "told",
            "tweeted",
            "urged",
            "vowed",
            "warned",
            "wrote",
            "writes",
        },
        key=lambda cue: (-len(cue), cue),
    )
)

WINDOW = 160


def _boundary_ok(text: str, start: int, end: int) -> bool:
    if start > 0 and (text[start - 1].isalnum() or text[start - 1] == "'"):
        return False
    if end < len(text) and (text[end].isalnum() or text[end] == "'"):
        return False
    return True


def find_quotative_candidates(
    text: str,
    *,
    inner_start: int,
    inner_end: int,
    outer_start: int,
    outer_end: int,
    window: int = WINDOW,
) -> list[dict[str, Any]]:
    """Return cue spans in a window immediately before/after the quote, not inside it."""
    spans = (
        (max(0, outer_start - window), inner_start),
        (outer_end, min(len(text), outer_end + window)),
    )
    lower = text.lower()
    found: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for left, right in spans:
        if right <= left:
            continue
        for cue in QUOTATIVE_CUES:
            start = left
            while True:
                index = lower.find(cue, start, right)
                if index < 0:
                    break
                end = index + len(cue)
                start = index + 1
                if not _boundary_ok(text, index, end):
                    continue
                key = (index, end)
                if key in seen:
                    continue
                seen.add(key)
                found.append({"start": index, "end": end, "surface": text[index:end]})
    found.sort(key=lambda row: (min(abs(row["start"] - outer_start), abs(row["end"] - outer_end)), row["start"]))
    return found
