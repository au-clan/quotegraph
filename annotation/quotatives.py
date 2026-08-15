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
    segments: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return cue spans in a window immediately before/after the quote, not inside it.

    For a merged interrupted quotation, ``segments`` also searches the bridge
    (``remarked SPEAKER``) between mark-spans.
    """
    if segments and len(segments) >= 2:
        ordered = sorted(segments, key=lambda seg: int(seg["outer_start"]))
        spans: list[tuple[int, int]] = [
            (max(0, int(ordered[0]["outer_start"]) - window), int(ordered[0]["inner_start"]))
        ]
        for left_seg, right_seg in zip(ordered, ordered[1:]):
            spans.append((int(left_seg["outer_end"]), int(right_seg["outer_start"])))
        last = ordered[-1]
        spans.append((int(last["outer_end"]), min(len(text), int(last["outer_end"]) + window)))
        inners = [(int(seg["inner_start"]), int(seg["inner_end"])) for seg in ordered]
        rank_outer_start, rank_outer_end = int(ordered[0]["outer_start"]), int(ordered[-1]["outer_end"])
    else:
        spans = [
            (max(0, outer_start - window), inner_start),
            (outer_end, min(len(text), outer_end + window)),
        ]
        inners = [(inner_start, inner_end)]
        rank_outer_start, rank_outer_end = outer_start, outer_end
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
                if any(inner_s <= index < end <= inner_e for inner_s, inner_e in inners):
                    continue
                key = (index, end)
                if key in seen:
                    continue
                seen.add(key)
                found.append({"start": index, "end": end, "surface": text[index:end]})
    found.sort(
        key=lambda row: (
            min(abs(row["start"] - rank_outer_start), abs(row["end"] - rank_outer_end)),
            row["start"],
        )
    )
    return found
