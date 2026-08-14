"""Person-mention filtering for quotegraph (post-merge).

Merging runs before attribution (MERGING.md).  Do not gate individual
atomic spans on in-quote person mentions — split halves of the same turn
often carry the entity in only one span.  Filter merged blocks instead via
:func:`merge_pipeline.block_passes_post_merge_filter`.
"""

from __future__ import annotations

import re

DEFAULT_BRIDGE_WINDOW_WORDS = 10


def _word_window(text: str, from_start: bool, window_words: int) -> str:
    words = re.findall(r"\w+|[^\w\s]", text or "")
    if not words:
        return ""
    if from_start:
        return " ".join(words[:window_words])
    return " ".join(words[-window_words:])


def bridge_has_person_entity(
    bridge: str,
    person_qids: set[str],
    bridge_mentions: set[str] | frozenset[str] | None = None,
    window_words: int = DEFAULT_BRIDGE_WINDOW_WORDS,
) -> bool:
    """True when a linked person appears in the bridge near quote boundaries."""

    if not bridge or not person_qids:
        return False
    if bridge_mentions:
        return bool(person_qids.intersection(bridge_mentions))

    left_window = _word_window(bridge, from_start=True, window_words=window_words)
    right_window = _word_window(bridge, from_start=False, window_words=window_words)
    # Without explicit entity linking on the bridge, callers should pass bridge_mentions.
    return bool(left_window or right_window)


def quote_is_candidate(
    quote_mentions: set[str] | frozenset[str],
    bridge_before_mentions: set[str] | frozenset[str] | None = None,
    bridge_after_mentions: set[str] | frozenset[str] | None = None,
    person_qids: set[str] | None = None,
    window_words: int = DEFAULT_BRIDGE_WINDOW_WORDS,
) -> bool:
    """Legacy per-span gate. Prefer post-merge filtering in merge_pipeline."""

    if not person_qids:
        return bool(quote_mentions)

    if quote_mentions.intersection(person_qids):
        return True

    before = bridge_before_mentions or set()
    after = bridge_after_mentions or set()
    if before.intersection(person_qids) or after.intersection(person_qids):
        return True
    return False


def filter_candidate_indices(
    quotes: list,
    bridges: list[str],
    quote_mentions: list[set[str]],
    bridge_mentions: list[set[str]] | None = None,
    person_qids: set[str] | None = None,
    window_words: int = DEFAULT_BRIDGE_WINDOW_WORDS,
) -> list[int]:
    """Return indices of quotes that pass the widened candidate filter."""

    bridge_mentions = bridge_mentions or [set() for _ in bridges]
    selected: list[int] = []
    for idx, mentions in enumerate(quote_mentions):
        before = bridge_mentions[idx - 1] if idx > 0 else set()
        after = bridge_mentions[idx] if idx < len(bridge_mentions) else set()
        if quote_is_candidate(
            mentions,
            bridge_before_mentions=before,
            bridge_after_mentions=after,
            person_qids=person_qids,
            window_words=window_words,
        ):
            selected.append(idx)
    return selected
