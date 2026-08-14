"""Merge-first pipeline: join quote spans, then filter by person mentions.

Per MERGING.md the merge step runs before attribution.  Person-mention
filtering must therefore happen on merged blocks, not on individual atomic
spans — otherwise split halves of the same turn are dropped because only
one span carries the linked entity.
"""

from __future__ import annotations

from dataclasses import dataclass

from quotegraph.attribution_aggregate import aggregate_mentions
from quotegraph.merger_patterns import load_merger_patterns
from quotegraph.quote_merger import (
    MergeBlock,
    QuoteCandidate,
    QuoteRole,
    QuoteTurnMerger,
    normalize_bridge,
)


@dataclass(frozen=True)
class MergedQuoteBlock:
    block: MergeBlock
    member_quotes: tuple[QuoteCandidate, ...]
    member_bridges: tuple[str, ...]
    mention_ids: frozenset[str]
    left_role: str
    right_role: str


def block_mention_ids(
    quotes: list[QuoteCandidate],
    bridges: list[str],
    start: int,
    end: int,
    *,
    bridge_mention_ids: list[tuple[str, ...]] | None = None,
    left_role: str = "utterance",
    right_role: str = "utterance",
) -> frozenset[str]:
    """Union person-entity ids across spans (and scare bridges) in one block."""

    member_quotes = quotes[start : end + 1]
    member_bridges = bridges[start:end] if end > start else []
    bridge_rows = bridge_mention_ids[start:end] if bridge_mention_ids else None
    return aggregate_mentions(
        member_quotes,
        member_bridges,
        bridge_mentions=bridge_rows,
        left_role=left_role,
        right_role=right_role,
    )


def span_passes_pre_merge_filter(
    quote: QuoteCandidate,
    bridge_before: set[str] | frozenset[str] | None = None,
    bridge_after: set[str] | frozenset[str] | None = None,
    person_ids: set[str] | None = None,
) -> bool:
    """Legacy per-span gate — misses split-quote siblings (see module docstring)."""

    if not person_ids:
        return bool(quote.mentioned_entities)
    quote_ids = set(quote.mentioned_entities)
    if quote_ids.intersection(person_ids):
        return True
    before = bridge_before or set()
    after = bridge_after or set()
    return bool(before.intersection(person_ids) or after.intersection(person_ids))


def block_has_person_speaker(
    quotes: list[QuoteCandidate],
    bridges: list[str],
    block: MergeBlock,
) -> bool:
    """True iff the merged block has any predicted person speaker.

    Recognises three signals (any one is sufficient):

    1. A member span carries ``speaker_qid`` (entity-linked candidate).
    2. A member span carries a non-empty ``speaker`` name string.
    3. A bridge between member spans matches a person-attribution pattern
       (e.g. ``Smith said``, ``Mr. Johnson wrote``) and is *not* flagged as
       institutional.
    """

    patterns = load_merger_patterns()
    for quote in quotes[block.start_index : block.end_index + 1]:
        if quote.speaker_qid:
            return True
        if quote.speaker and quote.speaker.strip():
            return True

    for idx in range(block.start_index, block.end_index):
        bridge = normalize_bridge(bridges[idx]).strip("\u201c\u201d\u2018\u2019\"' `")
        if not bridge:
            continue
        if patterns.institutional_bridge_re.search(bridge):
            continue
        if patterns.person_attribution_bridge_re.search(bridge):
            return True
    return False


def block_is_non_quote(block: MergeBlock) -> bool:
    """Block-wide non_quote classification.

    True iff *every* decision in the block has both roles flagged
    ``non_quote``.  A single utterance decision is enough to keep the block.
    """

    decisions = block.decisions
    if not decisions:
        return False
    return all(
        d.left_role is QuoteRole.NON_QUOTE and d.right_role is QuoteRole.NON_QUOTE
        for d in decisions
    )


def block_passes_speaker_filter(
    quotes: list[QuoteCandidate],
    bridges: list[str],
    block: MergeBlock,
) -> bool:
    """Drop blocks with no person speaker or marked as non_quote."""

    if block_is_non_quote(block):
        return False
    return block_has_person_speaker(quotes, bridges, block)


def block_passes_post_merge_filter(
    quotes: list[QuoteCandidate],
    bridges: list[str],
    block: MergeBlock,
    person_ids: set[str] | None = None,
    *,
    bridge_mention_ids: list[tuple[str, ...]] | None = None,
) -> bool:
    """Keep blocks whose merged unit carries at least one target person id."""

    mentions = block_mention_ids(
        quotes,
        bridges,
        block.start_index,
        block.end_index,
        bridge_mention_ids=bridge_mention_ids,
        left_role=block.left_role.value,
        right_role=block.right_role.value,
    )
    if not person_ids:
        return bool(mentions)
    return bool(mentions.intersection(person_ids))


def filter_merged_blocks(
    quotes: list[QuoteCandidate],
    bridges: list[str],
    blocks: list[MergeBlock],
    person_ids: set[str] | None = None,
    *,
    bridge_mention_ids: list[tuple[str, ...]] | None = None,
) -> list[MergedQuoteBlock]:
    kept: list[MergedQuoteBlock] = []
    for block in blocks:
        if not block_passes_post_merge_filter(
            quotes, bridges, block, person_ids, bridge_mention_ids=bridge_mention_ids
        ):
            continue
        kept.append(
            MergedQuoteBlock(
                block=block,
                member_quotes=tuple(quotes[block.start_index : block.end_index + 1]),
                member_bridges=tuple(bridges[block.start_index : block.end_index]),
                mention_ids=block_mention_ids(
                    quotes,
                    bridges,
                    block.start_index,
                    block.end_index,
                    bridge_mention_ids=bridge_mention_ids,
                    left_role=block.left_role.value,
                    right_role=block.right_role.value,
                ),
                left_role=block.left_role.value,
                right_role=block.right_role.value,
            )
        )
    return kept


def merge_then_filter(
    quotes: list[QuoteCandidate],
    bridges: list[str],
    merger: QuoteTurnMerger,
    person_ids: set[str] | None = None,
    *,
    bridge_mention_ids: list[tuple[str, ...]] | None = None,
) -> list[MergedQuoteBlock]:
    """Merge all adjacent spans first, then apply mention filter on blocks."""

    blocks = merger.merge_all(quotes, bridges, bridge_mentions=bridge_mention_ids)
    return filter_merged_blocks(
        quotes, bridges, blocks, person_ids, bridge_mention_ids=bridge_mention_ids
    )


def split_mention_recovery_stats(
    quotes: list[QuoteCandidate],
    bridges: list[str],
    blocks: list[MergeBlock],
    person_ids: set[str] | None = None,
    *,
    bridge_mention_ids: list[tuple[str, ...]] | None = None,
) -> dict[str, int]:
    """Count blocks saved by merge-then-filter vs naive per-span selection."""

    stats = {
        "merged_blocks": 0,
        "blocks_with_mentions": 0,
        "blocks_saved_from_split": 0,
        "spans_dropped_by_pre_merge": 0,
    }
    if not person_ids:
        return stats

    bridge_rows = bridge_mention_ids or [() for _ in bridges]
    for block in blocks:
        if block.end_index <= block.start_index:
            continue
        stats["merged_blocks"] += 1
        mentions = block_mention_ids(
            quotes,
            bridges,
            block.start_index,
            block.end_index,
            bridge_mention_ids=bridge_rows,
            left_role=block.left_role.value,
            right_role=block.right_role.value,
        )
        if not mentions.intersection(person_ids):
            continue
        stats["blocks_with_mentions"] += 1

        span_survives = []
        for idx in range(block.start_index, block.end_index + 1):
            before = set(bridge_rows[idx - 1]) if idx > 0 else set()
            after = set(bridge_rows[idx]) if idx < len(bridge_rows) else set()
            span_survives.append(
                span_passes_pre_merge_filter(quotes[idx], before, after, person_ids)
            )
        if not all(span_survives):
            stats["blocks_saved_from_split"] += 1
        stats["spans_dropped_by_pre_merge"] += sum(1 for ok in span_survives if not ok)
    return stats
