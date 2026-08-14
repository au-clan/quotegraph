#!/usr/bin/env python3
"""Merge-first DirectQuote run: merge blind, filter mentions on blocks, review samples."""

from __future__ import annotations

import argparse
import json
import random
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quotegraph.directquote_loader import load_paragraphs
from quotegraph.directquote_mentions import attach_mention_proxies, paragraph_person_ids
from quotegraph.merge_pipeline import (
    block_mention_ids,
    filter_merged_blocks,
    split_mention_recovery_stats,
    span_passes_pre_merge_filter,
)
from quotegraph.quote_merger import (
    DiskPairCache,
    MergeBlock,
    OpenAICompatibleAdjudicator,
    QuoteCandidate,
    QuoteTurnMerger,
    ReasonCode,
    resolve_api_key,
)


@dataclass
class RunError:
    paragraph_id: int
    error_type: str
    message: str


def _example_from_block(
    paragraph_id: int,
    quotes: list[QuoteCandidate],
    bridges: list[str],
    block: MergeBlock,
    bridge_rows: list[tuple[str, ...]],
    person_ids: set[str],
) -> dict[str, Any]:
    spans = []
    for idx in range(block.start_index, block.end_index + 1):
        before = set(bridge_rows[idx - 1]) if idx > 0 else set()
        after = set(bridge_rows[idx]) if idx < len(bridge_rows) else set()
        spans.append(
            {
                "index": idx,
                "quote_id": quotes[idx].quote_id,
                "text": quotes[idx].text,
                "mentions": list(quotes[idx].mentioned_entities),
                "pre_merge_would_keep": span_passes_pre_merge_filter(
                    quotes[idx], before, after, person_ids
                ),
            }
        )

    internal_bridges = [
        {"index": block.start_index + offset, "text": bridges[block.start_index + offset]}
        for offset in range(block.end_index - block.start_index)
    ]
    block_mentions = block_mention_ids(
        quotes,
        bridges,
        block.start_index,
        block.end_index,
        bridge_mention_ids=bridge_rows,
        left_role=block.left_role.value,
        right_role=block.right_role.value,
    )
    return {
        "paragraph_id": paragraph_id,
        "start_index": block.start_index,
        "end_index": block.end_index,
        "span_count": block.end_index - block.start_index + 1,
        "spans": spans,
        "internal_bridges": internal_bridges,
        "merged_text": block.text,
        "merged_quote_text": block.quote_text,
        "block_mentions": sorted(block_mentions),
        "post_merge_kept": bool(block_mentions.intersection(person_ids)),
        "saved_from_split_filter": any(not s["pre_merge_would_keep"] for s in spans)
        and bool(block_mentions.intersection(person_ids)),
        "left_role": block.left_role.value,
        "right_role": block.right_role.value,
        "decisions": [
            {
                "merge_prev": d.merge_prev,
                "merge_next": d.merge_next,
                "reason": d.reason_code.value,
                "source": d.source,
            }
            for d in block.decisions
        ],
    }


def run(args: argparse.Namespace) -> dict:
    paragraphs = load_paragraphs(args.input, use_gold_speakers=False)
    if args.limit_paragraphs:
        paragraphs = paragraphs[: args.limit_paragraphs]

    errors: list[RunError] = []
    adjudicator = OpenAICompatibleAdjudicator(
        model=args.model,
        key_path=Path.home() / "configs/keys/openai.txt",
        use_self_consistency=not args.no_self_consistency,
    )
    disk_cache = DiskPairCache(args.cache_dir) if args.cache_dir else None
    merger = QuoteTurnMerger(
        adjudicator=adjudicator,
        disk_cache=disk_cache,
        auto_accept_logprob_margin=args.logprob_margin,
        prefetch_llm=True,
    )

    stats: Counter[str] = Counter()
    merged_examples: list[dict[str, Any]] = []
    split_recovery_examples: list[dict[str, Any]] = []

    for paragraph in paragraphs:
        stats["paragraphs"] += 1
        person_ids = set(paragraph_person_ids(paragraph))
        tagged, bridge_rows = attach_mention_proxies(paragraph)
        quotes = list(tagged.quotes)
        bridges = list(tagged.bridges)
        stats["pairs"] += len(bridges)

        try:
            raw_blocks = merger.merge_all(quotes, bridges, bridge_mentions=bridge_rows)
            recovery = split_mention_recovery_stats(
                quotes, bridges, raw_blocks, person_ids, bridge_mention_ids=bridge_rows
            )
            for key, value in recovery.items():
                stats[key] += value

            kept = filter_merged_blocks(
                quotes, bridges, raw_blocks, person_ids, bridge_mention_ids=bridge_rows
            )
        except Exception as exc:
            stats["paragraph_exceptions"] += 1
            errors.append(
                RunError(paragraph.paragraph_id, type(exc).__name__, str(exc))
            )
            if args.verbose:
                traceback.print_exc()
            continue

        stats["raw_blocks"] += len(raw_blocks)
        stats["kept_blocks"] += len(kept)
        for block in raw_blocks:
            if block.end_index > block.start_index:
                ex = _example_from_block(
                    paragraph.paragraph_id,
                    quotes,
                    bridges,
                    block,
                    bridge_rows,
                    person_ids,
                )
                merged_examples.append(ex)
                if ex["saved_from_split_filter"]:
                    split_recovery_examples.append(ex)

        for item in kept:
            if item.block.end_index > item.block.start_index:
                stats["kept_merged_blocks"] += 1

    rng = random.Random(args.seed)
    sample_pool = split_recovery_examples or merged_examples
    rng.shuffle(sample_pool)
    samples = sample_pool[: args.inspect_n]

    stats["cache_entries"] = len(merger.cache)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "model": args.model,
        "pipeline": "merge_first_then_mention_filter",
        "use_gold_speakers": False,
        "stats": dict(stats),
        "error_count": len(errors),
        "errors": [asdict(e) for e in errors[:20]],
        "inspect_samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/tmp/DirectQuote/data/truecased.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/directquote_merge_first.json"))
    parser.add_argument("--examples", type=Path, default=Path("data/directquote_merge_examples.json"))
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--limit-paragraphs", type=int, default=0)
    parser.add_argument("--logprob-margin", type=float, default=1.5)
    parser.add_argument("--cache-dir", type=Path, default=Path(".quote_merger_cache/directquote_blind"))
    parser.add_argument("--no-self-consistency", action="store_true")
    parser.add_argument("--inspect-n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    resolve_api_key(key_path=Path.home() / "configs/keys/openai.txt")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    summary = run(args)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.examples.write_text(
        json.dumps(summary["inspect_samples"], indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "inspect_samples"}, indent=2))
    print(f"\nWrote {len(summary['inspect_samples'])} inspection examples to {args.examples}")


if __name__ == "__main__":
    main()
