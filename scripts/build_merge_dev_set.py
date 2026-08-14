#!/usr/bin/env python3
"""Sample atomic adjacent pairs for merge-dev-set annotation (MERGING.md §9.1)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from quotegraph.quote_merger import AtomicPairContext, MergeAnswer, QuoteTurnMerger
from quotegraph.quotebank_loader import extract_quotes_and_bridges


def _pair_record(article_id: str, left_idx: int, quotes, bridges, bucket: str) -> dict:
    left = quotes[left_idx]
    right = quotes[left_idx + 1]
    bridge = bridges[left_idx]
    return {
        "article_id": article_id,
        "pair_index": left_idx,
        "bucket": bucket,
        "left": {
            "text": left.text,
            "quote_id": left.quote_id,
            "speaker": left.speaker,
            "speaker_qid": left.speaker_qid,
            "speaker_probability": left.speaker_probability,
            "mentioned_entities": list(left.mentioned_entities),
            "local_probas": list(left.local_probas),
        },
        "right": {
            "text": right.text,
            "quote_id": right.quote_id,
            "speaker": right.speaker,
            "speaker_qid": right.speaker_qid,
            "speaker_probability": right.speaker_probability,
            "mentioned_entities": list(right.mentioned_entities),
            "local_probas": list(right.local_probas),
        },
        "bridge": bridge,
        "bridge_mentions": [],
        "labels": {
            "merge": "no",
            "left_role": "utterance",
            "right_role": "utterance",
        },
    }


def classify_bucket(left, right, bridge, bridge_mentions) -> str:
    decision = QuoteTurnMerger().decide_pair(
        AtomicPairContext(
            left=left,
            right=right,
            bridge=bridge,
            bridge_mentioned_entities=tuple(bridge_mentions),
        )
    )
    if bridge_mentions and not left.mentioned_entities and not right.mentioned_entities:
        return "bridge_only"
    if decision.source == "llm" or decision.reason_code.value == "ambiguous":
        return "ambiguous"
    if decision.merge is MergeAnswer.YES:
        return "hard_merge"
    return "hard_no_merge"


def sample_from_articles(articles: list[dict], total: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {
        "hard_merge": [],
        "hard_no_merge": [],
        "ambiguous": [],
        "bridge_only": [],
    }

    for article in articles:
        quotes, bridges = extract_quotes_and_bridges(article)
        if len(quotes) < 2:
            continue
        article_id = str(article.get("articleID") or article.get("article_id") or "")
        for idx in range(len(bridges)):
            left, right = quotes[idx], quotes[idx + 1]
            bucket = classify_bucket(left, right, bridges[idx], [])
            if len(buckets[bucket]) >= total:
                continue
            buckets[bucket].append(_pair_record(article_id, idx, quotes, bridges, bucket))

    targets = {
        "hard_merge": 100,
        "hard_no_merge": 100,
        "ambiguous": 200,
        "bridge_only": 100,
    }
    sampled: list[dict] = []
    for bucket, target in targets.items():
        pool = buckets[bucket]
        rng.shuffle(pool)
        sampled.extend(pool[: min(target, len(pool))])
    rng.shuffle(sampled)
    return sampled[:total]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, required=True, help="JSON list of Quotebank articles")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    articles = json.loads(args.articles.read_text(encoding="utf-8"))
    sampled = sample_from_articles(articles, args.total, args.seed)
    args.output.write_text(json.dumps(sampled, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(sampled)} pairs to {args.output}")


if __name__ == "__main__":
    main()
