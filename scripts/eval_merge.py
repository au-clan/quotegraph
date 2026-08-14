#!/usr/bin/env python3
"""Evaluate quote-merger configurations on a labeled dev set (MERGING.md §9)."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from quotegraph.quote_merger import (
    AtomicPairContext,
    MergeAnswer,
    OpenAICompatibleAdjudicator,
    QuoteCandidate,
    QuoteRole,
    QuoteTurnMerger,
)


def load_dev_set(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _quote_from_dict(data: dict) -> QuoteCandidate:
    return QuoteCandidate(
        text=data["text"],
        quote_id=data.get("quote_id"),
        speaker=data.get("speaker"),
        speaker_qid=data.get("speaker_qid"),
        speaker_probability=float(data.get("speaker_probability") or 0.0),
        mentioned_entities=tuple(data.get("mentioned_entities") or ()),
        local_probas=tuple(tuple(row) for row in data.get("local_probas") or ()),
    )


def _pair_from_record(record: dict) -> tuple[AtomicPairContext, dict]:
    left = _quote_from_dict(record["left"])
    right = _quote_from_dict(record["right"])
    context = AtomicPairContext(
        left=left,
        right=right,
        bridge=record["bridge"],
        bridge_mentioned_entities=tuple(record.get("bridge_mentions") or ()),
    )
    return context, record["labels"]


def _config_merger(name: str, args: argparse.Namespace) -> QuoteTurnMerger:
    if name == "heuristic_only":
        return QuoteTurnMerger(adjudicator=None)
    adjudicator = OpenAICompatibleAdjudicator(
        model=args.model,
        base_url=args.base_url,
        use_self_consistency=name.endswith("_self_consistency"),
    )
    margin = 999.0 if name.endswith("_no_threshold") else args.logprob_margin
    return QuoteTurnMerger(
        adjudicator=adjudicator,
        auto_accept_logprob_margin=margin,
    )


def precision_recall_f1(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        t = [item == label for item in y_true]
        p = [item == label for item in y_pred]
        scores.append(precision_recall_f1(t, p)["f1"])
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_config(name: str, records: list[dict], args: argparse.Namespace) -> dict:
    merger = _config_merger(name, args)
    merge_true: list[bool] = []
    merge_pred: list[bool] = []
    left_true: list[str] = []
    left_pred: list[str] = []
    right_true: list[str] = []
    right_pred: list[str] = []
    bucket_counts: Counter[str] = Counter()

    for record in records:
        context, labels = _pair_from_record(record)
        bucket = record.get("bucket", "ambiguous")
        bucket_counts[bucket] += 1
        decision = merger.decide_pair(context)
        merge_true.append(labels["merge"] == "yes")
        merge_pred.append(decision.merge is MergeAnswer.YES)
        left_true.append(labels["left_role"])
        left_pred.append(decision.left_role.value)
        right_true.append(labels["right_role"])
        right_pred.append(decision.right_role.value)

    role_labels = [role.value for role in QuoteRole]
    return {
        "config": name,
        "buckets": dict(bucket_counts),
        "merge": precision_recall_f1(merge_true, merge_pred),
        "left_role_macro_f1": macro_f1(left_true, left_pred, role_labels),
        "right_role_macro_f1": macro_f1(right_true, right_pred, role_labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-set", type=Path, required=True)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "heuristic_only",
            "llm_no_threshold",
            "llm_threshold",
            "llm_threshold_self_consistency",
        ],
    )
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--logprob-margin", type=float, default=1.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = load_dev_set(args.dev_set)
    results = [evaluate_config(name, records, args) for name in args.configs]
    payload = json.dumps(results, indent=2)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
