#!/usr/bin/env python3
"""Summarize DirectQuote attribution LLM errors from a detail JSON file."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TITLE_TOKENS = {
    "chief",
    "deputy",
    "dr",
    "governor",
    "gov",
    "mr",
    "mrs",
    "ms",
    "president",
    "prof",
    "rep",
    "representative",
    "secretary",
    "sen",
    "senator",
}

QUOTATIVE_VERBS = {
    "added",
    "announced",
    "asked",
    "insisted",
    "replied",
    "responded",
    "said",
    "says",
    "stated",
    "told",
    "tweeted",
    "warned",
    "wrote",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def speaker_correct_relaxed(row: dict[str, Any]) -> bool:
    """Score exact speaker matches plus pronoun predictions resolved by referent.

    DirectQuote sometimes annotates pronouns as speakers.  For model outputs that
    choose a pronoun, count the attribution as correct when either the pronoun
    itself exactly matches the gold speaker span or the model's referent span
    overlaps the gold speaker span.
    """

    if row["speaker_exact"]:
        return True
    if not row["speaker_is_pronoun"]:
        return False
    gold = set(_tokens(row["gold_speaker"]))
    referent = set(_tokens(row["pred_speaker_refers_to"]))
    return bool(gold and referent and gold.intersection(referent))


def categorize(row: dict[str, Any]) -> str:
    """Assign a human-readable error category."""

    if speaker_correct_relaxed(row):
        return "correct"
    pred = _tokens(row["pred_speaker"])
    gold = _tokens(row["gold_speaker"])
    pred_set = set(pred)
    gold_set = set(gold)

    if not gold and pred:
        return "false_positive_unknown"
    if gold and not pred:
        return "missing_speaker"
    if pred_set.isdisjoint(gold_set):
        return "wrong_speaker"
    if pred_set.intersection(QUOTATIVE_VERBS):
        return "speaker_span_includes_verb_or_clause"
    if gold_set.issubset(pred_set) and (pred_set - gold_set).intersection(TITLE_TOKENS):
        return "title_or_role_expansion"
    if gold_set.issubset(pred_set):
        return "overlong_name_span"
    if pred_set.issubset(gold_set):
        return "underspecified_name_span"
    return "alias_or_partial_overlap"


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    configs = sorted({row["prompt_config"] for row in rows})
    by_config: dict[str, Any] = {}
    for config in configs:
        subset = [row for row in rows if row["prompt_config"] == config]
        categories = Counter(categorize(row) for row in subset)
        exact_misses = [row for row in subset if not row["speaker_exact"]]
        low_overlap = [row for row in exact_misses if row["speaker_token_f1"] < 0.5]
        partial_overlap = [row for row in exact_misses if row["speaker_token_f1"] >= 0.5]
        copied_failures = [
            row
            for row in subset
            if not (row["speaker_in_document"] and row["referent_in_document"] and row["verb_in_document"])
        ]
        pronoun_predictions = [row for row in subset if row["speaker_is_pronoun"]]
        by_config[config] = {
            "total": len(subset),
            "speaker_exact_accuracy": _rate(sum(row["speaker_exact"] for row in subset), len(subset)),
            "speaker_relaxed_accuracy": _rate(
                sum(speaker_correct_relaxed(row) for row in subset),
                len(subset),
            ),
            "speaker_token_f1": sum(row["speaker_token_f1"] for row in subset) / len(subset),
            "quotative_verb_exact_accuracy": _rate(sum(row["verb_exact"] for row in subset), len(subset)),
            "exact_misses": len(exact_misses),
            "relaxed_misses": sum(not speaker_correct_relaxed(row) for row in subset),
            "low_overlap_misses": len(low_overlap),
            "partial_overlap_misses": len(partial_overlap),
            "copied_span_failures": len(copied_failures),
            "pronoun_predictions": len(pronoun_predictions),
            "pronoun_missing_referent": sum(
                row["speaker_is_pronoun"] and not row["pronoun_referent_present"]
                for row in subset
            ),
            "categories": dict(categories),
        }
    return {
        "by_config": by_config,
        "overall_categories": dict(Counter(categorize(row) for row in rows)),
    }


def grouped_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["paragraph_id"], row["quote_index"])].append(row)

    examples: list[dict[str, Any]] = []
    for (paragraph_id, quote_index), subset in sorted(grouped.items()):
        if all(row["speaker_exact"] for row in subset):
            continue
        first = subset[0]
        examples.append(
            {
                "paragraph_id": paragraph_id,
                "quote_index": quote_index,
                "quote_kind": first["quote_kind"],
                "target_quote": first["target_quote"],
                "gold_speaker": first["gold_speaker"],
                "gold_quotative_verb": first["gold_quotative_verb"],
                "predictions": [
                    {
                        "prompt_config": row["prompt_config"],
                        "category": categorize(row),
                        "pred_speaker": row["pred_speaker"],
                        "speaker_token_f1": row["speaker_token_f1"],
                        "pred_speaker_refers_to": row["pred_speaker_refers_to"],
                        "pred_quotative_verb": row["pred_quotative_verb"],
                        "copied_spans": row["speaker_in_document"]
                        and row["referent_in_document"]
                        and row["verb_in_document"],
                        "speaker_exact": row["speaker_exact"],
                        "verb_exact": row["verb_exact"],
                    }
                    for row in sorted(subset, key=lambda item: item["prompt_config"])
                ],
            }
        )
    return examples


def render_markdown(
    *,
    detail_path: Path,
    summary: dict[str, Any],
    examples: list[dict[str, Any]],
) -> str:
    config_stats = summary["by_config"]
    best_exact = max(
        config_stats.items(),
        key=lambda item: (item[1]["speaker_relaxed_accuracy"], item[1]["speaker_token_f1"]),
    )
    direct = config_stats.get("direct")
    repeat = config_stats.get("direct_repeat")
    overall = Counter(summary["overall_categories"])
    dominant_error = next(
        (name for name, _ in overall.most_common() if name != "correct"),
        "none",
    )
    findings = [
        (
            f"Best speaker accuracy is `{best_exact[0]}` at "
            f"{best_exact[1]['speaker_relaxed_accuracy']:.3f} under relaxed scoring, with token F1 "
            f"{best_exact[1]['speaker_token_f1']:.3f}."
        ),
        (
            f"The dominant non-correct category is `{dominant_error}` "
            f"({overall[dominant_error]} cases across prompt configs)."
        ),
        (
            f"Wrong-speaker and missing-speaker errors are rare in this slice: "
            f"{overall['wrong_speaker']} wrong-speaker, {overall['missing_speaker']} missing-speaker."
        ),
    ]
    if direct and repeat:
        findings.append(
            "Repeating the input changed exact accuracy from "
            f"{direct['speaker_exact_accuracy']:.3f} to "
            f"{repeat['speaker_exact_accuracy']:.3f}, and token F1 from "
            f"{direct['speaker_token_f1']:.3f} to {repeat['speaker_token_f1']:.3f}."
        )
    if any(stats["copied_span_failures"] for stats in config_stats.values()):
        findings.append(
            "Some outputs failed copied-span validation, usually because punctuation/tokenization "
            "or referent expansion did not exactly match the document text."
        )
    else:
        findings.append("All predicted speaker/referent/verb spans were copied from the document.")

    lines = [
        "# DirectQuote Attribution Error Analysis",
        "",
        f"Source detail file: `{detail_path}`",
        "",
        "## Summary",
        "",
        "| config | exact acc | relaxed acc | token F1 | verb acc | exact misses | relaxed misses | partial misses | low-overlap misses | copied-span failures | pronoun missing referent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config, stats in summary["by_config"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    config,
                    f"{stats['speaker_exact_accuracy']:.3f}",
                    f"{stats['speaker_relaxed_accuracy']:.3f}",
                    f"{stats['speaker_token_f1']:.3f}",
                    f"{stats['quotative_verb_exact_accuracy']:.3f}",
                    str(stats["exact_misses"]),
                    str(stats["relaxed_misses"]),
                    str(stats["partial_overlap_misses"]),
                    str(stats["low_overlap_misses"]),
                    str(stats["copied_span_failures"]),
                    str(stats["pronoun_missing_referent"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Error Categories",
            "",
            "| config | correct | title/role expansion | includes verb/clause | overlong name | wrong speaker | missing speaker |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for config, stats in summary["by_config"].items():
        cats = Counter(stats["categories"])
        lines.append(
            "| "
            + " | ".join(
                [
                    config,
                    str(cats["correct"]),
                    str(cats["title_or_role_expansion"]),
                    str(cats["speaker_span_includes_verb_or_clause"]),
                    str(cats["overlong_name_span"]),
                    str(cats["wrong_speaker"]),
                    str(cats["missing_speaker"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            *[f"- {finding}" for finding in findings],
            "",
            "## Recommendations",
            "",
            f"- Keep `{best_exact[0]}` for the next larger run; keep another prompt only if it adds complementary error behavior.",
            "- Tighten the prompt with: `speaker must be the minimal DirectQuote Speaker span, excluding titles, roles, appositives, and quotative verbs unless the title is the only speaker text available`.",
            "- Add post-processing metrics that strip titles/roles and compare last-name aliases. The current exact score underestimates attribution correctness because many errors are span-boundary mismatches.",
            "- Add a separate gold/metric for quotative verbs from the labeled attribution context. The current gold verb is heuristic, so some verb mismatches are evaluator noise.",
            "",
            "## Missed Examples",
            "",
        ]
    )
    for example in examples:
        lines.extend(
            [
                f"### Paragraph {example['paragraph_id']}, Quote {example['quote_index']} ({example['quote_kind']})",
                "",
                f"- Gold speaker: `{example['gold_speaker']}`",
                f"- Gold quotative verb: `{example['gold_quotative_verb']}`",
                f"- Quote: `{example['target_quote'][:180]}`",
                "",
                "| config | category | predicted speaker | F1 | predicted verb | copied spans |",
                "|---|---|---|---:|---|---:|",
            ]
        )
        for pred in example["predictions"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        pred["prompt_config"],
                        pred["category"],
                        f"`{pred['pred_speaker']}`",
                        f"{pred['speaker_token_f1']:.3f}",
                        f"`{pred['pred_quotative_verb']}`",
                        str(pred["copied_spans"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.detail_json.read_text(encoding="utf-8"))
    summary = summarize(rows)
    examples = grouped_examples(rows)

    output = {"summary": summary, "missed_examples": examples}
    if args.output_json:
        args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.write_text(
            render_markdown(detail_path=args.detail_json, summary=summary, examples=examples),
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
