#!/usr/bin/env python3
"""Score mojibake repair on the aligned A–C HTML cache."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import ftfy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quotegraph"))

from restore_text import align_key, fix_mojibake, fold_key, looks_mojibake


def encoding_match(pred: str, gold: str) -> bool:
    """Same word ignoring case; no leftover mojibake on the prediction."""
    if looks_mojibake(pred):
        return False
    if pred.lower() == gold.lower():
        return True
    return fold_key(pred) == fold_key(gold) and fold_key(pred) != ""


def load_cache(cache: Path) -> list[dict]:
    opener = gzip.open if str(cache).endswith(".gz") else open
    with opener(cache, "rt", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def align_fuzzy(a_words: list[str], b_words: list[str]) -> tuple[list[str], list[str]]:
    a_keys = [align_key(t) for t in a_words]
    b_keys = [align_key(t) for t in b_words]
    left: list[str] = []
    right: list[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=a_keys, b=b_keys, autojunk=False).get_opcodes():
        if tag == "equal":
            left.extend(a_words[i1:i2])
            right.extend(b_words[j1:j2])
    return left, right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/home/mculjak/datasets/quotegraph_poc/align_cache.jsonl.gz")
    parser.add_argument("--out", default="/home/mculjak/datasets/quotegraph_poc/mojibake_eval.json")
    args = parser.parse_args()
    rows = load_cache(Path(args.cache))
    stats = {k: Counter() for k in ("raw", "ftfy_only", "fix_mojibake")}
    examples = []
    n_docs = 0
    for rec in rows:
        raw = rec["dump_raw"]
        gold = rec["gold"]
        if not any(looks_mojibake(t) for t in raw):
            continue
        n_docs += 1
        preds = {
            "raw": raw,
            "ftfy_only": ftfy.fix_text(" ".join(raw), uncurl_quotes=False).split(),
            "fix_mojibake": fix_mojibake(" ".join(raw)).split(),
        }
        damaged_keys = {align_key(t) for t in raw if looks_mojibake(t)}
        for name, pred in preds.items():
            pred_al, gold_al = align_fuzzy(pred, gold)
            for p, g in zip(pred_al, gold_al):
                if align_key(p) != align_key(g):
                    continue
                if align_key(g) not in damaged_keys and not looks_mojibake(p):
                    continue
                stats[name]["n"] += 1
                stats[name]["ok"] += int(p == g)
                stats[name]["enc_ok"] += int(encoding_match(p, g))
        if len(examples) < 25:
            f_al, g_al = align_fuzzy(preds["fix_mojibake"], gold)
            mapped = {align_key(g): p for p, g in zip(f_al, g_al)}
            for d, g in zip(raw, gold):
                if not looks_mojibake(d):
                    continue
                pred = mapped.get(align_key(g), "")
                examples.append(
                    {
                        "phase": rec["phase"],
                        "dump": d,
                        "pred": pred,
                        "gold": g,
                        "ok": pred == g,
                        "enc_ok": encoding_match(pred, g) if pred else False,
                    }
                )
                if len(examples) >= 25:
                    break
    report = {
        "n_docs": n_docs,
        "systems": {
            k: {
                "exact_acc": v["ok"] / v["n"] if v["n"] else None,
                "encoding_acc": v["enc_ok"] / v["n"] if v["n"] else None,
                "n": v["n"],
                "ok": v["ok"],
                "enc_ok": v["enc_ok"],
            }
            for k, v in stats.items()
        },
        "examples": examples,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
