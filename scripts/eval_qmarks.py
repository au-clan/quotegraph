#!/usr/bin/env python3
"""Score mid-token ? restore against aligned HTML gold."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quotegraph"))

from restore_text import (
    align_key,
    fold_key,
    load_question_lexicon,
    restore_question_marks,
)


def is_name_smash(dump: str, gold: str) -> bool:
    if "?" not in dump or "?" in gold:
        return False
    if dump.endswith("?") and dump.count("?") == 1:
        return False
    return fold_key(dump) == fold_key(gold) and not gold.isascii()


def load_cache(cache: Path) -> list[dict]:
    opener = gzip.open if str(cache).endswith(".gz") else open
    with opener(cache, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/home/mculjak/datasets/quotegraph_poc/align_cache.jsonl.gz")
    parser.add_argument("--out", default="/home/mculjak/datasets/quotegraph_poc/qmark_eval.json")
    args = parser.parse_args()
    lex = load_question_lexicon()
    rows = load_cache(Path(args.cache))
    n = ok = 0
    examples = []
    for rec in rows:
        raw = rec["dump_raw"]
        gold = rec["gold"]
        pred = restore_question_marks(" ".join(raw), lex).split()
        pred_map = {align_key(p): p for p in pred}
        for d, g in zip(raw, gold):
            if not is_name_smash(d, g):
                continue
            n += 1
            p = pred_map.get(align_key(g), d)
            hit = p == g or p.lower() == g.lower()
            ok += int(hit)
            if len(examples) < 40:
                examples.append(
                    {"phase": rec["phase"], "dump": d, "pred": p, "gold": g, "ok": hit}
                )
    report = {
        "lexicon": len(lex),
        "n_name_smash": n,
        "acc": ok / n if n else None,
        "ok": ok,
        "examples": examples,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
