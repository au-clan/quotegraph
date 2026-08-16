#!/usr/bin/env python3
"""Truecaser eval on retrieved A–C articles, aligned through Quootstrap prep."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quotegraph"))

from dump_text import clean_dump_text
from quotebank_prep import html_to_ptb, start_jvm
from spinn3r import html_gz_path

LOWER_PHASES = {"A", "B", "C"}
TRAIN_PHASES = {"D", "E"}


def open_jsonl(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_html(path: Path) -> str:
    return gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")


def align_words(dump_words: list[str], html_words: list[str]) -> tuple[list[str], list[str]]:
    """Keep only SequenceMatcher equal blocks (lowercase identity)."""
    a = [t.lower() for t in dump_words]
    b = [t.lower() for t in html_words]
    dump_al: list[str] = []
    html_al: list[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            dump_al.extend(dump_words[i1:i2])
            html_al.extend(html_words[j1:j2])
    return dump_al, html_al


def has_upper(text: str) -> bool:
    return any("A" <= c <= "Z" for c in text)


def score_tokens(pred: list[str], gold: list[str]) -> tuple[int, int, int, int]:
    ok = n = cased_ok = cased_n = 0
    for p, g in zip(pred, gold):
        if p.lower() != g.lower():
            continue
        n += 1
        ok += int(p == g)
        if has_upper(g):
            cased_n += 1
            cased_ok += int(p == g)
    return ok, n, cased_ok, cased_n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="/home/mculjak/datasets/quotegraph_poc/sample.jsonl.gz")
    parser.add_argument("--html-dir", default="/home/mculjak/datasets/quotegraph_poc/html")
    parser.add_argument("--out", default="/home/mculjak/datasets/quotegraph_poc/truecase_eval.json")
    parser.add_argument("--min-coverage", type=float, default=0.9)
    parser.add_argument("--train-limit", type=int, default=100_000)
    parser.add_argument("--jars", default="")
    args = parser.parse_args()
    html_dir = Path(args.html_dir)
    start_jvm(Path(args.jars) if args.jars else None)

    from sacremoses import MosesTruecaser

    train_docs: list[list[str]] = []
    eval_rows: list[dict] = []
    n_html = 0
    with open_jsonl(Path(args.sample)) as handle:
        for line in handle:
            rec = json.loads(line)
            phase = rec.get("phase") or ""
            if phase in TRAIN_PHASES and len(train_docs) < args.train_limit:
                text = clean_dump_text(rec.get("content") or "")
                if text:
                    train_docs.append(text.split())
            if phase not in LOWER_PHASES:
                continue
            path = html_gz_path(html_dir, rec["article_id"])
            if not path.exists():
                continue
            n_html += 1
            eval_rows.append(rec)
    print(f"train_docs={len(train_docs)} html_abc={n_html}", flush=True)
    if not train_docs:
        sys.exit("no D/E dump text to train Moses")
    truecaser = MosesTruecaser()
    truecaser.train(train_docs, possibly_use_first_token=True)
    print("trained Moses truecaser", flush=True)

    stats = defaultdict(lambda: Counter())
    n_aligned = 0
    for i, rec in enumerate(eval_rows, 1):
        dump_ann = clean_dump_text(rec.get("content") or "")
        html_ptb = html_to_ptb(read_html(html_gz_path(html_dir, rec["article_id"])), rec.get("url") or "")
        html_ann = clean_dump_text(" ".join(html_ptb))
        dump_al, html_al = align_words(dump_ann.split(), html_ann.split())
        coverage = len(dump_al) / max(1, len(dump_ann.split()))
        stats[rec["phase"]]["n_html"] += 1
        stats[rec["phase"]]["coverage_sum"] += coverage
        if coverage < args.min_coverage or not dump_al:
            if i % 50 == 0:
                print(f"  {i}/{len(eval_rows)} aligned={n_aligned}", flush=True)
            continue
        dump_ann = " ".join(dump_al)
        html_ann = " ".join(html_al)
        if dump_ann.lower() != html_ann.lower() or not has_upper(html_ann):
            continue
        pred = truecaser.truecase(dump_ann)
        gold = html_ann.split()
        if len(pred) != len(gold):
            pred = " ".join(pred).split()
        ok, n, cased_ok, cased_n = score_tokens(pred, gold)
        if n == 0:
            continue
        n_aligned += 1
        bucket = stats[rec["phase"]]
        bucket["n_aligned"] += 1
        bucket["tok_ok"] += ok
        bucket["tok_n"] += n
        bucket["cased_ok"] += cased_ok
        bucket["cased_n"] += cased_n
        if i % 50 == 0:
            print(f"  {i}/{len(eval_rows)} aligned={n_aligned}", flush=True)

    report = {"html_abc": n_html, "aligned": n_aligned, "min_coverage": args.min_coverage, "phases": {}}
    for phase in ("A", "B", "C"):
        c = stats[phase]
        n_tok = c["tok_n"] or 1
        n_cased = c["cased_n"] or 1
        n_html_p = c["n_html"] or 1
        report["phases"][phase] = {
            "n_html": c["n_html"],
            "n_aligned": c["n_aligned"],
            "mean_coverage": c["coverage_sum"] / n_html_p,
            "token_acc": c["tok_ok"] / n_tok,
            "cased_token_acc": c["cased_ok"] / n_cased,
            "n_tokens": c["tok_n"],
            "n_cased_tokens": c["cased_n"],
        }
    tok_n = sum(stats[p]["tok_n"] for p in LOWER_PHASES)
    tok_ok = sum(stats[p]["tok_ok"] for p in LOWER_PHASES)
    cased_n = sum(stats[p]["cased_n"] for p in LOWER_PHASES)
    cased_ok = sum(stats[p]["cased_ok"] for p in LOWER_PHASES)
    report["token_acc"] = tok_ok / tok_n if tok_n else None
    report["cased_token_acc"] = cased_ok / cased_n if cased_n else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
