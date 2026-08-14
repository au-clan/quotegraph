#!/usr/bin/env python3
"""Turn sampled Spinn3r/Quotebank records into annotation-batch JSONL.

Cleans PTB dump bodies with ftfy + Moses detokenization. Keeps 100
quote-bearing articles per Quotebank date phase.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.quotes import find_quote_candidates
from quotegraph.dump_text import clean_dump_text, is_blocked_source

PHASES = ("A", "B", "C", "D", "E")
PHASE_RANGES = {
    "A": ("2008-09-01", "2010-07-13"),
    "B": ("2010-07-14", "2010-07-26"),
    "C": ("2010-07-27", "2013-04-28"),
    "D": ("2013-04-29", "2014-05-21"),
    "E": ("2014-05-22", "2020-04-30"),
}


def phase_from_date(date_str: str) -> str | None:
    day = (date_str or "")[:10]
    if len(day) < 10:
        return None
    for phase, (start, end) in PHASE_RANGES.items():
        if start <= day <= end:
            return phase
    return None


def domain_of(url: str) -> str:
    try:
        return url.split("/")[2]
    except IndexError:
        return ""


def to_batch_row(record: dict) -> dict:
    return {
        "article_id": record["article_id"],
        "phase": record["phase"],
        "title": record.get("title") or "",
        "url": record.get("url") or "",
        "date": record.get("date") or "",
        "source": domain_of(record.get("url") or ""),
        "text": record["text"],
        "text_source": "dump_cleaned",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="annotation/data/batch.jsonl")
    parser.add_argument("--per-phase", type=int, default=100)
    args = parser.parse_args()

    by_phase: dict[str, list[dict]] = defaultdict(list)
    n_loaded = 0
    n_blocked = 0
    with Path(args.input).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_loaded += 1
            if is_blocked_source(rec.get("url") or ""):
                n_blocked += 1
                continue
            dated = phase_from_date(rec.get("date") or "")
            if dated:
                rec["phase"] = dated
            rec["title"] = clean_dump_text(rec.get("title") or "")
            rec["text"] = clean_dump_text(rec.get("content") or "")
            rec["n_quotes"] = len(find_quote_candidates(rec["text"]))
            by_phase[rec["phase"]].append(rec)
    print(f"loaded {n_loaded} sampled records, skipped {n_blocked} blocked domains")

    selected: list[dict] = []
    for phase in PHASES:
        rows = by_phase.get(phase, [])
        rows.sort(
            key=lambda r: (
                0 if r["n_quotes"] > 0 else 1,
                r.get("date") or "",
                r["article_id"],
            )
        )
        keep = rows[: args.per_phase]
        n_with_quotes = sum(1 for r in keep if r["n_quotes"] > 0)
        print(
            f"phase {phase}: {len(rows)} recovered, keeping {len(keep)} "
            f"({n_with_quotes} with paired quotes)"
        )
        selected.extend(keep)

    selected.sort(key=lambda r: (r["phase"], r["date"], r["article_id"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for rec in selected:
            handle.write(json.dumps(to_batch_row(rec), ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} articles to {output}")


if __name__ == "__main__":
    main()
