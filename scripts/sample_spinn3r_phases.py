#!/usr/bin/env python3
"""Reservoir-sample Spinn3r/Quotebank input shards on jadranka.

Record tags (one article per blank-line-separated block):
    I  article id
    V  Quotebank encoding phase (A–E)
    U  URL
    D  datetime
    T  title
    C  PTB-tokenized body
    X  leftover name string from the converter (ignored)

Phases (Vaucher et al. / Quotebank phases.md):
    A  until 2010-07-13
    B  2010-07-14 to 2010-07-26
    C  2010-07-27 to 2013-04-28
    D  2013-04-29 to 2014-05-21
    E  since 2014-05-22
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cramjam

PHASES = ("A", "B", "C", "D", "E")
QUOTE_MARKS = (
    '"',
    "``",
    "''",
    "\u201c",
    "\u201d",
    "\u00ab",
    "\u00bb",
    "<blockquote>",
    "\\blockquote",
)
BLOCKED_DOMAINS = (
    "wordpress.com",
    "wikia.com",
    "gamereactor.eu",
    "myspace.com",
    "typepad.com",
)


def is_blocked_source(url: str) -> bool:
    host = url.split("/")[2].lower() if "://" in url else url.lower()
    if host.startswith("www."):
        host = host[4:]
    return any(host == domain or host.endswith("." + domain) for domain in BLOCKED_DOMAINS)


def iter_snappy_blocks(path: Path):
    with path.open("rb") as handle:
        while True:
            header = handle.read(4)
            if len(header) < 4:
                return
            uncompressed_len = int.from_bytes(header, "big")
            if uncompressed_len <= 0 or uncompressed_len > 20_000_000:
                return
            parts: list[bytes] = []
            got = 0
            while got < uncompressed_len:
                clen_bytes = handle.read(4)
                if len(clen_bytes) < 4:
                    return
                compressed_len = int.from_bytes(clen_bytes, "big")
                if compressed_len <= 0 or compressed_len > 20_000_000:
                    return
                blob = handle.read(compressed_len)
                if len(blob) < compressed_len:
                    return
                chunk = bytes(cramjam.snappy.decompress_raw(blob))
                parts.append(chunk)
                got += len(chunk)
            yield b"".join(parts)


def iter_records(path: Path):
    buffer = ""
    for block in iter_snappy_blocks(path):
        buffer += block.decode("utf-8", errors="replace")
        while "\n\n" in buffer:
            raw, buffer = buffer.split("\n\n", 1)
            record = parse_record(raw)
            if record:
                yield record


def parse_record(raw: str) -> dict | None:
    fields: dict[str, str] = {}
    for line in raw.split("\n"):
        if not line:
            continue
        tag, _, rest = line.partition("\t")
        if len(tag) == 1:
            fields[tag] = rest
    article_id = fields.get("I") or ""
    phase = fields.get("V") or ""
    url = fields.get("U") or ""
    content = fields.get("C") or ""
    if not article_id or phase not in PHASES:
        return None
    if not url.startswith("http"):
        return None
    if len(content) < 400:
        return None
    if not any(mark in content for mark in QUOTE_MARKS):
        return None
    if is_blocked_source(url):
        return None
    return {
        "article_id": article_id,
        "phase": phase,
        "url": url,
        "date": fields.get("D") or "",
        "title": fields.get("T") or "",
        "content": content,
    }


def reservoir_sample(paths: list[Path], per_phase: int, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {phase: [] for phase in PHASES}
    seen: dict[str, int] = {phase: 0 for phase in PHASES}
    ids: set[str] = set()
    n_kept = 0
    n_seen = 0
    for path in paths:
        print(f"scanning {path} ({path.stat().st_size / 1e9:.2f} GB)", flush=True)
        for record in iter_records(path):
            n_seen += 1
            if record["article_id"] in ids:
                continue
            phase = record["phase"]
            seen[phase] += 1
            bucket = buckets[phase]
            if len(bucket) < per_phase:
                bucket.append(record)
                ids.add(record["article_id"])
                n_kept += 1
            else:
                index = rng.randrange(seen[phase])
                if index < per_phase:
                    ids.discard(bucket[index]["article_id"])
                    bucket[index] = record
                    ids.add(record["article_id"])
            if n_seen % 100_000 == 0:
                filled = {p: len(buckets[p]) for p in PHASES}
                print(f"  seen={n_seen} kept={n_kept} filled={filled} counts={seen}", flush=True)
    filled = {p: len(buckets[p]) for p in PHASES}
    print(f"done seen={n_seen} phase_counts={seen} filled={filled}", flush=True)
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/shared/lovorka/spinn3r_for_quotebank")
    parser.add_argument("--per-phase", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    paths = [
        root / "sep2008-sep2018" / "part-r-00000.snappy",
        root / "oct2018-apr2020" / "part-r-00000.snappy",
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f"missing shards: {missing}")
    buckets = reservoir_sample(paths, args.per_phase, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output.open("w", encoding="utf-8") as handle:
        for phase in PHASES:
            for record in buckets[phase]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} records to {output}")
    for phase in PHASES:
        dates = sorted(r["date"] for r in buckets[phase] if r["date"])
        print(f"  {phase}: {len(buckets[phase])}  {dates[0][:10] if dates else '-'} .. {dates[-1][:10] if dates else '-'}")


if __name__ == "__main__":
    main()
