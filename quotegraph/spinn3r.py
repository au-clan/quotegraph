"""Read Quootstrap/Spinn3r Hadoop-Snappy article shards.

Record tags (blank-line-separated):
    I  article id
    V  dump encoding phase (A–E); prefer date for Quotebank phases
    U  URL
    D  datetime
    T  title
    C  Stanford-PTB body
    X  leftover name string (ignored)
"""

from __future__ import annotations

from pathlib import Path

import cramjam

PHASES = ("A", "B", "C", "D", "E")
PHASE_RANGES = {
    "A": ("2008-09-01", "2010-07-13"),
    "B": ("2010-07-14", "2010-07-26"),
    "C": ("2010-07-27", "2013-04-28"),
    "D": ("2013-04-29", "2014-05-21"),
    "E": ("2014-05-22", "2020-04-30"),
}
DUMP_ROOT = Path("/shared/lovorka/spinn3r_for_quotebank")


def phase_from_date(date_str: str) -> str | None:
    day = (date_str or "")[:10]
    if len(day) < 10:
        return None
    for phase, (start, end) in PHASE_RANGES.items():
        if start <= day <= end:
            return phase
    return None


def shard_paths(root: Path = DUMP_ROOT) -> list[Path]:
    paths: list[Path] = []
    for folder in ("sep2008-sep2018", "oct2018-apr2020"):
        paths.extend(sorted((root / folder).glob("part-r-*.snappy")))
    return paths


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
                parts.append(bytes(cramjam.snappy.decompress_raw(blob)))
                got += len(parts[-1])
            yield b"".join(parts)


def iter_field_dicts(path: Path):
    buffer = ""
    for block in iter_snappy_blocks(path):
        buffer += block.decode("utf-8", errors="replace")
        while "\n\n" in buffer:
            raw, buffer = buffer.split("\n\n", 1)
            fields: dict[str, str] = {}
            for line in raw.split("\n"):
                if not line:
                    continue
                tag, _, rest = line.partition("\t")
                if len(tag) == 1:
                    fields[tag] = rest
            if fields:
                yield fields


def html_gz_path(html_dir: Path, article_id: str) -> Path:
    return html_dir / article_id[:4] / f"{article_id}.html.gz"
