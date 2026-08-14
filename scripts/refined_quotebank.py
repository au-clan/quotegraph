#!/usr/bin/env python3
"""Run ReFinED entity linking on Quotebank parquet splits.

This script is intentionally scheduler-agnostic. HPC submit scripts create a
manifest for the split they own, then array tasks call ``run-chunk`` with their
chunk id.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


LOG = logging.getLogger("refined_quotebank")


OUTPUT_SCHEMA = pa.schema(
    [
        ("articleID", pa.string()),
        ("url", pa.string()),
        ("phase", pa.string()),
        ("mention_text", pa.string()),
        ("start", pa.int64()),
        ("end", pa.int64()),
        ("wikidata_qid", pa.string()),
        ("wikipedia_title", pa.string()),
        ("coarse_type", pa.string()),
        ("confidence", pa.float64()),
        ("chunk_id", pa.int32()),
        ("model_name", pa.string()),
        ("entity_set", pa.string()),
    ]
)


@dataclass(frozen=True)
class RowGroupUnit:
    path: str
    row_group: int
    start: int
    rows: int
    chunk_id: int


def parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(
        p
        for p in path.rglob("*.parquet")
        if not p.name.startswith("_") and not p.name.startswith(".")
    )
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return files


def scan_row_groups(input_path: Path) -> list[tuple[str, int, int]]:
    units: list[tuple[str, int, int]] = []
    for file_path in parquet_files(input_path):
        parquet = pq.ParquetFile(file_path)
        for row_group in range(parquet.num_row_groups):
            rows = parquet.metadata.row_group(row_group).num_rows
            if rows > 0:
                units.append((str(file_path), row_group, rows))
    if not units:
        raise ValueError(f"No non-empty row groups found in {input_path}")
    return units


def build_manifest(input_path: Path, output_path: Path, chunks: int) -> dict[str, int]:
    if chunks <= 0:
        raise ValueError("--chunks must be positive")

    row_groups = scan_row_groups(input_path)
    total_rows = sum(rows for _, _, rows in row_groups)
    target_rows = math.ceil(total_rows / chunks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments = 0
    with output_path.open("w", encoding="utf-8") as handle:
        global_start = 0
        for path, row_group, rows in row_groups:
            global_end = global_start + rows
            first_chunk = min(global_start // target_rows, chunks - 1)
            last_chunk = min((global_end - 1) // target_rows, chunks - 1)

            for chunk_id in range(first_chunk, last_chunk + 1):
                chunk_start = chunk_id * target_rows
                chunk_end = min((chunk_id + 1) * target_rows, total_rows)
                overlap_start = max(global_start, chunk_start)
                overlap_end = min(global_end, chunk_end)
                if overlap_start >= overlap_end:
                    continue
                entry = {
                    "chunk_id": chunk_id,
                    "path": path,
                    "row_group": row_group,
                    "start": overlap_start - global_start,
                    "rows": overlap_end - overlap_start,
                }
                handle.write(json.dumps(entry, separators=(",", ":")) + "\n")
                assignments += 1

            global_start = global_end

    return {"chunks": chunks, "total_rows": total_rows, "target_rows": target_rows, "assignments": assignments}


def load_manifest_chunk(manifest_path: Path, chunk_id: int) -> list[RowGroupUnit]:
    units: list[RowGroupUnit] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if int(entry["chunk_id"]) != chunk_id:
                continue
            units.append(
                RowGroupUnit(
                    path=str(entry["path"]),
                    row_group=int(entry["row_group"]),
                    start=int(entry["start"]),
                    rows=int(entry["rows"]),
                    chunk_id=chunk_id,
                )
            )
    return units


def as_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def nested_get(obj: Any, names: Iterable[str]) -> Any:
    for name in names:
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(name)
        else:
            obj = getattr(obj, name, None)
    return obj


def first_attr(obj: Any, names: Iterable[str]) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def span_to_row(
    *,
    span: Any,
    article_id: str | None,
    url: str | None,
    phase: str | None,
    chunk_id: int,
    model_name: str,
    entity_set: str,
) -> dict[str, Any] | None:
    if isinstance(span, (list, tuple)):
        mention = span[0] if len(span) > 0 else None
        entity = span[1] if len(span) > 1 else None
        coarse_type = span[2] if len(span) > 2 else None
    else:
        mention = first_attr(span, ("text", "mention", "surface_form"))
        entity = first_attr(span, ("predicted_entity", "entity", "linked_entity"))
        coarse_type = first_attr(span, ("coarse_mention_type", "coarse_type", "type"))

    qid = first_attr(entity, ("wikidata_entity_id", "wikidata_id", "qid"))
    title = first_attr(entity, ("wikipedia_entity_title", "wikipedia_title", "title"))

    if qid in (None, "", "None") and title in (None, "", "None"):
        return None

    start = first_attr(span, ("start", "start_char", "char_start"))
    end = first_attr(span, ("end", "end_char", "char_end"))
    confidence = first_attr(span, ("confidence", "score", "probability"))

    return {
        "articleID": article_id,
        "url": url,
        "phase": phase,
        "mention_text": as_string(mention),
        "start": int(start) if start is not None else None,
        "end": int(end) if end is not None else None,
        "wikidata_qid": as_string(qid),
        "wikipedia_title": as_string(title),
        "coarse_type": as_string(coarse_type),
        "confidence": float(confidence) if confidence is not None else None,
        "chunk_id": chunk_id,
        "model_name": model_name,
        "entity_set": entity_set,
    }


def load_refined(model_name: str, entity_set: str, device: str):
    from refined.inference.processor import Refined

    kwargs = {"model_name": model_name, "entity_set": entity_set}
    try:
        return Refined.from_pretrained(**kwargs, device=device)
    except TypeError:
        model = Refined.from_pretrained(**kwargs)
        if hasattr(model, "to"):
            model.to(device)
        return model


def process_text(model: Any, text: str) -> list[Any]:
    result = model.process_text(text)
    if isinstance(result, tuple):
        result = result[0]
    if result is None:
        return []
    return list(result)


def write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA) if rows else pa.Table.from_pylist([], schema=OUTPUT_SCHEMA)
    pq.write_table(table, output_path, compression="zstd")


def table_batches(units: list[RowGroupUnit], columns: list[str]) -> Iterable[pa.Table]:
    parquet_cache: dict[str, pq.ParquetFile] = {}
    for unit in units:
        parquet = parquet_cache.get(unit.path)
        if parquet is None:
            parquet = pq.ParquetFile(unit.path)
            parquet_cache[unit.path] = parquet
        table = parquet.read_row_group(unit.row_group, columns=columns)
        yield table.slice(unit.start, unit.rows)


def run_chunk(args: argparse.Namespace) -> None:
    units = load_manifest_chunk(args.manifest, args.chunk_id)
    if not units:
        LOG.warning("No manifest entries for chunk_id=%s; writing empty output", args.chunk_id)
        write_rows([], args.output_dir / f"part-{args.chunk_id:05d}.parquet")
        return

    required_columns = [args.id_col, args.content_col]
    optional_columns = [args.url_col, args.phase_col]
    first_file = pq.ParquetFile(units[0].path)
    schema_names = set(first_file.schema_arrow.names)
    columns = [c for c in required_columns + optional_columns if c and c in schema_names]
    missing = [c for c in required_columns if c not in schema_names]
    if missing:
        raise ValueError(f"Missing required columns in {units[0].path}: {missing}")

    model = load_refined(args.model_name, args.entity_set, args.device)
    rows: list[dict[str, Any]] = []
    processed = 0

    for table in table_batches(units, columns):
        records = table.to_pylist()
        for record in records:
            if args.max_articles is not None and processed >= args.max_articles:
                break
            text = record.get(args.content_col)
            if not text:
                continue
            article_id = as_string(record.get(args.id_col))
            url = as_string(record.get(args.url_col)) if args.url_col in record else None
            phase = as_string(record.get(args.phase_col)) if args.phase_col in record else None

            try:
                spans = process_text(model, str(text))
            except Exception:
                LOG.exception("ReFinED failed for articleID=%s", article_id)
                spans = []

            for span in spans:
                row = span_to_row(
                    span=span,
                    article_id=article_id,
                    url=url,
                    phase=phase,
                    chunk_id=args.chunk_id,
                    model_name=args.model_name,
                    entity_set=args.entity_set,
                )
                if row is not None:
                    rows.append(row)
            processed += 1

        if args.max_articles is not None and processed >= args.max_articles:
            break

    output_path = args.output_dir / f"part-{args.chunk_id:05d}.parquet"
    write_rows(rows, output_path)
    LOG.info("Processed %s articles, wrote %s spans to %s", processed, len(rows), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Create a chunk manifest for a parquet split")
    manifest.add_argument("--input", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--chunks", type=int, required=True)

    run = subparsers.add_parser("run-chunk", help="Run ReFinED for one manifest chunk")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--chunk-id", type=int, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--model-name", default=os.environ.get("REFINED_MODEL", "wikipedia_model_with_numbers"))
    run.add_argument("--entity-set", default=os.environ.get("REFINED_ENTITY_SET", "wikipedia"))
    run.add_argument("--device", default=os.environ.get("REFINED_DEVICE", "cuda:0"))
    run.add_argument("--id-col", default=os.environ.get("REFINED_ID_COL", "articleID"))
    run.add_argument("--content-col", default=os.environ.get("REFINED_CONTENT_COL", "content"))
    run.add_argument("--url-col", default=os.environ.get("REFINED_URL_COL", "url"))
    run.add_argument("--phase-col", default=os.environ.get("REFINED_PHASE_COL", "phase"))
    run.add_argument("--max-articles", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "manifest":
        summary = build_manifest(args.input, args.output, args.chunks)
        LOG.info("Manifest written to %s: %s", args.output, summary)
    elif args.command == "run-chunk":
        run_chunk(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
