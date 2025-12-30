import os
import argparse
import logging
import multiprocessing
import subprocess
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

CHUNK_SIZE = 256 * 1024 * 1024

SCHEMA = pa.schema([("instance", pa.string())])


def has_pigz():
    """Check if pigz is available for parallel decompression."""
    return shutil.which("pigz") is not None


def process_chunk_data(data, chunk_idx, output_dir, file_uuid):
    """Process a chunk of lines and write to parquet as raw strings.

    Each JSON line is stored as-is in the "instance" column (no parsing).

    Args:
        data: Raw bytes containing complete lines.
        chunk_idx: Index of this chunk.
        output_dir: Output directory for parquet files.
        file_uuid: UUID for this job (Spark-style naming).

    Returns:
        Number of records written.
    """
    instances = []
    for line in data.split(b"\n"):
        line = line.strip()
        if not line or line in (b"[", b"]", b","):
            continue
        # Remove trailing comma if present (Wikidata dump format)
        line = line.rstrip(b",")
        if line:
            # Store as string without parsing
            instances.append(line.decode("utf-8"))

    if not instances:
        return 0

    table = pa.table({"instance": instances}, schema=SCHEMA)

    part_path = os.path.join(
        output_dir,
        f"part-{chunk_idx:05d}-{file_uuid}.snappy.parquet"
    )
    pq.write_table(table, part_path, compression="snappy")

    return len(instances)


def gz_to_parquet_parallel(source_gz, dest_parquet, n_decompress, n_workers, chunk_size):
    """Convert gzip-compressed JSONL to Parquet with raw JSON strings.

    Stores each JSON line as a raw string in the "instance" column.
    No JSON parsing is performed - much faster and avoids schema issues.

    Args:
        source_gz: Path to the input gzip-compressed file.
        dest_parquet: Output directory path for parquet part files.
        n_decompress: Number of threads for decompression (pigz).
        n_workers: Number of parallel workers for processing.
        chunk_size: Size of chunks to read and process.
    """
    logging.info(f"Converting '{source_gz}' to '{dest_parquet}'")
    logging.info("Output schema: single column 'instance' (raw JSON strings)")
    logging.info(f"Decompress threads: {n_decompress}, Workers: {n_workers}, Chunk: {chunk_size // (1024*1024)}MB")

    os.makedirs(dest_parquet, exist_ok=True)
    compressed_size = os.path.getsize(source_gz)

    job_uuid = str(uuid.uuid4())
    logging.info(f"Job UUID: {job_uuid}")

    if has_pigz():
        logging.info(f"Using pigz with {n_decompress} threads")
        proc = subprocess.Popen(
            ["pigz", "-dc", "-p", str(n_decompress), source_gz],
            stdout=subprocess.PIPE,
            bufsize=chunk_size,
        )
        stream = proc.stdout
    else:
        import gzip
        logging.warning("pigz not found, using single-threaded gzip")
        proc = None
        stream = gzip.open(source_gz, "rb")

    total_records = 0
    total_chunks = 0
    futures = {}

    pbar = tqdm(
        total=compressed_size,
        unit="B",
        unit_scale=True,
        desc="Processing",
    )

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            chunk_idx = 0
            leftover = b""

            while True:
                data = stream.read(chunk_size)
                if not data:
                    break

                full_data = leftover + data

                last_newline = full_data.rfind(b"\n")
                if last_newline == -1:
                    leftover = full_data
                    continue

                to_process = full_data[:last_newline + 1]
                leftover = full_data[last_newline + 1:]

                future = executor.submit(
                    process_chunk_data, to_process, chunk_idx, dest_parquet, job_uuid
                )
                futures[future] = chunk_idx
                chunk_idx += 1

                pbar.update(len(data) // 8)

                # Collect completed futures
                done = [f for f in futures if f.done()]
                for f in done:
                    try:
                        n_recs = f.result()
                        total_records += n_recs
                        total_chunks += 1
                    except Exception as e:
                        logging.error(f"Chunk failed: {e}")
                    del futures[f]

                # Backpressure
                while len(futures) >= n_workers * 2:
                    f = next(as_completed(futures))
                    try:
                        n_recs = f.result()
                        total_records += n_recs
                        total_chunks += 1
                    except Exception as e:
                        logging.error(f"Chunk failed: {e}")
                    del futures[f]

            # Process remaining
            if leftover.strip():
                future = executor.submit(
                    process_chunk_data, leftover, chunk_idx, dest_parquet, job_uuid
                )
                futures[future] = chunk_idx

            for f in as_completed(futures):
                try:
                    n_recs = f.result()
                    total_records += n_recs
                    total_chunks += 1
                except Exception as e:
                    logging.error(f"Chunk failed: {e}")

            pbar.update(pbar.total - pbar.n)

    finally:
        pbar.close()
        stream.close()
        if proc:
            proc.wait()

    success_path = os.path.join(dest_parquet, "_SUCCESS")
    with open(success_path, "w") as f:
        pass

    logging.info(f"Done! Wrote {total_records:,} records to {total_chunks} parquet files.")
    logging.info(f"Output: {dest_parquet}/")
    logging.info("Read in PySpark: df = spark.read.parquet('path')")
    logging.info("Parse JSON in Spark: df.select(from_json(col('instance'), schema))")


if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser(
        description="Convert gzip JSONL to Parquet (raw JSON strings, no parsing)."
    )
    parser.add_argument("input_gz", type=str, help="Input .json.gz file")
    parser.add_argument("output_parquet", type=str, help="Output Parquet folder path")
    parser.add_argument(
        "--n_decompress",
        type=int,
        default=cpu_count // 2,
        help=f"Threads for pigz decompression (default: {cpu_count // 2})",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=cpu_count // 2,
        help=f"Workers for parquet writing (default: {cpu_count // 2})",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Chunk size in bytes (default: {CHUNK_SIZE // (1024*1024)}MB)",
    )
    args = parser.parse_args()

    gz_to_parquet_parallel(
        args.input_gz,
        args.output_parquet,
        args.n_decompress,
        args.n_workers,
        args.chunk_size
    )
