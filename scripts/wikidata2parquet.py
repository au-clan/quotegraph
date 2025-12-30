import os
import argparse
import logging
import multiprocessing
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import orjson
    def json_loads(s):
        return orjson.loads(s)
except ImportError:
    import json
    def json_loads(s):
        return json.loads(s)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

CHUNK_SIZE = 256 * 1024 * 1024


def has_pigz():
    """Check if pigz is available for parallel decompression."""
    return shutil.which("pigz") is not None


def parse_and_write_chunk(chunk_data, chunk_idx, output_dir, leftover_prefix=b""):
    """Parse a chunk of data and write to parquet.
    
    Args:
        chunk_data: Raw bytes containing JSON lines.
        chunk_idx: Index of this chunk.
        output_dir: Output directory for parquet files.
        leftover_prefix: Incomplete line from previous chunk.
    
    Returns:
        Tuple of (leftover_suffix, num_records, output_path).
    """
    try:
        data = leftover_prefix + chunk_data
        
        last_newline = data.rfind(b"\n")
        if last_newline == -1:
            return data, 0, None
        
        complete_data = data[:last_newline + 1]
        leftover = data[last_newline + 1:]
        
        records = []
        for line in complete_data.split(b"\n"):
            line = line.strip()
            if not line or line in (b"[", b"]", b","):
                continue
            line = line.rstrip(b",")
            if line:
                try:
                    records.append(json_loads(line))
                except (ValueError, TypeError):
                    pass
        
        if not records:
            return leftover, 0, None
        
        table = pa.Table.from_pylist(records)
        part_path = os.path.join(output_dir, f"part-{chunk_idx:05d}.parquet")
        pq.write_table(table, part_path, compression="snappy")
        
        return leftover, len(records), part_path
        
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_idx}: {e}")
        traceback.print_exc()
        return b"", 0, None


def chunk_reader_thread(stream, chunk_queue, chunk_size, pbar):
    """Thread that reads chunks from stream and puts them in queue."""
    chunk_idx = 0
    leftover = b""
    
    while True:
        data = stream.read(chunk_size)
        if not data:
            if leftover:
                chunk_queue.put((chunk_idx, leftover, True))
            chunk_queue.put(None)  # Sentinel
            break
        
        chunk_queue.put((chunk_idx, leftover + data, False))
        leftover = b""
        chunk_idx += 1
        
        pbar.update(len(data) // 8)


def gz_json_to_parquet_parallel(source_gz, dest_parquet, n_decompress, n_workers, chunk_size):
    """Convert gzip-compressed JSONL to Parquet with full parallelism.

    Uses pigz for parallel decompression, reads large chunks, and processes
    them in parallel with multiple workers.

    Args:
        source_gz: Path to the input gzip-compressed JSON lines file.
        dest_parquet: Output directory path for parquet part files.
        n_decompress: Number of threads for decompression (pigz).
        n_workers: Number of parallel workers for processing.
        chunk_size: Size of chunks to read and process.
    """
    logging.info(f"Converting '{source_gz}' to '{dest_parquet}'")
    logging.info(f"Decompress threads: {n_decompress}, Workers: {n_workers}, Chunk: {chunk_size // (1024*1024)}MB")

    os.makedirs(dest_parquet, exist_ok=True)
    compressed_size = os.path.getsize(source_gz)

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
                    process_chunk_data, to_process, chunk_idx, dest_parquet
                )
                futures[future] = chunk_idx
                chunk_idx += 1
                
                pbar.update(len(data) // 8)
                
                done = [f for f in futures if f.done()]
                for f in done:
                    try:
                        n_recs = f.result()
                        total_records += n_recs
                        total_chunks += 1
                    except Exception as e:
                        logging.error(f"Chunk failed: {e}")
                    del futures[f]
                
                while len(futures) >= n_workers * 2:
                    f = next(as_completed(futures))
                    try:
                        n_recs = f.result()
                        total_records += n_recs
                        total_chunks += 1
                    except Exception as e:
                        logging.error(f"Chunk failed: {e}")
                    del futures[f]
            
            if leftover.strip():
                future = executor.submit(
                    process_chunk_data, leftover, chunk_idx, dest_parquet
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

    logging.info(f"Done! Wrote {total_records:,} records to {total_chunks} parquet files.")
    logging.info(f"Output: {dest_parquet}/")


def process_chunk_data(data, chunk_idx, output_dir):
    """Process a chunk of complete JSON lines and write parquet.
    
    This runs in a worker process.
    """
    records = []
    for line in data.split(b"\n"):
        line = line.strip()
        if not line or line in (b"[", b"]", b","):
            continue
        line = line.rstrip(b",")
        if line:
            try:
                records.append(json_loads(line))
            except (ValueError, TypeError):
                pass
    
    if not records:
        return 0
    
    table = pa.Table.from_pylist(records)
    part_path = os.path.join(output_dir, f"part-{chunk_idx:05d}.parquet")
    pq.write_table(table, part_path, compression="snappy")
    
    return len(records)


if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    
    parser = argparse.ArgumentParser(
        description="Convert gzip JSONL to Parquet (fully parallel)."
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
        help=f"Workers for JSON parsing + parquet writing (default: {cpu_count // 2})",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Chunk size in bytes (default: {CHUNK_SIZE // (1024*1024)}MB)",
    )
    args = parser.parse_args()

    gz_json_to_parquet_parallel(
        args.input_gz, 
        args.output_parquet, 
        args.n_decompress,
        args.n_workers, 
        args.chunk_size
    )
