#!/usr/bin/env python3
"""Retry failed scrapes via URL canonicalization and optional Datastreamer."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quotegraph"))

from canonicalize import MATERIAL, has_material_rewrite, url_variants
from spinn3r import html_gz_path

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
WAYBACK = "https://archive.org/wayback/available"
DS_SEARCH = "https://api.platform.datastreamer.io/api/search"
MAX_BYTES = 2_000_000
_local = threading.local()
_log_lock = threading.Lock()


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = UA
        _local.s = s
    return _local.s


def is_html(content_type: str, body: bytes) -> bool:
    ct = (content_type or "").lower()
    if "html" in ct or "xml" in ct:
        return True
    head = body[:400].lower()
    return b"<html" in head or b"<!doctype" in head or b"<p" in head


def wayback_ts(date: str) -> str:
    digits = "".join(c for c in (date or "") if c.isdigit())
    return (digits + "0101000000")[:14]


def to_id_url(snapshot: str) -> str:
    if "/web/" not in snapshot:
        return snapshot
    prefix, rest = snapshot.split("/web/", 1)
    ts, _, orig = rest.partition("/")
    ts = ts.replace("id_", "")
    return f"{prefix}/web/{ts}id_/{orig}"


def fetch_url(url: str, timeout: int = 15) -> str | None:
    try:
        resp = session().get(url, timeout=timeout, allow_redirects=True, stream=True)
        raw = resp.raw.read(MAX_BYTES + 1, decode_content=True)
        ctype = resp.headers.get("Content-Type") or ""
        enc = resp.encoding or "utf-8"
        code = resp.status_code
        resp.close()
        if code != 200 or len(raw) > MAX_BYTES or not is_html(ctype, raw):
            return None
        return raw.decode(enc, errors="replace")
    except Exception:
        return None


def fetch_wayback(url: str, date: str) -> str | None:
    for attempt in range(3):
        try:
            lookup = session().get(
                WAYBACK,
                params={"url": url, "timestamp": wayback_ts(date)},
                timeout=15,
            )
            if lookup.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if lookup.status_code != 200:
                return None
            snap = ((lookup.json() or {}).get("archived_snapshots") or {}).get("closest") or {}
            if not snap.get("available"):
                return None
            return fetch_url(to_id_url(snap.get("url") or ""), timeout=20)
        except Exception:
            time.sleep(3)
    return None


def fetch_datastreamer(url: str, date: str, api_key: str) -> str | None:
    day = (date or "")[:10] or "2008-09-01"
    query = f'source.link:"{url}" AND doc_date:[{day} TO {day}]'
    try:
        resp = session().post(
            DS_SEARCH,
            headers={"apikey": api_key, "content-type": "application/json"},
            json={
                "query": {
                    "from": 0,
                    "size": 1,
                    "query": query,
                    "data_sources": ["wsl_news", "opoint_news"],
                }
            },
            timeout=20,
        )
        if resp.status_code != 200:
            return None
        results = (resp.json() or {}).get("results") or []
        if not results:
            return None
        body = ((results[0].get("content") or {}).get("body")) or ""
        title = ((results[0].get("content") or {}).get("title")) or ""
        if not body:
            return None
        return f"<html><head><title>{title}</title></head><body>{body}</body></html>"
    except Exception:
        return None


def open_jsonl(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def load_fail_rows(sample: Path, scrape_log: Path) -> list[dict]:
    fail_ids = set()
    for line in scrape_log.open():
        rec = json.loads(line)
        if rec.get("status") == "fail":
            fail_ids.add(rec["article_id"])
    rows = []
    for line in open_jsonl(sample):
        rec = json.loads(line)
        if rec.get("article_id") in fail_ids:
            rows.append(
                {k: rec.get(k) or "" for k in ("article_id", "url", "date", "phase")}
            )
    return rows


def recover_one(rec: dict, html_dir: Path, api_key: str, rewritten_only: bool) -> dict:
    dest = html_gz_path(html_dir, rec["article_id"])
    fail = dest.with_suffix(".fail")
    if dest.exists():
        return {"article_id": rec["article_id"], "status": "exists"}
    variants = url_variants(rec["url"])
    if rewritten_only:
        variants = [(lab, u) for lab, u in variants if lab != "orig"]
        variants.sort(key=lambda item: (0 if item[0] in MATERIAL else 1))
        variants = variants[:4]
    if not variants:
        return {"article_id": rec["article_id"], "status": "no_variant"}
    for label, url in variants:
        html = fetch_url(url)
        source = f"live:{label}"
        if html is None:
            html = fetch_wayback(url, rec["date"])
            source = f"wayback:{label}"
        if html is None and api_key:
            html = fetch_datastreamer(url, rec["date"], api_key)
            source = f"datastreamer:{label}"
        if html:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(gzip.compress(html.encode("utf-8", errors="replace")))
            if fail.exists():
                fail.unlink()
            return {
                "article_id": rec["article_id"],
                "status": "ok",
                "source": source,
                "url": url,
            }
    return {"article_id": rec["article_id"], "status": "fail"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="/home/mculjak/datasets/quotegraph_poc/sample.jsonl.gz")
    parser.add_argument("--out", default="/home/mculjak/datasets/quotegraph_poc")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rewritten-only", action="store_true", default=True)
    parser.add_argument("--all-fails", action="store_true")
    parser.add_argument("--datastreamer-key", default=os.environ.get("DATASTREAMER_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.all_fails:
        args.rewritten_only = False
    out = Path(args.out)
    html_dir = out / "html"
    log_path = out / "recover.jsonl"
    print("loading fail rows...", flush=True)
    rows = load_fail_rows(Path(args.sample), out / "scrape.jsonl")
    if args.rewritten_only:
        rows = [r for r in rows if has_material_rewrite(r["url"])]
    rows.sort(key=lambda r: ({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(r["phase"], 9), r["date"]))
    if args.limit:
        rows = rows[: args.limit]
    print(
        f"recover {len(rows)} fails rewritten_only={args.rewritten_only} "
        f"datastreamer={bool(args.datastreamer_key)} threads={args.threads}",
        flush=True,
    )
    n_ok = n_fail = n_skip = 0
    with ThreadPoolExecutor(max_workers=args.threads) as pool, log_path.open("a", encoding="utf-8") as log:
        in_flight: set = set()
        done_n = 0
        for rec in rows:
            in_flight.add(
                pool.submit(recover_one, rec, html_dir, args.datastreamer_key, args.rewritten_only)
            )
            if len(in_flight) >= args.threads * 8:
                finished = next(as_completed(in_flight))
                in_flight.remove(finished)
                done_n += 1
                result = finished.result()
                status = result["status"]
                n_ok += status == "ok"
                n_skip += status in {"exists", "no_variant"}
                n_fail += status == "fail"
                with _log_lock:
                    log.write(json.dumps(result, ensure_ascii=False) + "\n")
                    if done_n % 100 == 0:
                        log.flush()
                if done_n % 100 == 0:
                    print(f"  done={done_n}/{len(rows)} ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)
        for finished in as_completed(in_flight):
            done_n += 1
            result = finished.result()
            status = result["status"]
            n_ok += status == "ok"
            n_skip += status in {"exists", "no_variant"}
            n_fail += status == "fail"
            with _log_lock:
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
        log.flush()
    print(f"finished ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)


if __name__ == "__main__":
    main()
