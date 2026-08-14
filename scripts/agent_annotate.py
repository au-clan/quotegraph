#!/usr/bin/env python3
"""Annotate the gold batch with an isolated LLM agent per article.

Saves under ``annotation/data/annotations/<annotator>/`` (default: ``agent``).
Refuses to write to the human annotator ``default``.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from annotation.agent import (
    AGENT_ANNOTATOR,
    DEFAULT_MODEL,
    HUMAN_ANNOTATOR,
    annotate_article,
    make_openai_llm,
    prompt_payload,
    save_agent_prompts,
)
from annotation.schema import article_is_complete
from annotation.store import load_annotation, load_batch, prompt_path, save_annotation


def _should_skip(annotator: str, article_id: str, *, force: bool) -> bool:
    if force:
        return False
    saved = load_annotation(annotator, article_id)
    return bool(saved) and article_is_complete(saved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", default=AGENT_ANNOTATOR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--article-id", action="append", default=[])
    parser.add_argument(
        "--backfill-prompts",
        action="store_true",
        help="Write prompt sidecars for existing agent JSON without calling the LLM.",
    )
    args = parser.parse_args()

    if args.annotator == HUMAN_ANNOTATOR:
        raise SystemExit(
            f"Refusing to write to annotator {HUMAN_ANNOTATOR!r}. "
            f"Use --annotator {AGENT_ANNOTATOR} (or another name)."
        )

    articles = load_batch()
    if args.article_id:
        wanted = set(args.article_id)
        articles = [row for row in articles if row["article_id"] in wanted]
    if args.limit:
        articles = articles[: args.limit]

    if args.backfill_prompts:
        written = 0
        for article in articles:
            dest = prompt_path(args.annotator, article["article_id"])
            saved = load_annotation(args.annotator, article["article_id"])
            if not saved:
                continue
            if dest.exists() and not args.force:
                continue
            payload = prompt_payload(article, model=saved.get("agent_model") or args.model)
            save_agent_prompts(args.annotator, payload)
            written += 1
        print(f"wrote {written} prompt sidecars under annotator={args.annotator}")
        return

    todo = [row for row in articles if not _should_skip(args.annotator, row["article_id"], force=args.force)]
    print(
        f"annotator={args.annotator} model={args.model} "
        f"batch={len(articles)} todo={len(todo)} workers={args.workers}"
    )
    if not todo:
        return

    llm = make_openai_llm(args.model)

    def run_one(article: dict) -> tuple[str, str, int, int]:
        record = annotate_article(article, llm=llm)
        record["annotator"] = args.annotator
        record["agent_model"] = args.model
        prompts = record.pop("agent_prompts", None) or prompt_payload(article, model=args.model)
        prompts["model"] = args.model
        save_agent_prompts(args.annotator, prompts)
        save_annotation(args.annotator, record)
        n_keep = sum(1 for q in record["quotes"] if q.get("status") == "keep")
        n_rej = sum(1 for q in record["quotes"] if q.get("status") == "reject")
        status = "complete" if article_is_complete(record) else "incomplete"
        return record["article_id"], status, n_keep, n_rej

    errors = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, article): article["article_id"] for article in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc="articles"):
            article_id = futures[future]
            try:
                aid, status, n_keep, n_rej = future.result()
                tqdm.write(f"{aid} {status} keep={n_keep} reject={n_rej}")
            except Exception as exc:  # noqa: BLE001 — keep the batch moving
                errors += 1
                tqdm.write(f"{article_id} ERROR {exc}")
    if errors:
        raise SystemExit(f"{errors} article(s) failed")
    print(f"done annotator={args.annotator} wrote={len(todo)}")


if __name__ == "__main__":
    main()
