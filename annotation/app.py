"""Local annotation server for Quotegraph gold.

Run from the repo root:

    python -m uvicorn annotation.app:app --reload --port 8765
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from annotation.schema import article_is_complete, quote_is_complete, validation_errors
from annotation.store import (
    BATCH_PATH,
    article_status,
    load_annotation,
    load_batch,
    merge_saved,
    save_annotation,
)
from annotation.wikidata import search_entities

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Quotegraph annotation")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_ARTICLES: list[dict] | None = None


def reset_articles() -> None:
    global _ARTICLES
    _ARTICLES = None


def articles() -> list[dict]:
    global _ARTICLES
    if _ARTICLES is None:
        _ARTICLES = load_batch()
    return _ARTICLES


@app.post("/api/reload")
def reload_batch() -> dict:
    reset_articles()
    return {"n_articles": len(articles()), "batch_exists": BATCH_PATH.exists()}


def _index() -> dict[str, dict]:
    return {row["article_id"]: row for row in articles()}


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status() -> dict:
    return {
        "n_articles": len(articles()),
        "batch": str(BATCH_PATH),
        "batch_exists": BATCH_PATH.exists(),
    }


@app.get("/api/batch")
def batch(annotator: str = Query("default")) -> list[dict]:
    rows = []
    for article in articles():
        record = merge_saved(article, load_annotation(annotator, article["article_id"]))
        quotes = record.get("quotes") or []
        rows.append(
            {
                "article_id": record["article_id"],
                "title": record.get("title") or record["article_id"],
                "date": record.get("date") or "",
                "phase": record.get("phase") or "",
                "n_quotes": len(quotes),
                "n_done": sum(1 for q in quotes if quote_is_complete(q)),
                "status": article_status(record),
                "complete": article_is_complete(record),
            }
        )
    return rows


@app.get("/api/articles/{article_id}")
def get_article(article_id: str, annotator: str = Query("default")) -> dict:
    base = _index().get(article_id)
    if not base:
        raise HTTPException(404, f"Unknown article {article_id}")
    return merge_saved(base, load_annotation(annotator, article_id))


@app.put("/api/articles/{article_id}")
def put_article(article_id: str, payload: dict, annotator: str = Query("default")) -> dict:
    base = _index().get(article_id)
    if not base:
        raise HTTPException(404, f"Unknown article {article_id}")
    payload["article_id"] = article_id
    payload["text"] = base["text"]
    payload["text_sha256"] = base["text_sha256"]
    errors = validation_errors(payload)
    if errors:
        raise HTTPException(400, "; ".join(errors))
    saved = save_annotation(annotator, payload)
    return {
        "ok": True,
        "complete": article_is_complete(saved),
        "status": article_status(saved),
    }


@app.get("/api/wikidata")
def wikidata(q: str = Query(..., min_length=1)) -> list[dict]:
    try:
        return search_entities(q)
    except Exception as exc:  # noqa: BLE001 — surface search failures to the UI
        raise HTTPException(502, f"Wikidata search failed: {exc}") from exc
