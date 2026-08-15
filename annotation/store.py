"""Load article batches and persist per-article annotations."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from annotation.names import nearest_name_candidates
from annotation.quotes import find_quote_candidates, find_unmatched_quote_marks
from annotation.quotatives import QUOTATIVE_CUES, find_quotative_candidates
from annotation.schema import (
    article_is_complete,
    coerce_text_span,
    empty_quotative,
    empty_speaker,
    quote_from_candidate,
    quote_is_touched,
    quote_segments,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BATCH_PATH = DATA_DIR / "batch.jsonl"
ANNOTATIONS_DIR = DATA_DIR / "annotations"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def article_text(raw: dict[str, Any]) -> str:
    for key in ("text", "detokenized_content", "content", "body"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def normalize_article(raw: dict[str, Any], index: int) -> dict[str, Any]:
    text = article_text(raw)
    article_id = str(raw.get("article_id") or raw.get("articleID") or raw.get("id") or f"article-{index:04d}")
    candidates = find_quote_candidates(text)
    quotes = [
        quote_from_candidate(
            quote_id=f"{article_id}-q{i}",
            inner_start=c.inner_start,
            inner_end=c.inner_end,
            outer_start=c.outer_start,
            outer_end=c.outer_end,
            delimiter=c.delimiter,
            text=text,
        )
        for i, c in enumerate(candidates)
    ]
    for quote in quotes:
        attach_suggestions(text, quote)
    return {
        "article_id": article_id,
        "text": text,
        "text_sha256": _sha256(text),
        "title": raw.get("title") or "",
        "url": raw.get("url") or "",
        "date": raw.get("date") or raw.get("date_published") or "",
        "source": raw.get("source") or raw.get("domain") or "",
        "phase": raw.get("phase") or "",
        "text_source": raw.get("text_source") or "",
        "quotes": quotes,
        "unmatched_quotes": [{"offset": m.offset, "kind": m.kind} for m in find_unmatched_quote_marks(text)],
        "quotative_cues": list(QUOTATIVE_CUES),
        "skipped": False,
        "skip_reason": "",
        "notes": "",
        "meta": {k: raw[k] for k in raw if k not in {"text", "detokenized_content", "content", "body", "quotes"}},
    }


def load_batch(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or BATCH_PATH
    if not path.exists():
        return []
    articles: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            articles.append(normalize_article(json.loads(line), i))
    return articles


def annotation_path(annotator: str, article_id: str) -> Path:
    safe_annotator = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in annotator) or "default"
    safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in article_id)
    folder = ANNOTATIONS_DIR / safe_annotator
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{safe_id}.json"


def prompt_path(annotator: str, article_id: str) -> Path:
    """Sidecar for the LLM prompt; not loaded by the annotation UI."""
    return annotation_path(annotator, article_id).with_suffix(".prompt.json")


def prompt_text_path(annotator: str, article_id: str) -> Path:
    return annotation_path(annotator, article_id).with_name(
        annotation_path(annotator, article_id).stem + ".prompt.txt"
    )


def load_annotation(annotator: str, article_id: str) -> dict[str, Any] | None:
    path = annotation_path(annotator, article_id)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_annotation(annotator: str, record: dict[str, Any]) -> dict[str, Any]:
    path = annotation_path(annotator, record["article_id"])
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def _quote_outer_ranges(quote: dict[str, Any]) -> list[tuple[int, int]]:
    segments = quote_segments(quote)
    return [(int(seg["outer_start"]), int(seg["outer_end"])) for seg in segments]


def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return not (left[1] <= right[0] or right[1] <= left[0])


def _next_quote_id(article_id: str, quotes: list[dict[str, Any]]) -> str:
    numbers = [-1]
    for quote in quotes:
        match = re.search(r"q(\d+)$", str(quote.get("id") or ""), re.I)
        if match:
            numbers.append(int(match.group(1)))
    return f"{article_id}-q{max(numbers) + 1}"


def union_saved_quotes(base: dict[str, Any], saved_quotes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep labelled quotes and append newly detected spans that do not overlap."""
    if not saved_quotes:
        return list(base.get("quotes") or [])
    covered = [span for quote in saved_quotes for span in _quote_outer_ranges(quote)]
    combined = list(saved_quotes)
    taken_ids = {str(quote.get("id") or "") for quote in combined}
    for candidate in base.get("quotes") or []:
        if any(_ranges_overlap(span, seen) for span in _quote_outer_ranges(candidate) for seen in covered):
            continue
        extra = dict(candidate)
        if extra["id"] in taken_ids:
            extra["id"] = _next_quote_id(base["article_id"], combined)
        taken_ids.add(extra["id"])
        covered.extend(_quote_outer_ranges(extra))
        combined.append(extra)
    combined.sort(key=lambda quote: (int(quote.get("outer_start") or 0), -int(quote.get("outer_end") or 0)))
    return combined


def merge_saved(base: dict[str, Any], saved: dict[str, Any] | None) -> dict[str, Any]:
    if not saved:
        return base
    if saved.get("text_sha256") != base["text_sha256"]:
        # Text changed (re-extraction). Keep notes but rebuild quotes.
        merged = dict(base)
        merged["notes"] = saved.get("notes") or ""
        merged["skipped"] = saved.get("skipped") or False
        merged["skip_reason"] = saved.get("skip_reason") or ""
        return merged
    merged = dict(base)
    merged["quotes"] = union_saved_quotes(base, saved.get("quotes") or [])
    merged["skipped"] = saved.get("skipped") or False
    merged["skip_reason"] = saved.get("skip_reason") or ""
    merged["notes"] = saved.get("notes") or ""
    for quote in merged.get("quotes") or []:
        coerce_quote(quote)
        attach_suggestions(merged["text"], quote)
    return merged


def coerce_quote(quote: dict[str, Any]) -> dict[str, Any]:
    quote.setdefault("quotative", empty_quotative())
    quotative = quote.get("quotative") or {}
    if quotative.get("status") == "none":
        quotative["status"] = "implicit"
    speaker = quote.setdefault("speaker", empty_speaker())
    for key, default in empty_speaker().items():
        if key in {"intro_phrase", "first_span"}:
            speaker[key] = coerce_text_span(speaker.get(key))
        else:
            speaker.setdefault(key, default)
    for mention in quote.get("mentions") or []:
        mention.setdefault("qid", None)
        mention.setdefault("qid_label", "")
        mention.setdefault("qid_description", "")
        mention.setdefault("nil", False)
        mention["intro_phrase"] = coerce_text_span(mention.get("intro_phrase"))
        mention["first_span"] = coerce_text_span(mention.get("first_span"))
    return quote


def attach_suggestions(text: str, quote: dict[str, Any]) -> dict[str, Any]:
    segments = quote_segments(quote)
    quote["quotative_candidates"] = find_quotative_candidates(
        text,
        inner_start=int(quote.get("inner_start") or 0),
        inner_end=int(quote.get("inner_end") or 0),
        outer_start=int(quote.get("outer_start") or 0),
        outer_end=int(quote.get("outer_end") or 0),
        segments=segments if len(segments) >= 2 else None,
    )
    quote["name_candidates"] = nearest_name_candidates(text, center=int(quote.get("outer_start") or 0))
    return quote


def article_status(record: dict[str, Any]) -> str:
    if record.get("skipped"):
        return "skipped"
    quotes = record.get("quotes") or []
    if not quotes:
        return "done"
    n_done = sum(1 for q in quotes if quote_is_touched(q))
    if n_done == 0:
        return "new"
    if article_is_complete(record):
        return "done"
    return "in_progress"
