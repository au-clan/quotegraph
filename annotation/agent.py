"""Independent LLM annotator for the gold batch.

Writes to ``annotation/data/annotations/<annotator>/`` and never reads
human files under ``default/``.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
from typing import Any, Callable, Mapping

import httpx

from annotation.schema import (
    PRONOUNS,
    SENTIMENT_LABELS,
    empty_quotative,
    empty_speaker,
    infer_form,
    mention_from_span,
    quote_is_complete,
)
from annotation.store import attach_suggestions, coerce_quote
from annotation.wikidata import search_entities
from quotegraph.quote_merger import resolve_api_key

AGENT_ANNOTATOR = "agent"
DEFAULT_MODEL = "gpt-5-mini"
HUMAN_ANNOTATOR = "default"

CONTEXT_CHARS = 280
SPEAKER_WINDOW = 500
CHUNK_SIZE = 8
FULL_TEXT_LIMIT = 10_000

_WD_LOCK = threading.Lock()

SYSTEM_PROMPT = """\
You are an independent annotator for Quotegraph gold. Label every quotation-mark
span in the article. Do not look at anyone else's labels.

Rules:
- Direct quotes only. Keep real speech / writing attributed to a source.
  Reject titles/headlines, scare/emphasis quotes, and non-speech (nicknames,
  scare-quoted jargon, programme titles, scare-quoted irony).
- Reject reasons: title, scare, not_speech, other.
- Quotative: the journalist's reporting cue outside the quote (said, told,
  according to, added, announced, …). Copy the exact document substring.
  If attribution is clear but no cue is written, set quotative_status=implicit.
  Implicit is a real label, not a skip.
- Speaker: the source of the quote as a minimal person span copied exactly
  from the document. Use the name or pronoun only — not a title or role
  (bush not president bush; Obama not President Obama; she not Ms Smith).
  If that span is a pronoun, still copy the pronoun as speaker_surface and put
  the person's name in speaker_search for Wikidata. If you cannot identify a
  person speaker, speaker_status=cannot_identify. Organisations as speakers
  count as cannot_identify for this task (speaker must be a person).
- Mentions: named entities referred to inside the quote (people, organisations,
  places, named works). Copy exact substrings from the quote. Empty is fine
  when nothing is named. Do not mark generic nouns. Do not annotate the rest
  of the article. First-person I/we referring to the speaker should have
  search=the speaker's name.
- Sentiment: the speaker's attitude toward that mention in this quote:
  positive, negative, mixed, neutral, not_about.
- Copy spans exactly; do not paraphrase. Use the provided quote ids.
- For rejected quotes still fill the schema (empty strings, implicit,
  cannot_identify, empty mentions) but those fields are ignored.
"""

_MENTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "surface": {"type": "string"},
        "search": {"type": "string"},
        "sentiment": {"type": "string", "enum": list(SENTIMENT_LABELS)},
    },
    "required": ["surface", "search", "sentiment"],
    "additionalProperties": False,
}

_QUOTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "status": {"type": "string", "enum": ["keep", "reject"]},
        "reject_reason": {
            "anyOf": [
                {"type": "string", "enum": ["title", "scare", "not_speech", "other"]},
                {"type": "null"},
            ]
        },
        "quotative_status": {"type": "string", "enum": ["present", "implicit"]},
        "quotative_surface": {"type": "string"},
        "speaker_status": {"type": "string", "enum": ["identified", "cannot_identify"]},
        "speaker_surface": {"type": "string"},
        "speaker_search": {"type": "string"},
        "mentions": {"type": "array", "items": _MENTION_SCHEMA},
        "notes": {"type": "string"},
    },
    "required": [
        "id",
        "status",
        "reject_reason",
        "quotative_status",
        "quotative_surface",
        "speaker_status",
        "speaker_surface",
        "speaker_search",
        "mentions",
        "notes",
    ],
    "additionalProperties": False,
}

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"quotes": {"type": "array", "items": _QUOTE_SCHEMA}},
    "required": ["quotes"],
    "additionalProperties": False,
}

_TITLE_PREFIX = re.compile(
    r"^(?:mr|mrs|ms|dr|sir|dame|president|pope|king|queen|prime minister|"
    r"senator|sen|representative|rep|secretary|gov(?:ernor)?|gen(?:eral)?)\.?\s+",
    re.I,
)


def _minimal_person_span(surface: str) -> str:
    trimmed = (surface or "").strip()
    stripped = _TITLE_PREFIX.sub("", trimmed).strip()
    return stripped or trimmed


def locate_surface(
    text: str,
    surface: str,
    *,
    lo: int,
    hi: int,
    used: set[tuple[int, int]] | None = None,
    center: int | None = None,
    prefer_outside: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Find ``surface`` in ``text[lo:hi]``. Prefer unused, then outside, then nearest."""
    needle = (surface or "").strip()
    if not needle or hi <= lo:
        return None
    window = text[lo:hi]
    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = window.find(needle, start)
        if idx < 0:
            break
        matches.append((lo + idx, lo + idx + len(needle)))
        start = idx + 1
    if not matches:
        low = window.lower()
        key = needle.lower()
        start = 0
        while True:
            idx = low.find(key, start)
            if idx < 0:
                break
            matches.append((lo + idx, lo + idx + len(key)))
            start = idx + 1
    if used:
        matches = [span for span in matches if span not in used]
    if not matches:
        return None
    if prefer_outside is not None:
        inner_start, inner_end = prefer_outside
        outside = [span for span in matches if span[1] <= inner_start or span[0] >= inner_end]
        if outside:
            matches = outside
    if center is not None:
        matches.sort(key=lambda span: (abs(span[0] - center), span[0]))
    return matches[0]


def _clip(text: str, start: int, end: int) -> str:
    return text[max(0, start) : min(len(text), end)]


def _quote_block(article: dict[str, Any], quote: dict[str, Any]) -> str:
    text = article["text"]
    inner_start = int(quote["inner_start"])
    inner_end = int(quote["inner_end"])
    outer_start = int(quote["outer_start"])
    outer_end = int(quote["outer_end"])
    before = _clip(text, outer_start - CONTEXT_CHARS, outer_start).replace("\n", " ")
    after = _clip(text, outer_end, outer_end + CONTEXT_CHARS).replace("\n", " ")
    inner = text[inner_start:inner_end].replace("\n", " ")
    cues = quote.get("quotative_candidates") or []
    names = quote.get("name_candidates") or []
    cue_s = ", ".join(f"{c['surface']}@{c['start']}-{c['end']}" for c in cues[:8]) or "none"
    name_s = ", ".join(f"{c['surface']}@{c['start']}-{c['end']}" for c in names[:8]) or "none"
    return (
        f"### {quote['id']}\n"
        f"outer=[{outer_start},{outer_end}] inner=[{inner_start},{inner_end}]\n"
        f"BEFORE: {before}\n"
        f"QUOTE: {inner}\n"
        f"AFTER: {after}\n"
        f"quotative_candidates: {cue_s}\n"
        f"name_candidates: {name_s}\n"
    )


def build_user_prompt(article: dict[str, Any], quotes: list[dict[str, Any]]) -> str:
    text = article["text"]
    header = (
        f"article_id: {article['article_id']}\n"
        f"title: {article.get('title') or ''}\n"
        f"source: {article.get('source') or ''}  date: {article.get('date') or ''}  "
        f"phase: {article.get('phase') or ''}\n"
    )
    if len(text) <= FULL_TEXT_LIMIT:
        header += f"\nARTICLE TEXT:\n{text}\n"
    header += "\nAnnotate these quote candidates. Return JSON for every id listed.\n\n"
    return header + "\n".join(_quote_block(article, quote) for quote in quotes)


def prepare_article_quotes(article: dict[str, Any]) -> dict[str, Any]:
    """Deep copy with quotative/name suggestions attached (same input the LLM sees)."""
    record = copy.deepcopy(article)
    record["quotes"] = [copy.deepcopy(quote) for quote in article.get("quotes") or []]
    for quote in record["quotes"]:
        coerce_quote(quote)
        attach_suggestions(record["text"], quote)
    return record


def iter_chunk_prompts(
    article: dict[str, Any],
    *,
    chunk_size: int = CHUNK_SIZE,
) -> list[dict[str, Any]]:
    record = prepare_article_quotes(article)
    quotes = record["quotes"]
    messages: list[dict[str, Any]] = []
    if not quotes:
        return messages
    for start in range(0, len(quotes), chunk_size):
        chunk = quotes[start : start + chunk_size]
        messages.append(
            {
                "call": len(messages) + 1,
                "kind": "chunk",
                "quote_ids": [quote["id"] for quote in chunk],
                "user": build_user_prompt(record, chunk),
            }
        )
    return messages


def prompt_payload(
    article: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    messages: list[dict[str, Any]] | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, Any]:
    return {
        "article_id": article["article_id"],
        "model": model,
        "system": SYSTEM_PROMPT,
        "messages": messages if messages is not None else iter_chunk_prompts(article, chunk_size=chunk_size),
    }


def render_prompt_text(payload: dict[str, Any]) -> str:
    lines = [
        f"article_id: {payload.get('article_id') or ''}",
        f"model: {payload.get('model') or ''}",
        "",
        "## system",
        payload.get("system") or "",
        "",
    ]
    for message in payload.get("messages") or []:
        quote_ids = ",".join(message.get("quote_ids") or [])
        lines.extend(
            [
                f"## user call {message.get('call')} ({message.get('kind') or 'chunk'}) quotes={quote_ids}",
                message.get("user") or "",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def save_agent_prompts(annotator: str, payload: dict[str, Any]) -> None:
    from annotation.store import prompt_path, prompt_text_path

    article_id = str(payload["article_id"])
    prompt_path(annotator, article_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    prompt_text_path(annotator, article_id).write_text(render_prompt_text(payload), encoding="utf-8")


def _reasoning_effort(model: str) -> str:
    if model.startswith("gpt-5.4"):
        return "none"
    if model.startswith("gpt-5-nano"):
        return "minimal"
    if model.startswith("gpt-5"):
        return "low"
    return "minimal"


def call_openai_json(
    *,
    system: str,
    user: str,
    model: str,
    api_key: str,
    schema: Mapping[str, Any] = RESPONSE_SCHEMA,
    timeout: float = 120.0,
    max_retries: int = 4,
    max_output_tokens: int = 12_288,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "quotegraph_article_annotation",
                "schema": schema,
                "strict": True,
            },
        },
    }
    if model.startswith("gpt-5"):
        payload["max_completion_tokens"] = max_output_tokens
        payload["reasoning_effort"] = _reasoning_effort(model)
    else:
        payload["max_tokens"] = max_output_tokens
        payload["temperature"] = 0
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            try:
                response = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if response.status_code == 429:
                    time.sleep(float(response.headers.get("Retry-After", "2")))
                    continue
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise RuntimeError(f"{exc}; response={response.text[:1200]}") from exc
                data = response.json()
                content = data["choices"][0]["message"].get("content")
                if not content:
                    finish = data["choices"][0].get("finish_reason")
                    raise ValueError(f"Empty model response (finish_reason={finish})")
                parsed = json.loads(content)
                if not isinstance(parsed, dict) or "quotes" not in parsed:
                    raise ValueError("Model JSON missing quotes")
                return parsed
            except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(0.8 * (2**attempt))
    assert last_error is not None
    raise last_error


def link_wikidata(query: str, *, prefer_human: bool) -> dict[str, Any] | None:
    query = re.sub(r"\s+", " ", query or "").strip()
    if len(query) < 2 or query.lower() in PRONOUNS:
        return None
    with _WD_LOCK:
        hits = search_entities(query, limit=8)
    if not hits:
        return None
    pool = [row for row in hits if row.get("is_human")] if prefer_human else hits
    if prefer_human and not pool:
        return None
    pool = pool or hits
    key = query.lower()
    for row in pool:
        label = (row.get("label") or "").lower()
        if key == label or key in label or label in key:
            return row
    return pool[0]


def _apply_entity_link(entity: dict[str, Any], query: str, *, prefer_human: bool) -> None:
    hit = link_wikidata(query or entity.get("surface") or "", prefer_human=prefer_human)
    if hit:
        entity["qid"] = hit["qid"]
        entity["qid_label"] = hit.get("label") or ""
        entity["qid_description"] = hit.get("description") or ""
        entity["nil"] = False
    else:
        entity["qid"] = None
        entity["qid_label"] = ""
        entity["qid_description"] = ""
        entity["nil"] = True


def apply_quote_prediction(article: dict[str, Any], quote: dict[str, Any], pred: Mapping[str, Any]) -> None:
    text = article["text"]
    inner_start = int(quote["inner_start"])
    inner_end = int(quote["inner_end"])
    outer_start = int(quote["outer_start"])
    outer_end = int(quote["outer_end"])
    notes = [pred.get("notes") or ""]

    if pred.get("status") == "reject":
        quote["status"] = "reject"
        quote["reject_reason"] = pred.get("reject_reason") or "not_speech"
        quote["speaker"] = empty_speaker()
        quote["quotative"] = empty_quotative()
        quote["mentions"] = []
        quote["notes"] = " ".join(part for part in notes if part).strip()
        return

    quote["status"] = "keep"
    quote["reject_reason"] = None

    quotative_status = pred.get("quotative_status") or "implicit"
    quotative_surface = (pred.get("quotative_surface") or "").strip()
    quote["quotative"] = empty_quotative()
    if quotative_status == "present" and quotative_surface:
        located = None
        for cand in quote.get("quotative_candidates") or []:
            if cand["surface"].lower() == quotative_surface.lower():
                located = (int(cand["start"]), int(cand["end"]))
                break
        if located is None:
            located = locate_surface(
                text,
                quotative_surface,
                lo=max(0, outer_start - SPEAKER_WINDOW),
                hi=min(len(text), outer_end + SPEAKER_WINDOW),
                center=outer_start,
                prefer_outside=(inner_start, inner_end),
            )
        if located is not None:
            start, end = located
            quote["quotative"] = {
                "status": "present",
                "start": start,
                "end": end,
                "surface": text[start:end],
            }
        else:
            quote["quotative"]["status"] = "implicit"
            notes.append("agent: quotative surface not found; marked implicit")
    else:
        quote["quotative"]["status"] = "implicit"

    speaker_status = pred.get("speaker_status") or "cannot_identify"
    speaker_surface = _minimal_person_span(pred.get("speaker_surface") or "")
    quote["speaker"] = empty_speaker()
    if speaker_status == "identified" and speaker_surface:
        located = locate_surface(
            text,
            speaker_surface,
            lo=max(0, outer_start - SPEAKER_WINDOW),
            hi=min(len(text), outer_end + SPEAKER_WINDOW),
            center=outer_start,
            prefer_outside=(inner_start, inner_end),
        )
        if located is None:
            quote["speaker"]["status"] = "cannot_identify"
            notes.append("agent: speaker surface not found; cannot identify")
        else:
            start, end = located
            quote["speaker"] = empty_speaker()
            quote["speaker"].update(
                {
                    "status": "identified",
                    "start": start,
                    "end": end,
                    "surface": text[start:end],
                    "form": infer_form(text[start:end]),
                }
            )
            _apply_entity_link(
                quote["speaker"],
                pred.get("speaker_search") or text[start:end],
                prefer_human=True,
            )
    else:
        quote["speaker"]["status"] = "cannot_identify"

    used: set[tuple[int, int]] = set()
    mentions: list[dict[str, Any]] = []
    for i, raw in enumerate(pred.get("mentions") or []):
        surface = (raw.get("surface") or "").strip()
        if not surface:
            continue
        located = locate_surface(
            text,
            surface,
            lo=inner_start,
            hi=inner_end,
            used=used,
            center=inner_start,
        )
        if located is None:
            notes.append(f"agent: dropped mention {surface!r} (not inside quote)")
            continue
        used.add(located)
        mention = mention_from_span(f"{quote['id']}-m{i}", located[0], located[1], text)
        mention["sentiment"] = raw.get("sentiment") if raw.get("sentiment") in SENTIMENT_LABELS else "neutral"
        _apply_entity_link(mention, raw.get("search") or mention["surface"], prefer_human=False)
        mentions.append(mention)
    quote["mentions"] = mentions
    quote["notes"] = " ".join(part for part in notes if part).strip()


def _index_predictions(preds: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["id"]): row for row in preds if row.get("id")}


def annotate_article(
    article: dict[str, Any],
    *,
    llm: LlmFn,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, Any]:
    """Return a new record with every quote labelled. Does not read human annotations."""
    record = prepare_article_quotes(article)
    quotes = record["quotes"]
    prompt_messages: list[dict[str, Any]] = []
    if not quotes:
        record["skipped"] = False
        record["notes"] = (record.get("notes") or "").strip()
        record["agent_prompts"] = prompt_payload(record, messages=[])
        return record

    collected: dict[str, Mapping[str, Any]] = {}
    for start in range(0, len(quotes), chunk_size):
        chunk = quotes[start : start + chunk_size]
        prompt = build_user_prompt(record, chunk)
        prompt_messages.append(
            {
                "call": len(prompt_messages) + 1,
                "kind": "chunk",
                "quote_ids": [quote["id"] for quote in chunk],
                "user": prompt,
            }
        )
        parsed = llm(SYSTEM_PROMPT, prompt)
        collected.update(_index_predictions(list(parsed.get("quotes") or [])))

    missing = [quote["id"] for quote in quotes if quote["id"] not in collected]
    if missing:
        retry = [quote for quote in quotes if quote["id"] in missing]
        retry_prompt = build_user_prompt(record, retry)
        prompt_messages.append(
            {
                "call": len(prompt_messages) + 1,
                "kind": "retry",
                "quote_ids": [quote["id"] for quote in retry],
                "user": retry_prompt,
            }
        )
        parsed = llm(SYSTEM_PROMPT, retry_prompt)
        collected.update(_index_predictions(list(parsed.get("quotes") or [])))

    for quote in quotes:
        pred = collected.get(quote["id"])
        if pred is None:
            quote["status"] = "reject"
            quote["reject_reason"] = "other"
            quote["speaker"] = empty_speaker()
            quote["quotative"] = empty_quotative()
            quote["mentions"] = []
            quote["notes"] = "agent: omitted this candidate after retry"
            continue
        apply_quote_prediction(record, quote, pred)
        if quote["status"] == "keep" and not quote_is_complete(quote):
            if (quote.get("quotative") or {}).get("status") not in {"present", "implicit", "none"}:
                quote["quotative"] = empty_quotative()
                quote["quotative"]["status"] = "implicit"
            if (quote.get("speaker") or {}).get("status") == "unset":
                quote["speaker"]["status"] = "cannot_identify"

    record["skipped"] = False
    record["skip_reason"] = ""
    record["annotator"] = AGENT_ANNOTATOR
    record["agent_prompts"] = prompt_payload(record, messages=prompt_messages)
    return record


def make_openai_llm(model: str = DEFAULT_MODEL, api_key: str | None = None) -> LlmFn:
    key = resolve_api_key(api_key)

    def _call(system: str, user: str) -> dict[str, Any]:
        return call_openai_json(system=system, user=user, model=model, api_key=key)

    return _call
