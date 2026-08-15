"""Annotation record schema for Quotegraph gold."""

from __future__ import annotations

import copy
from typing import Any, Literal

QuoteStatus = Literal["pending", "keep", "reject"]
RejectReason = Literal["title", "scare", "not_speech", "other"]
SpeakerStatus = Literal["unset", "identified", "cannot_identify"]
QuotativeStatus = Literal["unset", "present", "implicit"]
MentionForm = Literal["proper", "pronoun", "nominal"]
Sentiment = Literal["positive", "negative", "mixed", "neutral", "not_about"]

PRONOUNS = frozenset(
    {
        "he",
        "she",
        "they",
        "him",
        "her",
        "them",
        "his",
        "hers",
        "their",
        "theirs",
        "i",
        "me",
        "my",
        "mine",
        "we",
        "us",
        "our",
        "ours",
        "you",
        "your",
        "yours",
    }
)

SENTIMENT_LABELS = ("positive", "negative", "mixed", "neutral", "not_about")


def infer_form(surface: str) -> MentionForm:
    token = surface.strip().lower()
    if token in PRONOUNS:
        return "pronoun"
    if token.startswith("the ") or token.startswith("a ") or token.startswith("an "):
        return "nominal"
    return "proper"


def empty_text_span() -> dict[str, Any]:
    return {"start": None, "end": None, "surface": ""}


def coerce_text_span(span: Any) -> dict[str, Any]:
    """Fill intro/first-span keys without dropping offsets already saved."""
    out = empty_text_span()
    if not isinstance(span, dict):
        return out
    if span.get("start") is not None:
        out["start"] = span["start"]
    if span.get("end") is not None:
        out["end"] = span["end"]
    if span.get("surface") is not None:
        out["surface"] = span["surface"]
    return out


def span_is_set(span: Any) -> bool:
    if not isinstance(span, dict):
        return False
    start, end = span.get("start"), span.get("end")
    return start is not None and end is not None


def empty_speaker() -> dict[str, Any]:
    return {
        "status": "unset",
        "start": None,
        "end": None,
        "surface": "",
        "qid": None,
        "qid_label": "",
        "qid_description": "",
        "nil": False,
        "form": "proper",
        "intro_phrase": empty_text_span(),
        "first_span": empty_text_span(),
    }


def empty_quotative() -> dict[str, Any]:
    return {
        "status": "unset",
        "start": None,
        "end": None,
        "surface": "",
    }


def mention_from_span(mention_id: str, start: int, end: int, text: str) -> dict[str, Any]:
    surface = text[start:end]
    return {
        "id": mention_id,
        "start": start,
        "end": end,
        "surface": surface,
        "form": infer_form(surface),
        "qid": None,
        "qid_label": "",
        "qid_description": "",
        "nil": False,
        "sentiment": None,
        "notes": "",
        "intro_phrase": empty_text_span(),
        "first_span": empty_text_span(),
    }


def quote_is_touched(quote: dict[str, Any]) -> bool:
    if quote.get("status") != "pending":
        return True
    speaker = quote.get("speaker") or {}
    if speaker.get("status") != "unset":
        return True
    if quote.get("mentions"):
        return True
    quotative = quote.get("quotative") or {}
    if quotative.get("status") not in (None, "unset"):
        return True
    return False


def _validate_optional_span(
    errors: list[str],
    quote_id: str,
    label: str,
    span: Any,
    n: int,
) -> None:
    if not isinstance(span, dict):
        return
    start, end = span.get("start"), span.get("end")
    if start is None and end is None:
        return
    if start is None or end is None:
        errors.append(f"{quote_id}: {label} span incomplete")
        return
    try:
        start_i, end_i = int(start), int(end)
    except (TypeError, ValueError):
        errors.append(f"{quote_id}: {label} offsets are not integers")
        return
    if not (0 <= start_i < end_i <= n):
        errors.append(f"{quote_id}: {label} span out of range")


def validation_errors(record: dict[str, Any]) -> list[str]:
    """Return human-readable problems; empty means the record is well-formed."""
    text = record.get("text") or ""
    n = len(text)
    errors: list[str] = []
    for quote in record.get("quotes") or []:
        qid = quote.get("id") or "?"
        try:
            inner_start = int(quote["inner_start"])
            inner_end = int(quote["inner_end"])
            outer_start = int(quote["outer_start"])
            outer_end = int(quote["outer_end"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{qid}: missing quote offsets")
            continue
        if not (0 <= outer_start <= inner_start <= inner_end <= outer_end <= n):
            errors.append(f"{qid}: quote offsets out of range")
        for i, segment in enumerate(quote.get("segments") or []):
            try:
                s_inner_s = int(segment["inner_start"])
                s_inner_e = int(segment["inner_end"])
                s_outer_s = int(segment["outer_start"])
                s_outer_e = int(segment["outer_end"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{qid}: segment {i} missing offsets")
                continue
            if not (0 <= s_outer_s <= s_inner_s <= s_inner_e <= s_outer_e <= n):
                errors.append(f"{qid}: segment {i} offsets out of range")
        speaker = quote.get("speaker") or {}
        _validate_optional_span(errors, qid, "intro phrase", speaker.get("intro_phrase"), n)
        _validate_optional_span(errors, qid, "first speaker span", speaker.get("first_span"), n)
        if speaker.get("status") == "identified":
            start, end = speaker.get("start"), speaker.get("end")
            if start is None or end is None:
                continue
            try:
                start_i, end_i = int(start), int(end)
            except (TypeError, ValueError):
                errors.append(f"{qid}: speaker offsets are not integers")
                continue
            if not (0 <= start_i < end_i <= n):
                errors.append(f"{qid}: speaker span out of range")
        quotative = quote.get("quotative") or {}
        if quotative.get("status") == "present":
            try:
                q_start, q_end = int(quotative["start"]), int(quotative["end"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{qid}: quotative missing offsets")
            else:
                if not (0 <= q_start < q_end <= n):
                    errors.append(f"{qid}: quotative span out of range")
        for mention in quote.get("mentions") or []:
            try:
                m_start, m_end = int(mention["start"]), int(mention["end"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{qid}: mention missing offsets")
                continue
            if not mention_inside_quote(quote, m_start, m_end):
                errors.append(f"{qid}: mention must lie inside the quote")
            _validate_optional_span(errors, qid, "mention intro phrase", mention.get("intro_phrase"), n)
            _validate_optional_span(errors, qid, "mention first span", mention.get("first_span"), n)
    return errors


def quote_from_candidate(
    quote_id: str,
    inner_start: int,
    inner_end: int,
    outer_start: int,
    outer_end: int,
    delimiter: str,
    text: str,
) -> dict[str, Any]:
    return {
        "id": quote_id,
        "status": "pending",
        "reject_reason": None,
        "inner_start": inner_start,
        "inner_end": inner_end,
        "outer_start": outer_start,
        "outer_end": outer_end,
        "delimiter": delimiter,
        "text": text[inner_start:inner_end],
        "speaker": empty_speaker(),
        "quotative": empty_quotative(),
        "mentions": [],
        "notes": "",
    }


def quote_segments(quote: dict[str, Any]) -> list[dict[str, Any]]:
    """Discontinuous quote spans. One segment unless the annotator merged."""
    segments = quote.get("segments") or []
    if len(segments) >= 2:
        return list(segments)
    return [
        {
            "inner_start": int(quote["inner_start"]),
            "inner_end": int(quote["inner_end"]),
            "outer_start": int(quote["outer_start"]),
            "outer_end": int(quote["outer_end"]),
            "delimiter": str(quote.get("delimiter") or ""),
            "text": quote.get("text") or "",
        }
    ]


def mention_inside_quote(quote: dict[str, Any], start: int, end: int) -> bool:
    for segment in quote_segments(quote):
        if int(segment["inner_start"]) <= start < end <= int(segment["inner_end"]):
            return True
    return False


def quotes_are_mergeable(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """True when ``right`` starts after ``left`` (interrupted quotation, not nesting)."""
    left_end = max(int(seg["outer_end"]) for seg in quote_segments(left))
    right_start = min(int(seg["outer_start"]) for seg in quote_segments(right))
    return right_start >= left_end


def _sync_covering_fields(quote: dict[str, Any], text: str) -> dict[str, Any]:
    segments = quote_segments(quote)
    for segment in segments:
        segment["text"] = text[int(segment["inner_start"]) : int(segment["inner_end"])]
    quote["outer_start"] = int(segments[0]["outer_start"])
    quote["outer_end"] = int(segments[-1]["outer_end"])
    quote["inner_start"] = int(segments[0]["inner_start"])
    quote["inner_end"] = int(segments[-1]["inner_end"])
    quote["text"] = " […] ".join(seg["text"] for seg in segments)
    if len(segments) >= 2:
        quote["segments"] = segments
    else:
        quote.pop("segments", None)
    return quote


def _prefer_speaker(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    ranked = {"identified": 2, "cannot_identify": 1, "unset": 0}
    pick = left
    if ranked.get((right.get("status") or "unset"), 0) > ranked.get((left.get("status") or "unset"), 0):
        pick = right
    return copy.deepcopy(pick)


def _prefer_quotative(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    ranked = {"present": 2, "implicit": 1, "none": 1, "unset": 0}
    pick = left
    if ranked.get((right.get("status") or "unset"), 0) > ranked.get((left.get("status") or "unset"), 0):
        pick = right
    return copy.deepcopy(pick)


def merge_quote_records(left: dict[str, Any], right: dict[str, Any], text: str) -> dict[str, Any]:
    """Join two quote candidates that are one interrupted quotation."""
    if not quotes_are_mergeable(left, right):
        raise ValueError("quotes overlap or are nested; not an interrupted quotation")
    merged = copy.deepcopy(left)
    segments = copy.deepcopy(quote_segments(left)) + copy.deepcopy(quote_segments(right))
    segments.sort(key=lambda seg: int(seg["outer_start"]))
    merged["segments"] = segments
    _sync_covering_fields(merged, text)
    ids: list[str] = []
    for src in (left, right):
        ids.extend(src.get("merged_from") or [src["id"]])
    merged["merged_from"] = list(dict.fromkeys(ids))
    merged["speaker"] = _prefer_speaker(left.get("speaker") or empty_speaker(), right.get("speaker") or empty_speaker())
    merged["quotative"] = _prefer_quotative(
        left.get("quotative") or empty_quotative(),
        right.get("quotative") or empty_quotative(),
    )
    mentions = list(left.get("mentions") or []) + list(right.get("mentions") or [])
    merged["mentions"] = [m for m in mentions if mention_inside_quote(merged, int(m["start"]), int(m["end"]))]
    left_status, right_status = left.get("status"), right.get("status")
    if left_status == "keep" or right_status == "keep":
        merged["status"] = "keep"
        merged["reject_reason"] = None
    elif left_status == "reject" and right_status == "reject":
        merged["status"] = "reject"
        merged["reject_reason"] = left.get("reject_reason") or right.get("reject_reason") or "other"
    else:
        merged["status"] = "pending"
        merged["reject_reason"] = None
    notes = [part for part in (left.get("notes") or "", right.get("notes") or "") if part]
    merged["notes"] = " ".join(notes)
    return merged


def _span_owner_index(quote: dict[str, Any], start: int | None, end: int | None) -> int:
    """Which piece of an interrupted quote a speaker/quotative span belongs to."""
    if start is None or end is None:
        return 0
    segments = quote_segments(quote)
    for i, segment in enumerate(segments):
        if int(segment["outer_start"]) <= int(start) < int(end) <= int(segment["outer_end"]):
            return i
        if i + 1 < len(segments) and int(segment["outer_end"]) <= int(start) < int(end) <= int(segments[i + 1]["outer_start"]):
            return i
    if int(end) <= int(segments[0]["outer_start"]):
        return 0
    return len(segments) - 1


def unmerge_quote_record(quote: dict[str, Any], text: str) -> list[dict[str, Any]]:
    """Split a merged quotation back into one candidate per mark-span."""
    segments = quote_segments(quote)
    if len(segments) < 2:
        return [copy.deepcopy(quote)]
    ids = list(quote.get("merged_from") or [])
    speaker = quote.get("speaker") or empty_speaker()
    quotative = quote.get("quotative") or empty_quotative()
    speaker_i = _span_owner_index(quote, speaker.get("start"), speaker.get("end")) if speaker.get("status") == "identified" else 0
    quotative_i = (
        _span_owner_index(quote, quotative.get("start"), quotative.get("end"))
        if quotative.get("status") == "present"
        else 0
    )
    pieces: list[dict[str, Any]] = []
    for i, segment in enumerate(segments):
        quote_id = ids[i] if i < len(ids) else f"{quote['id']}-p{i}"
        piece = quote_from_candidate(
            quote_id,
            int(segment["inner_start"]),
            int(segment["inner_end"]),
            int(segment["outer_start"]),
            int(segment["outer_end"]),
            str(segment.get("delimiter") or quote.get("delimiter") or ""),
            text,
        )
        piece["status"] = quote.get("status") or "pending"
        piece["reject_reason"] = quote.get("reject_reason")
        if i == speaker_i:
            piece["speaker"] = copy.deepcopy(speaker)
        if i == quotative_i:
            piece["quotative"] = copy.deepcopy(quotative)
        elif quotative.get("status") in {"implicit", "none"} and i == 0:
            piece["quotative"] = copy.deepcopy(quotative)
        piece["mentions"] = [
            copy.deepcopy(mention)
            for mention in quote.get("mentions") or []
            if int(segment["inner_start"]) <= int(mention["start"]) < int(mention["end"]) <= int(segment["inner_end"])
        ]
        pieces.append(piece)
    return pieces


def quote_is_complete(quote: dict[str, Any]) -> bool:
    if quote.get("status") == "pending":
        return False
    if quote.get("status") == "reject":
        return True
    speaker = quote.get("speaker") or {}
    if speaker.get("status") == "unset":
        return False
    if speaker.get("status") == "identified":
        if speaker.get("start") is None or speaker.get("end") is None:
            return False
        if not speaker.get("nil") and not speaker.get("qid"):
            return False
        if not span_is_set(speaker.get("first_span")):
            return False
    quotative = quote.get("quotative") or {}
    if quotative.get("status") not in {"present", "implicit", "none"}:
        return False
    if quotative.get("status") == "present" and (quotative.get("start") is None or quotative.get("end") is None):
        return False
    for mention in quote.get("mentions") or []:
        if mention.get("sentiment") not in SENTIMENT_LABELS:
            return False
        if not mention.get("nil") and not mention.get("qid"):
            return False
        if not span_is_set(mention.get("first_span")):
            return False
    return True


def article_is_complete(record: dict[str, Any]) -> bool:
    if record.get("skipped"):
        return True
    quotes = record.get("quotes") or []
    if not quotes:
        return True
    return all(quote_is_complete(q) for q in quotes)
