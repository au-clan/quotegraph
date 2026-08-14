"""Annotation record schema for Quotegraph gold."""

from __future__ import annotations

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
        speaker = quote.get("speaker") or {}
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
            if not (inner_start <= m_start < m_end <= inner_end):
                errors.append(f"{qid}: mention must lie inside the quote")
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
    return True


def article_is_complete(record: dict[str, Any]) -> bool:
    if record.get("skipped"):
        return True
    quotes = record.get("quotes") or []
    if not quotes:
        return True
    return all(quote_is_complete(q) for q in quotes)
