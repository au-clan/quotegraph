#!/usr/bin/env python3
"""Evaluate LLM speaker attribution prompts on DirectQuote.

The evaluator uses DirectQuote's CoNLL labels as gold speaker spans.  It asks
an LLM to return structured JSON containing only copied text fields:

* ``speaker``: free-text speaker span, including pronouns when appropriate.
* ``speaker_refers_to``: exact document span for the antecedent when
  ``speaker`` is a pronoun, otherwise an empty string.
* ``quotative_verb``: exact document span for the attribution verb.

No chain-of-thought or reasoning text is requested or accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import httpx

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quotegraph.directquote_loader import _extract_spans
from quotegraph.merger_patterns import load_merger_patterns
from quotegraph.quote_merger import resolve_api_key


DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_KEY_PATH = Path.home() / "configs/keys/openai.txt"
DEFAULT_GOOGLE_KEY_PATH = Path.home() / "configs/keys/google.txt"
DEFAULT_INPUT = Path("/tmp/DirectQuote/data/truecased.txt")

PRONOUNS = {
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "they",
    "them",
    "their",
    "theirs",
    "we",
    "us",
    "our",
    "ours",
    "i",
    "me",
    "my",
    "mine",
}

UNKNOWN_MARKERS = {"", "unknown", "none", "no speaker", "not stated", "not found", "n/a", "null"}
NONE_CANDIDATE = "None"
FIRST_SECOND_PERSON_PRONOUNS = {
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
ALL_PRONOUN_CANDIDATES = FIRST_SECOND_PERSON_PRONOUNS | {
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "they",
    "them",
    "their",
    "theirs",
}

ATTRIBUTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "speaker": {
            "type": "string",
            "description": "Exact speaker span copied from the document; may be a pronoun; empty if no speaker is stated.",
        },
        "speaker_refers_to": {
            "type": "string",
            "description": "Exact document span for the pronoun antecedent, or empty when speaker is not a pronoun.",
        },
        "quotative_verb": {
            "type": "string",
            "description": "Exact attribution verb copied from the document, such as said/told/asked/wrote; empty if none.",
        },
    },
    "required": ["speaker", "speaker_refers_to", "quotative_verb"],
    "additionalProperties": False,
}

_SPACY_NLP: Any | None | bool = None


@dataclass(frozen=True)
class GoldAttribution:
    paragraph_id: int
    quote_index: int
    quote_kind: str
    document: str
    marked_document: str
    target_quote: str
    left_context: str
    right_context: str
    gold_speaker: str
    gold_quotative_verb: str
    target_start: int
    target_end: int


@dataclass
class AttributionPrediction:
    speaker: str
    speaker_refers_to: str
    quotative_verb: str


@dataclass
class AttributionOutcome:
    paragraph_id: int
    quote_index: int
    quote_kind: str
    prompt_config: str
    target_quote: str
    gold_speaker: str
    pred_speaker: str
    gold_quotative_verb: str
    pred_quotative_verb: str
    pred_speaker_refers_to: str
    speaker_exact: bool
    speaker_token_f1: float
    verb_exact: bool
    predicted_empty: bool
    gold_empty: bool
    speaker_is_pronoun: bool
    pronoun_referent_present: bool
    speaker_in_document: bool
    referent_in_document: bool
    verb_in_document: bool
    from_cache: bool
    error: str


def _parse_paragraphs(path: Path) -> list[tuple[int, list[tuple[str, str]]]]:
    paragraphs: list[tuple[int, list[tuple[str, str]]]] = []
    current: list[tuple[str, str]] = []
    paragraph_id = 0
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip():
                if current:
                    paragraphs.append((paragraph_id, current))
                    current = []
                paragraph_id += 1
                continue
            word, tag = line.rsplit(" ", 1)
            current.append((word, tag))
    if current:
        paragraphs.append((paragraph_id, current))
    return paragraphs


def _speaker_tokens_around(
    tokens: list[tuple[str, str]],
    start: int,
    end: int,
    quote_kind: str,
    radius: int = 35,
) -> tuple[str, ...]:
    if quote_kind == "LeftSpeaker":
        indices = range(start - 1, max(start - radius, -1), -1)
    elif quote_kind == "RightSpeaker":
        indices = range(end, min(end + radius, len(tokens)))
    else:
        return ()

    collected: list[str] = []
    for idx in indices:
        word, tag = tokens[idx]
        if tag in {"B-Speaker", "I-Speaker"}:
            collected.append(word)
        elif collected:
            break
    if quote_kind == "LeftSpeaker":
        collected.reverse()
    return tuple(collected)


def _window_text(tokens: list[tuple[str, str]], start: int, end: int) -> str:
    return " ".join(word for word, _ in tokens[max(0, start) : min(len(tokens), end)])


def _document_with_target_marked(
    tokens: list[tuple[str, str]],
    start: int,
    end: int,
) -> str:
    words = [word for word, _ in tokens]
    marked = words[:start] + ["<TARGET_QUOTE>"] + words[start:end] + ["</TARGET_QUOTE>"] + words[end:]
    return " ".join(marked)


def _nearest_quotative_verb(
    tokens: list[tuple[str, str]],
    quote_start: int,
    quote_end: int,
    quote_kind: str,
) -> str:
    patterns = load_merger_patterns()
    if quote_kind == "LeftSpeaker":
        window = _window_text(tokens, max(0, quote_start - 35), quote_start)
        matches = list(patterns.attribution_re.finditer(window))
        return matches[-1].group(0) if matches else ""
    if quote_kind == "RightSpeaker":
        window = _window_text(tokens, quote_end, min(len(tokens), quote_end + 35))
        match = patterns.attribution_re.search(window)
        return match.group(0) if match else ""
    return ""


def _load_spacy_ner() -> Any | None:
    global _SPACY_NLP
    if _SPACY_NLP is False:
        return None
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
    except Exception:
        _SPACY_NLP = False
        return None
    if "ner" not in nlp.pipe_names:
        _SPACY_NLP = False
        return None
    _SPACY_NLP = nlp
    return nlp


ROLE_PREFIX_RE = re.compile(
    r"^(?:(?:Mr|Mrs|Ms|Miss|Dr|Prof|Professor|Rep|Representative|Sen|Senator|"
    r"Gov|Governor|President|Prime\s+Minister|Minister|Secretary|Deputy|"
    r"Chief|Officer|Attorney|Lawyer|Judge|Justice|Sir|Dame|Lord|Lady)\.?\s+)+",
    re.IGNORECASE,
)


def _canonical_candidate(text: str) -> str:
    candidate = re.sub(r"\s+", " ", text).strip(" \t\r\n\"'`.,:;!?()[]{}")
    candidate = re.sub(r"\s+([.'-])\s+", r" \1 ", candidate)
    if "," in candidate:
        candidate = candidate.split(",", 1)[0].strip()
    candidate = re.sub(r"^[A-Z][A-Za-z]+ '?s\s+", "", candidate)
    candidate = ROLE_PREFIX_RE.sub("", candidate).strip()

    words = candidate.split()
    lower_words = {w.lower().strip(".") for w in words}
    role_words = {
        "leader",
        "minister",
        "secretary",
        "attorney",
        "lawyer",
        "chairman",
        "chairwoman",
        "chief",
        "shadow",
        "employment",
        "conservative",
        "labour",
        "democrat",
        "republican",
    }
    if len(words) > 2 and lower_words.intersection(role_words):
        capital_words = [
            w
            for w in words
            if re.match(r"^[A-Z][A-Za-z.'-]*$", w) or re.match(r"^[A-Z]\.?$", w)
        ]
        if len(capital_words) >= 2:
            candidate = " ".join(capital_words[-2:])
    return candidate.strip(" \t\r\n\"'`.,:;!?()[]{}")


def _regex_person_mentions(text: str) -> list[str]:
    pattern = re.compile(
        r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Professor|Rep|Representative|Sen|Senator|"
        r"Gov|Governor|President|Prime Minister|Minister|Secretary|Deputy Secretary of State|"
        r"Chief|Officer|Attorney|Judge|Justice)?\.?\s*"
        r"[A-Z][A-Za-z.'-]*(?:\s+(?:[A-Z][A-Za-z.'-]*|[A-Z]\.?)){0,4}"
        r"(?:\s*,\s*[^,.!?]{1,80})?"
    )
    mentions: list[str] = []
    for match in pattern.finditer(text):
        raw = match.group(0).strip()
        if not raw:
            continue
        tokens = raw.split()
        if len(tokens) == 1 and tokens[0] in {"The", "This", "That", "While", "If", "But"}:
            continue
        mentions.append(raw)
    return mentions


def extract_speaker_candidates(example: GoldAttribution, *, limit: int = 24) -> list[str]:
    text = " ".join(part for part in [example.left_context, example.right_context] if part)
    mentions: list[str] = []
    nlp = _load_spacy_ner()
    if nlp is not None:
        doc = nlp(text)
        mentions.extend(ent.text for ent in doc.ents if ent.label_ == "PERSON")
    if not mentions:
        mentions.extend(_regex_person_mentions(text))

    candidates: list[str] = []
    seen: set[str] = set()
    for mention in mentions:
        for candidate in [mention, _canonical_candidate(mention)]:
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if not candidate:
                continue
            if candidate.lower() in ALL_PRONOUN_CANDIDATES:
                continue
            key = _normalise(candidate)
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
            if len(candidates) >= limit:
                return candidates
    return candidates


def load_examples(path: Path) -> list[GoldAttribution]:
    return load_examples_with_context(path, context_tokens=45)


def load_examples_with_context(path: Path, *, context_tokens: int) -> list[GoldAttribution]:
    examples: list[GoldAttribution] = []
    for paragraph_id, tokens in _parse_paragraphs(path):
        spans = _extract_spans(tokens)
        if not spans:
            continue
        document = " ".join(word for word, _ in tokens)
        for quote_index, (quote_kind, start, end) in enumerate(spans):
            speaker_tokens = _speaker_tokens_around(tokens, start, end, quote_kind)
            examples.append(
                GoldAttribution(
                    paragraph_id=paragraph_id,
                    quote_index=quote_index,
                    quote_kind=quote_kind,
                    document=document,
                    marked_document=_document_with_target_marked(tokens, start, end),
                    target_quote=_window_text(tokens, start, end),
                    left_context=_window_text(tokens, start - context_tokens, start),
                    right_context=_window_text(tokens, end, end + context_tokens),
                    gold_speaker=" ".join(speaker_tokens),
                    gold_quotative_verb=_nearest_quotative_verb(tokens, start, end, quote_kind),
                    target_start=start,
                    target_end=end,
                )
            )
    return examples


def build_prompt(
    example: GoldAttribution,
    config: str,
    *,
    prompt_scope: str = "document",
    candidate_mode: bool = False,
    candidates: list[str] | None = None,
) -> str:
    if prompt_scope == "document":
        base_input = (
            f"Document:\n{example.marked_document}\n\n"
            f"Target quote:\n{example.target_quote}\n\n"
            f"Left context:\n{example.left_context}\n\n"
            f"Right context:\n{example.right_context}"
        )
    elif prompt_scope == "context":
        base_input = (
            f"Context before target quote:\n{example.left_context}\n\n"
            f"Target quote:\n<TARGET_QUOTE> {example.target_quote} </TARGET_QUOTE>\n\n"
            f"Context after target quote:\n{example.right_context}"
        )
    else:
        raise ValueError(f"Unknown prompt scope: {prompt_scope}")
    if candidate_mode:
        candidate_lines = "\n".join(f"- {candidate}" for candidate in (candidates or []))
        base_input = (
            f"{base_input}\n\nCandidate speaker mentions:\n"
            f"{candidate_lines if candidate_lines else '- None'}"
        )
        output_rules = (
            "Return only JSON matching the schema. The speaker field must be exactly "
            "one item from Candidate speaker mentions, or exactly None if no candidate "
            "is the attributed speaker of the target quote. Do not output first- or "
            "second-person pronouns such as I, me, my, we, us, our, you, or your. "
            "Copy quotative_verb as an exact span from the provided text, or use an "
            "empty string when absent. Keep speaker_refers_to empty unless speaker is "
            "a third-person pronoun candidate. Do not explain."
        )
    else:
        output_rules = (
            "Return only JSON matching the schema. Copy speaker, speaker_refers_to, "
            "and quotative_verb as exact spans from the provided text, or use an empty "
            "string when absent. Speaker may be a pronoun if the document attributes "
            "the quote to a pronoun. If speaker is a pronoun, fill speaker_refers_to "
            "with the exact provided span that the pronoun refers to. Do not explain."
        )

    if config == "direct":
        task = "Identify who is the speaker of the target direct quotation."
        return f"{task}\n\n{base_input}\n\n{output_rules}"
    if config == "extraction":
        task = (
            "Extract the attribution fields for the target direct quotation: "
            "speaker span, pronoun referent span if needed, and quotative verb."
        )
        return f"{task}\n\n{base_input}\n\n{output_rules}"
    if config == "directquote_labels":
        task = (
            "DirectQuote annotates quotations as LeftSpeaker, RightSpeaker, or "
            "Unknown depending on where the speaker appears. Use the target quote "
            "and surrounding text to recover the Speaker span."
        )
        return f"{task}\n\n{base_input}\n\n{output_rules}"
    if config == "direct_repeat":
        task = "Identify who is the speaker of the target direct quotation."
        prompt = f"{task}\n\n{base_input}\n\n{output_rules}"
        return f"{prompt}\n\n{prompt}"
    raise ValueError(f"Unknown prompt config: {config}")


def _system_prompt() -> str:
    return (
        "You are an information extraction system for news quotation attribution. "
        "Do not reveal reasoning or chain-of-thought. Return only the requested JSON."
    )


def _reasoning_effort_for_model(model: str) -> str:
    if model.startswith("gpt-5.4"):
        return "none"
    return "minimal"


def _cache_key(
    model: str,
    config: str,
    example: GoldAttribution,
    *,
    prompt_text: str,
    provider: str,
    prompt_scope: str,
    context_tokens: int,
) -> str:
    raw = json.dumps(
        {
            "model": model,
            "provider": provider,
            "config": config,
            "prompt_scope": prompt_scope,
            "context_tokens": context_tokens,
            "paragraph_id": example.paragraph_id,
            "quote_index": example.quote_index,
            "document": example.document,
            "marked_document": example.marked_document,
            "target_quote": example.target_quote,
            "prompt_text": prompt_text,
            "schema": ATTRIBUTION_SCHEMA,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_cache(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(cache_dir: Path, key: str, data: Mapping[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _parse_prediction_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
        if not isinstance(parsed, dict):
            raise
        return parsed


def attribution_schema_for_candidates(candidates: list[str]) -> dict[str, Any]:
    schema = json.loads(json.dumps(ATTRIBUTION_SCHEMA))
    enum_values = list(dict.fromkeys([*candidates, NONE_CANDIDATE]))
    schema["properties"]["speaker"]["enum"] = enum_values
    schema["properties"]["speaker"]["description"] = (
        "Exactly one candidate speaker mention, or None if no candidate is the attributed speaker."
    )
    schema["properties"]["speaker_refers_to"]["description"] = (
        "Empty unless speaker is a third-person pronoun candidate."
    )
    return schema


def call_openai(
    *,
    model: str,
    api_key: str,
    prompt: str,
    timeout: float,
    max_retries: int,
    max_output_tokens: int,
    schema: Mapping[str, Any] = ATTRIBUTION_SCHEMA,
) -> tuple[AttributionPrediction, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "directquote_attribution",
                "schema": schema,
                "strict": True,
            },
        },
    }
    if model.startswith("gpt-5"):
        payload["max_completion_tokens"] = max_output_tokens
        payload["reasoning_effort"] = _reasoning_effort_for_model(model)
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
                    retry_after = float(response.headers.get("Retry-After", "1"))
                    time.sleep(retry_after)
                    continue
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    detail = response.text[:1000]
                    raise RuntimeError(f"{exc}; response={detail}") from exc
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not content:
                    finish = data["choices"][0].get("finish_reason")
                    raise ValueError(f"Empty model response (finish_reason={finish})")
                parsed = _parse_prediction_json(content)
                prediction = AttributionPrediction(
                    speaker=str(parsed["speaker"]),
                    speaker_refers_to=str(parsed["speaker_refers_to"]),
                    quotative_verb=str(parsed["quotative_verb"]),
                )
                return prediction, data
            except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(0.5 * (2**attempt))
    assert last_error is not None
    raise last_error


def call_google(
    *,
    model: str,
    api_key: str,
    prompt: str,
    timeout: float,
    max_retries: int,
    max_output_tokens: int,
    schema: Mapping[str, Any] = ATTRIBUTION_SCHEMA,
) -> tuple[AttributionPrediction, dict[str, Any]]:
    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": _system_prompt()}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
        },
    }
    if model == "gemma-4-26b-a4b-it":
        payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "high"}
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    last_error: Exception | None = None
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            try:
                response = client.post(url, headers=headers, json=payload)
                if response.status_code == 429:
                    last_error = RuntimeError(f"Rate limited: {response.text[:1000]}")
                    retry_after = float(response.headers.get("Retry-After", "1"))
                    time.sleep(retry_after)
                    continue
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    detail = response.text[:1000]
                    raise RuntimeError(f"{exc}; response={detail}") from exc
                data = response.json()
                parts = data["candidates"][0]["content"]["parts"]
                content = "".join(
                    str(part.get("text", ""))
                    for part in parts
                    if not part.get("thought")
                ).strip()
                if not content:
                    finish = data["candidates"][0].get("finishReason")
                    raise ValueError(f"Empty model response (finish_reason={finish})")
                parsed = _parse_prediction_json(content)
                prediction = AttributionPrediction(
                    speaker=str(parsed["speaker"]),
                    speaker_refers_to=str(parsed["speaker_refers_to"]),
                    quotative_verb=str(parsed["quotative_verb"]),
                )
                return prediction, data
            except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(0.5 * (2**attempt))
    assert last_error is not None
    raise last_error


def _normalise(text: str) -> str:
    text = text.strip()
    if text.lower() in UNKNOWN_MARKERS:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'`.,:;!?()[]{}")
    return text.lower()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", _normalise(text))


def _token_f1(predicted: str, gold: str) -> float:
    pred_tokens = _tokens(predicted)
    gold_tokens = _tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _is_pronoun(text: str) -> bool:
    return _normalise(text) in PRONOUNS


def _span_in_document(span: str, document: str) -> bool:
    norm_span = _normalise(span)
    if not norm_span:
        return True
    return norm_span in _normalise(document)


def evaluate_prediction(
    example: GoldAttribution,
    prediction: AttributionPrediction,
    *,
    config: str,
    from_cache: bool,
    error: str = "",
) -> AttributionOutcome:
    pred_speaker_norm = _normalise(prediction.speaker)
    gold_speaker_norm = _normalise(example.gold_speaker)
    pred_verb_norm = _normalise(prediction.quotative_verb)
    gold_verb_norm = _normalise(example.gold_quotative_verb)
    speaker_is_pronoun = _is_pronoun(prediction.speaker)
    return AttributionOutcome(
        paragraph_id=example.paragraph_id,
        quote_index=example.quote_index,
        quote_kind=example.quote_kind,
        prompt_config=config,
        target_quote=example.target_quote,
        gold_speaker=example.gold_speaker,
        pred_speaker=prediction.speaker,
        gold_quotative_verb=example.gold_quotative_verb,
        pred_quotative_verb=prediction.quotative_verb,
        pred_speaker_refers_to=prediction.speaker_refers_to,
        speaker_exact=pred_speaker_norm == gold_speaker_norm,
        speaker_token_f1=_token_f1(prediction.speaker, example.gold_speaker),
        verb_exact=pred_verb_norm == gold_verb_norm,
        predicted_empty=not pred_speaker_norm,
        gold_empty=not gold_speaker_norm,
        speaker_is_pronoun=speaker_is_pronoun,
        pronoun_referent_present=(not speaker_is_pronoun) or bool(_normalise(prediction.speaker_refers_to)),
        speaker_in_document=_span_in_document(prediction.speaker, example.document),
        referent_in_document=_span_in_document(prediction.speaker_refers_to, example.document),
        verb_in_document=_span_in_document(prediction.quotative_verb, example.document),
        from_cache=from_cache,
        error=error,
    )


def _empty_prediction() -> AttributionPrediction:
    return AttributionPrediction(speaker="", speaker_refers_to="", quotative_verb="")


def summarize(outcomes: list[AttributionOutcome]) -> dict[str, Any]:
    total = len(outcomes)
    if not total:
        return {}

    attributed = [o for o in outcomes if not o.gold_empty]
    pronoun_predictions = [o for o in outcomes if o.speaker_is_pronoun]
    errors = [o for o in outcomes if o.error]
    copied = [o for o in outcomes if o.speaker_in_document and o.referent_in_document and o.verb_in_document]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "total": total,
        "attributed_total": len(attributed),
        "unknown_total": total - len(attributed),
        "speaker_exact_accuracy": mean([float(o.speaker_exact) for o in outcomes]),
        "speaker_exact_accuracy_attributed": mean([float(o.speaker_exact) for o in attributed]),
        "speaker_token_f1": mean([o.speaker_token_f1 for o in outcomes]),
        "speaker_token_f1_attributed": mean([o.speaker_token_f1 for o in attributed]),
        "unknown_accuracy": mean([float(o.predicted_empty) for o in outcomes if o.gold_empty]),
        "quotative_verb_exact_accuracy": mean([float(o.verb_exact) for o in outcomes]),
        "all_output_spans_in_document_rate": len(copied) / total,
        "pronoun_prediction_count": len(pronoun_predictions),
        "pronoun_referent_present_rate": mean([float(o.pronoun_referent_present) for o in pronoun_predictions]),
        "error_count": len(errors),
        "from_cache_count": sum(1 for o in outcomes if o.from_cache),
        "quote_kind_counts": dict(Counter(o.quote_kind for o in outcomes)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    examples = load_examples_with_context(args.input, context_tokens=args.context_tokens)
    if args.include_unknowns:
        selected = examples
    else:
        selected = [e for e in examples if e.quote_kind != "Unknown"]
    if args.limit:
        selected = selected[: args.limit]

    provider = args.provider
    if provider == "auto":
        provider = "google" if args.model.startswith("gemma-") else "openai"
    key_path = args.google_key_path if provider == "google" else args.key_path
    api_key = resolve_api_key(key_path=key_path)
    grouped_outcomes: dict[str, list[AttributionOutcome]] = defaultdict(list)
    detail_rows: list[dict[str, Any]] = []

    for config in args.configs:
        print(f"Evaluating {config} on {len(selected)} quotes...", flush=True)
        for idx, example in enumerate(selected, start=1):
            candidates = extract_speaker_candidates(example) if args.candidate_mode else []
            schema = (
                attribution_schema_for_candidates(candidates)
                if args.candidate_mode
                else ATTRIBUTION_SCHEMA
            )
            prompt_text = build_prompt(
                example,
                config,
                prompt_scope=args.prompt_scope,
                candidate_mode=args.candidate_mode,
                candidates=candidates,
            )
            key = _cache_key(
                args.model,
                config,
                example,
                prompt_text=prompt_text,
                provider=provider,
                prompt_scope=args.prompt_scope,
                context_tokens=args.context_tokens,
            )
            cache_dir = args.cache_root / provider / args.model / args.prompt_scope / f"{args.context_tokens}tok" / config
            cached = _read_cache(cache_dir, key) if args.cache_root else None
            from_cache = cached is not None
            raw_response: dict[str, Any] = {}
            error = ""
            if cached:
                pred = AttributionPrediction(**cached["prediction"])
            else:
                try:
                    call_fn = call_google if provider == "google" else call_openai
                    pred, raw_response = call_fn(
                        model=args.model,
                        api_key=api_key,
                        prompt=prompt_text,
                        timeout=args.timeout,
                        max_retries=args.max_retries,
                        max_output_tokens=args.max_output_tokens,
                        schema=schema,
                    )
                    if args.cache_root:
                        _write_cache(
                            cache_dir,
                            key,
                            {
                                "prediction": asdict(pred),
                                "response_id": raw_response.get("id", ""),
                                "created": raw_response.get("created", 0),
                                "usage": raw_response.get("usage", {}),
                            },
                        )
                except Exception as exc:
                    pred = _empty_prediction()
                    error = f"{type(exc).__name__}: {exc}"
            outcome = evaluate_prediction(
                example,
                pred,
                config=config,
                from_cache=from_cache,
                error=error,
            )
            grouped_outcomes[config].append(outcome)
            row = asdict(outcome)
            if args.candidate_mode:
                row["speaker_candidates"] = candidates
                row["gold_speaker_candidate_overlap"] = [
                    candidate
                    for candidate in candidates
                    if set(_tokens(candidate)).intersection(_tokens(example.gold_speaker))
                ]
            if raw_response:
                row["usage"] = raw_response.get("usage", {})
            detail_rows.append(row)
            if args.progress_every and idx % args.progress_every == 0:
                summary = summarize(grouped_outcomes[config])
                print(
                    f"  {idx}/{len(selected)} "
                    f"speaker_acc={summary['speaker_exact_accuracy_attributed']:.3f} "
                    f"token_f1={summary['speaker_token_f1_attributed']:.3f}",
                    flush=True,
                )

    summaries = {
        config: summarize(outcomes)
        for config, outcomes in grouped_outcomes.items()
    }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "model": args.model,
        "provider": provider,
        "configs": args.configs,
        "prompt_scope": args.prompt_scope,
        "context_tokens": args.context_tokens,
        "candidate_mode": args.candidate_mode,
        "limit": args.limit,
        "include_unknowns": args.include_unknowns,
        "summary": summaries,
        "details": detail_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=Path("data/directquote_attribution_llm.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "google"],
        default="auto",
        help="API provider. Defaults to google for gemma-* model names and openai otherwise.",
    )
    parser.add_argument("--key-path", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--google-key-path", type=Path, default=DEFAULT_GOOGLE_KEY_PATH)
    parser.add_argument(
        "--prompt-scope",
        choices=["document", "context"],
        default="document",
        help="Use the full marked paragraph document or only left/right context around the target quote.",
    )
    parser.add_argument("--context-tokens", type=int, default=45)
    parser.add_argument(
        "--candidate-mode",
        action="store_true",
        help="Extract candidate speaker mentions and force speaker to one candidate or None.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["direct", "direct_repeat", "extraction", "directquote_labels"],
        choices=["direct", "direct_repeat", "extraction", "directquote_labels"],
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--include-unknowns", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=Path(".quote_merger_cache/directquote_attribution"))
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)

    summary = {k: v for k, v in result.items() if k != "details"}
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    detail_path = args.output.with_name(args.output.stem + "_detail.json")
    detail_path.write_text(json.dumps(result["details"], indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {args.output}")
    print(f"Wrote detail to {detail_path}")


if __name__ == "__main__":
    main()
