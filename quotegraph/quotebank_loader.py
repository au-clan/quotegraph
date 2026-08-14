"""Extract aligned quote spans and bridges from Quotebank article records."""

from __future__ import annotations

import ast
from typing import Any, Mapping

from quotegraph.quote_merger import QuoteCandidate, SpeakerTop
from quotegraph.utils import get_ends_dict


def _speaker_top_from_quotation(quotation: Mapping[str, Any]) -> SpeakerTop:
    name = quotation.get("globalTopSpeaker")
    probas = quotation.get("globalProbas") or quotation.get("localProbas") or []
    best_qid = None
    best_prob = 0.0
    for entry in probas:
        if entry.get("speaker") != name:
            continue
        qids = entry.get("qids") or []
        prob = float(entry.get("probability") or entry.get("proba") or 0.0)
        if qids and prob >= best_prob:
            best_qid = qids[0]
            best_prob = prob
    if best_qid is None and name:
        for entry in probas:
            qids = entry.get("qids") or []
            prob = float(entry.get("probability") or entry.get("proba") or 0.0)
            if qids and prob > best_prob:
                best_qid = qids[0]
                best_prob = prob
                name = entry.get("speaker") or name
    return SpeakerTop(name=name, qid=best_qid, probability=best_prob)


def _local_probas(quotation: Mapping[str, Any]) -> tuple[tuple[str, str, float], ...]:
    rows: list[tuple[str, str, float]] = []
    for entry in quotation.get("localProbas") or quotation.get("globalProbas") or []:
        qids = entry.get("qids") or []
        if not qids:
            continue
        rows.append(
            (
                str(entry.get("speaker") or ""),
                str(qids[0]),
                float(entry.get("probability") or entry.get("proba") or 0.0),
            )
        )
    return tuple(rows)


def _quote_text(quotation: Mapping[str, Any]) -> str:
    for key in ("quotation", "quote", "text"):
        value = quotation.get(key)
        if value:
            return str(value)
    return ""


def _mentioned_entities(quotation: Mapping[str, Any], article: Mapping[str, Any]) -> tuple[str, ...]:
    qids: set[str] = set()
    for mention in quotation.get("mentions") or []:
        qid = mention.get("qid") or mention.get("wikidata_id")
        if qid:
            qids.add(str(qid))
    quote_id = quotation.get("quoteID")
    if quote_id and article.get("names"):
        for name in article["names"]:
            try:
                ids = ast.literal_eval(name.get("ids", "[]"))
            except (SyntaxError, ValueError, TypeError):
                ids = []
            if not ids:
                continue
            offsets = name.get("offsets")
            if offsets is None:
                continue
            qids.update(str(q) for q in ids)
    return tuple(sorted(qids))


def extract_quotes_and_bridges(
    article: Mapping[str, Any],
    *,
    detokenized: bool = False,
) -> tuple[list[QuoteCandidate], list[str]]:
    """Build ``quotes`` and ``bridges`` lists from one Quotebank article."""

    content = article.get("detokenized_content") if detokenized else article.get("content")
    if not content:
        content = article.get("content") or article.get("detokenized_content") or ""

    quotations = list(article.get("quotations") or [])
    if not quotations:
        return [], []

    quotations.sort(key=lambda q: q.get("quotationOffset", 0))
    tokens = content.split(" ") if not detokenized else content.split()
    ends = get_ends_dict(content, quotations)

    quotes: list[QuoteCandidate] = []
    for quotation, end_tok in zip(quotations, ends):
        speaker = _speaker_top_from_quotation(quotation)
        quotes.append(
            QuoteCandidate(
                text=_quote_text(quotation),
                quote_id=quotation.get("quoteID"),
                speaker=speaker.name,
                speaker_qid=speaker.qid,
                speaker_probability=speaker.probability,
                mentioned_entities=_mentioned_entities(quotation, article),
                local_probas=_local_probas(quotation),
                metadata={"quotationOffset": quotation.get("quotationOffset")},
            )
        )

    bridges: list[str] = []
    for idx in range(len(quotations) - 1):
        left_end = ends[idx]
        right_start = quotations[idx + 1].get("quotationOffset", 0)
        open_mark = right_start - 1
        if left_end == -1 or open_mark < 0 or open_mark >= len(tokens):
            bridges.append("")
            continue
        bridge_tokens = tokens[left_end + 1 : open_mark]
        bridges.append(" ".join(bridge_tokens))

    return quotes, bridges


def bridge_mentioned_entities(
    bridge: str,
    article: Mapping[str, Any],
    *,
    window_words: int = 10,
) -> frozenset[str]:
    """Placeholder for bridge entity linking; callers can override with NER output."""

    _ = (bridge, article, window_words)
    return frozenset()
