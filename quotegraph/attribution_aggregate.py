"""Re-aggregate Quotebank speaker probabilities across merged quote spans."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from quotegraph.quote_merger import QuoteCandidate, SpeakerTop


@dataclass(frozen=True)
class AggregatedAttribution:
    speaker: SpeakerTop
    local_probas: tuple[tuple[str, str, float], ...]


def aggregate_local_probas(
    quotes: list[QuoteCandidate],
) -> AggregatedAttribution:
    """Sum local probas across spans and pick the argmax QID."""

    totals: dict[tuple[str, str], float] = defaultdict(float)
    for quote in quotes:
        for name, qid, prob in quote.local_probas:
            totals[(name, qid)] += prob

    if not totals:
        speaker_name = next((q.speaker for q in quotes if q.speaker), None)
        speaker_qid = next((q.speaker_qid for q in quotes if q.speaker_qid), None)
        speaker_prob = max((q.speaker_probability for q in quotes), default=0.0)
        return AggregatedAttribution(
            speaker=SpeakerTop(speaker_name, speaker_qid, speaker_prob),
            local_probas=(),
        )

    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    (name, qid), prob = ranked[0]
    local_probas = tuple((n, q, p) for (n, q), p in ranked)
    return AggregatedAttribution(
        speaker=SpeakerTop(name, qid, prob),
        local_probas=local_probas,
    )


def aggregate_mentions(
    quotes: list[QuoteCandidate],
    bridges: list[str],
    *,
    bridge_mentions: list[tuple[str, ...]] | None = None,
    left_role: str = "utterance",
    right_role: str = "utterance",
) -> frozenset[str]:
    """Union in-quote mentions; include bridge mentions for scare-role units."""

    mentions: set[str] = set()
    for quote in quotes:
        mentions.update(quote.mentioned_entities)

    if not bridges:
        return frozenset(mentions)

    bridge_mentions = bridge_mentions or [() for _ in bridges]
    if left_role == "scare" and bridge_mentions:
        mentions.update(bridge_mentions[0])
    if right_role == "scare" and bridge_mentions:
        mentions.update(bridge_mentions[-1])
    return frozenset(mentions)
