"""QuoteGraph - Named Entity Recognition for Quote Attribution."""

from quotegraph.attribution_aggregate import AggregatedAttribution, aggregate_local_probas
from quotegraph.candidate_selection import (
    DEFAULT_BRIDGE_WINDOW_WORDS,
    filter_candidate_indices,
    quote_is_candidate,
)
from quotegraph.entity_extractor import (
    TrieHolder,
    build_trie_from_qid2alias,
    find_entities,
    find_entities_with_offsets,
    load_trie_from_mapping,
)
from quotegraph.quote_merger import (
    AtomicPairContext,
    Confidence,
    DEFAULT_LOGPROB_MARGIN,
    DiskPairCache,
    MergeAnswer,
    MergeBlock,
    MergeDecision,
    OpenAICompatibleAdjudicator,
    OpenAIQuoteMergeAdjudicator,
    PairDecision,
    QuoteCandidate,
    QuoteRole,
    QuoteTurnMerger,
    ReasonCode,
    SpeakerContinuity,
    SpeakerTop,
)
from quotegraph.quotebank_loader import extract_quotes_and_bridges

__all__ = [
    "AggregatedAttribution",
    "AtomicPairContext",
    "Confidence",
    "DEFAULT_BRIDGE_WINDOW_WORDS",
    "DEFAULT_LOGPROB_MARGIN",
    "DiskPairCache",
    "MergeAnswer",
    "MergeBlock",
    "MergeDecision",
    "OpenAICompatibleAdjudicator",
    "OpenAIQuoteMergeAdjudicator",
    "PairDecision",
    "QuoteCandidate",
    "QuoteRole",
    "QuoteTurnMerger",
    "ReasonCode",
    "SpeakerContinuity",
    "SpeakerTop",
    "TrieHolder",
    "aggregate_local_probas",
    "build_trie_from_qid2alias",
    "extract_quotes_and_bridges",
    "filter_candidate_indices",
    "find_entities",
    "find_entities_with_offsets",
    "load_trie_from_mapping",
    "quote_is_candidate",
]
