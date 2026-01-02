"""QuoteGraph - Named Entity Recognition for Quote Attribution."""

from quotegraph.entity_extractor import (
    build_trie_from_qid2alias,
    find_entities,
    find_entities_with_offsets,
    load_trie_from_mapping,
    TrieHolder,
)

__all__ = [
    "build_trie_from_qid2alias",
    "find_entities",
    "find_entities_with_offsets",
    "load_trie_from_mapping",
    "TrieHolder",
]
