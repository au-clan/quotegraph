"""
Gazetteer-based Named Entity Recognizer for Quote Attribution.

Uses the Quootstrap two-pass methodology (Pavllo et al.):
1. First pass: Find full name matches from the trie
2. Second pass: Resolve partial mentions as unambiguous suffixes of identified names

NOTE: Text is assumed to be pre-tokenized with Stanford CoreNLP tokens,
joined by spaces (e.g., "Joe Biden , the president ." with punctuation separated).
"""

import argparse
import ast
import re
import string
from pathlib import Path
from typing import Any

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import BooleanType
from pygtrie import StringTrie  # type: ignore

# =============================================================================
# Constants
# =============================================================================

PUNCT = "".join(x for x in string.punctuation if x not in "[]")
TARGET_QUOTE_TOKEN = "[TARGET_QUOTE]"
MASK_TOKEN = "[MASK]"

DEFAULT_BLACKLIST_PATH = Path(__file__).parent.parent / "data" / "title_blacklist.txt"

# Stopwords to skip when looking for partial mentions
STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "dare", "ought", "used", "it", "its", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "mine", "yours", "hers", "ours",
    "theirs", "who", "whom", "whose", "which", "what", "where", "when",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "also", "now", "here", "there",
    "then", "once", "if", "because", "although", "though", "while",
    "about", "after", "before", "between", "into", "through", "during",
    "above", "below", "up", "down", "out", "off", "over", "under", "again",
    "further", "any", "said", "says", "say", "told", "tell", "according",
})


def load_blacklist(path: str | Path | None = None) -> set[str]:
    """Load honorifics/titles blacklist from file (one word per line)."""
    path = path or DEFAULT_BLACKLIST_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {
                line.strip().lower()
                for line in f
                if line.strip() and not line.startswith("#")
            }
    except FileNotFoundError:
        return {"president", "manager"}


BLACK_LIST = load_blacklist()


# =============================================================================
# Phase Transformations (Quotebank encoding issues)
# =============================================================================

PHASE_TRANSFORMERS = {
    "A": lambda a: a.lower().encode("utf-8").decode("Latin-1"),
    "B": lambda a: "".join(c if ord(c) < 256 else "?" for c in a.lower()),
    "C": lambda a: "".join(c if ord(c) < 128 else "?" for c in a.lower()),
    "D": lambda a: "".join(c if ord(c) < 128 else "?" for c in a),
    "E": lambda a: a,
}


# =============================================================================
# Trie Building
# =============================================================================

def _add_to_trie(trie: StringTrie, key: str, qid: str, metadata: Any = None) -> None:
    """Add entry to trie, handling multiple entities with same alias."""
    val = (qid, metadata)
    if key not in trie:
        trie[key] = val
    else:
        existing = trie[key]
        if isinstance(existing, list):
            if not any(e[0] == qid for e in existing):
                existing.append(val)
        elif existing[0] != qid:
            trie[key] = [existing, val]


def build_trie_from_qid2alias(
    qid2alias: dict[str, list[str]],
    phase: str | None = None,
) -> StringTrie:
    """
    Build trie from QID -> aliases mapping.
    
    Single-word aliases are only indexed if they match the entity's label
    (first item in aliases list). This prevents false positives from common
    short names like "Obama" matching in the first pass - they'll still
    match as partial mentions if the full name was found.
    """
    trie = StringTrie(delimiter="/")
    transformer = PHASE_TRANSFORMERS.get(phase, PHASE_TRANSFORMERS["E"]) if phase else lambda x: x

    for qid, aliases in qid2alias.items():
        if len(aliases) == 0:
            continue
        
        # First alias is the label
        
        for alias in aliases:
            if not alias or not alias.strip():
                continue
            tokens = transformer(alias).split()
            if not tokens:
                continue
            
            key = "/".join(tokens).lower()
            
            # Skip single-word aliases unless they ARE the label
            if len(tokens) == 1:
                continue
            
            _add_to_trie(trie, key, qid)

    return trie


def load_trie_from_mapping(
    alias_to_qids: dict[str, list[str]],
    qid_to_label: dict[str, str] | None = None,
) -> StringTrie:
    """
    Create trie from alias -> QIDs mapping.
    
    If qid_to_label is provided, single-word aliases are only indexed
    if they match the entity's label.
    """
    trie = StringTrie(delimiter="/")
    
    for alias, qids in alias_to_qids.items():
        tokens = alias.split()
        if not tokens:
            continue
        
        key = "/".join(tokens).lower()
        is_single_word = len(tokens) == 1
        
        for qid in qids:
            # Skip single-word aliases unless they match the label
            if is_single_word and qid_to_label:
                label = qid_to_label.get(qid, "")
                label_key = "/".join(label.split()).lower() if label else ""
                if key != label_key:
                    continue
            
            _add_to_trie(trie, key, qid)
    
    return trie


def create_trie(names: list[tuple]) -> StringTrie:
    """
    Create trie from Spark (name, qid) tuples format.
    
    Single-word names are only indexed if they're the first (label) name
    seen for that QID in the names list.
    """
    trie = StringTrie(delimiter="/")
    
    # First pass: collect entries and track labels (first name per QID)
    qid_labels: dict[str, str] = {}  # qid -> label_key
    entries: list[tuple] = []  # (tokens, key, q_name, qid_full)
    
    for name, qid in names:
        try:
            q_name = ast.literal_eval(qid[0])[1]
        except (ValueError, SyntaxError, IndexError, TypeError):
            continue
        
        tokens = name.split()
        if not tokens:
            continue
        
        key = "/".join(tokens).lower()
        
        # First name seen for this QID is treated as the label
        if q_name not in qid_labels:
            qid_labels[q_name] = key
        
        entries.append((tokens, key, q_name, qid))
    
    # Second pass: add to trie, skipping single-word non-labels
    for tokens, key, q_name, qid in entries:
        if len(tokens) == 1 and key != qid_labels.get(q_name):
            continue
        _add_to_trie(trie, key, q_name, qid)
    
    return trie


class TrieHolder:
    """Cache tries per phase for Spark partitions."""

    def __init__(self, qid2alias: dict[str, list[str]]):
        self.qid2alias = qid2alias
        self._tries: dict[str, StringTrie] = {}

    def get_trie(self, phase: str | None = None) -> StringTrie:
        cache_key = phase or "_default_"
        if cache_key not in self._tries:
            self._tries[cache_key] = build_trie_from_qid2alias(self.qid2alias, phase)
        return self._tries[cache_key]


# =============================================================================
# Entity Extraction - Core Two-Pass Logic
# =============================================================================

def _extract_qids(value: Any) -> list[str]:
    """Extract QIDs from trie value."""
    if isinstance(value, tuple):
        return [value[0]]
    if isinstance(value, list):
        return [v[0] if isinstance(v, tuple) else v for v in value]
    return [str(value)]


def _find_full_matches(tokens: list[str], trie: StringTrie) -> tuple[list, dict]:
    """First pass: Find all full name matches."""
    full_matches = []
    matched_full_names = {}

    i = 0
    while i < len(tokens):
        best_match, best_end = None, i

        for j in range(i + 1, len(tokens) + 1):
            key = "/".join(tokens[i:j]).lower()
            if trie.has_key(key):
                best_match = (i, j, key, trie[key])
                best_end = j
            elif not trie.has_subtrie(key):
                break

        if best_match:
            full_matches.append(best_match)
            matched_full_names[best_match[2]] = best_match[3]
            i = best_end
        else:
            i += 1

    return full_matches, matched_full_names


def _find_partial_matches(
    tokens: list[str],
    matched_full_names: dict[str, Any],
    matched_positions: set[int],
    blacklist: set[str],
) -> list[tuple]:
    """
    Second pass: Resolve partial mentions as unambiguous suffixes.
    
    Skips stopwords and honorifics. Only matches if exactly ONE
    identified full name ends with the partial mention.
    """
    partial_matches = []
    skip_words = STOPWORDS | blacklist

    i = 0
    while i < len(tokens):
        if i in matched_positions or tokens[i].lower() in skip_words or len(tokens[i]) <= 1:
            i += 1
            continue

        best_partial, best_end = None, i

        for j in range(i + 1, len(tokens) + 1):
            if any(pos in matched_positions for pos in range(i, j)):
                break

            partial_tokens = tokens[i:j]
            # Skip if all tokens are stopwords/honorifics
            if not any(t.lower() not in skip_words and len(t) > 1 for t in partial_tokens):
                continue

            partial_key = "/".join(partial_tokens).lower()
            matches = [
                (fk, fv) for fk, fv in matched_full_names.items()
                if fk.endswith("/" + partial_key) or fk == partial_key
            ]

            if len(matches) == 1:
                best_partial = (i, j, matches[0][0], matches[0][1])
                best_end = j
            elif len(matches) > 1:
                break  # Ambiguous

        if best_partial:
            partial_matches.append(best_partial)
            i = best_end
        else:
            i += 1

    return partial_matches


# =============================================================================
# Public API
# =============================================================================

def find_entities_with_offsets(text: str, trie: StringTrie) -> list[dict[str, Any]]:
    """
    Find entities in text with token positions.
    
    Args:
        text: Space-separated tokens
        trie: StringTrie from build_trie_from_qid2alias()
    
    Returns:
        List of {"name", "start_token", "end_token", "qids"} dicts
    """
    if not text or not text.strip():
        return []

    tokens = text.split()
    if not tokens:
        return []

    full_matches, matched_full_names = _find_full_matches(tokens, trie)
    matched_positions = {pos for s, e, _, _ in full_matches for pos in range(s, e)}
    partial_matches = _find_partial_matches(tokens, matched_full_names, matched_positions, BLACK_LIST)

    return [
        {
            "name": " ".join(tokens[s:e]),
            "start_token": s,
            "end_token": e,
            "qids": _extract_qids(v),
        }
        for s, e, _, v in sorted(full_matches + partial_matches, key=lambda x: x[0])
    ]


# Spark output schema
ENTITY_STRUCT = T.StructType([
    T.StructField("name", T.StringType(), nullable=False),
    T.StructField("start_token", T.IntegerType(), nullable=False),
    T.StructField("end_token", T.IntegerType(), nullable=False),
    T.StructField("qids", T.ArrayType(T.StringType()), nullable=False),
])


# =============================================================================
# CoreNLP Text Processing (for Quotebank pipeline)
# =============================================================================

def _fix_special_tokens(tokens: list[str]) -> list[str]:
    """Reconstruct bracketed tokens like [TARGET_QUOTE]."""
    out, current = [], ""
    for token in tokens:
        if token == "[":
            current = "["
        elif token == "]":
            out.append(current + "]")
            current = ""
        elif current:
            current += token
        else:
            out.append(token)
    return out


def _fix_punct_tokens(tokens: list[str]) -> list[str]:
    """Filter punctuation tokens for CoreNLP text."""
    return [
        t for t in tokens
        if t and (
            (t.startswith("[") and t.endswith("]")) or
            (not all(c in PUNCT for c in t) and t not in ("'s", "s'", "'", "n't"))
        )
    ]


def find_entities(
    text: str,
    trie: StringTrie,
    mask: str = MASK_TOKEN,
) -> tuple[str, dict[str, tuple[list[int], Any]]]:
    """
    Find and mask entities in CoreNLP pre-tokenized text.
    
    Returns:
        (masked_text, entities_dict) where entities_dict maps QID to ([positions], metadata)
    """
    if not text or not text.strip():
        return "", {}

    tokens = _fix_punct_tokens(_fix_special_tokens(text.split()))
    if not tokens:
        return "", {}

    full_matches, matched_full_names = _find_full_matches(tokens, trie)
    matched_positions = {pos for s, e, _, _ in full_matches for pos in range(s, e)}
    partial_matches = _find_partial_matches(tokens, matched_full_names, matched_positions, BLACK_LIST)

    all_matches = sorted(full_matches + partial_matches, key=lambda x: x[0])
    pos_to_entity = {s: (e, v) for s, e, _, v in all_matches}

    count, entities, out = 1, {}, []
    i = 0
    while i < len(tokens):
        if i in pos_to_entity:
            end, value = pos_to_entity[i]
            entities[count] = value
            count += 1
            out.append(mask)
            i = end
        else:
            out.append(tokens[i])
            i += 1

    masked = "".join(" " + t if not t.startswith("'") and t not in PUNCT else t for t in out).strip()

    # Reduce: group by QID
    reduced = {}
    for pos, value in entities.items():
        for item in (value if isinstance(value, list) else [value]):
            try:
                qid, meta = item[0], item[1]
                if qid in reduced:
                    reduced[qid][0].append(pos)
                else:
                    reduced[qid] = ([pos], meta)
            except (IndexError, TypeError):
                continue

    return masked, reduced


# Backwards compatibility
find_entites = find_entities


# =============================================================================
# Quotebank Spark Pipeline
# =============================================================================

def get_targets(entities: dict, target: str) -> tuple[list[int], bool]:
    """Get target entity positions and ambiguity flag."""
    targets = entities.get(target)
    return (targets[0], len(targets[0]) > 1) if targets else ([0], False)


def check_speaker_in_entities(speaker: str, names: list[tuple]) -> bool:
    """Check if speaker Q-ID exists in names list."""
    if speaker in ("-1", "none", "not_quote", "not_mentioned", "not_en", "ambiguous", "other"):
        return True
    for name, qid in names:
        try:
            if ast.literal_eval(qid[0])[1] == speaker:
                return True
        except (ValueError, SyntaxError, IndexError, TypeError):
            continue
    return False


def transform(x: Row) -> Row | None:
    """Transform quote row for training (with speaker labels)."""
    if not check_speaker_in_entities(x.speaker, x.names):
        return None

    trie = create_trie(x.names)
    full_text = re.sub(r"\"+", "", " ".join([x.leftContext, TARGET_QUOTE_TOKEN, x.rightContext]))

    try:
        masked_text, entities = find_entities(full_text, trie)
    except Exception:
        return None

    targets, ambiguous = get_targets(entities, x.speaker)
    return Row(
        articleUID=x.articleUID, articleOffset=x.articleOffset,
        speaker=x.speaker, quotation=x.quotation,
        full_text=full_text, masked_text=masked_text,
        entities=entities, targets=targets, ambiguous=ambiguous,
        domain=getattr(x, "domain", ""), pattern=getattr(x, "pattern", ""),
    )


def transform_test(x: Row) -> Row | None:
    """Transform quote row for test (without speaker labels)."""
    trie = create_trie(x.names)
    full_text = re.sub(r"\"+", "", " ".join([x.leftContext, TARGET_QUOTE_TOKEN, x.rightContext]))

    try:
        masked_text, entities = find_entities(full_text, trie)
    except Exception:
        return None

    return Row(
        articleUID=x.articleUID, articleOffset=x.articleOffset,
        quotation=x.quotation, full_text=full_text,
        masked_text=masked_text, entities=entities,
    )


@F.udf(returnType=BooleanType())
def is_all_lower(masked_text: str | None) -> bool:
    """Check if text (excluding special tokens) is all lowercase."""
    if not masked_text:
        return True
    return re.sub(r"\[MASK\]|\[QUOTE\]|\[TARGET_QUOTE\]", "", masked_text).islower()


def extract_entities(
    spark: SparkSession,
    *,
    merged_path: str,
    speakers_path: str,
    output_path: str,
    nb_partition: int = 200,
    compression: str = "gzip",
    ftype: str = "parquet",
    kind: str = "train",
) -> None:
    """Main Quotebank pipeline: extract entities from quotes."""
    df = (
        spark.read.parquet(merged_path) if ftype == "parquet"
        else spark.read.json(merged_path).repartition(nb_partition)
    ).dropna(subset=["quotation"])

    joined = df.join(spark.read.json(speakers_path), on="articleUID")
    transform_fn = transform if kind == "train" else transform_test

    transformed = (
        joined.rdd.map(transform_fn)
        .filter(lambda x: x is not None)
        .toDF()
        .withColumn("nb_entities", F.size("entities"))
        .filter("nb_entities > 0")
    )

    if kind == "train":
        transformed = transformed.withColumn("nb_targets", F.size("targets"))

    transformed.write.parquet(output_path, "overwrite", compression=compression)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract entities from Quotebank quotes")
    parser.add_argument("-m", "--merged", required=True, help="Path to merged quotes")
    parser.add_argument("-s", "--speakers", required=True, help="Path to speakers JSON")
    parser.add_argument("-o", "--output", required=True, help="Output path")
    parser.add_argument("--kind", required=True, choices=["train", "test"])
    parser.add_argument("-l", "--local", action="store_true")
    parser.add_argument("-n", "--nb_partition", type=int, default=200)
    parser.add_argument("--compression", default="gzip")
    parser.add_argument("--ftype", default="parquet", choices=["parquet", "json"])
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .master("local[24]").appName("EntityExtractorLocal")
        .config("spark.driver.memory", "16g")
        .config("spark.executor.memory", "32g")
        .getOrCreate()
    ) if args.local else SparkSession.builder.appName("EntityExtractor").getOrCreate()

    extract_entities(
        spark,
        merged_path=args.merged,
        speakers_path=args.speakers,
        output_path=args.output,
        nb_partition=args.nb_partition,
        compression=args.compression,
        ftype=args.ftype,
        kind=args.kind,
    )
