"""
Gazetteer-based Named Entity Recognizer for Quote Attribution.

Matches PERSON entity names from a Wikidata gazetteer against text contexts
surrounding quotes. First-name references require prior full-name activation.
"""

import argparse
import ast
import logging
import re
import string

import pyspark.sql.functions as F
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import BooleanType
from pygtrie import StringTrie  # type: ignore

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PUNCT = "".join(x for x in string.punctuation if x not in "[]")
TARGET_QUOTE_TOKEN = "[TARGET_QUOTE]"
MASK_TOKEN = "[MASK]"
MIN_NAME_LEN = 3

BLACK_LIST = {"president", "manager"}
HONORIFICS = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "dame", "lord", "lady",
    "rev", "hon", "sen", "rep", "gov", "gen", "col", "maj", "capt", "lt",
    "sgt", "rabbi", "imam", "fr", "brother", "sister", "saint", "st",
    "jr", "sr", "ii", "iii", "iv", "v", "esq", "phd", "md", "dds", "jd",
}
SPECIAL_MARKERS = {"-1", "none", "not_quote", "not_mentioned", "not_en", "ambiguous", "other"}


def flatten_entities(value):
    """Yield (q_name, qid, is_first_name_only) tuples from nested structure."""
    if isinstance(value, tuple) and len(value) >= 2 and isinstance(value[0], str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from flatten_entities(item)


def normalize_text(text):
    """Normalize hyphens and initials."""
    if not text:
        return ""
    text = re.sub(r"(?<!\[)(\w)-(\w)(?!\])", r"\1 \2", text)
    text = re.sub(r"\b([A-Za-z])\.\s*", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def filter_tokens(tokens):
    """Filter out blacklisted words and honorifics."""
    return [t for t in tokens if t.lower() not in BLACK_LIST and t.lower().rstrip(".") not in HONORIFICS]


def create_trie(names):
    """Create trie from entity names with first-name activation tracking."""
    trie = StringTrie(delimiter="/")

    def add(key, q_name, qid, first_only):
        val = (q_name, qid, first_only)
        if key not in trie:
            trie[key] = val
        else:
            existing = trie[key]
            if isinstance(existing, list):
                if not any(e[0] == q_name for e in existing):
                    existing.append(val)
            elif existing[0] != q_name:
                trie[key] = [existing, val]

    def index_name(first, last, q_name, qid):
        """Add name variants to trie."""
        if first and last and len(first) >= 2 and len(last) >= 2:
            add(f"{first}/{last}".lower(), q_name, qid, False)
        if first and len(first) >= MIN_NAME_LEN:
            add(first.lower(), q_name, qid, True)  # First-name requires activation
        if last and len(last) >= MIN_NAME_LEN:
            add(last.lower(), q_name, qid, False)

    for name, qid in names:
        try:
            q_name = ast.literal_eval(qid[0])[1]
        except (ValueError, SyntaxError, IndexError, TypeError):
            continue
        if not name or not name.strip():
            continue

        tokens = filter_tokens(normalize_text(name).split())
        if not tokens:
            continue

        first = tokens[0] if len(tokens) > 1 else None
        last = tokens[-1] if tokens else None
        index_name(first, last, q_name, qid)

        # Handle hyphenated variants
        if "-" in name:
            orig_tokens = filter_tokens(name.split())
            if orig_tokens:
                orig_first = orig_tokens[0] if len(orig_tokens) > 1 else None
                orig_last = orig_tokens[-1]
                index_name(orig_first, orig_last, q_name, qid)

    return trie


def fix_punct_tokens(tokens):
    """Separate punctuation and strip honorifics."""
    out = []
    for token in tokens:
        if not token:
            continue
        if token.startswith("[") and token.endswith("]"):
            out.append(token)
            continue
        while token and token[0] in PUNCT:
            token = token[1:]
        if not token or token.lower().rstrip(".") in HONORIFICS:
            continue
        if token[-1] in PUNCT:
            if token[:-1]:
                out.append(token[:-1])
            out.append(token[-1])
        elif token.endswith("'s"):
            if token[:-2]:
                out.append(token[:-2])
            out.append("'s")
        elif token.endswith("s'"):
            if token[:-1]:
                out.append(token[:-1])
            out.append("'")
        else:
            out.append(token)
    return out


def find_entities(text, trie, mask=MASK_TOKEN, require_caps=True):
    """Find and mask named entities. First-name refs require prior full-name match."""
    if not text or not text.strip():
        return "", {}

    tokens = fix_punct_tokens(normalize_text(text).split())
    if not tokens:
        return "", {}

    activated = set()
    entities, out = {}, []
    start, count = 0, 1

    def filter_match(match):
        """Filter candidates by activation status. Returns (filtered, has_full_name)."""
        if not match:
            return None, False
        filtered, has_full = [], False
        for e in flatten_entities(match):
            is_first_only = e[2] if len(e) > 2 else False
            if not is_first_only:
                has_full = True
                filtered.append(e)
            elif e[0] in activated:
                filtered.append(e)
        return (filtered[0] if len(filtered) == 1 else filtered) if filtered else None, has_full

    def accept(matched_toks):
        """Check capitalization for single-token matches."""
        if not require_caps or len(matched_toks) > 1:
            return True
        t = matched_toks[0]
        return t[0].isupper() if t and not t.startswith("[") else True

    def activate(match):
        for e in flatten_entities(match):
            activated.add(e[0])

    def record_match(match, matched_toks, is_full):
        nonlocal count
        entities[count] = match
        count += 1
        out.append(mask)
        if len(matched_toks) > 1 or is_full:
            activate(match)

    i = 0
    while i < len(tokens):
        key = "/".join(tokens[start:i + 1]).lower()
        if not key:
            out.append(tokens[i])
            start = i + 1
            i += 1
            continue

        try:
            if trie.has_subtrie(key):
                if i == len(tokens) - 1:
                    matched = tokens[start:i + 1]
                    filtered, is_full = filter_match(list(trie[key:]))
                    if filtered and accept(matched):
                        entities[count] = filtered
                        out.append(mask)
                        if is_full:
                            activate(filtered)
                    else:
                        out.extend(matched)
                i += 1
            elif key in trie:
                matched = tokens[start:i + 1]
                filtered, is_full = filter_match(trie[key])
                if filtered and accept(matched):
                    record_match(filtered, matched, is_full)
                    start = i + 1
                    i += 1
                else:
                    out.append(tokens[start])
                    start += 1
            elif start < i:
                old_key = "/".join(tokens[start:i]).lower()
                matched = tokens[start:i]
                filtered, is_full = filter_match(list(trie[old_key:]) if old_key else None)
                if filtered and accept(matched):
                    record_match(filtered, matched, is_full)
                else:
                    out.extend(matched)
                start = i if trie.has_node(tokens[i].lower()) else i + 1
                if start == i + 1:
                    out.append(tokens[i])
                i += 1
            else:
                out.append(tokens[i])
                start = i + 1
                i += 1
        except (KeyError, TypeError):
            out.append(tokens[i])
            start = i + 1
            i += 1

    if not out:
        return "", {}

    result = "".join(" " + t if not t.startswith("'") and t not in PUNCT else t for t in out).strip()

    # Group by Q-name
    grouped = {}
    for idx, val in entities.items():
        for e in flatten_entities(val):
            q_name, qinfo = e[0], e[1]
            if q_name in grouped:
                grouped[q_name][0].append(idx)
            else:
                grouped[q_name] = ([idx], qinfo)

    return result, grouped


find_entites = find_entities  # Backwards compatibility


def get_targets(entities, target):
    """Get target entity positions and ambiguity flag."""
    t = entities.get(target)
    if t and isinstance(t, tuple) and t[0] and isinstance(t[0], list):
        return t[0], len(t[0]) > 1
    return [0], False


def _transform(x, include_speaker=True):
    """Transform quote row by extracting entities."""
    try:
        names = x.names
        left, right = x.leftContext, x.rightContext
        speaker = x.speaker if include_speaker else None
    except AttributeError:
        return None

    if include_speaker:
        # Check speaker in entities
        if speaker not in SPECIAL_MARKERS:
            found = False
            for name, qid in (names or []):
                try:
                    if ast.literal_eval(qid[0])[1] == speaker:
                        found = True
                        break
                except (ValueError, SyntaxError, IndexError, TypeError):
                    continue
            if not found:
                return None

    trie = create_trie(names)
    full_text = re.sub(r"\"+", "", f"{left} {TARGET_QUOTE_TOKEN} {right}")

    try:
        masked_text, entities = find_entities(full_text, trie)
    except Exception:
        return None

    if include_speaker:
        targets, ambiguous = get_targets(entities, speaker)
        return Row(
            articleUID=x.articleUID, articleOffset=x.articleOffset, speaker=speaker,
            quotation=x.quotation, full_text=full_text, masked_text=masked_text,
            entities=entities, targets=targets, ambiguous=ambiguous,
            domain=getattr(x, "domain", ""), pattern=getattr(x, "pattern", ""),
        )
    return Row(
        articleUID=x.articleUID, articleOffset=x.articleOffset,
        quotation=x.quotation, full_text=full_text, masked_text=masked_text, entities=entities,
    )


def transform(x):
    return _transform(x, include_speaker=True)


def transform_test(x):
    return _transform(x, include_speaker=False)


@F.udf(returnType=BooleanType())
def is_all_lower(text):
    """Check if text (excluding special tokens) is all lowercase."""
    if text is None:
        return True
    t = re.sub(r"(\[MASK\]|\[QUOTE\]|\[TARGET_QUOTE\])", "", text)
    return t == t.lower()


def extract_entities(spark, *, merged_path, speakers_path, output_path,
                     nb_partition, compression="gzip", ftype="parquet", kind="train"):
    """Main pipeline: read data, extract entities, write parquet."""
    logger.info(f"Starting entity extraction: kind={kind}, input={merged_path}")

    df = (spark.read.parquet(merged_path) if ftype == "parquet"
          else spark.read.json(merged_path).repartition(nb_partition))
    df = df.dropna(subset=["quotation"])
    joined = df.join(spark.read.json(speakers_path), on="articleUID")

    transform_fn = transform if kind == "train" else transform_test
    transformed = (joined.rdd.map(transform_fn).filter(lambda x: x is not None)
                   .toDF().withColumn("nb_entities", F.size("entities")).filter("nb_entities > 0"))
    if kind == "train":
        transformed = transformed.withColumn("nb_targets", F.size("targets"))

    transformed.write.parquet(output_path, "overwrite", compression=compression)
    logger.info(f"Entity extraction complete: output={output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract named entities from quote contexts")
    parser.add_argument("-m", "--merged", required=True, help="Path to merged quotes")
    parser.add_argument("-s", "--speakers", required=True, help="Path to speakers folder")
    parser.add_argument("-o", "--output", required=True, help="Output path")
    parser.add_argument("--kind", required=True, choices=["train", "test"], help="Data type")
    parser.add_argument("-l", "--local", action="store_true", help="Run locally")
    parser.add_argument("-n", "--nb_partition", type=int, default=200, help="Number of partitions")
    parser.add_argument("--compression", default="gzip", help="Compression algorithm")
    parser.add_argument("--ftype", default="parquet", choices=["parquet", "json"], help="Input type")
    args = parser.parse_args()

    if args.local:
        spark = (SparkSession.builder.master("local[24]").appName("EntityExtractorLocal")
                 .config("spark.driver.memory", "16g").config("spark.executor.memory", "32g").getOrCreate())
    else:
        spark = SparkSession.builder.appName("EntityExtractor").getOrCreate()

    extract_entities(spark, merged_path=args.merged, speakers_path=args.speakers,
                     output_path=args.output, nb_partition=args.nb_partition,
                     compression=args.compression, ftype=args.ftype, kind=args.kind)
