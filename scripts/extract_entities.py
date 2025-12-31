"""
Gazetteer-based Named Entity Recognizer for Quote Attribution.

Matches PERSON entity names from a Wikidata gazetteer against text contexts
surrounding quotes. Assumes Wikidata alias table is complete.

Changes (2025-12-31):
- Replaced eval() with ast.literal_eval() for security
- Added comprehensive error handling and logging
- Added text normalization (hyphens, initials, possessives, family plurals)
- Added honorific/suffix stripping (Dr., Jr., etc.)
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

# Words to filter from entity names
BLACK_LIST = {"president", "manager"}
HONORIFICS = {
    "mr",
    "mrs",
    "ms",
    "miss",
    "dr",
    "prof",
    "sir",
    "dame",
    "lord",
    "lady",
    "rev",
    "hon",
    "sen",
    "rep",
    "gov",
    "gen",
    "col",
    "maj",
    "capt",
    "lt",
    "sgt",
    "rabbi",
    "imam",
    "fr",
    "brother",
    "sister",
    "saint",
    "st",
    # Name suffixes
    "jr",
    "sr",
    "ii",
    "iii",
    "iv",
    "v",
    "esq",
    "phd",
    "md",
    "dds",
    "jd",
}

SPECIAL_MARKERS = {
    "-1",
    "none",
    "not_quote",
    "not_mentioned",
    "not_en",
    "ambiguous",
    "other",
}


def normalize_text(text):
    """Normalize hyphens (Jean-Pierre -> Jean Pierre) and initials (J. -> J)."""
    if not text:
        return ""
    text = re.sub(
        r"(?<!\[)(\w)-(\w)(?!\])", r"\1 \2", text
    )  # Hyphens (preserve [TARGET_QUOTE])
    text = re.sub(r"\b([A-Za-z])\.\s*", r"\1 ", text)  # Initials
    return re.sub(r"\s+", " ", text).strip()


def normalize_person_token(token):
    """Normalize family plurals: Bidens/bidens -> Biden/biden."""
    if (
        not token
        or len(token) < 4
        or not token.endswith("s")
        or token.endswith("ss")
    ):
        return token
    second_last = token[-2].lower()
    if second_last in "aeiou":  # Obamas -> Obama
        return token[:-1]
    token_lower = token.lower()
    if token_lower.endswith(("ens", "ans", "ons")):  # Bidens -> Biden
        return token[:-1]
    return token


def create_trie(names):
    """Create trie from entity names for prefix matching."""
    trie = StringTrie(delimiter="/")
    for name, qid in names:
        try:
            q_name = ast.literal_eval(qid[0])[1]
        except (ValueError, SyntaxError, IndexError, TypeError):
            continue
        if not name or not name.strip():
            continue

        # Add normalized and original (if hyphenated) versions
        normalized = normalize_text(name)
        variants = [normalized]
        if name != normalized and "-" in name:
            variants.append(name)

        for variant in variants:
            tokens = [
                x
                for x in variant.split()
                if x.lower() not in BLACK_LIST
                and x.lower().rstrip(".") not in HONORIFICS
            ]
            if not tokens:
                continue
            for i in range(len(tokens)):
                key = "/".join(tokens[i:]).lower()
                if key:
                    trie[key] = (q_name, qid)
    return trie


def fix_punct_tokens(tokens):
    """Separate punctuation, strip honorifics, normalize family plurals."""
    if not tokens:
        return []
    out = []
    for token in tokens:
        if not token:
            continue
        # Strip leading punctuation
        if token.startswith("[") and token.endswith("]"):
            out.append(token)
            continue
        while token and token[0] in PUNCT:
            token = token[1:]
        if not token:
            continue
        # Skip honorifics/suffixes
        if token.lower().rstrip(".") in HONORIFICS:
            continue
        # Handle punctuation and possessives
        if token[-1] in PUNCT:
            base = normalize_person_token(token[:-1])
            if base:
                out.append(base)
            out.append(token[-1])
        elif token.endswith("'s") and len(token) >= 2:
            if token[:-2]:
                out.append(token[:-2])
            out.append("'s")
        elif token.endswith("s'") and len(token) >= 2:
            base = normalize_person_token(token[:-1])
            if base:
                out.append(base)
            out.append("'")
        else:
            out.append(normalize_person_token(token))
    return out


def reduce_entities(entities):
    """Group entity occurrences by Q-name."""
    out = {}
    for i, value in entities.items():
        if value is None:
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, tuple) and len(item) >= 2:
                q_name, qinfo = item[0], item[1]
                if q_name in out:
                    out[q_name][0].append(i)
                else:
                    out[q_name] = ([i], qinfo)
    return out


def find_entities(text, trie, mask=MASK_TOKEN):
    """Find and mask named entities in text using gazetteer trie."""
    if not text or not text.strip():
        return "", {}

    text = normalize_text(text)
    tokens = fix_punct_tokens(text.split())
    if not tokens:
        return "", {}

    start, count, entities, out = 0, 1, {}, []

    for i in range(len(tokens)):
        key = "/".join(tokens[start : i + 1]).lower()
        if not key:
            out.append(tokens[i])
            start = i + 1
            continue

        try:
            if trie.has_subtrie(key):
                if i == len(tokens) - 1:  # End of string
                    match = list(trie[key:])
                    if match:
                        entities[count] = (
                            match[0] if len(match) == 1 else match
                        )
                        out.append(mask)
                    else:
                        out.extend(tokens[start : i + 1])
            elif key in trie:
                entities[count] = trie[key]
                count += 1
                out.append(mask)
                start = i + 1
            elif start < i:  # Partial match
                old_key = "/".join(tokens[start:i]).lower()
                match = list(trie[old_key:]) if old_key else None
                if match:
                    entities[count] = match[0] if len(match) == 1 else match
                    count += 1
                    out.append(mask)
                else:
                    out.extend(tokens[start:i])
                start = i if trie.has_node(tokens[i].lower()) else i + 1
                if start == i + 1:
                    out.append(tokens[i])
            else:
                out.append(tokens[i])
                start = i + 1
        except (KeyError, TypeError):
            out.append(tokens[i])
            start = i + 1

    if not out:
        return "", {}

    retokenized = "".join(
        " " + t if not t.startswith("'") and t not in PUNCT else t for t in out
    ).strip()
    return retokenized, reduce_entities(entities)


# Backwards compatibility alias
find_entites = find_entities


def get_targets(entities, target_entity):
    """Get target entity positions and ambiguity flag."""
    targets = entities.get(target_entity)
    if not targets:
        return [0], False
    if (
        isinstance(targets, tuple)
        and targets[0]
        and isinstance(targets[0], list)
    ):
        return targets[0], len(targets[0]) > 1
    return [0], False


def check_speaker_in_entities(speaker, names):
    """Check if speaker is in entity list or is a special marker."""
    if speaker in SPECIAL_MARKERS:
        return True
    if not names:
        return False
    for name, qid in names:
        try:
            if ast.literal_eval(qid[0])[1] == speaker:
                return True
        except (ValueError, SyntaxError, IndexError, TypeError):
            continue
    return False


def transform(x):
    """Transform quote row by extracting entities (with speaker labels)."""
    try:
        speaker, names = x.speaker, x.names
        left_context, right_context = x.leftContext, x.rightContext
    except AttributeError:
        return None

    if not check_speaker_in_entities(speaker, names):
        return None

    trie = create_trie(names)
    full_text = re.sub(
        r"\"+", "", f"{left_context} {TARGET_QUOTE_TOKEN} {right_context}"
    )

    try:
        masked_text, entities = find_entities(full_text, trie)
    except Exception:
        return None

    targets, ambiguous = get_targets(entities, speaker)

    return Row(
        articleUID=x.articleUID,
        articleOffset=x.articleOffset,
        speaker=speaker,
        quotation=x.quotation,
        full_text=full_text,
        masked_text=masked_text,
        entities=entities,
        targets=targets,
        ambiguous=ambiguous,
        domain=getattr(x, "domain", ""),
        pattern=getattr(x, "pattern", ""),
    )


def transform_test(x):
    """Transform quote row by extracting entities (without speaker labels)."""
    try:
        names = x.names
        left_context, right_context = x.leftContext, x.rightContext
    except AttributeError:
        return None

    trie = create_trie(names)
    full_text = re.sub(
        r"\"+", "", f"{left_context} {TARGET_QUOTE_TOKEN} {right_context}"
    )

    try:
        masked_text, entities = find_entities(full_text, trie)
    except Exception:
        return None

    return Row(
        articleUID=x.articleUID,
        articleOffset=x.articleOffset,
        quotation=x.quotation,
        full_text=full_text,
        masked_text=masked_text,
        entities=entities,
    )


@F.udf(returnType=BooleanType())
def is_all_lower(masked_text):
    """Check if text (excluding special tokens) is all lowercase."""
    if masked_text is None:
        return True
    text = re.sub(r"(\[MASK\]|\[QUOTE\]|\[TARGET_QUOTE\])", "", masked_text)
    return text == text.lower()


def extract_entities(
    spark,
    *,
    merged_path,
    speakers_path,
    output_path,
    nb_partition,
    compression="gzip",
    ftype="parquet",
    kind="train",
):
    """Main pipeline: read data, extract entities, write parquet."""
    logger.info(
        f"Starting entity extraction: kind={kind}, input={merged_path}"
    )

    df = (
        spark.read.parquet(merged_path)
        if ftype == "parquet"
        else spark.read.json(merged_path).repartition(nb_partition)
    )
    df = df.dropna(subset=["quotation"])
    speakers = spark.read.json(speakers_path)
    joined = df.join(speakers, on="articleUID")

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

    transformed.write.parquet(
        output_path, "overwrite", compression=compression
    )
    logger.info(f"Entity extraction complete: output={output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract named entities from quote contexts"
    )
    parser.add_argument(
        "-m",
        "--merged",
        required=True,
        help="Path to merged quotes (.parquet/.json)",
    )
    parser.add_argument(
        "-s",
        "--speakers",
        required=True,
        help="Path to speakers folder (.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path for transformed data",
    )
    parser.add_argument(
        "--kind", required=True, choices=["train", "test"], help="Data type"
    )
    parser.add_argument(
        "-l", "--local", action="store_true", help="Run locally"
    )
    parser.add_argument(
        "-n",
        "--nb_partition",
        type=int,
        default=200,
        help="Number of partitions",
    )
    parser.add_argument(
        "--compression", default="gzip", help="Compression algorithm"
    )
    parser.add_argument(
        "--ftype",
        default="parquet",
        choices=["parquet", "json"],
        help="Input file type",
    )
    args = parser.parse_args()

    if args.local:
        spark = (
            SparkSession.builder.master("local[24]")
            .appName("EntityExtractorLocal")
            .config("spark.driver.memory", "16g")
            .config("spark.executor.memory", "32g")
            .getOrCreate()
        )
    else:
        spark = SparkSession.builder.appName("EntityExtractor").getOrCreate()

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
