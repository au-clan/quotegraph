#!/usr/bin/env python3
"""
Parse Wikidata redirects from XML (SPARQL result format) and save as a pickled dictionary.

The XML file is expected to be in SPARQL result format with bindings like:
    <binding name='redirect'>
        <uri>http://www.wikidata.org/entity/Q123</uri>
    </binding>
    <binding name='target'>
        <uri>http://www.wikidata.org/entity/Q456</uri>
    </binding>

The output is a dictionary mapping redirect QIDs to target QIDs: {"Q123": "Q456", ...}
"""

import argparse
import logging
import pickle
import re
from typing import Iterator

from tqdm import tqdm

logger = logging.getLogger(__name__)


def iter_lines(filepath: str) -> Iterator[str]:
    """Iterate over lines of a file, handling bz2, gzip, or plain text."""

    with open(filepath, "r", encoding="utf-8") as f:
        yield from f


def parse_redirects(filepath: str) -> dict[str, str]:
    """
    Parse Wikidata redirect XML file and return a mapping dict.

    Args:
        filepath: Path to the XML file

    Returns:
        Dictionary mapping redirect QIDs to target QIDs
    """
    binding_pattern = re.compile(r"<binding name='(redirect|target)'>")

    uri_pattern = re.compile(
        r"<uri>http://www\.wikidata\.org/entity/([A-Z][0-9]+)</uri>"
    )

    redirect_mapping: dict[str, str] = {}
    current_redirect: dict[str, str] = {}
    current_binding_type: str = ""

    pbar = tqdm(iter_lines(filepath), desc="Parsing redirects", unit=" lines")
    for line in pbar:
        line = line.strip()

        if not line:
            continue

        if len(current_redirect) == 2:
            redirect_mapping[current_redirect["redirect"]] = current_redirect[
                "target"
            ]
            current_redirect = {}

        binding_match = binding_pattern.search(line)
        if binding_match:
            current_binding_type = binding_match.group(1)
            continue

        uri_match = uri_pattern.search(line)
        if uri_match:
            entity_id = uri_match.group(1)
            current_redirect[current_binding_type] = entity_id

    if len(current_redirect) == 2:
        redirect_mapping[current_redirect["redirect"]] = current_redirect[
            "target"
        ]

    return redirect_mapping


def resolve_redirect_chains(
    redirect_mapping: dict[str, str],
) -> dict[str, str]:
    """
    Resolve redirect chains to their final targets.

    For example, if A -> B and B -> C, this will update A -> C.

    Args:
        redirect_mapping: Dictionary mapping redirect QIDs to target QIDs

    Returns:
        Dictionary with all redirects pointing to their final targets
    """
    resolved = {}

    for source in redirect_mapping:
        target = redirect_mapping[source]
        visited = {source, target}

        while target in redirect_mapping:
            next_target = redirect_mapping[target]
            if next_target in visited:
                break
            visited.add(next_target)
            target = next_target

        resolved[source] = target

    return resolved


def main():
    parser = argparse.ArgumentParser(
        description="Parse Wikidata redirects XML and save as pickled dictionary."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input XML file (can be .xml, .xml.bz2, or .xml.gz)",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output pickle file",
    )
    parser.add_argument(
        "--resolve-chains",
        action="store_true",
        help="Resolve redirect chains to final targets",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("Parsing redirects from: %s", args.input_file)

    redirect_mapping = parse_redirects(args.input_file)

    logger.info("Found %s redirects", f"{len(redirect_mapping):,}")

    if args.resolve_chains:
        logger.info("Resolving redirect chains...")
        redirect_mapping = resolve_redirect_chains(redirect_mapping)

    logger.info("Saving to: %s", args.output_file)

    with open(args.output_file, "wb") as f:
        pickle.dump(redirect_mapping, f)

    logger.info("Done!")


if __name__ == "__main__":
    main()
