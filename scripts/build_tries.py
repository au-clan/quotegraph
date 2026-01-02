"""
Pre-build tries for named entity extraction.

Builds and serializes tries for each Quotebank phase (A-E), allowing
extract_entities.py to load pre-built tries instead of building on the fly.

Usage:
    python scripts/build_tries.py --qid2alias qid2alias.pkl --output tries/
"""

import argparse
import gc
import pickle
from pathlib import Path

from quotegraph.entity_extractor import build_trie_from_qid2alias
from quotegraph.utils import load_pickle

PHASES = ["A", "B", "C", "D", "E"]


def build_and_save_trie(qid2alias: dict, phase: str, output_dir: Path) -> Path:
    """Build and save a trie for a single phase."""
    print(f"  Building trie for phase {phase}...")
    trie = build_trie_from_qid2alias(qid2alias, phase=phase)
    print(f"    {len(list(trie.keys())):,} keys")

    output_path = output_dir / f"trie_phase_{phase}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(trie, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"    Saved: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Pre-build tries for entity extraction")
    parser.add_argument("--qid2alias", required=True, help="Path to qid2alias.pkl")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--phases", nargs="+", default=PHASES, choices=PHASES,
                        help=f"Phases to build (default: {PHASES})")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.qid2alias}...")
    qid2alias = load_pickle(args.qid2alias)
    print(f"Loaded {len(qid2alias):,} QIDs\n")

    output_paths = []
    for phase in args.phases:
        output_paths.append(build_and_save_trie(qid2alias, phase, output_dir))
        gc.collect()

    # Save metadata
    metadata = {
        "qid2alias_path": args.qid2alias,
        "num_qids": len(qid2alias),
        "phases": args.phases,
        "trie_files": {p: str(output_dir / f"trie_phase_{p}.pkl") for p in args.phases},
    }
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nTotal: {sum(p.stat().st_size for p in output_paths) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
