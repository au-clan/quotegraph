#!/usr/bin/env python3
"""Evaluate quote merging + person-speaker filtering on DirectQuote.

Uses DirectQuote's CoNLL labels to construct gold:

* **Per quote span**: ``has_person_speaker`` iff the span is tagged
  ``LeftSpeaker`` or ``RightSpeaker`` (i.e. attributed to a person in the
  corpus).  ``Unknown`` spans count as *no* person speaker.
* **Per atomic adjacent pair**: ``should_merge`` iff the bridge contains a
  gold ``B-Speaker``/``I-Speaker`` token whose name overlaps with the
  speaker name attached to *both* adjacent spans, OR the bridge is pure
  quote punctuation, OR both spans are tagged with the same kind
  (Left/Right) and share Speaker tokens.  This is a precision-leaning
  proxy: we never claim ``should_merge=yes`` without explicit evidence in
  the gold annotation.

Pipeline under test:

1. Heuristics (with institutional denylist + person-attribution gate).
2. LLM adjudication (``gpt-5-mini`` by default) on ambiguous pairs only.
3. Block-level speaker filter combining role classification and
   predicted-speaker presence.

We DO NOT pass gold speaker names into the merger.  This mirrors the real
setting where speaker candidates upstream are noisy / absent.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from quotegraph.directquote_loader import (
    DirectQuoteParagraph,
    _extract_spans,
    load_paragraphs,
    paragraph_from_tokens,
)
from quotegraph.merge_pipeline import (
    block_has_person_speaker,
    block_is_non_quote,
    block_passes_speaker_filter,
)
from quotegraph.quote_merger import (
    AtomicPairContext,
    DiskPairCache,
    MergeAnswer,
    OpenAICompatibleAdjudicator,
    QuoteCandidate,
    QuoteRole,
    QuoteTurnMerger,
    _heuristic_pair_decision,
    resolve_api_key,
)


# ---------------------------------------------------------------------------
# Gold-label construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldSpan:
    paragraph_id: int
    quote_index: int
    kind: str  # LeftSpeaker | RightSpeaker | Unknown
    text: str
    speaker_tokens: tuple[str, ...]


@dataclass(frozen=True)
class GoldPair:
    paragraph_id: int
    pair_index: int
    bridge: str
    left: GoldSpan
    right: GoldSpan
    bridge_speaker_tokens: tuple[str, ...]
    should_merge: bool
    bridge_has_person: bool
    institutional_proxy: bool


def _speaker_tokens_around(
    tokens: list[tuple[str, str]],
    span_start: int,
    span_end: int,
    kind: str,
    radius: int = 25,
) -> tuple[str, ...]:
    """Return ``Speaker`` tokens adjacent to a quote span on its labelled side."""

    if kind == "LeftSpeaker":
        indices = range(max(0, span_start - radius), span_start)
    elif kind == "RightSpeaker":
        indices = range(span_end, min(len(tokens), span_end + radius))
    else:
        return ()

    collected: list[str] = []
    streak_started = False
    for idx in indices if kind == "RightSpeaker" else reversed(list(indices)):
        word, tag = tokens[idx]
        if tag in {"B-Speaker", "I-Speaker"}:
            collected.append(word)
            streak_started = True
        elif streak_started:
            # we've left the contiguous speaker tag block
            break
    if kind == "LeftSpeaker":
        collected.reverse()
    return tuple(collected)


def _bridge_speaker_tokens(
    tokens: list[tuple[str, str]], start: int, end: int
) -> tuple[str, ...]:
    return tuple(
        word for word, tag in tokens[start:end] if tag in {"B-Speaker", "I-Speaker"}
    )


INSTITUTIONAL_PROXY_RE = re.compile(
    r"\bthe\s+(?:report|agency|company|department|administration|committee|"
    r"filing|document|piece|editorial|statement|lawsuit|brief|memo|order|"
    r"ruling|opinion|complaint|indictment|motion|petition|judgment|"
    r"news\s+release|press\s+release|suit|note|tweet|email|letter|release|"
    r"notice|warning)\b|"
    r"\bit\s+(?:said|states|reads|notes|reports|added|continues)\b|"
    r"\b(?:an?\s+)?editorial\s+(?:in|said|read)\b|"
    r"\bin\s+a\s+statement\b|"
    r"\baccording\s+to\s+the\s+(?:report|agency|company|department|filing|"
    r"document|statement|lawsuit|brief|news\s+release|press\s+release|"
    r"editorial|memo)\b|"
    r"\b(?:said|read)\s+an?\s+(?:editorial|statement|filing|brief|memo|"
    r"complaint|indictment|motion|order|ruling|petition|lawsuit|press\s+release|"
    r"news\s+release)\b|"
    r"\bthe\s+(?:company|filing|suit|report|agency)\s+(?:stated|said|added|"
    r"alleges|claims|reads|states|notes)\b",
    re.IGNORECASE,
)


def _norm_tokens(tokens: tuple[str, ...]) -> frozenset[str]:
    return frozenset(t.lower().strip(".,;:'\"") for t in tokens if t.isalpha())


def build_gold_pairs(path: str | Path) -> list[GoldPair]:
    path = Path(path)
    raw_paragraphs: list[tuple[int, list[tuple[str, str]], DirectQuoteParagraph]] = []
    current: list[tuple[str, str]] = []
    pid = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                if current:
                    paragraph = paragraph_from_tokens(pid, current)
                    if paragraph is not None:
                        raw_paragraphs.append((pid, current, paragraph))
                    pid += 1
                    current = []
                continue
            word, tag = line.rsplit(" ", 1)
            current.append((word, tag))
    if current:
        paragraph = paragraph_from_tokens(pid, current)
        if paragraph is not None:
            raw_paragraphs.append((pid, current, paragraph))

    pairs: list[GoldPair] = []
    for paragraph_id, tokens, _ in raw_paragraphs:
        spans = _extract_spans(tokens)
        if len(spans) < 2:
            continue
        gold_spans: list[GoldSpan] = []
        for quote_index, (kind, span_start, span_end) in enumerate(spans):
            gold_spans.append(
                GoldSpan(
                    paragraph_id=paragraph_id,
                    quote_index=quote_index,
                    kind=kind,
                    text=" ".join(word for word, _ in tokens[span_start:span_end]),
                    speaker_tokens=_speaker_tokens_around(
                        tokens, span_start, span_end, kind
                    ),
                )
            )
        for pair_index, (left_span, right_span) in enumerate(zip(spans, spans[1:])):
            bridge_start, bridge_end = left_span[2], right_span[1]
            bridge_text = " ".join(word for word, _ in tokens[bridge_start:bridge_end])
            bridge_speakers = _bridge_speaker_tokens(tokens, bridge_start, bridge_end)
            left_tokens = _norm_tokens(gold_spans[pair_index].speaker_tokens)
            right_tokens = _norm_tokens(gold_spans[pair_index + 1].speaker_tokens)
            bridge_tokens_norm = _norm_tokens(bridge_speakers)

            pure_punct = bool(bridge_text.strip()) and not re.search(r"[A-Za-z0-9]", bridge_text)

            same_via_bridge = bool(
                bridge_tokens_norm
                and (
                    bridge_tokens_norm & left_tokens
                    or bridge_tokens_norm & right_tokens
                )
            )
            same_via_spans = bool(left_tokens & right_tokens) and bool(left_tokens)
            should_merge = pure_punct or same_via_bridge or same_via_spans

            both_unknown = (
                gold_spans[pair_index].kind == "Unknown"
                and gold_spans[pair_index + 1].kind == "Unknown"
            )
            inst_proxy = (
                INSTITUTIONAL_PROXY_RE.search(bridge_text) is not None
                and not bridge_speakers
                and both_unknown
            )

            pairs.append(
                GoldPair(
                    paragraph_id=paragraph_id,
                    pair_index=pair_index,
                    bridge=bridge_text,
                    left=gold_spans[pair_index],
                    right=gold_spans[pair_index + 1],
                    bridge_speaker_tokens=bridge_speakers,
                    should_merge=should_merge,
                    bridge_has_person=bool(bridge_speakers),
                    institutional_proxy=inst_proxy,
                )
            )
    return pairs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


@dataclass
class PairOutcome:
    paragraph_id: int
    pair_index: int
    bridge: str
    left_text: str
    right_text: str
    gold_should_merge: bool
    gold_bridge_has_person: bool
    gold_institutional: bool
    pred_merge: bool
    pred_left_role: str
    pred_right_role: str
    pred_reason: str
    pred_source: str


@dataclass
class BlockOutcome:
    paragraph_id: int
    start: int
    end: int
    member_text: list[str]
    member_kinds: list[str]
    gold_has_person: bool
    pred_kept: bool
    pred_non_quote: bool
    pred_has_speaker: bool
    decisions: list[dict[str, Any]]


def _quote_candidates(
    gold_spans: list[GoldSpan],
    *,
    speaker_noise: float = 0.0,
    seed: int = 0,
) -> list[QuoteCandidate]:
    """Build merger input candidates.

    ``speaker_noise`` simulates *upstream* attribution as in production:
    we use DirectQuote's gold speaker name with probability
    ``1 - speaker_noise`` and drop the name otherwise.  At
    ``speaker_noise == 1.0`` no speaker hints are passed (the strictest
    setting, which matches the original heuristics-only eval).
    """

    import random

    rng = random.Random(seed)
    candidates: list[QuoteCandidate] = []
    for span in gold_spans:
        speaker_name: str | None = None
        if span.kind != "Unknown" and span.speaker_tokens and rng.random() >= speaker_noise:
            speaker_name = " ".join(span.speaker_tokens)
        candidates.append(
            QuoteCandidate(
                text=span.text,
                quote_id=f"{span.paragraph_id}:{span.quote_index}",
                speaker=speaker_name,
            )
        )
    return candidates


def evaluate_config(
    pairs_by_paragraph: dict[int, list[GoldPair]],
    *,
    adjudicator: Any | None,
    cache_dir: Path | None,
    label: str,
    speaker_noise: float = 1.0,
) -> dict[str, Any]:
    cache = DiskPairCache(cache_dir) if cache_dir else None
    merger = QuoteTurnMerger(
        adjudicator=adjudicator,
        disk_cache=cache,
        auto_accept_logprob_margin=1.5,
        prefetch_llm=False,
    )

    pair_outcomes: list[PairOutcome] = []
    block_outcomes: list[BlockOutcome] = []
    decision_sources: Counter[str] = Counter()
    decision_reasons: Counter[str] = Counter()

    paragraph_ids = sorted(pairs_by_paragraph.keys())
    for paragraph_id in paragraph_ids:
        pairs = pairs_by_paragraph[paragraph_id]
        gold_spans = [pairs[0].left] + [p.right for p in pairs]
        quotes = _quote_candidates(gold_spans, speaker_noise=speaker_noise, seed=paragraph_id)
        bridges = [p.bridge for p in pairs]

        blocks = merger.merge_all(quotes, bridges)

        decision_per_pair: dict[int, Any] = {}
        for block in blocks:
            for offset, decision in enumerate(block.decisions):
                pair_index = block.start_index + offset
                if pair_index < len(bridges):
                    decision_per_pair.setdefault(pair_index, decision)

        # Per-pair: compare pred merge vs gold should_merge
        for pair_index, gold_pair in enumerate(pairs):
            # find which block contains this pair index
            containing = next(
                (b for b in blocks if b.start_index <= pair_index <= b.end_index - 1 or
                 (b.start_index == pair_index and b.end_index == pair_index)),
                None,
            )
            # pred merge: the pair is merged iff some block spans pair_index .. pair_index+1
            pred_merge = any(
                b.start_index <= pair_index and pair_index + 1 <= b.end_index
                for b in blocks
            )

            # capture decision context if available from any block that touched this pair
            relevant_decision = None
            for b in blocks:
                if b.start_index <= pair_index <= b.end_index:
                    for d in b.decisions:
                        if d.merge_prev or d.merge_next:
                            relevant_decision = d
                            break
                    if relevant_decision is None and b.decisions:
                        relevant_decision = b.decisions[0]
                    if relevant_decision is not None:
                        break

            pred_left_role = (
                relevant_decision.left_role.value if relevant_decision else QuoteRole.UTTERANCE.value
            )
            pred_right_role = (
                relevant_decision.right_role.value if relevant_decision else QuoteRole.UTTERANCE.value
            )
            pred_reason = relevant_decision.reason_code.value if relevant_decision else "heuristic_only"
            pred_source = relevant_decision.source if relevant_decision else "heuristic"

            decision_sources[pred_source] += 1
            decision_reasons[pred_reason] += 1

            pair_outcomes.append(
                PairOutcome(
                    paragraph_id=paragraph_id,
                    pair_index=pair_index,
                    bridge=gold_pair.bridge,
                    left_text=gold_pair.left.text,
                    right_text=gold_pair.right.text,
                    gold_should_merge=gold_pair.should_merge,
                    gold_bridge_has_person=gold_pair.bridge_has_person,
                    gold_institutional=gold_pair.institutional_proxy,
                    pred_merge=pred_merge,
                    pred_left_role=pred_left_role,
                    pred_right_role=pred_right_role,
                    pred_reason=pred_reason,
                    pred_source=pred_source,
                )
            )

        # Per-block: gold has-person-speaker
        for block in blocks:
            members = gold_spans[block.start_index : block.end_index + 1]
            gold_has_person = any(s.kind != "Unknown" for s in members)
            pred_has_speaker = block_has_person_speaker(quotes, bridges, block)
            pred_non_quote = block_is_non_quote(block)
            pred_kept = block_passes_speaker_filter(quotes, bridges, block)
            block_outcomes.append(
                BlockOutcome(
                    paragraph_id=paragraph_id,
                    start=block.start_index,
                    end=block.end_index,
                    member_text=[s.text for s in members],
                    member_kinds=[s.kind for s in members],
                    gold_has_person=gold_has_person,
                    pred_kept=pred_kept,
                    pred_non_quote=pred_non_quote,
                    pred_has_speaker=pred_has_speaker,
                    decisions=[
                        {
                            "merge_prev": d.merge_prev,
                            "merge_next": d.merge_next,
                            "left_role": d.left_role.value,
                            "right_role": d.right_role.value,
                            "reason": d.reason_code.value,
                            "source": d.source,
                        }
                        for d in block.decisions
                    ],
                )
            )

    # ---- Pair-level merge metrics ----
    tp = sum(1 for o in pair_outcomes if o.pred_merge and o.gold_should_merge)
    fp = sum(1 for o in pair_outcomes if o.pred_merge and not o.gold_should_merge)
    fn = sum(1 for o in pair_outcomes if not o.pred_merge and o.gold_should_merge)
    tn = sum(1 for o in pair_outcomes if not o.pred_merge and not o.gold_should_merge)
    merge_metrics = _prf(tp, fp, fn)
    merge_metrics["tn"] = tn
    merge_metrics["total"] = len(pair_outcomes)

    # Bucketed merge precision: high-risk subset (no person speaker in bridge)
    high_risk = [o for o in pair_outcomes if not o.gold_bridge_has_person]
    hr_tp = sum(1 for o in high_risk if o.pred_merge and o.gold_should_merge)
    hr_fp = sum(1 for o in high_risk if o.pred_merge and not o.gold_should_merge)
    hr_fn = sum(1 for o in high_risk if not o.pred_merge and o.gold_should_merge)
    high_risk_metrics = _prf(hr_tp, hr_fp, hr_fn)
    high_risk_metrics["total"] = len(high_risk)

    # ---- Block-level keep filter metrics ----
    btp = sum(1 for b in block_outcomes if b.pred_kept and b.gold_has_person)
    bfp = sum(1 for b in block_outcomes if b.pred_kept and not b.gold_has_person)
    bfn = sum(1 for b in block_outcomes if not b.pred_kept and b.gold_has_person)
    btn = sum(1 for b in block_outcomes if not b.pred_kept and not b.gold_has_person)
    block_metrics = _prf(btp, bfp, bfn)
    block_metrics["tn"] = btn
    block_metrics["total"] = len(block_outcomes)

    # ---- Institutional drop precision (gold proxy) ----
    inst_pairs = [o for o in pair_outcomes if o.gold_institutional]
    inst_merge_rate = (
        sum(1 for o in inst_pairs if o.pred_merge) / len(inst_pairs) if inst_pairs else 0.0
    )
    inst_role_nonquote = (
        sum(
            1
            for o in inst_pairs
            if o.pred_left_role == "non_quote" or o.pred_right_role == "non_quote"
        )
        / len(inst_pairs)
        if inst_pairs
        else 0.0
    )

    return {
        "label": label,
        "n_paragraphs": len(paragraph_ids),
        "n_pairs": len(pair_outcomes),
        "n_blocks": len(block_outcomes),
        "decision_sources": dict(decision_sources),
        "decision_reasons": dict(decision_reasons),
        "merge_metrics": merge_metrics,
        "merge_metrics_high_risk_no_person_bridge": high_risk_metrics,
        "block_keep_metrics": block_metrics,
        "institutional_proxy": {
            "total": len(inst_pairs),
            "false_merge_rate": inst_merge_rate,
            "non_quote_role_rate": inst_role_nonquote,
        },
        "pair_outcomes": [asdict(o) for o in pair_outcomes],
        "block_outcomes": [asdict(b) for b in block_outcomes],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/tmp/DirectQuote/data/truecased.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/directquote_eval.json"))
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["heuristics", "gpt-5-mini"],
        help="Subset of {heuristics, gpt-5-mini, gpt-5-nano}",
    )
    parser.add_argument("--limit-paragraphs", type=int, default=0)
    parser.add_argument("--cache-root", type=Path, default=Path(".quote_merger_cache"))
    parser.add_argument(
        "--speaker-noise",
        type=float,
        default=1.0,
        help="Fraction of upstream speakers to drop. 1.0 = blind (default), "
        "0.2 = simulate 80%% recall noisy attribution.",
    )
    args = parser.parse_args()

    print("Building gold pairs...", flush=True)
    pairs = build_gold_pairs(args.input)
    if args.limit_paragraphs:
        allowed = set()
        for p in pairs:
            allowed.add(p.paragraph_id)
            if len(allowed) >= args.limit_paragraphs:
                break
        pairs = [p for p in pairs if p.paragraph_id in allowed]
    pairs_by_paragraph: dict[int, list[GoldPair]] = defaultdict(list)
    for p in pairs:
        pairs_by_paragraph[p.paragraph_id].append(p)
    print(f"  {sum(len(v) for v in pairs_by_paragraph.values())} pairs across {len(pairs_by_paragraph)} paragraphs")

    results: list[dict[str, Any]] = []
    for config in args.configs:
        print(f"\n=== Evaluating: {config} ===", flush=True)
        t0 = time.time()
        if config == "heuristics":
            res = evaluate_config(
                pairs_by_paragraph,
                adjudicator=None,
                cache_dir=None,
                label=config,
                speaker_noise=args.speaker_noise,
            )
        else:
            adj = OpenAICompatibleAdjudicator(
                model=config,
                key_path=Path.home() / "configs/keys/openai.txt",
                use_self_consistency=False,
            )
            cache_dir = args.cache_root / f"directquote_eval_{config}"
            res = evaluate_config(
                pairs_by_paragraph,
                adjudicator=adj,
                cache_dir=cache_dir,
                label=config,
                speaker_noise=args.speaker_noise,
            )
        res["elapsed_seconds"] = time.time() - t0
        results.append(res)
        # Print key metrics
        m = res["merge_metrics"]
        bm = res["block_keep_metrics"]
        ins = res["institutional_proxy"]
        print(
            f"  merge P/R/F1: {m['precision']:.3f}/{m['recall']:.3f}/{m['f1']:.3f}  "
            f"block-keep P/R/F1: {bm['precision']:.3f}/{bm['recall']:.3f}/{bm['f1']:.3f}  "
            f"inst false-merge: {ins['false_merge_rate']:.3f}  "
            f"non_quote role: {ins['non_quote_role_rate']:.3f}  "
            f"sources: {res['decision_sources']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Save full results (pair/block outcomes can be large)
    summary = []
    for r in results:
        summary.append({k: v for k, v in r.items() if k not in {"pair_outcomes", "block_outcomes"}})
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    detail_path = args.output.with_name(args.output.stem + "_detail.json")
    detail_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote summary to {args.output}")
    print(f"Wrote detail to {detail_path}")


if __name__ == "__main__":
    main()
