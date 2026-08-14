"""Load regex/heuristic patterns for the quote merger from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import yaml

PATTERNS_PATH = Path(__file__).with_name("merger_patterns.yaml")


@dataclass(frozen=True)
class MergerPatterns:
    attribution_re: re.Pattern[str]
    new_speaker_bridge_re: re.Pattern[str]
    post_sentence_named_attribution_re: re.Pattern[str]
    forward_attribution_re: re.Pattern[str]
    reply_turn_re: re.Pattern[str]
    background_intro_re: re.Pattern[str]
    speaker_switch_re: re.Pattern[str]
    boilerplate_role_re: re.Pattern[str]
    non_quote_re: re.Pattern[str]
    institutional_bridge_re: re.Pattern[str]
    person_attribution_bridge_re: re.Pattern[str]
    abbreviations_before_period: frozenset[str]
    pronoun_speakers: frozenset[str]
    speaker_titles: frozenset[str]


@lru_cache(maxsize=1)
def load_merger_patterns(path: Path | None = None) -> MergerPatterns:
    path = path or PATTERNS_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    verbs = sorted(re.escape(v) for v in raw["attribution_verbs"])
    attribution_re = re.compile(r"\b(" + "|".join(verbs) + r")\b", re.IGNORECASE)

    def compile_flagged(name: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
        return re.compile(raw[name], flags)

    return MergerPatterns(
        attribution_re=attribution_re,
        new_speaker_bridge_re=compile_flagged("new_speaker_bridge"),
        post_sentence_named_attribution_re=compile_flagged("post_sentence_named_attribution"),
        forward_attribution_re=compile_flagged("forward_attribution"),
        reply_turn_re=compile_flagged("reply_turn"),
        background_intro_re=compile_flagged("background_intro"),
        speaker_switch_re=compile_flagged("speaker_switch"),
        boilerplate_role_re=compile_flagged("boilerplate_role"),
        non_quote_re=compile_flagged("non_quote"),
        institutional_bridge_re=compile_flagged("institutional_bridge"),
        person_attribution_bridge_re=compile_flagged("person_attribution_bridge", flags=0),
        abbreviations_before_period=frozenset(raw["abbreviations_before_period"]),
        pronoun_speakers=frozenset(raw["pronoun_speakers"]),
        speaker_titles=frozenset(raw["speaker_titles"]),
    )
