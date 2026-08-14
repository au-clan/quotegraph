"""Hybrid quote-turn merging for news quote attribution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx

from quotegraph.merger_patterns import load_merger_patterns

try:
    import diskcache
except ImportError:  # pragma: no cover
    diskcache = None  # type: ignore[assignment]


DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LOGPROB_MARGIN = 1.5
DEFAULT_SELF_CONSISTENCY_SAMPLES = 2
DEFAULT_API_KEY_PATHS = (
    Path("openai/keys/openai.txt"),
    Path("configs/keys/openai.txt"),
    Path.home() / "configs/keys/openai.txt",
    Path.home() / "openai/keys/openai.txt",
)
SYSTEM_PROMPT_PATH = Path(__file__).with_name("merger_prompts") / "system.txt"
SPEAKER_PROB_THRESHOLD = 0.6

PAIR_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "merge": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "left_role": {"type": "string", "enum": ["utterance", "scare", "non_quote"]},
        "right_role": {"type": "string", "enum": ["utterance", "scare", "non_quote"]},
        "speaker_continuity": {
            "type": "string",
            "enum": ["same", "different", "unknown"],
        },
    },
    "required": ["merge", "left_role", "right_role", "speaker_continuity"],
}


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReasonCode(str, Enum):
    ARTICLE_BOUNDARY = "article_boundary"
    AMBIGUOUS = "ambiguous"
    CONTINUED_NEWS_QUOTE = "continued_news_quote"
    DIFFERENT_SPEAKERS = "different_speakers"
    EMPTY_NEIGHBOR = "empty_neighbor"
    EXPLICIT_SPEAKER_SWITCH = "explicit_speaker_switch"
    HEURISTIC_ONLY = "heuristic_only"
    LLM_ERROR = "llm_error"
    NON_QUOTE = "non_quote"
    SAME_ATTRIBUTION_CLAUSE = "same_attribution_clause"
    SAME_SPEAKER = "same_speaker"
    TOO_DISTANT = "too_distant"


class QuoteRole(str, Enum):
    UTTERANCE = "utterance"
    SCARE = "scare"
    NON_QUOTE = "non_quote"


class MergeAnswer(str, Enum):
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


class SpeakerContinuity(str, Enum):
    SAME = "same"
    DIFFERENT = "different"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SpeakerTop:
    name: str | None = None
    qid: str | None = None
    probability: float = 0.0


@dataclass(frozen=True)
class QuoteCandidate:
    text: str
    quote_id: str | None = None
    speaker: str | None = None
    speaker_qid: str | None = None
    speaker_probability: float = 0.0
    start: int | None = None
    end: int | None = None
    mentioned_entities: tuple[str, ...] = ()
    local_probas: tuple[tuple[str, str, float], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AtomicPairContext:
    left: QuoteCandidate
    right: QuoteCandidate
    bridge: str
    bridge_mentioned_entities: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairDecision:
    merge: MergeAnswer
    left_role: QuoteRole
    right_role: QuoteRole
    speaker_continuity: SpeakerContinuity
    logprob_margin: float = 0.0
    reason_code: ReasonCode = ReasonCode.AMBIGUOUS
    source: str = "heuristic"

    def should_merge(self) -> bool:
        return self.merge is MergeAnswer.YES

    @property
    def confidence(self) -> Confidence:
        if self.source == "heuristic":
            return Confidence.HIGH
        if self.logprob_margin >= DEFAULT_LOGPROB_MARGIN:
            return Confidence.HIGH
        if self.logprob_margin >= 0.75:
            return Confidence.MEDIUM
        return Confidence.LOW


@dataclass(frozen=True)
class MergeDecision:
    merge_prev: bool
    merge_next: bool
    confidence: Confidence
    reason_code: ReasonCode
    left_role: QuoteRole = QuoteRole.UTTERANCE
    right_role: QuoteRole = QuoteRole.UTTERANCE
    logprob_margin: float = 0.0
    rationale: str = ""
    source: str = "heuristic"


@dataclass(frozen=True)
class MergeBlock:
    start_index: int
    end_index: int
    quote_ids: tuple[str | None, ...]
    text: str
    quote_text: str
    decisions: tuple[MergeDecision, ...]
    left_role: QuoteRole = QuoteRole.UTTERANCE
    right_role: QuoteRole = QuoteRole.UTTERANCE


Adjudicator = Callable[[AtomicPairContext], PairDecision]


def resolve_api_key(api_key: str | None = None, key_path: str | Path | None = None) -> str:
    if api_key:
        return api_key.strip()
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    paths = (Path(key_path).expanduser(),) if key_path is not None else DEFAULT_API_KEY_PATHS
    for path in paths:
        if path.exists():
            return path.read_text().strip()
    searched = ", ".join(str(p) for p in paths)
    raise FileNotFoundError(
        "OpenAI API key not found. Set OPENAI_API_KEY or create one of: "
        f"{searched}"
    )


def load_system_prompt(path: Path | None = None) -> str:
    prompt_path = path or SYSTEM_PROMPT_PATH
    return prompt_path.read_text(encoding="utf-8").strip()


def compact_quote(text: str, max_edge_tokens: int = 16) -> str:
    tokens = text.split()
    if len(tokens) <= max_edge_tokens * 2:
        return text.strip()
    head = " ".join(tokens[:max_edge_tokens])
    tail = " ".join(tokens[-max_edge_tokens:])
    return f"{head} [...QUOTE_MIDDLE_OMITTED...] {tail}"


def normalize_bridge(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _speaker_tokens(speaker: str) -> list[str]:
    patterns = load_merger_patterns()
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9.'-]*", speaker)]
    return [t.strip(".") for t in tokens if t.strip(".") not in patterns.speaker_titles]


def _speaker_is_pronoun(speaker: str) -> bool:
    patterns = load_merger_patterns()
    tokens = _speaker_tokens(speaker)
    return len(tokens) == 1 and tokens[0] in patterns.pronoun_speakers


def _speakers_compatible(left: QuoteCandidate, right: QuoteCandidate) -> bool:
    if (
        left.speaker_qid
        and right.speaker_qid
        and left.speaker_probability >= SPEAKER_PROB_THRESHOLD
        and right.speaker_probability >= SPEAKER_PROB_THRESHOLD
    ):
        return left.speaker_qid == right.speaker_qid

    left_name = left.speaker
    right_name = right.speaker
    if not left_name or not right_name:
        return False
    left_tokens = _speaker_tokens(left_name)
    right_tokens = _speaker_tokens(right_name)
    if not left_tokens or not right_tokens:
        return False
    if left_tokens == right_tokens:
        return True
    if _speaker_is_pronoun(left_name) or _speaker_is_pronoun(right_name):
        return False
    return left_tokens[-1] == right_tokens[-1] and (
        len(left_tokens) == 1 or len(right_tokens) == 1
    )


def _different_known_speakers(left: QuoteCandidate, right: QuoteCandidate) -> bool:
    if (
        left.speaker_qid
        and right.speaker_qid
        and left.speaker_probability >= SPEAKER_PROB_THRESHOLD
        and right.speaker_probability >= SPEAKER_PROB_THRESHOLD
        and left.speaker_qid != right.speaker_qid
    ):
        return True
    if not left.speaker or not right.speaker:
        return False
    if _speaker_is_pronoun(left.speaker) or _speaker_is_pronoun(right.speaker):
        return False
    return not _speakers_compatible(left, right)


def _same_known_speakers(left: QuoteCandidate, right: QuoteCandidate) -> bool:
    if (
        left.speaker_qid
        and right.speaker_qid
        and left.speaker_probability >= SPEAKER_PROB_THRESHOLD
        and right.speaker_probability >= SPEAKER_PROB_THRESHOLD
    ):
        return left.speaker_qid == right.speaker_qid
    return bool(left.speaker and right.speaker and _speakers_compatible(left, right))


def _bridge_is_quote_punctuation(text: str) -> bool:
    bridge = normalize_bridge(text)
    return bool(bridge) and not re.search(r"[A-Za-z0-9]", bridge)


def _bridge_is_short_attribution(text: str) -> bool:
    patterns = load_merger_patterns()
    bridge = normalize_bridge(text)
    if not bridge or _word_count(bridge) > 45:
        return False
    return bool(patterns.attribution_re.search(bridge))


def _independent_attribution_matches(text: str) -> list[Any]:
    patterns = load_merger_patterns()
    bridge = normalize_bridge(text)
    matches = []
    for match in patterns.attribution_re.finditer(bridge):
        prefix = bridge[max(0, match.start() - 24) : match.start()].lower()
        if re.search(r"\b(who|which|that)\s+(?:had\s+|has\s+|have\s+|also\s+)?$", prefix):
            continue
        matches.append(match)
    return matches


def _has_post_sentence_named_attribution(bridge: str) -> bool:
    patterns = load_merger_patterns()
    for match in patterns.post_sentence_named_attribution_re.finditer(bridge):
        prior_words = re.findall(r"[A-Za-z]+", bridge[: match.start()])
        if prior_words and prior_words[-1].lower() in patterns.abbreviations_before_period:
            continue
        return True
    return False


def _bridge_has_speaker_switch_risk(text: str) -> bool:
    patterns = load_merger_patterns()
    bridge = normalize_bridge(text).strip("\"' ")
    if patterns.speaker_switch_re.search(bridge):
        return True
    if patterns.new_speaker_bridge_re.search(bridge):
        return True
    if _has_post_sentence_named_attribution(bridge):
        return True
    if patterns.forward_attribution_re.search(bridge):
        return True
    if patterns.reply_turn_re.search(bridge):
        return True
    if patterns.background_intro_re.search(bridge) and patterns.attribution_re.search(bridge):
        return True
    return False


def _bridge_is_pure_local_attribution(text: str) -> bool:
    bridge = normalize_bridge(text).strip("\"' ")
    if not bridge or _word_count(bridge) > 35:
        return False
    if _bridge_has_speaker_switch_risk(bridge):
        return False
    matches = _independent_attribution_matches(bridge)
    if len(matches) != 1:
        return False
    match = matches[0]
    words_before = _word_count(bridge[: match.start()])
    if words_before <= 6:
        return True
    return "," in bridge[: match.start()] and words_before <= 24


_BRIDGE_QUOTE_CHARS = '"\u201c\u201d\u2018\u2019\'` '


def _bridge_is_institutional(bridge: str) -> bool:
    patterns = load_merger_patterns()
    norm = normalize_bridge(bridge).strip(_BRIDGE_QUOTE_CHARS)
    if not norm or _word_count(norm) > 35:
        return False
    if patterns.person_attribution_bridge_re.search(norm):
        return False
    return bool(patterns.institutional_bridge_re.search(norm))


def _bridge_has_person_attribution(bridge: str) -> bool:
    patterns = load_merger_patterns()
    norm = normalize_bridge(bridge).strip(_BRIDGE_QUOTE_CHARS)
    return bool(patterns.person_attribution_bridge_re.search(norm))


def _heuristic_roles(left: QuoteCandidate, right: QuoteCandidate, bridge: str) -> tuple[QuoteRole, QuoteRole]:
    patterns = load_merger_patterns()
    norm = normalize_bridge(bridge)
    if patterns.boilerplate_role_re.search(norm):
        return QuoteRole.NON_QUOTE, QuoteRole.NON_QUOTE
    if patterns.boilerplate_role_re.search(left.text):
        return QuoteRole.NON_QUOTE, QuoteRole.UTTERANCE
    if patterns.boilerplate_role_re.search(right.text):
        return QuoteRole.UTTERANCE, QuoteRole.NON_QUOTE
    if _bridge_is_institutional(bridge):
        return QuoteRole.NON_QUOTE, QuoteRole.NON_QUOTE
    return QuoteRole.UTTERANCE, QuoteRole.UTTERANCE


def _hard_no_for_pair(left: QuoteCandidate, right: QuoteCandidate, bridge: str) -> PairDecision | None:
    patterns = load_merger_patterns()
    norm = normalize_bridge(bridge)
    left_role, right_role = _heuristic_roles(left, right, bridge)

    if "\n\n" in bridge and not _bridge_is_short_attribution(norm):
        return PairDecision(
            MergeAnswer.NO, left_role, right_role, SpeakerContinuity.DIFFERENT,
            reason_code=ReasonCode.ARTICLE_BOUNDARY,
        )
    if patterns.non_quote_re.search(norm):
        return PairDecision(
            MergeAnswer.NO, QuoteRole.NON_QUOTE, QuoteRole.NON_QUOTE, SpeakerContinuity.UNKNOWN,
            reason_code=ReasonCode.NON_QUOTE,
        )
    if _bridge_is_institutional(bridge):
        return PairDecision(
            MergeAnswer.NO, QuoteRole.NON_QUOTE, QuoteRole.NON_QUOTE, SpeakerContinuity.UNKNOWN,
            reason_code=ReasonCode.NON_QUOTE,
        )
    if _different_known_speakers(left, right):
        return PairDecision(
            MergeAnswer.NO, left_role, right_role, SpeakerContinuity.DIFFERENT,
            reason_code=ReasonCode.DIFFERENT_SPEAKERS,
        )
    if not _same_known_speakers(left, right) and _bridge_has_speaker_switch_risk(norm):
        return PairDecision(
            MergeAnswer.NO, left_role, right_role, SpeakerContinuity.DIFFERENT,
            reason_code=ReasonCode.EXPLICIT_SPEAKER_SWITCH,
        )
    if _word_count(norm) > 90 and not _bridge_is_short_attribution(norm):
        return PairDecision(
            MergeAnswer.NO, left_role, right_role, SpeakerContinuity.UNKNOWN,
            reason_code=ReasonCode.TOO_DISTANT,
        )
    return None


def _heuristic_pair_decision(
    left: QuoteCandidate,
    right: QuoteCandidate,
    bridge: str,
) -> PairDecision | None:
    hard_no = _hard_no_for_pair(left, right, bridge)
    if hard_no is not None:
        return hard_no

    norm = normalize_bridge(bridge)
    left_role, right_role = _heuristic_roles(left, right, bridge)
    if _bridge_is_quote_punctuation(norm):
        return PairDecision(
            MergeAnswer.YES, left_role, right_role, SpeakerContinuity.SAME,
            reason_code=ReasonCode.CONTINUED_NEWS_QUOTE,
        )
    if _same_known_speakers(left, right) and _word_count(norm) <= 60:
        return PairDecision(
            MergeAnswer.YES, left_role, right_role, SpeakerContinuity.SAME,
            reason_code=ReasonCode.SAME_SPEAKER,
        )
    if _bridge_is_pure_local_attribution(norm):
        if (
            _same_known_speakers(left, right)
            or _bridge_has_person_attribution(bridge)
        ):
            return PairDecision(
                MergeAnswer.YES, left_role, right_role, SpeakerContinuity.SAME,
                reason_code=ReasonCode.SAME_ATTRIBUTION_CLAUSE,
            )
    return None


def pair_cache_key(context: AtomicPairContext) -> str:
    payload = {
        "left_text": context.left.text,
        "right_text": context.right.text,
        "bridge": context.bridge,
        "left_speaker": context.left.speaker,
        "right_speaker": context.right.speaker,
        "left_speaker_qid": context.left.speaker_qid,
        "right_speaker_qid": context.right.speaker_qid,
        "left_speaker_probability": context.left.speaker_probability,
        "right_speaker_probability": context.right.speaker_probability,
        "bridge_mentions": context.bridge_mentioned_entities,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class DiskPairCache:
    def __init__(self, directory: str | Path | None = None) -> None:
        if diskcache is None:
            raise ImportError("diskcache is required for DiskPairCache")
        cache_dir = Path(directory or ".quote_merger_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(cache_dir))

    def get(self, key: str) -> PairDecision | None:
        raw = self._cache.get(key)
        return _pair_decision_from_dict(raw) if raw is not None else None

    def set(self, key: str, decision: PairDecision) -> None:
        self._cache[key] = _pair_decision_to_dict(decision)


def _pair_decision_to_dict(decision: PairDecision) -> dict[str, Any]:
    return {
        "merge": decision.merge.value,
        "left_role": decision.left_role.value,
        "right_role": decision.right_role.value,
        "speaker_continuity": decision.speaker_continuity.value,
        "logprob_margin": decision.logprob_margin,
        "reason_code": decision.reason_code.value,
        "source": decision.source,
    }


def _pair_decision_from_dict(raw: Mapping[str, Any]) -> PairDecision:
    return PairDecision(
        merge=MergeAnswer(raw["merge"]),
        left_role=QuoteRole(raw["left_role"]),
        right_role=QuoteRole(raw["right_role"]),
        speaker_continuity=SpeakerContinuity(raw["speaker_continuity"]),
        logprob_margin=float(raw.get("logprob_margin") or 0.0),
        reason_code=ReasonCode(raw.get("reason_code", ReasonCode.AMBIGUOUS.value)),
        source=str(raw.get("source") or "cache"),
    )


def build_llm_prompt(context: AtomicPairContext) -> str:
    def quote_block(name: str, quote: QuoteCandidate) -> str:
        speaker_bits = [quote.speaker or "unknown"]
        if quote.speaker_qid:
            speaker_bits.append(f"qid={quote.speaker_qid}")
        if quote.speaker_probability:
            speaker_bits.append(f"p={quote.speaker_probability:.2f}")
        return (
            f"{name}: id={quote.quote_id or 'unknown'} speaker={' '.join(speaker_bits)}\n"
            f"{compact_quote(quote.text)}"
        )

    bridge = normalize_bridge(context.bridge) or "<empty>"
    bridge_mentions = (
        ", ".join(context.bridge_mentioned_entities)
        if context.bridge_mentioned_entities
        else "none"
    )
    return "\n\n".join(
        [
            "Task: classify merge and roles for this adjacent quote pair.",
            quote_block("left_quote", context.left),
            f"bridge:\n{bridge[:300]}",
            f"bridge_mentioned_entities: {bridge_mentions}",
            quote_block("right_quote", context.right),
            "Return JSON with merge, left_role, right_role, speaker_continuity.",
        ]
    )


def _enum_logprob_margin(content_logprobs: Sequence[Mapping[str, Any]]) -> float:
    enum_values = {
        "yes", "no", "uncertain", "utterance", "scare", "non_quote",
        "same", "different", "unknown",
    }
    best = 0.0
    for entry in content_logprobs:
        token = str(entry.get("token", "")).strip().strip('"')
        if token not in enum_values:
            continue
        top = float(entry.get("logprob", -999.0))
        alternatives = entry.get("top_logprobs") or []
        if len(alternatives) < 2:
            continue
        second = float(alternatives[1].get("logprob", -999.0))
        best = max(best, top - second)
    return best


class OpenAICompatibleAdjudicator:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        key_path: str | Path | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        max_output_tokens: int = 64,
        extra_body: Mapping[str, Any] | None = None,
        system_prompt: str | None = None,
        logprob_margin_threshold: float = DEFAULT_LOGPROB_MARGIN,
        self_consistency_samples: int = DEFAULT_SELF_CONSISTENCY_SAMPLES,
        use_self_consistency: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.key_path = key_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self.extra_body = dict(extra_body or {})
        self.system_prompt = system_prompt or load_system_prompt()
        self.logprob_margin_threshold = logprob_margin_threshold
        self.self_consistency_samples = self_consistency_samples
        self.use_self_consistency = use_self_consistency
        self._is_vllm = "localhost" in self.base_url or "127.0.0.1" in self.base_url
        self._is_openai = "api.openai.com" in self.base_url
        self._uses_reasoning_effort = self._is_openai and self.model.startswith("gpt-5")
        self._supports_logprobs = not (self._is_openai and self.model.startswith("gpt-5"))

    def __call__(self, context: AtomicPairContext) -> PairDecision:
        return self.adjudicate(context)

    def adjudicate(self, context: AtomicPairContext) -> PairDecision:
        parsed, margin = self._request_once(context, temperature=0.0)
        decision = _pair_decision_from_llm(parsed, margin)
        if (
            self.use_self_consistency
            and self._supports_logprobs
            and margin < self.logprob_margin_threshold
            and self.self_consistency_samples > 0
        ):
            votes = [decision.merge]
            for _ in range(self.self_consistency_samples):
                sample, _ = self._request_once(context, temperature=0.3)
                votes.append(MergeAnswer(sample["merge"]))
            decision = _apply_majority_vote(decision, votes)
        return decision

    async def adjudicate_many(
        self,
        contexts: Sequence[AtomicPairContext],
        concurrency: int = 16,
    ) -> list[PairDecision]:
        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(context: AtomicPairContext) -> PairDecision:
            async with semaphore:
                try:
                    return await asyncio.to_thread(self.adjudicate, context)
                except Exception:
                    left_role, right_role = _heuristic_roles(context.left, context.right, context.bridge)
                    return PairDecision(
                        MergeAnswer.NO,
                        left_role,
                        right_role,
                        SpeakerContinuity.UNKNOWN,
                        reason_code=ReasonCode.LLM_ERROR,
                        source="llm",
                    )

        return list(await asyncio.gather(*(run_one(ctx) for ctx in contexts)))

    def _request_once(
        self,
        context: AtomicPairContext,
        *,
        temperature: float,
    ) -> tuple[dict[str, Any], float]:
        prompt = build_llm_prompt(context)
        token_budget = max(self.max_output_tokens, 128 if self._uses_reasoning_effort else self.max_output_tokens)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if not self._uses_reasoning_effort:
            payload["temperature"] = temperature
        if self._uses_reasoning_effort:
            payload["max_completion_tokens"] = token_budget
            payload["reasoning_effort"] = "minimal"
        else:
            payload["max_tokens"] = token_budget
        if self._supports_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
        if self._is_vllm or "guided_json" in self.extra_body:
            payload["extra_body"] = {"guided_json": self.extra_body.get("guided_json", PAIR_JSON_SCHEMA)}
        else:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "quote_pair_decision",
                    "schema": PAIR_JSON_SCHEMA,
                    "strict": True,
                },
            }
        for key, value in self.extra_body.items():
            if key != "guided_json":
                payload[key] = value

        data = self._post_json(f"{self.base_url}/chat/completions", payload)
        content = data["choices"][0]["message"]["content"]
        if not content:
            finish = data["choices"][0].get("finish_reason")
            raise ValueError(f"Empty LLM content (finish_reason={finish})")
        parsed = json.loads(content)
        if self._supports_logprobs:
            logprobs = data["choices"][0].get("logprobs") or {}
            margin = _enum_logprob_margin(logprobs.get("content") or [])
        else:
            margin = DEFAULT_LOGPROB_MARGIN
        return parsed, margin

    def _post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        headers = {
            "Authorization": f"Bearer {resolve_api_key(self.api_key, self.key_path)}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.post(url, headers=headers, json=payload)
                    if response.status_code == 429:
                        retry_after = float(response.headers.get("Retry-After", "1"))
                        time.sleep(retry_after)
                        continue
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        response.raise_for_status()
                    if response.status_code >= 500:
                        response.raise_for_status()
                    return response.json()
                except httpx.HTTPError as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        time.sleep(0.5 * (2**attempt))
        assert last_error is not None
        raise last_error


def _pair_decision_from_llm(parsed: Mapping[str, Any], margin: float) -> PairDecision:
    return PairDecision(
        merge=MergeAnswer(parsed.get("merge", "uncertain")),
        left_role=QuoteRole(parsed.get("left_role", "utterance")),
        right_role=QuoteRole(parsed.get("right_role", "utterance")),
        speaker_continuity=SpeakerContinuity(parsed.get("speaker_continuity", "unknown")),
        logprob_margin=margin,
        reason_code=ReasonCode.AMBIGUOUS,
        source="llm",
    )


def _apply_majority_vote(decision: PairDecision, votes: Sequence[MergeAnswer]) -> PairDecision:
    counts = {MergeAnswer.YES: 0, MergeAnswer.NO: 0, MergeAnswer.UNCERTAIN: 0}
    for vote in votes:
        counts[vote] += 1
    winner = max(counts, key=lambda key: counts[key])
    if winner is MergeAnswer.UNCERTAIN:
        winner = MergeAnswer.NO
    return PairDecision(
        merge=winner,
        left_role=decision.left_role,
        right_role=decision.right_role,
        speaker_continuity=decision.speaker_continuity,
        logprob_margin=decision.logprob_margin,
        reason_code=decision.reason_code,
        source=decision.source,
    )


OpenAIQuoteMergeAdjudicator = OpenAICompatibleAdjudicator


class QuoteTurnMerger:
    def __init__(
        self,
        adjudicator: Adjudicator | None = None,
        cache: dict[str, PairDecision] | None = None,
        disk_cache: DiskPairCache | None = None,
        auto_accept_logprob_margin: float = DEFAULT_LOGPROB_MARGIN,
        prefetch_llm: bool = True,
    ) -> None:
        self.adjudicator = adjudicator
        self.cache = cache if cache is not None else {}
        self.disk_cache = disk_cache
        self.auto_accept_logprob_margin = auto_accept_logprob_margin
        self.prefetch_llm = prefetch_llm

    def decide_pair(self, context: AtomicPairContext) -> PairDecision:
        if len(normalize_bridge(context.bridge)) > 300:
            left_role, right_role = _heuristic_roles(context.left, context.right, context.bridge)
            return PairDecision(
                MergeAnswer.NO, left_role, right_role, SpeakerContinuity.UNKNOWN,
                reason_code=ReasonCode.TOO_DISTANT,
            )

        heuristic = _heuristic_pair_decision(context.left, context.right, context.bridge)
        if heuristic is not None:
            return heuristic

        if self.adjudicator is None:
            left_role, right_role = _heuristic_roles(context.left, context.right, context.bridge)
            return PairDecision(
                MergeAnswer.NO, left_role, right_role, SpeakerContinuity.UNKNOWN,
                reason_code=ReasonCode.AMBIGUOUS,
            )

        cached = self._get_cached_pair(context)
        if cached is not None:
            return cached

        try:
            llm_decision = self.adjudicator(context)
        except Exception:
            left_role, right_role = _heuristic_roles(context.left, context.right, context.bridge)
            llm_decision = PairDecision(
                MergeAnswer.NO, left_role, right_role, SpeakerContinuity.UNKNOWN,
                reason_code=ReasonCode.LLM_ERROR, source="llm",
            )

        if (
            llm_decision.source == "llm"
            and llm_decision.logprob_margin < self.auto_accept_logprob_margin
            and llm_decision.merge is not MergeAnswer.NO
        ):
            llm_decision = PairDecision(
                merge=MergeAnswer.NO,
                left_role=llm_decision.left_role,
                right_role=llm_decision.right_role,
                speaker_continuity=llm_decision.speaker_continuity,
                logprob_margin=llm_decision.logprob_margin,
                reason_code=ReasonCode.AMBIGUOUS,
                source=llm_decision.source,
            )

        self._store_cached_pair(context, llm_decision)
        return llm_decision

    def merge_around(
        self,
        quotes: list[QuoteCandidate],
        bridges: list[str],
        current_index: int,
        bridge_mentions: list[tuple[str, ...]] | None = None,
    ) -> MergeBlock:
        if len(bridges) != max(0, len(quotes) - 1):
            raise ValueError("bridges must have len(quotes) - 1 entries")
        if current_index < 0 or current_index >= len(quotes):
            raise IndexError("current_index outside quotes")

        start = end = current_index
        decisions: list[MergeDecision] = []

        while True:
            merge_prev = False
            merge_next = False
            prev_pair: PairDecision | None = None
            next_pair: PairDecision | None = None

            if start > 0:
                bridge_idx = start - 1
                prev_pair = self.decide_pair(
                    AtomicPairContext(
                        left=quotes[start - 1],
                        right=quotes[start],
                        bridge=bridges[bridge_idx],
                        bridge_mentioned_entities=(bridge_mentions[bridge_idx] if bridge_mentions else ()),
                    )
                )
                merge_prev = prev_pair.should_merge()

            if end + 1 < len(quotes):
                bridge_idx = end
                next_pair = self.decide_pair(
                    AtomicPairContext(
                        left=quotes[end],
                        right=quotes[end + 1],
                        bridge=bridges[bridge_idx],
                        bridge_mentioned_entities=(bridge_mentions[bridge_idx] if bridge_mentions else ()),
                    )
                )
                merge_next = next_pair.should_merge()

            reason = ReasonCode.HEURISTIC_ONLY
            source = "heuristic"
            confidence = Confidence.HIGH
            margin = 0.0
            left_role = QuoteRole.UTTERANCE
            right_role = QuoteRole.UTTERANCE
            if prev_pair is not None:
                reason = prev_pair.reason_code
                source = prev_pair.source
                confidence = prev_pair.confidence
                margin = prev_pair.logprob_margin
                left_role = prev_pair.left_role
            if next_pair is not None:
                if next_pair.source == "llm" or not merge_prev:
                    reason = next_pair.reason_code
                    source = next_pair.source
                    confidence = next_pair.confidence
                margin = max(margin, next_pair.logprob_margin)
                right_role = next_pair.right_role

            decisions.append(
                MergeDecision(
                    merge_prev=merge_prev,
                    merge_next=merge_next,
                    confidence=confidence,
                    reason_code=reason,
                    left_role=left_role,
                    right_role=right_role,
                    logprob_margin=margin,
                    source=source,
                )
            )

            changed = False
            if merge_prev and start > 0:
                start -= 1
                changed = True
            if merge_next and end + 1 < len(quotes):
                end += 1
                changed = True
            if not changed:
                break

        text, quote_text = _block_texts(quotes, bridges, start, end)
        return MergeBlock(
            start_index=start,
            end_index=end,
            quote_ids=tuple(q.quote_id for q in quotes[start : end + 1]),
            text=text,
            quote_text=quote_text,
            decisions=tuple(decisions),
            left_role=decisions[-1].left_role if decisions else QuoteRole.UTTERANCE,
            right_role=decisions[-1].right_role if decisions else QuoteRole.UTTERANCE,
        )

    def merge_all(
        self,
        quotes: list[QuoteCandidate],
        bridges: list[str],
        bridge_mentions: list[tuple[str, ...]] | None = None,
    ) -> list[MergeBlock]:
        if len(bridges) != max(0, len(quotes) - 1):
            raise ValueError("bridges must have len(quotes) - 1 entries")

        if self.prefetch_llm and self.adjudicator is not None:
            self._prefetch_ambiguous_pairs(quotes, bridges, bridge_mentions)

        blocks: list[MergeBlock] = []
        idx = 0
        while idx < len(quotes):
            block = self.merge_around(quotes, bridges, idx, bridge_mentions=bridge_mentions)
            blocks.append(block)
            idx = block.end_index + 1
        return blocks

    def _prefetch_ambiguous_pairs(
        self,
        quotes: list[QuoteCandidate],
        bridges: list[str],
        bridge_mentions: list[tuple[str, ...]] | None,
    ) -> None:
        pending: list[AtomicPairContext] = []
        for idx, bridge in enumerate(bridges):
            context = AtomicPairContext(
                left=quotes[idx],
                right=quotes[idx + 1],
                bridge=bridge,
                bridge_mentioned_entities=(bridge_mentions[idx] if bridge_mentions else ()),
            )
            if self._get_cached_pair(context) is not None:
                continue
            if _heuristic_pair_decision(context.left, context.right, context.bridge) is not None:
                continue
            pending.append(context)

        if not pending or self.adjudicator is None:
            return

        if hasattr(self.adjudicator, "adjudicate_many"):
            decisions = asyncio.run(self.adjudicator.adjudicate_many(pending))
            for context, decision in zip(pending, decisions):
                self._store_cached_pair(context, decision)
            return

        for context in pending:
            self.decide_pair(context)

    def _get_cached_pair(self, context: AtomicPairContext) -> PairDecision | None:
        key = pair_cache_key(context)
        if key in self.cache:
            return self.cache[key]
        if self.disk_cache is not None:
            cached = self.disk_cache.get(key)
            if cached is not None:
                self.cache[key] = cached
                return cached
        return None

    def _store_cached_pair(self, context: AtomicPairContext, decision: PairDecision) -> None:
        key = pair_cache_key(context)
        self.cache[key] = decision
        if self.disk_cache is not None:
            self.disk_cache.set(key, decision)


def _block_texts(
    quotes: list[QuoteCandidate],
    bridges: list[str],
    start: int,
    end: int,
) -> tuple[str, str]:
    raw_parts: list[str] = []
    quote_parts: list[str] = []
    for idx in range(start, end + 1):
        if idx > start:
            raw_parts.append(bridges[idx - 1])
        raw_parts.append(quotes[idx].text)
        quote_parts.append(quotes[idx].text)
    return "".join(raw_parts), " ".join(quote_parts)


__all__ = [
    "AtomicPairContext",
    "Confidence",
    "DEFAULT_LOGPROB_MARGIN",
    "DEFAULT_MODEL",
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
    "build_llm_prompt",
    "compact_quote",
    "load_system_prompt",
    "pair_cache_key",
    "resolve_api_key",
]
