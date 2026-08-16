"""Restore casing and damaged characters on dump text."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import ftfy

_WORD = re.compile(r"\S+")


def fold_key(tok: str) -> str:
    """ASCII letters/digits only. ``beyonc?`` and ``Beyoncé`` both → ``beyonc``."""
    return "".join(c.lower() for c in tok if c.isascii() and c.isalnum())


def align_key(tok: str) -> str:
    return fold_key(tok) or tok.lower()


def wildcard_match(dump: str, cand: str) -> bool:
    if "?" not in dump:
        return fold_key(dump) == fold_key(cand)
    pat = "".join("." if c == "?" else re.escape(c) for c in dump)
    return re.fullmatch(pat, cand, flags=re.I) is not None


def fills_encoding(dump: str, cand: str) -> bool:
    """True when ``?`` slots are filled by non-ASCII or non-alnum (e.g. é, ')."""
    if "?" not in dump or len(dump) != len(cand):
        return False
    filled = False
    for src, dst in zip(dump, cand):
        if src == "?":
            if dst.isascii() and dst.isalnum():
                return False
            filled = True
        elif src.lower() != dst.lower():
            return False
    return filled


def looks_mojibake(tok: str) -> bool:
    if any(c in tok for c in "ÃÂâ"):
        return True
    if tok.startswith(("€œ", "€˜", "€™", "€")):
        return True
    return bool(tok.startswith("€") and len(tok) > 1 and tok[1].isalpha())


def needs_char_fix(tok: str) -> bool:
    return "?" in tok or looks_mojibake(tok)


def is_encoding_damage(dump: str, gold: str) -> bool:
    """Tokens the HTML gold shows were smashed, not real question marks."""
    if looks_mojibake(dump):
        return dump != gold
    if "?" not in dump:
        return False
    if "?" in gold:
        return False
    return align_key(dump) == align_key(gold) and dump.lower() != gold.lower()


def build_lexicon(token_lists: list[list[str]]) -> dict[str, str]:
    """Majority surface per fold_key. Prefer non-ASCII (phase E), not D's ``?`` smash."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for tokens in token_lists:
        for tok in tokens:
            key = fold_key(tok)
            if not key:
                continue
            if tok.isascii() and tok.islower():
                continue
            counts[key][tok] += 1
    out: dict[str, str] = {}
    for key, cnt in counts.items():
        non_ascii = [(surf, n) for surf, n in cnt.items() if not surf.isascii()]
        if non_ascii:
            out[key] = max(non_ascii, key=lambda item: item[1])[0]
        else:
            out[key] = cnt.most_common(1)[0][0]
    return out


def lexicon_by_length(lexicon: dict[str, str]) -> dict[int, list[str]]:
    by_len: dict[int, list[str]] = defaultdict(list)
    for surf in lexicon.values():
        by_len[len(surf)].append(surf)
    return by_len


def _unique_wildcard(tok: str, by_len: dict[int, list[str]]) -> str | None:
    hits = [surf for surf in by_len.get(len(tok), []) if wildcard_match(tok, surf)]
    if len(hits) == 1:
        return hits[0]
    return None


_EDGE_PUNCT = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


def apply_lexicon(
    text: str,
    lexicon: dict[str, str],
    by_len: dict[int, list[str]] | None = None,
) -> str:
    if by_len is None:
        by_len = lexicon_by_length(lexicon)

    def restore_core(tok: str) -> str:
        if "?" not in tok:
            return tok
        cand = lexicon.get(fold_key(tok))
        if cand and fills_encoding(tok, cand):
            return cand
        unique = _unique_wildcard(tok, by_len)
        if unique and fills_encoding(tok, unique):
            return unique
        return tok

    def repl(match: re.Match[str]) -> str:
        tok = match.group()
        prefix, core, suffix = _EDGE_PUNCT.match(tok).groups()
        if not core:
            return tok
        restored = restore_core(core)
        if restored == core:
            restored = restore_core(tok)
            return restored
        return prefix + restored + suffix

    return _WORD.sub(repl, text)


_QUESTION_LEXICON: dict[str, str] | None = None
_QUESTION_BY_LEN: dict[int, list[str]] | None = None
DEFAULT_QUESTION_LEXICON = Path(
    os.environ.get("QUOTEGRAPH_LEXICON", "/home/mculjak/datasets/quotegraph_poc/lexicon_E.json")
)


def question_lexicon(lexicon: dict[str, str]) -> dict[str, str]:
    """Keep only non-ASCII surfaces so D-style ``Fran?ois`` cannot win."""
    return {key: surf for key, surf in lexicon.items() if not surf.isascii()}


def load_question_lexicon(path: Path | None = None) -> dict[str, str]:
    global _QUESTION_LEXICON, _QUESTION_BY_LEN
    if _QUESTION_LEXICON is not None and path is None:
        return _QUESTION_LEXICON
    lex_path = Path(path) if path is not None else DEFAULT_QUESTION_LEXICON
    if not lex_path.exists():
        _QUESTION_LEXICON = {}
        _QUESTION_BY_LEN = {}
        return _QUESTION_LEXICON
    raw = json.loads(lex_path.read_text(encoding="utf-8"))
    _QUESTION_LEXICON = question_lexicon(raw)
    _QUESTION_BY_LEN = lexicon_by_length(_QUESTION_LEXICON)
    return _QUESTION_LEXICON


def restore_question_marks(text: str, lexicon: dict[str, str] | None = None) -> str:
    """Replace encoding ``?`` from a phase-E lexicon; leave real questions alone."""
    if "?" not in text:
        return text
    if lexicon is None:
        lexicon = load_question_lexicon()
    if not lexicon:
        return text
    by_len = _QUESTION_BY_LEN if lexicon is _QUESTION_LEXICON else lexicon_by_length(lexicon)
    return apply_lexicon(text, lexicon, by_len)


def project_aligned(dump_words: list[str], gold_words: list[str]) -> list[str]:
    """Copy gold surface when fold keys match (case + unicode)."""
    out = []
    for d, g in zip(dump_words, gold_words):
        if align_key(d) == align_key(g):
            out.append(g)
        else:
            out.append(d)
    return out


# UTF-8 curly quotes / dashes read as Windows-1252, often split by PTB.
_SPLIT_OPEN = re.compile(r"â\s+€\s*œ")
_SPLIT_APOS = re.compile(r"â\s+€\s*™")
_SPLIT_CLOSE = re.compile(r"â\s+€\s*")
_SPLIT_LSQUO = re.compile(r"â\s+€\s*˜")
_SPLIT_EURO = re.compile(r"â\s+€")
_SPLIT_CONTRACTION = re.compile(r"â\s+(s|t|re|ve|ll|d)\b", re.I)
_LEAD_OPEN = re.compile(r"(^|[\s“”\"'])€œ")
_LEAD_EURO_WORD = re.compile(r"(^|[\s“”\"'])€(?=[A-Za-z])")
_GLUE_OPEN = re.compile(r"(\S)([“])")
_GLUE_CLOSE = re.compile(r"([”])(?=[A-Za-z])")
_CP1252 = (
    ("â€œ", "“"),
    ("â€™", "’"),
    ("â€", "”"),
    ("â€˜", "‘"),
    ("â€”", "—"),
    ("â€“", "–"),
    ("â€", "”"),
)


def fix_mojibake(text: str) -> str:
    """Rejoin PTB-split cp1252 sequences, then ftfy. Leaves real ``€100`` alone."""
    if not text or not any(c in text for c in "ÃÂâ€"):
        return text
    text = _SPLIT_OPEN.sub("â€œ", text)
    text = _SPLIT_APOS.sub("â€™", text)
    text = _SPLIT_CLOSE.sub("â€", text)
    text = _SPLIT_LSQUO.sub("â€˜", text)
    text = _SPLIT_CONTRACTION.sub(r"'\1", text)
    text = _SPLIT_EURO.sub("â€", text)
    for src, dst in _CP1252:
        text = text.replace(src, dst)
    text = _LEAD_OPEN.sub(r"\1“", text)
    text = _LEAD_EURO_WORD.sub(r"\1", text)
    # Unknown leftover â is usually a truncated quote or possessive, not a letter.
    text = re.sub(r"â(?=\s|$)", "", text)
    text = _GLUE_OPEN.sub(r"\1 \2", text)
    text = _GLUE_CLOSE.sub(r"\1 ", text)
    return ftfy.fix_text(text, uncurl_quotes=False)


def ftfy_words(words: list[str]) -> list[str]:
    return fix_mojibake(" ".join(words)).split()


def sentence_truecase(text: str) -> str:
    """Capitalize the first letter of the text and after . ! ? (skipping quotes)."""
    chars = list(text)
    cap = True
    for i, ch in enumerate(chars):
        if cap and ch.isalpha():
            chars[i] = ch.upper()
            cap = False
        elif ch in ".!?":
            cap = True
    return "".join(chars)


def map_pred_words(src_words: list[str], pred_words: list[str]) -> list[str]:
    """Project a possibly retokenized prediction onto ``src_words``."""
    out = list(src_words)
    src_keys = [align_key(t) for t in src_words]
    pred_keys = [align_key(t) for t in pred_words]
    for tag, i1, i2, j1, j2 in SequenceMatcher(
        a=src_keys, b=pred_keys, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            out[i1:i2] = pred_words[j1:j2]
    return out


def vote_surfaces(candidates: list[str]) -> str:
    """Exact majority; ties keep the first candidate (Moses)."""
    if not candidates:
        return ""
    counts = Counter(candidates)
    best, n = counts.most_common(1)[0]
    if n >= 2:
        return best
    return candidates[0]


def ensemble_truecase(src_text: str, pred_texts: list[str]) -> str:
    """Majority-vote surfaces from several truecasers onto the source tokens."""
    src_words = src_text.split()
    if not src_words or not pred_texts:
        return src_text
    mapped = [map_pred_words(src_words, pred.split()) for pred in pred_texts]
    return " ".join(vote_surfaces([row[i] for row in mapped]) for i in range(len(src_words)))
