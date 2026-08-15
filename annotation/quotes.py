"""Detect direct-quote candidates: every span between quotation marks."""

from __future__ import annotations

from dataclasses import dataclass

PTB_OPEN = "``"
PTB_CLOSE = "''"

# Dedicated opening / closing marks (ASCII double quote is handled as a toggle).
OPEN_MARKS = frozenset({"\u201c", "\u201e", "\u00ab", "\u2018"})
CLOSE_MARKS = frozenset({"\u201d", "\u00bb", "\u2019"})
TOGGLE_MARKS = frozenset({'"'})

# Single quotes that are almost always apostrophes when adjacent to a letter.
_APOSTROPHE = frozenset({"'", "\u2019", "\u2018"})


@dataclass(frozen=True)
class QuoteCandidate:
    """One delimited span.

    ``inner_start``/``inner_end`` cover the text between the marks.
    ``outer_start``/``outer_end`` cover the marks as well.
    """

    inner_start: int
    inner_end: int
    outer_start: int
    outer_end: int
    delimiter: str

    def inner_text(self, text: str) -> str:
        return text[self.inner_start : self.inner_end]


def _is_letter(ch: str) -> bool:
    return ch.isalpha()


@dataclass(frozen=True)
class UnmatchedMark:
    offset: int
    kind: str


def _apostrophe_context(text: str, i: int) -> bool:
    """True if this mark sits between letters (don't, Obama's), not a delimiter."""
    return i > 0 and _is_letter(text[i - 1]) and i + 1 < len(text) and _is_letter(text[i + 1])


def _blockquote_open_len(text_low: str, i: int) -> int:
    if text_low.startswith("\\blockquote", i):
        return len("\\blockquote")
    if text_low.startswith("<blockquote", i):
        end = text_low.find(">", i)
        if end != -1:
            return end + 1 - i
    return 0


def _unmatched_double_closer_is_opener(text: str, i: int) -> bool:
    """True for continuation wraps (faith:”... they're...”) not stray citation marks.

    ``reported” boston globe`` is a leftover PTB closer, not an opening quote.
    """
    j = i - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    if j >= 0 and text[j] in ":,":
        return True
    k = i + 1
    while k < len(text) and text[k].isspace():
        k += 1
    return k < len(text) and text[k] in ".…"


def _blockquote_close_len(text_low: str, i: int) -> int:
    if text_low.startswith("\\endblockquote", i):
        return len("\\endblockquote")
    if text_low.startswith("</blockquote>", i):
        return len("</blockquote>")
    return 0


def _scan(text: str, *, min_inner_chars: int = 1) -> tuple[list[QuoteCandidate], list[UnmatchedMark]]:
    """Pair every quotation-mark span, including nested curly / guillemet pairs.

    ASCII ``"`` and PTB ```` / ``''`` toggle (they do not nest). Apostrophes
    between letters are ignored. ASCII ``'`` is never a delimiter. A new
    curly opener ``“`` while a curly span is already open ends that
    paragraph (multi-paragraph quotations omit closers until the last para).
    """
    found: list[QuoteCandidate] = []
    stack: list[tuple[int, int, str]] = []  # (outer_start, inner_start, kind)
    i = 0
    n = len(text)
    text_low = text.lower()

    def emit(outer_start: int, inner_start: int, inner_end: int, outer_end: int, kind: str) -> None:
        if inner_end - inner_start < min_inner_chars:
            return
        if kind == "curly-single" and inner_end - inner_start < 3:
            return
        found.append(
            QuoteCandidate(
                inner_start=inner_start,
                inner_end=inner_end,
                outer_start=outer_start,
                outer_end=outer_end,
                delimiter=kind,
            )
        )

    while i < n:
        if text.startswith(PTB_OPEN, i):
            stack.append((i, i + 2, "ptb"))
            i += 2
            continue
        if text.startswith(PTB_CLOSE, i):
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][2] == "ptb":
                    outer_start, inner_start, kind = stack.pop(j)
                    emit(outer_start, inner_start, i, i + 2, kind)
                    break
            i += 2
            continue

        ch = text[i]

        open_bq = _blockquote_open_len(text_low, i)
        if open_bq:
            stack.append((i, i + open_bq, "blockquote"))
            i += open_bq
            continue
        close_bq = _blockquote_close_len(text_low, i)
        if close_bq:
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][2] == "blockquote":
                    outer_start, inner_start, kind = stack.pop(j)
                    emit(outer_start, inner_start, i, i + close_bq, kind)
                    break
            i += close_bq
            continue

        if ch in TOGGLE_MARKS:
            if stack and stack[-1][2] == "ascii":
                outer_start, inner_start, kind = stack.pop()
                emit(outer_start, inner_start, i, i + 1, kind)
            else:
                stack.append((i, i + 1, "ascii"))
            i += 1
            continue

        if ch in OPEN_MARKS:
            if ch in _APOSTROPHE and _apostrophe_context(text, i):
                i += 1
                continue
            kind = "guillemet" if ch == "\u00ab" else ("curly-single" if ch == "\u2018" else "curly")
            # Multi-paragraph quotation (AP/CMOS): each paragraph opens with “
            # and only the last paragraph has a closer. A new “ while one is
            # already open ends the previous paragraph, then starts the next.
            if kind == "curly" and stack and stack[-1][2] == "curly":
                outer_start, inner_start, prev_kind = stack.pop()
                inner_end = i
                while inner_end > inner_start and text[inner_end - 1].isspace():
                    inner_end -= 1
                emit(outer_start, inner_start, inner_end, inner_end, prev_kind)
            stack.append((i, i + 1, kind))
            i += 1
            continue

        if ch in CLOSE_MARKS:
            if ch in _APOSTROPHE and _apostrophe_context(text, i):
                i += 1
                continue
            if ch == "\u00bb":
                want = "guillemet"
            elif ch == "\u2019":
                want = "curly-single"
            else:
                want = "curly"
            closed = False
            for j in range(len(stack) - 1, -1, -1):
                if stack[j][2] == want:
                    outer_start, inner_start, kind = stack.pop(j)
                    emit(outer_start, inner_start, i, i + 1, kind)
                    closed = True
                    break
            # Same-direction wrap: ”...they're...” (continuation after a colon,
            # or a detokenizer that emitted closers on both ends). Do not do
            # this for ’ — it is usually an unmatched apostrophe.
            if not closed and want == "curly" and _unmatched_double_closer_is_opener(text, i):
                stack.append((i, i + 1, want))
            i += 1
            continue

        i += 1

    found.sort(key=lambda q: (q.outer_start, -q.outer_end))
    unmatched = [UnmatchedMark(offset=outer_start, kind=kind) for outer_start, _, kind in stack]
    return found, unmatched


def find_quote_candidates(text: str, *, min_inner_chars: int = 1) -> list[QuoteCandidate]:
    """Return every paired quotation-mark span, including nested pairs."""
    candidates, _ = _scan(text, min_inner_chars=min_inner_chars)
    return candidates


def find_unmatched_quote_marks(text: str) -> list[UnmatchedMark]:
    """Opening marks that never closed — surface these for a manual fix."""
    _, unmatched = _scan(text)
    return unmatched
