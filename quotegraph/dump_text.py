"""Clean Quootstrap/Spinn3r dump bodies with off-the-shelf tools.

The jadranka shards are already jsoup-stripped, Stanford-PTB-tokenized text.
This module does not fetch HTML. It detokenizes and removes obvious converter
junk so quote offsets are measured on readable text.
"""

from __future__ import annotations

import re

import ftfy

_MOSES = None

# Converter leftovers that jsoup did not strip. Not linguistic content.
_CONVERTER_JUNK = re.compile(
    r"(?is)"
    r"(?:var\s+\w+\s*=\s*escape\s*\([^;]{0,500}\)\s*;?)"
    r"|(?:openwin\s*\([^)]{0,300}\)\s*;?)"
    r"|(?:document\.write\s*\([^)]{0,200}\)\s*;?)"
    r"|(?:navigator\.\w+(?:\.\w+)*)"
    r"|(?:\{\*\s*[^*]*\*})"
)
_SPACED_HTML = re.compile(
    r"<\s*/?\s*[a-z][a-z0-9]*\b(?:\s+\w+\s*=\s*\"[^\"]*\")*\s*/?\s*>",
    re.I,
)
_LEADING_CHROME = re.compile(
    r"(?is)^(?:your browser's security settings are preventing some features from appearing\.\s*"
    r"|your browser does not support iframes\.\s*)+"
)
_TRAILING_CHROME = re.compile(
    r"(?is)\s*(?:share this:.*|click to share on .*"
    r"|you must fill out the comment body in order to submit a comment\.?)\s*$"
)

BLOCKED_DOMAINS = (
    "wordpress.com",
    "wikia.com",
    "gamereactor.eu",
    "myspace.com",
    "typepad.com",
)


def is_blocked_source(url: str) -> bool:
    """True for blog/wiki hosts excluded from the gold sample."""
    host = url.split("/")[2].lower() if "://" in url else url.lower()
    if host.startswith("www."):
        host = host[4:]
    return any(host == domain or host.endswith("." + domain) for domain in BLOCKED_DOMAINS)


def _moses():
    global _MOSES
    if _MOSES is None:
        from sacremoses import MosesDetokenizer

        _MOSES = MosesDetokenizer(lang="en")
    return _MOSES


_OPEN_Q = "@@OPENQ@@"
_CLOSE_Q = "@@CLOSEQ@@"
# Longest-first: </blockquote> before <blockquote, \endblockquote before \blockquote.
_QUOTE_DELIM = re.compile(
    r"``|''|</blockquote>|<blockquote[^>]*>|\\endblockquote|\\blockquote",
    re.I,
)


def _split_quote_delims(tok: str) -> list[str]:
    """Split PTB / blockquote marks out of mixed tokens (``hello'')."""
    pieces: list[str] = []
    last = 0
    for match in _QUOTE_DELIM.finditer(tok):
        if match.start() > last:
            pieces.append(tok[last : match.start()])
        pieces.append(match.group())
        last = match.end()
    if last < len(tok):
        pieces.append(tok[last:])
    return [p for p in pieces if p] or [tok]


def _quote_placeholder(tok: str) -> str | None:
    low = tok.lower()
    if tok == "``" or low == "\\blockquote" or (
        low.startswith("<blockquote") and low.endswith(">")
    ):
        return _OPEN_Q
    if tok == "''" or low in {"</blockquote>", "\\endblockquote"}:
        return _CLOSE_Q
    return None


def _ptb_tokens_for_moses(tokens: list[str]) -> list[str]:
    """Map Stanford PTB tokens onto what MosesDetokenizer expects.

    Leave quote direction in placeholders. Moses treats `` and '' as
    independent quote toggles, and folding both to ASCII \" crosses pairs
    whenever a closer appears before its opener. ``<blockquote>`` and
    ``\\blockquote`` wraps are the same as `` / '' in ``find_quotation_ends``.
    """
    out: list[str] = []
    for raw in tokens:
        for tok in _split_quote_delims(raw):
            placeholder = _quote_placeholder(tok)
            if placeholder is not None:
                out.append(placeholder)
                continue
            if tok == "n't" and out:
                out[-1] += "n't"
                continue
            out.append(tok)
    return out


def clean_dump_text(text: str) -> str:
    """Detokenize a PTB dump body and drop obvious script/template junk."""
    if not text or not text.strip():
        return ""
    text = ftfy.fix_text(text)
    detok = _moses().detokenize(_ptb_tokens_for_moses(text.split()))
    detok = detok.replace(_OPEN_Q + " ", "“").replace(" " + _CLOSE_Q, "”")
    detok = detok.replace(_OPEN_Q, "“").replace(_CLOSE_Q, "”")
    detok = re.sub(r"\s+”", "”", detok)
    detok = re.sub(r"“\s+", "“", detok)
    detok = _CONVERTER_JUNK.sub(" ", detok)
    detok = _SPACED_HTML.sub(" ", detok)
    detok = _LEADING_CHROME.sub("", detok)
    detok = _TRAILING_CHROME.sub("", detok)
    return re.sub(r"[ \t]+", " ", detok).strip()
