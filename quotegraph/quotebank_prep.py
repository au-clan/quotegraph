"""Reproduce Quootstrap HTML → PTB conversion (EntryWrapperBuilder).

Uses jsoup 1.10.3 + Stanford CoreNLP 3.8.0 PTBTokenizer with the same
whitelist, retained tags, and tokenizer options as the dump converter.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve

JSOUP_URL = "https://repo1.maven.org/maven2/org/jsoup/jsoup/1.10.3/jsoup-1.10.3.jar"
CORENLP_URL = (
    "https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.8.0/"
    "stanford-corenlp-3.8.0.jar"
)
TOKENIZER_SETTINGS = (
    "tokenizeNLs=false, americanize=false, normalizeCurrency=false, "
    "normalizeParentheses=false, normalizeOtherBrackets=false, "
    "unicodeQuotes=false, ptb3Ellipsis=true, escapeForwardSlashAsterisk=false, "
    "untokenizable=noneKeep, normalizeSpace=false"
)
HTML_WHITELIST = [
    "a", "b", "blockquote", "br", "caption", "cite", "code", "col", "colgroup",
    "dd", "div", "dl", "dt", "em", "h1", "h2", "h3", "h4", "h5", "h6", "i",
    "center", "font", "abbr", "li", "ol", "p", "pre", "q", "small", "span",
    "strike", "strong", "sub", "sup", "table", "tbody", "td", "tfoot", "th",
    "thead", "tr", "u", "ul",
]
HTML_RETAINED = {
    "a", "blockquote", "br", "caption", "cite",
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "q",
}

_jvm = False
_cleaner = None
_factory = None


def default_jar_dir() -> Path:
    return Path(os.environ.get("QUOTEGRAPH_JARS", "/home/mculjak/datasets/quotegraph_poc/jars"))


def ensure_jars(jar_dir: Path | None = None) -> tuple[Path, Path]:
    jar_dir = Path(jar_dir or default_jar_dir())
    jar_dir.mkdir(parents=True, exist_ok=True)
    jsoup = jar_dir / "jsoup-1.10.3.jar"
    corenlp = jar_dir / "stanford-corenlp-3.8.0.jar"
    if not jsoup.exists():
        print(f"downloading {JSOUP_URL}", flush=True)
        urlretrieve(JSOUP_URL, jsoup)
    if not corenlp.exists():
        print(f"downloading {CORENLP_URL}", flush=True)
        urlretrieve(CORENLP_URL, corenlp)
    return jsoup, corenlp


def start_jvm(jar_dir: Path | None = None) -> None:
    global _jvm, _cleaner, _factory
    if _jvm:
        return
    import jpype
    from jpype import JClass

    jsoup, corenlp = ensure_jars(jar_dir)
    jpype.startJVM(classpath=[str(jsoup), str(corenlp)], convertStrings=True)
    whitelist = JClass("org.jsoup.safety.Whitelist")()
    for tag in HTML_WHITELIST:
        whitelist.addTags(tag)
    whitelist.addAttributes("a", "href", "title", "target", "rel")
    whitelist.addAttributes("blockquote", "cite")
    whitelist.addAttributes("col", "span", "width")
    whitelist.addAttributes("colgroup", "span", "width")
    whitelist.addAttributes("ol", "start", "type")
    whitelist.addAttributes("q", "cite")
    whitelist.addAttributes("table", "summary", "width")
    whitelist.addAttributes("td", "abbr", "axis", "colspan", "rowspan", "width")
    whitelist.addAttributes("th", "abbr", "axis", "colspan", "rowspan", "scope", "width")
    whitelist.addAttributes("ul", "type")
    whitelist.addProtocols("a", "href", "ftp", "http", "https", "mailto", "#")
    whitelist.addProtocols("blockquote", "cite", "http", "https")
    whitelist.addProtocols("cite", "cite", "http", "https")
    whitelist.addProtocols("q", "cite", "http", "https")
    _cleaner = JClass("org.jsoup.safety.Cleaner")(whitelist)
    _factory = JClass("edu.stanford.nlp.process.CoreLabelTokenFactory")()
    _jvm = True


def clean_content(html: str, url: str = "") -> str:
    """jsoup whitelist clean + unwrap structural tags; return body HTML."""
    start_jvm()
    from jpype import JClass

    doc = JClass("org.jsoup.Jsoup").parse(html or "", url or "")
    doc = _cleaner.clean(doc)
    strip = [t for t in HTML_WHITELIST if t not in HTML_RETAINED]
    doc.select(",".join(strip)).unwrap()
    doc.outputSettings().prettyPrint(False)
    body = doc.body()
    return str(body.html()) if body is not None else ""


def tokenize_ptb(text: str) -> list[str]:
    """Stanford PTBTokenizer with Quootstrap options. Uses word(), not toString()."""
    start_jvm()
    from jpype import JClass
    from jpype.types import JString

    reader = JClass("java.io.StringReader")(JString(text or ""))
    ptbt = JClass("edu.stanford.nlp.process.PTBTokenizer")(reader, _factory, TOKENIZER_SETTINGS)
    tokens: list[str] = []
    while ptbt.hasNext():
        tokens.append(str(ptbt.next().word()))
    return tokens


def html_to_ptb(html: str, url: str = "") -> list[str]:
    return tokenize_ptb(clean_content(html, url))
