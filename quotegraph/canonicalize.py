"""Turn dump URLs into ones Wayback / live servers actually stored."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

TRACKING = {
    "feedtype",
    "feedname",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
    "tracking_source",
    "campaign_id",
    "odyssey",
    "ref",
    "rss",
    "psp",
}
_PATH_ID = re.compile(r"id[A-Z0-9]{6,}|\.html(?:$|/)|/\d{8}/|/\d{5,}", re.I)


def _set_scheme_host(url: str, scheme: str | None = None, host: str | None = None) -> str:
    p = urlparse(url)
    return urlunparse(
        (
            scheme or p.scheme,
            host or p.netloc,
            p.path,
            p.params,
            p.query,
            p.fragment,
        )
    )


def unwrap_embedded(url: str) -> str | None:
    """Philly /r?40=http://..., and similar click-wrappers."""
    for _, value in parse_qsl(urlparse(url).query, keep_blank_values=True):
        dest = unquote(value).strip()
        if dest.startswith("http://") or dest.startswith("https://"):
            return dest
    return None


def unwrap_yahoo(url: str) -> str | None:
    host = urlparse(url).netloc.lower()
    if "rd.yahoo.com" not in host or "*" not in url:
        return None
    dest = unquote(url.split("*", 1)[1]).strip()
    if dest.startswith("http://") or dest.startswith("https://"):
        return dest
    return None


def strip_tracking(url: str) -> str | None:
    p = urlparse(url)
    if not p.query:
        return None
    kept = [
        (k, v)
        for k, v in parse_qsl(p.query, keep_blank_values=True)
        if k.lower() not in TRACKING and not k.lower().startswith("utm_")
    ]
    if len(kept) == len(parse_qsl(p.query, keep_blank_values=True)):
        return None
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(kept), p.fragment))


def strip_query_if_path_identifies(url: str) -> str | None:
    p = urlparse(url)
    if not p.query or not _PATH_ID.search(p.path):
        return None
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))


MATERIAL = {"yahoo", "embedded", "no_tracking", "no_query"}


def url_variants(url: str, host_tweaks: bool = True) -> list[tuple[str, str]]:
    """(label, url) including the original, then rewritten forms."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    def add(label: str, candidate: str | None) -> None:
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        out.append((label, candidate))

    add("orig", url)
    inner = unwrap_yahoo(url) or unwrap_embedded(url)
    add("yahoo", unwrap_yahoo(url))
    add("embedded", unwrap_embedded(url))
    base = inner or url
    add("no_tracking", strip_tracking(base))
    add("no_query", strip_query_if_path_identifies(base))
    if host_tweaks:
        for label, cand in list(out):
            if label == "orig":
                continue
            p = urlparse(cand)
            host = p.netloc
            if p.scheme == "http":
                add(label + "_https", _set_scheme_host(cand, scheme="https"))
            if host.startswith("www."):
                add(label + "_bare", _set_scheme_host(cand, host=host[4:]))
            elif host and "." in host:
                add(label + "_www", _set_scheme_host(cand, host="www." + host))
    return out


def has_material_rewrite(url: str) -> bool:
    labs = {lab for lab, _ in url_variants(url, host_tweaks=False)}
    return bool(labs & {"yahoo", "embedded", "no_tracking"})
