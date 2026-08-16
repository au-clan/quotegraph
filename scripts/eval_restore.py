#!/usr/bin/env python3
"""Compare truecasers and character restorers on aligned A–C HTML."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import ftfy
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quotegraph"))

from dump_text import clean_dump_text
from restore_text import (
    align_key,
    apply_lexicon,
    build_lexicon,
    ensemble_truecase,
    fix_mojibake,
    ftfy_words,
    is_encoding_damage,
    lexicon_by_length,
    project_aligned,
    sentence_truecase,
)
from spinn3r import html_gz_path

LOWER = {"A", "B", "C"}
TRAIN = {"D", "E"}
CORENLP_HOME = Path("/home/mculjak/.cache/stanza/1.12.0/corenlp")


def open_jsonl(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_html(path: Path) -> str:
    return gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")


def align_fuzzy(a_words: list[str], b_words: list[str]) -> tuple[list[str], list[str]]:
    a_keys = [align_key(t) for t in a_words]
    b_keys = [align_key(t) for t in b_words]
    dump_al: list[str] = []
    html_al: list[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=a_keys, b=b_keys, autojunk=False).get_opcodes():
        if tag == "equal":
            dump_al.extend(a_words[i1:i2])
            html_al.extend(b_words[j1:j2])
    return dump_al, html_al


def align_raw_ftfy_gold(
    raw: list[str], ftfy_w: list[str], html: list[str]
) -> tuple[list[str], list[str], list[str]]:
    a_keys = [align_key(t) for t in ftfy_w]
    b_keys = [align_key(t) for t in html]
    raw_al: list[str] = []
    ftfy_al: list[str] = []
    html_al: list[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=a_keys, b=b_keys, autojunk=False).get_opcodes():
        if tag == "equal":
            raw_al.extend(raw[i1:i2])
            ftfy_al.extend(ftfy_w[i1:i2])
            html_al.extend(html[j1:j2])
    return raw_al, ftfy_al, html_al


def has_upper(text: str) -> bool:
    return any(c.isupper() for c in text)


def score_case(pred: list[str], gold: list[str]) -> tuple[int, int, int, int]:
    ok = n = cased_ok = cased_n = 0
    for p, g in zip(pred, gold):
        if align_key(p) != align_key(g):
            continue
        n += 1
        ok += int(p == g)
        if has_upper(g):
            cased_n += 1
            cased_ok += int(p == g)
    return ok, n, cased_ok, cased_n


def score_pred_text(pred_text: str, gold: list[str]) -> tuple[int, int, int, int]:
    pred_al, gold_al = align_fuzzy(pred_text.split(), gold)
    return score_case(pred_al, gold_al)


def score_chars(dump: list[str], pred: list[str], gold: list[str]) -> tuple[int, int]:
    ok = n = 0
    for d, p, g in zip(dump, pred, gold):
        if not is_encoding_damage(d, g):
            continue
        n += 1
        ok += int(p == g)
    return ok, n


def pick_candidates(by_phase: dict[str, list[dict]], max_align: int) -> list[dict]:
    try_per = max(80, max_align * 2)
    chosen: list[dict] = []
    for phase in ("A", "B", "C"):
        rows = by_phase[phase]
        damaged = [r for r in rows if r["has_q"] or r["has_moj"]]
        clean = [r for r in rows if not (r["has_q"] or r["has_moj"])]
        take = damaged[: try_per // 2] + clean[: try_per - min(len(damaged), try_per // 2)]
        if len(take) < try_per:
            take = (damaged + clean)[:try_per]
        chosen.extend(take)
    return chosen


def build_cache(sample: Path, html_dir: Path, cache: Path, max_align: int, min_coverage: float) -> None:
    from quotebank_prep import html_to_ptb, start_jvm

    start_jvm()
    try_per = max(80, max_align * 2)
    keep_per = try_per * 2
    by_phase: dict[str, list[dict]] = {p: [] for p in LOWER}
    train_docs: list[list[str]] = []
    e_docs: list[list[str]] = []
    moses_path = cache.with_name("moses_truecase.model")
    lex_path = cache.with_name("lexicon_E.json")
    need_moses = not moses_path.exists()
    need_lex = not lex_path.exists()
    print("indexing sample for HTML A–C and D/E train", flush=True)
    with open_jsonl(sample) as handle:
        for i, line in enumerate(handle, 1):
            rec = json.loads(line)
            phase = rec.get("phase") or ""
            if need_moses and phase in TRAIN and len(train_docs) < 40_000:
                text = clean_dump_text(rec.get("content") or "")
                if text:
                    train_docs.append(text.split())
            if need_lex and phase == "E" and len(e_docs) < 40_000:
                raw = (rec.get("content") or "").split()
                if raw:
                    e_docs.append(raw)
            if phase in LOWER and len(by_phase[phase]) < keep_per:
                path = html_gz_path(html_dir, rec["article_id"])
                if path.exists():
                    content = rec.get("content") or ""
                    by_phase[phase].append(
                        {
                            "article_id": rec["article_id"],
                            "phase": phase,
                            "url": rec.get("url") or "",
                            "content": content,
                            "has_q": "?" in content,
                            "has_moj": "Ã" in content or "â" in content,
                        }
                    )
            if i % 100_000 == 0:
                print(
                    f"  scanned={i} train={len(train_docs)} "
                    + " ".join(f"{p}={len(by_phase[p])}" for p in LOWER),
                    flush=True,
                )
            if (
                (not need_moses or len(train_docs) >= 40_000)
                and (not need_lex or len(e_docs) >= 40_000)
                and all(len(by_phase[p]) >= try_per for p in LOWER)
            ):
                print(f"  index early-stop at {i}", flush=True)
                break

    from sacremoses import MosesTruecaser

    moses_path = cache.with_name("moses_truecase.model")
    lex_path = cache.with_name("lexicon.json")
    if train_docs and not moses_path.exists():
        moses = MosesTruecaser()
        moses.train(train_docs, save_to=str(moses_path), possibly_use_first_token=True)
        print(f"saved Moses {moses_path}", flush=True)
    if e_docs and not lex_path.exists():
        lexicon = build_lexicon(e_docs)
        lex_path.write_text(json.dumps(lexicon, ensure_ascii=False), encoding="utf-8")
        print(f"saved E lexicon n={len(lexicon)}", flush=True)
    del train_docs, e_docs

    chosen = pick_candidates(by_phase, max_align)
    quota = max(80, max_align // 3)
    print(f"aligning {len(chosen)} html articles, {quota} per phase", flush=True)
    aligned_by: dict[str, list[dict]] = {p: [] for p in LOWER}
    for i, rec in enumerate(chosen, 1):
        phase = rec["phase"]
        if len(aligned_by[phase]) >= quota:
            if all(len(aligned_by[p]) >= quota or p == "B" and len(aligned_by[p]) >= 80 for p in LOWER):
                break
            continue
        dump_raw = clean_dump_text(rec.get("content") or "", fix_encoding=False).split()
        if not dump_raw:
            continue
        dump_ftfy = ftfy_words(dump_raw)
        html_ptb = html_to_ptb(read_html(html_gz_path(html_dir, rec["article_id"])), rec.get("url") or "")
        html_ann = clean_dump_text(" ".join(html_ptb), fix_encoding=True).split()
        raw_al, ftfy_al, gold = align_raw_ftfy_gold(dump_raw, dump_ftfy, html_ann)
        coverage = len(ftfy_al) / max(1, len(dump_ftfy))
        if coverage < min_coverage or not ftfy_al or not has_upper(" ".join(gold)):
            if i % 100 == 0:
                print(
                    f"  tried={i} "
                    + " ".join(f"{p}={len(aligned_by[p])}" for p in LOWER),
                    flush=True,
                )
            continue
        aligned_by[phase].append(
            {
                "article_id": rec["article_id"],
                "phase": phase,
                "coverage": coverage,
                "dump_raw": raw_al,
                "dump_ftfy": ftfy_al,
                "gold": gold,
            }
        )
        if i % 100 == 0:
            print(
                f"  tried={i} "
                + " ".join(f"{p}={len(aligned_by[p])}" for p in LOWER),
                flush=True,
            )

    aligned = [row for phase in LOWER for row in aligned_by[phase]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wt", encoding="utf-8") as handle:
        for row in aligned:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"wrote {len(aligned)} aligned to {cache} "
        + " ".join(f"{p}={len(aligned_by[p])}" for p in LOWER),
        flush=True,
    )


def load_cache(cache: Path) -> list[dict]:
    opener = gzip.open if str(cache).endswith(".gz") else open
    with opener(cache, "rt", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if len(rows) == 1 and isinstance(rows[0], dict) and "rows" in rows[0]:
        return rows[0]["rows"]
    return rows


class CoreNLP:
    def __init__(self, port: int = 9010):
        self.port = port
        self.proc = None

    def start(self) -> None:
        if not CORENLP_HOME.exists():
            raise FileNotFoundError(CORENLP_HOME)
        self.proc = subprocess.Popen(
            [
                "java",
                "-mx4g",
                "-cp",
                f"{CORENLP_HOME}/*",
                "edu.stanford.nlp.pipeline.StanfordCoreNLPServer",
                "-port",
                str(self.port),
                "-timeout",
                "60000",
                "-quiet",
                "-preload",
                "tokenize,ssplit,truecase",
            ],
            cwd=str(CORENLP_HOME),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        url = f"http://127.0.0.1:{self.port}"
        for _ in range(60):
            try:
                if requests.get(url, timeout=1).status_code < 500:
                    print("CoreNLP server up", flush=True)
                    return
            except Exception:
                time.sleep(1)
        raise RuntimeError("CoreNLP server failed to start")

    def annotate(self, text: str) -> dict:
        props = {
            "annotators": "tokenize,ssplit,truecase",
            "outputFormat": "json",
            "truecase.overwriteText": "true",
            "tokenize.whitespace": "true",
            "ssplit.eolonly": "true",
        }
        resp = requests.post(
            f"http://127.0.0.1:{self.port}/",
            params={"properties": json.dumps(props)},
            data=text.encode("utf-8"),
            timeout=45,
        )
        resp.raise_for_status()
        return resp.json()

    def from_doc(self, data: dict) -> str:
        true_toks = []
        for sent in data.get("sentences") or []:
            for tok in sent.get("tokens") or []:
                true_toks.append(tok.get("truecaseText") or tok.get("word") or "")
        return " ".join(true_toks)

    def close(self) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except Exception:
                self.proc.kill()


def stanza_ner_capitalize(text: str, nlp) -> str:
    doc = nlp(text)
    out = []
    for sent in doc.sentences:
        for tok in sent.tokens:
            word = tok.text
            ner = getattr(tok, "ner", None) or "O"
            if ner != "O" and word:
                word = word[:1].upper() + word[1:]
            out.append(word)
    return " ".join(out)


def pack_case(c: Counter) -> dict:
    return {
        "token_acc": c["ok"] / c["n"] if c["n"] else None,
        "cased_token_acc": c["cased_ok"] / c["cased_n"] if c["cased_n"] else None,
        "n_tokens": c["n"],
        "n_cased": c["cased_n"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="/home/mculjak/datasets/quotegraph_poc/sample.jsonl.gz")
    parser.add_argument("--html-dir", default="/home/mculjak/datasets/quotegraph_poc/html")
    parser.add_argument("--cache", default="/home/mculjak/datasets/quotegraph_poc/align_cache.jsonl.gz")
    parser.add_argument("--out", default="/home/mculjak/datasets/quotegraph_poc/restore_eval.json")
    parser.add_argument("--max-align", type=int, default=1200)
    parser.add_argument("--min-coverage", type=float, default=0.75)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--no-stanza", action="store_true")
    args = parser.parse_args()
    cache = Path(args.cache)

    if args.score_only and not cache.exists():
        sys.exit(f"missing cache {cache}")
    if not args.score_only and (args.rebuild_cache or not cache.exists()):
        build_cache(Path(args.sample), Path(args.html_dir), cache, args.max_align, args.min_coverage)
    if args.cache_only:
        return
    rows = load_cache(cache)
    print(f"loaded cache n={len(rows)}", flush=True)

    lex_path = cache.with_name("lexicon_E.json")
    if not lex_path.exists():
        lex_path = cache.with_name("lexicon.json")
    lexicon = json.loads(lex_path.read_text(encoding="utf-8")) if lex_path.exists() else {}
    lex_len = lexicon_by_length(lexicon)
    print(f"lexicon={len(lexicon)}", flush=True)

    from sacremoses import MosesTruecaser

    moses_path = cache.with_name("moses_truecase.model")
    if moses_path.exists():
        moses = MosesTruecaser(str(moses_path))
        print("loaded Moses", flush=True)
        have_moses = True
    else:
        moses = MosesTruecaser()
        have_moses = False

    truecase_fn = None
    try:
        import truecase

        truecase_fn = truecase.get_true_case
        print("loaded truecase package", flush=True)
    except Exception as exc:
        print(f"truecase package unavailable: {exc}", flush=True)

    stanza_nlp = None
    if not args.no_stanza:
        try:
            import stanza

            stanza_nlp = stanza.Pipeline(
                "en",
                processors="tokenize,ner",
                tokenize_pretokenized=True,
                download_method=None,
            )
            print("loaded stanza NER", flush=True)
        except Exception as exc:
            print(f"stanza unavailable: {exc}", flush=True)

    corenlp = None
    try:
        corenlp = CoreNLP()
        corenlp.start()
    except Exception as exc:
        print(f"CoreNLP unavailable: {exc}", flush=True)
        corenlp = None

    def moses_fn(text: str) -> str:
        return moses.truecase(text, return_str=True)

    def lex_apply(text: str) -> str:
        return apply_lexicon(text, lexicon, lex_len)

    def unigram_fn(text: str) -> str:
        out = []
        for tok in text.split():
            key = "".join(c.lower() for c in tok if c.isascii() and c.isalnum())
            out.append(lexicon.get(key, tok) if key else tok)
        return " ".join(out)

    case_stats: dict[str, Counter] = defaultdict(Counter)
    phase_case: dict[str, dict[str, Counter]] = {p: defaultdict(Counter) for p in LOWER}
    phase_docs: Counter = Counter()
    char_stats: dict[str, Counter] = defaultdict(Counter)
    combo_stats: dict[str, Counter] = defaultdict(Counter)
    n_rows = 0
    n_char_rows = 0
    examples: list[dict] = []

    for i, rec in enumerate(rows, 1):
        gold = rec["gold"]
        dump_raw = rec["dump_raw"]
        dump_ftfy = fix_mojibake(" ".join(dump_raw)).split()
        src = " ".join(dump_ftfy)
        lexed = lex_apply(src)
        phase = rec.get("phase") or ""
        n_rows += 1
        if phase in LOWER:
            phase_docs[phase] += 1

        preds: dict[str, str] = {
            "identity": src,
            "sentence_start": sentence_truecase(src),
            "unigram_de": unigram_fn(src),
        }
        if have_moses:
            try:
                preds["moses_de"] = moses_fn(src)
            except Exception:
                pass
        if truecase_fn:
            try:
                preds["truecase_pypi"] = truecase_fn(src)
            except Exception:
                pass
        if corenlp:
            try:
                preds["stanford_truecase"] = corenlp.from_doc(corenlp.annotate(src))
            except Exception as exc:
                if i < 5:
                    print(f"CoreNLP article fail: {exc}", flush=True)
        if stanza_nlp:
            try:
                preds["stanza_ner_cap"] = stanza_ner_capitalize(src, stanza_nlp)
            except Exception as exc:
                if i < 5:
                    print(f"stanza article fail: {exc}", flush=True)
        top3 = [
            preds[name]
            for name in ("moses_de", "truecase_pypi", "stanford_truecase")
            if name in preds
        ]
        if len(top3) >= 2:
            ens = ensemble_truecase(src, top3)
            preds["ensemble_top3"] = ens
            preds["ensemble_top3_sent"] = sentence_truecase(ens)
        if i == 1:
            print("systems", sorted(preds), flush=True)

        for name, pred in preds.items():
            ok, n, cased_ok, cased_n = score_pred_text(pred, gold)
            case_stats[name]["ok"] += ok
            case_stats[name]["n"] += n
            case_stats[name]["cased_ok"] += cased_ok
            case_stats[name]["cased_n"] += cased_n
            case_stats[name]["docs"] += 1
            if phase in phase_case:
                phase_case[phase][name]["ok"] += ok
                phase_case[phase][name]["n"] += n
                phase_case[phase][name]["cased_ok"] += cased_ok
                phase_case[phase][name]["cased_n"] += cased_n

        char_preds = {
            "raw": dump_raw,
            "ftfy": dump_ftfy,
            "ftfy_lexicon": lex_apply(" ".join(dump_ftfy)).split(),
            "html_project": project_aligned(dump_ftfy, gold),
        }
        if any(is_encoding_damage(d, g) for d, g in zip(dump_raw, gold)):
            n_char_rows += 1
            for name, pred in char_preds.items():
                if len(pred) != len(gold):
                    pred = (pred + [""] * len(gold))[: len(gold)]
                ok, n = score_chars(dump_raw, pred, gold)
                char_stats[name]["ok"] += ok
                char_stats[name]["n"] += n
                if name == "ftfy_lexicon" and len(examples) < 30:
                    for d, p, g in zip(dump_raw, pred, gold):
                        if is_encoding_damage(d, g):
                            examples.append(
                                {
                                    "phase": rec["phase"],
                                    "dump": d,
                                    "pred": p,
                                    "gold": g,
                                    "ok": p == g,
                                }
                            )
                            if len(examples) >= 30:
                                break

        combo_src = {
            "ftfy_lexicon": lexed,
        }
        if have_moses:
            try:
                combo_src["ftfy_lexicon_moses"] = moses_fn(lexed)
            except Exception:
                pass
        if "stanford_truecase" in preds:
            if lexed == src:
                combo_src["ftfy_lexicon_stanford"] = preds["stanford_truecase"]
            else:
                try:
                    combo_src["ftfy_lexicon_stanford"] = corenlp.from_doc(corenlp.annotate(lexed))
                except Exception:
                    pass
        for name, pred in combo_src.items():
            ok, n, cased_ok, cased_n = score_pred_text(pred, gold)
            combo_stats[name]["ok"] += ok
            combo_stats[name]["n"] += n
            combo_stats[name]["cased_ok"] += cased_ok
            combo_stats[name]["cased_n"] += cased_n
            pred_words = pred.split()
            if len(pred_words) == len(gold):
                cok, cn = score_chars(dump_raw, pred_words, gold)
                combo_stats[name]["char_ok"] += cok
                combo_stats[name]["char_n"] += cn
        if i % 50 == 0:
            print(f"  scored {i}/{len(rows)}", flush=True)

    report = {
        "n_aligned": n_rows,
        "n_docs_by_phase": dict(phase_docs),
        "n_with_damaged_chars": n_char_rows,
        "truecasers": {k: pack_case(v) for k, v in case_stats.items()},
        "truecasers_by_phase": {
            p: {k: pack_case(v) for k, v in systems.items()}
            for p, systems in phase_case.items()
        },
        "char_restorers": {
            k: {"acc": v["ok"] / v["n"] if v["n"] else None, "n": v["n"]}
            for k, v in char_stats.items()
        },
        "combined": {
            k: {
                **pack_case(v),
                "char_acc": v["char_ok"] / v["char_n"] if v["char_n"] else None,
                "n_char": v["char_n"],
            }
            for k, v in combo_stats.items()
        },
        "examples": examples,
    }
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("==== truecasers by phase ====", flush=True)
    for phase in ("A", "B", "C"):
        print(f"phase {phase} docs={phase_docs.get(phase, 0)}", flush=True)
        for name, stats in sorted(report["truecasers_by_phase"].get(phase, {}).items()):
            tok = stats["token_acc"]
            cased = stats["cased_token_acc"]
            print(
                f"  {name:22} tok={tok:.4f} cased={cased:.4f} n={stats['n_tokens']} cased_n={stats['n_cased']}"
                if tok is not None and cased is not None
                else f"  {name:22} {stats}",
                flush=True,
            )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if corenlp:
        corenlp.close()


if __name__ == "__main__":
    main()
