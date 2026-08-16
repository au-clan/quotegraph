#!/bin/bash
# Sample 1M dump articles, scrape HTML, evaluate Moses truecaser on A–C.
set -euo pipefail
ROOT="${ROOT:-/home/mculjak/git/quotegraph}"
DATA="${DATA:-/home/mculjak/datasets/quotegraph_poc}"
export PYTHONPATH="$ROOT"
export QUOTEGRAPH_JARS="$DATA/jars"
mkdir -p "$DATA"
python3 -m pip install --user -q ftfy sacremoses JPype1
python3 "$ROOT/scripts/sample_1m.py" --output "$DATA/sample.jsonl.gz" --n 1000000
python3 "$ROOT/scripts/scrape_html.py" --input "$DATA/sample.jsonl.gz" --out "$DATA" --threads 40
python3 "$ROOT/scripts/eval_truecaser.py" \
  --sample "$DATA/sample.jsonl.gz" \
  --html-dir "$DATA/html" \
  --out "$DATA/truecase_eval.json"
