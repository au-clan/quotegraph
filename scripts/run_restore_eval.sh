#!/bin/bash
# Build aligned A–C cache and compare truecasers / character restorers.
set -euo pipefail
ROOT="${ROOT:-/home/mculjak/git/quotegraph}"
DATA="${DATA:-/home/mculjak/datasets/quotegraph_poc}"
export PYTHONPATH="$ROOT"
export QUOTEGRAPH_JARS="${QUOTEGRAPH_JARS:-$DATA/jars}"
export CORENLP_HOME="${CORENLP_HOME:-/home/mculjak/.cache/stanza/1.12.0/corenlp}"
MAX_ALIGN="${MAX_ALIGN:-1200}"

python3 -m pip install --user -q truecase nltk requests || true
python3 -m pip install --user stanza >"$DATA/stanza_install.log" 2>&1 &
STANZA_PID=$!

python3 "$ROOT/scripts/eval_restore.py" \
  --sample "$DATA/sample.jsonl.gz" \
  --html-dir "$DATA/html" \
  --cache "$DATA/align_cache.jsonl.gz" \
  --out "$DATA/restore_eval.json" \
  --max-align "$MAX_ALIGN" \
  --min-coverage 0.75 \
  --rebuild-cache \
  --cache-only

wait "$STANZA_PID" || echo "stanza install failed; scoring without it"

python3 "$ROOT/scripts/eval_restore.py" \
  --sample "$DATA/sample.jsonl.gz" \
  --html-dir "$DATA/html" \
  --cache "$DATA/align_cache.jsonl.gz" \
  --out "$DATA/restore_eval.json" \
  --max-align "$MAX_ALIGN" \
  --min-coverage 0.75 \
  --score-only
