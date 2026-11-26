#!/usr/bin/env bash
# Cola de experimentos en GPU local. Uso: bash kaggle_runner/local_queue.sh <snapshot_src_dir> "name|args" ...
# Cada experimento escribe en local_runs/<name>/ (OOF, summary, registro.csv, log.txt).
set -u
SNAP="$1"; shift
R="$(cd "$(dirname "$0")/.." && pwd)"
PY="$R/.venv/Scripts/python"
for spec in "$@"; do
  name="${spec%%|*}"; args="${spec#*|}"
  out="$R/local_runs/$name"; mkdir -p "$out"
  if [ -f "$out/DONE" ]; then echo "skip $name"; continue; fi
  echo "$(date +%H:%M) START $name :: $args"
  ( cd "$SNAP" && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PASSWORD_DATA_DIR="$R/data" PASSWORD_OUT_DIR="$out" \
      "$PY" -u run_cv.py $args > "$out/log.txt" 2>&1 ) && touch "$out/DONE"
  echo "$(date +%H:%M) END $name rc=$? :: $(grep -E 'cv_media|CV media|ponderada' "$out/log.txt" | tail -2 | tr '\n' ' ')"
done
echo QUEUE_DONE
