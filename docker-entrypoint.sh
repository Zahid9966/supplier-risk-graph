#!/bin/sh
# Build the data products on first start (simulate -> bronze -> silver -> gold -> evaluate -> score), then serve.
set -e
if [ ! -f "$SRG_DATA/gold/scores.parquet" ]; then
  echo "[entrypoint] no gold artifacts yet - running the full pipeline (a few minutes)"
  python -m srg.cli run --fresh
fi
exec python -m srg.cli serve --host 0.0.0.0 --port 8000
