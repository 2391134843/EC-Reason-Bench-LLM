#!/usr/bin/env bash
# Shared bootstrap for the API runners: load .env, resolve paths, ensure data exists.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$HERE/../.." && pwd)"     # .../benchmark-main
if [ -f "$HERE/.env" ]; then
  set -a; . "$HERE/.env"; set +a
else
  echo "!! $HERE/.env not found. Copy .env.example -> .env and add your key." >&2
  exit 1
fi
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in run/api/.env}"
export OPENAI_API_KEY OPENAI_BASE_URL
MODEL="${MODEL:-gpt-4o-mini}"
WORKERS="${WORKERS:-8}"
LIMIT="${LIMIT:-0}"
SPLITS="${SPLITS:-30 30-50 price promiscuous}"
CHANNELS="${CHANNELS:-all}"
MODE="${MODE:-cascade}"

cd "$BENCH_DIR"
# build the benchmark data if missing (no API/GPU needed for these)
[ -f out/items/30.jsonl ]    || { echo "[bootstrap] building items...";    python3 -m ecreason.build_items; }
[ -f out/evidence/30.jsonl ] || { echo "[bootstrap] building evidence..."; python3 -m ecreason.evidence --topk 5; }
echo "[run] model=$MODEL base_url=${OPENAI_BASE_URL:-<openai>} workers=$WORKERS limit=$LIMIT splits=[$SPLITS]"
