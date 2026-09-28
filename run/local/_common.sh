#!/usr/bin/env bash
# Shared bootstrap for local runners: load .env, point the OpenAI client at the local
# vLLM server, ensure benchmark data exists.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$HERE/../.." && pwd)"     # .../benchmark-main
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-16}"
LIMIT="${LIMIT:-0}"
SPLITS="${SPLITS:-30 30-50 price promiscuous}"
CHANNELS="${CHANNELS:-all}"
MODE="${MODE:-cascade}"

# talk to the local vLLM server through the OpenAI-compatible backend
export OPENAI_BASE_URL="http://localhost:${PORT}/v1"
export OPENAI_API_KEY=EMPTY

cd "$BENCH_DIR"
[ -f out/items/30.jsonl ]    || { echo "[bootstrap] building items...";    python3 -m ecreason.build_items; }
[ -f out/evidence/30.jsonl ] || { echo "[bootstrap] building evidence..."; python3 -m ecreason.evidence --topk 5; }

# wait for the server to be reachable
echo "[run] waiting for vLLM at $OPENAI_BASE_URL ..."
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then echo "[run] server up."; break; fi
  sleep 3
  if [ "$i" = 60 ]; then echo "!! server not reachable; start ./serve_vllm.sh first." >&2; exit 1; fi
done
echo "[run] model=$MODEL workers=$WORKERS limit=$LIMIT splits=[$SPLITS]"
