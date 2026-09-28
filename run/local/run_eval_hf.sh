#!/usr/bin/env bash
# Direct HuggingFace in-process eval (no server). Simplest path; single process / GPU.
# Slower than the vLLM server route; good for a quick check on one GPU.
#   CHANNELS=all LIMIT=30 ./run_eval_hf.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$HERE/../.." && pwd)"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LIMIT="${LIMIT:-30}"
SPLITS="${SPLITS:-30 30-50 price promiscuous}"
CHANNELS="${CHANNELS:-all}"
MODE="${MODE:-cascade}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$BENCH_DIR"
[ -f out/items/30.jsonl ]    || python3 -m ecreason.build_items
[ -f out/evidence/30.jsonl ] || python3 -m ecreason.evidence --topk 5
python3 -m ecreason.run_eval \
  --backend "hf:${MODEL}" \
  --channels ${CHANNELS} \
  --mode "${MODE}" \
  --splits ${SPLITS} \
  --workers 1 \
  --limit "${LIMIT}"
