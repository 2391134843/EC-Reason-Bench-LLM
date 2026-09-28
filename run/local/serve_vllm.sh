#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for a local open model.
# Run this in one terminal; then run ./run_ablation.sh in another.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/.env" ] && { set -a; . "$HERE/.env"; set +a; }

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "[vllm] serving $MODEL on :$PORT (TP=$TP, HF_ENDPOINT=$HF_ENDPOINT)"
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL"
