#!/usr/bin/env bash
# Head-to-head: zero-shot direct prediction vs. M1--M4 against a local vLLM
# server (start ./serve_vllm.sh first).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

python3 -m ecreason.compare \
  --backend "openai:${MODEL}" \
  --splits ${SPLITS} \
  --workers "${WORKERS}" \
  --limit "${LIMIT}" \
  --with-sc \
  --sc-k "${K:-3}"

echo "Done. See out/results/compare_openai_${MODEL//[:\/]/_}.md"
