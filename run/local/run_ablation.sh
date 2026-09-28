#!/usr/bin/env bash
# Full ablation against a LOCAL vLLM server (start ./serve_vllm.sh first).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

python3 -m ecreason.ablation \
  --backend "openai:${MODEL}" \
  --mode "${MODE}" \
  --splits ${SPLITS} \
  --workers "${WORKERS}" \
  --limit "${LIMIT}"

echo "Done. See out/results/ablation_openai_${MODEL//[:\/]/_}.md"
