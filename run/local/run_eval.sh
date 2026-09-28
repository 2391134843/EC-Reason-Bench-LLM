#!/usr/bin/env bash
# Single-condition eval against a LOCAL vLLM server (start ./serve_vllm.sh first).
#   CHANNELS=none ./run_eval.sh                # closed-book
#   CHANNELS="blast hmmer" ./run_eval.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

python3 -m ecreason.run_eval \
  --backend "openai:${MODEL}" \
  --channels ${CHANNELS} \
  --mode "${MODE}" \
  --splits ${SPLITS} \
  --workers "${WORKERS}" \
  --limit "${LIMIT}"
