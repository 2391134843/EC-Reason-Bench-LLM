#!/usr/bin/env bash
# M3 - Mechanism-grounded protein chain-of-thought (closed-book by default), against a
# LOCAL vLLM server (start ./serve_vllm.sh first).
# Distinct from the cascade (no EC-tree walk) and from retrieval (no tools): a single-pass CoT
# structured along the EC definition (reaction class -> bond/group -> cofactor -> substrate),
# grounded in intrinsic sequence motifs.
#   ./run_mechcot.sh                 # closed-book (headline)
#   MECHCOT_CHANNELS=all ./run_mechcot.sh  # optional: inject retrieval evidence
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

python3 -m ecreason.mechanism_cot \
  --backend "openai:${MODEL}" \
  --channels ${MECHCOT_CHANNELS:-none} \
  --splits ${SPLITS} \
  --workers "${WORKERS}" \
  --limit "${LIMIT}"
