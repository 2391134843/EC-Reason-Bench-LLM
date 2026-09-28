#!/usr/bin/env bash
# M2 - Intelligent Search: agentic ReAct retrieval over the EC-tree cascade, against a LOCAL vLLM
# server (start ./serve_vllm.sh first). The model decides WHICH evidence tool to call and WHEN to
# stop, under a per-enzyme search budget. Reports accuracy + search cost (tool calls per enzyme).
#   ./run_search.sh                 # default budget
#   MAX_SEARCH=6 ./run_search.sh    # larger search budget
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

python3 -m ecreason.search \
  --backend "openai:${MODEL}" \
  --max-search "${MAX_SEARCH:-4}" \
  --splits ${SPLITS} \
  --workers "${WORKERS}" \
  --limit "${LIMIT}"
