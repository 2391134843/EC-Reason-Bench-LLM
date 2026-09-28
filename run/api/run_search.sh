#!/usr/bin/env bash
# M2 - Intelligent Search: agentic ReAct retrieval over the EC-tree cascade, via your API model.
# The model itself decides WHICH evidence tool to call (blast/hmmer/esm_knn/structure/active_site)
# and WHEN to stop, under a per-enzyme search budget. Distinct from the passive "dump all evidence"
# open-book condition. Reports accuracy + search cost (tool calls per enzyme).
#   ./run_search.sh                 # default budget
#   MAX_SEARCH=6 ./run_search.sh    # larger search budget
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

python3 -m ecreason.search \
  --backend "openai:${MODEL}" \
  --max-search "${MAX_SEARCH:-4}" \
  --splits ${SPLITS} \
  --workers "${WORKERS}" \
  --limit "${LIMIT}"
