#!/usr/bin/env bash
# Single-condition eval via your API model (channels/mode from .env or args).
# Usage: ./run_eval.sh                 # uses CHANNELS/MODE from .env
#        CHANNELS="blast hmmer" ./run_eval.sh
#        CHANNELS=none ./run_eval.sh    # closed-book
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

python3 -m ecreason.run_eval \
  --backend "openai:${MODEL}" \
  --channels ${CHANNELS} \
  --mode "${MODE}" \
  --splits ${SPLITS} \
  --workers "${WORKERS}" \
  --limit "${LIMIT}"
