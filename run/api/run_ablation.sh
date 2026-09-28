#!/usr/bin/env bash
# Full ablation (blind -> closed-book -> +each channel -> open-book) via your API model.
# Usage: cp .env.example .env && edit .env && ./run_ablation.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

python3 -m ecreason.ablation \
  --backend "openai:${MODEL}" \
  --mode "${MODE}" \
  --splits ${SPLITS} \
  --workers "${WORKERS}" \
  --limit "${LIMIT}"

echo "Done. See out/results/ablation_openai_${MODEL//[:\/]/_}.md"
