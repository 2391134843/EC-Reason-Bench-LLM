#!/usr/bin/env bash
# M4 - Self-Consistency CoT: sample K independent reasoning chains and prefix-vote, via your API
# model. Wraps either base reasoner (BASE=cascade wraps M1, BASE=mechcot wraps M3). Costs ~K x
# the base method; commits the deepest EC prefix the chains agree on and abstains where they don't.
#   ./run_sc.sh                          # K=3 over the cascade (M1)
#   BASE=mechcot K=7 ./run_sc.sh         # self-consistency over the mechanism CoT (M3)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

python3 -m ecreason.self_consistency \
  --backend "openai:${MODEL}" \
  --base "${BASE:-cascade}" \
  --k "${K:-3}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --channels ${SC_CHANNELS:-none} \
  --splits ${SPLITS} \
  --workers "${WORKERS}" \
  --limit "${LIMIT}"
