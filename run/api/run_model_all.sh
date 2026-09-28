#!/usr/bin/env bash
# Run ALL 9 EC-Reason-Bench conditions for ONE model on the gateway, in background, logged.
# Generic / reusable across models. Tags are prefixed by $TAG so different models don't clobber.
#
# Usage:
#   MODEL_OVERRIDE=gpt-5.5 TAG=g55 nohup setsid bash run/api/run_model_all.sh >out/results/logs/nohup_g55.out 2>&1 3>&- 4>&- </dev/null &
#
# Watch:
#   tail -n 40 out/results/logs/run_$TAG.log
#   tail -f "$(ls -t out/results/logs/${TAG}_*.log | head -1)"
#   ls out/results/metrics_${TAG}_*.json | wc -l        # of 18 passes (9 conditions x big/small)
#   python3 -m ecreason.summarize --prefix ${TAG}_ --out summary_$TAG
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$HERE/../.." && pwd)"
cd "$BENCH_DIR"
set -a; . "${ENV_FILE:-$HERE/.env}"; set +a   # ENV_FILE may select another local configuration.
export OPENAI_API_KEY OPENAI_BASE_URL
export ECREASON_PROGRESS=1
MODEL="${MODEL_OVERRIDE:-${MODEL:-deepseek-v4-pro}}"   # override on CLI, ignore .env's MODEL
TAG="${TAG:-model}"                                    # tag prefix, e.g. g55
WORKERS="${WORKERS_OVERRIDE:-64}"
BIG_SPLITS="30 30-50"; BIG_LIMIT="${BIG_LIMIT:-0}"
SMALL_SPLITS="price promiscuous"
K="${K:-3}"; MAXSEARCH="${MAXSEARCH:-4}"

LOGDIR="out/results/logs"; mkdir -p "$LOGDIR"
MASTER="$LOGDIR/run_${TAG}.log"
echo "===== RUN START $(date '+%F %T')  model=$MODEL tag=$TAG workers=$WORKERS k=$K =====" | tee -a "$MASTER"

run_pass () {  # $1=cond  $2=pass  $3..=module + extra args
  local cond="$1"; shift
  local pass="$1"; shift
  local tag="${TAG}_${cond}_${pass}"
  local splits limit log rc t0 dt head
  if [ "$pass" = big ]; then splits="$BIG_SPLITS"; limit="$BIG_LIMIT"; else splits="$SMALL_SPLITS"; limit=0; fi
  log="$LOGDIR/${tag}.log"
  if [ -f "out/results/metrics_${tag}.json" ]; then
    echo "[$(date '+%T')] SKIP  $tag (already done)" | tee -a "$MASTER"; return
  fi
  echo "[$(date '+%T')] START $tag  splits=[$splits] limit=$limit" | tee -a "$MASTER"
  t0=$(date +%s)
  python3 -m "$@" --backend "openai:${MODEL}" --workers "$WORKERS" \
      --splits $splits --limit "$limit" --tag "$tag" >"$log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then
    head=$(grep -oE '"L4": [0-9.]+' "$log" | head -1)
    echo "[$(date '+%T')] DONE  $tag in ${dt}s  ($head)" | tee -a "$MASTER"
  else
    echo "[$(date '+%T')] FAIL  $tag rc=$rc in ${dt}s (see $log)" | tee -a "$MASTER"
  fi
}
run_cond () { run_pass "$1" big "${@:2}"; run_pass "$1" small "${@:2}"; }

# order: cheap + headline first; expensive M4 last
run_cond b0_closed  ecreason.baselines        --channels none
run_cond m1         ecreason.run_eval         --mode cascade --channels none
run_cond m3         ecreason.mechanism_cot    --channels none
run_cond m2         ecreason.search           --max-search "$MAXSEARCH"
run_cond casc_open  ecreason.run_eval         --mode cascade --channels all
run_cond b0_open    ecreason.baselines        --channels all
run_cond m3_open    ecreason.mechanism_cot    --channels all
run_cond m4         ecreason.self_consistency --base cascade --k "$K" --channels none
run_cond m4_open    ecreason.self_consistency --base cascade --k "$K" --channels all

echo "[$(date '+%T')] SUMMARIZE" | tee -a "$MASTER"
python3 -m ecreason.summarize --prefix "${TAG}_" --out "summary_${TAG}" --title "$MODEL" >>"$MASTER" 2>&1
echo "===== RUN DONE $(date '+%F %T') =====" | tee -a "$MASTER"
