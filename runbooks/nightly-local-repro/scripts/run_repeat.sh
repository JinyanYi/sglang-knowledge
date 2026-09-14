#!/usr/bin/env bash
# Repeat a local NPU performance repro inside an existing container.
# Usage:
#   CONTAINER=sglang-nightly-repro TEST=/path/to/repro.py RUNS=3 \
#     bash run_repeat.sh
set -euo pipefail

CONTAINER="${CONTAINER:-sglang-nightly-repro}"
TEST="${TEST:?set TEST=/path/to/repro.py}"
RUNS="${RUNS:-3}"
SGLANG_PYTHON="${SGLANG_PYTHON:-$HOME/community/sglang/python}"
OUTDIR="${OUTDIR:-$HOME/nightly_repro/runs_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.txt"
{
  echo "outdir=$OUTDIR"
  echo "container=$CONTAINER"
  echo "test=$TEST"
  echo "image=$(docker inspect -f '{{.Image}}' "$CONTAINER" 2>/dev/null || echo unknown)"
  echo "started=$(date -Iseconds)"
} | tee "$SUMMARY"

extract() {
  local log=$1
  echo "---- $(basename "$log") ----"
  grep -E 'max_running_requests was reduced|Output token throughput|Peak output|Total token|Concurrency|Accept length|Mean TTFT|Median TTFT|P90 TTFT|Mean TPOT|Median TPOT|P90 TPOT|P99 TPOT|Mean E2E|Median E2E|AssertionError|FAILED \(|^OK$|Ran [0-9]+ test' "$log" || true
}

for i in $(seq 1 "$RUNS"); do
  LOG="$OUTDIR/run${i}.log"
  echo "========== RUN $i START $(date -Iseconds) ==========" | tee -a "$SUMMARY"
  docker exec "$CONTAINER" bash -lc "
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    export PYTHONPATH=${SGLANG_PYTHON}:\${PYTHONPATH}
    cd \"\$HOME\"
    python3 -u \"$TEST\"
  " >"$LOG" 2>&1 || true
  echo "========== RUN $i END $(date -Iseconds) ==========" | tee -a "$SUMMARY"
  extract "$LOG" | tee -a "$SUMMARY"
done

echo "finished=$(date -Iseconds)" | tee -a "$SUMMARY"
echo "summary=$SUMMARY"
