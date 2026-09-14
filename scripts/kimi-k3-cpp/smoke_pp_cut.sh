#!/usr/bin/env bash
# Single-node / small PD smoke for Kimi-K3 cut layers + optional PP.
# Does not require all 4 machines.
#
# Examples:
#   # 1) Single-node PP only (framework path)
#   MODE=pp_only KEEP_LAYERS=24 TP=8 PP=2 bash smoke_pp_cut.sh
#
#   # 2) Minimal PD (run on two machines separately)
#   MODE=pd_prefill KEEP_LAYERS=12 TP=16 PP=1 bash smoke_pp_cut.sh   # node A
#   MODE=pd_decode  KEEP_LAYERS=12 TP=16 bash smoke_pp_cut.sh         # node B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-pp_only}"
KEEP_LAYERS="${KEEP_LAYERS:-24}"
TP="${TP:-8}"
PP="${PP:-2}"
MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}"
CODE_ROOT="${CODE_ROOT:-/home/y00970600/KimiK3/sglang-kimiK3}"
OVERRIDE_JSON="${OVERRIDE_JSON:-${SCRIPT_DIR}/overrides/k3_cut_l${KEEP_LAYERS}.json}"
HTTP_PORT="${HTTP_PORT:-30001}"
DIST_INIT="${DIST_INIT:-127.0.0.1:5001}"
HCCL_IFNAME="${HCCL_IFNAME:-}"

export PYTHONPATH="${CODE_ROOT}/python:${PYTHONPATH:-}"
export SGLANG_NPU_USE_MULTI_STREAM=0
export SGLANG_NPU_PROFILING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-800}"

if [[ -n "${HCCL_IFNAME}" ]]; then
  export HCCL_SOCKET_IFNAME="${HCCL_IFNAME}"
  export GLOO_SOCKET_IFNAME="${HCCL_IFNAME}"
fi

python3 "${SCRIPT_DIR}/prepare_k3_cut_override.py" \
  --config "${MODEL_PATH}/config.json" \
  --keep "${KEEP_LAYERS}" \
  -o "${OVERRIDE_JSON}"

OVERRIDE_ARGS="$(cat "${OVERRIDE_JSON}")"
HALF=$((KEEP_LAYERS / 2))

BASE=(
  --model-path "${MODEL_PATH}"
  --tokenizer-path "${MODEL_PATH}"
  --trust-remote-code
  --device npu
  --dtype bfloat16
  --quantization modelslim
  --attention-backend ascend
  --host 0.0.0.0
  --port "${HTTP_PORT}"
  --mem-fraction-static 0.72
  --max-running-requests 4
  --watchdog-timeout 9000
  --moe-a2a-backend deepep
  --deepep-mode auto
  --json-model-override-args "${OVERRIDE_ARGS}"
)

case "${MODE}" in
  pp_only)
    if [[ "${KEEP_LAYERS}" -lt 24 && "${PP}" -gt 1 ]]; then
      echo "[smoke] WARN: KEEP<24 with PP>1 splits mid attn_res block; forcing PP=1" >&2
      PP=1
    fi
    if [[ "${PP}" -gt 1 ]]; then
      export SGLANG_PP_LAYER_PARTITION="${SGLANG_PP_LAYER_PARTITION:-${HALF},${HALF}}"
    else
      unset SGLANG_PP_LAYER_PARTITION || true
    fi
    EXTRA=(
      --nnodes 1
      --node-rank 0
      --dist-init-addr "${DIST_INIT}"
      --tp-size "${TP}"
      --pp-size "${PP}"
      --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-2048}"
    )
    ;;
  pd_prefill)
    if [[ "${KEEP_LAYERS}" -lt 24 && "${PP}" -gt 1 ]]; then
      echo "[smoke] WARN: KEEP<24 with PP>1 splits mid attn_res block; forcing PP=1" >&2
      PP=1
    fi
    if [[ "${PP}" -gt 1 ]]; then
      export SGLANG_PP_LAYER_PARTITION="${SGLANG_PP_LAYER_PARTITION:-${HALF},${HALF}}"
    else
      unset SGLANG_PP_LAYER_PARTITION || true
    fi
    EXTRA=(
      --disaggregation-mode prefill
      --disaggregation-bootstrap-port "${BOOTSTRAP_PORT:-8998}"
      --disaggregation-transfer-backend "${TRANSFER_BACKEND:-mooncake}"
      --nnodes 1
      --node-rank 0
      --dist-init-addr "${DIST_INIT}"
      --tp-size "${TP}"
      --pp-size "${PP}"
      --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-2048}"
    )
    ;;
  pd_decode)
    unset SGLANG_PP_LAYER_PARTITION || true
    EXTRA=(
      --disaggregation-mode decode
      --disaggregation-bootstrap-port "${BOOTSTRAP_PORT:-8998}"
      --disaggregation-transfer-backend "${TRANSFER_BACKEND:-mooncake}"
      --nnodes 1
      --node-rank 0
      --dist-init-addr "${DIST_INIT}"
      --tp-size "${TP}"
      --pp-size 1
    )
    ;;
  *)
    echo "MODE must be pp_only|pd_prefill|pd_decode" >&2
    exit 1
    ;;
esac

echo "[smoke] MODE=${MODE} KEEP=${KEEP_LAYERS} TP=${TP} PP=${PP}"
python -m sglang.launch_server "${BASE[@]}" "${EXTRA[@]}"
