#!/usr/bin/env bash
# Kimi-K3 PD + Chunked-PP bring-up (cut layers). Do NOT replace multistream.sh.
#
# Usage (on each node, inside container with host network):
#   ROLE=prefill NODE_RANK=0 KEEP_LAYERS=24 bash cpp_pd.sh
#   ROLE=prefill NODE_RANK=1 KEEP_LAYERS=24 bash cpp_pd.sh
#   ROLE=decode  NODE_RANK=0 KEEP_LAYERS=24 bash cpp_pd.sh
#   ROLE=decode  NODE_RANK=1 KEEP_LAYERS=24 bash cpp_pd.sh
#
# Default topology (4x A3):
#   Prefill: 2 nodes, tp=16, pp=2 (KEEP must be multiple of 24 for clean block split;
#            default KEEP=24 -> SGLANG_PP_LAYER_PARTITION=12,12)
#   Decode:  2 nodes, tp=32, pp=1
#   Spec/DSPARK off; multi-stream off; fixed chunked-prefill on Prefill.
#   For KEEP=12 PD-only smoke, set PP_SIZE=1 (or use smoke_pp_cut.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLE="${ROLE:?set ROLE=prefill|decode}"
NODE_RANK="${NODE_RANK:?set NODE_RANK=0|1}"
KEEP_LAYERS="${KEEP_LAYERS:-24}"

MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}"
CODE_ROOT="${CODE_ROOT:-/home/y00970600/KimiK3/sglang-kimiK3}"
OVERRIDE_JSON="${OVERRIDE_JSON:-${SCRIPT_DIR}/overrides/k3_cut_l${KEEP_LAYERS}.json}"

# Cluster IPs (override via env if needed)
PF_HOSTS="${PF_HOSTS:-192.168.25.209,192.168.25.212}"
DC_HOSTS="${DC_HOSTS:-192.168.25.216,192.168.25.217}"
PF_DIST="${PF_DIST:-192.168.25.209:5001}"
DC_DIST="${DC_DIST:-192.168.25.216:5002}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
HTTP_PORT="${HTTP_PORT:-30001}"

# NIC: set explicitly or leave empty to skip
HCCL_IFNAME="${HCCL_IFNAME:-enp196s0f0}"

export PYTHONPATH="${CODE_ROOT}/python:${PYTHONPATH:-}"
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE=1
export SGLANG_NPU_USE_MULTI_STREAM=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1200}"
export DEEPEP_NORMAL_LONG_SEQ_ROUND=64
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
# profiling off by default for bring-up
export SGLANG_NPU_PROFILING="${SGLANG_NPU_PROFILING:-0}"

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

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --tokenizer-path "${MODEL_PATH}"
  --trust-remote-code
  --device npu
  --dtype bfloat16
  --quantization modelslim
  --attention-backend ascend
  --host 0.0.0.0
  --port "${HTTP_PORT}"
  --mem-fraction-static "${MEM_FRACTION:-0.75}"
  --max-running-requests "${MAX_RUNNING:-16}"
  --watchdog-timeout 9000
  --reasoning-parser kimi_k3
  --moe-a2a-backend deepep
  --deepep-mode auto
  --json-model-override-args "${OVERRIDE_ARGS}"
  --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}"
  --disaggregation-transfer-backend "${TRANSFER_BACKEND:-mooncake}"
)

if [[ "${ROLE}" == "prefill" ]]; then
  # Prefer cutting on attn_res_block_size=12 boundaries.
  PP="${PP_SIZE:-2}"
  if [[ "${KEEP_LAYERS}" -lt 24 && "${PP}" -gt 1 ]]; then
    echo "[cpp_pd] WARN: KEEP_LAYERS=${KEEP_LAYERS}<24 with pp=${PP} splits mid attn_res block; forcing PP=1" >&2
    PP=1
  fi
  if [[ "${PP}" -gt 1 ]]; then
    export SGLANG_PP_LAYER_PARTITION="${SGLANG_PP_LAYER_PARTITION:-${HALF},${HALF}}"
  else
    unset SGLANG_PP_LAYER_PARTITION || true
  fi
  NNODES=2
  TP=16
  DP=1
  DIST_ADDR="${PF_DIST}"
  CHUNK="${CHUNKED_PREFILL_SIZE:-4096}"
  EXTRA=(
    --disaggregation-mode prefill
    --nnodes "${NNODES}"
    --node-rank "${NODE_RANK}"
    --dist-init-addr "${DIST_ADDR}"
    --tp-size "${TP}"
    --pp-size "${PP}"
    --dp-size "${DP}"
    --chunked-prefill-size "${CHUNK}"
  )
  # Optional second step: export ENABLE_DYNAMIC_CHUNKING=1
  if [[ "${ENABLE_DYNAMIC_CHUNKING:-0}" == "1" ]]; then
    export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR="${SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR:-0.75}"
    EXTRA+=(--enable-dynamic-chunking)
    EXTRA+=(--chunked-prefill-size "${DYNAMIC_INIT_CHUNK:-12288}")
  fi
elif [[ "${ROLE}" == "decode" ]]; then
  unset SGLANG_PP_LAYER_PARTITION || true
  NNODES=2
  TP=32
  PP=1
  DP=1
  DIST_ADDR="${DC_DIST}"
  EXTRA=(
    --disaggregation-mode decode
    --nnodes "${NNODES}"
    --node-rank "${NODE_RANK}"
    --dist-init-addr "${DIST_ADDR}"
    --tp-size "${TP}"
    --pp-size "${PP}"
    --dp-size "${DP}"
  )
else
  echo "ROLE must be prefill or decode" >&2
  exit 1
fi

echo "[cpp_pd] ROLE=${ROLE} NODE_RANK=${NODE_RANK} KEEP=${KEEP_LAYERS}"
echo "[cpp_pd] tp=${TP} pp=${PP} dist=${DIST_ADDR} partition=${SGLANG_PP_LAYER_PARTITION:-none}"
echo "[cpp_pd] override=${OVERRIDE_JSON}"
echo "[cpp_pd] NOTE: DSPARK disabled for first CPP bring-up"

python -m sglang.launch_server "${COMMON_ARGS[@]}" "${EXTRA[@]}"
