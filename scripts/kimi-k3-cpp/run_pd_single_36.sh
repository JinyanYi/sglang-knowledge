#!/bin/bash
# Single-node PD on 80.5.17.36: Prefill dies 0-7, Decode dies 8-15, Router :8000
# No json-model-override. Edit model config.json yourself before launch (see comments).
# KEEP layers should already be set in $MODEL_PATH/config.json (recommend 24 for 8+8 die).
#
# Usage:
#   bash run_pd_single_36.sh          # start P + D + router
#   bash run_pd_single_36.sh stop     # kill background P/D
#
# Client -> http://80.5.17.36:8000

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_DIR="${SCRIPT_DIR}/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

ACTION="${1:-start}"
# Point to a weight dir whose config.json is already cut (or the original after you edited it).
MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-W4A8}"
PF_IP="80.5.17.36"
MF_STORE_URL="${ASCEND_MF_STORE_URL:-tcp://127.0.0.1:24670}"

PF_PORT=30001
DC_PORT=30002
BOOTSTRAP_PORT=8998
ROUTER_PORT=8000

stop_all() {
    for f in "${PID_DIR}/prefill.pid" "${PID_DIR}/decode.pid" "${PID_DIR}/router.pid"; do
        if [[ -f "$f" ]]; then
            pid="$(cat "$f")"
            if kill -0 "$pid" 2>/dev/null; then
                echo "kill $f -> $pid"
                kill "$pid" 2>/dev/null || true
                sleep 2
                kill -9 "$pid" 2>/dev/null || true
            fi
            rm -f "$f"
        fi
    done
    echo "stopped"
}

if [[ "$ACTION" == "stop" ]]; then
    stop_all
    exit 0
fi

stop_all

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE=1
export SGLANG_NPU_USE_MULTI_STREAM=0

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export HCCL_BUFFSIZE=800
export DEEPEP_NORMAL_LONG_SEQ_ROUND=64
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512
export HCCL_OP_EXPANSION_MODE=AIV
export ASCEND_CUSTOM_OPP_PATH=/home/z30071866/cann9.1.0/cann-9.1.0-beta.3/opp/vendors/custom_transformer
export LD_LIBRARY_PATH=/home/z30071866/cann9.1.0/cann-9.1.0-beta.3/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}

export SGLANG_NPU_PROFILING=0
export PYTHONPATH=/home/y00970600/KimiK3/sglang-kimiK3/python:$PYTHONPATH
export ASCEND_MF_STORE_URL="${MF_STORE_URL}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
unset SGLANG_ENABLE_SPEC_V2 || true
unset SGLANG_RAGGED_VERIFY_MODE || true
unset SGLANG_PP_LAYER_PARTITION || true

mapfile -t LOCAL_IPS < <(hostname -I | tr ' ' '\n' | grep -E '^[0-9.]+$')
echo "LOCAL_IPS=${LOCAL_IPS[*]}"

find_iface_by_ip() {
    local target_ip="$1"
    python3 - "$target_ip" <<'PY'
import fcntl, os, socket, struct, sys
target = sys.argv[1]

def iface_ip(name: str):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        return socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,
            struct.pack("256s", name[:15].encode()),
        )[20:24])
    except OSError:
        return None
    finally:
        s.close()

for name in sorted(os.listdir("/sys/class/net")):
    if name == "lo":
        continue
    if iface_ip(name) == target:
        print(name)
        break
PY
}

MATCH=0
for lip in "${LOCAL_IPS[@]}"; do
    if [[ "$lip" == "$PF_IP" ]]; then
        MATCH=1
        break
    fi
done
if [[ "$MATCH" -ne 1 ]]; then
    echo "ERROR: this script must run on ${PF_IP}, LOCAL_IPS=${LOCAL_IPS[*]}"
    exit 1
fi

IFACE="$(find_iface_by_ip "${PF_IP}")"
if [[ -z "${IFACE}" ]]; then
    echo "ERROR: cannot find NIC for ${PF_IP}"
    exit 1
fi
export HCCL_SOCKET_IFNAME="${IFACE}"
export GLOO_SOCKET_IFNAME="${IFACE}"
echo "Single-node PD on ${PF_IP} iface=${IFACE} MODEL=${MODEL_PATH}"

# Sanity: print layers from config (no override)
python3 - "$MODEL_PATH/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
tc = c.get("text_config") or c
print(f"config num_hidden_layers={tc.get('num_hidden_layers')} heads={tc.get('num_attention_heads')}")
lac = tc.get("linear_attn_config") or {}
print(f"  full_attn_layers={lac.get('full_attn_layers')}")
print(f"  kda_layers count={len(lac.get('kda_layers') or [])}")
PY

TS="$(date +%Y-%m-%d_%H-%M-%S)"
PF_LOG="${LOG_DIR}/single36_prefill_${TS}.log"
DC_LOG="${LOG_DIR}/single36_decode_${TS}.log"
RT_LOG="${LOG_DIR}/single36_router_${TS}.log"

# ---- Prefill: dies 0-7 ----
(
  export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  sglang serve \
    --model-loader-extra-config '{"enable_multithread_load": true}' \
    --dist-init-addr 127.0.0.1:5001 --nnodes 1 --node-rank 0 \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$MODEL_PATH" \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --dtype bfloat16 \
    --tp-size 8 \
    --pp-size 1 \
    --base-gpu-id 0 \
    --disable-radix-cache \
    --mem-fraction-static 0.72 \
    --chunked-prefill-size 4096 \
    --max-running-requests 4 \
    --host 0.0.0.0 \
    --port "${PF_PORT}" \
    --reasoning-parser kimi_k3 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend ascend \
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
    --watchdog-timeout 9000
) >"$PF_LOG" 2>&1 &
echo $! >"${PID_DIR}/prefill.pid"
echo "Prefill pid=$(cat "${PID_DIR}/prefill.pid") log=$PF_LOG"

sleep 5

# ---- Decode: dies 8-15 ----
(
  export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
  export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
  sglang serve \
    --model-loader-extra-config '{"enable_multithread_load": true}' \
    --dist-init-addr 127.0.0.1:5002 --nnodes 1 --node-rank 0 \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$MODEL_PATH" \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --dtype bfloat16 \
    --tp-size 8 \
    --pp-size 1 \
    --base-gpu-id 0 \
    --disable-radix-cache \
    --mem-fraction-static 0.72 \
    --max-running-requests 4 \
    --host 0.0.0.0 \
    --port "${DC_PORT}" \
    --reasoning-parser kimi_k3 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend ascend \
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
    --watchdog-timeout 9000
) >"$DC_LOG" 2>&1 &
echo $! >"${PID_DIR}/decode.pid"
echo "Decode  pid=$(cat "${PID_DIR}/decode.pid") log=$DC_LOG"

echo "Waiting for ports; client hits :${ROUTER_PORT} after ready"
echo "  tail -f $PF_LOG"
echo "  tail -f $DC_LOG"
echo "stop later: bash $0 stop"

for i in $(seq 1 120); do
    if (echo >/dev/tcp/127.0.0.1/${PF_PORT}) 2>/dev/null \
       && (echo >/dev/tcp/127.0.0.1/${DC_PORT}) 2>/dev/null; then
        echo "ports ${PF_PORT}/${DC_PORT} are up"
        break
    fi
    sleep 10
done

python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://127.0.0.1:${PF_PORT}" \
  --decode  "http://127.0.0.1:${DC_PORT}" \
  --host 0.0.0.0 \
  --port "${ROUTER_PORT}" \
  --mini-lb 2>&1 | tee "$RT_LOG"
