#!/bin/bash
# Single-node PD on 80.5.17.34: Prefill dies 0-7, Decode dies 8-15, Router :8000
# KEEP=24 (colleague: ~24 layers on 8 die). No DSPARK. Codebase unchanged.
#
# Usage:
#   bash run_pd_single_34.sh          # start P + D + router (foreground waits on router)
#   bash run_pd_single_34.sh stop     # kill background P/D
#
# Client -> http://80.5.17.34:8000  (router is still required for PD)

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_DIR="${SCRIPT_DIR}/pids"
mkdir -p "$LOG_DIR" "$PID_DIR" "${SCRIPT_DIR}/overrides"

ACTION="${1:-start}"
MODEL_PATH=/home/weights/Kimi-K3-W4A8
KEEP_LAYERS="${KEEP_LAYERS:-24}"
PF_IP="80.5.17.34"
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

# fresh start
stop_all

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE=1
export SGLANG_NPU_USE_MULTI_STREAM=0

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
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
echo "Single-node PD on ${PF_IP} iface=${IFACE} MF_STORE=${ASCEND_MF_STORE_URL} KEEP=${KEEP_LAYERS}"

OVERRIDE_JSON="${SCRIPT_DIR}/overrides/k3_cut_l${KEEP_LAYERS}.json"
python3 - "$MODEL_PATH/config.json" "$KEEP_LAYERS" "$OVERRIDE_JSON" <<'PY'
import json, sys
from pathlib import Path
cfg = json.load(open(sys.argv[1]))
tc = cfg.get("text_config") or cfg
keep = int(sys.argv[2])
n = int(tc["num_hidden_layers"])
if keep < 1 or keep > n:
    raise SystemExit(f"KEEP={keep} invalid, full layers={n}")
# Must copy FULL text_config: config.update() replaces text_config wholesale.
tc_out = dict(tc)
tc_out["num_hidden_layers"] = keep
lac = dict(tc.get("linear_attn_config") or {})
lac["kda_layers"] = [i for i in lac.get("kda_layers", []) if i <= keep]
lac["full_attn_layers"] = [i for i in lac.get("full_attn_layers", []) if i <= keep]
tc_out["linear_attn_config"] = lac
out = {"text_config": tc_out}
Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[3]).write_text(json.dumps(out))
print(f"override -> {sys.argv[3]}  layers {n}->{keep}  has_heads={tc_out.get('num_attention_heads')}")
PY
OVERRIDE_ARGS="$(cat "$OVERRIDE_JSON")"

TS="$(date +%Y-%m-%d_%H-%M-%S)"
PF_LOG="${LOG_DIR}/single34_prefill_keep${KEEP_LAYERS}_${TS}.log"
DC_LOG="${LOG_DIR}/single34_decode_keep${KEEP_LAYERS}_${TS}.log"
RT_LOG="${LOG_DIR}/single34_router_${TS}.log"

# ---- Prefill: dies 0-7 ----
(
  export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
  # Prefer visible-device split; if your image ignores this, --base-gpu-id below still applies.
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
    --json-model-override-args "${OVERRIDE_ARGS}" \
    --watchdog-timeout 9000
) >"$PF_LOG" 2>&1 &
echo $! >"${PID_DIR}/prefill.pid"
echo "Prefill pid=$(cat "${PID_DIR}/prefill.pid") log=$PF_LOG  (8 die, port ${PF_PORT})"

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
    --json-model-override-args "${OVERRIDE_ARGS}" \
    --watchdog-timeout 9000
) >"$DC_LOG" 2>&1 &
echo $! >"${PID_DIR}/decode.pid"
echo "Decode  pid=$(cat "${PID_DIR}/decode.pid") log=$DC_LOG  (8 die, port ${DC_PORT})"

echo "Waiting for Prefill/Decode to become ready (check logs)..."
echo "  tail -f $PF_LOG"
echo "  tail -f $DC_LOG"
echo "After both show ready, starting router on :${ROUTER_PORT}"
echo "Press Ctrl+C later then: bash $0 stop"

# Optional: wait until ports accept connections (best-effort)
for i in $(seq 1 120); do
    if (echo >/dev/tcp/127.0.0.1/${PF_PORT}) 2>/dev/null \
       && (echo >/dev/tcp/127.0.0.1/${DC_PORT}) 2>/dev/null; then
        echo "ports ${PF_PORT}/${DC_PORT} are up"
        break
    fi
    sleep 10
done

# ---- Router (REQUIRED for PD client path) ----
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://127.0.0.1:${PF_PORT}" \
  --decode  "http://127.0.0.1:${DC_PORT}" \
  --host 0.0.0.0 \
  --port "${ROUTER_PORT}" \
  --mini-lb 2>&1 | tee "$RT_LOG"
