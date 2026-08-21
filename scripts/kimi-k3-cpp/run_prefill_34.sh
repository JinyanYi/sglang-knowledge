#!/bin/bash
# Prefill @ 80.5.17.34  — copy this file to the Prefill machine and run.
# Pair with the Decode script on 80.5.17.38. Same KEEP_LAYERS on both.
#
# Default: KEEP=48, tp=16, pp=1  (PD smoke; same TP as decode)
# CPP:     ENABLE_PP=1 -> tp=8, pp=2, PARTITION=24,24
#
# Codebase: no change needed (0728_dspark).
# Weights: do NOT edit /home/weights/.../config.json; override is generated below.

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR" "${SCRIPT_DIR}/overrides"

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE=1
export SGLANG_NPU_USE_MULTI_STREAM=0

MODEL_PATH=/home/weights/Kimi-K3-W4A8
KEEP_LAYERS="${KEEP_LAYERS:-48}"
ENABLE_PP="${ENABLE_PP:-0}"

PF_IP="80.5.17.34"
export ASCEND_MF_STORE_URL="${ASCEND_MF_STORE_URL:-tcp://${PF_IP}:24670}"

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
export HCCL_BUFFSIZE=1200
export DEEPEP_NORMAL_LONG_SEQ_ROUND=64
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512

export HCCL_OP_EXPANSION_MODE=AIV
export ASCEND_CUSTOM_OPP_PATH=/home/z30071866/cann9.1.0/cann-9.1.0-beta.3/opp/vendors/custom_transformer
export LD_LIBRARY_PATH=/home/z30071866/cann9.1.0/cann-9.1.0-beta.3/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}

export SGLANG_NPU_PROFILING=0
export PYTHONPATH=/home/y00970600/KimiK3/sglang-kimiK3/python:$PYTHONPATH

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0
# first PD/CPP bring-up: no DSPARK
unset SGLANG_ENABLE_SPEC_V2 || true
unset SGLANG_RAGGED_VERIFY_MODE || true

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

dump_ifaces() {
    python3 <<'PY'
import fcntl, os, socket, struct

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
    print(f"{name}: {iface_ip(name)}")
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
    echo "ERROR: Prefill script must run on ${PF_IP}, LOCAL_IPS=${LOCAL_IPS[*]}"
    echo "If LOCAL_IPS has no 80.5.17.x, container likely is NOT --network host"
    dump_ifaces
    exit 1
fi

IFACE="$(find_iface_by_ip "${PF_IP}")"
if [[ -z "${IFACE}" ]]; then
    echo "ERROR: cannot find NIC for ${PF_IP}"
    dump_ifaces
    exit 1
fi
export HCCL_SOCKET_IFNAME="${IFACE}"
export GLOO_SOCKET_IFNAME="${IFACE}"
echo "Prefill -> ${PF_IP} iface=${IFACE} MF_STORE=${ASCEND_MF_STORE_URL} KEEP=${KEEP_LAYERS}"

# --- cut layers via launch override (weights on disk stay full) ---
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
block = int(tc.get("attn_res_block_size") or 12)
if keep % block != 0:
    print(f"[warn] KEEP={keep} not multiple of attn_res_block_size={block}")
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
print(f"override -> {sys.argv[3]}  layers {n}->{keep}  has_heads={tc_out.get('num_attention_heads')}  full_attn={lac['full_attn_layers']}")
PY
OVERRIDE_ARGS="$(cat "$OVERRIDE_JSON")"

if [[ "${ENABLE_PP}" == "1" ]]; then
    if [[ "${KEEP_LAYERS}" -lt 24 ]]; then
        echo "ERROR: ENABLE_PP=1 needs KEEP>=24 (attn_res_block_size=12)"
        exit 1
    fi
    HALF=$((KEEP_LAYERS / 2))
    export SGLANG_PP_LAYER_PARTITION="${SGLANG_PP_LAYER_PARTITION:-${HALF},${HALF}}"
    TP=8
    PP=2
    echo "CPP: tp=${TP} pp=${PP} PARTITION=${SGLANG_PP_LAYER_PARTITION}"
else
    unset SGLANG_PP_LAYER_PARTITION || true
    TP=16
    PP=1
    echo "PD-only: tp=${TP} pp=${PP}"
fi

sglang serve \
    --model-loader-extra-config '{"enable_multithread_load": true}' \
    --dist-init-addr "${PF_IP}:5001" --nnodes 1 --node-rank 0 \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$MODEL_PATH" \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --dtype bfloat16 \
    --tp-size "${TP}" \
    --pp-size "${PP}" \
    --disable-radix-cache \
    --mem-fraction-static 0.72 \
    --chunked-prefill-size 4096 \
    --max-running-requests 8 \
    --host 0.0.0.0 \
    --port 30001 \
    --reasoning-parser kimi_k3 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend ascend \
    --disaggregation-bootstrap-port 8998 \
    --json-model-override-args "${OVERRIDE_ARGS}" \
    --watchdog-timeout 9000 2>&1 | tee \
        "${LOG_DIR}/prefill34_keep${KEEP_LAYERS}_pp${PP}_$(date +%Y-%m-%d_%H-%M-%S).log"
status=${PIPESTATUS[0]}
exit "$status"
