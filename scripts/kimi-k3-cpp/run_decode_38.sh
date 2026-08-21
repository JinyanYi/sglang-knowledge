#!/bin/bash
# Decode @ 80.5.17.38  — copy this file to the Decode machine and run.
# Pair with the Prefill script on 80.5.17.34. KEEP_LAYERS must match Prefill.
# No PP on decode.
#
# Codebase: no change needed.
# Weights: do NOT edit config.json on disk; override is generated below.

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

PF_IP="80.5.17.34"
DC_IP="80.5.17.38"
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
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
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
    if [[ "$lip" == "$DC_IP" ]]; then
        MATCH=1
        break
    fi
done
if [[ "$MATCH" -ne 1 ]]; then
    echo "ERROR: Decode script must run on ${DC_IP}, LOCAL_IPS=${LOCAL_IPS[*]}"
    echo "If LOCAL_IPS has no 80.5.17.x, container likely is NOT --network host"
    dump_ifaces
    exit 1
fi

IFACE="$(find_iface_by_ip "${DC_IP}")"
if [[ -z "${IFACE}" ]]; then
    echo "ERROR: cannot find NIC for ${DC_IP}"
    dump_ifaces
    exit 1
fi
export HCCL_SOCKET_IFNAME="${IFACE}"
export GLOO_SOCKET_IFNAME="${IFACE}"
echo "Decode -> ${DC_IP} iface=${IFACE} MF_STORE=${ASCEND_MF_STORE_URL} KEEP=${KEEP_LAYERS}"

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

sglang serve \
    --model-loader-extra-config '{"enable_multithread_load": true}' \
    --dist-init-addr "${DC_IP}:5002" --nnodes 1 --node-rank 0 \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$MODEL_PATH" \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --dtype bfloat16 \
    --tp-size 16 \
    --pp-size 1 \
    --disable-radix-cache \
    --mem-fraction-static 0.72 \
    --max-running-requests 8 \
    --host 0.0.0.0 \
    --port 30001 \
    --reasoning-parser kimi_k3 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend ascend \
    --disaggregation-bootstrap-port 8998 \
    --json-model-override-args "${OVERRIDE_ARGS}" \
    --watchdog-timeout 9000 2>&1 | tee \
        "${LOG_DIR}/decode38_keep${KEEP_LAYERS}_$(date +%Y-%m-%d_%H-%M-%S).log"
status=${PIPESTATUS[0]}
exit "$status"
