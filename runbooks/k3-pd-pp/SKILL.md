---
name: k3-pd-pp
description: >-
  Bring up Kimi-K3 PD-disaggregated and Chunked-PP serving on multi-node Ascend.
  Use when launching K3 prefill/decode split, running the cut-layer smoke ladder,
  generating --json-model-override-args for layer truncation, setting
  SGLANG_PP_LAYER_PARTITION, or comparing baseline vs CPP vs dynamic chunking TTFT.
---

# Kimi-K3 PD / Chunked PP 起服

配套脚本：[`scripts/kimi-k3-cpp/`](../../scripts/kimi-k3-cpp/)
验收清单：[`scripts/kimi-k3-cpp/ACCEPTANCE.md`](../../scripts/kimi-k3-cpp/ACCEPTANCE.md)
项目导航：[`projects/kimi-k3/`](../../projects/kimi-k3/README.md)

> 本文 = **起服步骤**。
> **切点怎么选、裁几层、参数怎么填** → [`pp-partition-and-cut.md`](./pp-partition-and-cut.md)
> **层结构与代码就绪度** → [`knowledge/models/kimi-k3-layer-structure.md`](../../knowledge/models/kimi-k3-layer-structure.md)
> **后续 DSparkPP 源码结论（非本 baseline）** → [`knowledge/models/kimi-k3-dspark-pp-pd.md`](../../knowledge/models/kimi-k3-dspark-pp-pd.md)

> 前置：[`npu-docker-env/`](../npu-docker-env/SKILL.md)（容器内 PYTHONPATH）、
> [`npu-device-selection/`](../npu-device-selection/SKILL.md)（选卡与端口）。
> 分叉定位：[`../../debugging/pp-divergence/`](../../debugging/pp-divergence/SKILL.md)。

---

## §0 脚本一览

| 脚本 | 用途 |
|---|---|
| `prepare_k3_cut_override.py` | 生成裁层用的 `--json-model-override-args` |
| `smoke_pp_cut.sh` | **单机/小规模冒烟**，支持 `pp_only` / `pd_prefill` / `pd_decode` 三模式 |
| `run_prefill_34.sh` / `run_decode_38.sh` | 两机 PD（Prefill / Decode 各一台） |
| `run_pd_single_34.sh` / `run_pd_single_36.sh` | 单机 PD 单角色 |
| `cpp_pd.sh` | **四机 PD**（2 Prefill + 2 Decode） |
| `run_router.sh` | PD router，客户端打 `:8000` |

---

## §1 裁层 override（起服第一步）

```bash
python3 prepare_k3_cut_override.py --keep 24 -o overrides/k3_cut_l24.json
# 然后传给 server：--json-model-override-args "$(cat overrides/k3_cut_l24.json)"
```

### 三个必须知道的点

1. **必须输出完整的 `text_config` 副本**。SGLang 做 `config.update(override)`
   会**整体替换** `text_config`；只给部分字段会丢 `num_attention_heads`
   → `get_hf_text_config` 里 AssertionError。脚本已处理，别手写 JSON。
2. 本文旧版冒烟基线选 `--keep` 为 `attn_res_block_size`（默认 12）的倍数，
   例如 `12, 24, 36, ...`，脚本对非对齐值会 warn。该选择用于缩小旧路径变量；
   不把它表述为后续版本一律不可用的硬限制。
3. **trim 要做全**：`num_hidden_layers` + `linear_attn_config.kda_layers`
   + `full_attn_layers` 都要裁到 `<= keep`。只改 `num_hidden_layers` 会配置自相矛盾。

脚本会顺带打印建议的 `SGLANG_PP_LAYER_PARTITION`（默认对半：`12,12`）。

---

## §2 冒烟阶梯（按顺序，别跳）

```bash
# 1) 单机只验 PP 框架通路
MODE=pp_only KEEP_LAYERS=24 TP=8 PP=2 bash smoke_pp_cut.sh     # PARTITION=12,12

# 2) 最小 PD（两台机各跑一个角色）
MODE=pd_prefill KEEP_LAYERS=12 TP=16 PP=1 bash smoke_pp_cut.sh  # 节点 A
MODE=pd_decode  KEEP_LAYERS=12 TP=16      bash smoke_pp_cut.sh  # 节点 B

# 3) 升到 24 层 + Prefill 开 PP
#    Prefill pp=2 (PARTITION=12,12)，Decode pp=1
```

**每一步都要 ready + 一条端到端请求通过**才进下一步。

### smoke 脚本的关键环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODE` | `pp_only` | `pp_only` / `pd_prefill` / `pd_decode` |
| `KEEP_LAYERS` | `24` | 裁层后保留层数 |
| `TP` | `8` | |
| `PP` | `2` | |
| `MODEL_PATH` | 权重目录 | 以本机实际路径为准 |
| `CODE_ROOT` | 代码仓 | 会作为 `PYTHONPATH` 前缀**追加** |
| `HTTP_PORT` | `30001` | |
| `DIST_INIT` | `127.0.0.1:5001` | |
| `HCCL_BUFFSIZE` | `800` | 显存紧时下调 |

脚本内固定 export（**不要改**，改了对比就失真）：

```bash
export SGLANG_NPU_USE_MULTI_STREAM=0
export SGLANG_NPU_PROFILING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

---

## §3 四机 PD（`cpp_pd.sh`）

拓扑：**2 Prefill + 2 Decode**，各角色内 `NODE_RANK` 0/1。

```bash
# Prefill 节点
ROLE=prefill NODE_RANK=0 KEEP_LAYERS=24 bash cpp_pd.sh
# Decode 节点
ROLE=decode  NODE_RANK=0 KEEP_LAYERS=24 bash cpp_pd.sh
```

**本 runbook 的基线参数**：裁 24 层 + Prefill `pp=2` / Decode `pp=1` + 关 DSPARK。
这是为了控制首轮实验变量，不是对后续 DSparkPP addon 的兼容性判断。

| 角色 | 关键参数 |
|---|---|
| Prefill | `--disaggregation-mode prefill --nnodes 2 --tp-size 16 --pp-size 2 --dp-size 1` |
| Decode | `--disaggregation-mode decode --nnodes 2 --tp-size 32 --pp-size 1` |

- Prefill：`tp16 × pp2` 用满 2 机 × 16 die = 32 die，`SGLANG_PP_LAYER_PARTITION=12,12`
- Decode 保持 `pp=1`（**CPP 收益主要在 Prefill TTFT**）
- `chunked-prefill-size` 先固定 `4096` 或 `8192`（仅 Prefill）

完整推导与降级路径见 [`pp-partition-and-cut.md §3`](./pp-partition-and-cut.md)。

**起之前 checklist**：

- [ ] 四机容器 host 网络、网卡 / `/etc/hosts` / `HCCL_IF_IP` 正常
- [ ] override 已生成
- [ ] **关掉** DSPARK、multi-stream、`enable_pdmux`
- [ ] 端口避开同事：HTTP `30001`；Prefill dist `5001`；Decode dist `5002`

ready 后先长输入 **8K**，再 **32K**。

### router

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://<PREFILL_HOST>:30001 \
  --decode  http://<DECODE_HOST>:30001 \
  --host 0.0.0.0 --port 8000 --mini-lb
```

客户端打 router 的 `:8000`。真实主机地址以本机脚本为准，公开文档不写死。

---

## §4 CPP 对比实验（同 KEEP=24、同 PD）

| 实验 | 配置 | 记录 |
|---|---|---|
| 基线 | `pp=1`，固定 chunk 4096 | TTFT |
| CPP | `pp=2`，同 chunk，`PARTITION=12,12` | TTFT |
| Dynamic | 加 `--enable-dynamic-chunking`，init chunk ≈ 2–3×，smooth `0.65–0.8` | TTFT |

**验收要点**：

- Log/trace 里能看到多个 chunk **跨 PP stage 重叠**（否则 CPP 没生效）
- stage 不均就调 `SGLANG_PP_LAYER_PARTITION`（旧版满血基线可先试 `48,45`；以当前分支配置/验证为准）

---

## §5 OOM 与降级路径

按这个顺序退让：

1. 降 `HCCL_BUFFSIZE`
2. 降 `--mem-fraction-static`
3. 降 `--max-running-requests`
4. 退回 `1P+1D` + `KEEP=12`

> ⚠️ **不要把裁层模型的 TTFT 直接当满血结论。** 裁层只验证框架通路与调度，
> 满血需要更多机器或接受极紧显存。

---

## 踩坑账本

| 坑 | 症状 | 处理 |
|---|---|---|
| 手写 override JSON 只给部分字段 | `get_hf_text_config` AssertionError | 用脚本生成**完整** `text_config` |
| `--keep` 不是 block_size 倍数 | 配置不自洽 | 取 12 的倍数 |
| 只改 `num_hidden_layers` | `kda_layers` 越界 | 一并 trim `linear_attn_config` |
| 冒烟跳过 `pp_only` 直接上 PD | 分不清是 PP 还是 PD 的问题 | 按 §2 阶梯走 |
| 没关 DSPARK / multi-stream | 变量太多，结果不可归因 | 首版全关 |
| 拿裁层 TTFT 当满血结论 | 结论错误 | 明确标注裁层数 |

## Living doc

脚本改了 → 同步本文；K3 相关的具体故障 →
[`../../debugging/`](../../debugging/README.md)。
