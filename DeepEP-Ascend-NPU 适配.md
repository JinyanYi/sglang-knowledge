# DeepEP on Ascend NPU 适配笔记（Kimi-K3）

记录在 Ascend 910（A3）上跑 SGLang + DeepEP（Kimi-K3 W4A8，PD 分离）遇到的三个问题和根因结论。
**结论先行：前两个问题的根因都在 DeepEP kernel 层 / 运行配置，不是 SGLang 集成问题，也不是 INT8/BF16 或 CUDA Graph 的问题。**

---

## 1. 背景：`--moe-a2a-backend` 是什么

它决定模型 MoE 的**跨 rank 通信方式**（`MoeA2ABackend`，见 `sglang/srt/layers/moe/utils.py`）。真正影响 K3 行为的是 `none` 和 `deepep` 两种：

| | `none`（TP 内 MoE，ep=1） | `deepep`（EP a2a，SP-MoE） |
|---|---|---|
| token 分布 | 每个 rank 持有完整 batch | 每个 rank 只持有 token 分片（scattered） |
| expert 计算 | 每 rank 算全部专家在 TP 上的分片，输出 all-reduce | 每 rank 只算自己管辖的 1/ep 个专家，token 被 dispatch/combine 移动 |
| 通信 | MLP 内 all-reduce | DeepEP 内核跨卡移动 token |

要点：
- **`deepep` 就是"开跨卡专家并行"的开关**。没有单独的 "EP 开关"，也没有"配套开关"；`--deepep-mode`（normal/ll/auto）只是 DeepEP 内部选哪种 dispatch。
- `--moe-a2a-backend none` = 关掉 DeepEP，回到 TP 内 MoE。之前"关掉 deepep 就不崩"正是因为 896 专家的 DeepEP 内核根本没有被调用。
- K3 里开 `deepep` 后 shared experts 需要按 tp1 复制（各 rank token 不同，无法 TP 分片归约），见 `sglang/srt/models/kimi_k3.py` 的 `_ep_a2a` 分支。

## 2. DeepEP 的两种 dispatch 模式

- **normal**：prefill / extend 用。`aclnnCamMoeDispatchNormal`（内核 `cam_moe_dispatch_normal`）。
- **low-latency**：decode 用。`aclnnMoeLowLatencyDispatchV2`。
- `--deepep-mode auto` 按 batch 是否含 prefill/extend 自动切换（`DeepEPMode.resolve`）。

**Kimi-K3 真实形状：`hidden=7168, num_experts=896, num_experts_per_token(topk)=16`**（见 `/data/weights/kimi-k3-5layers/config.json`）。

---

## 3. 问题一：Prefill 崩溃 `507015` / "MPU address access is invalid"

**现象**：prefill 首轮 dispatch 就崩，错误签名固定为：

```
RuntimeError: ... error code is 507015
fftsplus aivector error ... errorStr: The MPU address access is invalid
```

栈：`buffer.dispatch → normal_strategy.dispatch:116 → _intranode_dispatch → runtime.intranode_dispatch`

**根因（已定位）**：DeepEP **normal** dispatch 的**多轮路径（round>1）**在 **num_experts=896（> 512）** 下越界，越界点在内核 `notify_dispatch.h` 的 UB 分配，不是"专家数上限 512"：
- A3 的 tiling/布局校验上限其实是 **1024**（`csrc/deepep/ops/op_host/cam_moe_dispatch_normal_tiling.cc` 的 `MOE_EXPERT_MAX_NUM=1024`、`dispatch_layout_tiling.cc` 的 `MAX_MOE_EXPERTS_NUM=1024`），896 专家在显式校验下全部合法。**512 是 `ops2/`（A2/910B）路径的上限，不是 A3 的**。
- 真正崩点是 `notify_dispatch.h`：多轮时 `batchRounds` 按专家数分流，但原实现把 `numLocalExperts >= 128` 放最前，在 ep≤4 且 numExperts>512 时恒命中 → `batchRounds=16`；UB 按 `16 × 896` 分配，`sendDataAlignLen≈172KB` 再加其他 buffer 超 **192KB 上限** → 内核写越界 → AIVector 非法访问。
- 二分复现：384 专家通过，896 专家（无论 topk=8 还是 16）必崩 → **与 topk 无关，是专家数**。

**修复（已应用、未提交）**：`csrc/deepep/ops/op_kernel/notify_dispatch.h`：
- 新增 `round==1 → batchRounds=1` 特判（单轮不需要多轮批处理，UB 最小）；
- 调整分支顺序：`numExperts > 512` 的"降半批"分支提到最前，消除死代码；
- 验证：896 + round=64 + 128 token（realRound=1）通过；修复前同配置必崩 507015。

**未决问题**：realRound≥2（单次 dispatch token > per_round_tokens=512）时，`CamMoeDispatchNormal do tiling failed, ret=-1`（host 侧 `OP_TILING_CHECK` 拒绝执行）。根因疑似窗口校验（`cam_moe_dispatch_normal_tiling.cc` 的 `actualSize > maxWindowSize`，round>1 时 combine 走 double buffer 使需求增大），**尚未定性**，与 decode 的 "HCCL_BUFFSIZE too SMALL" 是同一条链路。

**注意（容易误判）**：`DEEP_NORMAL_MODE_USE_INT8_QUANT=1` 是**干扰项**——安装版 deep_ep 的 normal dispatch 根本不读它（`quant_mode=None` 直接走 BF16），节点实际跑的就是 BF16。INT8/BF16 不是判别变量。

**结论**：不是"A3 不支持 896 专家"（A3 上限 1024），而是多轮路径有两处问题：① notify UB 越界（已修）；② realRound≥2 的 tiling 校验失败（未修）。**单轮（round=1）+ 小 chunk 端到端可用**。

## 4. 问题二：Decode tiling 失败 "HCCL_BUFFSIZE is too SMALL"

**现象**：decode 用 low-latency dispatch 时 tiling 直接失败：

```
[MoeLowLatencyDispatchV2] HCCL_BUFFSIZE is too SMALL, maxBs=64, h=7168, epWorldSize=8,
localMoeExpertNum=112, k=16, NEEDED=1733MB, HCCL_BUFFSIZE=800MB
```

**根因**：low-latency 的窗口公式（`moe_distribute_dispatch_v2_tiling.cpp` 的 `CheckWinSize`）：

```
actualSize = (maxBs × dispatch_size × ep × local_experts + maxBs × combine_size × (k+shared)) × 2
```

896 专家 + topk16 下：EP8（decode 实际，maxBs=64）需 **~1.73GB**，EP4（maxBs=128）需 **~3.3GB**。脚本只给了 `HCCL_BUFFSIZE=800`。

**修复**：decode 节点 `HCCL_BUFFSIZE=800 → 2048`（按 MiB 计，`HCCL_BUFFSIZE` 经 `mc2_tiling_utils.h` 读取，`uint16_t` 无碍）。EP4 的测试环境要 4096。改完必须重启进程（启动时读 env）。

**补充**："关了 graph 才暴露" 是因为 graph 复播跳过 tiling；eager 才每次做 tiling。

**K3 满血 PD（2026-08-13）**：EP16 + `HCCL_BUFFSIZE=2048` 仍可能 **没有** too SMALL、只有 `HcclAllocComResourceByTiling ret=24`。那是 leftover（KV 分完后 ~4.7GB @ f=0.92）不够 HCCL **整块** 申请，不是公式 1733MB 没盖住。处理：降 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`（maxBs）并相应降 `HCCL_BUFFSIZE`，或把 fraction 降到刚好比 `1−post/pre` 高一点以增大 leftover。见 [Memory Breakdown §6](./Memory%20Breakdown%20and%20case%20study.md)。

## 5. 问题三：CUDA Graph bs 必须是 attn_tp_size 的倍数

**现象**：decode 节点 `--cuda-graph-bs 1 2` 无效，启动时 `assert len(capture_bs) > 0` 失败。

**原因**（`get_batch_sizes_to_capture` / `get_cuda_graph_batch_size_alignment`）：
- 开了 `deepep`（a2a 非 none）→ `require_attn_tp_gather()=True` → 对齐基数 = `attn_tp_size` = 8。
- `capture_bs` 里不满足 `bs % 8 == 0` 的值全部被过滤 → 1、2 被剔除 → 空列表 assert。
- 根本原因：scattered 模式下每 rank 只拿 `bs/attn_tp` 个 token，必须是整数；且 DeepEP a2a 用固定几何 buffer，**decode graph bucket 必须跨 EP rank 完全一致**，否则几何不匹配 → 非法访问（代码注释引用 issue #30242）。

**结论**：`deepep` + tp=8 时用 `--cuda-graph-bs 8 16 ...`；`--moe-a2a-backend none` 时 `1 2` 合法（对齐基数=1）。

---

## 6. 版本信息（2026-08-10 排查时）

- deep_ep wheel：`1.0.0+48942e37.cann.9.0.0.b250`（构建 commit 48942e37，2026-07-27）
- DeepEP OPP：`custom_opp_compiler_version=9.0.0`（vendors/customize）
- sgl-kernel-npu HEAD：`feca6c9`（含 k3-pp 分支）；A3 路径 `MOE_EXPERT_MAX_NUM=1024`（`ops/`），`ops2/`（A2）才是 512；notify 修复在本机工作区未提交
- 容器 `sglang-yjy`（`lmsysorg/sglang:main-cann9.0.0-a3`）deep_ep 与节点完全同版，可作本地复现环境

## 7. 复现方法（本地隔离测试）

```bash
# prefill 崩溃复现（normal dispatch, BF16, 896 专家）
docker exec sglang-yjy bash -lc '
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export DEEPEP_NORMAL_LONG_SEQ_ROUND=64
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512
export HCCL_BUFFSIZE=800
cd /home/y00970600/personal/sgl-kernel-npu/tests/python/deepep
python3 test_intranode.py --num-processes 4 --num-tokens 128 \
  --hidden 7168 --num-topk 16 --num-experts 896 --quant-type bf16'
# 预期：修复前 507015 + MPU invalid（与节点一致）；384 专家则 PASSED。
#       带 notify 修复后：128 token（realRound=1）应通过；
#       realRound≥2（--num-tokens 600/1024）→ tiling ret=-1（未决问题）

# decode tiling 复现（low-latency）
python3 test_low_latency.py --num-processes 4 --num-tokens 128 \
  --hidden 7168 --num-topk 16 --num-experts 896 --quant-type no
# 预期：HCCL_BUFFSIZE too SMALL
```

注意：507015 之后该 NPU 进程上下文不可复用，需重启进程；复现用的空闲卡（如 4-7）不会影响其他占用。
