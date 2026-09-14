# DeepEP on Ascend NPU：概念与机制

> **本文只讲「为什么」。**
> 具体的报错、定位过程、修复命令已拆至
> [`debugging/deep-ep/`](../../debugging/deep-ep/SKILL.md) 与
> [`debugging/cases/`](../../debugging/cases/)（三个 case 见文末）。
>
> 起因：在 Ascend 910（A3）上跑 SGLang + DeepEP（Kimi-K3 W4A8，PD 分离）时遇到的三个问题。
> **结论先行：前两个问题的根因都在 DeepEP kernel 层 / 运行配置，不是 SGLang 集成问题，
> 也不是 INT8/BF16 或 CUDA Graph 的问题。**

---

## 1. `--moe-a2a-backend` 是什么

它决定模型 MoE 的**跨 rank 通信方式**（`MoeA2ABackend`，见 `sglang/srt/layers/moe/utils.py`）。真正影响 K3 行为的是 `none` 和 `deepep` 两种：

| | `none`（TP 内 MoE，ep=1） | `deepep`（EP a2a，SP-MoE） |
|---|---|---|
| token 分布 | 每个 rank 持有完整 batch | 每个 rank 只持有 token 分片（scattered） |
| expert 计算 | 每 rank 算全部专家在 TP 上的分片，输出 all-reduce | 每 rank 只算自己管辖的 1/ep 个专家，token 被 dispatch/combine 移动 |
| 通信 | MLP 内 all-reduce | DeepEP 内核跨卡移动 token |

要点：

- **`deepep` 就是「开跨卡专家并行」的开关**。没有单独的 "EP 开关"，也没有"配套开关"；
  `--deepep-mode`（normal/ll/auto）只是 DeepEP 内部选哪种 dispatch。
- `--moe-a2a-backend none` = 关掉 DeepEP，回到 TP 内 MoE。
  之前「关掉 deepep 就不崩」正是因为 896 专家的 DeepEP 内核根本没有被调用
  → 这是**判断问题是否出在 DeepEP 的最快二分手段**。
- **开 `deepep` 后 shared experts 需要按 tp1 复制**：各 rank token 不同，
  无法用 TP 分片归约（见 `sglang/srt/models/kimi_k3.py` 的 `_ep_a2a` 分支）。

## 2. 两种 dispatch 模式

- **normal**：prefill / extend 用。`aclnnCamMoeDispatchNormal`（内核 `cam_moe_dispatch_normal`）。
- **low-latency**：decode 用。`aclnnMoeLowLatencyDispatchV2`。
- `--deepep-mode auto` 按 batch 是否含 prefill/extend 自动切换（`DeepEPMode.resolve`）。

**Kimi-K3 真实形状：`hidden=7168, num_experts=896, topk=16`。**

---

## 3. 三条机制性约束（为什么会这样）

这三条是**平台/实现决定的常量与公式**，不依赖某次失败，所以留在这里。
对应的报错现象见下节的 case 链接。

### 3.1 A3 的专家数上限是 1024，不是 512

| 路径 | 常量 | 值 |
|---|---|---|
| `csrc/deepep/ops/op_host/cam_moe_dispatch_normal_tiling.cc` | `MOE_EXPERT_MAX_NUM` | **1024** |
| `csrc/deepep/ops/op_host/dispatch_layout_tiling.cc` | `MAX_MOE_EXPERTS_NUM` | **1024** |
| `ops2/`（**A2 / 910B** 路径） | — | 512 |

**896 专家在 A3 上是合法的。**「512 是 A3 上限」是常见误传 —— 那是个 A2 的数字。
→ 相关 case：[`2026-08-deepep-prefill-507015-notify-dispatch-ub.md`](../../debugging/cases/2026-08-deepep-prefill-507015-notify-dispatch-ub.md)

### 3.2 low-latency 的窗口公式

`moe_distribute_dispatch_v2_tiling.cpp` 的 `CheckWinSize`：

```
actualSize = (maxBs × dispatch_size × ep × local_experts
            + maxBs × combine_size × (k + shared)) × 2
```

**为什么必须这么大**：low-latency 走的是固定几何的通信窗口，
`dispatch` 与 `combine` 各占一份，且要 double buffer（那个 `× 2`）。
`maxBs` 越大、`ep` / `local_experts` / `topk` 越大，需求线性增长。

896 专家 + topk16 下：EP8（`maxBs=64`）需 ~1.73GB，EP4（`maxBs=128`）需 ~3.3GB。
→ 相关 case：[`2026-08-deepep-decode-hccl-buffsize-too-small.md`](../../debugging/cases/2026-08-deepep-decode-hccl-buffsize-too-small.md)

### 3.3 开 `deepep` 后 graph bs 的对齐基数变成 `attn_tp_size`

`get_cuda_graph_batch_size_alignment`：开 `deepep` → `require_attn_tp_gather()=True`
→ 对齐基数 = `attn_tp_size`。

**为什么必须是整数倍**，两个原因缺一不可：

1. **scattered 模式下每 rank 只拿 `bs / attn_tp` 个 token，必须是整数**；
2. DeepEP a2a 用**固定几何 buffer**，decode graph bucket 必须**跨 EP rank 完全一致**，
   否则几何不匹配 → 非法访问（代码注释引用 issue #30242）。

所以 tp=8 时 `--cuda-graph-bs` 只能是 `8 16 ...`；
`--moe-a2a-backend none` 时对齐基数回到 1，`1 2` 合法。
→ 相关 case：[`2026-08-deepep-cuda-graph-bs-alignment.md`](../../debugging/cases/2026-08-deepep-cuda-graph-bs-alignment.md)

---

## 4. 关于 `DEEP_NORMAL_MODE_USE_INT8_QUANT`

**它是 normal dispatch 的干扰项**：安装版 `deep_ep` 的 normal dispatch
**根本不读这个变量**（`quant_mode=None` 直接走 BF16）。

这个变量在别的地方（DeepEP hang 场景）**是**有效诱因，见
[`debugging/deep-ep/SKILL.md §4`](../../debugging/deep-ep/SKILL.md)。
**用在哪条路径上要分清**，不要笼统地说「int8 量化有问题」。

---

## 相关 case（故障排查）

| Case | 症状 |
|---|---|
| [`2026-08-deepep-prefill-507015-notify-dispatch-ub.md`](../../debugging/cases/2026-08-deepep-prefill-507015-notify-dispatch-ub.md) | prefill 首轮崩 `507015` / MPU invalid（`notify_dispatch.h` 多轮 UB 越界） |
| [`2026-08-deepep-decode-hccl-buffsize-too-small.md`](../../debugging/cases/2026-08-deepep-decode-hccl-buffsize-too-small.md) | decode tiling 失败 `HCCL_BUFFSIZE is too SMALL` |
| [`2026-08-deepep-cuda-graph-bs-alignment.md`](../../debugging/cases/2026-08-deepep-cuda-graph-bs-alignment.md) | `assert len(capture_bs) > 0` |

## 版本与复现环境

见 [`debugging/deep-ep/SKILL.md §8`](../../debugging/deep-ep/SKILL.md)。
