---
name: precision-dump-compare
description: >-
  Locate numeric divergence in SGLang NPU by dumping per-layer hidden states and
  comparing across parallel configs (TP/PP). Use when accuracy drops, outputs
  repeat/乱码, PP1 vs PP4 disagree, or you need to find which layer first
  diverges. Covers SGLANG_K3_DUMP_HIDDEN env vars, sp_sharded all-gather
  requirement, filename scheme, and which metrics to compute.
---

# 精度 / 数值分叉定位

> 相关：[`../pp-divergence/`](../pp-divergence/SKILL.md)（PP 专用路径）、
> [`../../knowledge/inference/`](../../knowledge/inference/)（数值原理不在本文）。

## 核心原则（先记住，能省一半时间）

1. **不要只比 `norm` / `sum`**。这两个量对「哪个元素错了」几乎不敏感，看不出分叉点。
   要 dump 完整 tensor，算 **`max_abs` / `mean_abs` / `relative L2` / `cosine`**。
2. **先测 noise floor**。同一配置重复跑 ≥3 次，得到「什么都不改也有多大波动」。
   比 noise floor 还小的差异 **不是 bug**。
3. **只引入一个变量**。要查 PP 就固定 TP（`PP1 TP16` vs `PP4 TP16`），不要两边都变。
4. **对比前确认配置真的是你以为的那样**（见 §3 的 DP 陷阱）。

---

## §1 hidden state dump（以 Kimi K3 为例）

### 开关

| 环境变量 | 作用 |
|---|---|
| `SGLANG_K3_DUMP_HIDDEN=1` | 启用 dump |
| `SGLANG_K3_DUMP_HIDDEN_LAYERS` | 覆盖层列表，默认 `0,23,47,59,92` |
| 另设 `SGLANG_K3_DUMP_DIR` | 输出目录，不同配置必须分开 |

### 插桩位置

`kimi_k3.py` 中每层 `self.layers[i](...)` 返回后、**DSpark capture 前**：

```python
hidden_states, residual, sp_sharded = self.layers[i](...)
# ← dump 插这里
```

### 必须遵守的约束

| 约束 | 原因 |
|---|---|
| `sp_sharded=True` 时，所有参与 rank **都要**执行 debug-only `_sp_all_gather_rows()` | **不能只让 TP0 进 collective**，否则整组 hang |
| 只让 **TP rank 0** 写文件 | 避免同 stage 重复文件 |
| 只处理 **prefill / EXTEND**，跳过 DECODE / IDLE | decode 会淹没真正关心的 prefill 数据 |
| 真实 token 行数用 `sum(forward_batch.extend_seq_lens_cpu)` 并**裁掉 padding** | padding 会污染比较 |
| 存 **float32 CPU tensor**（直接存 tensor，别转 numpy） | 方便 `torch.load` 后直接比 |
| 文件名含 **global layer + PP rank + 真实 token 行数** | 见下方陷阱 |
| 不要动主路径的 `hidden_states` / `sp_sharded` | dump 必须是纯旁路 |
| `os.makedirs(..., exist_ok=True)` 建父目录 | shell 侧不一定建了 |

### 文件名陷阱（真踩过）

文件名 **必须带 token 行数**，例如 `layer023_pp00_tokens96.pt`。
否则 health check（4 token）会**覆盖** curl prefill（96 token）——
你以为在比 prefill，其实比的是 health check。

---

## §2 比什么

拿到两边的 `.pt` 之后，逐 token 算：

| 指标 | 用途 |
|---|---|
| `max_abs` | 最坏单点误差，定位「是不是有个别 token 炸了」 |
| `mean_abs` | 整体量级 |
| `relative L2` | 相对误差，跨量级可比 |
| `cosine` | 方向一致性；**逐 token** 算，不要对整个矩阵算一个值 |
| mismatch ratio | 超过阈值的元素占比 |

**单独看 `norm` / `sum` 不足以定位问题**——这是最容易犯的错。

---

## §3 对比前必须确认的事

| 检查 | 为什么 |
|---|---|
| **baseline 真的关掉 DP 了吗** | 开着 DP 时 TP0 保存不稳定，无法一对一比较 |
| 两边用的是不是同一份代码 | `git show HEAD:<file>` 核对，别信工作区脏改动 |
| 并行配置差异是否只有一个变量 | TP 和 PP 同时变 → 结论不可归因 |
| `max_running_requests` 有没有被砍 | 不影响精度，但说明显存状态不同 |

> **重要**：TP 变化本身就会引入 all-reduce 顺序/数值差异，
> **TP64 vs TP16 不是 bit-exact，也不能期望 bit-exact**。
> 要分离 PP 的影响，必须在**同 TP** 下比 PP1 vs PP4。

---

## §4 常见结论模式

| 现象 | 通常意味着 |
|---|---|
| 前几层只有噪声级差异，输出却乱码 | 分叉在**后层累积**，或在某个 PP boundary 之后；继续往前层找是浪费 |
| 差异在某层**阶跃**变大 | 分叉点就在该层，去看该层的算子实现 |
| 每个 PP boundary 后第一层差异变大 | PP 传输/拼接有问题 → 转 [`../pp-divergence/`](../pp-divergence/SKILL.md) |
| 重复运行差异量级不稳定 | 还没测出 noise floor，先跑 3 次 |
| NPU vs GPU 精度差一个固定比例 | 查算子是否漏了某个 scaling factor —— 见 [`../cases/2026-hunyuan3-topk-scaling-factor.md`](../cases/2026-hunyuan3-topk-scaling-factor.md) |

## Living doc

新增 dump 开关/指标 → 改本文；某个具体模型的完整定位过程 → 写 [`../cases/`](../cases/)。
