---
name: pp-divergence
description: >-
  Find where pipeline-parallel (PP) execution first diverges from a PP=1 baseline
  on SGLang NPU. Use when PP1 vs PP4 disagree, PP outputs are 乱码 while early
  layers look fine, or you need to prove whether PP send/recv is correct.
  Covers the cleanest experiment design, which tensors to compare at each
  boundary, and how to fit more layers into a PP1 reference run.
---

# PP 逐层分叉定位

> 前置：[`../precision/`](../precision/SKILL.md)（dump 与指标怎么用）。
> 本文只讲 **PP 特有的**实验设计与观察点。

## §0 最容易踩的概念坑

> **不是「local layer ID vs global layer ID」，而是「local buffer position vs global layer ID」。**

这个区分搞错，后面所有层的对应关系都是错的。
`attn_res` 这类结构是**按 buffer position 索引**的，不是按全局层号。

---

## §1 实验设计（先做对这一步）

### 最干净的对照

```
A: PP=1, TP=16, num_hidden_layers=24
B: PP=4, TP=16, num_hidden_layers=24, SGLANG_PP_LAYER_PARTITION=6,6,6,6
```

**TP 完全一致，只引入 PP 变量** —— 这是唯一能把差异归因到 PP 的设计。

### 为什么不要两边都是 TP4

TP4 下 PP1 装不下足够的层，只能比前面几层，而前几层通常只有噪声级差异
→ 你会得出「没问题」或「看不出来」的错误结论。

> **不建议继续硬挤 PP1 TP4。** `PP1 TP16 vs PP4 TP16` 的 24 层对照才是当前最干净、
> 最快出结论的实验。

### 层裁剪

用 `--json-model-override-args` 改 `num_hidden_layers`，**记得同时 trim**
`linear_attn_config`（否则配置自相矛盾）。

---

## §2 观察点（按顺序，不要跳）

| # | 比什么 | 期望 |
|---|---|---|
| 1 | layer 0–5：PP1 vs PP4 rank0 | 一致（差异应在 noise floor 内） |
| 2 | **PP4 rank0 的 `pre-send` vs rank1 的 `post-recv`** | **完全一致 / bitwise 一致** |
| 3 | PP1 的 layer6 输入 vs PP4 rank1 的 layer6 输入 | 一致 |
| 4 | layer 6 / 12 / 18（**每个 PP boundary 后的第一层**） | 差异从这里开始变大 |

第 3 步要同时比较三样东西，缺一不可：

- `hidden_states`
- **完整的** `attn_res.block_residual`（不是只有 norm）
- `num_valid_blocks`

### 为什么先看 boundary 后第一层

「最终乱码但前层只有噪声级差异」时，真实 divergence 往往在
**后层累积**或**某个 boundary 之后**才出现。盲目往后层扫 norm 是最慢的路径。

---

## §3 判定标准

先得到 **PP1 自身的 noise floor**（同配置跑 ≥3 次）。

- 差异 ≤ noise floor → **不是 PP bug**，别继续查。
- 差异在某层阶跃 → 分叉点确定，去查该层算子。
- boundary 两侧不一致 → PP 传输/拼接/缓冲区索引有问题（回到 §0 的概念坑）。

---

## §4 PP1 装不下层时的手段（按代价排序）

| 手段 | 代价 |
|---|---|
| `--disable-cuda-graph` | 显存，无速度损失（但测的不再是生产路径） |
| `--max-running-requests 1` | 只影响并发，数值不受影响 ✅ 优先 |
| 降低 `--max-total-tokens` | 同上 ✅ |
| 减小 `--chunked-prefill-size` | 同上 ✅ |
| 调低 `--mem-fraction-static` | KV 池变小 |
| `--cpu-offload-gb N` | **很慢**；NPU quantization 路径已处理 CPU offload，但只适合小规模验证兼容性 |
| 「后半段 replay」 | 从正常跑 dump 某 boundary 的 `hidden_states + attn_res bank`，只加载后续 layer slice 离线执行。**最省显存，但需要专门入口**，普通 serving 流程做不到 |

## Living doc

发现新的 boundary 相关坑 → 补本文；若是某个模型/某次事故的完整过程 →
写 [`../cases/`](../cases/)。
