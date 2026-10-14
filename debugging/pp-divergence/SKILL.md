---
name: pp-divergence
description: >-
  Find where pipeline-parallel (PP) execution first diverges from a PP=1 baseline
  on SGLang NPU. Use when PP1 vs PP>1 disagree, PP outputs are 乱码 while early
  layers look fine, or you need to prove whether PP send/recv is correct.
  Covers a controlled comparison design, which tensors to compare at each
  boundary, and how to fit more layers into a PP1 reference run.
---

# PP 逐层分叉定位

> 前置：[`../precision/`](../precision/SKILL.md)（dump 与指标怎么用）。
> 本文只讲 **PP 特有的**实验设计与观察点。
> K3 的项目入口、版本范围和运行基线见 [`../../projects/kimi-k3/`](../../projects/kimi-k3/README.md)。

## §0 最容易踩的概念坑

> **不是「local layer ID vs global layer ID」，而是「local buffer position vs global layer ID」。**

这个区分搞错，后面所有层的对应关系都是错的。
`attn_res` 这类结构是**按 buffer position 索引**的，不是按全局层号。

---

## §1 实验设计（先做对这一步）

### K3 旧版基线下的推荐对照

```
A: PP=1, TP=16, num_hidden_layers=24
B: PP=2, TP=16, num_hidden_layers=24, SGLANG_PP_LAYER_PARTITION=12,12
```

**TP 保持一致，并尽量只改变 PP**，能显著缩小归因范围；仍应记录版本、采样、输入、
图模式及其他开关，避免把未控制变量误归因给 PP。
`12,12` 只是 K3 旧版冒烟基线；后续分支可使用其他已经验证的分区，不能由此反推
12 对齐是架构硬限制。

### 为什么不要两边都是 TP4

若 TP4 下 PP1 装不下足够层，只能比较前面几层，可能看不见后段或边界后的差异；
应把这个覆盖范围限制写进结论。

> 在具备相同资源与旧版配置时，`PP1 TP16 vs PP2 TP16` 的 24 层对照是一个短、
> 可复核的起点；它不替代对当前模型、层数和分区的覆盖评估。

### 层裁剪

用 `--json-model-override-args` 改 `num_hidden_layers`，**记得同时 trim**
`linear_attn_config`（否则配置自相矛盾）。

---

## §2 观察点（按顺序，不要跳）

| # | 比什么 | 期望 |
|---|---|---|
| 1 | layer 0–11：PP1 vs PP2 rank0 | 一致（差异应在 noise floor 内） |
| 2 | **PP2 rank0 的 `pre-send` vs rank1 的 `post-recv`** | **完全一致 / bitwise 一致** |
| 3 | PP1 的 layer12 输入 vs PP2 rank1 的 layer12 输入 | 一致 |
| 4 | layer 12（**PP boundary 后的第一层**） | 检查是否出现超出 noise floor 的突变 |

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

- 差异 ≤ noise floor → 当前测量点尚无 PP 特有分叉证据；保留输入、层覆盖和复跑条件。
- 差异在某层阶跃 → 候选分叉点；先复跑确认，再检查该层算子与输入状态。
- boundary 两侧不一致 → 优先检查 PP 传输、拼接和缓冲区索引（回到 §0 的概念坑）。

---

## §4 PP1 装不下层时的手段（按代价排序）

| 手段 | 代价 |
|---|---|
| `--disable-cuda-graph` | 释放图相关显存，但调试运行可能变慢，且不再是生产图路径 |
| `--max-running-requests 1` | 只影响并发，数值不受影响 ✅ 优先 |
| 降低 `--max-total-tokens` | 同上 ✅ |
| 减小 `--chunked-prefill-size` | 同上 ✅ |
| 调低 `--mem-fraction-static` | KV 池变小 |
| `--cpu-offload-gb N` | **很慢**；NPU quantization 路径已处理 CPU offload，但只适合小规模验证兼容性 |
| 「后半段 replay」 | 从正常跑 dump 某 boundary 的 `hidden_states + attn_res bank`，只加载后续 layer slice 离线执行。**最省显存，但需要专门入口**，普通 serving 流程做不到 |

## Living doc

发现新的 boundary 相关坑 → 补本文；若是某个模型/某次事故的完整过程 →
写 [`../cases/`](../cases/)。
