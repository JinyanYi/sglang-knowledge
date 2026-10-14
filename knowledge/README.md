# Knowledge —— 为什么是这样

**这一类只放「原理 / 架构 / 设计」**：不依赖某次失败，也不告诉你要敲什么命令。
判据：**半年后重读依然成立**，且换了机型、换了镜像、换了任务也还有用。

| 不属于这里 | 该去哪 |
|---|---|
| 「我遇到了 X 报错」 | [`../debugging/`](../debugging/README.md) |
| 「怎么把服务起起来 / 怎么复现」 | [`../runbooks/`](../runbooks/README.md) |
| 「这次调出来的参数是多少」 | [`../runbooks/`](../runbooks/README.md)（参数属于操作，不属于原理） |

**写在文档里的「踩坑」要有边界**：只留「为什么会这样」的原理性解释（例如「`deepep` 模式下 shared expert 必须按 tp1 复制，因为各 rank token 不同无法 TP 分片归约」）；具体的报错串、修复命令、重试次数一律挪去 debugging。

---

## inference —— 推理机制

| 文档 | 内容 |
|---|---|
| [prefill-decode.md](./inference/prefill-decode.md) | 两阶段计算特性、Arithmetic Intensity、为什么需要 PD 分离 |
| [cuda-graphs.md](./inference/cuda-graphs.md) | CUDA Graph、Piecewise CUDA Graph 与 SGLang 实现 |
| [speculative-decoding.md](./inference/speculative-decoding.md) | N-gram / EAGLE / EAGLE-3 / DFlash 原理对比与使用 |
| [multi-token-prediction.md](./inference/multi-token-prediction.md) | MTP 训练方式、加速原理、SGLang 实现 |
| [sampling-params.md](./inference/sampling-params.md) | Temperature 等采样参数的作用机制与实现 |

## memory —— 显存

| 文档 | 内容 |
|---|---|
| [memory-budget-and-scheduling.md](./memory/memory-budget-and-scheduling.md) | Static/Dynamic 划分、`mem-fraction-static`、KV 池 profiling、PrefillAdder |

## models —— 模型与架构

K3 跨主题内容从 [项目入口](../projects/kimi-k3/README.md) 找；架构正文仍保留在本目录，实验时间线不混入原理页。

**GDN/SSM 学习笔记**：[Qwen3.6 的状态、投影与 shape](./models/qwen3-6-gdn-state-and-shapes.md)。记录 2026-10-13 的逐步学习、正确回答摘录与明天的接续点；从 q/k/v、TP 分片、conv/temporal 状态读到 PR #43845。

后续目标：[Qwen GDN 完整算法](./models/qwen3-6-gdn-algorithm.md)；原始资料：[GDN、KDA 与 Kimi K3 的论文阅读顺序及本地 PDF](./models/papers/README.md)。

| 文档 | 内容 |
|---|---|
| [kimi-k3-latent-moe-ep.md](./models/kimi-k3-latent-moe-ep.md) | K3 latent MoE + EP 架构分析 |
| [kimi-k3-layer-structure.md](./models/kimi-k3-layer-structure.md) | K3 层结构（93 层 / KDA×3+MLA×1 / `attn_res_block_size=12`）、PP 代码就绪度、为什么首版必须关 spec 等 |
| [kimi-k3-dspark-pp-pd.md](./models/kimi-k3-dspark-pp-pd.md) | K3 DSpark prompt draft KV、PP stage 所有权、partial-load 显存收益、PD pointer 映射与 DeepEP 语义 |
| [qwen3-8-day0-pr34585.md](./models/qwen3-8-day0-pr34585.md) | Qwen3.8 模型创新点 + #34585 适配全景 |
| [multimodal-end-to-end-flow.md](./models/multimodal-end-to-end-flow.md) | 多模态请求端到端链路（Qwen3.6-35B-A3B） |

## npu —— Ascend 平台

| 文档 | 内容 |
|---|---|
| [npu-graph-runner.md](./npu/npu-graph-runner.md) | NPU Graph 机制：`cpu_update_input`、属性名随架构/forward 模式变化、与 CUDA Graph 的 4 点差异、完整工作流 |
| [overlap-schedule.md](./inference/overlap-schedule.md) | Overlap schedule **被自动关掉的 6 个条件**（PP / sparse head / diffusion / mamba / MPS / CPU+spec）、PP 与 PD 分离的区别、与 NPU `OVERLAP_PLAN_STREAM` 的区别 |
| [deepep-ascend-npu.md](./npu/deepep-ascend-npu.md) | `--moe-a2a-backend` 语义、两种 dispatch、**三条机制性约束**（A3 专家上限 1024 / low-latency 窗口公式 / graph bs 对齐基数） |

> 该文原有的「三个坑」已拆为独立 case，见
> [`debugging/cases/`](../debugging/cases/)：`507015` UB 越界、`HCCL_BUFFSIZE too SMALL`、graph bs 对齐。
> 原理留在本文，故障过程留在 case，两边用链接互指。

## security

| 文档 | 内容 |
|---|---|
| [cve-2026-15969.md](./security/cve-2026-15969.md) | SafeUnpickler 黑名单绕过原理、无鉴权利用条件、PoC |

---

## 维护约定

1. 新文档先问：**这是「为什么」还是「怎么做」还是「怎么修」**。只有「为什么」进这里。
2. 长文顶部的「太长不看 / 结论先行」保留 —— 原理文也要能 30 秒扫完。
3. 一篇文档只讲一个主题；跨主题用相对链接互指，不要复制段落（会腐烂）。
