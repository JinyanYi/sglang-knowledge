# Kimi-K3 层结构与 PP 代码就绪度

> **本文只讲「是什么 / 为什么」。**
> 切点怎么选、裁几层、起服务用什么参数 →
> [`runbooks/k3-pd-pp/pp-partition-and-cut.md`](../../runbooks/k3-pd-pp/pp-partition-and-cut.md)
> 与 [`runbooks/k3-pd-pp/SKILL.md`](../../runbooks/k3-pd-pp/SKILL.md)。
>
> 审计基准：分支 `0728_dspark` @ `e0263a9`（2026-08）。代码位置以该分支为准。

---

## 1. 层结构（满血 93 层）

| 项 | 值 |
|---|---|
| `num_hidden_layers` | **93** |
| 注意力模式 | 每 4 层：`KDA×3 + MLA×1`（config 里层号 **1-based**） |
| `full_attn_layers` | `4,8,12,…,92,93`（24 个 MLA） |
| `kda_layers` | 其余 69 个 |
| MoE | `first_k_dense_replace=1`，`moe_layer_freq=1` → 几乎每层 MoE |
| Experts | `num_experts=896`，`num_experts_per_tok=16`，`num_shared_experts=2` |
| **`attn_res_block_size`** | **12** |
| DSA / index topk | **无** |

0-based 前 12 层示例：`KDA, KDA, KDA, MLA` 重复 3 次，正好一个 `attn_res` block。

### 两个关键含义

**`attn_res_block_size=12` 不是普通 Transformer residual。**
它是 K3 特有的 **block residual**：每 12 层才做一次 residual 归并。
这直接决定了 PP 切点必须落在 12 的倍数上（见 runbook 的切点表）。

**没有 DSA 跨 stage 元数据。**
不像 DeepSeek-V3 的 DSA 需要跨 PP stage 传 `topk_indices`；
K3 的 KDA cache / MoE 都是**逐层本地状态**，跨 stage 只需传 activation + residual。
→ **断点没有特殊的架构障碍**，唯一的约束就是 `attn_res_block_size=12` 的对齐。

---

## 2. PP 代码就绪度（`kimi_k3.py`）

### 2.1 已具备

| 能力 | 位置 |
|---|---|
| `get_pp_group` / `make_layers` / `start_layer` / `end_layer` | `KimiLinearModel.__init__` ~L1104–1129 |
| 非本 rank `PPMissingLayer`（embed / norm / lm_head） | 首末 rank 分支 |
| `PPProxyTensors({hidden_states, residual})` | 非 last rank return ~L1226 |
| 外层 `KimiK3ForCausalLM` / VLM wrapper 透传 `start_layer`/`end_layer` | ~L1327+ |

对照 `kimi_linear` / `deepseek_v2`：**语言主干 PP 骨架已齐，不是从零接 PP。**

**`PPProxyTensors` 传的是 `hidden_states` + `residual`（即 block residual）** →
跨 stage 功能上可切。

### 2.2 机制性约束：为什么这些必须关

这是**代码事实**，不是偏好。每条的「为什么」比「关掉」这个动作更重要。

| 项 | 结论 | 为什么 |
|---|---|---|
| **DSPARK / speculative** | 首版 CPP **关** | 现网开着，但 PP+spec 变量多，且部分路径**强制 draft 在 pp0** |
| **PD-Multiplexing (`enable_pdmux`)** | **不要开** | 断言 `pp_size==1` 且不能 disagg。要 PD 就用真正的 `--disaggregation-mode prefill/decode` |
| **PD disaggregation + PP** | 高风险，先裁层 + 小 pp | Ascend/Mooncake KV 传输路径有 `pp_size > 1` 分支（按当前 PP stage 的层切 KV）。**框架声称支持，但未充分验证** |
| **DeepEP** | 可保留 | 与 TP/EP 同开；**PP 只传 activation，EP 仍在 stage 内**。OOM 时再降 `HCCL_BUFFSIZE` |
| **双流 multi-stream** | 首版关 | 与 CPP **正交**，关掉只为减少噪声 |
| **dynamic chunking** | **第二步**再开 | 需 `pp_size>1`；先固定 `--chunked-prefill-size` |

> **附带后果（不是可选项）**：`pp_size > 1` 时框架会**自动**关掉 overlap schedule，
> 启动时打 `Pipeline parallelism is incompatible with overlap schedule.`。
> **PP 会关、PD 分离不会** —— 完整条件见
> [`inference/overlap-schedule.md`](../inference/overlap-schedule.md)。

### 2.3 一个已知噪音

`init_zbal_on_npu` 在 `pp_size>1` 时会打 **error log**（要求 zbal mix mode），
**不一定硬退出**。若 PP 起不来再查 `SGLANG_ZBAL_LOCAL_MEM_SIZE` / mix alloc。

> 这是典型的「看起来像致命错误但其实不是」——别被这条 log 误导去改无关的东西。

---

## 相关

- [`runbooks/k3-pd-pp/pp-partition-and-cut.md`](../../runbooks/k3-pd-pp/pp-partition-and-cut.md) — 切点与裁层怎么定
- [`runbooks/k3-pd-pp/SKILL.md`](../../runbooks/k3-pd-pp/SKILL.md) — 起服步骤
- [`debugging/pp-divergence/`](../../debugging/pp-divergence/SKILL.md) — PP 结果对不上时怎么定位
