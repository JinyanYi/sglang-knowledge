# Kimi K3 Latent MoE 与 EP 通信详解

> 模型：Kimi K3（Moonshot，2026-07 开源）
> 结构相关：hidden=7168，latent=3584，896 个路由专家 top-16（激活率 1.8%），2 个共享专家，专家 intermediate=3072
> 代码位置：[`kimi_k3.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py)（下文省略 `python/sglang/` 前缀）
>
> **太长不看**：
> 1. Latent MoE 的收益不是"总参数更大"，而是**同样算力/带宽下每次 token 可调用的专家组合变多**（NVIDIA 是 E 和 top-k 同时 ×d/ℓ，专家总参数基本守恒），同时 a2a 通信量、权重加载带宽除以 d/ℓ；
> 2. EP（专家并行）必然要 all-to-all：路由把 token 分给了分布在所有 rank 上的专家，token 必须按专家归属"搬家"，这就是 dispatch（去程 a2a）+ combine（回程 a2a）；
> 3. K3 的 SP-MoE 用 **reduce-scatter** 把 o_proj 归约和 MoE 计算重叠起来，MoE 前端计算除以 attn_tp；
> 4. K3 的专家路由 = `MoEGate`（GEMM 出分数）→ `moe_fused_gate`（sigmoid + 训练好的 bias + top-16）→ FusedMoE（dispatch/GEMM/combine）。

---

## 目录

1. [技术演进：为什么会有 Latent MoE](#一技术演进为什么会有-latent-moe)
2. [前置知识：四个并集操作（Collectives）](#二前置知识四个并集操作collectives)
3. [为什么 EP 必须用 all-to-all](#三为什么-ep-必须用-all-to-all)
4. [K3 Latent MoE 的结构（算法 + 代码映射）](#四k3-latent-moe-的结构算法--代码映射)
5. [sglang 里的执行路径](#五sglang-里的执行路径)
6. [K3 vs Qwen：MoE 代码对照](#六k3-vs-qwenmoe-代码对照)
7. [激活函数背景：SwiGLU → SiTU-GLU](#七激活函数背景swiglu--situ-glu)
8. [量化路径](#八量化路径)
9. [关键代码索引](#九关键代码索引)

---

## 一、技术演进：为什么会有 Latent MoE

### 1.1 三代 MoE

| 代 | 代表 | 核心改动 | 解决的问题 |
|----|------|---------|-----------|
| 经典 MoE | Shazeer 2017 → Switch → Mixtral → DeepSeek-V3 | 总参数与激活算力解耦，专家越多知识容量越大 | 参数多、算力可控 |
| Latent MoE | NVIDIA Nemotron 3（arXiv 2512.20856，2025-12） | 专家在**低维潜空间**计算（token 先 down-proj 再进专家） | EP 的 a2a 通信 + 权重带宽瓶颈 |
| Stable LatentMoE | Kimi K3（2026-07） | 896 专家 top-16（稀疏度 56×），加 RMSNorm + SiTU-GLU + Quantile Balancing | 极端稀疏下的训练稳定性 |

### 1.2 NVIDIA Latent MoE 的机制与"收益错位"

- **机制**：token 经共享降维矩阵投影到潜空间 ℓ（通常 d/4），**路由和专家 FFN 都在潜空间算**，再 up-proj 还原。
- **收益不是"总参更大"**：NVIDIA 的做法是专家数 E 和 top-k **同时放大 d/ℓ 倍（约 4×）**。因为单个专家输入维小了 4 倍，单专家参数/算力也小 ~4 倍 → 4×专家 × 1/4 尺寸，**专家总参数基本守恒**。真正变的是：
  - 每个 token 的**专家组合空间呈指数级变大**（多样性）；
  - **Accuracy per Byte / per FLOP** 上升；
  - **EP 推理的 a2a 流量和专家权重加载带宽正比于 token 维 d，降维后直接 ÷d/ℓ**——这才是它真正的工程靶子。
- 哲学上和 MLA 同一套：**凡是一个 token 要跟大量候选（KV 头/专家）打交道，先降维再算。**

### 1.3 K3 "Stable" 两个字来自哪

896 专家、激活率 1.8%，稀疏度 56×，放大了两个训练问题：

1. **激活爆炸**：routed 路径 ≈ 4 次连续矩阵乘，2.8T 规模下中间激活易爆。对策：专家聚合与 up-proj 之间插 **RMSNorm** + 用 **SiTU-GLU** 替换 SwiGLU（输出有界，见[第七节](#七激活函数背景swiglu--situ-glu)）。
2. **负载失衡**：896 专家远超传统负载均衡能处理的规模。对策：**Quantile Balancing**（训练期每步从路由分数分位数一步解出每个专家的 bias，替代 DeepSeek-V3 的固定步长微调）。

> ⚠️ 概念澄清：K3 的难点**不是"选谁当专家"**（top-16/896 的排序计算量可忽略），而是"选了之后怎么稳定、均衡、不浪费"。
> - 路由训练稳定 → RMSNorm + SiTU（训练期）
> - 路由负载均衡 → Quantile Balancing（训练期）
> - 跨几百张卡的系统级负载 → MoonEP（训练）/ EPLB、Waterfill（推理，sglang）

---

## 二、前置知识：四个并集操作（Collectives）

P 个 rank，每个持有自己那份数据。四个操作的区别只在于"谁把什么发给谁、发完手里剩什么"。

### 2.1 All-Reduce（全归约）

```
每个 rank 有部分值 → 求和 → 每个 rank 拿到相同的完整总和
  R0: a ──┐
  R1: b ──┼── SUM(a,b,c,d) ──→ R0,R1,R2,R3 都得到总和
  R2: c ──┤
  R3: d ──┘
```
用途：TP 行并行输出归约；K3 非 EP 时 latent 空间专家输出的归约；共享专家 TP 切分的部分和。

### 2.2 All-Gather（全收集）

```
每个 rank 持有全量向量的 1/P 分片 → 每个 rank 拼出完整向量
  R0 有 x[0:2]，R1 有 x[2:4]，R2 有 x[4:6]，R3 有 x[6:8]
  → 四个 rank 都得到 x[0:8]
```
用途：把按 rank 切分的 batch/向量拼回完整（K3 SP-MoE 的 MoE 尾部 `_sp_all_gather_rows`）。**"完整 batch"= 全部 token × 全宽度（如 [B, 7168]）被复制 P 份，通信冗余 P 倍。**

### 2.3 Reduce-Scatter（归约-散射）

```
每个 rank 持有完整向量 → 求和 → 每个 rank 只拿回总和的一份分片
  R0 持有 a、b、c、d → 归约后 R0 拿 Σ[0]，R1 拿 Σ[1]，R2 拿 Σ[2]，R3 拿 Σ[3]
```
用途：只需要归约后的一份分片时省一半通信。它是 all-reduce 的前半段：`all-reduce = reduce-scatter + all-gather`。**K3 的 SP-MoE 就是它（见[5.4](#54-d-sp-moesp_moe)）。**

### 2.4 All-to-All（全交换）

```
每个 rank 向所有其他 rank 发送【不同的】数据，也从所有其他 rank 接收各自不同的数据
  R0 → R1: 数据A1     R1 → R0: 数据B1
  R0 → R2: 数据A2     R2 → R0: 数据B2
  ...（每对 rank 之间交换的内容都不同）
```
本质是"矩阵转置/重排"：数据按目标 rank 分区后全网换手。**专家并行（EP）的 token dispatch 就是它。**

**一句话区分**：all-reduce / all-gather 是把同一个完整对象分发/复制给所有人；reduce-scatter 是归约后每人留一份分片；all-to-all 是每对 rank 交换内容不同的数据。

---

## 三、为什么 EP 必须用 all-to-all

EP 的玩法：**把 896 个专家的权重物理分到 P 个 rank 上**（每 rank 持有 896/P 个专家的完整权重）。

```
router 给每个本地 token 选出 top-16 专家
→ 这些专家大多在【别的 rank】上
→ 计算前：把"要被专家 e 处理的 token"送到持有 e 的 rank   ← dispatch（a2a）
→ 每个 rank 用本地专家算完
→ 计算后：把部分和送回 token 原来的 rank 拼回去            ← combine（a2a）
```

token 按专家归属"各走各路"，天然就是 a2a 的通信形态。

**为什么不直接用 all-gather**：all-gather 会把完整 batch 复制给每个 rank（冗余 P 倍）；a2a 只发"对方真正需要的 token"，通信量小得多。

**为什么不全放专家权重**：那就是 TP 切专家（`--moe-a2a-backend none`），见下表——两种取舍：

| | 权重怎么放 | token 怎么走 | 通信量瓶颈 |
|---|---|---|---|
| TP 专家（a2a=none） | 每 rank 全专家 1/TP 切片 | token 不动，专家输出 all-reduce | 专家权重**加载带宽**（每 rank 读全量专家权重） |
| EP（deepep/megamoe） | 每 rank 只放本地专家 | **a2a dispatch + combine** | a2a 流量 ∝ token × latent 维 |

**Latent MoE 和 EP 是互相成就的**：EP 的 a2a 流量正比于 token 维 d，K3 把专家空间压到 3584（d/2），NVIDIA 压到 d/4 → a2a 流量直接减半 / 减到 1/4。

sglang 落点：[`create_moe_dispatcher`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L137-L178) —— `a2a=none` 用 `StandardDispatcher`（TP 权重切分），`deepep/megamoe` 用 DeepEP / MegaMoE dispatcher（a2a 搬 token）。

### 3.1 Dispatch 的细节

- 发的是**副本**，不是"搬走"：token 原数据留在原 rank（combine 要用），发出去的是 latent 向量副本；
- 一条 token 的 top-16 专家分布在多个 rank → 副本**多播**给所有持有其专家的 rank；
- 工程上不是逐条发，而是**批量置换（permutation）+ a2a**：router 先算完全局分配，按目标 rank 分组一次发（DeepEP 的 `permute_fusion`）；
- combine 回程把部分和送回 token 归属 rank 累加；
- **EP a2a 模式下 combine 返回的已是全量求和 → 不需要再 all-reduce**（K3 里 `_routed_needs_reduce = tp_size > 1 and a2a == none`）；
- 共享专家在 EP 下改为 **tp1 复制**（每个 rank 完整权重，本地算，不进 a2a）。

---

## 四、K3 Latent MoE 的结构（算法 + 代码映射）

### 4.1 算法

```
h (7168)
  → z = W_down · h                        [3584]    # ① 降维（每层共享一套）
  → z = RMSNorm(z)                                 # ② 中间 norm（防激活爆炸）
  → router_logits = W_g · z                        # ③ 路由在 latent 上跑
  → top-16 选专家（sigmoid 打分 + bias）
  → 专家：gate_up = W1_e·z → SiTU → W2_e·act       # ④ 专家输入维 = 3584
  → combine + latent all-reduce（TP 时）           # ⑤ 必须在 norm 之前：sum(norm(x)) ≠ norm(sum(x))
  → z = RMSNorm(z)                                 # ⑥ 聚合后再 norm
  → out = W_up · z                        [7168]    # ⑦ 升维
  → + shared_experts(h)                            # 共享专家仍在原始 hidden 空间
```

### 4.2 代码映射（`KimiK3MoE`，[`kimi_k3.py#L354`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L354)）

| 算法部件 | 代码 | 说明 |
|---|---|---|
| `W_down` / `W_up` | `routed_expert_down_proj` / `routed_expert_up_proj`（[`kimi_k3.py#L541-L560`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L541-L560)） | **`ReplicatedLinear`（每 rank 完整权重）**：W_up 要输出完整 7168，latent 归约必须在 norm 之前 |
| 中间 norm | `routed_expert_norm`（`latent_moe_use_norm` 时启用） | RMSNorm(3584) |
| 路由 | `self.gate = MoEGate(...)`（[`kimi_k3.py#L390`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L390)） | 在 latent 上打分 |
| 专家 | `get_moe_impl_class(...)(hidden_size=self.moe_hidden_size, ...)`（[`kimi_k3.py#L411-L413`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L411-L413)） | FusedMoE，`gate_up_interleaved=False` |
| 共享专家 | `shared_experts = KimiK3MLP(...)` | 在原始 7168 空间 |

**gate_up 与两组 down/up（易混点）**：

| | 作用 | 形状 | 代码 |
|---|---|---|---|
| `W_down` / `W_up`（全局共享） | 7168 ↔ 3584 升降维 | [3584,7168] / [7168,3584] | `routed_expert_down/up_proj` |
| `W1_e` / `W2_e`（每专家各一套） | 3584 → 中间 → 3584 | [2·3072,3584] / [3584,3072] | FusedMoE 的 w1/w2/w3（[`kimi_k3.py#L2730-L2735`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L2730-L2735)） |

### 4.3 选专家的端到端链路（推理侧）

```
MoEGate.forward                    [deepseek_v2.py#L494-L548]
  └─ GEMM：h @ W_g^T → raw logits [N, 896]（fp32）  ⚠️ 不加 bias

TopK（moe_fused_gate kernel）      [topk.py#L1575] → [moe_fused_gate.py#L143-L146]
  └─ scores = sigmoid(logits)
     biased = activated + bias[None,:]     ← bias 只影响排序
     → 16 轮 argmax 取 top-16              ← 权重用 bias-free 的 activated
  └─ K3 专属加速：moe_route_radix.route_radix（原生 CUDA radix-select，比 triton 快 3.1-3.5x，bit-identical）

FusedMoE                                [fused_moe_triton/layer.py#L1404]
  └─ dispatch → grouped GEMM（w1→SiTU→w2）→ combine
```

> 为什么 K3 走 triton `moe_fused_gate` 而不是 flashinfer：`fused_topk_deepseek` 要求 `num_experts<=384 且 topk<=8`，JIT grouped kernel 要求 `experts<=512 且 topk<=8`，896/16 都超了。
>
> **bias 的来源**：训练期 Quantile Balancing 每步解出、冻结在 checkpoint 里的 `e_score_correction_bias`（896 个 float32/层，[`deepseek_v2.py#L483-L485`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v2.py#L483-L485) 声明，[`kimi_k3.py#L2842-L2852`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L2842-L2852) 从 checkpoint 填入）。

---

## 五、sglang 里的执行路径

`KimiK3MoE.forward` 按配置分派（[`kimi_k3.py#L1164`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L1164)）：

### 5.1 A. unfused（兜底）

三个 GEMM 分开（shared gate_up、gate、down_proj 各读一遍 h），`experts(...)` → `_reduce_latent`（TP all-reduce + norm）→ `up_proj` → `_add3(out, shared, prefix_sum)`。SBO 时共享专家放侧流与 a2a 重叠。

### 5.2 B. fused front（plain TP，a2a=none）

`_merge_front_weights` 把 shared.gate_up + gate + latent down_proj 合成一个大权重 `[H, gu+E+latent]`，**一次 GEMM 出三个 slice**（[`_forward_fused`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L1033-L1162)）：
- 共享专家 down 输出与 MoE topk 求和落进**同一个对称 buffer**（`k3_ar_fusion.symm_buffer(MOE_LATENT_SHARED)`）→ **一次 fused all-reduce** 全归约（可把 RMSNorm 融进 AR epilogue，要求 latent=3584=NORM_DIM、hidden=7168=2·NORM_DIM）；
- up_proj 尾段在 decode 尺寸下换 `gemm_ag_up_proj`（[`kimi_k3.py#L1131`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L1131)）：每 rank 只算 1/8 列 GEMV → 组播 all-gather → 与 shared 输出 `add3`（约 1.5-2x）。

### 5.3 C. EP a2a（megamoe / deepep）

不做 DP gather：每 rank 直接 dispatch 本地 token（或 SP shard），每个全局 token 恰好 dispatch 一次。共享专家改 tp1 复制。`_use_mega_moe` 时走 `deep_gemm.fp8_fp4_mega_moe`：**a2a 分发 + grouped GEMM + SiTU + combine 一步到位**。SiTU 靠 `activation_clamp=0.03125` 哨兵值选中（DeepGEMM 里 β=4.0/25.0 是烧进去的，checkpoint 必须恰好等于这俩值，否则 assert 失败）。

### 5.4 D. SP-MoE（`_sp_moe`，[`kimi_k3.py#L1922-L1926`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L1922-L1926)）

条件：`EP a2a + attn_tp>1`。核心是 **reduce-scatter**：

```
attention 输出（全 batch，TP 部分和）
  → o_proj reduce_results=False（不自己做归约）
  → k3_sp_collective.reduce_scatter_res（reduce-scatter，attn-res 残差折进 epilogue）
每个 rank 持有 1/attn_tp 的 token shard
  → MoE 整块在 shard 上跑（latent down/gate/tp1 共享专家/a2a dispatch）
  → _sp_all_gather_rows
全 batch 拼回
```

收益：MoE 前端计算除以 attn_tp；共享专家 tp1 省 all-reduce；a2a 无 attn_tp 倍冗余。`k3_sp_collective`（[`k3_sp_collective.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/k3_sp_collective.py)）只在 SM103 + TP4/8 + megamoe/deepep + CARv2 multicast 下启用，否则退 NCCL。

### 5.5 通信图对比

| 模式 | attention 后 | MoE 区域 | 尾部 |
|---|---|---|---|
| plain TP（fused front） | o_proj 全 reduce | 每 rank 全 batch，latent+shared 一次 fused AR | up_proj(复制) + add3 |
| EP a2a (megamoe) | o_proj 全 reduce | a2a dispatch 本地 token，共享专家 tp1 | up_proj + add3 |
| SP-MoE | o_proj **reduce-scatter**（[`kimi_k3.py#L2049-L2080`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L2049-L2080)） | shard 上 a2a dispatch | all-gather + add3 |

---

## 六、K3 vs Qwen：MoE 代码对照

**Qwen**（[`qwen2_moe.py#L304-L311`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2_moe.py#L304-L311) 构造 + [`L503-L511`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2_moe.py#L503-L511) 前向）：

```python
self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, ...)
self.experts = get_moe_impl_class(quant_config)(
    ..., hidden_size=config.hidden_size,        # ← 专家直接吃 hidden
    intermediate_size=config.moe_intermediate_size, ...)

def _forward_router_experts(self, hidden_states):
    router_logits, _ = self.gate(hidden_states)      # 对 h 路由
    topk_output = self.topk(hidden_states, router_logits)
    return self.experts(hidden_states, topk_output)  # 专家直接吃 h
```

**K3**（[`kimi_k3.py#L541-L560`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L541-L560) 构造 + `_forward_unfused`）：

```python
self.gate = MoEGate(config, ...)                      # 带 e_score_correction_bias
self.routed_expert_down_proj = ReplicatedLinear(hidden_size, self.moe_hidden_size, ...)
self.routed_expert_norm = RMSNorm(self.moe_hidden_size, ...)  # latent_moe_use_norm
self.experts = get_moe_impl_class(...)(
    ..., hidden_size=self.moe_hidden_size,            # ← 专家输入维 = latent！
    intermediate_size=config.moe_intermediate_size, ...)
self.routed_expert_up_proj = ReplicatedLinear(self.moe_hidden_size, hidden_size, ...)

routed_input, _ = self.routed_expert_down_proj(hidden_states)   # 先降维
expert_output = self.experts(routed_input, topk_output)         # 专家在 latent 算
latent = self._reduce_latent(expert_output)                     # latent 归约 + norm
out, _ = self.routed_expert_up_proj(latent)                     # 再升维
return _add3(out, shared_output, prefix_sum)
```

**核心差异**：多了 `W_down`/`W_up` + 中间 norm；专家和路由输入维从 7168 变 3584；激活 SiLU→SiTU；路由 gate 输入是 z 不是 h；K3 有 `_merge_front_weights` 的 GEMM 合并优化。专家算子本身（FusedMoE）两家共用同一套基础设施。

---

## 七、激活函数背景：SwiGLU → SiTU-GLU

### 7.1 GLU 家族速记

普通 FFN：`out = W₂·σ(W₁·x)`（σ 固定）。GLU 家族：**门控分支本身是学出来的投影**。

```
SwiGLU:  out = W_d · ( SiLU(W_g·x) ⊙ W_u·x )
                    └─gate开关─┘  └─up原始信号─┘
```

- `up` = 普通 FFN 里 `W₁·x` 的角色（"要被门控的原始信号"）；`gate` = 额外多学的开关；`down` = 回收投影；
- 三个可学习投影：`W_g`（[m,d]）、`W_u`（[m,d]）、`W_d`（[d,m]）；工程上 W_g/W_u 合并成 `[2m,d]`（gate_up_proj）；
- 名字来源：LLaMA checkpoint 里就叫 `gate_proj/up_proj/down_proj`，`up` 沿用了"第一层投影把维度升上去"的叫法，**不代表维度方向**（gate 和 up 都是 d→m）；
- 为什么中间维取 ~2/3：多一个 `W_u`，同样宽度参数多 50%，Shazeer 论文发现取 2/3 参数守恒。

### 7.2 SiTU-GLU（K3）

```python
# layers/activation.py#L184-L209，SituAndMul
gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)   # SiTU
up   = linear_beta * torch.tanh(up / linear_beta)             # 软截断（保险丝）
return gate * up
```

- K3 参数：β₁=4.0、β₂=25.0（[`kimi_k3.py#L411-L413`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L411-L413) 传给 FusedMoE 的 `gemm1_alpha` / `gemm1_clamp_limit`）；
- 数学性质：`β·tanh(x/β)` 原点附近≈线性（梯度不饱和），大值被钳在 ±β；β₁·β₂=100 → **这条链路的输出永远不会超过 100**；
- 目的：解决"激活爆炸"（Stable LatentMoE 的第一根保险丝）；
- 与 SwiGLU 的关系：同门控结构，只是把 SiLU 换成 SiTU 并对 up 也做软截断；
- DeepGEMM 的 MegaMoE 路径里 SiTU 是烧进 kernel 的（β=4.0/25.0 硬编码 + `activation_clamp=0.03125` 哨兵），所以 checkpoint 值必须精确等于 4.0/25.0。

---

## 八、量化路径

| 路径 | 说明 |
|---|---|
| **MXFP4（官方）** | `Mxfp4Config(is_checkpoint_mxfp4_serialized=True)`，runner 自动选 `flashinfer_mxfp4`（trtllm-gen，需 SiTU cubin pool，见 [`overrides.py#L505-L546`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/overrides.py#L505-L546)）；`route_quant_handoff` 把"路由 + 打包 + 量化"fuse 进 fused-front 预发射 |
| **w4a8（社区部署版）** | 大概率 marlin/trtllm 的 int4-group + int8 activation MoE 路径（`--moe-runner-backend marlin`），attention/共享专家保持 bf16；未拿到 config 前不下定论 |

注意事项：K3 checkpoint 是 `gate_up_interleaved=False`（w1/w3 非交错布局）；非 flashinfer_mxfp4 的 runner 需要 dense buffer（`moe_front_needs_contiguous`）。

---

## 九、关键代码索引

| 主题 | 位置 |
|---|---|
| KimiK3MoE 定义 | [`kimi_k3.py#L354`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L354) |
| down/up/norm 构造 | [`kimi_k3.py#L541-L560`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L541-L560) |
| 专家激活参数（SiTU） | [`kimi_k3.py#L411-L413`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L411-L413) |
| TopK 构造（含 bias） | [`kimi_k3.py#L424-L450`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L424-L450) |
| MoEGate（GEMM + bias 参数） | [`deepseek_v2.py#L455-L548`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v2.py#L455-L548) |
| 选专家 kernel（sigmoid+bias+topk） | [`moe_fused_gate.py#L143-L219`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/moe/moe_fused_gate.py#L143-L219) |
| K3 radix-select 加速 | [`moe_fused_gate.py#L293-L315`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/moe/moe_fused_gate.py#L293-L315) |
| fused front（合并 GEMM） | [`kimi_k3.py#L1033-L1162`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L1033-L1162) |
| gemm_ag_up_proj | [`kimi_k3.py#L1131`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L1131) |
| EP dispatcher 选择 | [`layer.py#L137-L178`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L137-L178) |
| SP-MoE 判定 | [`kimi_k3.py#L1922-L1926`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L1922-L1926) |
| SP-MoE 的 reduce-scatter | [`kimi_k3.py#L2049-L2080`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L2049-L2080) / [`k3_sp_collective.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/k3_sp_collective.py) |
| SituAndMul | [`activation.py#L184-L209`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/activation.py#L184-L209) |
| 专家权重映射（w1/w2/w3） | [`kimi_k3.py#L2730-L2735`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py#L2730-L2735) |

---

## 相关文档

- Quantile Balancing（训练侧背景）：待写
- Kimi K3 架构全景 / KDA / 启动链路：待写
