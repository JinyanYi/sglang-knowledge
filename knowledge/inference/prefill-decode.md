# Prefill vs Decode：LLM 推理的两个阶段

> 本文专注于 Prefill 和 Decode 阶段本身的工作原理、计算特性差异，以及为什么需要 PD 分离。
> CUDA Graph 相关内容已迁移至：[cuda-graphs.md](./cuda-graphs.md)
> MTP 相关内容请见：[multi-token-prediction.md](./multi-token-prediction.md)

---

## 目录

1. [Transformer Decoder 背景](#1-transformer-decoder-背景)
2. [Prefill 阶段：处理 Prompt](#2-prefill-阶段处理-prompt)
3. [Decode 阶段：逐 Token 生成](#3-decode-阶段逐-token-生成)
4. [KV Cache 详解](#4-kv-cache-详解)
5. [Prefill vs Decode 核心区别](#5-prefill-vs-decode-核心区别)
6. [Arithmetic Intensity：计算密度的鸿沟](#6-arithmetic-intensity计算密度的鸿沟)
7. [为什么需要 PD 分离](#7-为什么需要-pd-分离)
8. [总结](#8-总结)

---

## 1. Transformer Decoder 背景

### Decoder 的本质："预测下一个词"的机器

GPT 这类 decoder-only 模型，本质上是一个**"根据前文预测下一个词"**的机器。每个 decoder layer 包含三个核心部分：

```
┌──────────────────────────────────────────────────────────────┐
│                     Decoder Layer                            │
├──────────────────────────────────────────────────────────────┤
│  输入: [x₁, x₂, x₃, ..., xₙ]  (词向量序列)                  │
│                          ↓                                   │
│  ┌──────────────────────┐                                    │
│  │  Masked Self-Attn    │  ← 只能看到前面的词（因果注意力）    │
│  │  (Q, K, V)           │                                    │
│  └──────────────────────┘                                    │
│                          ↓                                   │
│  ┌──────────────────────┐                                    │
│  │    MLP (FFN)         │  ← 非线性变换，提取特征             │
│  └──────────────────────┘                                    │
│                          ↓                                   │
│  输出: [h₁, h₂, h₃, ..., hₙ]  (隐藏状态)                    │
└──────────────────────────────────────────────────────────────┘
```

### 因果注意力（Causal Attention）

位置 i 的 token 只能看到位置 1~i 的 token，看不到未来的 token：

```
对于输入序列 [x₁, x₂, x₃]:
注意力矩阵被 mask 后变成：
        x₁   x₂   x₃
    ┌──────────────────┐
x₁  │  ✓    ×    ×    │  ← x₁ 只能看到自己
x₂  │  ✓    ✓    ×    │  ← x₂ 能看到 x₁ 和自己
x₃  │  ✓    ✓    ✓    │  ← x₃ 能看到 x₁, x₂, 和自己
    └──────────────────┘
```

---

## 2. Prefill 阶段：处理 Prompt

### Prefill 在做什么

Prefill 阶段**一次性处理整个用户输入**（完整 prompt），并为后续 Decode 建立 KV Cache。可以理解为模型的"编码阶段"。

**核心任务**：
1. 处理 N 个 token 的 prompt
2. 计算出所有位置的 Key 和 Value 并缓存
3. 生成第一个输出 token

### 执行流程

```
输入: "今天天气真好" (6 个 token)
         ↓
┌────────────────────────────────────────────────────────────┐
│                    Embedding Layer                         │
│  Token IDs → 词向量矩阵 [1, 6, 4096]                      │
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│              Decoder Layer 1                               │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Q = W_Q(hidden)  → [1, 32, 6, 128]               │   │
│  │  K = W_K(hidden)  → [1, 32, 6, 128]  ← 缓存到 KV  │   │
│  │  V = W_V(hidden)  → [1, 32, 6, 128]  ← Cache      │   │
│  └────────────────────────────────────────────────────┘   │
│         ↓                                                 │
│  CausalAttention(Q, K, V) → [1, 32, 6, 128]              │
│         ↓                                                 │
│  MLP → [1, 6, 4096]                                      │
└────────────────────────────────────────────────────────────┘
         ↓
      ... (逐层处理所有 decoder layers)
         ↓
┌────────────────────────────────────────────────────────────┐
│                    Output Layer                            │
│  取最后一个位置的 logits → 采样 → 第 1 个生成的 token      │
└────────────────────────────────────────────────────────────┘
```

### 关键特征

- **输入规模**：整个 prompt（N 个 token，N 可能 10~100K）
- **计算量**：O(N² × d) —— Attention 计算完整 N×N 矩阵
- **并行度**：高（所有 N 个位置同时计算）
- **内存访问**：写入为主（创建 KV Cache）

---

## 3. Decode 阶段：逐 Token 生成

### Decode 在做什么

Decode 阶段是**逐 token 生成**的循环。每次迭代基于已生成的 token 和历史 KV Cache，生成下一个 token。可以理解为"解码阶段"。

**核心任务**：
1. 输入上一步生成的 1 个 token
2. 读取 KV Cache 中的历史信息
3. 计算新位置的注意力
4. 生成下一个 token，更新 KV Cache

### 执行流程

```
当前状态:
- 已输入: "今天天气真好" (6 个 token)
- KV Cache: 已缓存前 6 个 token 的 K、V

迭代 1（生成第 7 个 token）:
  input = x₆ (第 6 个 token 的 embedding → 1 个 token)
  
  → 经过所有 decoder layers:
     Layer 1:
       Q = W_Q(x₆)           → [1, 32, 1, 128]
       K = [K₁, K₂, ..., K₆]  ← 直接从 KV Cache 读取！
       V = [V₁, V₂, ..., V₆]  ← 直接从 KV Cache 读取！
       Attention(Q, K, V) → 只计算一个位置（1 个 query vs 6 个 KV）
       MLP → [1, 4096]
     Layer 2:
       ...（同样方式处理）
  
  → 采样得到新 token t₇（比如 "，"）
  → KV Cache 更新: [K₁, ..., K₆, K₇], [V₁, ..., V₆, V₇]

迭代 2（生成第 8 个 token）:
  input = x₇ (新生成的 token "，" → 1 个 token)
  → 同样利用已有的 KV Cache（现在包含 7 个 token）
  → 只计算第 8 个位置的输出
  → 采样得到 t₈
  → KV Cache 更新: [K₁,...,K₈], [V₁,...,V₈]

... 循环继续，直到遇到 <EOS>
```

### 关键特征

- **输入规模**：每次 1 个 token
- **计算量**：O(L × d) —— Attention 是 1 个 query 对所有 L 个 KV
- **并行度**：低（必须串行，每次依赖前一步的 token）
- **内存访问**：读取为主（读取整个 KV Cache，追加少量新 KV）

---

## 4. KV Cache 详解

### 为什么需要 KV Cache

在自回归生成中，每次 Decode 都需要重新计算注意力。如果不缓存，每次都要重新计算所有 token 的 K 和 V，造成大量重复计算：

```
每次重复计算（无 KV Cache）:
Forward 1:  K = W_K([t1])
Forward 2:  K = W_K([t1, t2])    ← 重新算了 t1 的 K
Forward 3:  K = W_K([t1, t2, t3])  ← 重新算了 t1, t2 的 K

有 KV Cache:
Forward 1:  K₁ = W_K(t1) → 缓存
Forward 2:  K₂ = W_K(t2) → 缓存，复用 K₁
Forward 3:  K₃ = W_K(t3) → 缓存，复用 K₁, K₂
```

### KV Cache 的存储结构

每个 decoder layer 的 KV Cache 包含两个张量：

```
kv_cache = (key_cache, value_cache)

# key_cache 形状:   [batch_size, num_heads, max_seq_len, head_dim]
# value_cache 形状: [batch_size, num_heads, max_seq_len, head_dim]

# 例如，对于 7B 模型：
# batch_size = 1, num_heads = 32, max_seq_len = 4096, head_dim = 128
# 每个 layer 的 KV Cache 大小: 1 × 32 × 4096 × 128 × 2 × 2bytes ≈ 64 MB
# 32 个 layers: ≈ 2 GB (FP16)
```

### KV Cache 的生命周期

```
Prefill 阶段：
  - KV Cache 是空的（或预分配但未写入）
  - 一次性写入 N 个 slot（整个 prompt 的 K、V）
  - 通过 out_cache_loc 指定写入位置

Decode 阶段：
  - 每次读取全部历史 K、V
  - 追加当前位置的 1 个新 K、V
  - cache_len 持续增长

内存占用随时间增长：
  时间 →  Prefill    Decode-1    Decode-2    Decode-3   ...
  Cache    [N]        [N+1]       [N+2]       [N+3]     ...
```

---

## 5. Prefill vs Decode 核心区别

### 计算特征对比

| 维度 | Prefill | Decode |
|------|---------|--------|
| **输入** | 整个 prompt（N 个 token） | 上一步的 1 个 token |
| **本质** | 编码 prompt，创建 KV Cache | 从 KV Cache 读取，逐 token 生成 |
| **计算方式** | 并行计算所有 N 个位置 | 串行计算 1 个位置 |
| **计算复杂度** | O(N² × d) — N×N 完整注意力 | O(L × d) — 1 个 query 对所有 L 个 KV |
| **并行度** | 高（GPU 可同时处理 N 个位置） | 低（必须等上一步结果） |
| **内存访问模式** | 写入为主（创建 KV Cache） | 读取为主（读取整个 KV Cache） |
| **KV Cache** | 一次性写入 N 个 K、V | 读取全部历史 + 追加 1 个 |
| **Attention 计算** | N×N 矩阵（因果掩码） | 1×L 向量 |

### 计算量随序列长度变化

```
Prefill 计算量: O(N²) 
  N=100:  10,000 单位
  N=1000: 1,000,000 单位
  N=4096: 16,777,216 单位

Decode 计算量: O(L)（L = cache_len）
  L=100:  100 单位
  L=1000: 1000 单位
  L=4096: 4096 单位

差距随序列长度增长急速扩大：
  N=1000 → Prefill 每层 ≈ 378 GFLOPs, Decode 每步 ≈ 0.4 GFLOPs
  差距约 1000 倍！
```

---

## 6. Arithmetic Intensity：计算密度的鸿沟

### Arithmetic Intensity 是什么

Arithmetic Intensity（AI）是衡量一个计算任务"计算密集程度"的核心指标：

```
Arithmetic Intensity = FLOPs / Bytes

- FLOPs: 需要的浮点运算次数
- Bytes: 需要从内存读取/写入的数据量（权重 + 中间结果）

单位: FLOPs/Byte（每读入 1 字节数据，能做多少次计算）
```

**直观理解**：
- **高 AI**（如矩阵乘法）：数据加载到寄存器后，反复使用多次 → **计算瓶颈**
- **低 AI**（如逐元素操作）：数据加载一次，只做一次运算 → **内存瓶颈**

### Roofline 模型

```
性能 (FLOPs/s)
    ↑
    │        ┌──────────────────── compute-bound
    │        │ 性能受算力上限限制  (高 AI)
    │        │
    │        │
    │        │
    │   ─────┘  memory-bound
    │  /        (低 AI) ────── 性能受内存带宽限制
    │ /
    │/
    └──────────────────────────→ Arithmetic Intensity (FLOPs/Byte)
                    ↑
              交叉点 (Ridge Point)
```

- AI **高于** Ridge Point → **Compute-bound**：GPU 算力（TFLOPs）是瓶颈
- AI **低于** Ridge Point → **Memory-bound**：GPU 带宽（GB/s）是瓶颈

### Prefill 的 AI 分析

#### 计算量

每层 Transformer（N=1000, d=4096）：

```
1. QKV 计算: hidden = [N, d] × [d, 3d] → [N, 3d]
   FLOPs = 2 × N × d × 3d = 6Nd²

2. Attention: Q[N, h, d/h] × K.T[d/h, N] → score[N, N]
   FLOPs = 2 × N² × d

3. Attention 输出: score[N, N] × V[N, d] → [N, d]
   FLOPs = 2 × N² × d

4. MLP (FFN):
   FLOPs = 2 × N × d × 4d + 2 × N × 4d × d = 16Nd²

总计（每层）≈ 2N²d + 22Nd² FLOPs
```

代入 N=1000, d=4096：

```
FLOPs ≈ 2 × 10⁶ × 4096 + 22 × 1000 × 4096²
     ≈ 8.2 × 10⁹ + 3.7 × 10¹¹
     ≈ 3.78 × 10¹¹ FLOPs ≈ 378 GFLOPs
```

#### 内存访问量

```
1. 权重加载: model_weights（约 7GB，一次性）
2. 输入/输出: [N, d] × 2 × 2 bytes ≈ 16 MB
3. KV Cache 写入: [N, d] × 2 × 2 bytes ≈ 16 MB

总内存访问 ≈ 7GB + 32MB ≈ 7GB（权重主导）
```

#### 计算强度

```
AI_Prefill = 378 GFLOPs / 7 GB ≈ 54 FLOPs/Byte
```

H100 Ridge Point ≈ 80 FLOPs/Byte，短 prompt 时 Prefill 可能在 Ridge Point 附近徘徊。
但随 seq_len 增长：

```
N=1000:  AI ≈ 54   FLOPs/Byte（混合 bound）
N=2000:  AI ≈ 108  FLOPs/Byte ✅ 进入 compute-bound
N=4096:  AI ≈ 220  FLOPs/Byte → 强 compute-bound
```

**原因**：Attention 的 O(N²) 部分占比随 N 增大而增大，计算量增长远快于内存访问量（O(N)）。

### Decode 的 AI 分析

#### 计算量

每层 Transformer（L=1000, d=4096）：

```
1. Q 计算: q = [1, d] × [d, d] → [1, d]
   FLOPs = 2d²

2. K_new, V_new 计算:
   FLOPs = 2 × 2d² = 4d²

3. Attention: Q[1, d] × K_cache.T[d, L] → score[1, L]
   FLOPs = 2 × L × d

4. Attention 输出: score[1, L] × V_cache[L, d] → [1, d]
   FLOPs = 2 × L × d

5. MLP: FLOPs = 16d²

总计 ≈ 22d² + 4Ld FLOPs
```

```
FLOPs ≈ 22 × 4096² + 4 × 1000 × 4096
     ≈ 3.7 × 10⁸ + 1.6 × 10⁷
     ≈ 3.86 × 10⁸ FLOPs ≈ 386 MFLOPs
```

#### 内存访问量

```
1. 权重加载: model_weights（约 7GB，每次必须重新加载！）
   - 权重在 HBM 中，每次计算前加载到 SRAM
   - H100 L2 Cache 只有 50MB，放不下 140GB 的权重

2. KV Cache 读取: [L, d] × 2 × 2 bytes ≈ 16 MB
3. KV Cache 写入: [1, d] × 2 × 2 bytes ≈ 16 KB

总内存访问 ≈ 7GB + 16MB（权重主导）
```

#### 计算强度

```
AI_Decode = 386 MFLOPs / 7 GB ≈ 0.055 FLOPs/Byte
```

### Prefill vs Decode AI 对比

| 指标 | Prefill (N=1000) | Decode (L=1000) | 差距 |
|------|-----------------|-----------------|------|
| **FLOPs** | 378 GFLOPs | 386 MFLOPs | 1000x |
| **内存访问** | ~7 GB | ~7 GB | ≈1x |
| **Arithmetic Intensity** | 54 FLOPs/Byte | 0.055 FLOPs/Byte | 1000x |
| **瓶颈** | Compute-bound（长序列） | Memory-bound | - |
| **GPU 算力利用率** | > 70% | < 10% | - |

**AI_Decode ≈ 0.055 vs AI_Prefill ≈ 54 → 差距 1000 倍！**

H100 Ridge Point ≈ 80 FLOPs/Byte，Decode 只有 0.055，远低于 Ridge Point → **极度 Memory-bound**。

### 就算 Decode 多个 token，依然是 Memory-bound 吗？

假设一次 decode K 个 token（MTP 或批量解码）：

```
总 FLOPs ≈ K × (22d² + 4Ld)
总内存访问 ≈ 7GB (权重) + K × 16MB (KV read) + K × 16KB (KV write)

对于 K=32, L=1000, d=4096:
FLOPs = 32 × 386 MFLOPs ≈ 12.4 GFLOPs
内存访问 = 7GB + 32 × 16MB ≈ 7.5GB
AI = 12.4 / 7.5 ≈ 1.65 FLOPs/Byte

H100 Ridge Point ≈ 80 FLOPs/Byte

1.65 << 80 → 仍然是 Memory-bound！
```

要达到 compute-bound（AI > 80）需要：

```
FLOPs / Bytes > 80
→ K × 386M / (7G + K × 16M) > 80
→ K 无解！（权重 7GB 的硬瓶颈）
```

**结论**：只要权重必须每次加载，Decode 就无法达到 compute-bound。权重加载是 Decode 的硬瓶颈。

---

## 7. 为什么需要 PD 分离

### 不分离的问题

Prefill 和 Decode 的硬件需求完全不同，放在同一 GPU 上会导致资源浪费：

```
场景：70B 模型，用户请求 prompt=1000 tokens，生成 1000 tokens
单 GPU：

时间线：
[ Prefill 1000 tokens ] → [ Decode 1000 tokens ]
   ~50ms (compute-bound)     ~500ms (memory-bound)

GPU 利用率分析：
- Prefill 阶段：算力利用率 70%，带宽利用率 30%
- Decode 阶段：算力利用率 5%，带宽利用率 60%
- 整体利用率：< 30%
```

### 分离后的优化

```
方案：2 个 GPU，一个专门 Prefill，一个专门 Decode

Prefill GPU（计算优化）：
- 需要高算力（TFLOPs）
- 大 batch size，算力吃满
- 快速生成 KV Cache → 传给 Decode GPU

Decode GPU（内存优化）：
- 需要高带宽（GB/s）
- KV Cache 常驻显存，避免重复传输
- 专注低延迟解码
```


| 维度 | 不分离 | 分离后 |
|------|--------|--------|
| Prefill 吞吐 | 受 Decode 干扰 | 算力独立，吞吐 ↑ 3-5x |
| Decode 延迟 | 受 Prefill 阻塞 | 带宽专注，延迟 ↓ 2-3x |
| 硬件利用率 | < 30% | Prefill > 70%，Decode > 60% |
| 成本 | 同一 GPU 做两件事 | 可用不同硬件做不同事 |

### 业界实践

- **vLLM / SGLang**：支持 PD 分离，Prefill 和 Decode 可以在不同 GPU 甚至不同机器上
- **Google (PaLM)**：Prefill 用 TPU v4，Decode 用 TPU v4 的不同配置
- **DeepSeek**：使用 PD 分离架构，提升整体推理效率

---

## 8. 总结

### 一句话理解

- **Prefill** = 编码 prompt = 处理大批数据 → **Compute-bound**（缺算力）
- **Decode** = 逐词生成 = 频繁读内存 → **Memory-bound**（缺带宽）

### Prefill vs Decode 最终对比

| 维度 | Prefill | Decode |
|------|---------|--------|
| **本质** | 处理 prompt，创建 KV Cache | 从 KV Cache 读取，逐 token 生成 |
| **计算复杂度** | O(N²) — N 是 prompt 长度 | O(L) — L 是 KV Cache 长度 |
| **Arithmetic Intensity** | 高（~54 FLOPs/Byte） | 极低（~0.055 FLOPs/Byte） |
| **瓶颈** | Compute-bound（算力） | Memory-bound（带宽） |
| **GPU 算力利用率** | > 70% | < 10% |
| **优化方向** | 大 batch、矩阵乘法优化 | 减少权重加载、优化 KV Cache 访问 |
| **PD 分离收益** | 算力资源独立调度 | 带宽资源独立调度 |

---

> **关联笔记**：
> - [cuda-graphs.md](./cuda-graphs.md) — CUDA Graph 和 Piecewise CUDA Graph 的详细实现
> - [multi-token-prediction.md](./multi-token-prediction.md) — MTP 推测解码加速原理
> - [speculative-decoding.md](./speculative-decoding.md) — 投机推理方法详解（N-gram、EAGLE、DFlash）
