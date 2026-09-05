# Multi-Token Prediction (MTP) 笔记

> 论文：[Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) (Meta, ICML 2024)
> 参考资料：[知乎 MTP 讲解](https://zhuanlan.zhihu.com/p/15037286337)
> SGLang 文档：[Speculative Decoding](https://docs.sglang.ai/references/speculative_decoding.html)

***

## 目录

1. [传统 Next-Token Prediction 的问题](#1-传统-next-token-prediction-的问题)
2. [MTP 的训练方式](#2-mtp-的训练方式)
3. [MTP 推理加速：Self-Speculative Decoding](#3-mtp-推理加速self-speculative-decoding)
4. [为什么能加速：核心原理分析](#4-为什么能加速核心原理分析)
5. [Transformer 注意力的并行特性](#5-transformer-注意力的并行特性)
6. [SGLang 中的 MTP 实现](#6-sglang-中的-mtp-实现)
7. [性能数据](#7-性能数据)

***

## 1. 传统 Next-Token Prediction 的问题

### 训练阶段的低效性

- **局部性**：只关注预测下一个 token，容易陷入局部模式，难以学习长距离依赖关系和全局语义
- **"硬决策"忽视**：倾向于捕捉简单的局部模式，忽略了那些对生成文本的整体质量有重要影响的"决策点"
- **数据需求大**：为了克服局部性，需要大量的训练数据

### 推理阶段的串行瓶颈

- 生成 N 个 token 需要 N 次 forward pass，无法并行
- 每次 forward 只推进 1 个 token 位置

***

## 2. MTP 的训练方式

### 核心思路

在训练时，让模型一次性预测多个未来 token，而不是仅仅预测下一个 token。

### 训练架构

```
输入: [t1, t2, t3, t4, t5, t6, t7, t8]

                      ┌─────────────────────┐
                      │  共享 Transformer    │  ← 共享主干（Shared Trunk）
                      │    Encoder (主干)    │
                      └──────────┬──────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
             ▼                   ▼                   ▼
        ┌─────────┐        ┌─────────┐        ┌─────────┐
        │输出头 1  │        │输出头 2  │  ...   │输出头 N  │  ← 独立输出头
        └────┬────┘        └────┬────┘        └────┬────┘
             │                  │                  │
             ▼                  ▼                  ▼
        预测 t2              预测 t3           预测 t_{N+1}
```

**关键设计**：

- **共享主干**：模型主体是标准的 Transformer，提取输入的特征表示
- **独立输出头**：每个待预测的 token 都有一个独立的输出头（轻量线性层），并行预测

### 损失函数

```python
# 训练时 N 个输出头的 loss 之和
loss = CE(head1, target_token1) + CE(head2, target_token2) + ... + CE(headN, target_tokenN)
```

### 内存优化

前向和反向传播时，依次计算每个输出头的梯度，而不是一次性计算所有头的梯度，避免同时存储所有输出头的梯度信息，降低 GPU 内存占用。

### MTP 跟主模型一起训练吗？

是的。MTP 论文中是直接修改训练目标，让模型在训练时就学习多 token 预测。但 MTP 模型的参数就是主模型本身（加上额外的小输出头），**不需要额外加载一个独立的 MTP 模型**。

在 SGLang 的实践中，有两种情况：

1. **原生 MTP 模型**：如 Qwen3.5-MTP，模型本身就带有 MTP 模块（额外的小网络 + 输出头），训练时一起训练好的
2. **后训练 MTP 模块**：如 EAGLE 系列，在已训练好的主模型上，额外训练一个轻量级的 draft 模块

***

## 3. MTP 推理加速：Self-Speculative Decoding

### 核心思想

利用 MTP 训练出来的额外输出头，让主模型自己担任"draft 模型"和"验证模型"双重角色：

1. **Draft 阶段**：用 MTP 的多个输出头并行预测多个 token
2. **Verify 阶段**：用主模型的 next-token prediction head 验证预测结果

### 推理流程

```
Step 1: MTP 生成草稿 token
  输入: [context]
  MTP 输出头 1 → 预测 tokenA
  MTP 输出头 2 → 预测 tokenB
  MTP 输出头 3 → 预测 tokenC
  MTP 输出头 4 → 预测 tokenD
  
  Draft tokens: [tokenA, tokenB, tokenC, tokenD]

Step 2: 主模型验证（1 次 forward）
  输入: [context, tokenA, tokenB, tokenC, tokenD]
  
  → 主模型在位置 context+1 的预测 = tokenA ? ✅
  → 主模型在位置 context+2 的预测 = tokenB ? ✅
  → 主模型在位置 context+3 的预测 = tokenC ? ✅
  → 主模型在位置 context+4 的预测 = tokenD ? ❌ (实际预测 tokenE)

Step 3: 接受/拒绝
  ✅ 接受 tokenA, tokenB, tokenC → 白赚了 3 个 token
  ❌ 拒绝 tokenD → 使用主模型预测的 tokenE
  
  输出: [tokenA, tokenB, tokenC, tokenE]
```

### EAGLE 变体

> EAGLE 的具体实现细节已迁移至：[speculative-decoding.md](./speculative-decoding.md)

简要说明：EAGLE 不是在 token 层面做预测，而是在**特征层面**做预测：

1. **特征预测**：draft 模型预测主模型的最后一层 hidden state（特征向量），而不是直接预测 token
2. **特征转 token**：预测出来的特征再经过 LM Head 转成 token
3. **树状扩展**：每个位置扩展 top-k 个候选分支，形成候选树
4. **验证**：主模型一次 forward 验证整个候选树

```
EAGLE draft 模型:
输入: (token_k, hidden_state_k) → 预测 hidden_state_{k+1} → LM Head → token_{k+1}

EAGLE 比直接 MTP 好的原因：
- 特征的分布比 token 的分布更平滑、更可预测（连续空间 vs 离散空间）
- 预测特征比预测 token 更容易学习
```

***

## 4. 为什么能加速：核心原理分析

### 直觉误区

> "主模型验证也需要 forward pass，那和直接生成有什么区别？"

**核心区别在于**：

- 传统生成：**N 个 token → N 次 forward pass**
- MTP 验证：**N 个 token → 1 次 forward pass**

### 数学计算

#### 传统自回归（生成 4 个 token）

```
Forward 1: [context]                   → 得到 tokenA
Forward 2: [context, tokenA]           → 得到 tokenB
Forward 3: [context, tokenA, tokenB]   → 得到 tokenC
Forward 4: [context, tokenA, tokenB, tokenC] → 得到 tokenD

Foward pass 次数 = 4
计算量 ≈ 序列长度平方之和
```

#### MTP 验证（验证 4 个 draft token）

```
Forward 1: [context, tokenA, tokenB, tokenC, tokenD]
→ 一次性得到所有位置的预测，与 draft 对比

Forward pass 次数 = 1
计算量 ≈ (context_len + 4)²
```

#### 加速比计算

设主模型一次 forward = T\_main，MTP 模型一次 forward = T\_mtp ≈ 0.05\~0.1 × T\_main

```
传统方法 4 个 token: 4 × T_main
MTP 方法 4 个 token: T_mtp + T_main

假设接受率 70%:
平均每次生成 4 × 0.7 = 2.8 个 token
每个 token 代价 = (T_mtp + T_main) / 2.8

加速比 = 4 / ((T_mtp/T_main + 1) / 0.7) = 4 / ((0.1 + 1) / 0.7) ≈ 2.5x
```

**关键洞察**：生成 token 需要的 forward pass 次数从 N 次降到了 1 次，这是加速的本质。

### 为什么验证比生成的 forward 还长，反而更快？

传统 4 次 forward 的累计计算量（用注意力复杂度 O(L²) 估算）:

```
1² + 2² + 3² + 4² = 30 单位

1 次验证 forward (长度 = context_len + 4，context 固定不变):
(context_len + 4)² >> 每步积累的平方和？不是的！

实际上传统方法每次 forward 都是完整序列长度:
100² + 101² + 102² + 103² ≈ 41214
验证 1 次: 104² ≈ 10816

MTP 验证 ≈ 传统方法的 1/4 计算量
```

***

## 5. Transformer 注意力的并行特性

### 核心理解：为什么一次 forward 可以验证所有 token

这是理解 MTP 加速的**最关键原理**。

### Transformer 矩阵运算的并行性

与 RNN 不同，Transformer 不是从左到右逐个处理 token，而是**一次性处理整个序列**：

```python
# 输入矩阵 X: [seq_len, hidden_dim]
# 不是逐行处理，而是整个矩阵同时计算

Q = X @ W_q  # 所有位置的 Query 同时计算
K = X @ W_k  # 所有位置的 Key 同时计算
V = X @ W_v  # 所有位置的 Value 同时计算

attention_scores = Q @ K.T  # 注意力分数矩阵一次性计算
# [t1看t1, t1看t2, t1看t3, t1看t4]
# [t2看t1, t2看t2, t2看t3, t2看t4]
# [t3看t1, t3看t2, t3看t3, t3看t4]
# [t4看t1, t4看t2, t4看t3, t4看t4]
```

### 因果掩码不破坏并行性

因果掩码只是限制信息流，但不改变**计算模式的并行性**：

```python
masked_scores = attention_scores + causal_mask
# 位置的 t2 虽然只能看到 t1,t2，但仍然是和其他位置同时算出来的

output = softmax(masked_scores) @ V
# output[0] = f(t1)
# output[1] = f(t1, t2)
# output[2] = f(t1, t2, t3)
# output[3] = f(t1, t2, t3, t4)
# 全部同时算出！
```

### 验证时的应用

```python
# 输入: [context, draft1, draft2, draft3, draft4]
# 一次 forward

logits = model.forward(extended_sequence)  # [L+4, vocab_size]

# 同时得到每个位置的预测
pred_at_context_end = logits[L-1]    # 验证 draft1 是否匹配
pred_at_draft1      = logits[L]      # 验证 draft2 是否匹配
pred_at_draft2      = logits[L+1]    # 验证 draft3 是否匹配
pred_at_draft3      = logits[L+2]    # 验证 draft4 是否匹配

# 所有验证同时完成
is_correct = (pred_at_draft1.argmax() == draft1) &
             (pred_at_draft2.argmax() == draft2) &
             ...
```

**这就是 MTP 加速的底层原理**：Transformer 的并行特性使得一次性验证多个 token 成为可能。

***

## 6. SGLang 中的 MTP 实现

### 支持的算法

| 算法         | 描述                   | 是否需要独立 draft 模型 |
| ---------- | -------------------- | --------------- |
| MTP        | 原生多头预测（模型自带）         | 不需要（内置在模型中）     |
| EAGLE-2    | 特征级别预测 + 树扩展         | 需要额外训练          |
| EAGLE-3    | EAGLE-2 改进版，去掉特征预测目标 | 需要额外训练          |
| STANDALONE | 小 LLM 作为 draft 模型    | 需要              |

> 完整的投机推理方法对比（包括 N-gram、DFlash）详见：[speculative-decoding.md](./speculative-decoding.md)

### MTP 模型架构（以 Qwen3.5-MTP 为例）

```python
class Qwen3_5ForCausalLMMTP(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        # 融合层：将输入嵌入和主模型隐藏状态融合
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.pre_fc_norm_embedding = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # MTP 模型通常只有 1 层 Transformer
        config.num_hidden_layers = 1
        self.model = Qwen3_5ForCausalLM(config, quant_config, is_nextn=True)

    def forward(self, input_ids, positions, forward_batch, **kwargs):
        # 1. 获取输入嵌入
        input_embeds = self.model.embed_tokens(input_ids)
        
        # 2. 获取主模型的隐藏状态（通过 forward_batch 传递）
        hidden_states = forward_batch.spec_info.hidden_states
        
        # 3. 融合输入嵌入和隐藏状态
        input_embeds = self.pre_fc_norm_embedding(input_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
        
        # 4. 通过融合层 + 1 层 Transformer
        hidden_states = self.fc(hidden_states)
        hidden_states = self.model(input_ids, positions, forward_batch, hidden_states)
        
        # 5. 输出 logits
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
```

### 关键参数

| 参数                               | 说明                    | 默认值  |
| -------------------------------- | --------------------- | ---- |
| `--speculative-algorithm`        | 推测解码算法                | 无    |
| `--speculative-num-steps`        | draft 深度（自回归步数）       | 5    |
| `--speculative-eagle-topk`       | 每步扩展的分支数              | 4    |
| `--speculative-num-draft-tokens` | 验证容量（总 draft token 数） | 自动计算 |
| `--speculative-draft-model-path` | draft 模型路径            | None |

### 启动示例

```bash
# MTP 模型（自带 MTP 模块）
python3 -m sglang.launch_server \
    --model XiaomiMiMo/MiMo-7B-RL \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 2

# EAGLE-3 加速
python3 -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16
```

***

## 7. 性能数据

### SGLang 官方数据（LLaMA 3.1 8B, H100）

| 方法        | 吞吐量 (tokens/s) | 加速比   |
| --------- | -------------- | ----- |
| 无推测解码     | 158.34         | 1x    |
| + EAGLE-2 | 244.10         | 1.54x |
| + EAGLE-3 | 373.25         | 2.36x |

### MTP 论文数据（13B 模型）

- 在 HumanEval 上提升 12%
- 在 MBPP 上提升 17%
- 4-token 预测实现 3 倍推理加速
- 8-token 预测实现 6.4 倍推理加速

***

## 常见误区澄清

### 1. "MTP 验证和生成一样，都是 forward pass，为什么能省时间？"

**回答**：生成 N 个 token 需要 N 次 forward（串行），但验证 N 个 token 只需要 **1 次 forward**（并行）。所以是 N → 1 的差距，不是 1 → 1。

### 2. "为什么主模型不能直接一次生成多个 token？"

**回答**：因果注意力的设计使得 token 之间存在依赖关系（token\_k 依赖 token\_1...token\_{k-1}），无法并行生成。MTP 的"生成"是由独立的 draft 模型完成的（代价只有主模型的 5-10%），主模型只负责**验证**。

### 3. "验证的序列更长，计算量不是更大吗？"

**回答**：传统方法 N 次 forward 的累计计算量是 O(N × L²)，而 MTP 1 次验证是 O((L+N)²)。当 N 较大时，MTP 的计算量显著小于传统方法。而且验证只需要主模型计算**1 次**KV Cache，不需要每步都重新计算。

### 4. "MTP 和 EAGLE 是什么关系？"

**回答**：

- **MTP 论文**：Meta 提出的一种训练方法 + 推理加速方案。训练时用多输出头，推理时用这些头做自推测解码
- **EAGLE**：SGLang 中主流的推测解码方案。在已训练好的主模型上，额外训练一个轻量级 draft 模块，在**特征级别**（而非 token 级别）做预测
- EAGLE 更实用，因为它不需要修改主模型的训练过程，可以后训练方式接入
- 完整的 EAGLE、N-gram、DFlash 对比详见：[speculative-decoding.md](./speculative-decoding.md)

