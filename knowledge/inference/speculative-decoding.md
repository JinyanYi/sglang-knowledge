# 投机推理（Speculative Decoding）方法详解

> 本文档系统介绍了 SGLang 支持的各类投机解码方法：N-gram、EAGLE/EAGLE-3、DFlash，以及它们的原理、对比和在 SGLang 中的使用方法。
> MTP 的原理和训练方式详见：[multi-token-prediction.md](./multi-token-prediction.md)

---

## 目录

1. [投机推理概述](#1-投机推理概述)
2. [N-gram 投机解码](#2-n-gram-投机解码)
3. [EAGLE / EAGLE-3](#3-eagle--eagle-3)
4. [DFlash](#4-dflash)
5. [方法对比](#5-方法对比)
6. [SGLang 使用指南](#6-sglang-使用指南)
7. [常见问题](#7-常见问题)

---

## 1. 投机推理概述

### 核心思想

投机推理（Speculative Decoding）的核心是**"先猜后验"**：

```
传统自回归:  [context] → forward → token1 → forward → token2 → forward → token3
              (每次只能生成 1 个 token，串行)

投机推理:    [context] → 快速 Draft 生成 [token1, token2, token3]
                        → 主模型 1 次 forward 验证
                        → 接受正确的，拒绝错误的
```

### 为什么能加速

- **Draft 阶段**：用轻量方法快速生成多个候选 token（代价低）
- **Verify 阶段**：主模型 1 次 forward 验证所有候选（利用 Transformer 并行性）
- **净效果**：N 个 token 从 N 次 forward 降为 1-2 次 forward

### 投机推理的两大要素

| 要素 | 说明 |
|------|------|
| **Draft 方法** | 如何快速生成候选 token（N-gram / EAGLE / DFlash 的区别所在） |
| **Verify 机制** | 主模型如何验证候选 token（所有方法共用同一套验证逻辑） |

---

## 2. N-gram 投机解码

### 2.1 核心原理

N-gram 投机解码是一种**无需训练**的方法，利用文本中已有的 n-gram 模式来生成候选 token。

```
工作原理:
1. 从当前上下文提取 n-gram 模式
2. 在已生成的文本中查找匹配的模式
3. 根据匹配结果预测后续 token
4. 主模型验证
```

### 2.2 为什么不需要训练

N-gram 方法完全基于**统计匹配**，不需要任何额外的模型或参数：

- **数据源**：直接使用当前对话的上下文和已生成的文本
- **匹配方式**：前缀树（Trie）存储和检索 n-gram 模式
- **预测逻辑**：找到匹配的 n-gram 后，取其后续 token 作为候选

### 2.3 适用场景

N-gram 在以下场景表现优异：

| 场景 | 原因 |
|------|------|
| **代码生成** | 代码有大量重复模式（函数调用、循环结构、API 调用） |
| **长上下文** | 上下文越长，可匹配的 n-gram 越多 |
| **结构化文本** | JSON、XML、表格等有固定格式 |
| **重复性内容** | 技术文档、法律合同等 |

### 2.4 局限性

- **创造性任务**：诗歌、故事等需要创造力的任务效果差
- **短上下文**：可匹配的 n-gram 少，接受率低
- **全新内容**：没有历史模式可参考时无法工作

### 2.5 SGLang 中的实现

N-gram 在 SGLang 中的核心组件：

- `NGRAMWorker`：负责 N-gram 投机解码的完整流程
- `NgramCorpus`：前缀树存储，支持 BFS 匹配
- `NgramVerifyInput`：验证逻辑

```python
# SGLang 源码: sglang/srt/speculative/ngram_worker.py
class NGRAMWorker:
    def __init__(self, server_args, gpu_id, tp_rank, ...):
        self.ngram_corpus = NgramCorpus(
            min_bfs_breadth=server_args.speculative_ngram_min_bfs_breadth,
            max_bfs_breadth=server_args.speculative_ngram_max_bfs_breadth,
            match_type=server_args.speculative_ngram_match_type,
            capacity=server_args.speculative_ngram_capacity,
            max_trie_depth=server_args.speculative_ngram_max_trie_depth,
            draft_token_num=server_args.speculative_num_draft_tokens,
        )

    def forward_batch_generation(self, batch):
        self._prepare_for_speculative_decoding(batch)
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, next_token_ids, num_accepted_drafts = verify_input.verify(
            batch, logits_output, self.page_size, vocab_mask
        )
        self._update_ngram_corpus(batch)  # 更新 n-gram 语料库
```

---

## 3. EAGLE / EAGLE-3

### 3.1 核心原理

EAGLE（Extrapolation-based Acceleration for Generation with LLMs）是一种**需要单独训练**的投机解码方法。

```
EAGLE 的工作方式:
1. Draft 模型预测主模型的 hidden state（特征向量）
2. 预测的特征通过 LM Head 转为 token
3. 树状扩展：每个位置扩展 top-k 个候选分支
4. 主模型一次 forward 验证整个候选树
```

### 3.2 与 MTP 的区别

| 维度 | MTP（原生） | EAGLE |
|------|------------|-------|
| **训练方式** | 与主模型联合训练 | 在主模型训练完成后单独训练 |
| **模型架构** | 共享主干 + 多个输出头 | 独立的轻量 Draft 模型 |
| **预测目标** | 直接预测 token | 预测 hidden state（特征） |
| **适用模型** | 需要修改训练流程的模型 | 任何已训练好的模型 |
| **灵活性** | 低（需要重新训练主模型） | 高（可后训练接入） |

**关键洞察**：EAGLE 预测特征而非 token，因为特征空间的分布比 token 空间更平滑、更可预测（连续空间 vs 离散空间）。

### 3.3 EAGLE-2 vs EAGLE-3

| 维度 | EAGLE-2 | EAGLE-3 |
|------|---------|---------|
| **训练目标** | 特征预测 + token 预测 | 仅 token 预测 |
| **Draft 模型** | 1 层 Transformer | 1 层 Transformer |
| **复杂度** | 稍高 | 稍低 |
| **性能** | 好 | 更好（简化训练目标后更稳定） |

### 3.4 适用场景

- **通用场景**：适用于大多数文本生成任务
- **对话生成**：接受率稳定，加速比可靠
- **代码生成**：效果也不错，但可能不如 N-gram
- **需要后训练加速**：不想重新训练主模型时

---

## 4. DFlash

### 4.1 核心原理

DFlash（Block Diffusion for Flash Speculative Decoding）是一种**基于扩散模型**的投机解码框架，由 Z-Lab 在 2026 年 2 月提出（论文：[arXiv:2602.06036](https://arxiv.org/abs/2602.06036)）。

```
DFlash 的创新:
传统方法（EAGLE/MTP/N-gram）: Draft tokens 逐个生成（自回归）→ 串行
DFlash: 整个 block 的 tokens 一次性并行生成 → 并行
```

### 4.2 为什么更快

| 维度 | EAGLE-3 | DFlash |
|------|---------|--------|
| **Draft 方式** | 自回归（逐个生成） | 扩散模型（并行生成整个 block） |
| **Draft 模型** | 1 层 Transformer | 5+ 层 Transformer（因为并行，可以更深） |
| **Draft 成本** | 随 token 数线性增长 | 基本恒定（一次 forward pass） |
| **加速上限** | ~2-3× | ~4-6× |

### 4.3 训练方式

DFlash 的 Draft 模型是**扩散模型**，不是 Transformer：

1. 从主模型提取 hidden states 作为条件
2. 训练扩散模型，以 hidden states 为条件预测未来 block 的 tokens
3. 扩散模型学习从噪声中并行生成整个 block 的 tokens

### 4.4 适用场景

- **追求极致加速**：需要最高加速比的场景
- **资源充足**：有足够显存和算力训练扩散 Draft 模型
- **通用任务**：论文显示在 GSM8K、Math500、Code 等任务上都有 ~6× 加速

---

## 5. 方法对比

### 5.1 综合对比表

| 方法 | Draft 方式 | 训练需求 | 加速比 | 最佳场景 | 特点 |
|------|-----------|----------|--------|----------|------|
| **N-gram** | 字符串匹配（前缀树） | **零训练** | ~1.5-2× | 代码生成、长上下文、结构化文本 | 简单、特定场景效果好 |
| **EAGLE-3** | 自回归（1层 Transformer） | 微调 Draft 模型 | ~2-3× | 通用场景、对话生成 | 稳定、通用、后训练接入 |
| **MTP（原生）** | 自回归（共享主干） | 联合训练主模型 | ~2-4× | 需要最高质量的场景 | 质量最高、需要重新训练 |
| **DFlash** | **扩散模型（并行）** | 训练扩散 Draft 模型 | **~4-6×** | 追求极致加速 | 最快、需要单独训练 |

### 5.2 选择建议

```
需要加速？
  │
  ├─ 代码生成 / 长上下文 / 结构化文本
  │   └─ 使用 N-gram（零训练，简单有效）
  │
  ├─ 通用场景 / 对话生成
  │   ├─ 已有 EAGLE draft 模型 → 使用 EAGLE-3
  │   └─ 没有 draft 模型 → 先训练 EAGLE draft
  │
  ├─ 追求极致加速
  │   └─ 使用 DFlash（需要训练扩散 Draft 模型）
  │
  └─ 从零训练新模型
      └─ 考虑原生 MTP（联合训练，质量最高）
```

---

## 6. SGLang 使用指南

### 6.1 通用参数说明

| 参数 | 说明 | 默认值 | 适用方法 |
|------|------|--------|----------|
| `--speculative-algorithm` | 推测解码算法 | 无 | 所有 |
| `--speculative-num-steps` | Draft 深度（自回归步数） | 5 | EAGLE/MTP/DFlash |
| `--speculative-eagle-topk` | 每步扩展的分支数 | 4 | EAGLE |
| `--speculative-num-draft-tokens` | 验证容量（总 draft token 数） | 自动计算 | 所有 |
| `--speculative-draft-model-path` | Draft 模型路径 | None | EAGLE/DFlash |

### 6.2 N-gram 专用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--speculative-ngram-min-bfs-breadth` | BFS 最小广度 | 1 |
| `--speculative-ngram-max-bfs-breadth` | BFS 最大广度 | 8 |
| `--speculative-ngram-match-type` | 匹配类型（BFS/DFS） | BFS |
| `--speculative-ngram-capacity` | 前缀树容量 | 1000000 |
| `--speculative-ngram-max-trie-depth` | 前缀树最大深度 | 18 |
| `--speculative-ngram-external-sam-budget` | 外部 SAM 预算 | 0 |
| `--speculative-ngram-external-corpus-max-tokens` | 外部语料库最大 token 数 | 10000000 |

### 6.3 N-gram 启动示例

```bash
python -m sglang.launch_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --speculative-algorithm NGRAM \
    --speculative-num-steps 5 \
    --speculative-num-draft-tokens 16 \
    --speculative-ngram-max-trie-depth 18 \
    --speculative-ngram-capacity 1000000 \
    --speculative-ngram-match-type BFS
```

### 6.4 EAGLE-3 启动示例

```bash
python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16
```

### 6.5 DFlash 启动示例

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3-Coder-30B-A3B-DFlash \
    --tp-size 1 \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --mem-fraction-static 0.75
```

### 6.6 MTP（原生）启动示例

```bash
# 小米 MiMo 模型（自带 MTP 模块）
python -m sglang.launch_server \
    --model XiaomiMiMo/MiMo-7B-RL \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 2

# Qwen3.5-MTP 模型
python -m sglang.launch_server \
    --model Qwen/Qwen3.5-MTP \
    --speculative-algorithm MTP \
    --speculative-num-steps 4 \
    --speculative-num-draft-tokens 8
```

---

## 7. 常见问题

### 7.1 N-gram 和 EAGLE 哪个更好？

**取决于场景**：

- **代码生成**：N-gram 通常更好（代码有大量重复模式）
- **对话生成**：EAGLE 更稳定（通用性更强）
- **长上下文**：N-gram 接受率更高（可匹配的模式更多）
- **短上下文**：EAGLE 更好（N-gram 可匹配的模式少）

### 7.2 为什么 EAGLE 预测特征而不是 token？

特征空间的分布比 token 空间更平滑、更可预测：

- **Token 空间**：离散、高维、分布不均匀
- **特征空间**：连续、低维（相对）、分布更平滑

预测特征后再通过 LM Head 转为 token，比直接预测 token 更容易学习。

### 7.3 DFlash 比 EAGLE 快多少？

根据论文数据：

| 模型 | 任务 | EAGLE-3 加速 | DFlash 加速 |
|------|------|-------------|-------------|
| Qwen3-8B | GSM8K | ~2.5× | ~6× |
| Qwen3-8B | Math500 | ~2.8× | ~6× |
| Qwen3-4B | Code | ~2.3× | ~5.9× |

DFlash 的加速比约为 EAGLE-3 的 **2-2.5 倍**。

### 7.4 如何选择 speculative-num-steps 和 speculative-num-draft-tokens？

- `speculative-num-steps`：Draft 深度，越大生成的候选越多，但 Draft 成本也越高
- `speculative-num-draft-tokens`：验证容量，应 >= num_steps × eagle_topk

**经验法则**：

| 方法 | num-steps | eagle-topk | num-draft-tokens |
|------|-----------|------------|------------------|
| EAGLE-3（保守） | 3 | 4 | 16 |
| EAGLE-3（激进） | 5 | 8 | 64 |
| N-gram | 5 | N/A | 16 |
| DFlash | 根据模型推荐 | N/A | 根据模型推荐 |

---

## 关联笔记

- [multi-token-prediction.md](./multi-token-prediction.md) — MTP 的训练方式、推理加速原理和 SGLang 实现
- [prefill-decode.md](./prefill-decode.md) — Prefill 和 Decode 阶段的计算特性差异
- [cuda-graphs.md](./cuda-graphs.md) — CUDA Graph 和 Piecewise CUDA Graph 详解
