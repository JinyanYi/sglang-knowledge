# Hunyuan3-Preview NPU 精度 Bug 排查全过程

- 模型：Hunyuan3-Preview（约 295B 参数）
- 结构：80层 Transformer，1层 Dense FFN + 79层 MoE，hidden_size=4096，192个路由专家，top_k=8
- **太长不看**：NPU 的 `hardware_backend/npu/moe/topk.py` 里的 `fused_topk_npu` 方法，少乘了一个 `routed_scaling_factor`，导致每个 MoE 层的路由专家输出只有正确值的约 35%，精度严重下降。

---

# 一、问题描述

Hunyuan3-Preview 在 NPU 上部署时，有严重的精度问题。官方 GPQA Diamond 数据集精度 87.2%，NPU 上 40 道题只有 17.5%。

并伴有**输出无限循环**（模型一直重复同样的内容，直到 token 数量打满 65536 才停下来）。

```text
┌─────────────┬──────────────┬──────────┬──────────┬───────┬─────────┬─────────┐
│ Model       │ Dataset      │ Metric   │ Subset   │   Num │   Score │ Cat.0   │
├─────────────┼──────────────┼──────────┼──────────┼───────┼─────────┼─────────┤
│ Hy3-preview │ gpqa_diamond │ mean_acc │ default  │    40 │   0.175 │ default │
└─────────────┴──────────────┴──────────┴──────────┴───────┴─────────┴─────────┘

Hy3-preview  gpqa_diamond  40  925.395s  TTFT=21683ms  TPOT=39.88ms
```

错题分两类：

1. **14 道题**：打满 65536 tokens（死循环，没有输出最终答案）
2. **19 道题**：有输出但选错

---

# 二、问题定位

## 1. 缩小范围：逐一排除特性

**原则：遇到精度问题，第一步是尽量去掉所有高级特性，缩小搜索空间。**

现代推理框架有很多加速特性，任何一个都可能引入 bug，所以首先将它们逐一排除。

去掉了 CUDA graph（NPU 叫 npu graph）和 MTP（Multi-Token Prediction / EAGLE speculative decoding）之后，用最简启动参数：

```bash
python3 -m sglang.launch_server \
        --model-path /mnt/paas/weights/Hy3-preview \
        --attention-backend ascend \
        --reasoning-parser hunyuan --tool-call-parser hunyuan \
        --device npu \
        --tp-size 16 \       # 模型太大，至少需要 16 张 NPU 卡才装得下
        --trust-remote-code \
        --host 127.0.0.1 \
        --mem-fraction-static 0.84 \
        --port 9999 \
        --prefill-max-requests 1 \
        --max-running-requests 1 \
        --mm-attention-backend ascend_attn \
        --dtype bfloat16 \
        --base-gpu-id 0 \
```

去掉cuda graph 和 mtp 之后，由于输出速度过慢，只跑了两条 GPQAD就杀掉了 
 （而且输出速度变得特别慢，因此证明这两个特性很有用）

这两条都答对了，但是这两道题，打开cuda graph和mtp也能答对。
证明问题不在这两个特性，遂即从此往后的测试都带上这两个特性以加速跑测。

重新组一个 batch 跑，问题还是很严重。

**结论**：排除了 cuda graph 和 MTP 的影响，但依然不知道问题在哪。

这个时候需要去GPU上测一下精度以拿到一个baseline。如果GPU精度也不对，就是框架的问题；否则就是NPU部分的实现问题。

## 2. 获取 GPU baseline：确认是 NPU 特有问题

**原则：有 GPU 就先测 GPU，拿到正确答案作为对照，确定是框架问题还是 NPU 适配问题。GPU不行再在NPU上用Huggingface Transformer框架跑**

在 GPU（H20，单卡 96G）上用相同框架测试：

```bash
python3 -m sglang.launch_server \
        --model-path /home/weights/Hy3-preview \
        --reasoning-parser hunyuan --tool-call-parser hunyuan \
        --tp-size 8 \        # H20 单卡 96G，8卡能装下
        --trust-remote-code \
        --host 127.0.0.1 \
        --mem-fraction-static 0.85 \
        --port 9999 \
        --prefill-max-requests 1 \
        --max-running-requests 1 \
        --mm-attention-backend ascend_attn \
        --dtype bfloat16 \
        --base-gpu-id 0 \
```

GPU上测完，64条 精度0.87， 符合官方精度，排除了框架问题。

这个时候就很难受了，框架没问题，GPU精度正确，NPU上特性也没问题，那问题在哪呢？

**结论**：SGLang 框架本身没有问题，问题出在 NPU 适配层。

---

## 3. 发现突破口：第一个 Token 就不同

> **温度参数（temperature）是什么？** temperature 控制模型输出的随机性。temperature=0 时，模型每次都选概率最大的 token，输出完全固定（greedy decoding）；temperature>0 时，输出有随机性，多次运行结果不同。调成 0 方便对比，因为两边都确定性输出，只要输出不同就一定是计算出了问题。

找到一道"复现率 100%"的题目，将 temperature 设为 0，固定输出：

```text
题目（原题略）：给定弹性散射的相移 δ₀=90°, δ₁=67°, δ₂=55°, δ₃=30°, δ₄=13°，
      求散射振幅在入射方向上的虚部。
答案：B) 251.271 fm
```

- **GPU（greedy）**：输出正确推导过程，最终 `\boxed{B}`
- **NPU（greedy）**：进入死循环，一直重复计算步骤

**关键发现**：将两边输出的**第一个 token** 拿来对比，它们就不一样了！

这说明问题出在 **Prefill 阶段**（即处理输入 prompt 的过程），而不是 Decode 阶段。Prefill 阶段的输出是第一个 token 的 logits（每个词的概率分布），如果这一步就算错了，后续所有输出都会跟着偏。

> **Prefill vs Decode**：Transformer 推理分两个阶段。Prefill 是一次性处理完整个输入 prompt，生成第一个 token；Decode 是每次生成一个新 token，循环直到结束。如果 Prefill 计算有误，第一个 token 就错了，后续 decode 完全是在错误的基础上继续，很容易陷入循环。

---

## 4. 逐层打印：二分定位出问题的层

**方法**：在模型的每一层之后打印 hidden state（隐状态）的统计量（norm、mean、std），对比 GPU 和 NPU。第一个出现明显差异的层，就是 bug 所在。

> **Hidden State 是什么？** Transformer 里，输入经过 embedding 变成一个向量，这个向量依次经过每一层的处理，逐步"学到"输入的含义。每一层的输出叫做 hidden state，形状是 `[token数量, hidden_size]`。我们打印它的统计量（norm 是向量的整体大小，mean 是均值，std 是标准差）来判断两边计算是否一致。

在 `hunyuan_v3.py` 的模型前向函数里，每隔 8 层打印一次 hidden state 的统计量（完整代码见 results.md 的"定位方法"一节）。

第一轮粗粒度打印结果（每 8 层一个 checkpoint，取 last token，即最后一个位置的 hidden state）：


| Layer             | NPU last_norm | GPU last_norm | 差异          |
| ----------------- | ------------- | ------------- | ----------- |
| L000              | 1.903         | 1.907         | ≈ 0% ✓      |
| L007              | 0.334         | 0.590         | **−43%** ✗  |
| L015              | 0.602         | 0.896         | −33% ✗      |
| L023              | 0.946         | 1.831         | −48% ✗      |
| L031              | 1.418         | 2.995         | −53% ✗      |
| L039              | 4.120         | 7.060         | −42% ✗      |
| L079 (final)      | 64.7          | 79.1          | −18% ✗      |
| prefill last_norm | **81.8**      | **74.9**      | **+9.2%** ✗ |


**分析**：Layer 0 几乎完全一致，Layer 7 就已经差了 43%。说明 **问题在 L001～L007 这 7 层里**，其中很可能在第一个出问题的层（L001）。

细化打印，每层都打：


| Layer | NPU last_norm | GPU last_norm | 差异                       |
| ----- | ------------- | ------------- | ------------------------ |
| L000  | 1.902         | 1.907         | ≈ 0% ✓ 密集FFN层，正常         |
| L001  | 0.426         | 0.533         | **−20%** ✗ 第一个MoE层，开始出问题 |
| L002  | 0.421         | 0.601         | −30% ✗ 误差累积              |
| L003  | 0.320         | 0.539         | −41% ✗                   |
| L004  | 0.519         | 0.785         | −34% ✗                   |


**结论：问题从第 1 层（第一个 MoE 层）开始产生，并随层数增加不断累积。**

---

## 5. 理解模型结构：为什么 L0 没问题而 L1 开始出问题？

在进一步 debug 之前，先理解一下 HunyuanV3 的结构，这样后面的分析才能对号入座。

HunyuanV3-Preview 的配置（`config.json` 关键字段）：

```json
{
  "num_hidden_layers": 80,
  "first_k_dense_replace": 1,
  "hidden_size": 4096,
  "num_experts": 192,
  "num_experts_per_tok": 8,
  "num_shared_experts": 1,
  "moe_intermediate_size": 192,
  "route_norm": true,
  "router_scaling_factor": 2.826,
  "num_attention_heads": 32,
  "num_key_value_heads": 8
}
```

关键字段解释：

- `first_k_dense_replace = 1`：**前 1 层（L000）是普通的 Dense FFN，从 L001 起全是 MoE 层**。这直接解释了为什么 L000 没问题而 L001 开始出问题——出问题的是 MoE 相关的代码。
- `num_experts = 192`：共有 192 个路由专家
- `num_experts_per_tok = 8`：每个 token 选 8 个专家（top-k=8）
- `num_shared_experts = 1`：除路由专家外，还有 1 个 shared expert（所有 token 都会经过）

**每一层（L001 及以后）的计算结构：**

```
输入 x  [tokens, 4096]
    │
    ▼
┌──────────────────┐
│  RMSNorm         │  层归一化，稳定训练
└────────┬─────────┘
         │
    ┌────▼────┐
    │Attention │  自注意力，捕捉 token 间关系
    └────┬────┘
         │ (残差连接)
    ┌────▼────┐
    │ RMSNorm │  再归一化
    └────┬────┘
         │
    ┌────▼────────────────────────────┐
    │  Gate (self.gate)               │  x @ W_gate → logits [tokens, 192]
    │  TopK (self.topk)               │  sigmoid(logit + bias) → 每个 token 选 top-8 专家
    └────┬────────────────────────────┘
         │  topk_ids, topk_weights（routing 结果，顺序执行，不是 fork）
         │
         │ (这里才 fork)
    ┌────┴────────────────────┐
    │                         │
    ▼                         ▼
┌──────────────┐    ┌──────────────────────────────┐
│  shared_mlp  │    │  experts (FusedMoE)           │
│(HYV3FeedForward)  │  192 个路由专家                 │
│  所有 token  │    │  每个 token 按 topk_ids         │
│  都经过，    │    │  路由到对应的 8 个专家，          │
│  无 routing │    │  输出加权合并 (topk_weights)     │
└──────┬───────┘    └──────────────┬───────────────┘
       │                           │
       └──────────┬────────────────┘
                  ▼
         output = shared_out + expert_out
                  │
                  ▼ (残差连接)
         hidden_state for next layer
```

> **shared_mlp vs 路由专家的区别**：
>
> - `shared_mlp`（代码中的 `self.shared_mlp`）是一个普通的 FFN，**所有 token 无条件经过**，没有任何路由逻辑，参数量等效于 `num_shared_experts`（=1）个专家的 FFN。
> - `experts`（`self.experts`，`FusedMoE`）是 192 个路由专家，**每个 token 只经过 top-8 个**，由 gate + topk 的结果决定。
> - 两者并行计算，结果相加后才是这一层 MoE 的最终输出。

理解了结构之后，继续细化打印 L001 内部各个环节：


| 打印点        | 含义                       | NPU norm    | GPU norm    | 差异         |
| ---------- | ------------------------ | ----------- | ----------- | ---------- |
| LN_OUT     | 层归一化后                    | 0.4739      | 0.4740      | ≈ 0% ✓     |
| ATTN_OUT   | 注意力输出                    | 0.6199      | 0.6192      | ≈ 0% ✓     |
| MoE_IN     | MoE 输入（post-attn norm 后） | 1.4751      | 1.4740      | ≈ 0% ✓     |
| ROUTER     | 路由 logits                | 188.09      | 188.03      | ≈ 0% ✓     |
| SHARED     | Shared MLP 输出            | 0.09238     | 0.09229     | ≈ 0% ✓     |
| **EXPERT** | **路由专家输出**               | **0.03621** | **0.10254** | **−65% ✗** |
| layer out  | 层最终输出                    | 0.426       | 0.533       | −20% ✗     |


**结论清晰：ROUTER 之前全部正确，Shared MLP 正确，只有路由专家（EXPERT）的输出差了约 2.83 倍。**

---

## 6. 卡住了：TP size 不同导致 shape 对不上

发现了 EXPERT 输出有问题之后，下一步自然是想进一步打印 expert 内部的中间值（gemm 输入、gemm 输出、激活函数输出等），和 GPU 对比。

**但这里遇到了一个障碍**：

- NPU 需要 tp=16（模型太大，8 张 NPU 装不下）
- GPU 跑的是 tp=8

TP（Tensor Parallelism，张量并行）会把权重矩阵切分到多张卡上。每张卡只持有全部权重的一部分，具体的切分粒度由 `moe_tp_size`（MoE 模块内部使用的 TP 大小，受 EP 配置影响）决定，不一定等于启动参数里的 `--tp-size`。

实际打印出来的 shared_mlp 的 gate_up 权重可以直接验证这一点：

- NPU `--tp-size 16`：`[DBG FFN_W_GATE_UP] L001 shape=[192, 4096]`（每卡 192 行）
- GPU `--tp-size 8`：`[DBG FFN_W_GATE_UP] L001 shape=[384, 4096]`（每卡 384 行）

输入 `x` 同样被切分，形状也不一致。所以没法直接把 NPU 的中间激活值和 GPU 的对比——**形状就不匹配，无法逐元素比较**。

**这就卡住了**。需要想个办法让两边跑在相同的 TP size 下。

---

## 7. 解决办法：临时缩小模型层数

精度 bug 不需要全部 80 层都存在才能复现。Layer 1 就已经出问题了，所以**只保留前 10 层就够了**。

如果只有 10 层，显存需求大幅减少，NPU 就可以用 tp=8，和 GPU 的 tp=8 对齐，形状就一致了！

操作方法：直接修改模型 config.json，把 `num_hidden_layers` 从 80 改成 10：

```json
// 改之前
"num_hidden_layers": 80

// 改之后（临时，调试用）
"num_hidden_layers": 10
```

> **注意**：这只是临时改动用于调试，不是真的改模型，推理结果当然不再有意义，只是为了能在相同 TP size 下对比中间值。

这样改了之后，NPU 也可以用 tp=8 启动了，两边的权重形状完全对齐：

- 两边的 `FFN_W_GATE_UP L001 shape=[384, 4096]`，norm 也完全一致（`32.70472`）
- 两边的 MoE_IN、ROUTER、SHARED 都一致

两边对齐之后，进一步在 expert 内部逐步打印（`gmm1 → swiglu → gmm2`）：


| 阶段                           | 含义            | NPU norm | GPU norm  | 差异         |
| ---------------------------- | ------------- | -------- | --------- | ---------- |
| after_routing (dispatched_x) | 分发给专家的 token  | 41.59    | 41.57     | ≈ 0% ✓     |
| after_gmm1 (gate_up)         | gate_up 矩阵乘法后 | 28.68    | 28.66     | ≈ 0% ✓     |
| after_swiglu (act)           | SwiGLU 激活后    | 0.9966   | 0.9964    | ≈ 0% ✓     |
| after_gmm2 (down)            | down 矩阵乘法后    | 1.826    | 1.826     | ≈ 0% ✓     |
| **topk_weights**             | **路由权重**      | **4.97** | **14.03** | **−65% ✗** |


**结论更进一步**：expert 内部的所有矩阵乘法（gmm1、gmm2）和激活函数（swiglu）全部正确。唯一有问题的是 **`topk_weights`**（路由权重）——NPU 算出来的权重只有 GPU 的 35%！

---

## 8. 锁定根因：路由权重的缩放系数

对 `topk_weights` 做更详细的分析：

```
NPU per-token weight sum ≈ 1.000（精确归一化，每个 token 的 8 个专家权重之和 = 1）
GPU per-token weight sum ≈ 2.826（每个 token 的权重之和固定为 2.826）
```

GPU 的每个 token 权重之和是**完全一致的固定常数 2.826**，不是随机误差，说明这是一个**固定的乘法缩放系数**被漏掉了。

> 模型的 config 里有一个参数：`"router_scaling_factor": 2.826`，这就是答案。

追踪代码调用链：

```
HYV3MoEFused.forward()
    → self.topk(hidden_states, router_logits)   # TopK 模块，计算路由权重
        ├── GPU 路径: TopK.forward_cuda()
        │       → select_experts()
        │           → biased_topk_impl()         # layers/moe/topk.py:959
        │               → renormalize            # 归一化，sum变为1
        │               → topk_weights *= routed_scaling_factor  # ← 乘以2.826 ✓
        │
        └── NPU 路径: TopK.forward_npu()
                → fused_topk_npu()               # hardware_backend/npu/moe/topk.py
                    → npu_moe_gating_top_k(
                          routed_scaling_factor=(
                              1 if renormalize    # ← 故意传1，不在op内部缩放
                              else ...
                          )
                      )
                    # renorm 之后 sum = 1
                    # ← 这里少了 topk_weights *= routed_scaling_factor ✗
```

**GPU 路径**（`biased_topk_impl`，`layers/moe/topk.py:1004-1012`）：

```python
if renormalize:
    topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights / topk_weights_sum   # renorm，sum变为1
    if apply_routed_scaling_factor_on_output:
        topk_weights *= routed_scaling_factor        # ← 再乘回缩放系数
```

**NPU 路径**（`fused_topk_npu`）：

```python
topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
    ...,
    routed_scaling_factor=(
        1 if renormalize else topk_config.routed_scaling_factor
        # renormalize=True 时传 1，让 op 内部不做缩放（正确）
    ),
)
topk_weights = topk_weights.to(torch.float32)
# ← 缺少：topk_weights *= routed_scaling_factor
```

NPU 路径的意图是正确的：当需要 renormalize 时，不想让 NPU op 在内部乘 scaling factor（因为 renorm 之后 scaling factor 才有意义），所以传了 `routed_scaling_factor=1`。但是，**renorm 之后忘记在外部补上 `*= routed_scaling_factor` 这一步**，导致权重只有正确值的 35%。

---

## 9. 修复：补上缺失的一行

文件：`python/sglang/srt/hardware_backend/npu/moe/topk.py`，函数 `fused_topk_npu`

```python
# 在 npu_moe_gating_top_k 调用之后，加入以下代码：

topk_weights = topk_weights.to(torch.float32)

# When renormalize=True, we pass routed_scaling_factor=1 to the NPU op
# (correct, to avoid the op applying the scale before renorm).
# But the GPU path (biased_topk_impl) applies the scale AFTER renorm.
# Mirror that behavior here.
if (
    renormalize
    and topk_config.apply_routed_scaling_factor_on_output
    and topk_config.routed_scaling_factor is not None
    and topk_config.routed_scaling_factor != 1.0
):
    topk_weights = topk_weights * topk_config.routed_scaling_factor
```

---

## 10. 修复效果验证

修复后，NPU 的各项数值与 GPU 对齐：

**L001 expert 输出对比：**


| 指标                       | 修复前 NPU | 修复后 NPU     | GPU     | 说明        |
| ------------------------ | ------- | ----------- | ------- | --------- |
| `topk_weights` sum/token | ≈ 1.000 | ≈ **2.826** | ≈ 2.826 | ✓ 对齐      |
| `[DBG EXPERT] L001` norm | 0.03621 | **0.10197** | 0.10254 | 差异 < 0.6% |
| `[DBG layer] L001` norm  | 0.42634 | **0.53104** | 0.53338 | 差异 < 0.5% |


**全层 prefill hidden state norm 对比：**


| Layer             | 修复前 NPU    | 修复后 NPU    | GPU        | 修复后偏差     |
| ----------------- | ---------- | ---------- | ---------- | --------- |
| L000              | 1.90477    | 1.90546    | 1.90690    | 0.07%     |
| L001              | 0.42634    | 0.53104    | 0.53338    | 0.44%     |
| L002              | 0.42134    | 0.59946    | 0.60119    | 0.29%     |
| L003              | 0.31973    | 0.53973    | 0.53915    | 0.11%     |
| L004              | 0.51994    | 0.78774    | 0.78534    | 0.31%     |
| L005              | 0.46888    | 0.62643    | 0.62092    | 0.89%     |
| L006              | 0.36527    | 0.68068    | 0.67898    | 0.25%     |
| L007              | 0.33331    | 0.59052    | 0.58988    | 0.11%     |
| **prefill final** | **76.768** | **73.314** | **73.202** | **0.15%** |


修复后，prefill 阶段的 hidden state 与 GPU 的偏差从原来的 **>20%** 降到了 **≤1%**，前几个生成 token 的 logit 排序与 GPU 一致，死循环问题消失。

---

# 三、经验总结：NPU 精度 Debug 通用流程

这次 debug 走了一些弯路，但也形成了一套可复用的方法论：

```
1. 最小复现（缩小范围）
   └─ 关掉所有高级特性（cuda graph、speculation 等）
   └─ 找一道在 GPU 正确、NPU 错误的题，固定 temp=0

2. 确认是哪个阶段（prefill vs decode）
   └─ 比较第一个 token，如果就不同 → prefill 出问题
   └─ 在 CausalLM.forward 最后加 hidden state 日志，比较整体 norm

3. 层级二分定位（找出第一个出问题的层）
   └─ 每 N 层打一个 checkpoint，对比 GPU vs NPU
   └─ 找到第一个差异明显的层

4. 解决 TP size 不一样的问题
   └─ 修改 config.json 的 num_hidden_layers，把模型缩小到能在低 TP size 下运行
   └─ 对齐 TP size，形状就对齐了

5. 算子级 dump（逐步缩小到具体算子）
   └─ 在关键位置 torch.save 中间值，用脚本对比
   └─ 从 routing → gmm1 → act → gmm2 → finalize 逐步对比
   └─ 找到第一个 norm ratio 偏离 1.0 的位置

6. 注意事项
   └─ get_is_capture_mode() 判断必须加，否则 NPU graph capture 时 .item() 会同步 stream 导致崩溃
   └─ TP=16 时只在 rank 0 打日志，避免 16 倍重复输出
   └─ 打印 norm/mean/std/sum 比打印原始 tensor 更直观，且不会撑爆日志
```

---

# 附：为什么 `routed_scaling_factor` 是必须的？

这个参数的必要性源于 HunyuanV3 使用的 **sigmoid 路由方式**。

**Softmax 路由**（传统方式）：所有专家分数加起来为 1，renorm 之后 sum 还是 1，大小天然稳定，不需要额外缩放。

**Sigmoid 路由**（HunyuanV3 等现代大 MoE）：每个专家独立打分（`score = sigmoid(logit + bias)`），分数之间没有竞争关系。top-k 之后 renorm，sum 强制变成 1。但这把路由的置信度信息丢掉了——不管选出来的 8 个专家打分高低，renorm 后权重一样。

`router_scaling_factor = 2.826` 是**训练时校准的超参数**：在正常 token 分布下，top-8 专家的原始 sigmoid 分数之和期望值就是 2.826。renorm 后乘以 2.826，等效于"直接用原始 sigmoid 分数加权"的期望效果。

模型参数是按照"输入专家加权输出，权重 sum ≈ 2.826"来训练的，如果推理时 sum 只有 1.0（少了 2.826 倍），每一层 MoE 对 residual stream 的贡献只有正确值的 35%，80 层累积下来，hidden state 完全偏离训练分布，精度急剧下降。