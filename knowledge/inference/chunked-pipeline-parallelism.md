# Chunked Pipeline Parallelism：TP / CP 通信与 Dynamic Chunking

> 阅读：[Pipeline Parallelism in SGLang: Scaling to Million-Token Contexts and Beyond](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/)（2026-01-15）。
> 笔记与源码核对：2026-10-17。只记阅读时需要拆开理解的几个问题；PCP / DCP 是延伸阅读。
> Prefill / Decode 基础见 [prefill-decode.md](./prefill-decode.md)；起服与实验见 [K3 PD / Chunked PP runbook](../../runbooks/k3-pd-pp/SKILL.md)。

## 先记住这几件事

- **TP 的 All-Reduce 合并的是同一个输出的部分贡献；CP 的 KV All-Gather 收集的是不同 token 的数据。**
- Attention TP 在各 head 经过输出投影之后归约；不能把不同 head 的原始输出直接相加。
- **CP 不等于必须 All-Gather 全部 KV。** SGLang 的 MLA DCP 可以保留 KV 分片，再合并局部 Attention 输出。
- Dynamic chunking 先预测等耗时的 chunk 长度，再用 smooth factor 在预测长度和初始长度之间折中。

---

## 1. 文章为什么要用 Chunked PP

PP 按层划分 stage，主要在 stage 边界传递 hidden states。一个 stage 可以包含多层，所以跨 stage 通信不必像常规 TP 那样在每层的 Attention、MLP 后都发生。

长 prompt 如果整段进入流水线，后面的 stage 要等很久。Chunked PP 把 prompt 分成多块：stage 0 处理 chunk 2 时，stage 1 可以处理 chunk 1，让不同 stage 同时有工作。

但 **fixed token length 不等于 fixed runtime**。越靠后的 chunk，其 query 要关注的历史 KV 越多，同样数量的新 token 会越来越慢。Dynamic chunking 就是为了缓和这个耗时差异。

## 2. 为什么 TP 是 All-Reduce，CP 是 All-Gather

### 2.1 MLP TP：同一个 token 的特征分开算，输出贡献相加

以 down projection 为例，$h$ 是 MLP 中间特征：

$$
h=[h_0,h_1],\qquad
W=\begin{bmatrix}W_0\\W_1\end{bmatrix}
$$

于是：

$$
y=hW=h_0W_0+h_1W_1
$$

这里切的是**同一个 token 的 feature / channel 维度**。如果有 $S$ 个 token，$h$ 的 shape 从 $[S,D_{\mathrm{ff}}]$ 分成两个 $[S,D_{\mathrm{ff}}/2]$，$S$ 没有被切。

一个手算例子：

$$
h=[1,2,3,4],\qquad
W=\begin{bmatrix}1&2\\0&1\\2&0\\1&1\end{bmatrix}
$$

- GPU 0 用前两个特征和前两行权重，得到 $y_0=[1,4]$。
- GPU 1 用后两个特征和后两行权重，得到 $y_1=[10,4]$。
- All-Reduce(SUM) 得到 $y=[11,8]$，每张卡拿到完整输出。

**两张卡的局部输出 shape 相同，但各自只包含部分输入特征的贡献。** 实现中通常由 up projection 直接产生各卡的 $h_i$，不必先算出完整 $h$ 再切。

### 2.2 Attention TP：归约发生在输出投影之后

Attention 的计算顺序是：

$$
QK^\top/\sqrt d+\mathrm{mask}
\;\longrightarrow\; P=\mathrm{softmax}(\cdot)
\;\longrightarrow\; a_i=P_iV_i
\;\longrightarrow\; W_O
$$

不同卡负责不同 head，先独立算出 $a_0,a_1$。多头输出在数学上需要拼接，再做输出投影：

$$
y=[a_0,a_1]\begin{bmatrix}W_{O,0}\\W_{O,1}\end{bmatrix}
=a_0W_{O,0}+a_1W_{O,1}
$$

所以可以各卡先乘自己的 $W_O$ 分片，再 All-Reduce。**相加的是经过 $W_O$ 后对同一个输出的贡献，原始 head 输出仍然是不同的特征。**

### 2.3 CP：本卡的 query 需要其他 token 的 KV

只看同一个 head：GPU 0 持有 token 1 的 KV，GPU 1 持有 token 2 的 Q、KV。GPU 1 要算 token 2 的输出，需要：

$$
[p_{21},p_{22}]=\mathrm{softmax}([q_2k_1^\top,q_2k_2^\top]/\sqrt d),
\qquad o_2=p_{21}v_1+p_{22}v_2
$$

它缺少 $k_1,v_1$，因此需要取得其他卡的 KV。All-Gather 沿 token 维收集这些数据；直接把 KV 相加会丢失位置对应关系，而且每个位置的 Attention 权重不同。

文章用 All-Reduce / All-Gather 概括常见前向路径。实际 TP 可以结合 Reduce-Scatter，CP 也可以分块环传 KV，或采用下面的局部结果合并。

## 3. 延伸：SGLang 的 PCP 和 DCP

**PCP（Prefill Context Parallelism）**：prefill 有很多 query token，各卡负责其中一部分，再取得计算这些 query 所需的 KV。比如一张卡负责偶数位置的 query，另一张负责奇数位置，分担长 prompt 的计算。

**DCP（Decode Context Parallelism）**：每个请求通常只有一个新 query，将它需要访问的历史 KV 按位置分散到不同卡。各卡取得所需 query 后，读取本地 KV，计算局部 Attention，再交换并合并结果。

以 SGLang 的 **MLA decode 路径**为例，合并的是局部输出和 LSE（log-sum-exp）。局部 softmax 的分母各不相同，所以不能直接把局部输出相加，需要用 LSE 恢复全局归一化。这样就能让整段历史 KV 保持分片。

这是两种典型的数据流；具体通信、缓存布局和支持范围仍取决于模型与 backend。参见 [MLA PCP 实现说明](https://github.com/sgl-project/sglang/pull/23292)和 [DCP 文档](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/dcp.mdx)。

## 4. Dynamic chunking 的公式在求什么

设 $T(n)$ 是 profiling 拟合的累计 prefill 耗时模型，$L$ 是已处理的 prefix 长度，$x$ 是下一块长度，$C_0$ 是初始 chunk size。

文章写的是：

$$
\underbrace{T(L+x)-T(L)}_{\text{下一块新增工作的预计耗时}}
=\underbrace{T(C_0)}_{\text{第一块的预计耗时}}
$$

问题就是：**下一块切多大，才能和第一块差不多耗时？** 左边相减是在估计新增工作的代价，历史 KV 会被复用，并不会重新执行前缀。

源码拟合 $T(n)=an^2+bn+c$，目标耗时实际取 $T(C_0)-T(0)$，去掉常数项。代入后得到：

$$
ax^2+(2aL+b)x-[T(C_0)-T(0)]=0
$$

代码用一元二次方程的正根求 $x$。$L$ 增大时，$2aLx$ 这一项变大，为了保持目标耗时，预测的 $x$ 就要变小。

### 手算一次

以下只是教学模型：长度以 K token 为单位，$T(n)=n^2+2n$ ms，初始块为 4K。

- 第一块：$T(4)=24$ ms。
- 已有 4K prefix，下一块仍取 4K：$T(8)-T(4)=80-24=56$ ms。
- 下一块改取 2K：$T(6)-T(4)=48-24=24$ ms。

所以模型预测第二块应缩到 **2K**。实际流程是先 profiling、拟合，再根据 prefix 长度预测；不是在运行过程中计时到点就截断 chunk。

## 5. Ratio / smooth factor 具体乘在哪里

原始预测之后，还有一次插值：

$$
\boxed{x_{\mathrm{smooth}}=C_0+\sigma(x_{\mathrm{pred}}-C_0)}
$$

$\sigma$ 对应 `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR`。继续用初始 4K、预测 2K 的例子：

- $\sigma=0$：4K，保持初始大小。
- $\sigma=0.5$：3K，采用一半缩减量。
- $\sigma=0.75$：2.5K，采用 75% 缩减量。
- $\sigma=1$：2K，完全采用预测值。

**参照值始终是初始 chunk size $C_0$，不是把上一块不断乘以 ratio。** 小于 1 时保留更大的 chunk，接受后续块更慢一些，减少过小 chunk 带来的调度开销。

预测模型并不能精确描述所有 kernel shape 的效率。实现还会先用 $C_0/4$ 托底，再向下对齐到 `max(page_size, 64)`，应用 context / token 上限并重新对齐。因此最终长度不一定满足严格等耗时，最终上限也可能压过前面的托底值。

### 源码对应

当前实现位于 [`dynamic_chunk_sizer.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler_components/dynamic_chunk_sizer.py)，重点看 `ChunkSizePredictor`：

1. `fit()`：由序列长度和实测耗时拟合 $a,b,c$。
2. `set_target_latency()`：设置 $T(C_0)-T(0)$。
3. `predict_next_chunk_size()`：解二次方程，再做 smoothing、托底、对齐和上限限制。

源码链接指向 `main`；后续实现可能迁移，优先按类名、函数名和环境变量搜索。
