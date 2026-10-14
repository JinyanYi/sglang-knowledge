# GDN → KDA → Kimi K3：原始论文与阅读路线

整理日期：2026-10-13。这些是之后慢慢读的原始资料，配套 [Qwen 基础笔记](../qwen3-6-gdn-state-and-shapes.md) 与 [完整 GDN 算法](../qwen3-6-gdn-algorithm.md)。

三份 PDF 从 arXiv 的固定版本地址下载，保持原文，不改写论文。下面的导读是自己的学习路线；算法公式统一使用 LaTeX。

## 1. Gated Delta Networks：先读懂矩阵如何更新

- 论文：**Gated Delta Networks: Improving Mamba2 with Delta Rule**。
- 作者：Songlin Yang、Jan Kautz、Ali Hatamizadeh；ICLR 2025。
- 固定版本：arXiv **2412.06464v3**，2025-03-06，22 页。
- [本地 PDF](./gated-deltanet-2412.06464v3.pdf)、[arXiv 页面](https://arxiv.org/abs/2412.06464v3)、[在线全文](https://arxiv.org/html/2412.06464v3)、[原始代码](https://github.com/NVlabs/GatedDeltaNet)。

**第一遍阅读顺序**：§2.2 的 delta rule → §3.1 的 gated delta rule / 式 (10) → §3.4 的 block 架构。能解释单步后，再读 §3.3 的 chunk 算法与附录 A。

对应问题：旧矩阵对当前 k 的预测是什么？为什么写入的是预测误差与 k 的外积？遗忘门与写入门分别改变什么？q 从哪个版本的矩阵读取？

**这一阶段的完成标准**：能够用一个小矩阵演示两次写入与不同 q 的读取，并把每条更新对应到 SGLang kernel 的求和轴和广播轴。

## 2. Kimi Linear：理解 KDA 怎样扩展 GDN

- 论文：**Kimi Linear: An Expressive, Efficient Attention Architecture**。
- 作者：Kimi Team。
- 固定版本：arXiv **2510.26692v1**，2025-10-30，28 页。
- [本地 PDF](./kimi-linear-2510.26692v1.pdf)、[arXiv 页面](https://arxiv.org/abs/2510.26692v1)、[在线全文](https://arxiv.org/html/2510.26692v1)、[官方模型仓库](https://github.com/MoonshotAI/Kimi-Linear)。

**第一遍阅读顺序**：§2.2 回顾 linear attention / DeltaNet / GDN → §3 的 KDA 式 (1) → §4 的模型参数化。之后再读 §3.1 与附录 B/C 的 chunk 推导。

KDA 的一项核心变化是把每个 head 的标量遗忘乘数，变成沿 key 特征轴的向量。下面统一到 Qwen 笔记的状态方向：

$$
S_t\in\mathbb R^{d_v\times d_k},
\qquad
D_t=\operatorname{Diag}(\boldsymbol\alpha_t)
\in\mathbb R^{d_k\times d_k}.
$$

GDN 使用一个标量对整块矩阵进行遗忘：

$$
\overline S_t=\alpha_t S_{t-1}.
$$

KDA 让不同 key 通道具有不同的遗忘比例：

$$
\overline S_t=S_{t-1}D_t.
$$

其余预测、delta 修正与读取可以继续写成：

$$
\begin{aligned}
\widehat{\mathbf v}_t&=\overline S_t\mathbf k_t,\\
S_t&=\overline S_t
+\beta_t\left(\mathbf v_t-\widehat{\mathbf v}_t\right)\mathbf k_t^{\mathsf T},\\
\mathbf o_t&=S_t\mathbf q_t.
\end{aligned}
$$

读取的 scale 可由实现另行处理，此处只比较核心递推。紧凑形式为：

$$
S_t=S_{t-1}D_t\left(I-\beta_t\mathbf k_t\mathbf k_t^{\mathsf T}\right)
+\beta_t\mathbf v_t\mathbf k_t^{\mathsf T}.
$$

### 读论文时必须注意转置

Kimi Linear 论文把状态记为 `[dk,dv]`。令论文中的矩阵为 R，则：

$$
\begin{aligned}
R_t&=S_t^{\mathsf T},\\
R_t&=\left(I-\beta_t\mathbf k_t\mathbf k_t^{\mathsf T}\right)D_tR_{t-1}
+\beta_t\mathbf k_t\mathbf v_t^{\mathsf T},\\
\mathbf o_t&=R_t^{\mathsf T}\mathbf q_t.
\end{aligned}
$$

两个写法表示同一递推。**对角遗忘矩阵与 delta 转移矩阵通常不能交换顺序。** 方阵 shape 都是 $128\times128$ 时，即使把语义轴弄反也可能不触发 shape 报错，因此学习时会先用 $d_v=3$、$d_k=2$ 的非方阵。

对应问题：当前状态方向下，遗忘发生在行还是列？它为何作用于 key 通道？转置后，各矩阵的左右顺序为何变化？

## 3. Kimi K3：再读真实模型对 KDA 的改动

- 论文：**Kimi K3: Open Frontier Intelligence**。
- 作者：Kimi Team。
- 固定版本：arXiv **2607.24653v2**，2026-08-07，47 页。
- [本地 PDF](./kimi-k3-2607.24653v2.pdf)、[arXiv 页面](https://arxiv.org/abs/2607.24653v2)、[在线全文](https://arxiv.org/html/2607.24653v2)、[官方开放公告](https://www.kimi.com/news/kimi-k3-open-source)。

**第一遍只读 §2.1，尤其 §2.1.1 的 KDA。** 全模型的 AttnRes、MoE、训练与部署可以之后分主题阅读。

这一节需要抓住三处：

1. KDA 的核心递推继续使用通道级遗忘，再做 delta 修正。
2. K3 调整了遗忘门的参数化，限制 log decay 的下界。
3. K3 使用输入相关的 full-rank 输出门；它与原 Kimi Linear 的输出门参数化有差异。

### K3 的 lower-bounded decay

论文的遗忘 logits 由输入相关投影与 head bias 得到。记该向量为 z_decay，避免与 Qwen 的输出门 z 混淆。K3 §2.1.1、式 (5) 使用：

$$
\begin{aligned}
\mathbf g_t^{(p)}
&=g_{\min}\,\sigma\!\left(e^{A_p}\mathbf z_{t,\mathrm{decay}}^{(p)}\right),
\qquad g_{\min}=-5,\\
\boldsymbol\alpha_t^{(p)}&=\exp\!\left(\mathbf g_t^{(p)}\right).
\end{aligned}
$$

这里的指数、sigmoid 均逐分量计算。因而每个 key 通道的 log decay 在 $-5$ 和 $0$ 之间，保留比例在 $e^{-5}$ 和 $1$ 之间。论文将这一数值范围限制与 chunk 内的倒数缩放、矩阵乘法实现联系起来；读通单步之后再研究这层优化动机。

**不能将原 Kimi Linear 的负 softplus 遗忘门直接当作 K3 的最终参数化，也不能把 Qwen 的标量门公式直接套在 K3 上。**

### K3 的输出门

用论文的原始输出符号，式 (6) 为：

$$
\mathbf y_t
=W_o\left[
\sigma\left(W_g\mathbf x_t\right)
\odot\operatorname{RMSNorm}\left(\widetilde{\mathbf o}_t\right)
\right].
$$

这里的 sigmoid 输出门应与 Qwen 这条路径的默认 SiLU 输出门分别理解；两者都在状态读取之后作用于输出特征。

K3 的骨干以三层 KDA 配一层 Gated MLA 的模式组织，并在末尾增加全局注意力层，见 §2.1。读 SGLang 时，KDA 的递推状态与 MLA 的 KV cache 要沿各自层类型追踪。

## 4. 对接已经有的 SGLang 笔记

- [Qwen 的状态与 shape](../qwen3-6-gdn-state-and-shapes.md)：先复习自己已经算通的部分。
- [Qwen 的完整 GDN 算法](../qwen3-6-gdn-algorithm.md)：将公式对应到固定提交的源码。
- [Kimi K3 层结构](../kimi-k3-layer-structure.md)：已有的工程结构笔记，保留其原文注明的源码基准。
- [K3 项目入口](../../../projects/kimi-k3/README.md)：后续需要 PP/PD、状态传递与实现证据时再进入。

论文描述模型算法；工程笔记中的具体实现结论取决于对应源码提交。未来逐行阅读 K3 时再固定实际 checkout 与配置，避免将旧分支的支持情况当成当前社区实现。

**明天的接续点不变：每个 value head 的 128×128 矩阵，怎样由 k/v 写入、由 q 读取。** 三篇论文是后面的学习资料，第一步只需一个小矩阵。
