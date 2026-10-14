# Qwen3.6 GDN：完整算法与源码学习目标

整理日期：2026-10-13。配套基础笔记：[状态、投影与 shape](./qwen3-6-gdn-state-and-shapes.md)。论文原文：[本地阅读目录](./papers/README.md)。

**这篇是最终要读懂的地图。** 下一次仍从单个 head 的小矩阵开始；暂时看不懂下方完整公式很正常。目标是将公式、tensor 的轴和源码中的写入对应起来，最后能够独立审查 PR #43845。

源码基准固定为 [`fe7fc57d8d091adca7c350f086428fce366884c0`](https://github.com/JinyanYi/sglang/tree/fe7fc57d8d091adca7c350f086428fce366884c0)，具体参数来自本次实际使用的 Qwen3.6-35B-A3B checkpoint。数学采用列向量；状态采用 SGLang 这条 kernel 路径的 `[dv,dk]` 顺序。

## 1. 先认清这条计算链

对一个 GDN 层、一个请求、一个 token，主要过程是：

1. 本层输入投影为 q、k、v，以及输出门 z、遗忘门输入 a、写入门输入 b。
2. q/k/v 分别经过短因果卷积和 SiLU，q/k 再做 L2 归一化。
3. a/b 决定当前 token 对旧矩阵的遗忘与修正强度。
4. 用 k/v 更新 temporal 矩阵，再用 q 读取。
5. 对读取结果做 RMSNorm、输出门控、输出投影，交给后续网络。

其中，conv 保存短卷积所需的近期输入，temporal 保存递推矩阵。模型权重在推理中固定，两个请求状态随 token 更新。

以下符号里的 t 是 token 位置，p 是 **value head 编号**。暂时省去 layer、request、TP rank 三个下标，理解单块矩阵后再补回来。

## 2. 输入投影：模型权重如何生成 q/k/v

令本层完成输入归一化后的特征为：

$$
\mathbf h_t\in\mathbb R^{2048}.
$$

它是当前位置在这一层的特征，通常已经经过前面许多层；只有网络最初的输入才直接来自 embedding。q/k/v 都由它生成：

$$
\begin{aligned}
\mathbf q_t^{\mathrm{raw}}&=W_q\mathbf h_t,&
\mathbf k_t^{\mathrm{raw}}&=W_k\mathbf h_t,&
\mathbf v_t^{\mathrm{raw}}&=W_v\mathbf h_t,\\
\mathbf z_t&=W_z\mathbf h_t,&
\mathbf a_t&=W_a\mathbf h_t,&
\mathbf b_t&=W_b\mathbf h_t.
\end{aligned}
$$

W 是训练得到的模型权重。上式按数学含义展开；源码把多组投影合并执行，以减少 kernel 调用。`in_proj_qkvz` 合并 q/k/v/z，`in_proj_ba` 合并 b/a。

TP2 下，一个 rank、一个 token 的拆分结果为：

```text
q_raw: [8, 128]     = 1024 channels
k_raw: [8, 128]     = 1024 channels
v_raw: [16, 128]    = 2048 channels
z:     [16, 128]    = 2048 channels
a:     [16]        = 16 scalar gate inputs
b:     [16]        = 16 scalar gate inputs

mixed_qkv: [4096]
```

因此，卷积的 4096 个通道仅包含 q/k/v。**z、a、b 不进入这段 q/k/v 短卷积，也不作为额外历史逐 token 存入 temporal。** 它们在处理当前输入时重新算出。

阅读入口：[投影构造](https://github.com/JinyanYi/sglang/blob/fe7fc57d8d091adca7c350f086428fce366884c0/python/sglang/srt/models/qwen3_5.py#L405)、[投影结果拆分](https://github.com/JinyanYi/sglang/blob/fe7fc57d8d091adca7c350f086428fce366884c0/python/sglang/srt/models/qwen3_5.py#L671)。

## 3. 短卷积：让当前特征带上局部历史

令打包后的原始投影为：

$$
\mathbf u_t^{\mathrm{raw}}
=\operatorname{Concat}\left(
\mathbf q_t^{\mathrm{raw}},
\mathbf k_t^{\mathrm{raw}},
\mathbf v_t^{\mathrm{raw}}
\right)\in\mathbb R^{4096}.
$$

本次卷积 kernel size 为 4。对每个通道 c，它独立进行：

$$
u_{t,c}^{\mathrm{conv}}
=\operatorname{SiLU}\left(
\sum_{\ell=0}^{3}w_{c,\ell}\,u_{t-\ell,c}^{\mathrm{raw}}
\right),
\qquad
\operatorname{SiLU}(x)=x\,\sigma(x).
$$

这里按历史 lag 给权重重新编号，tensor 中权重的物理存放顺序另看卷积实现。depthwise 的含义是：每个通道都有自己的时间卷积权重，此步骤沿时间混合，通道间的线性混合已经由投影完成。

开始处理 t 前，conv 保留 t-1、t-2、t-3 的原始投影；处理完成后保留 t、t-1、t-2。序列开头没有的历史按相应初始化处理。滑动窗口不会因已经处理的 token 总数而无限增长。

将卷积结果拆回 q/k/v，得到每组 q/k 的 128 维向量和每组 v 的 128 维向量。此处的 v 是卷积与激活后的结果。

q/k 随后做 L2 归一化。记：

$$
\begin{aligned}
\widetilde{\mathbf q}_t^{(j)}
&=\operatorname{L2Norm}\left(\mathbf q_t^{\mathrm{conv},(j)}\right),\\
\mathbf k_t^{(j)}
&=\operatorname{L2Norm}\left(\mathbf k_t^{\mathrm{conv},(j)}\right),\\
\mathbf q_t^{(j)}&=\frac{\widetilde{\mathbf q}_t^{(j)}}{\sqrt{d_k}},
\qquad d_k=128.
\end{aligned}
$$

最后一行将读取缩放吸收到 q 中，方便后面写成一次矩阵乘法。实际实现可以在 kernel 内乘这个 scale，未必提前构造缩放后的 tensor。Triton 参考递推使用向量平方和加 epsilon 后开方做 L2Norm；NPU 由对应 L2Norm 算子处理，不能据此断言各 backend 的浮点结果逐位相同。

阅读入口：[NPU prefill 的归一化与 scale](https://github.com/JinyanYi/sglang/blob/fe7fc57d8d091adca7c350f086428fce366884c0/python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py#L343)、[参考递推的归一化](https://github.com/JinyanYi/sglang/blob/fe7fc57d8d091adca7c350f086428fce366884c0/python/sglang/kernels/ops/attention/fla/fused_recurrent.py#L86)。

## 4. Head 分组：8 组 q/k 服务 16 个矩阵

本地 value head p 使用的 q/k head 为：

$$
j(p)=\left\lfloor\frac{p}{2}\right\rfloor,
\qquad p\in\{0,\ldots,15\}.
$$

例如，value heads 0、1 共享 q/k head 0；value heads 2、3 共享 q/k head 1。

但是每个 value head 都拥有自己的 v、a/b 门控和 temporal 矩阵。共享 q/k 不会把两个矩阵合成一个。

$$
S_t^{(p)}\in\mathbb R^{d_v\times d_k}
=\mathbb R^{128\times128}.
$$

因此单层、单槽位、单 rank 的 temporal 为 `[16,128,128]`。阅读入口：[kernel 的 head 映射](https://github.com/JinyanYi/sglang/blob/fe7fc57d8d091adca7c350f086428fce366884c0/python/sglang/kernels/ops/attention/fla/fused_recurrent.py#L44)。

## 5. 两个状态门：遗忘多少、修正多少

对 value head p，Qwen 这条 GDN 路径计算：

$$
\begin{aligned}
g_t^{(p)}
&=-\exp\!\left(A_{\log}^{(p)}\right)
\operatorname{softplus}\!\left(a_t^{(p)}+d_{\mathrm{bias}}^{(p)}\right),\\
\alpha_t^{(p)}&=\exp\!\left(g_t^{(p)}\right),\\
\beta_t^{(p)}&=\sigma\!\left(b_t^{(p)}\right).
\end{aligned}
$$

其中：

$$
\operatorname{softplus}(x)=\log(1+e^x),
\qquad
\sigma(x)=\frac{1}{1+e^{-x}}.
$$

模型参数是每个 head 的 `A_log` 和 `dt_bias`；a/b 则是当前输入投影得到的数。本式的 $d_{\mathrm{bias}}$ 对应源码 `dt_bias`。g 是 log decay，理想实数计算中为负；alpha 是实际遗忘乘数，处于 0 和 1 之间。beta 控制沿当前 k 的修正强度，也处于 0 和 1 之间。

**源码传递的 g 与公式中的 alpha 不是同一个数：kernel 要先对 g 取指数。** 此处每个 value head 的 alpha 是标量，该 head 的整个矩阵按同一比例遗忘。

阅读入口：[fused_gdn_gating](https://github.com/JinyanYi/sglang/blob/fe7fc57d8d091adca7c350f086428fce366884c0/python/sglang/kernels/ops/attention/fla/fused_gdn_gating.py#L44)。

## 6. 核心递推：完整的 Gated Delta Rule

下面固定一个 value head，省去 p 和 j(p) 下标。首先遗忘旧记忆：

$$
\overline S_t=\alpha_t S_{t-1}
\in\mathbb R^{d_v\times d_k}.
$$

然后向这块记忆询问：“当前 key 关联的 value，已有预测是什么？”

$$
\widehat{\mathbf v}_t=\overline S_t\mathbf k_t
\in\mathbb R^{d_v}.
$$

当前实际 v 与预测的差异，乘上写入强度：

$$
\boldsymbol\delta_t
=\beta_t\left(\mathbf v_t-\widehat{\mathbf v}_t\right)
\in\mathbb R^{d_v}.
$$

将修正量沿当前 key 方向写回，得到新状态：

$$
S_t=\overline S_t+\boldsymbol\delta_t\mathbf k_t^{\mathsf T}
\in\mathbb R^{d_v\times d_k}.
$$

最后用 q 从 **已经包含当前 token 更新** 的矩阵读取：

$$
\mathbf o_t=S_t\mathbf q_t\in\mathbb R^{d_v}.
$$

展开后，状态更新的紧凑形式为：

$$
\boxed{
S_t=\alpha_t S_{t-1}
\left(I-\beta_t\mathbf k_t\mathbf k_t^{\mathsf T}\right)
+\beta_t\mathbf v_t\mathbf k_t^{\mathsf T}
}
$$

这与 [GDN 原论文 §3.1、式 (10)](https://arxiv.org/html/2412.06464v3#S3.SS1) 的状态方向一致。上面的分步展开用于解释源码中的预测与修正。

### 为什么叫 delta rule：这块矩阵在调整 key → value 的映射

把遗忘后的矩阵当作起点，并考虑当前关联的误差：

$$
\mathcal L_t(S)=\frac12\left\|S\mathbf k_t-\mathbf v_t\right\|_2^2.
$$

它对 S 的梯度为：

$$
\nabla_S\mathcal L_t(S)
=\left(S\mathbf k_t-\mathbf v_t\right)\mathbf k_t^{\mathsf T}.
$$

在起点做一次步长为 beta 的修正，正好得到前述更新：

$$
S_t=\overline S_t-\beta_t\nabla_S\mathcal L_t(\overline S_t).
$$

这给矩阵记忆一个具体解释：它随当前序列调整 key 到 value 的映射。推理时这些变化发生在请求的状态 S 上，固定模型参数 W 并没有因此训练更新。这里的目标函数用于解释递推含义；实际推理 kernel 直接计算等价公式。

### 单个矩阵元素到底保存什么

行 i 对应一个 value 分量，列 j 对应一个 key 方向：

$$
S_{t,ij}
=\alpha_t S_{t-1,ij}
+\beta_t\left(
v_{t,i}-\alpha_t\sum_{\ell=1}^{d_k}S_{t-1,i\ell}k_{t,\ell}
\right)k_{t,j}.
$$

读取第 i 个输出分量：

$$
o_{t,i}=\sum_{j=1}^{d_k}S_{t,ij}q_{t,j}.
$$

**128 行和 128 列都是特征维度，均不是 128 个历史 token。** 历史关联通过连续更新混合在这些系数中。固定大小的矩阵容量有限，不能保证精确存下任意多条独立历史。

学习时可先假设 key 的范数严格等于 1。此时右乘当前 key 得到：

$$
S_t\mathbf k_t
=(1-\beta_t)\overline S_t\mathbf k_t+\beta_t\mathbf v_t.
$$

这说明 beta 将当前关联的预测往实际 v 拉近。若一个读取方向与当前 key 正交，则 delta 修正不会改变该方向的读取；遗忘本身仍会改变旧记忆。实际 epsilon 归一化与有限精度会让上述单位范数条件只能近似成立。

## 7. 从 128 维读取结果回到本层输出

每个 head 的读取结果做 RMSNorm，然后乘上来自当前输入的输出门：

$$
\begin{aligned}
\operatorname{RMSNorm}_{\boldsymbol\gamma}(\mathbf o)
&=\boldsymbol\gamma\odot
\frac{\mathbf o}{\sqrt{\frac1{d_v}\sum_{i=1}^{d_v}o_i^2+\epsilon}},\\
\mathbf r_t^{(p)}
&=\operatorname{RMSNorm}_{\boldsymbol\gamma}\left(\mathbf o_t^{(p)}\right)
\odot\operatorname{SiLU}\left(\mathbf z_t^{(p)}\right).
\end{aligned}
$$

本次 checkpoint 的 epsilon 为 1e-6；这条实现 `norm_before_gate=True`，默认输出门激活为 Swish/SiLU。其他模型若设置 `output_gate_type`，需按其配置重新核对。gamma 是训练得到的归一化权重。

本地 16 个 head 形成 `[16,128]`，展平成 2048 个分量。两个 TP rank 合起来对应全局 32 个 value heads，即 4096 个分量。数学上的全局输出为：

$$
\mathbf y_t=W_o\operatorname{Concat}_{p=0}^{31}\left(\mathbf r_t^{(p)}\right)
\in\mathbb R^{2048}.
$$

TP 的 row-parallel 输出投影由各 rank 计算局部贡献，通信路径完成合并。当前构造设置 `reduce_results=False`，不能仅看到 `out_proj` 就认定归约已经在该函数内部完成；这个调用链属于之后的源码阅读内容。

这里的 y 仍是 **本层输出特征**。加入层边界的残差处理、后续层与 MoE、最终归一化、LM head 和采样后，才会生成下一个 token。

阅读入口：[RMSNormGated 与 out_proj 构造](https://github.com/JinyanYi/sglang/blob/fe7fc57d8d091adca7c350f086428fce366884c0/python/sglang/srt/models/qwen3_5.py#L503)、[门控归一化实现](https://github.com/JinyanYi/sglang/blob/fe7fc57d8d091adca7c350f086428fce366884c0/python/sglang/kernels/ops/attention/fla/layernorm_gated.py#L487)、[forward](https://github.com/JinyanYi/sglang/blob/fe7fc57d8d091adca7c350f086428fce366884c0/python/sglang/srt/models/qwen3_5.py#L975)。

## 8. 把公式对上最短的 kernel

先读通用 Triton 递推作为数学参考，再追 NPU 的实际 dispatch。以下是基准源码中的核心片段，省去装载、归一化、循环与存储：

```python
b_h *= exp(b_g)
b_v -= tl.sum(b_h * b_k[None, :], 1)
b_v *= b_beta
b_h += b_v[:, None] * b_k[None, :]
b_o = tl.sum(b_h * b_q[None, :], 1)
```

- 第一行：b_h 从旧状态变成遗忘后的矩阵。
- 第二行：读取该矩阵对当前 k 的预测，从实际 v 中减去。
- 第三行：乘上 beta；此后名为 b_v 的局部变量已经表示修正量，含义随代码改变。
- 第四行：构造修正量与 k 的外积，更新 b_h。
- 第五行：用 q 读取新矩阵。

b_h 的 tile 是 `[BV,BK]`，两条轴分别来自 dv、dk。`tl.sum(...,1)` 沿 key 轴求和。实际 head 可能由多个 tile 执行，需要结合地址和 mask 阅读，不能把 tile 尺寸直接当成模型的完整 head_dim。

阅读入口：[核心递推第 90 行](https://github.com/JinyanYi/sglang/blob/fe7fc57d8d091adca7c350f086428fce366884c0/python/sglang/kernels/ops/attention/fla/fused_recurrent.py#L90)。NPU 的 prefill 使用 chunk 算子，decode/verify 使用其对应实现，具体分派从 [Ascend GDN backend](https://github.com/JinyanYi/sglang/blob/fe7fc57d8d091adca7c350f086428fce366884c0/python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py) 继续追。

## 9. Prefill 的 chunk 算法与逐 token 更新是什么关系

理解上述递推后，再读 [GDN 论文 §3.3 与附录 A](https://arxiv.org/html/2412.06464v3#S3.SS3)。其目标是把一段连续的递推组织成适合矩阵乘法的计算。

这里的 chunk length C 是一次并行处理的 token 数；conv kernel size 4 是局部卷积宽度。它们控制不同的计算步骤。

数学上，chunk 算法计算同一套因果状态更新，输出每个位置并把末状态传给下个 chunk。有限精度下，不同运算顺序可以产生数值差异。学习时先能手算单步，再理解一个 chunk 的输入状态、块内输出和末状态，最后研究 WY 表示与硬件优化。

## 10. 怎样从算法走到 checkpoint 与 MTP

对这个 GDN 层，处理完前缀位置 n 后，继续执行需要保留：

$$
\left(C_n,S_n\right),
$$

其中 C 保存后续短卷积需要的原始投影窗口，S 保存所有本地 value heads 的递推矩阵。给定下一步本层输入和相同模型参数，这对状态允许递推继续运行。

这解释了为什么只保存 S 或只保存卷积窗口都不够。混合模型的其他 full-attention 层还需要自己的 KV cache，整个服务恢复也需要正确的前缀位置与槽位元数据。

无量化时，保存与恢复应保留状态的数值及其消费者要求的布局。int8 temporal checkpoint 引入量化误差，恢复的是近似状态；功能测试通过与逐位无误差是不同的判断。

MTP/verify 会处理多个候选位置，再根据接受长度选择正确状态。实现使用扩展卷积窗口和中间状态，必须结合实际消费者检查。这也是本次 PR 从活动 buffer 的真实 shape 分配 checkpoint、保留完整 conv 快照的原因。**仅凭单步公式还不能完成对整个 MTP 回滚路径的证明。**

## 11. Ultimate goal：能自己解释到什么程度

以下是后续学习的完成标准，不代表今天已经掌握：

- [ ] 能解释 128×128 的两条轴，并用非方阵例子区分 dv 与 dk。
- [ ] 能用一个小矩阵演示 key/value 写入、预测误差修正和 query 读取。
- [ ] 能从投影一路说明卷积、归一化、head 复用、alpha/beta、输出门的顺序与 shape。
- [ ] 能把完整递推展开成矩阵元素更新，并对应 kernel 的广播和求和轴。
- [ ] 能区分固定模型权重、请求状态、池中槽位和被选中的 batch。
- [ ] 能解释 prefill 的 chunk 末状态如何接到 decode。
- [ ] 能沿 prefix checkpoint 的保存、恢复、槽位释放解释一个请求的状态生命周期。
- [ ] 能说明 MTP 接受不同长度时，conv/temporal 分别恢复到哪个前缀位置。
- [ ] 能独立检查 PR #43845 的真实 shape、完整窗口、原 buffer 写入和量化误差边界。
- [ ] 能将 GDN 与 Kimi K3 的 KDA 公式互相转置比较，识别通道级遗忘门的变化。

明天只开始前两项。后面按理解进展继续，阅读论文和优化 kernel 无需一次完成。

## 原始资料

- [Gated Delta Networks: Improving Mamba2 with Delta Rule，v3](https://arxiv.org/abs/2412.06464v3)：核心递推与 chunk 推导。原始代码：[NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)。
- [Transformers 的 Qwen3.5 文档](https://huggingface.co/docs/transformers/v5.13.0/model_doc/qwen3_5)：说明 Qwen3.6 复用该系列模型类；具体 shape 仍以实际 checkpoint 为准。
- [Transformers Qwen3.5 参考实现，v5.13.0](https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)：后续交叉阅读入口，矩阵方向与 backend 布局需单独核对。
- [GDN → KDA → Kimi K3 论文阅读顺序](./papers/README.md)：已经保存原始 PDF，附版本与先读章节。
