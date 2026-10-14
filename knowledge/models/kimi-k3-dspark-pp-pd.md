# Kimi K3：DSpark、PP/PD 与 partial-load 结论

> 用途：给下一位 agent 的短上下文。先读本页，再决定是否继续翻源码或运行日志。
>
> 源码核对日期：2026-09-09。必须区分两个代码基准：
> `pdcpp@b54f247f18` 含 K3 hybrid KV 配对修复 `579dd83e8d`、Mamba state 校验
> `4ba2aac818`，但不含这里讨论的 DSpark PP addon；该路径来自
> `k3-dspark-prefill-partial-load@dbfe6e937d`，PP 核心提交 `34b531835e`。
> 社区 `main@708f51e44b` 当时的 K3 DSpark capture 仍限制 PP=1。
> 以下是这些版本的源码结论，**没有**对应四机 PP/PD 端到端实测背书。
>
> 关联原理：[K3 latent MoE / EP](./kimi-k3-latent-moe-ep.md)、
> [DeepEP on Ascend NPU](../npu/deepep-ascend-npu.md)。本页不重复它们的通用展开。

[项目入口](../../projects/kimi-k3/README.md) · [运行与验收](../../runbooks/k3-pd-pp/SKILL.md) · [PP 精度案例](../../debugging/cases/2026-08-k3-pp-precision.md)

## 30 秒结论

- PP=1 时，P 端捕获 target features，执行 `FC -> hidden RMSNorm -> draft K/V
  projection`，直接把 prompt draft KV 写入本地 draft KV pool，再随 target KV 传给 D。
- 上述 addon 的 PP>1 路径中，各 stage 只算本地 FC 列块贡献，线上传递的是 FP32
  `dspark_ctx_acc[tokens, hidden_size]`；只有最后 PP stage 做 RMSNorm、生成并发送
  真正的 draft KV。不要把中间量叫作 draft KV。
- RMSNorm 非线性要求先求完整 FC 和，但不要求 gather 到唯一一张卡。当前实现中各参与
  TP rank 都能得到本地 token 的完整投影向量，因此各自在本地做 RMSNorm，再生成自己负责的 KV heads。
- partial load 的主价值是降低加载峰值。PP=1 没有 FC 列裁剪，若原版也能启动且最终
  KV 容量相同，稳态 token throughput 预期基本不变。
- PP>1 额外省下原来常驻的 FC 无用列；是否转化为吞吐，要看它是否提高所有 PP rank
  `MIN` 同步后的 `max_total_num_tokens`，并且 P 端确实是系统瓶颈。

## 1. PP=1 的 prompt draft KV 链路

对 dense DSpark draft，P 端不是把 target hidden 发给 D 再重算，而是先在 P 端完成：

```text
target 指定层 hidden
  -> concat features
  -> 完整 FC（当前 dense draft 为复制的 nn.Linear）
  -> hidden RMSNorm
  -> 每个 draft layer 的本 rank K/V projection
  -> K-Norm + RoPE
  -> draft KV pool
  -> 与 target KV 一起做 P -> D cache transfer
```

P/D 两边使用对应 cache indices；D 收到的是已经物化的本地 KV shard，不重做 FC/RMSNorm。

DP/EP 不改变这条数学依赖：target 模型先完成该层所需的专家归并，DSpark 捕获的是
本 DP token shard 的完整 feature。完整单-token 向量是 RMSNorm 的输入，但“完整”不等于
“必须集中到 TP rank 0”。

## 2. PP>1 为什么传 `dspark_ctx_acc`

若完整 FC 按输入 features 分成各 stage 的列块：

```text
z = W0*x0 + W1*x1 + ... + Ws*xs
c = RMSNorm(z)
```

不能改成 `sum(RMSNorm(Ws*xs))`。因此每个 stage 计算 `Ws*xs`，以 FP32 累加到
`dspark_ctx_acc` 后交给下一 stage；最后 stage 拿到完整 `z` 后才做 RMSNorm 和 K/V writer。

对象所有权：

- 每个 PP stage：生成并发送自己负责的 target KV。
- 非最后 stage：只持有本地 FC 列块，不需要 draft transformer layers，也不分配 draft KV pool。
- 最后 stage：持有本地 FC 列块、hidden norm、K/V-only writer 和 draft KV pool，并发送 prompt draft KV。

## 3. partial load 到底新增省了什么

### 优化前（已经支持 PP+DSpark，但未 partial load）

初始化会先完整构造并加载 draft，然后：

1. `prune_to_ctx_kv_injection()` 删除 MLP、O projection、Markov/confidence head；
2. 非最后 PP stage 把 `draft_model.layers` 整体清空；
3. 非最后 PP stage 的 `alloc_memory_pool()` 直接返回，不分配 draft KV pool；
4. 随后才进行 target/draft KV pool sizing、backend/graph 初始化和 server ready。

所以中间 stage 的 draft layers 在旧版只是**加载峰值**，不是 ready 后的常驻占用；draft
KV pool 在新旧两版都不存在。

### partial-load 之后

- 非最后 stage：从一开始不构造 draft layers，只加载本 stage 的 FC 列块。
- 最后 stage：只构造 prompt KV 所需的 K/V-only layers，并只加载本地 FC 列块。
- `dspark_ctx_acc`、必要 K/V 计算以及 P->D draft-KV 数据量没有减少。

因此新增的稳态显存收益主要是 **FC 无用列**（以及最后 stage 的 Q 等残余）；直接计算量
基本没变。

容量预算修正要单独看：非最后 stage 没有 draft KV pool，因此不应在每-token cell size
里虚扣 draft KV。这是独立正确性/容量修正，不等同于 partial weight loading。

## 4. 我们的 PP4 拓扑下 FC 节省量

“8 机 4P4D”在本次讨论中表示 4 台 prefill host + 4 台 decode host；PP4 版本的 P 侧为
TP16/PP4/DP2，D 侧为 TP64/PP1/DP4。这是讨论中的配置，非本页实测拓扑；
与历史笔记七机 3P4D 分开。以下 HBM 数字只针对 P 侧 draft FC。

已讨论配置：K3 93 层，PP partition `24,23,23,23`；DSpark 捕获 target layer ids
`[7, 23, 51, 67, 83]`；draft hidden size 7168，FC 为 BF16。K3 target 使用 W8A8
**不代表 DSpark FC 也是 INT8**。

完整 FC：

```text
7168 * (5 * 7168) * 2 bytes = 490 MiB / 卡
每个捕获 feature 对应的 FC 列块 = 98 MiB / 卡
```

stage 到 feature 的归属为：

| PP rank | 本地捕获 feature | 保留 FC | 每卡节省 | 16 卡整机节省 |
|---|---:|---:|---:|---:|
| 0 | 7 | 98 MiB | 392 MiB | 6.125 GiB |
| 1 | 23 | 98 MiB | 392 MiB | 6.125 GiB |
| 2 | 51, 67 | 196 MiB | 294 MiB | 4.594 GiB |
| 3（last） | 83 | 98 MiB | 392 MiB | 6.125 GiB |

前三台非 last P host 合计约 16.844 GiB，但分散在 48 张卡上。layer 23 虽位于前一
stage 边界附近，其 capture 由下一 consumer 所在 stage 负责，所以归 PP rank 1。

PP=1 时五个 features 都在同一 stage：完整 FC 仍为 490 MiB/卡，**FC 列裁剪收益为 0**。

## 5. 对吞吐的判断标准

不要用“峰值省了多少”直接宣称 tokens/s 提升。需要依次证明：

```text
瓶颈 rank 常驻显存下降
  -> PP 全局 MIN 后的 max_total_num_tokens 上升
  -> 实际负载获得更高并发或更少容量等待
  -> P 端处理能力上升
  -> D 端没有先成为瓶颈
```

建议 A/B 至少记录：加载峰值、ready 时间、各 rank ready 后 free HBM、最终
`max_total_num_tokens` / `max_running_requests`、同一 workload 的 P/D utilization 和
端到端 output tokens/s。

## 6. K3 PP+PD pointer 修复与 DSV4 的边界

`579dd83e8d` 修的是 K3 `HybridLinearKVPool`：PP stage 只注册稀疏的 full-attention
layers，而 PP=1 decode 注册全模型条目；按平坦 list 位置切片会错，必须按 global layer ID
配对（K/V 重复 layer ID 时按 occurrence 顺序消费）。

DeepSeek-V4 不走这个 `is_hybrid_mla_backend` 分支。它通过
`mla_compression_ratios + prefill_start_layer/end_layer`，分别计算 C4/C128 前缀计数，切分：

- 主 KV：C4 KV / index K / index scale；
- C4 attention/indexer state；
- C128 state；
- SWA KV。

所以 DSV4 不会原样复现 K3 这次 sparse hybrid pointer bug；但当前看到的 DSV4 注册
端到端测试是 PP1，不能据此声称 DSV4 的 PP>1 + PD 已被完整运行验证。

`4ba2aac818` 是 Mamba persistent-state slot 配对/长度校验，也不适用于 DSV4 的
SWA/C4/C128 state。

## 7. 当前 DeepEP 参数是什么意思

2026-09-09 核对的本地脚本中可见（不等于远端当前运行参数）：

```text
--moe-a2a-backend deepep --deepep-mode auto
```

DeepEP 的通信语义是 personalized all-to-all，但实现接口是 `dispatch -> local experts ->
combine`，不是简单调用一次普通 HCCL all-to-all：

- dispatch：按 router top-k 把 token hidden/metadata 发给 expert owner ranks；
- combine：把 expert 输出送回 token owner，并按 routing weight 合并；
- `auto`：extend/prefill batch 解析为 `normal`，普通 decode batch 解析为 `low_latency`；
- 显式 `normal`：P/D 都只走 normal，不进入 low-latency。

DeepEP 会把 EP size 调到 TP size。于是 PP4/TP16 时，每个 PP stage 有独立 EP16 group；
PP1/TP64 时，每个角色内是一个 EP64 group。因此本页 PP4 P + PP1 D 的拓扑是
P 的四个 EP16 group 与 D 的一个 EP64 group；P/D 不共用专家通信组，二者之间传 KV，
不做跨 P/D 的 MoE all-to-all。

## 8. 下一位 agent 的最短源码入口

在相邻 `sglang` 仓库查这些标识即可：

- `dspark_worker_v2.py`：`_prefill_load_plan`、`_context_only_pp_rank`、
  `_accumulate_pp_context`、`inject_target_hidden`。
- `dspark_pp.py`：`context_feature_slice`、`accumulate_context`。
- `dflash.py` / `dspark.py`：`project_target_hidden`、`write_target_hidden_kv`、
  `prune_to_ctx_kv_injection`、K/V-only load。
- `disaggregation/prefill.py`：draft KV 注册与 `transfer_draft_cache`。
- `disaggregation/ascend/conn.py`：K3 layer-ID pairing 与 DSV4 C4/C128 slicing。
- `pool_configurator.py` / `kv_cache_configurator.py`：draft KV cell-size 核算与 PP capacity MIN。
- `layers/moe/utils.py` / `token_dispatcher/deepep.py`：DeepEP auto mode、dispatch/combine。
