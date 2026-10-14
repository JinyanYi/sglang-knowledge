# K3 DeepEP：后续请求挂起、wheel 回归与 low-latency 假设

- **日期**：2026-08-29 至 2026-09-03 的历史四节点混部实验
- **范围**：Kimi-K3 W4A8，`TP16 / PP4 / DP2`，每个 PP stage 在单一节点内；DeepEP。
- **结论级别**：本页分开记录可观测现象、固定请求回归和静态实现假设。它们均**不能**外推为其他版本、其他节点布局或 P/D 分离拓扑的已证根因。

> 导航：[`projects/kimi-k3/evidence-map.md`](../../projects/kimi-k3/evidence-map.md)。
> 归档来源：[`archive/2026-09-09-note/README.md`](../../projects/kimi-k3/archive/2026-09-09-note/README.md)，原始笔记 §9–10。

---

## 可观测现象（已记录，不等于根因）

连续请求实验中，服务在首请求或后续请求后可能挂起：客户端与 health 超时；约
9.5–10 分钟后出现 NPU aicore timeout，进程退出。旧 wheel 上也记录过“首请求完成后，
第二请求卡住”；另一次纯空闲约 47 分钟没有自发崩溃。

这支持“问题由请求触发、且先于本次 wheel 重装存在”的观察；**不证明**是 scheduler、
PP 同步、DeepEP 或任一特定通信路径的唯一根因。最初的“scheduler/PP 竞态”写法是当时
的推断，后来已被新的通信假设替代，二者都不应当作定论。

可检索证据：`seqtest-002623.log`、
`dp2pp4tp16-2026-08-29_16-18-02.log`（历史文件名，不含运行身份信息）。
原始行：2026-08-29/30，§10.3，1999–2011。

## kernel wheel：仅单请求回归已验证

在四个参与节点安装 `sgl_kernel_npu` 的 `k3cppp@f89ddee` 构建 wheel 后，安装检查确认
包可导入、conv-state cast 标记存在。固定贪心请求与旧 wheel 对照：两边均为 115 token、
`finish_reason=stop`，token 序列一致，logprob 差异为噪声级，未见乱码或 AclNN 错误。

因此可写成：**该 wheel 未破坏这一次固定请求的精度回归**。不可写成 determinism 或连续
稳定性通过：名为 `newwheel-rerun-2026-08-29_15-05-52.txt` 的 rerun 响应无法解析，正是触发
挂起的请求，并未完成。

- 相关提交：kernel `f3903b3`（conv-state dtype 修复）、`f89ddee`（含 notify-dispatch 修复）。
- 证据文件：`dp2-pp4-tp16-k3cppp-newwheel-2026-08-29_14-53-36.txt`、
  `dp2-pp4-tp16-k3cppp-2026-08-29_13-31-43.txt`、`kverify2.sh`。
- 原始行：2026-08-29，§10.1–10.2，1979–1997。

## `auto` / `normal`：有价值的观察，有限的适用范围

历史记录称：`auto + cuda graph` 能 capture 但曾发生挂起；`normal + DeepEP` 曾通过第 1、
第 2 个 `grab_logprob` 请求，输出与单请求基线一致。normal 的变长 layout 被 hook 禁图，
所以这里的取舍是“当时该配方的两请求可用性”对“图能力”，不是 PP 与 DeepEP 互斥。

这是一组**特定环境的回归观察**，应复测完整参数、wheel、日志和至少连续两请求后才用于
新的部署决策。单请求 200 不足以验收该问题。

- 记录的工作配置：`--moe-a2a-backend deepep --deepep-mode normal`。
- 原始行：2026-09-03，§10.6，2033–2045；单请求终验在 2026-08-30，2029–2031。

## low-latency 的静态分析与待验证项

原笔记的代码阅读提出：此布局中 EP 自动调整为 TP=16，EP 组落在单节点；normal 路径有
`intranode_dispatch`，而 low-latency v2 被认为缺少等价的 intra-node 数据面。因此提出让
`MoeLowLatencyDispatchV2/CombineV2` 增加同机 HCCS 数据面、跨机仍走 RDMA 的方向。

这是**静态分析和修复设计**，不是抓到通道残留或因果闭环的运行时证据。此前关于
`comm_alg="hierarchy"` 的方案已被代码核查否定；不要再次建议传入这个不存在的算法值。
现有 `DEEPEP_LL_DEBUG` 仅会打印 soc、rank、RDMA/NVL rank、comm_alg 与 EP 组，下一次
`auto + graph` 复现应先采集这些字段，再修改 kernel。

- 涉及文件：`deep_ep.cpp`、`moe_low_latency_dispatch_v2.cpp`、
  `moe_distribute_combine_v2_*`、`cam_moe_dispatch_normal.h`。
- 原始行：2026-09-03，§10.6，2050–2058；早期通信因果链见 2020–2027，保留为假设。

## 最小复测顺序

1. 记录实际导入的 SGLang、kernel、DeepEP 版本和完整启动参数。
2. 固定输入与采样设置，先做一请求基线，再做至少两次连续请求。
3. 对 `auto` 与 `normal` 分别保存各 rank 日志、health、请求输出与 logprob。
4. 仅在 `auto` 复现时启用 `DEEPEP_LL_DEBUG=1`；把日志证据和静态推断分开报告。

> 不能将本页推论直接用于七节点 P-only/D-only 拓扑；请先看
> [`2026-09-k3-pd-topology-config.md`](./2026-09-k3-pd-topology-config.md)。
