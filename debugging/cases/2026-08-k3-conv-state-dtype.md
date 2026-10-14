# K3 conv-state dtype：历史 NPU decode 事件

> 日期：2026-08 历史记录。本文是证据归档，**不是本次复现**，也不是当前环境状态。已移除用户名、地址、容器名和绝对路径；工件只保留 basename。导航见[归档入口](../../projects/kimi-k3/archive/2026-09-09-note/README.md)。

## 已确认：失败症状

- 历史 decode 调用 `causal_conv1d_update_npu` 时，`IndexPutImpl`/`aclnnIndexPutImpl` 报参数错误：目标预期 `DT_FLOAT`，实际为 `DT_BFLOAT16`（原始 `K3/209-note.md` 行 339-347）。
- 历史 DSpark 请求要求返回 logprob 时得到 HTTP 400；响应明确称 DSpark speculative decoding 不支持 `return_logprob`（385-407）。因此 DSpark 腿不能同时作为该接口 logprob 精度对照。

这只确立一次具体 dtype 失败和接口限制；不能把它写成所有 PP 乱码、服务卡死或精度问题的共同根因。

## 假设：当时的路径解释，尚未按源码复查

原记录推断：K3 Mamba cache 的 `conv` 为 bf16，内核内部 fp32 计算后以 in-place 写回，触发 bf16 目标与 fp32 写入的冲突；DSpark verify 路径据称绕开了触发此调用的 `forward_decode`，而 DSpark-off 的 PP4 腿进入了该路径（349-368）。

这些是当时的源码阅读解释，未保存锁定 revision 的最小复现、调用栈比对或 PP×DSpark 交叉实验。故不得断言“与 PP 无关”或“DSpark 修复了 dtype”。

## 历史 workaround 与版本证据

- 当时建议在调用前把 `conv_states` 转 fp32 并持久回写 cache；另列出内核接受 bf16、cache 默认 fp32、或让 `newbase` 开 DSpark 三个方向（355-368）。
- 开 DSpark 是**回避路径的 workaround，不是治本**；且受上述 HTTP 400 限制。
- 一组历史记录列出 `sglang@35b551f08`、kernel `@53e3543`，以及含 “preserve convolution state dtype” 修复的 `feca6c9` wheel；四机脚本 MD5 曾漂移，三个节点的 `newbase.sh` 缺 DeepEP 行（603-622）。这些是记录中的版本/配置快照，不是本次核验。
- 另一后续分支记录为 `k3cppp@f89ddee`，含 `f3903b3` conv-state dtype 修复（1957-1959）。历史 PP1、DSpark-off、指定 wheel 的单请求输出正常，但仅是有限请求证据，不能外推至 PP 拓扑或连续服务。

## 复验边界

1. 锁定 sglang、kernel、实际导入 wheel、CANN/NPU 运行时；先确认进程而非源码树。
2. 对 DSpark on/off 与 PP1/PP>1 保存 traceback、cache dtype、调用路径和请求接口结果。
3. 分开验收：无 dtype 错误、文本/token/logprob 精度、连续请求稳定性。

相关 PP 证据见[PP 精度案例](2026-08-k3-pp-precision.md)。
