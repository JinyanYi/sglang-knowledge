# K3 PP 精度：历史证据与结论边界

> 日期：2026-08 历史记录。本文记录证据，**不是本次复现**或当前 runbook。已移除用户名、地址、容器名和绝对路径；工件只保留 basename。导航见[证据地图](../../projects/kimi-k3/evidence-map.md)与[归档入口](../../projects/kimi-k3/archive/2026-09-09-note/README.md)。

## 已确认的历史观察

- 早期 PP4 `newbase`、DSpark-off 腿 HTTP 200，却出现空 `content`、控制 token 和 `finish_reason=length`；这是一条精度失败记录，不应以 HTTP 200 当成功（796-811）。
- 早期 proxy send/recv 摘要在有限字段上匹配；它只覆盖那次打印的局部摘要。block size=2 时，PP1/PP4 的 layer 0/1 对齐、从 layer 2 出现差异，不能据摘要声称所有层都一致（1209-1236）。

## 同 TP32 的 93 层 PP1 vs PP2 dump（2026-08-28）

- 条件：同 TP32、无 DP，PP1 为 93 层；PP2 分段为 `47,46`，因此可以比较同一 `[94,7168]` token 行（1811-1825）。
- `ppsend`/`pprecv` 保存的 hidden 与 residual 逐元素 max diff=0；46→47 层交界没有额外突变（1862-1876）。这证明**已保存的边界消息未被传错**，不等价于端到端数值或生成正确。
- 差异逐层积累，且 PP2 rerun 自身也出现同量级漂移；后半段不能归因于 PP hop（1876-1880、1892-1893）。attn-res 仅在 0/12/24/... 更新，非更新层不变，不应把 cosine 或静态缓存当主证据（1876）。
- PP1 TP32 的 93 层 prefill dump 虽完成，decode 随后 OOM，未取得完整生成/logprob 基线（1857-1860）。因此不能声称 PP1 TP32 全模型会正常生成。

## 后续有限精度通过（2026-08-28/29）

- DP2×PP4×TP16 的固定 greedy 单请求，93 层，使用 DP attention、DeepEP 与 attention-TP 相关设置；迁至 `k3cppp` 后，两个旧 env 设置改为 CLI flag，记录称回显为 true（1903-1936、1970-1971）。
- `k3cppp@ccaf0adc96` 加 kernel `f89ddee`（含 `f3903b3`）的一次对照：两侧 115 token，文本逐字符一致，仅三处分词边界差异，`finish_reason=stop`，logprob 健康（1968-1975）。可引用工件：`dp2-pp4-tp16-nodump-2026-08-28_19-53-00.txt`、`dp2-pp4-tp16-k3cppp-2026-08-29_13-31-43.txt`。
- 新 wheel 单请求与旧 wheel token 一致、logprob 差异为噪声级；当时“determinism rerun 通过”已被更正为**未通过**，因为 rerun 响应未完成（1977-1997）。

## 不能外推的结论

- 未隔离 DP attention、DeepEP、shared-experts attention-TP、dense-MLP attention-TP 的因果；不能声称三项开关缺一必然乱码。
- 单请求通过不覆盖多请求、长上下文、PD 传输、所有 DP 路由、性能或连续稳定性。
- 版本、wheel、脚本与节点配置均可能漂移；尤其应确认实际导入的 wheel 及每节点脚本一致性。

## 建议的复验顺序

1. 记录源码 commit、实际 wheel、拓扑、层数、TP/PP/DP、开关与请求参数。
2. 先关闭 dump，完成 PP1/PP>1 的固定 greedy 生成和 token/logprob 比较；再单独开启 dump 定位边界张量。
3. 独立测试连续请求、长上下文和 PD；不要以 health 或 HTTP 200 代替精度或稳定性。

相关 dtype 事件见[conv-state dtype 案例](2026-08-k3-conv-state-dtype.md)。
