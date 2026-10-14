# K3 P/D + PP 拓扑变更检查单

> 适用：K3 的 P/D、PP、DP、DeepEP 变更。背景与证据边界见
> [`../../debugging/cases/2026-09-k3-pd-topology-config.md`](../../debugging/cases/2026-09-k3-pd-topology-config.md)。
> 项目导航与历史证据见 [`../../projects/kimi-k3/`](../../projects/kimi-k3/README.md)。

## 改脚本前

- [ ] 定义一个拓扑；不要把七节点 3P4D、八节点讨论和历史四节点回归混为同一次验收。
- [ ] 核对每个参与节点的真实模型配置，特别是 `attn_res_block_size=12`；全部节点必须一致。
- [ ] 记录实际 `PYTHONPATH`/容器导入路径、SGLang commit、kernel/DeepEP 版本，不能只看宿主 checkout。
- [ ] PP1：清除 `SGLANG_PP_LAYER_PARTITION`；PP>1：核对所有节点的最终同一 partition。
- [ ] 核对 `TP % DP == 0`、attention head 约束、DeepEP EP 专家整除和实际可用设备数。
- [ ] 按 DP 分摊后再检查 `chunked-prefill-size` 能被实际 page size 整除。
- [ ] DP1 保留 `--enable-dp-attention --enable-dp-lm-head`，并确认代码含保留显式 flag 的修复；DP2 也保留两项。
- [ ] 核对 P、D、router 的 HTTP、dist-init、bootstrap、网卡和可达性；不要只测 HTTP 端口。

## 启动与验收

- [ ] 完整重启受影响的分布式实例；配置恢复不完整重启不构成验证。
- [ ] `--skip-server-warmup` 只用于缩短服务 warmup；另做一次真实 P→D 请求。
- [ ] 固定输入、采样设置和模型版本，保存文本、token、所有 rank 日志和最终参数回显；仅在当前接口支持时保存 logprob（历史 DSpark 路径曾明确不支持）。
- [ ] 依次验证：ready → 固定单请求 → P→D 传输 → 连续请求 → 精度/长上下文。
- [ ] 将“源码核查”“单请求成功”“连续稳定”和“精度通过”分别记录，不互相替代。

## DeepEP 特别提醒

历史四节点 `auto/normal` 观察不能直接解释新的 P/D 拓扑。若出现挂起，采集当前各 rank
日志、版本与通信参数后再定位；不要先把静态 low-latency 分析写成根因。
