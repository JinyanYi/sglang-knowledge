# Kimi-K3：导航、当前状态与证据边界

> 默认从本页按主题进入；只读文档和归档，不执行脚本。历史笔记中的错误断言会被保留，
> 但不是当前权威。

维护日期：2026-09-10。本页“当前”指这次文档整理，不表示远端运行状态已刷新。
先读本页和一个主题即可；需要原始证据时才展开归档。

## 先按问题选择唯一正文

### 理解架构与数据所有权

- [K3 层结构与 block residual](../../knowledge/models/kimi-k3-layer-structure.md)
- [K3 latent-MoE / EP](../../knowledge/models/kimi-k3-latent-moe-ep.md)
- [DSpark 在 PP/P-D 中的 draft KV](../../knowledge/models/kimi-k3-dspark-pp-pd.md)

这些 `knowledge/` 页面是架构事实的唯一正文；不要从实验日志反推实现已经成立。

### 调整 PP/P-D 脚本或拓扑

- [K3 P/D + PP runbook](../../runbooks/k3-pd-pp/SKILL.md)
- [PP partition 与裁层决策](../../runbooks/k3-pd-pp/pp-partition-and-cut.md)
- [拓扑变更检查单](../../runbooks/k3-pd-pp/topology-checklist.md)

这些 `runbooks/` 页面是操作决策的唯一正文。变更前先核对实际导入的源码、模型配置、
版本与所有节点最终参数；不要执行归档中出现的命令。

### 分析精度、PP 或 DeepEP 异常

- [PP divergence 取证方法](../../debugging/pp-divergence/SKILL.md)
- [K3 PP 精度与逐层证据](../../debugging/cases/2026-08-k3-pp-precision.md)
- [K3 conv-state dtype 回归](../../debugging/cases/2026-08-k3-conv-state-dtype.md)
- [K3 DeepEP 后续请求挂起](../../debugging/cases/2026-08-k3-deepep-followup-hang.md)
- [K3 P/D 拓扑与配置边界](../../debugging/cases/2026-09-k3-pd-topology-config.md)

这些 `debugging/` case 是故障结论和证据边界的唯一正文。特别是 DeepEP 的历史
`auto/normal` 观察不能外推到不同拓扑；静态核查、单请求成功、连续稳定和精度通过也
不能互相替代。

## 当前可用结论的简表述

- PP 的 hidden/residual 实验曾确认一次 PP send/recv 数值对齐；不能据此排除所有数值漂移，
  也不能归因“PP 传坏”。
- 某一四节点 `TP16/PP4/DP2` 配方有固定请求精度对齐与单请求 wheel 回归；它不是新环境的
  连续稳定性保证。
- 七节点 3P4D 是历史脚本/静态核查记录，尚非该页写作时的重新运行验收；`TP48/PP1` 与
  当前 DeepEP 自动 EP 的专家整除约束冲突。

## 需要追溯时

进入 [`evidence-map.md`](./evidence-map.md)，按主题选择一个归档片段。归档仅保留笔记
文字与来源行段；**源日志、权重、运行环境和容器内容没有随 TXT 真实转存**。原始
`K3/209-note.md` 未被修改。
