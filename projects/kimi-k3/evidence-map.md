# Kimi-K3 历史证据地图

> 用途：从一个明确主题定位一个脱敏的原始笔记片段。不是递归搜索入口，也不是当前
> 配置或运行结论。归档保留旧错误断言以便审计；当前权威分别在 `knowledge/`、
> `runbooks/` 和 `debugging/cases/`。

归档说明见 [`archive/2026-09-09-note/README.md`](./archive/2026-09-09-note/README.md)。
这些 TXT 只保存笔记文字；源日志、权重、运行环境和容器内容均未真实转存，原始
`K3/209-note.md` 也未改动。

## 启动背景、早期修复与单机/PP1 记录

- [01-bootstrap-and-environment.txt](./archive/2026-09-09-note/01-bootstrap-and-environment.txt)
  — 原始行 1–338；启动背景与环境记录。**状态：历史上下文，需现场核验。**
- [02-conv-dtype-and-pp1.txt](./archive/2026-09-09-note/02-conv-dtype-and-pp1.txt)
  — 原始行 339–675；conv dtype、PP1 记录与早期推断。**状态：混合证据，勿直接复用。**
- [03-logprob-and-first-pp-dumps.txt](./archive/2026-09-09-note/03-logprob-and-first-pp-dumps.txt)
  — 原始行 676–1238；固定请求与首批 dump。**状态：探索性实验。**

## PP 数值与 AttnResidual

- [04-hidden-and-attnres-dumps.txt](./archive/2026-09-09-note/04-hidden-and-attnres-dumps.txt)
  — 原始行 1239–1810；原始 debug 输出。**状态：附录，不以 sum 猜测定因。**
- [05-same-tp32-pp-comparison.txt](./archive/2026-09-09-note/05-same-tp32-pp-comparison.txt)
  — 原始行 1811–1902；同 TP32、全层 PP1/PP2 对比。**状态：PP send/recv 对齐已记录；
  完整 PP1 生成因 OOM 未验证。**

对应当前 case：[K3 PP 精度与逐层证据](../../debugging/cases/2026-08-k3-pp-precision.md)。

## DP2、wheel 与 DeepEP 后续请求

- [06-dp2-wheel-and-deepep-hang.txt](./archive/2026-09-09-note/06-dp2-wheel-and-deepep-hang.txt)
  — 原始行 1903–2060；DP2/PP4 固定请求、wheel 单请求回归、挂起与 low-latency 假设。
  **状态：固定请求回归与现象记录；根因假设待复现。**

对应当前 case：
[conv-state dtype](../../debugging/cases/2026-08-k3-conv-state-dtype.md)、
[DeepEP 后续请求挂起](../../debugging/cases/2026-08-k3-deepep-followup-hang.md)。

## P/D 拓扑、配置与 PR 静态审查

- [07-pd-topology-and-pr-review.txt](./archive/2026-09-09-note/07-pd-topology-and-pr-review.txt)
  — 原始行 2061–2230；七节点 3P4D 输入、配置错误、约束与 PR 审查。
  **状态：源码/参数核查；非七节点 NPU 运行验收。**

对应当前 case：
[P/D 拓扑与配置边界](../../debugging/cases/2026-09-k3-pd-topology-config.md)。

## 读取规则

1. 先在项目 README 选一个主题。
2. 只有需要证据时打开本页对应的一项 TXT；从其指定行段读取，不扫全盘。
3. 将归档中命令、路径、提交和“根因”措辞视为待核验历史记录；执行前回到 runbook/case。
