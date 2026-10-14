# K3 P/D 拓扑与配置：七节点 3P4D 记录、约束和验收边界

- **日期**：2026-09-08/09
- **证据性质**：用户提供的脚本、已确认配置错误和本地源码静态核查；本次记录**未重新运行**七节点 NPU 测试。
- **脱敏约定**：节点仅按 P1–P3、D1–D4 表示；路径使用 `<MODEL_PATH>`、`<USER>` 占位符。

> 导航：[`projects/kimi-k3/evidence-map.md`](../../projects/kimi-k3/evidence-map.md)。
> 操作前检查：[`runbooks/k3-pd-pp/topology-checklist.md`](../../runbooks/k3-pd-pp/topology-checklist.md)。
> 归档来源：[`archive/2026-09-09-note/README.md`](../../projects/kimi-k3/archive/2026-09-09-note/README.md)，原始笔记 §11。

## 七节点 3P4D：历史输入，不是运行验收

当时脚本的目标为：Prefill 三节点 `TP16 / PP3 / DP2`、partition `31,31,31`；Decode 四节点
`TP64 / PP1 / DP4`。Prefill attention-TP 为 8、Decode 为 16；DeepEP 由框架自动调整为
EP16/EP64。两侧均传 DP attention、DP LM-head、shared-expert/dense-MLP attention-TP 开关；
P 禁图，D 保留图参数，且两侧配置了 `--skip-server-warmup`。

这描述的是**脚本输入和静态参数关系**。它不证明服务 ready、真实 P→D KV 传输、精度、
连续请求或图捕获已经通过；实际 backend、捕图与导入 checkout 都须由该次运行日志确认。
原始行：2063–2073。

## 已确认的配置错误：AttnResidual bank 宽度不一致

P2、D2 的模型配置曾将 `attn_res_block_size` 改为 2，而其他节点为 12。对 `31,31,31`：

```text
P2 end_layer=62: ceil(62 / 2)  = 31
P3 end_layer=93: ceil(93 / 12) = 8
```

因此发送 bank `[8,31,7168]`、接收端容量 `[8,8,7168]`，精确解释 shape mismatch。所有节点
恢复为 12 时，三个 stage 的宽度为 3/6/8。该 bank 是 attention residual snapshot，**不是
KV cache**。恢复配置后必须完整重启受影响实例；对话没有提供恢复后的七节点精度验收。

不要重新归因到 partition：此前“partition 不一致”或“31 是错误层分配”是已否定的早期猜测。
原始行：2102–2117。

## 参数约束（静态核查）

- 开启 DP attention、未启 CP 时，attention-TP=`TP/DP`，所以 TP16 不可配 DP3；仅此规则下
  DP1/2/4/8/16 合法，其他 head 等限制仍要单独检查。
- 现有 DeepEP auto 下 EP=TP。三台 P 改为 `TP48 / PP1` 时，`896 % 48 = 32`，专家不能均分；
  DP3 不会把 EP48 变为 EP16，显式 `--ep-size 16` 也不能绕过该自动调整。
- CLI `chunked-prefill-size` 会先按 DP 整除，再验 page 对齐。例如 DP3 下 4096 变 1365；
  应选 `DP × page_size × 整数`。6144 仅在 page_size=128 或 256 时是示例，不是已确认默认值。
- DP1 非 scale 时，旧实现会覆写显式 DP flags；`pdcpp@b54f247f18` 保留显式 flags，但这只是
  源码/隔离函数验证，**不是**七节点 NPU 精度验证。相关先例：`k3cppp@72f575f93d`。

原始行：2075–2094、2119–2164。

## warmup、脚本漂移与验收

`--skip-server-warmup` 是正确参数：它跳过 `_execute_server_warmup` 的短请求，不能跳过权重、
通信初始化或捕图，也不能作为真实 P→D KV 传输的验收。首次真实请求仍可能触发懒初始化。

脚本中的 `SGLANG_PP_LAYER_PARTITION` 可继承环境值；要检查最终生效值。PP1 时清除该变量，
PP>1 时使所有节点使用同一最终 partition。P/D/router 的端口、网卡、bootstrap、源码导入
路径和权重配置也必须以运行进程为准。单条 curl 成功不能代表多请求、各 DP 路由、传输与
长上下文精度通过。

四节点混部的 DeepEP auto/normal 观察不适用于本页七节点 P-only/D-only 拓扑的根因判断。
原始行：2166–2188。

## 8 节点 4P4D：仅讨论，未运行

另一份讨论中的 4P4D/八节点构想不属于上述七节点实验，也没有运行验收。不得把它写成
“七节点的后续成功配置”，更不能将其参数、端口或推断结果反填到 3P4D。若采用新拓扑，
应作为新实验单独记录：完整参数、真实提交、节点分配、固定输入、P→D 传输、精度与连续
请求均须重新验证。

## 可检索依据

本地静态核查涉及 `python/sglang/srt/` 下的 `arg_groups/overrides.py`、`parallel_hook.py`、
`validation_hook.py`、`layers/moe/token_dispatcher/deepep.py`、`models/kimi_k3.py`、
`layers/attn_residual.py`、`entrypoints/http_server.py`。原始记录还引用 PR head
`2526297b77ebb124ee709712c60629479ad3c113`；它是静态审查对象，不代表运行服务已使用该代码。
