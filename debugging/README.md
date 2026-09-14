# Debugging —— 挂了怎么查

**这一类只放「具体故障的排查」**：症状 → 根因 → 修复。
判据：**由一次真实的失败驱动**，读完之后你该能修好一个正在报错的东西。

| 不属于这里 | 该去哪 |
|---|---|
| 「PD 分离为什么存在、原理是什么」 | [`../knowledge/`](../knowledge/README.md) |
| 「怎么把 K3 服务正常起起来」 | [`../runbooks/`](../runbooks/README.md) |
| 「怎么复现 nightly 失败 case」 | [`../runbooks/nightly-local-repro/`](../runbooks/nightly-local-repro/SKILL.md)（操作） |

---

## 速查：报错串 → 去哪

先在这张表里搜你看到的报错。**找不到就补一行**（见文末「维护约定」）。

| 报错 / 症状 | 判断 | 文档 |
|---|---|---|
| `EI0020` / `Communication_Error_Bind_IP_Port` | 端口段被占 | [hccl/](./hccl/SKILL.md) §1 |
| `Communication_Error_Get_Socket` + `HcclAllreduce` | NPU 间 link 故障，换 Phy 对 | [hccl/](./hccl/SKILL.md) §2 |
| `hccl port` 用满 / 多 job 抢端口 | 端口段耗尽 | [hccl/](./hccl/SKILL.md) §1 |
| `SynchronizeExecutedTask: report three minutes timeout!` | AI Core 死锁（plog） | [deep-ep/](./deep-ep/SKILL.md) §1 |
| `507014` / `ACL_ERROR_RT_AICORE_TIMEOUT` | DeepEP kernel **没返回**（卡死/trap） | [deep-ep/](./deep-ep/SKILL.md) §3.1 |
| `507015` / `The MPU address access is invalid` | DeepEP **写越界**（多轮 UB） | [cases/](./cases/2026-08-deepep-prefill-507015-notify-dispatch-ub.md) |
| `[MoeLowLatencyDispatchV2] HCCL_BUFFSIZE is too SMALL` | 配置量不够，日志给了 NEEDED | [cases/](./cases/2026-08-deepep-decode-hccl-buffsize-too-small.md) |
| `HcclAllocComResourceByTiling ret=24` | leftover 不够**整块**申请 | [cases/](./cases/2026-08-deepep-decode-hccl-buffsize-too-small.md) |
| `assert len(capture_bs) > 0` | 开 deepep 后 bs 被对齐过滤空了 | [cases/](./cases/2026-08-deepep-cuda-graph-bs-alignment.md) |
| `CamMoeDispatchNormal do tiling failed, ret=-1` | `realRound≥2` 窗口校验（未决） | [cases/](./cases/2026-08-deepep-prefill-507015-notify-dispatch-ub.md) |
| `_intranode_dispatch` 卡住、`use_quant=True` | DeepEP normal 模式 int8 | [deep-ep/](./deep-ep/SKILL.md) §5 |
| `ModuleNotFoundError: tbe` / `500001` / `AclSetCompileopt` | PYTHONPATH 覆盖了 CANN | [../runbooks/npu-docker-env/](../runbooks/npu-docker-env/SKILL.md) |
| `GEInitializeV2 failed` / `Failed to init tbe` | 同上，且没 source set_env | [../runbooks/npu-docker-env/](../runbooks/npu-docker-env/SKILL.md) |
| 精度掉了、输出复读、score 暴跌 | 分层 dump 对比 | [precision/](./precision/SKILL.md) §1 |
| PP1 vs PP4 结果不一致 | 逐层分叉定位 | [pp-divergence/](./pp-divergence/SKILL.md) §1 |
| ViT + NPU Graph `seqlen mismatch` | 存的是 per-seq 而非 cumulative | [cases/2026-08-vit-npu-graph-cu-seqlens-tnd-mismatch.md](./cases/2026-08-vit-npu-graph-cu-seqlens-tnd-mismatch.md) |
| 服务 `200 OK` 但 progress 卡 0% | silent hang，翻 plog | [deep-ep/](./deep-ep/SKILL.md) §1 |
| `max_running_requests was reduced from X to Y` | 不是错误，但会毁掉性能数字 | [../runbooks/nightly-local-repro/](../runbooks/nightly-local-repro/SKILL.md) §7 |

---

## 目录

| 目录 | 处理什么 |
|---|---|
| [hccl/](./hccl/SKILL.md) | 多机/多 job 通信：端口段、Phy 对 link、`HCCL_BUFFSIZE`、buffsize 默认值回退 |
| [deep-ep/](./deep-ep/SKILL.md) | DeepEP：silent hang、507014、`_intranode_dispatch`、plog/ py-spy / coredump 取证链、notify_dispatch 编译 |
| [precision/](./precision/SKILL.md) | 精度：hidden dump → cosine/relative L2/max-mean abs → 定位分叉层；`grab_logprob` |
| [pp-divergence/](./pp-divergence/SKILL.md) | PP 分层：PP1 vs PP4 对齐、local buffer position vs global layer id、`attn_res_block_size` |
| [cases/](./cases/) | **单个 case 的完整复盘**（一次一件事，带时间线） |

### cases/ 的格式

每个 case 一个文件，`YYYY-MM-<slug>.md`，固定四段：

```markdown
# <一句话症状>

- 日期：
- 环境：<机型 / 分支 / 镜像 / 并行方式>
- 结论：<一句话根因>

## 症状        ← 看到的报错原文
## 排查过程     ← 怎么一步步缩小范围的（这最值钱）
## 根因
## 修复 & 验证
```

已有：

| Case | 症状 | 结论 |
|---|---|---|
| [`2026-hunyuan3-topk-scaling-factor.md`](./cases/2026-hunyuan3-topk-scaling-factor.md) | Hy3-preview 精度 17.5%（GPU 87.2%）+ 输出复读 | `fused_topk_npu` 少乘 `routed_scaling_factor` → 修复后 87.88% |
| [`2026-08-deepep-prefill-507015-notify-dispatch-ub.md`](./cases/2026-08-deepep-prefill-507015-notify-dispatch-ub.md) | prefill 首轮崩 `507015` | `notify_dispatch.h` 多轮 UB 越界（**不是** 512 专家上限） |
| [`2026-08-deepep-decode-hccl-buffsize-too-small.md`](./cases/2026-08-deepep-decode-hccl-buffsize-too-small.md) | decode tiling `HCCL_BUFFSIZE is too SMALL` | 窗口公式算出来的需求没满足；日志里的 NEEDED 就是答案 |
| [`2026-08-deepep-cuda-graph-bs-alignment.md`](./cases/2026-08-deepep-cuda-graph-bs-alignment.md) | `assert len(capture_bs) > 0` | 开 deepep 后对齐基数 = `attn_tp_size` |
| [`2026-08-vit-npu-graph-cu-seqlens-tnd-mismatch.md`](./cases/2026-08-vit-npu-graph-cu-seqlens-tnd-mismatch.md) | ViT + NPU Graph `seqlen mismatch` | 存的是 per-seq 差分，#25768 后应为 cumulative；**仅高并发多图 batch 暴露** |

---

## 维护约定（下个 agent 必读）

1. **一个坑只写一处**。同一个根因在 runbook 和 debug 里都出现时：runbook 写「照做」，debug 写「为什么 + 怎么定位」，用相对链接互指。
2. **新坑优先进 `cases/`**，攒够 2-3 个同类再抽成 `hccl/` `deep-ep/` 这样的专题。别一上来就建专题。
3. **报错串要原文**。下一个人是靠 grep 报错找到这里的，写「通信错误」这种转述等于没写。
4. **写清「不是什么」**。例：`max_running_requests was reduced` 不是错误，但会让吞吐数字系统性偏低 —— 这类「看起来像 bug 但不是」的判断最省时间。
5. 踩坑账本式的表格**只保留在 `../runbooks/nightly-local-repro/`**（它是按操作流程组织的），本目录用独立文件，不要两个地方各维护一份。
