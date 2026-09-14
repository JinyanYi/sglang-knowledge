# Decode tiling 失败：`HCCL_BUFFSIZE is too SMALL`

- **日期**：2026-08-10 排查；2026-08-13 补充满血 PD 情况
- **环境**：Ascend 910 A3；Kimi-K3 W4A8；PD 分离 **decode 侧**；**low-latency** dispatch；
  EP8（`maxBs=64`）实际；`hidden=7168`、`num_experts=896`、`topk=16`
- **结论**：`HCCL_BUFFSIZE` 给小了，撑不住 low-latency 的窗口公式。
  **不是 bug，是配置量算错**。

> 相关：[`../deep-ep/SKILL.md`](../deep-ep/SKILL.md)、
> [`2026-08-deepep-prefill-507015-notify-dispatch-ub.md`](./2026-08-deepep-prefill-507015-notify-dispatch-ub.md)（同一条窗口/尺寸链路）、
> [`../../knowledge/memory/memory-budget-and-scheduling.md`](../../knowledge/memory/memory-budget-and-scheduling.md)（leftover 怎么算）。

---

## 症状

decode 用 low-latency dispatch 时，**tiling 直接失败**：

```
[MoeLowLatencyDispatchV2] HCCL_BUFFSIZE is too SMALL, maxBs=64, h=7168, epWorldSize=8,
localMoeExpertNum=112, k=16, NEEDED=1733MB, HCCL_BUFFSIZE=800MB
```

注意日志已经把 **NEEDED** 算给你了 —— 直接照着调即可。

---

## 根因

low-latency 的窗口公式（`moe_distribute_dispatch_v2_tiling.cpp` 的 `CheckWinSize`）：

```
actualSize = (maxBs × dispatch_size × ep × local_experts
            + maxBs × combine_size × (k + shared)) × 2
```

896 专家 + topk16 下的实际需求：

| 配置 | maxBs | 需求 |
|---|---|---|
| EP8（decode 实际） | 64 | **~1.73 GB** |
| EP4 | 128 | **~3.3 GB** |

脚本只给了 `HCCL_BUFFSIZE=800`（MB），远不够。

---

## 修复 & 验证

```bash
export HCCL_BUFFSIZE=2048   # decode 节点：800 → 2048（按 MiB 计）
# EP4 的测试环境要 4096
```

- `HCCL_BUFFSIZE` 经 `mc2_tiling_utils.h` 读取，`uint16_t`，2048/4096 都无溢出问题。
- **改完必须重启进程** —— 该值在**启动时**读取，运行中改无效。

### 本地复现

```bash
python3 test_low_latency.py --num-processes 4 --num-tokens 128 \
  --hidden 7168 --num-topk 16 --num-experts 896 --quant-type no
# 预期：HCCL_BUFFSIZE too SMALL
```

---

## 为什么「关了 graph 才暴露」

**graph 复播跳过 tiling**，只有 eager 才每次做 tiling。
所以同一份配置，开 graph 能跑、关 graph 就报错 —— 不是 graph 的问题，
是 eager 才把尺寸校验走了一遍。

---

## 满血 PD 的遗留情况（2026-08-13）

EP16 + `HCCL_BUFFSIZE=2048` 时，可能**不再报** `too SMALL`，
而是变成：

```
HcclAllocComResourceByTiling ret=24
```

**这是另一回事**：leftover（KV 分完后 ~4.7GB @ `f=0.92`）不够 HCCL **整块**申请，
**不是**公式的 1733MB 没盖住。

处理方式（二选一）：

1. 降 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`（即 `maxBs`），
   并**相应调低** `HCCL_BUFFSIZE`；
2. 把 `--mem-fraction-static` 降到刚好比 `1 − post/pre` 高一点，以**增大 leftover**。

leftover 怎么算见
[`../../knowledge/memory/memory-budget-and-scheduling.md`](../../knowledge/memory/memory-budget-and-scheduling.md)。

---

## 一句话给下一个 agent

日志里的 **NEEDED** 就是答案，照着调；改完**重启进程**。
若报错变成 `HcclAllocComResourceByTiling ret=24`，那是 leftover 不够整块申请，
**再加大 `HCCL_BUFFSIZE` 也没用**，得降 `maxBs` 或腾 leftover。
