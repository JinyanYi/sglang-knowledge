---
name: deep-ep-hang
description: >-
  Diagnose DeepEP / MoE a2a hangs and AI Core timeouts on Ascend NPU: silent
  hang with 200 OK but zero progress, error 507014 (ACL_ERROR_RT_AICORE_TIMEOUT),
  _intranode_dispatch trap, and the plog / py-spy / coredump evidence chain.
  Also covers which build flag compiles notify_dispatch.h.
---

# DeepEP：hang 与 AI Core 超时

> 概念部分（`--moe-a2a-backend` 语义、normal/low-latency dispatch 机制）
> 见 [`../../knowledge/npu/deepep-ascend-npu.md`](../../knowledge/npu/deepep-ascend-npu.md)。
> 本文只讲**怎么查**。

## §0 先分清「真超时」还是「卡死后被超时兜住」

错误文本 `errorStr: timeout or trap error` **两种都有可能**：

- 真慢 → 等一会能过
- kernel **trap / 死锁** → 永远不会返回，被超时机制兜住报出来

**判断方法**：去 plog 看 timeout 模式（§2）。30 秒内能定性。

---

## §1 silent hang：最容易被误诊的一种

### 症状

- sglang stdout **一直 `200 OK`**，health check 正常
- 但 bench 的 tqdm 卡在 `0/200` 不动
- `npu-smi` 显示 **1 张卡 AICore 100%，其余 3 张 0%**
- 最后 HTTP retry 雪崩 → `AssertionError: 0.0`

### 定性

**1 张 100% + 其余 0%** = 发起方在自旋等，其余在等通信 → **collective 死锁**。

> 此时**不要**先去抓 coredump（每个 20+ GB），先翻 plog。

---

## §2 plog 取证（第一手段）

### 文件在哪

| 路径 | 价值 |
|---|---|
| `/root/ascend/log/run/plog/plog-<PID>_*.log` | **主用**：RUNTIME / HCCL / HCCP 详细日志，每 PID 一份 |
| `/root/ascend/log/debug/plog/` | 几乎没用（167 B 或 0 B），仅 SIGABRT 时多一行 |
| `/root/ascend/log/run/device-N/` | 硬件错误（507014 等）查这里 |

PID 用 `ps -ef | grep scheduler_TP` 拿。

### 拷出来（< 10 MB，全拷）

```bash
mkdir -p "$HOME/nightly_repro/coredump/plog"
for pid in $(docker exec <CONTAINER> ps -ef | grep 'scheduler_TP' | grep -v grep | awk '{print $2}'); do
  src=$(docker exec <CONTAINER> ls /root/ascend/log/run/plog/plog-${pid}_*.log 2>/dev/null | head -1)
  docker cp <CONTAINER>:$src "$HOME/nightly_repro/coredump/plog/plog-TPx_EPx.${pid}.log"
done
```

### 搜什么

| Pattern | 含义 |
|---|---|
| `SynchronizeExecutedTask: report three minutes timeout!` | RUNTIME 每 3 分钟报一次。**连续 2 次 `task_id`/`sqHead`/`pendingNum` 不变 = 100% 死锁** |
| `stream_id=N, sq_id=M, task_id=K, sqHead=H, pendingNum=P` | 死锁 task 精确位置。**4 个 scheduler 卡在同一 `stream_id=45` 且 `task_id` 一致 = 同一个 collective 死锁** |
| `TryToRecycleProgPool` / `TryToRecycleModulesPool` | runtime 尝试回收，**失败** = 没人能动这个 task |
| `create hccl transport:communicator[...], local rank[R], remote rank[R']` | 建连记录。dispatch hang 时常看到 `ReduceScatter_group_name_3` / `AllGather_group_name_3*` 之后**再无新条目** |
| `[ERROR]` / `[FATAL]` / `507014` | 显式 driver 错误。**可能没有**（silent hang 时就没有） |

### 4-scheduler 对比

```bash
for f in $HOME/nightly_repro/coredump/plog/plog-*.log; do
  echo "--- $(basename $f) ---"; grep "three minutes timeout" "$f"
done
```

- **task_id 相同** → 被动方，等同一个 collective
- **task_id 不同** → 发起方 / 不同阶段
- 几次 timeout 之间 task_id 不变 → 真死锁（不是慢）

### plog vs sglang stdout

| 信息 | sglang log | plog |
|---|---|---|
| fired up / throughput | ✅ | ❌ |
| **HCCL collective 状态 / 死锁** | ❌ | ✅ |
| **AI Core / stream 真实在做什么** | ❌ | ✅ |
| driver 级 507014 / EZ 错误 | 偶有透传 | ✅（device 子目录更全） |

---

## §3 `507014` vs `507015`（别搞混）

| 错误码 | 含义 | 场景 | 去哪 |
|---|---|---|---|
| **507014** | `ACL_ERROR_RT_AICORE_TIMEOUT` — **没返回**（超时/trap） | DeepEP hang | 本文 §3.1 |
| **507015** | `MPU address access is invalid` — **写越界了**（AIVector 非法访问） | prefill 首轮崩，多轮 UB 分配 | [`../cases/2026-08-deepep-prefill-507015-notify-dispatch-ub.md`](../cases/2026-08-deepep-prefill-507015-notify-dispatch-ub.md) |

一个是「等不到」，一个是「写坏了」，排查方向完全不同。

## §3.1 `507014`：`ACL_ERROR_RT_AICORE_TIMEOUT`

### 典型栈

```
deep_ep/strategies/normal_strategy.py:120   dispatch
  → :231  _intranode_dispatch
    → runtime.intranode_dispatch
RuntimeError: ... AclrtSynchronizeStreamWithTimeout(copy_stream), error code is 507014
[Error]: The aicore execution times out.
EZ9999 ... fftsplus aicore error, core id is N ... errorStr: timeout or trap error
```

### 读法

| 字段 | 含义 |
|---|---|
| `507014` | `ACL_ERROR_RT_AICORE_TIMEOUT`，AI Core 执行超时 |
| `The aicore execution times out` | Runtime 明确报 AIC 超时 |
| `timeout or trap error` | 真超时 **或** trap 后被超时兜住（两者都可能） |
| `Device:N` / `chipId:N, dieId:N, core id is N` | 出问题的具体 NPU 与 AI Core |
| `retCode=0x25, [aicore timeout]` | Device 侧 kernel 任务超时 |

Host 是在 `AclrtSynchronizeStreamWithTimeout` 上**等 kernel 返回**等超时了 ——
即 kernel 没在预期时间内完成，不是 CPU 侧 recv 超时，更不是 CI 墙钟超时。

---

## §4 常见诱因

```
--moe-a2a-backend deepep --deepep-mode auto
  + DEEP_NORMAL_MODE_USE_INT8_QUANT=1
  + Qwen3 系列 MoE w8a8
```

**已验证无效的手段**：`--base-gpu-id N` 重启换卡**绕不开**。

快速验证是不是 DeepEP 引起：`--moe-a2a-backend none` 关掉 DeepEP 回到 TP 内 MoE。
若关掉就不崩，问题在 DeepEP kernel 层，不是 SGLang 集成层。

---

## §5 拿 Python 栈：`py-spy dump`

plog 确认死锁后，用 py-spy 定位到具体代码行：

```bash
py-spy dump --pid <scheduler_pid>
```

**典型落点**：

```
deep_ep/strategies/normal_strategy.py:250  _intranode_dispatch
locals: quant_type="int8"  use_quant=True
```

看到 `use_quant=True` 就要怀疑 int8 量化路径（对照 `DEEP_NORMAL_MODE_USE_INT8_QUANT`）。

---

## §6 拿 native C 栈：coredump（最后手段）

```bash
# ⚠️ 用 SIGABRT，不要用 SIGQUIT —— Python 默认 SIGQUIT handler 会吞掉信号，不出 core
for pid in $(docker exec <CONTAINER> ps -ef | grep scheduler_TP | grep -v grep | awk '{print $2}'); do
  docker exec <CONTAINER> kill -ABRT $pid
done
sleep 30
docker exec <CONTAINER> ls -lh /tmp/core-file/   # core_pattern: /tmp/core-file/core.%t.%e.%p
```

**磁盘**：每个 core 20+ GB（含 30 GB 权重），4 个共 80+ GB。
**保留 1 个**（`AICore 100%` 那个 = 发起方，最有诊断价值），其余 `rm -f`。

---

## §7 编译：`notify_dispatch.h` 归哪个 flag

| flag | 编什么 |
|---|---|
| `-a kernels` | **只**编 `sgl_kernel_npu` |
| `-a deepep` | 才编 `notify_dispatch.h` → custom OPP → `deep_ep` wheel |

改了 `notify_dispatch.h` 却用 `-a kernels` 编，会以为没生效。

---

## §8 版本与复现环境（2026-08-10 排查时）

| 组件 | 版本 |
|---|---|
| `deep_ep` wheel | `1.0.0+48942e37.cann.9.0.0.b250`（构建 commit `48942e37`，2026-07-27） |
| DeepEP OPP | `custom_opp_compiler_version=9.0.0`（`vendors/customize`） |
| `sgl-kernel-npu` HEAD | `feca6c9`（含 k3-pp 分支） |
| A3 专家数上限 | `MOE_EXPERT_MAX_NUM=1024`（`ops/`）；`ops2/`（A2）才是 512 |
| `notify_dispatch.h` 修复 | 在本机工作区，**未提交** |
| 复现容器 | `sglang-yjy`（`lmsysorg/sglang:main-cann9.0.0-a3`），`deep_ep` 与节点**完全同版**，可作本地复现环境 |

---

## §9 相关 case（非 hang 类）

本文只覆盖 **hang / 超时**。DeepEP 另外三个已知故障是独立 case：

| Case | 症状 | 错误签名 |
|---|---|---|
| [`2026-08-deepep-prefill-507015-notify-dispatch-ub.md`](../cases/2026-08-deepep-prefill-507015-notify-dispatch-ub.md) | prefill 首轮崩（多轮 UB 越界） | `507015` / `The MPU address access is invalid` |
| [`2026-08-deepep-decode-hccl-buffsize-too-small.md`](../cases/2026-08-deepep-decode-hccl-buffsize-too-small.md) | decode tiling 失败 | `[MoeLowLatencyDispatchV2] HCCL_BUFFSIZE is too SMALL` |
| [`2026-08-deepep-cuda-graph-bs-alignment.md`](../cases/2026-08-deepep-cuda-graph-bs-alignment.md) | graph 启动 assert | `assert len(capture_bs) > 0` |

概念与机制（`--moe-a2a-backend` 语义、两种 dispatch、三条机制性约束）见
[`../../knowledge/npu/deepep-ascend-npu.md`](../../knowledge/npu/deepep-ascend-npu.md)。

---

## 踩坑账本

| 坑 | 症状 | 处理 |
|---|---|---|
| 把 trap 当慢 | 傻等，永远不过 | plog 看 task_id 是否变化 |
| 先看 coredump | 80+ GB，慢 | 先 plog（30s 定性），再 py-spy，最后才 coredump |
| `kill -QUIT` 拿不到 core | 无 core 文件 | 用 `kill -ABRT`；Python 吞 SIGQUIT |
| 4 个 core 全留 | 磁盘爆 | 留 1 个（AICore 100% 那个） |
| 换 `--base-gpu-id` 想绕开 | 依旧 hang | 已验证无效；换 backend 或关量化 |
| 改了 `notify_dispatch.h` 没生效 | 行为不变 | 用 `-a deepep` 编，不是 `-a kernels` |

## Living doc

新的 DeepEP hang 模式 → 补本文；单次事故完整时间线 → [`../cases/`](../cases/)。
