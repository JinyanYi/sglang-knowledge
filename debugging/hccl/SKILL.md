---
name: hccl-multinode
description: >-
  Diagnose HCCL communication failures on Ascend NPU multi-node/multi-job
  SGLang runs: port range exhaustion (EI0020, hccl port 用满), base port
  misconfiguration, HcclAllreduce socket errors from bad NPU links, HCCL_BUFFSIZE
  defaults and fallback. Use when HCCL init errors, port bind failures, or
  collective hangs appear on 2-8 node runs.
---

# HCCL / 多机通信故障

> 相关：[`../deep-ep/`](../deep-ep/SKILL.md)（kernel 卡死 / 507014）、
> [`../../runbooks/npu-device-selection/`](../../runbooks/npu-device-selection/SKILL.md)（选卡）。

## §1 端口耗尽（`EI0020` / `Communication_Error_Bind_IP_Port`）

### 症状

```
Communication_Error_Bind_IP_Port(EI0020)
```
或日志里出现 `65536` 这种**非法 TCP 端口**。

### 关键认知（最常被搞错的三条）

1. **`65536` 不是被占用的端口**，是 HCCL 端口搜索撞到上界后的失败值。看到它就别再去 `ss -lnt` 找谁占了 65536。
2. **官方默认不是 `65536+`，是 `60000-60031`**。老脚本里「HCCL 从 `HCCL_IF_BASE_PORT` 默认 65536 起，检查 65536..65550」是**错误结论**，见到就删。
3. **报错发生在 HCCL lazy init 阶段不等于 lazy init 是根因**。首次 PP `isend` 才触发 `hcclGetRootInfo`，`ProcessGroupHCCL` 按需建 communicator。加不加打印只改变时序，不决定 init 与否。

### 参数事实

| 项 | 值 |
|---|---|
| `HCCL_IF_BASE_PORT` 合法范围 | `1024 – 65520` |
| HCCL 从 base 起占用 | 连续 **32** 个端口 |
| 同一作业各机 | 配**相同**值 |
| 不同并行作业 | 必须**不重叠**的端口段 |
| OS 侧 | 必须 `ip_local_reserved_ports` 预留，否则会被普通 TCP 抢 |

### 排查顺序（按这个走，别跳）

```bash
# 1. 有没有残留作业占着资源
ss -s                                   # 看总连接数
ss -lnt | wc -l                         # 监听端口数
sysctl net.ipv4.ip_local_reserved_ports # 空的 = 没预留，问题高发

# 2. 清理脚本是否真的匹配到了进程
pgrep -af 'sglang'                      # 现在多是 `sglang serve`
```

**坑**：清理脚本常只匹配 `sglang.launch_server`，而实际启动命令是 `sglang serve` → 脚本报成功但进程没死。
修正后要**检查退出码非零即失败**，不要静默继续。

### 修复

```bash
# 四台机统一设置，选一个不撞 HTTP / dist-init / 其他服务的段
export HCCL_IF_BASE_PORT=55000
sysctl -w net.ipv4.ip_local_reserved_ports=55000-55031
```

多 job 并行时每 job 错开 32 的整数倍（`55000` / `55032` / `55064` …）。
单机多 job 另见 `HCCL_NPU_SOCKET_PORT_RANGE`（见 [`../../runbooks/nightly-local-repro/`](../../runbooks/nightly-local-repro/SKILL.md) 踩坑账本）。

### 什么时候才怀疑代码

**彻底独占 + 完全清理后仍稳定复现**，才去查「PP 是否重复创建了 HCCL process group」。
在此之前归因到代码是浪费时间。

---

## §2 `Communication_Error_Get_Socket` / `HcclAllreduce` 失败

### 症状

服务起来了，但一发请求就 `Connection reset` 或进程自杀，栈里是：

```
Communication_Error_Get_Socket ... HcclAllreduce
```

### 判断

这是 **NPU 之间那条 link 的物理/链路故障**，不是端口、不是参数、不是代码。

### 处理

**换一对空闲 Phy 重启**（例如 8/9 → 4/5）。
不要去调 `HCCL_BUFFSIZE`、不要改 port、不要改 `--base-gpu-id` 之后在同一对卡上重试。

选卡方法见 [`../../runbooks/npu-device-selection/`](../../runbooks/npu-device-selection/SKILL.md)。

---

## §3 `HCCL_BUFFSIZE`：默认值会骗人

### 事实

- 没显式设置时，按 `parallel_state.py` 的回退逻辑，**实际用的是默认 200**（MB）。
- 看到「没设 HCCL_BUFFSIZE」不要以为用了某个大值——先去代码确认回退值。

### 怎么定

```bash
export HCCL_BUFFSIZE=512   # 示例；3000 这类大值在 tight memory 下过重
```

权衡：越大通信越快，但吃显存。显存紧时调大反而 OOM 或挤压 KV 池
（连带 `max_running_requests` 被下调，见下）。

### 连带效应

`HCCL_BUFFSIZE` / 显存估算导致并发被砍时，日志会打：

```
max_running_requests was reduced from 103 to 17
```

**这不是错误，但会毁掉你的性能数字**：TTFT 系统性偏高、吞吐系统性偏低。
任何性能对比报告里必须写明这一行，否则结论不可信。

---

## 踩坑账本

| 坑 | 症状 | 处理 |
|---|---|---|
| 老脚本写「HCCL base 65536+」 | 照着查 65536 谁占用，查不到 | 删掉该结论；官方默认 `60000-60031` |
| 把 `65536` 当被占端口 | 反复 `ss`/`lsof` 无果 | 它是搜索上界失败值 |
| 以为 lazy init 是根因 | 加/去打印，问题依旧 | 打印只改时序；先查资源占用 |
| 清理脚本漏 `sglang serve` | kill 报成功，进程还在 | `pgrep -af 'sglang'` 核对；检查退出码 |
| `ip_local_reserved_ports` 为空 | 端口被普通 TCP 抢占 | `sysctl -w` 预留，且四台一致 |
| `HcclAllreduce` socket 失败 | 一发请求就 reset | 换 Phy 对，别调参数 |
| 未显式设 `HCCL_BUFFSIZE` | 以为用了大 buffer | 默认回退 **200**；查 `parallel_state.py` |
| `max_running_requests` 被砍 | TTFT 高、吞吐低 | 不是 bug，但报告必须注明 |

## Living doc

新增 HCCL 报错 → 先补到本文「踩坑账本」；同一根因出现 ≥2 次且有完整时间线 →
抽成 [`../cases/`](../cases/) 独立 case。
