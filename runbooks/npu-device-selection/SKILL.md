---
name: npu-device-selection
description: >-
  Pick free Ascend NPUs and non-conflicting ports before launching SGLang runs:
  npu-smi inspection, Phy-ID vs card index, --base-gpu-id mapping, avoiding stale
  HBM on just-released cards, and coexistence with other users' jobs. Use before
  any benchmark, repro, or multi-job run on a shared NPU host.
---

# NPU 巡检与选卡

> 每次起服务 / 跑 bench / 复现之前先做这一步。
> 选错卡的代价：性能数字系统性偏低，且极难归因。

## §1 查占用

```bash
npu-smi info                      # 找 "No running processes found in NPU X"
npu-smi info -t board -i <id>     # 单卡详情
ss -lnt | grep -E ':(30001|5001|5002|9930|9999)\b' || true
```

**必须确认两件事都没撞**：NPU 空闲 **且** 端口空闲。只查一个会翻车。

## §2 Phy-ID ≠ 卡槽序号（最容易搞错）

`--base-gpu-id` 接受的是 **Phy-ID**，不是 `npu-smi` 里 NPU 0/1/2/3 的逻辑序号。

**映射关系因机器拓扑而异**，必须看该机 `npu-smi info` 的实际输出确认。已知样例：

| 机器拓扑 A | |
|---|---|
| NPU 0 | `--base-gpu-id 0` |
| NPU 1 | `--base-gpu-id 2` |
| NPU 2 | `--base-gpu-id 4` |
| NPU 3 | `--base-gpu-id 6` |

| 机器拓扑 B（Phy-ID 直接可用） | |
|---|---|
| Phy 6,7 | `--base-gpu-id 6` |
| Phy 8,9 | `--base-gpu-id 8` |
| Phy 10,11 | `--base-gpu-id 10` |

> 不要跨机器照抄映射表。看到别人的脚本里有 `NPU1→2` 就以为通用，是最常见的错误。

**TP2 需要两个连续 Phy-ID**。选卡时按对选（8/9、10/11…），不要选 8/10。

## §3 避开「刚释放」的卡

**症状**：卡看起来空闲（`No running processes`），但首跑性能异常差（TPOT 炸、吞吐腰斩），
第二次第三次才恢复正常。

**原因**：上一家的 HBM 还没完全释放 / 相邻卡仍在忙。

**处理**：

1. 优先选 `npu-smi` 明确报 `No running processes` **且**旁边那张也空的成对 Phy。
2. 避开刚跑完的卡，或**再观察一轮**确认稳定空闲。
3. **至少跑 3 次**，不要把首跑当稳态 → 详见
   [`nightly-local-repro`](../nightly-local-repro/SKILL.md) 的 RUNS=3 约定。

## §4 多 job 共存

同机跑多个任务时，每个 job 都要独立：

| 资源 | 隔离方式 |
|---|---|
| NPU | 不同的成对 Phy |
| HTTP / dist 端口 | 不同端口（`ss -lnt` 确认） |
| HCCL 端口段 | `HCCL_NPU_SOCKET_PORT_RANGE`，不同 job 错开 |

```bash
export HCCL_NPU_SOCKET_PORT_RANGE=36100-36150   # 每个 job 不同
```

多机场景还要错开 `HCCL_IF_BASE_PORT`（每段 32 个端口）→
见 [`../../debugging/hccl/`](../../debugging/hccl/SKILL.md) §1。

## §5 起不来 / 一发请求就挂

| 症状 | 判断 | 去哪 |
|---|---|---|
| `Communication_Error_Get_Socket` + `HcclAllreduce` | **该 Phy 对的 link 故障** | 换一对，别调参数（`[../../debugging/hccl/](../../debugging/hccl/SKILL.md) §2`） |
| `EI0020` / 端口 bind 失败 | 端口段冲突 | 同上 §1 |
| `ModuleNotFoundError: tbe` / `500001` | 环境问题，不是卡的问题 | [`../npu-docker-env/`](../npu-docker-env/SKILL.md) |

**关键**：换卡能解决的是「link 故障」；环境问题换卡没用，别在换卡上浪费时间。

## 踩坑账本

| 坑 | 症状 | 处理 |
|---|---|---|
| 把 NPU 序号当 Phy-ID | 起不来 / 用了别人在跑的卡 | 看 `npu-smi info` 实际输出 |
| 跨机器照抄映射表 | 同上 | 映射因拓扑而异 |
| 只查 NPU 不查端口 | 端口撞车，bind 失败 | 两者都查 |
| 选了刚释放的卡 | 首跑性能异常差 | 观察一轮 / RUNS=3 不看首跑 |
| TP2 选了不连续的 Phy | 通信异常 | 按对选连续 Phy |
| link 故障时调参数 | 反复失败 | 直接换 Phy 对 |

## Living doc

新机器拓扑 → 把该机的映射表补到 §2；新的选卡陷阱 → 补「踩坑账本」。
