# Runbooks —— 怎么把它跑起来

**这一类只放「操作步骤」**：照着敲就能把事情做成。
判据：**正常路径下的标准动作**，不需要先遇到一个 bug。

| 不属于这里 | 该去哪 |
|---|---|
| 「为什么这么设计 / 原理是什么」 | [`../knowledge/`](../knowledge/README.md) |
| 「挂了、报错了、怎么定位」 | [`../debugging/`](../debugging/README.md) |
| 「一次具体事故的复盘」 | [`../debugging/cases/`](../debugging/cases/) |

**边界约定**：runbook 只写「照做」，不写「为什么」。遇到坑就一行指过去：

> 若起不来并报 `ModuleNotFoundError: tbe` → 见 [`../debugging/`](../debugging/README.md)

同样，debugging 里的文档只写定位与根因，命令细节链回对应 runbook。

---

## 目录

| Runbook | 场景 | 状态 |
|---|---|---|
| [k3-pd-pp/](./k3-pd-pp/SKILL.md) | K3 多机 PD / Chunked PP 起服步骤 | ✅ 已有 |
| [k3-pd-pp/pp-partition-and-cut.md](./k3-pd-pp/pp-partition-and-cut.md) | PP 切点怎么选、裁几层、首版拓扑与参数速查 | ✅ 已有 |
| [git-workflow.md](./git-workflow.md) | git / PR 硬规则（不擅自 commit、多仓 stash/pull 顺序、自检） | ✅ 已有 |
| [npu-docker-env/](./npu-docker-env/SKILL.md) | 容器内 Python/CANN 环境：source 顺序、PYTHONPATH 追加 | ✅ 已有 |
| [npu-device-selection/](./npu-device-selection/SKILL.md) | `npu-smi` 巡检、选空闲 Phy 对、`--base-gpu-id` 映射 | ⚠️ **待补** |
| [nightly-watch/](./nightly-watch/SKILL.md) | 早上看护 public nightly，筛 `qwen3_6_35b_a3b` | ✅ 已有 |
| [nightly-local-repro/](./nightly-local-repro/SKILL.md) | 内网 NPU 机 docker 复现失败 case | ✅ 已有（§9 plog 取证已拆至 debugging） |
| [agent-tooling/](./agent-tooling/) | 本机 agent 相关路径地图（Cursor 数据落盘位置等） | ✅ 已有 |

## 关联脚本

| 脚本 | 配套 runbook |
|---|---|
| `scripts/kimi-k3-cpp/*.sh`（`cpp_pd.sh` / `run_pd_single_34.sh` / `run_pd_single_36.sh` / `run_prefill_34.sh` / `run_decode_38.sh` / `smoke_pp_cut.sh` / `run_router.sh`） | [k3-pd-pp/SKILL.md](./k3-pd-pp/SKILL.md) |
| `scripts/kimi-k3-cpp/prepare_k3_cut_override.py` | [k3-pd-pp/SKILL.md §1](./k3-pd-pp/SKILL.md)（裁层 override 生成） |
| `scripts/kimi-k3-cpp/ACCEPTANCE.md` | [k3-pd-pp/pp-partition-and-cut.md §5](./k3-pd-pp/pp-partition-and-cut.md)（验收） |
| `runbooks/nightly-watch/scripts/check_nightly.py` | [nightly-watch/](./nightly-watch/SKILL.md) |
| `runbooks/nightly-local-repro/scripts/run_repeat.sh` | [nightly-local-repro/](./nightly-local-repro/SKILL.md) |

---

## 通用前提（每个 runbook 都默认遵守）

1. **不占用户交互容器**（记作 `<USER_DEV_CONTAINER>`）。复现/压测另起专用容器。
2. **不擅自 `git commit` / `git push` / 改 `git config`**。用户明确要求时才做。
3. 真实镜像、模型路径、proxy 端口以**本机 `docker.sh` / 日更文件 / 用户当场指定**为准；公开文档一律 `$HOME` / `<PLACEHOLDER>`。
4. 拉代码前先看工作区脏不脏：脏就 `git stash push -m "agent-temp-..."` 再 `pull --ff-only`，**不要盲目 `stash pop` 到新 HEAD 上制造冲突**。
5. 长任务后台跑 + 轮询，**不要用裸 `OK` / `200 OK` 当结束标志**。

## 维护约定

1. **改了脚本就改 runbook**，反之亦然。脚本和说明不一致是这类文档最常见的腐烂方式。
2. 新增踩坑追加到对应 runbook 的「踩坑账本」表格；若同一根因出现 ≥2 次，抽成 [`../debugging/cases/`](../debugging/cases/) 独立文件。
3. 命令要能直接复制执行，变量用 `${VAR:-默认值}` 形式给全。
