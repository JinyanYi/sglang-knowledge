# Agent skills（SGLang / NPU 看护）

这些 skill 给 **cron agent** 和下一个接手的 Cursor agent 用。

| Skill | 用途 |
|-------|------|
| [nightly-npu-qwen36-35b](./nightly-npu-qwen36-35b/SKILL.md) | 查公开 Nightly，只筛 Qwen3.6-35B-A3B（`scripts/check_nightly.py`） |
| [nightly-npu-local-repro](./nightly-npu-local-repro/SKILL.md) | 内网 NPU 机 docker 复现失败 case（`scripts/run_repeat.sh`） |
| [npu-docker-pythonpath](./npu-docker-pythonpath/SKILL.md) | 容器内 PYTHONPATH / CANN `tbe` 坑 |

日更说明原文（本机私有）：`$HOME/daily task for agent.md`

公开文档里的路径一律用 `$HOME` / `<PLACEHOLDER>`；真实镜像、模型、proxy 以本机 `docker.sh` 或用户当场指定为准。

## 给下一个 agent 的硬规则

1. **先读再跑**：至少读 `nightly-npu-qwen36-35b` +（有失败时）`nightly-npu-local-repro`。
2. **这是活文档**：过程中遇到任何新问题、过时步骤、或可复用命令，**必须回写对应 `SKILL.md`**（追加「踩坑账本」），不要只写在对话里。
3. **相关文件一起改**：本机 `daily task for agent.md` 里的 checklist 若与 skill 不一致，同步更新。
4. **不要占用户交互容器**（`<USER_DEV_CONTAINER>`）；网络先试本机已配置的 `http_proxy`（端口问用户 / 看日更）。
