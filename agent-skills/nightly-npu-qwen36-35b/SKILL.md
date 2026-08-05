---
name: nightly-npu-qwen36-35b
description: >-
  Triage SGLang public Nightly Test NPU for Qwen3.6-35B-A3B only. Use when the
  user asks about nightly-test-npu.yml, morning NPU nightly watch, schedule
  runs, or Qwen3.6-35B-A3B performance/accuracy job failures on
  sgl-project/sglang.
---

# Nightly NPU：只看护 Qwen3.6-35B-A3B

每天早上看护公开仓 workflow，**只关心 schedule nightly 里名字含 `qwen3_6_35b_a3b` 的 job**。

- Workflow: https://github.com/sgl-project/sglang/actions/workflows/nightly-test-npu.yml
- 日更任务原文（本机私有）：`$HOME/daily task for agent.md`

## 范围

| 做 | 不做 |
|----|------|
| `event=schedule` 的 Nightly Test NPU | PR / workflow_dispatch（除非用户点名） |
| job 名含 `qwen3_6_35b_a3b` | `qwen3_6_27b`、`qwen3_vl_30b_a3b` 等 |
| 每个 matched job 的结论 + URL | 把整个 workflow failure 当成“我们的模型挂了” |
| failure 时抽出 AssertionError / 关键原因 | 无鉴权硬下 job logs（常 403） |

## 查法（可复用）

无 `gh`、无 token 时用 GitHub REST：

```text
1) GET .../actions/workflows/nightly-test-npu.yml/runs?event=schedule&per_page=5
   → 取最新（或指定日期 created_at 的）run_id / html_url

2) GET .../actions/runs/<run_id>/jobs?per_page=100
   → 过滤 name 含 qwen3_6_35b_a3b

3) failure 时：
   - GET .../actions/jobs/<job_id> 看失败 step（多为 Run test）
   - GET .../check-runs/<job_id>/annotations（信息有限）
   - 抓 public job HTML，搜 AssertionError / FAILED
   - 勿依赖 GET .../actions/jobs/<job_id>/logs（无 token 常 403）
```

一键脚本（本机）：

```bash
python3 personal/sglang-knowledge/agent-skills/nightly-npu-qwen36-35b/scripts/check_nightly.py
python3 personal/sglang-knowledge/agent-skills/nightly-npu-qwen36-35b/scripts/check_nightly.py --date 2026-07-22
python3 personal/sglang-knowledge/agent-skills/nightly-npu-qwen36-35b/scripts/check_nightly.py --run-id 30471373497
```

## 失败怎么读

- 性能：常见 `AssertionError: <actual> not greater than or equal to <threshold>`
  - threshold 通常是测试文件里 `output_token_throughput * 0.98`
- 精度：看 accuracy case（如 `*_aime26`）日志里的 score / assert
- 基础设施：`setUpClass` / compile / OOM 等，和吞吐门槛分开报

## 输出格式（给用户）

```markdown
Nightly: <run_url> (created <UTC>, conclusion <...>)

| case | result | job |
|------|--------|-----|
| ... | success/failure | <url> |

Failures:
- <case>: <one-line root cause>
```

## 失败之后

对每个失败的 `qwen3_6_35b_a3b` job，继续走本地复现 skill：

`personal/sglang-knowledge/agent-skills/nightly-npu-local-repro/SKILL.md`

## Living doc

这是活文档。triage / API / 脚本若有新坑或更好写法，**立刻改本 SKILL.md**（以及
`nightly-npu-local-repro` / `daily task for agent.md`），不要只留在聊天里。

## 踩坑账本

| 坑 | 症状 | 处理 |
|----|------|------|
| `127.0.0.1:1082` proxy 未监听 | `Connection refused` | 先无 proxy 直连 `api.github.com`；本机也可能开在 `1080` |
| Ascend **Full Test (NPU)** 与 public Nightly 不是同一 workflow | 用户贴 `ascend/sglang/actions/runs/...` | 用 `repos/ascend/sglang/actions/runs/<id>/jobs` 筛 `qwen3_6_35b_a3b`；失败常见是 **TPOT 上限**（`not less than or equal to`）而非吞吐 |

## 校准样例

- `2026-07-22` run `29946789102`：6 个 `qwen3_6_35b_a3b_*` 全 success
- `2026-07-29` run `30471373497`：仅 `1p_in128k_out1k_prefix90_50ms` failure  
  `298.41 not greater than or equal to 302.036`（吞吐）
