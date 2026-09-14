---
name: cursor-skill-locations
description: >-
  Maps where Cursor skills, rules, chats, plans, canvases, and other local
  agent data live on this Windows machine. Use when looking for skills, agent
  transcripts, User Rules, hooks, MCP config, state.vscdb, or when another
  agent asks where Cursor 沉淀 / 本地数据 is. Living doc: add new skill roots
  as they appear. Never write real usernames; use $HOME / <USER>.
---

# Cursor 本地沉淀位置（给下一个 agent）

本机是 **Windows**。公开文档一律用 `$HOME` / `<USER>` / `<PLACEHOLDER>`，不要写真实用户名。

**先记住：Cursor 的 skill 在这些地方。**

| 类型 | 路径 | 说明 |
|------|------|------|
| **本仓库活文档（优先读）** | `personal/sglang-knowledge/runbooks/` | 团队/项目操作手册；本文件也在这里 |
| **Cursor 内置 skill** | `$HOME/.cursor/skills-cursor/` | 系统同步，**禁止往这里写** |
| **个人 skill（跨项目）** | `$HOME/.cursor/skills/` | Cursor 官方个人 skill 根；本机创建时可能还不存在 |
| **项目 skill** | `<repo>/.cursor/skills/` | 跟仓库走；当前 `work` 根目录 **没有** `.cursor/` |
| **Agents/Codex 镜像的内置 skill** | `$HOME/.agents/skills/` | 与 `skills-cursor` 同类内置副本 |
| **Codex system skill** | `$HOME/.codex/skills/.system/` | Codex 系统 skill |

内置 skill 示例（`$HOME/.cursor/skills-cursor/`）：`create-skill`、`create-rule`、`create-hook`、`canvas`、`share`、`origin`、`sdk`、`babysit`、`review-bugbot`、`review-security` 等。

新建个人 skill：**写到 `$HOME/.cursor/skills/<name>/SKILL.md`**，或本仓库对应三分目录
（`runbooks/<name>/SKILL.md` / `debugging/<name>/SKILL.md`）。永远不要写进 `skills-cursor`。

---

## 两棵根目录

Cursor 本机数据就这两处（Windows）：

1. **Agent / Cursor 专用**：`$HOME/.cursor/`
2. **编辑器用户数据（VS Code 风格）**：`$HOME/AppData/Roaming/Cursor/`（即 `%APPDATA%/Cursor`）

Linux/macOS 第二棵树不在 `AppData`；本机以 Windows 为准。

---

## `$HOME/.cursor/`（可读沉淀大多在这）

| 内容 | 路径 | 备注 |
|------|------|------|
| 对话转录（Agent transcripts） | `$HOME/.cursor/projects/<workspace-slug>/agent-transcripts/` | 每个工作区一份；可读 jsonl |
| Canvas | `$HOME/.cursor/projects/<workspace-slug>/canvases/` | |
| 聊天附件 | `$HOME/.cursor/projects/<workspace-slug>/uploads/` | |
| 终端快照 | `$HOME/.cursor/projects/<workspace-slug>/terminals/` | |
| MCP 工具描述缓存 | `$HOME/.cursor/projects/<workspace-slug>/mcps/` | |
| Plan | `$HOME/.cursor/plans/` | |
| 内置 Skills | `$HOME/.cursor/skills-cursor/` | 只读 |
| 个人 Skills | `$HOME/.cursor/skills/` | 跨项目 |
| 自定义 Subagent | `$HOME/.cursor/agents/` | 可为空 |
| 用户级 Hooks | `$HOME/.cursor/hooks.json` + `hooks/` | 可没有 |
| 用户级 MCP | `$HOME/.cursor/mcp.json` | 可没有 |
| AI 改码追踪库 | `$HOME/.cursor/ai-tracking/ai-code-tracking.db` | SQLite |
| 扩展 | `$HOME/.cursor/extensions/` | |
| 插件 | `$HOME/.cursor/plugins/` | |
| IDE 状态 | `$HOME/.cursor/ide_state.json` | |
| argv | `$HOME/.cursor/argv.json` | |

工作区 slug 是路径的消毒名，例如本机 Desktop `work` 工作区类似：

`$HOME/.cursor/projects/c-Users-<USER>-Desktop-work/`

SSH 远程窗口会各自再有一份（`projects/` 下另有 `home`、`mnt`、临时目录等）。**不要把内网主机名/IP 写进本 skill。**

读历史对话：先打开对应项目的 `agent-transcripts/`。

---

## `%APPDATA%/Cursor`（聊天库、设置、索引）

根：`$HOME/AppData/Roaming/Cursor/`

| 内容 | 路径 | 备注 |
|------|------|------|
| Chat / Composer / **User Rules** / 记忆 | `User/globalStorage/state.vscdb` | SQLite，体积可到 GB 级；**不是明文 md** |
| 同目录 WAL / backup | `state.vscdb-wal`、`state.vscdb.backup` | 备份聊天/规则时一起拷 |
| 全局 storage 元数据 | `User/globalStorage/storage.json` | |
| 编辑器设置 | `User/settings.json` | |
| 本地文件编辑历史 | `User/History/` | 类似 VS Code Local History |
| 工作区索引 / retrieval | `User/workspaceStorage/<hash>/` | 每个工作区一个 hash 目录 |
| 日志 | `logs/` | 排障，不算知识沉淀 |

**User Rules**（Cursor Settings 里的用户规则）不在单独 `.md` 里，落在 `state.vscdb`，并可能跟账号同步。

备份 Chat/规则：拷 `User/globalStorage/state.vscdb`（及 wal/backup）。

当前 `work` 文件夹对应的 workspaceStorage hash 会变；用该目录下 `workspace.json` 的 `folder` 字段对照 `file:///.../work`，不要写死 hash。

---

## 项目仓库内（可进 Git）

| 内容 | 路径 |
|------|------|
| 项目 Rules | `<repo>/.cursor/rules/*.mdc` |
| 项目 Skills | `<repo>/.cursor/skills/` |
| 项目 Hooks | `<repo>/.cursor/hooks.json` + `.cursor/hooks/` |
| Agent 说明 | `<repo>/AGENTS.md`、遗留 `.cursorrules` |

当前 Desktop `work` **根目录没有** `.cursor/`。子仓里可能自带 `AGENTS.md`（例如各份 `sglang/docs/AGENTS.md`），那是项目文档，不是 Cursor 全局配置。

固化团队规范：写成项目 `.cursor/rules/` 或本仓库 `runbooks/`，不要只放在 User Rules 数据库里。

---

## 给下一个 agent 的用法

1. 找 **操作手册 / 起服步骤**：读 [`runbooks/`](../README.md)。
2. 找 **故障排查**：读 [`debugging/README.md`](../../debugging/README.md) 的「报错串 → 去哪」表。
3. 找 **原理 / 架构**：读 [`knowledge/`](../../knowledge/README.md)。
4. 找 **Cursor 个人/内置 skill**：上表 `$HOME/.cursor/skills*` 与 `$HOME/.agents/skills`。
5. 找 **某次对话原文**：`$HOME/.cursor/projects/<slug>/agent-transcripts/`。
6. 找 **侧边栏 Chat / User Rules**：`state.vscdb`（需 sqlite 查询，不要当 markdown 打开）。
7. 新发现其它 skill 根（远程机 `$HOME/.cursor/skills`、别的仓库 `.cursor/skills`）：**追加到本文件表格**，不要只写在聊天里。
8. 路径脱敏：用户名 → `<USER>` 或直接 `$HOME`；内网 IP/主机名/容器真名 → `<PLACEHOLDER>`。

## Living doc

本机又出现新的 skill 根、规则目录、或 Cursor 改了存储布局时，立刻改本文件，
并同步 [`runbooks/README.md`](../README.md) 与根 [`README.md`](../../README.md) 的索引。

> 本文件原在 `agent-skills/` 下，随 2026-09-02 的三分重构迁至
> `runbooks/agent-tooling/`（它回答的是「本机数据在哪」= 操作类，不是知识也不是故障）。
