# Git / PR 工作规范

> 这一类属于「怎么做」→ [`runbooks/`](./README.md)。
> 之所以单独成文：**这些规则原本只存在于 Cursor 的 User Rules 里，而 User Rules 正文
> 并不落在本机 `state.vscdb`**（只有一条空描述符），跟着账号走，换机器就没了。
> 固化在这里才能进 Git、能分享。

## 硬规则（违反会造成真实损失）

| 规则 | 原因 |
|---|---|
| **不擅自 `git commit` / `git push`** | 用户明确要求时才做 |
| **不改 `git config`** | 会污染用户环境 |
| **不打印 GitHub token / 带凭据的 origin URL** | 泄露 |
| **不动用户已有的未提交改动** | 那是用户有意保留的东西（常见：诊断日志、临时 print） |
| 提交前 `git diff --check` + 审阅完整 diff | 防止覆盖别人的改动 |

典型场景：用户说「改这两处，**不提交 git**」→ 改完跑 `py_compile` / `bash -n` / lint，
**然后停在那里**，不要顺手 commit。

## 多仓协作（community / ascend / personal）

| 仓库 | 角色 |
|---|---|
| `<SGLANG_PYTHON>`（community sglang） | **运行时**，跟 `main` |
| `ascend/sglang` | **测试用例**，分支 `testcases` |
| `personal/sglang` | 个人改动 |

### 拉更新的安全顺序

```bash
# 1. 先看脏不脏
git status --short

# 2. 脏就 stash 指定文件（不要 stash 整个工作区）
git stash push -m "agent-temp-<reason>" -- <dirty-files...>

# 3. 拉
git pull --ff-only origin <branch>

# 4. community 侧不要盲目 stash pop；ascend 侧测试用例改动可以 pop
git stash pop || true
```

**坑**：

- 不 stash 时 `git pull --ff-only` 会直接 **Abort**。
- **不要盲目 `stash pop` 到新 HEAD 上** —— 会制造冲突，且可能盖掉刚拉下来的文件。
  曾有 stash 整文件覆盖 `environ.py` 导致
  `AttributeError: Envs has no attribute SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK`
  → 应从 `origin/main` 补回缺失 Env，而不是用过旧的 stash 覆盖。

### 覆盖拷贝 test 包时

```bash
\cp -rf "$ASCEND/python/sglang/test" "${SGLANG_PYTHON}/sglang/"
```

**必须用 `\cp`**：交互 shell 里 `cp` 常是 `alias cp='cp -i'`，会对每个文件问
`overwrite?`，脚本直接卡死。

## 改完自检（改代码 / 改脚本后固定做）

```bash
python -m py_compile <changed.py>     # Python
bash -n <changed.sh>                   # Shell
git diff --check                       # 空白/冲突标记
git diff                               # 完整审阅
```

## PR / code review

- 回报 PR 状态写明 **review 结论 + commit 状态**，不要只说「看过了」。
- merge message 按仓库既有风格，不确定就先问。
- 评审时区分三类问题：**阻塞 / 建议 / nit**，别混在一起。

## 相关

- [`../debugging/README.md`](../debugging/README.md) — 故障排查
- [`agent-tooling/`](./agent-tooling/) — 本机路径地图
