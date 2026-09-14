# `--cuda-graph-bs 1 2` 被过滤导致 `assert len(capture_bs) > 0`

- **日期**：2026-08-10 排查
- **环境**：Ascend 910 A3；Kimi-K3 W4A8；PD 分离 decode 侧；开了 `deepep`；tp=8
- **结论**：不是参数写错。开了 `deepep` 之后，graph bs 的**对齐基数变成 `attn_tp_size`**，
  `1`/`2` 被合法过滤掉了。

> 相关：[`../deep-ep/SKILL.md`](../deep-ep/SKILL.md)、
> [`../../knowledge/inference/cuda-graphs.md`](../../knowledge/inference/cuda-graphs.md)（graph 捕获原理）。

---

## 症状

decode 节点配 `--cuda-graph-bs 1 2`，启动时直接失败：

```
assert len(capture_bs) > 0
```

看起来像是「参数没传进去」，实际是被**过滤空了**。

---

## 根因

调用链：`get_batch_sizes_to_capture` → `get_cuda_graph_batch_size_alignment`。

1. 开了 `deepep`（a2a 非 `none`）→ `require_attn_tp_gather()` 返回 `True`
2. → 对齐基数 = **`attn_tp_size` = 8**
3. → `capture_bs` 里不满足 `bs % 8 == 0` 的值**全部被过滤**
4. → `1`、`2` 被剔除 → 空列表 → assert 失败

### 为什么必须有这个对齐（不是随意的限制）

- **scattered 模式下每 rank 只拿 `bs / attn_tp` 个 token，必须是整数**；
- 且 DeepEP a2a 用**固定几何 buffer**，decode graph bucket 必须**跨 EP rank 完全一致**，
  否则几何不匹配 → 非法访问（代码注释引用 issue #30242）。

---

## 修复 & 验证

| 配置 | 合法的 `--cuda-graph-bs` |
|---|---|
| `deepep` + tp=8 | `8 16 ...`（8 的倍数） |
| `--moe-a2a-backend none` | `1 2` 合法（对齐基数 = 1） |

**验证**：改成 `--cuda-graph-bs 8 16` 后正常启动。

---

## 一句话给下一个 agent

看到 `assert len(capture_bs) > 0` 先别怀疑参数没生效 ——
**检查是不是开了 `deepep`**，开了就必须是 `attn_tp_size` 的倍数。
