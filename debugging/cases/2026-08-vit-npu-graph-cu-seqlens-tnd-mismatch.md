# ViT NPU Graph：`cu_seqlens` 存成了 per-seq 差分，应为 cumulative

- **日期**：2026-08（Plan `tnd_seqlen_mismatch` 记录）
- **环境**：Ascend NPU；Qwen3.6 系列多模态；`SGLANG_VIT_ENABLE_CUDA_GRAPH=1`（走 ViT NPU Graph）
- **结论**：`ViTNpuGraphRunner` 在 #25768 之后**没跟上 TND 语义变化** ——
  TND 要 **cumulative**，graph runner 仍按旧的 unpad 语义存 **per-seq 长度**。
  两个上游改动叠加后才暴露。
- **改动**：1 行（去掉一次差分）

> 前置概念：[`knowledge/npu/npu-graph-runner.md`](../../knowledge/npu/npu-graph-runner.md)
> （NPU Graph 的 `cpu_update_input` 机制 —— 理解本文「为什么图里存的是缓存值」的关键）。

---

## 症状

ViT 走 NPU Graph 时 seqlen 不匹配。特征：**只在高并发 / 多图 batch 下出现**。

---

## 根因

### 两个上游改动的叠加

| PR | 影响 |
|---|---|
| **#25768** | TND 语义改为需要 **cumulative** 的 `cu_seqlens` |
| **#24013** | 支持多图 batch → `T` 增大 |

**为什么单独一个都不炸**：

- #25768 之后 graph runner 没改，仍存 per-seq 长度 —— 但**单图时 per-seq 的末项碰巧等于 `T`**，
  看起来「没问题」。
- #24013 之后多图 batch 让 `T` 增大，**「末项碰巧 == T」这个巧合不再成立** → 暴露。

> 这类 bug 的典型特征：**两个改动各自都合理，叠加才炸**，且只在特定 batch 形状下出现。
> 定位时要同时看近期的语义变更和 batch 形状变更，只查一个找不到。

---

## 修复

文件：`sglang/python/sglang/srt/hardware_backend/npu/graph_runner/vit_npu_graph_runner.py`
方法：`create_graph`

```python
# before —— 存 per-seq 长度（旧 unpad 语义）
seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
self.cu_seq_lens[graph_key] = seq_lens.to("cpu").to(torch.int32)

# after —— 存 cumulative（与 TND / eager 路径对齐）
self.cu_seq_lens[graph_key] = cu_seqlens[1:].to("cpu").to(torch.int32)
```

**就这一处。** `VisionAscendAttention` 的 graph 分支继续用 `seq_len_arg = cu_seqlens`，无需改。

---

## 明确不改的部分（避免扩大 diff）

| 不改 | 原因 |
|---|---|
| `vision.py`（含 `forward_metadata` 分支） | 本用例 `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` 走 `forward_with_npu_graph`，**不传 metadata** |
| `mm_utils` / #24013 | 多图 batch 是**有意保留**的功能，不是 bug |
| 额外 assert | 非必需，会扩大 diff |

> 定位时容易误伤 `mm_utils` —— 它只是让 `T` 变大的**触发条件**，不是根因。

---

## 给 review 的解释口径

1. #25768 后 TND 要 cumulative；graph runner 仍按旧 unpad 存 per-seq
2. #24013 多图 batch 后 `T` 增大，单图时「末项碰巧 == T」不再成立
3. 本改动只把 runner 存的长度改成 `cu_seqlens[1:]`，**与 eager 路径对齐**

---

## 验证

```bash
# 复跑（或同等高并发 + ViT graph 的用例）
test_npu_qwen3_6_27b_1p_in1080p_30_out256_50ms
```

**注意要高并发**：单图 batch 下这个问题本来就不显现，低并发复跑会误判为「已修复」。

---

## 一句话给下一个 agent

看到 ViT + NPU Graph 的 seqlen 不匹配，先查
`vit_npu_graph_runner.py` 里存的是 **per-seq 还是 cumulative** ——
#25768 之后必须是 cumulative。复现**必须用高并发多图 batch**。
