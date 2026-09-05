# NPU Graph Runner：机制与关键设计

> **本文只讲「它是什么 / 为什么这么设计」。**
> 具体故障（如 ViT graph 的 `cu_seqlens` 语义不匹配）见
> [`debugging/cases/2026-08-vit-npu-graph-cu-seqlens-tnd-mismatch.md`](../../debugging/cases/2026-08-vit-npu-graph-cu-seqlens-tnd-mismatch.md)。

---

## 1. 它解决什么问题

**decode 阶段每一步的 forward 逻辑完全相同**：输入 shape 相同、算子序列相同，只有数据不同。
传统做法是每步重走「Python 解释器 → 算子调度 → NPU 执行」，有大量重复开销。

**NPU Graph 的核心思想**：先跑一次 forward，把整个计算过程**录制**成一张静态图
（含所有 kernel launch 的顺序与依赖），之后每次用新数据**回放**，省掉 Python 层调度开销。

> 与 CUDA Graph 的概念一致，差异在下面的实现细节。

---

## 2. 类层次

```
CudaGraphRunner (通用基类, sglang/srt/model_executor/cuda_graph_runner.py)
  ├── NPUGraphRunner  ★（昇腾专用）
  ├── CPUGraphRunner
  └── ...
```

`NPUGraphRunner` 继承并**重写**关键方法，把 CUDA 专用逻辑替换为昇腾实现。

### 一个容易看漏的初始化技巧

```python
def __init__(self, model_runner: ModelRunner):
    sglang.srt.model_executor.cuda_graph_runner.patch_model = patch_model_npu  # ★ 偷梁换柱
    super().__init__(model_runner)
```

`capture()` 内部用的是**模块级** `patch_model` 变量。NPU 版在调用 `super().__init__()`
**之前**就把这个变量换掉，于是父类捕获图时自动使用 NPU 的编译后端 `npugraph_ex`。

> 读代码时如果只看 `capture()`，会以为用的是默认的 `torch.compile(backend="inductor")`。

---

## 3. 与 CUDA Graph 的四个关键差异

| # | 差异 | 说明 |
|---|---|---|
| 1 | `torch.npu.NPUGraph()` | 替代 `torch.cuda.CUDAGraph()` |
| 2 | **`cpu_update_input` 机制** | ⭐ 见下节，这是最重要的差异 |
| 3 | `cache_loc` 用 **int32** | CUDA 用 int64 |
| 4 | 编译后端 `npugraph_ex` | 替代 inductor |

---

## 4. `cpu_update_input`：为什么图里能改「长度」

**CUDA Graph 是冻结的** —— 录完不能改输入 shape 相关参数。

**NPU Graph 支持在 replay 时通过 CPU 更新图中某些 Tensor 的值，而无需重新录制**：

```python
def _update_inputs(self, seq_lens):
    if isinstance(self.update_attr_type, torch.Tensor):
        seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
    self.graphs[self.bs].update(cpu_update_input=[{self.update_attr_name: seq_lens}])
```

这让 NPU Graph 能处理**变长序列**。

> ### ⚠️ 这直接导致一类 bug
>
> 既然长度是「录制时缓存、replay 时更新」的，那么**录制时存的值的语义**必须和
> replay 时期望的语义一致。上游改了语义（例如从 per-seq 改成 cumulative）而
> graph runner 没跟上，就会静默算错 —— 且**只在输入形状让巧合失效时才暴露**。
> 实例见 [`debugging/cases/2026-08-vit-npu-graph-cu-seqlens-tnd-mismatch.md`](../../debugging/cases/2026-08-vit-npu-graph-cu-seqlens-tnd-mismatch.md)。

---

## 5. 属性名随架构 / forward 模式而变

`_init_arch_map()` 根据 Attention 架构配置 replay 时更新的属性：

| 条件 | 属性名 | 类型 |
|---|---|---|
| DLLM 模式，MLA / MHA | `actual_seq_lengths_kv` | MLA: `list`，MHA: `torch.Tensor` |
| 非 DLLM，MLA | `actual_seq_lengths_kv` | `list` |
| 非 DLLM，MHA | `context_lens` | `torch.Tensor` |

**更隐蔽的一点**：`forward_mode` 也会影响。例如 `target_verify`（投机解码的验证阶段）
下图里注册的属性名是 **`actual_seq_kvlen`**，不是 `actual_seq_lengths_kv`：

```python
if forward_batch.forward_mode.is_target_verify():
    self.update_attr_name = "actual_seq_kvlen"   # 图里实际注册的名字
    self.update_attr_type = []                   # list
```

**用错属性名的后果**：`graph.update()` 不会报错，只是**静默无效** → 序列长度没更新 → 推理结果错。

> 这是一个典型的「模式适配」问题：**必须按当前 forward 模式用对应的属性名**。
> 而且失败是静默的，不会抛异常。

---

## 6. 完整工作流

```
【启动时】
ModelRunner.init_device_graphs()
  → 检测 device="npu"
  → NPUGraphRunner(model_runner)
      → __init__ 替换 patch_model 为 patch_model_npu
      → 父类 __init__ → self.capture()
          → 遍历各 batch size
          → patch_model_npu 包装 forward（torch.compile + npugraph_ex）
          → _capture_graph() 进入 torch.npu.graph() 上下文录制
          → NPUGraph 存进 self.graphs

【推理时（每步 decode）】
ModelRunner.forward_decode()
  → NPUGraphRunner.replay(forward_batch)
      → 准备输入（copy 进预分配 buffer）
      → 新线程: _update_inputs(seq_lens) → graph.update(cpu_update_input=...)
      → 主线程: graph.replay()
      → join 等待
      → 返回输出
```

**线程化的原因**：让 CPU 更新属性与 NPU 执行回放**部分重叠**，省时间。
NSA 架构不需要更新，直接 `replay()`。

---

## 7. 一页总结

| 角度 | 说明 |
|---|---|
| 业务目的 | 加速 NPU 上的推理（主要是 decode 阶段） |
| 核心技术 | `torch.npu.NPUGraph` 录制 + `torch.compile` + `npugraph_ex` 后端 |
| vs CUDA | ① `NPUGraph` ② `cpu_update_input` ③ `cache_loc` int32 ④ `npugraph_ex` |
| 最容易出事的地方 | `replay()` 的线程化 seq_lens 更新 + `_update_inputs()` 的属性名/语义 |
| 失败模式 | **静默** —— 属性名错或语义不匹配都不报错，只是结果错 |

## 相关

- [`debugging/cases/2026-08-vit-npu-graph-cu-seqlens-tnd-mismatch.md`](../../debugging/cases/2026-08-vit-npu-graph-cu-seqlens-tnd-mismatch.md)
- [`inference/cuda-graphs.md`](../inference/cuda-graphs.md) — CUDA Graph 通用原理
- [`../../runbooks/npu-docker-env/`](../../runbooks/npu-docker-env/SKILL.md) — 环境配置（不是本文主题）
