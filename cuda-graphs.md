# CUDA Graphs & Piecewise CUDA Graph 笔记

## 目录

1. [要解决的问题：Kernel Launch Overhead](#1-要解决的问题kernel-launch-overhead)
2. [CUDA Graph 的核心思想](#2-cuda-graph-的核心思想)
3. [三个阶段的 Workflow](#3-三个阶段的-workflow)
4. [Graph 的结构细节](#4-graph-的结构细节)
5. [使用 CUDA Graph 的典型模式](#5-使用-cuda-graph-的典型模式)
6. [CUDA Graph 的局限性](#6-cuda-graph-的局限性)
7. [LLM Decode 阶段为什么适合传统 CUDA Graph](#7-llm-decode-阶段为什么适合传统-cuda-graph)
8. [LLM Prefill 阶段为什么不适合传统 CUDA Graph](#8-llm-prefill-阶段为什么不适合传统-cuda-graph)
9. [Piecewise CUDA Graph 详解](#9-piecewise-cuda-graph-详解)
10. [参考](#10-参考)

***

## 1. 要解决的问题：Kernel Launch Overhead

在 CUDA 编程中，每次通过 `<<<>>>` 启动一个 kernel（或执行 memory copy 等操作）时，CPU 端的 driver 都需要执行一系列准备工作：

- 参数校验与打包
- Kernel 描述符的构建
- 命令提交到 GPU 前端队列

这些操作本身是有开销的（微秒级别）。对于执行时间很短的 kernel（比如几微秒），launch overhead 可能占端到端时间的 **很大比例**。

```cpp
// 传统 stream 方式：每次循环都要付一次 launch overhead
for (int istep = 0; istep < NSTEP; istep++) {
    for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++) {
        shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
    }
    cudaStreamSynchronize(stream);
}
```

实测数据（V100, kernel 执行 \~2.9μs）：

| 方式                 | 每 kernel 有效耗时 | 说明            |
| ------------------ | ------------- | ------------- |
| 每次 launch 后 sync   | \~9.6μs       | 完全暴露所有开销      |
| 移出 sync，overlap 启动 | \~3.8μs       | 开销被部分隐藏       |
| **CUDA Graph**     | **\~3.4μs**   | 进一步压缩到接近纯执行时间 |

***

## 2. CUDA Graph 的核心思想

**将一批 GPU 操作（kernel、memcpy 等）定义为一个有向无环图（DAG），一次定义、多次执行。**

```
传统方式:   launch kernel1 -> launch kernel2 -> launch kernel3 -> ...
              overhead       overhead       overhead

CUDA Graph:  定义图 + 实例化（一次性开销大）-> 启动整个图（开销极小）
                                                     |
                                           一次 CPU 操作启动多个 GPU 操作
```

关键洞察：**把多次 CPU launch overhead 合并为一次**。

***

## 3. 三个阶段的 Workflow

### 3.1 Definition（定义）

创建一个描述计算任务的图，包含节点和边。

- **节点（Node）**：一个 GPU 操作，如 kernel launch、memcpy、memset、event、child graph 等
- **边（Edge）**：节点间的依赖关系，约束执行顺序

两种创建方式：

- **Graph API**：通过 `cudaGraphCreate()` + `cudaGraphAddNode()` 等 API 手动构建图
- **Stream Capture**：用 `cudaStreamBeginCapture()` / `cudaStreamEndCapture()` 包裹现有 stream 代码，自动捕获生成图（**更常用，对现有代码侵入小**）

```cpp
// Stream Capture 方式：将现有代码包裹在其中即可
cudaGraph_t graph;
cudaStreamBeginCapture(stream);
kernel_A<<<..., stream>>>(...);
kernel_B<<<..., stream>>>(...);
kernel_C<<<..., stream>>>(...);
cudaStreamEndCapture(stream, &graph);    // 此时 graph 定义完毕
```

### 3.2 Instantiation（实例化）

将图定义"编译"成可执行形式：

```cpp
cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
```

这一步会做大量的预设置和校验工作（一次性开销较大，\~400μs 级别），但换来了后续启动时的极低开销。

### 3.3 Execution（执行）

在 stream 中启动实例化后的图：

```cpp
cudaGraphLaunch(instance, stream);   // 一次 CPU 调用，启动图中所有 GPU 操作
cudaStreamSynchronize(stream);
```

`instance` 可以被任意次重复 launch，无需重新实例化。

***

## 4. Graph 的结构细节

### 节点类型

- `kernel` — GPU kernel launch
- `memcpy` / `memset` — 数据拷贝/置零
- `empty` — 空节点，仅用于表示依赖关系
- `host` — 在 CPU 上执行的函数调用
- `event` — CUDA Event 的 record/wait
- `child graph` — 嵌套子图
- `semaphore` — 外部信号量 signal/wait
- `conditional` — 条件分支节点（CUDA 12+）
- `memory` — 内存分配/释放节点

### 边的语义

默认情况下，一条边表示：**上游节点完全完成后，下游节点才能开始**。CUDA 12.3+ 引入了 edge data 机制，可以定义更细粒度的依赖关系（如 Programmatic Dependent Launch）。

***

## 5. 使用 CUDA Graph 的典型模式

```cpp
bool graphCreated = false;
cudaGraph_t graph;
cudaGraphExec_t instance;

for (int istep = 0; istep < NSTEP; istep++) {
    if (!graphCreated) {
        // 第一次迭代：创建并实例化图
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++) {
            shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
        }
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        graphCreated = true;
    }
    // 后续迭代：直接启动图，不再有逐个 launch 的开销
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);
}
```

***

## 6. CUDA Graph 的局限性

### 图是静态的（Static/Fixed）

一旦实例化（`cudaGraphInstantiate`），图的结构、节点数、参数（如 grid/block size、kernel 函数指针等）就被固定下来了。如果下一次迭代中：

- **Kernel 数量变了**（比如 attention 的序列长度变了，需要不同的 block 数）
- **Kernel 参数变了**（比如不同 layer 用不同的权重指针）
- **控制流变了**（比如条件分支走向不同）

就需要**重新实例化**（`cudaGraphInstantiate`），而这又是一个很重的操作（几百微秒）。

### CUDA Graph 的参数更新机制

CUDA 提供了 `cudaGraphExecKernelNodeSetParams()` 来更新图中 kernel 节点的参数（如指针地址、标量值），**无需重新实例化**。但 grid/block 维度是 graph 结构的一部分，无法通过此接口更新。

***

## 7. LLM Decode 阶段为什么适合传统 CUDA Graph

### Decode 的计算特征

Decode 阶段是**逐 token 生成**的循环。每次迭代只处理 1 个新 token：

```
当前状态:
- 已输入: "今天天气真好" (6 个 token)
- KV Cache: 已缓存前 6 个 token 的 K、V

迭代 1（生成第 7 个 token）:
  input = x₆ (第 6 个 token 的 embedding)
  → 经过所有 decoder layers:
     Q = W_Q(x₆)
     K = [K₁, K₂, ..., K₆]  ← 直接从 cache 读取！
     V = [V₁, V₂, ..., V₆]  ← 直接从 cache 读取！
     Attention(Q, K, V) → 只计算一个位置（1 个 query）
  → 采样得到新 token t₇
  → KV Cache 更新: [K₁, ..., K₆, K₇], [V₁, ..., V₆, V₇]
```

### 为什么 launch config 是固定的

```
对于 attention 计算：
  - Q 的形状永远是 [1, num_heads, 1, head_dim]  ← 1 个 token
  - attention kernel 的 grid 由 num_heads 和 query length 决定
  - query length = 1 → grid 的维度是固定的

对于 MLP (FFN) 计算：
  - 输入形状永远是 [1, hidden_size]  ← 1 个 token
  - grid/block 完全固定
```

所以不管 KV cache 增长到多长（cache\_len 可能 1→1000），**每次 decode 的 kernel launch config 都是一样的**。

### 参数更新方式

KV Cache 的指针变化可以通过 `cudaGraphExecKernelNodeSetParams()` 更新，不需要重新实例化：

```cpp
// 更新图中某个 kernel 节点的参数，无需重新实例化整个图
cudaGraphExecKernelNodeSetParams(instance, node, &newParams);
```

每次 Decode 只需要：

1. 更新 KV cache 的偏移量指针（新 token 追加的位置）
2. 更新 position encoding 等标量参数
3. 直接 `cudaGraphLaunch`，不需要重新实例化

### Decode CUDA Graph 的核心流程

```
cudaGraph_t graph 包含了 N 个 decoder layer 的完整计算链：
┌─────────────────────────────────────────────────────────┐
│  1. Embedding: token_id → embedding vector              │
│  2. Layer 1:  Attention(1 query) → MLP                  │
│  3. Layer 2:  Attention(1 query) → MLP                  │
│  ...                                                    │
│  4. Layer N:  Attention(1 query) → MLP                  │
│  5. Output:  Linear → logits → sample                   │
└─────────────────────────────────────────────────────────┘
每次 launch 前只更新参数（KV cache pointer, position id 等）。
```

### 注意：计算量并非固定

Decode 的 **attention 计算量随 KV cache 长度线性增长**：

```
第 1 步: attention 对 6 个 KV 做计算
第 100 步: attention 对 106 个 KV 做计算
```

虽然 Q 始终只有 1 个，但 K、V 的序列长度在增长。**计算量变化不影响 CUDA Graph 的适用性**，因为 grid/block 由 query length（固定的 1）决定。

***

## 8. LLM Prefill 阶段为什么不适合传统 CUDA Graph

### Prefill 的计算特征

Prefill 阶段**一次性处理整个用户输入**（N 个 token），并为后续 Decode 创建 KV Cache：

```
输入: "今天天气真好" (6 tokens)
         ↓
┌────────────────────────────────────────────────────────────┐
│                    Embedding Layer                         │
│  [1, 6] → [1, 6, 4096] (词向量)                           │
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│              Decoder Layer 1                               │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Q = W_Q(hidden)  → [1, 32, 6, 128]               │   │
│  │  K = W_K(hidden)  → [1, 32, 6, 128]  ← 缓存到 KV  │   │
│  │  V = W_V(hidden)  → [1, 32, 6, 128]  ← Cache      │   │
│  └────────────────────────────────────────────────────┘   │
│         ↓                                                 │
│  Attention(Q, K, V) → [1, 32, 6, 128]                    │
│         ↓                                                 │
│  MLP → [1, 6, 4096]                                      │
└────────────────────────────────────────────────────────────┘
```

### 为什么 launch config 是动态的

**Prefill 的 seq\_len（N）每次都不一样**。用户的 prompt 可以是 10 个 token、100 个、1000 个...

关键差异在 attention 计算的 grid 配置：

```python
# Flash Attention 的 grid 通常这样设置：
gridDim.x = (seq_len + block_M - 1) // block_M  # 取决于 seq_len！
gridDim.y = num_heads
gridDim.z = batch_size
```

当 N=10 时，grid 可能是 `(1, 32, 1)`；当 N=1000 时，grid 变成 `(63, 32, 1)`。

```
对比：
Decode:  Q[1, h, 1, d]  → grid 由 num_heads 决定 → 固定
Prefill: Q[1, h, N, d]  → grid 由 seq_len N 和 num_heads 共同决定 → 动态
```

### 为什么传统 CUDA Graph 不适用

一旦实例化，grid/block 就固定了。如果预先把 N=100 的图实例化了，来了一个 N=200 的请求：

- 直接 launch 旧的图？→ grid 不对，计算量不够 → **结果错误**
- 重新实例化？→ \~400μs 开销 → **白费了 Graph 的优势**
- 每个 N 都实例化一个？→ 存几百个图 → 内存爆炸

**解决方案**：Piecewise CUDA Graph —— 把大图拆成小片，只重新实例化需要变化的部分。

***

## 9. Piecewise CUDA Graph 详解

### 9.1 核心直觉

不把整个 Prefill 打包成一个图，而是拆成小片（piece），每片单独用 CUDA Graph 管理：

```
传统 Graph:       [整个模型前向] → 一个大图 → 必须一次性实例化，完全静态

Piecewise Graph:  [Attention]  → 子图 A (按 seq_len 分桶实例化)
                  [MLP]        → 子图 B (形状固定，一次实例化就行)
                  [RMS Norm]   → 子图 C (形状固定)
                  ...
                  运行时根据实际 seq_len 选择对应的子图 A，动态组合
```

这样只有动态变化的子图（如 Attention）需要按不同大小实例化多个版本，静态部分（如 MLP、Norm）只需实例化一次。

### 9.2 SGLang 的实现架构

SGLang 的 Piecewise CUDA Graph 使用了 `torch.compile` 的 custom backend 机制。整体流程：

```
model.forward wrapper
→ torch.compile(..., backend=SGLangBackend)
→ FX graph（获取模型的计算图）
→ split_graph() 在注册的 split ops 处切割
→ split_gm（顶层"缝合图"，串联各个 piece）
→ 用 CUDAPiecewiseBackend 替换可捕捉的子模块
→ 运行时：split ops 走 eager 模式 + 其余子模块走 graph replay
```

#### 三个关键步骤

1. **Split（切割）**：`torch.compile` 追踪模型 forward 后拿到 FX 图，在 `CompilationConfig.split_ops` 中注册的算子处切割（如 attention、all-reduce）。切割后得到一列子模块：`submod_0`, `submod_1`, ... 可捕捉的（capturable）子模块和被切出的 split-op 子模块交替排列。
2. **Replace（替换）**：每个可捕捉子模块被替换为一个 `CUDAPiecewiseBackend` 实例。split-op 子模块（attention、all-reduce 等）保持原样，运行时走 eager 模式。
3. **Dispatch（派发）**：运行时执行 `split_gm`（缝合图），按序调用每个子模块：
   - split-op 子模块 → eager 执行
   - CUDAPiecewiseBackend → 经历三个阶段：
     - **Compile warmup**：通用形状编译路径
     - **Capture**：对每个捕捉大小，跑一次 warmup 后录制 CUDA graph
     - **Steady-state replay**：稳定后每次 forward 只需 replay 已录制的 graph

### 9.3 PiecewiseCudaGraphRunner 的三阶段生命周期

```
┌─────────────────────────────────────────────────────────────────┐
│  Compile 阶段                                                    │
│  - 用 dummy forward 预热 JIT kernel                              │
│  - 用 torch.compile 包裹模型，触发 Dynamo tracing                │
│  - 切割 FX graph，创建 CUDAPiecewiseBackend 实例                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  Capture 阶段                                                    │
│  - 从大到小遍历所有 capture sizes（大到小，为了内存复用）          │
│  - 对每个 size 跑两次 forward（一次 warmup，一次 CUDA graph capture）│
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  Replay 阶段                                                     │
│  - 二分查找最小的 captured size >= 实际 token 数                  │
│  - 将输入拷贝到 static buffer 并做 zero-padding                  │
│  - replay 对应的 CUDA graph                                      │
│  - 把输出 slice 回实际 token 数                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 9.4 capture\_one\_batch\_size 源码分析

核心代码在 `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`：

```python
def capture_one_batch_size(self, num_tokens: int):
    buffers = self.buffers
    bs = 1

    # 从预分配的 buffer 中 slice 出当前 size 需要的输入
    input_ids = buffers.input_ids[:num_tokens]
    positions = buffers.positions[:num_tokens]
    out_cache_loc = buffers.out_cache_loc[:num_tokens]  # KV 写入位置

    with torch.device(self.device):
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=bs,
            input_ids=input_ids,
            seq_lens=torch.tensor([num_tokens], device=self.device),
            out_cache_loc=out_cache_loc,
            extend_num_tokens=num_tokens,
            extend_seq_lens=torch.tensor([num_tokens], device=self.device),
        )

    # 跑两次：第一次 warmup，第二次 capture
    def run_once():
        with set_forward_context(forward_batch, ...):
            self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )

    for _ in range(2):
        self.device_module.synchronize()
        self.model_runner.tp_group.barrier()
        run_once()
```

关键点：

- **bs=1**：PCG 对 batch\_size=1 的场景优化，每个请求单独 capture
- **num\_tokens 可变**：循环调用此方法，对每个 `capture_num_tokens` 列表中的值做 capture
- **跑两次**：第一次 warmup 建立 CUDA graph 上下文，第二次真正录制
- **从大到小 capture**：最大 size 先 capture，后续小 size 复用大 size 分配的内存

### 9.5 Shape 配置（分桶策略）

PCG 预定义了一组 token 数（capture sizes），运行时实际 token 数通过二分查找 round up 到最近的 captured size，然后 pad 到该 size 后 replay。如果超过最大 captured size，则 fallback 到普通非 graph 的 forward 路径。

默认的自动生成策略（从细到粗）：

| Token 范围    | 步长  |
| ----------- | --- |
| 4 – 32      | 4   |
| 48 – 256    | 16  |
| 288 – 512   | 32  |
| 576 – 1024  | 64  |
| 1280 – 4096 | 256 |
| 4096+       | 512 |

这种从细到粗的设计很合理：短 prompt 时相对变化率大（+4 已经是 50% 的变化），长 prompt 时相对变化率小（+256 对 4096 只有 6%），在内存开销和命中率之间取得平衡。

### 9.6 Replay 时的 padding 机制

```python
def replay_prepare(self, forward_batch: ForwardBatch, **kwargs):
    num_tokens = len(forward_batch.input_ids)
    # 二分查找最近的桶
    index = bisect.bisect_left(self.capture_num_tokens, num_tokens)
    static_num_tokens = self.capture_num_tokens[index]

    if static_num_tokens != num_tokens:
        # zero-padding 多出来的位置
        buffers.input_ids[num_tokens:static_num_tokens].zero_()
        buffers.positions[num_tokens:static_num_tokens].zero_()

    # 拷贝实际数据到 static buffer
    buffers.input_ids[:num_tokens].copy_(forward_batch.input_ids)
    buffers.positions[:num_tokens].copy_(forward_batch.positions)
```

### 9.7 Prefill vs Decode CUDA Graph 对比总览

| 维度               | Prefill                        | Decode                          |
| ---------------- | ------------------------------ | ------------------------------- |
| 输入               | 完整的用户 prompt（N 个 token）        | 上一次生成的 token（1 个 token）         |
| 计算方式             | 并行计算所有位置                       | 串行计算单个位置                        |
| 计算复杂度            | O(N²)（完整注意力矩阵）                 | O(cache\_len)（一行注意力，随 cache 增长） |
| Kernel grid 是否固定 | ❌ 随 N 变                        | ✅ 固定                            |
| 能否用传统 CUDA Graph | ❌ 每个 N 都要不同图                   | ✅ 参数更新即可                        |
| 需要的技术            | **Piecewise CUDA Graph**       | 传统 CUDA Graph                   |
| CUDA Graph 节省的开销 | 阶段内多个 kernel 的 launch overhead | 整个 decode 流程的 launch overhead   |

***

## 10. 参考

- [NVIDIA CUDA Programming Guide - CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [CUDA Graphs 入门（知乎）](https://zhuanlan.zhihu.com/p/631187683)
- [NVIDIA cuda-samples - simpleCudaGraphs](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/simpleCudaGraphs/simpleCudaGraphs.cu)
- [SGLang Piecewise CUDA Graph 源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py)

***

> **关联笔记**：[prefill-decode-cuda-graph.md](./prefill-decode.md) — Prefill 和 Decode 阶段的详细工作原理及 PD 分离原因

