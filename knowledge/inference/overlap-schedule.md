# Overlap Schedule：什么情况下会被关掉

> **本文只讲「为什么 / 什么条件下」。**
> 起 K3 PP 服务时它会带来什么后果 →
> [`runbooks/k3-pd-pp/pp-partition-and-cut.md`](../../runbooks/k3-pd-pp/pp-partition-and-cut.md)。

---

## 1. 它是什么

`--disable-overlap-schedule`（server arg，默认 `False` = 开启 overlap）。

定义：`python/sglang/srt/server_args.py:993-1000`

```python
disable_overlap_schedule: A[
    bool,
    Arg(
        help="Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.",
        resolvable=True,
    ),
    NS("schedule"),
] = False
```

作用：**让 CPU 调度器与 GPU/NPU model worker 重叠执行** —— 调度下一步的同时，当前步还在设备上跑。
关掉它会有明显的吞吐损失，所以「被自动关掉」是需要知道的事。

`resolvable=True` 意味着它**可以被框架在启动期改写**（这正是下面这些自动禁用的实现基础）。

---

## 2. 自动禁用条件（完整清单）

按代码位置整理。这些都是 **`resolvable` 后置处理**在启动期把值改成 `True`，
**不是用户可选项** —— 手动传 `--disable-overlap-schedule=false` 也拦不住。

| # | 触发条件 | 代码位置 | 启动日志 |
|---|---|---|---|
| 1 | **`pp_size > 1`（pipeline parallel）** ★ | `arg_groups/overrides.py:2608` | `Pipeline parallelism is incompatible with overlap schedule.` |
| 2 | **`SGLANG_EMBEDDINGS_SPARSE_HEAD` 被设置** | `arg_groups/overrides.py:1946` | `Overlap scheduler is disabled when using sparse head for embedding model.` |
| 3 | **diffusion LLM**（`dllm_algorithm` 非空） | `arg_groups/overrides.py:2662` | `Overlap schedule is disabled because of using diffusion LLM inference` |
| 4 | **mamba radix cache 落到 `no_buffer`** | `arg_groups/overrides.py:1594` | （无日志） |
| 5 | **device 为 `mps` 且未用 MLX** | `server_args.py:4444` | （无日志） |
| 6 | **CPU + 投机解码** | `arg_groups/speculative_hook.py:14` | `Overlap schedule is not implemented for speculative decoding on CPU.` |

### 第 4 条是「协商」不是「单向禁用」

这条比较特殊，是**双向**的 —— overlap 与 mamba radix cache 策略互相决定：

```python
1594:    if view.mamba_radix_cache_strategy == "auto":
1595:        wants_overlap = not view.disable_overlap_schedule
1596:        wants_paging = view.page_size is not None and view.page_size > 1
1597:        if (wants_overlap or wants_paging) and supports_mamba_cache_extra_buffer(
1598:            view, model_arch
1599:        ):
1600:            declared["mamba_radix_cache_strategy"] = "extra_buffer"
1601:        else:
1602:            declared["mamba_radix_cache_strategy"] = "no_buffer"
1603:            declared["disable_overlap_schedule"] = True
```

即：**想要 overlap → 若架构支持就用 `extra_buffer`；不支持就退回 `no_buffer` 并关掉 overlap。**
所以反过来说也成立：**手动关掉 overlap 会让 mamba radix cache 走 `no_buffer`**（当 `page_size==1` 时）。

---

## 3. 最易混淆的一点：PP 关，PD 分离**不关**

| 特性 | 是否自动关 overlap |
|---|---|
| **PP（`pp_size > 1`）** | ✅ **会** |
| **PD 分离（`disaggregation_mode` 非空）** | ❌ **不会** |

`arg_groups/pd_disaggregation_hook.py`（全文 157 行）**没有任何对 overlap 的处置** ——
它只处理 `dcp_size` 校验、decode 侧 radix cache 强制改 chunk cache、
`staging buffer` 与 transfer backend 的兼容性检查。

> 记忆法：**切流水线（PP）会关 overlap；拆机器（PD）不会。**

---

## 4. NPU 上还有另一个叫「overlap stream」的开关，别混

`python/sglang/srt/environ.py:1132`

```python
SGLANG_ENABLE_OVERLAP_PLAN_STREAM = EnvBool(False)
```

| | `disable_overlap_schedule` | `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` |
|---|---|---|
| 层次 | 调度层（CPU scheduler ↔ worker） | NPU 图重放层 |
| 默认值 | `False`（**默认开启**） | `False`（**默认关闭，opt-in**） |
| 是否会被自动改 | **会**（上表 6 条） | **不会**（无任何自动禁用逻辑） |
| 唯一读取点 | 多处 | `hardware_backend/npu/graph_runner/npu_graph_runner.py:229` |
| 控制什么 | 调度与执行是否重叠 | 投机解码图重放时是否把 `mrope_positions` 拷进静态图 buffer |

后者只在 **NPU + 投机解码 + 多模态 mrope + 图重放** 这条窄路径上生效。

另外 `server_args.py:5586` 有 `envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.set(False)`，
名字最像，但那是 **HIP（AMD）+ DeepSeek-V4** 分支，**与 NPU 无关**。

---

## 5. 怎么确认自己被关了

1. **看启动日志**：第 1/2/3/6 条都会打 `logger.warning`，直接搜 `Overlap` 即可
2. **看 resolved 后的参数**：该字段是 `resolvable=True`，启动末期的值才是最终值，
   不要只看你命令行传了什么
3. 第 4、5 条**不打日志**，只能靠上面条件反推

---

## 相关

- [`models/kimi-k3-layer-structure.md`](../models/kimi-k3-layer-structure.md) §2.2 — 开 K3 PP 时的机制性约束
- [`runbooks/k3-pd-pp/pp-partition-and-cut.md`](../../runbooks/k3-pd-pp/pp-partition-and-cut.md) — PP 切点与首版拓扑
- [`inference/cuda-graphs.md`](./cuda-graphs.md) — 图捕获/重放（与 NPU graph runner 相关）
