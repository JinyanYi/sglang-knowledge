# 显存预算与 Prefill 调度

> 关联文档：[Prefill vs Decode](./prefill-decode.md)、[投机推理](./speculative-decoding.md)、[CUDA Graphs](./cuda-graphs.md)

---

## 1. Static vs Dynamic（组件名）

```mermaid
flowchart TB
    subgraph total ["整卡显存"]

        subgraph static ["Static — 长期占用"]
            S1["Target 模型权重"]
            S2["Mamba 主池"]
            S3["KV 池"]
            S4["Draft 权重"]
            S5["Draft KV"]
            S6["Mamba 投机中间态"]
            S7["ViT 权重"]
        end

        subgraph virtual ["动态预留空位（只记账）"]
            V1["pre × (1 − fraction)"]
        end

        subgraph dynamic ["Dynamic — graph / 峰值 / 运行时"]
            D1["通信缓冲"]
            D2["Target Decode Graph"]
            D3["Draft Graph"]
            D4["Draft Extend Graph"]
            D5["图像 / Feature 缓冲"]
            D6["Prefill 激活峰值"]
            D7["运行时临时 + 碎片"]
        end
    end
```

| 组件 | Static | Dynamic |
|------|--------|---------|
| 主模型 | 权重 | — |
| Mamba | 主池、投机中间态 | — |
| KV | KV 池 | — |
| 投机推理模型 | Draft 权重、 KV | Draft / Extend Graph |
| CUDA Graph | — | 主模型 Decode Graph |
| 多模态 | ViT 权重 | 图像 / Feature 缓冲 |
| Prefill | — | 激活峰值 |
| 通信 | — | HCCL 等 |
| 内存预算 | — | `pre×(1−fraction)` 仅预留 |

---

## 2. 启动时 KV 池怎么定（Concrete Example）

调用链（[`scheduler.py#L862`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L862) — `init_model_worker()`）：

```
init_model_worker()                    # scheduler
  ├─ init_tp_model_worker()            # 加载主模型权重，记录加载前空闲
  ├─ maybe_init_draft_worker()         # 加载 draft 模型权重
  └─ init_memory_pools()              # 协调所有 worker 分配内存池
       └─ tp_worker.alloc_memory_pool()
            └─ model_runner.alloc_memory_pool()
                 └─ _resolve_memory_pool_config()   # 测量加载后空闲，算 KV 预算
                      └─ _init_pools()              # 实际分配 KV + Mamba 池
```

> **关键**：`_profile_available_bytes()` 在 `alloc_memory_pool()` 内调用，此时主模型和 draft 模型均已加载完毕，`available_gpu_memory` 反映两者共同占用后的剩余显存，比仅加载主模型后少约 1.84 GB（draft 权重占用）。

### 2.1 公式与源码

```
# pre = 加载任何权重前的可用显存（init_torch_distributed 末尾测量）
# post = 主模型+draft 模型全部加载后的可用显存（alloc_memory_pool 时测量）

reserve_gb   = pre × (1 − mem_fraction_static)     ← 动态预留空位，不 malloc
rest_gb      = post − reserve_gb
rest_gb      −= Mamba 投机中间态（若开投机）
rest_gb      −= Mamba 主池
available_bytes = rest_gb × 2³⁰
KV tokens    = (available_bytes // cell_size) 按 page 对齐
final tokens = min(KV tokens, max_total_tokens)    ← 若设置了 cap
```

#### ① 测 `pre`（加载任何模型前）

[`model_runner.py#L1282`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L1282) — `init_torch_distributed()` 末尾：

```python
pre_model_load_memory = get_available_gpu_memory(
    self.device,
    self.gpu_id,
    distributed=get_world_group().world_size > 1,
    cpu_group=get_world_group().cpu_group,
)
return pre_model_load_memory
```

#### ② 动态预留 + `rest`（主模型 + draft 模型均加载完之后）

[`model_runner_kv_cache_mixin.py#L85`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L85) — `_profile_available_bytes()`：

```python
# 此时主模型和 draft 模型均已加载，available_gpu_memory 比仅加载主模型时更小
available_gpu_memory = get_available_gpu_memory(
    self.device,
    self.gpu_id,
    distributed=get_world_group().world_size > 1,
    cpu_group=get_world_group().cpu_group,
)

rest_memory = available_gpu_memory - pre_model_load_memory * (
    1 - self.mem_fraction_static
)

if self.mambaish_config is not None:
    rest_memory = self.handle_max_mamba_cache(rest_memory)

return int(rest_memory * (1 << 30))
```

#### ③ Mamba 投机中间态（从 `rest` 里扣预算）

[`model_runner_kv_cache_mixin.py#L138`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L138) — `handle_max_mamba_cache()`，显式指定 `max_mamba_cache_size` 且开投机时：

```python
if has_spec_dec:
    ratio = self._calculate_mamba_ratio()
    capped_reqs = min(
        server_args.max_running_requests
        // (self.dp_size if server_args.enable_dp_attention else 1),
        server_args.max_mamba_cache_size // ratio,
    )
    intermediate_size = (
        config.mamba2_cache_params.mamba_cache_per_req
        * capped_reqs
        * server_args.speculative_num_draft_tokens
    )
    total_rest_memory = total_rest_memory - (intermediate_size / (1 << 30))
```

#### ④ Mamba 主池（继续从 `rest` 里扣预算）

[`model_runner_kv_cache_mixin.py#L219`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L219)：

```python
mamba_state_memory = (
    server_args.max_mamba_cache_size
    * config.mamba2_cache_params.mamba_cache_per_req
    / (1 << 30)
)
return total_rest_memory - mamba_state_memory
```

#### ⑤ `rest` → KV token 数

[`pool_configurator.py#L222`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/pool_configurator.py#L222) — `DefaultPoolConfigurator.calculate_pool_sizes()`：

```python
def calculate_pool_sizes(
    self, available_bytes: int, page_size: int
) -> MemoryPoolConfig:
    max_total_num_tokens = available_bytes // self._cell_size
    max_total_num_tokens = max_total_num_tokens // page_size * page_size
    return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)
```

入口在 [`_resolve_memory_pool_config()`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L1051)：

```python
available_bytes = self._profile_available_bytes(pre_model_load_memory)
configurator = create_memory_pool_configurator(self)
config = configurator.calculate_pool_sizes(available_bytes, page_size)
```

#### ⑥ `max_total_tokens` 上限（--max-total-tokens）

[`model_runner_kv_cache_mixin.py#L952`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L952) — `_apply_token_constraints()`：

```python
user_limit = self.server_args.max_total_tokens

if user_limit is not None:
    if user_limit > token_capacity:
        logging.warning(
            f"max_total_tokens={user_limit} is larger than the profiled value "
            f"{token_capacity}. Use the profiled value instead."
        )
    token_capacity = min(token_capacity, user_limit)

return token_capacity
```

[`_resolve_memory_pool_config()`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L1063) 里调用：

```python
constrained = self._apply_token_constraints(config.max_total_num_tokens)
if constrained != config.max_total_num_tokens:
    config = configurator.calculate_pool_sizes_from_max_tokens(
        constrained, page_size
    )
```

**注意**：`min(cap, profiled)` 只能**缩小**池子；cap 大于 profiled 时打 warning，仍用 profiled 值。

### 2.2 实测数字（Qwen3.6-35B-A3B，TP=2，NPU）

**⑦ = ④ − ⑤ − ⑥**，**KV 池大小由 ⑦ 决定**，不是「加载后空闲减 Mamba 剩下的 free」。

#### 关键：加载后空闲已含 draft 权重

```
加载后空闲 ≈ 26.09 GB   ← 主模型 + MTP draft 均加载完后的空闲（不是整卡 60G）
③ = 加载前空闲 × (1−f)  ← 只在算 KV 预算时从②扣掉，不是再占一块物理显存
④ = 加载后空闲 − ③      ← ③ 只在这里出现一次
⑦ = ④ − ⑤ − ⑥         ← 全部给 KV 预算
⑧ = ⑦ 换成 bytes ÷ cell_size
```

#### Case A：实测（fraction = **0.8**）→ 479488 tokens

| 步骤 | 计算 | 数值 |
|------|------|------|
| ① 加载前空闲 | | **60.77 GB** |
| ② 两模型加载后空闲 | | **26.09 GB** |
| ③ 动态预留 | 60.77 × (1−0.8) | **12.15 GB** |
| ④ 预算起点 | 26.09 − 12.15 | **13.94 GB** |
| ⑤ Mamba 中间态 | | **7.55 GB** |
| ⑥ Mamba 主池 | | **1.97 GB** |
| **⑦ KV 可用预算** | 13.94 − 7.55 − 1.97 | **≈ 4.42 GB** |
| ⑧ KV 池 | 479488 tokens × 10240 | **≈ 4.57 GB** |

（⑦ 与 ⑧ 的微小差异来自 mamba_cache_per_req 内部精度取整。）

#### Case B：若 fraction = **0.9**（同一机器，只改 fraction）

| 步骤 | 计算 | 数值 |
|------|------|------|
| ③ 动态预留 | 60.77 × (1-0.9) | **6.08 GB** |
| ④ 预算起点 | 26.09 − 6.08 | **20.01 GB** |
| ⑤⑥ Mamba | 同左 | **7.55 + 1.97 = 9.52 GB** |
| **⑦ KV 可用预算** | 20.01 − 9.52 | **≈ 10.49 GB** |
| ⑧ KV 池（估） | 10.49×2³⁰÷10240 | **≈ 1.10M tokens** |

0.9 时 KV 预算 ~10.5G，约是 0.8 时的 **2.4 倍**。

#### ⑩ 池分配后剩余 free ≈ 11G 是什么？

这是 **KV + Mamba 真分配完之后** 的物理空闲，**不是** ⑦：

```
② 两模型加载后空闲       26.09 GB
  − Mamba 真分配          ~9.52 GB
  − KV 真分配            ~4.57 GB   ← 由 ⑦ 决定
─────────────────────────────────
剩余 free               ~12 GB     ← 给 Graph 等用
```

**~12G 是「池子分完还剩多少 free」；4.57G 是「KV 池分到多少」——两个不同量。**

```mermaid
flowchart TB
    POST["② 两模型加载后空闲 ≈ 26.1G"]
    subgraph budget ["预算记账（定池子大小）"]
        R4["④ 预算起点 = ② − 加载前空闲×(1−f)"]
        M5["⑤ 中间态 ~7.6G"]
        M6["⑥ 主池 ~2G"]
        R7["⑦ KV可用预算 ≈ 4.4G（f=0.8）"]
        KV8["⑧ KV 池 ≈ 4.6G / 479k tokens"]
    end
    subgraph physical ["物理分配（从②里实际扣减）"]
        PM["Mamba ~9.5G"]
        PKV["KV ~4.6G"]
        FREE["剩余 free ~12G"]
    end
    POST --> R4
    R4 --> M5 --> M6 --> R7 --> KV8
    POST --> PM --> PKV --> FREE
```

### 2.3 参考数值（Qwen3.6-35B-A3B，TP=2，NPU，fraction=0.8）

| 含义 | 代码变量 | 值 |
|------|---------|-----|
| ① 加载前空闲 | `pre_model_load_memory` | 60.77 GB |
| ② 两模型加载后空闲 | `available_gpu_memory` | 26.09 GB |
| ③ 动态预留 | `pre_model_load_memory × (1−f)` | 12.15 GB |
| ④ 预算起点 | `rest_memory`（扣③后） | 13.94 GB |
| ⑤ Mamba 中间态 | — | 7.55 GB |
| ⑥ Mamba 主池 | — | 1.97 GB |
| ⑦ KV 可用预算 | `rest_memory`（扣⑤⑥后） | **≈ 4.42 GB** |
| ⑧ KV 池（实分配） | — | **479488 tokens ≈ 4.57 GB** |
| 池分配后剩余 free | — | **~12 GB** |
| NPU graph | — | **~1.3 GB** |

要点：

- **draft 模型（~1.84 GB）已计入②**，所以②= 26.09 GB 而非 27.93 GB。
- fraction **0.9** 会把 ④ 从 13.94 抬到 20.01，⑦ 抬到 **~10.49G**，KV 池约 **1.10M tokens**。
- 若同时加 `--max-total-tokens N`，则 `min(profiled, N)` 决定最终池大小（只能缩小，不能扩大）。

---

## 3. 运行时：PrefillAdder 在哪判断？

| 阶段 | 文件 | 函数 |
|------|------|------|
| 调度入口 | `scheduler.py` | `get_next_batch_to_run()` → `get_new_batch_prefill()` |
| 组 batch | `scheduler.py` | `_get_new_batch_prefill_raw()` |
| 准入判断 | `schedule_policy.py` | `PrefillAdder.add_one_req()` |

### 3.1 剩余 token 预算

```python
rem_total_tokens = pool.available_size() + tree_cache.evictable_size() - offset
```

- `disable-radix-cache` 时 **evictable = 0**。
- 每条请求准入检查：`input + max_new + page_size` 不能超过 `rem_total_tokens`。
- 加入后 `offset` 增加（含未来 output 预留）。

### 3.2 返回值

| 结果 | 含义 | 调度器 |
|------|------|--------|
| `CONTINUE` | 还能加 | 继续扫队列 |
| `NO_TOKEN` | KV 池不够 | `batch_is_full=True`，**break** |
| `OTHER` | 触达 `prefill-max-requests` / `max-prefill-tokens` 等 | **break** |

`get_next_batch_to_run` **优先 prefill**；`get_new_batch_prefill()` 返回 `None`（含 `batch_is_full`）才 decode。

### 3.3 日志字段对照

| 日志 | 来源 | 与 PrefillAdder |
|------|------|-----------------|
| `full token usage` | `pool_stats_observer.py` | 同一 KV 池的 `used/size`，Adder 用 `available` |
| `#queue-req` | scheduler | `waiting_queue` 长度 |
| `#running-req` | scheduler | `running_batch` 长度 |

---

## 4. Case：3.5k 输入，fraction=0.8，decode@94（失败形态）

### 4.1 启动参数摘要

```
mem-fraction-static 0.8
max-running-requests 122
max-mamba-cache-size 122
prefill-max-requests 12
disable-radix-cache          → Mamba ratio=1，evictable=0
无 max-total-tokens
```

### 4.2 日志摘要

| 阶段 | 现象 |
|------|------|
| Prefill | 1 + 12×7 + **9** = **94** 路进 running |
| 最后一轮 prefill | 只加了 **9** 条（不是 12），`full token usage: 0.70` |
| Decode | **#running-req: 94**，**#queue-req: 28** 一直不降 |
| 池子 | `#full token: 347392`，`usage: 0.72` → 池约 **483k tokens** |

### 4.3 根因链

```
fraction 0.8 → KV 池约 48 万 token（0.9 时约 110 万）
    ↓
每条 3.5k 输入准入约需 input+max_new+page ≈ 5128 token
122 路全占需 ≈ 62 万 token 预算 → 池物理上不够
    ↓
Prefill 到 94 路时 usage≈0.70，再加第 10～12 条 → NO_TOKEN
    ↓
batch_is_full → 不再组 prefill → decode@94
    ↓
28 条留在 waiting_queue（#queue-req: 28）
```

**不是 Mamba 卡住**：`disable-radix-cache` + ratio=1 → 122 槽可用，decode 时 `mamba num: 94`（未满 122）。

**是 KV 池偏小 + PrefillAdder `NO_TOKEN`**。

### 4.4 与稳定 case 对比

| | 3.5k 失败 (0.8) | 3.5k 稳定 (0.9) | 3.5k 稳定 (0.9 + max-total-tokens 659840) |
|--|-----------------|-----------------|------------------------------------------|
| KV 池 | ~479k tokens | ~1100k tokens | ~660k tokens |
| Prefill 完成 | 94 / 122 | 122 / 122 | 122 / 122 |
| Decode | @94，queue 28 | @122 | @122 |
| 剩余 free | ~12 GB | ~6 GB | ~10 GB |

单条准入估算（output 1500）：

```
3500 + 1500 + 128 ≈ 5128 token / 请求
94 × 5128 ≈ 481k  → 与 0.70×479k 一致
```

---

## 5. Case：64k 输入，decode@32（Mamba 限制）

启动：`max-running-requests 40`，`max-mamba-cache-size 160`，`extra_buffer`（ratio=5），`fraction 0.71`。

```
有效并发 = min(40, 160÷5) = 32
```

日志：`#running-req: 32`，`#queue-req: 8` → **Mamba 槽位限制**，不是 KV 单独问题（与 3.5k@94 不同）。

---

## 6. Case：Kimi-K3 满血 W4A8 PD（4×A3，93 层，DeepEP）

拓扑（round3 起）：Prefill 209+212 与 Decode 216+217 **都是** `tp=16 pp=2 nnodes=2`，`SGLANG_PP_LAYER_PARTITION=48,45`，`ep_size=16`。权重 `/home/weights/Kimi-K3-w4a8-int-moe`（磁盘 ~1.49TB，93 层）。每 die 64GB。Prefill DeepEP 可开关；Decode **常开** DeepEP（LL）。冒烟：`8k.sh` in=8000 out=1000 conc=1。

### 6.1 `mem-fraction-static` **管不到** weight-load OOM

公式里 `pre` 是 `init_torch_distributed` **之后**才测的。`mem-fraction` 只在 **权重装完之后** 算 KV（`reserve = pre × (1−f)`）。`create_weights` / `torch.empty` 炸的时候还没走到 `alloc_memory_pool`。

Round0（2026-08-13）四机在 `modelslim_w4a8_int8_moe.create_weights` 炸：

```
Load weight begin. avail mem=17.81 GB
NPU out of memory. Tried to allocate 296.00 MiB
  (NPU 10; 61.27 GiB total; 17.36 GiB allocated; 183 MiB free)
```

**当时误判**：以为 HCCL=2048 从卡上划走了 ~44GB（PyTorch 看不见）。
**更正（round2/4 干净卡实测）**：Load weight begin **avail≈60.8–61.1 GB**，HCCL=800 / 1200 / **2048 都一样**。Round0 的 17GB 是 **NPU 上残留占用**（上次崩没清干净 / 别人进程），不是 2GB HCCL 吃掉 44GB。`HCCL_BUFFSIZE` 多数是 **第一次集合通信 / DeepEP tiling 时**才真正要连续块，不是 load 前就把卡掏空。

满血 93 层 **必须 pp=2**。Decode round2 `pp=1 tp=32`：每个 rank 扛全部 93 层，create_weights 把干净卡 61GB 打满 OOM。Prefill `pp=2` 实测：

| | PP0 (209) | PP1 (212) |
|--|-----------|-----------|
| Load weight end usage | **53.58 GB** | **51.20 GB** |
| Load weight end avail | ~7.3–7.5 GB | ~9.6–9.9 GB |
| Memory pool end avail | ~4.85–5.18 GB | ~7.2–7.55 GB |
| `max_total_num_tokens` | **176384**（8k×1 足够） | 同切 |

粗算：磁盘 1486GB × (48/93) / 16 die ≈ **48GB/die** 量级，与 PP0 ~53.6GB 同阶（含激活/量化辅助）。

### 6.2 DeepEP 对 HCCL 的下限（decode LL）

`MoeLowLatencyDispatchV2` 窗口（见 DeepEP 笔记）：

```
actualSize ∝ maxBs × (ep × local_experts + (k+shared))
ep × local_experts = num_experts = 896   ← 不随 EP 变小
```

历史：EP8、maxBs=64 → **1733MB**，HCCL=800 会打 `HCCL_BUFFSIZE is too SMALL`。
本次 `pp=2` 后 **EP=16**，`--cuda-graph-bs 16`，`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 曾设 64。

**`ret=24` ≠ `HCCL_BUFFSIZE is too SMALL`：**

| 现象 | 含义 |
|------|------|
| 日志有 `NEEDED=1733MB, HCCL_BUFFSIZE=800MB` | CheckWinSize 公式不够，升 buffsize |
| 只有 `HcclAllocComResourceByTiling ret=24`，**没有** too SMALL | 公式过了，但 HCCL 按 **整块 BUFFSIZE** 从 **KV 池分完后的 leftover** 申请连续内存失败 |

Round4：env=2048 已读到，PP0 leftover≈**4.7GB**（≈ `pre×(1−0.92)`），仍 ret=24。再升 2048 没用。应 **缩小 maxBs**（dispatch tokens 64→16，窗口约 433MB）并把 HCCL **降到 800**，必要时略降 fraction 把 leftover 从 4.7 抬到 ~6GB。fraction 不能低于 `1−post/pre`≈0.88，否则 KV rest 变负。

Round5 实测印证 leftover 公式：`f=0.90` → leftover **5.9–6.2GB**，HCCL=800 + maxBs=16 一次过 tiling，Decode `fired up`。随后 router `:8077` + `8k.sh` **Successful requests=1**（8000+1000，E2E 106s）。

Prefill 走 DeepEP **normal** 不是 LL，HCCL=800 已过 Memory pool + `fired up`。不必跟 Decode 绑死同一 buffsize。

### 6.3 调参方向（8k 冒烟，DeepEP 要开）

| 旋钮 | 对 weight-load | 对 KV / 8k | 对 DeepEP |
|------|----------------|------------|-----------|
| `HCCL_BUFFSIZE` | 干净卡上 **几乎不影响** load begin avail | **按整块**从 leftover 申请（在 KV 之后） | Prefill normal：800 够。Decode LL：先把 maxBs/dispatch tokens 降到窗口 < leftover，再设 buffsize；盲目 2048 会在 leftover≈4.7GB 上 `ret=24` |
| ↑ `mem-fraction-static` | 无 | leftover≈`pre×(1−f)` **变小**；KV 变大 | tiling 更挤 |
| ↓ `mem-fraction-static` | 无 | leftover 变大；下限 `f ≥ 1−post/pre`≈0.88 | 给 HCCL 连续块 |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 无 | 无 | **maxBs**，线性缩小 LL 窗口。8k×1 用 16 即可 |
| `--disable-cuda-graph` | 无 | 省 graph | 冒烟关；正式测 Decode 再开 |
| **pp=2（P 和 D）** | **满血 93 层唯一能装下的切法** | 每 stage ~48/45 层 | `ep_size=16` |

8k×1 约需 9k token，KV=176k 已富余。NPU：`expandable_segments` **不能**和 `max_split_size_mb` 同开（round1 立刻 `_npu_init` RuntimeError）。

### 6.4 实验轮次

| Round | mem-f | P HCCL | D 并行 | D HCCL | graph | 结果 |
|-------|-------|--------|--------|--------|-------|------|
| 0 | 0.75 | 2048 | tp32 pp1 | 2048 | on | create_weights OOM，avail≈**17GB**（卡不干净，不是 HCCL） |
| 1 | 0.92 | 800 | tp32 pp1 | 1200 | off | `expandable_segments` + `max_split_size_mb:256` → `_npu_init` 立刻炸 |
| 2 | 0.92 | 800 | tp32 pp1 | 1200 | off | 干净卡 load begin≈**61GB**。Prefill 权重+KV 成功，Uvicorn `:30001`。Decode **pp=1 满 93 层 OOM** |
| 3 | 0.92 | 800 | **tp16 pp2** | 1200 | off | Decode 权重装完；随后 **`HcclAllocComResourceByTiling ret=24`** |
| 4 | 0.92 | 800 | tp16 pp2 | **2048** | off | env 确认 `HCCL_BUFFSIZE=2048`。权重 53.58GB / pool leftover **PP0≈4.7GB**。**无** `too SMALL`，仍 `HcclAllocComResourceByTiling ret=24`（2048 整块从 4.7GB leftover 申请失败） |
| 5 | **0.90** | 800 | tp16 pp2 | **800** | off | **成功 + 8k 冒烟通过**。`max_dispatch_tokens=16`。PP0 leftover≈**5.9–6.2GB**（≈`pre×0.10`），KV=**84608**。`fired up`。`8k.sh` 1/1，in=8000 out=1000，E2E **106s**，output **9.41 tok/s**（graph 关着；TTFT/ITL 报表为 0，mini-lb 流式统计不可信） |

脚本备份：

- `K3/backups/round0_mem0.75_hccl2048/`
- `K3/backups/round1_mem0.92_hcclP800_D1200_bad_alloc_conf/`
- `K3/backups/round2_mem0.92_hcclP800_D1200_decode_pp1/`
- `K3/backups/round3_decode_pp2_hccl1200/`
- `K3/backups/round4_mem0.92_hcclP800_D2048_decode_pp2/`
- `K3/backups/round5_mem0.90_hccl800_dispatch16/`

启动（容器 `sglang-yjy-k3`，不要 wall / 不要动别人的容器）：

```
# 209: NODE_RANK=0 USE_DEEPEP=1 bash prefill.sh
# 212: NODE_RANK=1 USE_DEEPEP=1 bash prefill.sh
# 216: NODE_RANK=0 bash decode.sh
# 217: NODE_RANK=1 bash decode.sh
# 209 ready 后: bash router.sh && bash 8k.sh
```

---

## 7. 调参速查

| 目标 | 手段 |
|------|------|
| 增大 KV 池 | ↑ `mem-fraction-static`；可选 `max-total-tokens` cap |
| 更多路 decode | 查 `max-mamba-cache-size ÷ ratio`（radix/extra_buffer 影响 ratio） |
| 每轮 prefill 条数 | `prefill-max-requests` |
| 单轮 prefill token | `max-prefill-tokens` |
| **weight-load OOM（avail≈17GB）** | 先确认卡干净（残留占用）；HCCL 在干净卡上几乎不改 load begin；满血 93 层用 **pp=2** |
| **装完权重 rest 变负** | ↑ fraction（少 reserve） |
| DeepEP LL tiling `ret=24`（无 too SMALL） | leftover 不够整块 `HCCL_BUFFSIZE`：降 maxBs/`NUM_MAX_DISPATCH_TOKENS`，并降 buffsize；或略降 fraction 增大 leftover（下限 ≈0.88） |
| DeepEP LL `HCCL_BUFFSIZE is too SMALL` | 升 decode buffsize（EP8/maxBs64 ≈ 1733MB） |
| 准入逻辑代码 | `schedule_policy.py` → `PrefillAdder` |

---

## 标签

`sglang` `显存` `KV-cache` `mem-fraction-static` `PrefillAdder` `调度` `mamba` `NPU` `HCCL_BUFFSIZE` `DeepEP` `Kimi-K3`
