# K3 PP 切点与裁层怎么定

> 配套：[`SKILL.md`](./SKILL.md)（起服步骤）、
> [`../../knowledge/models/kimi-k3-layer-structure.md`](../../knowledge/models/kimi-k3-layer-structure.md)（层结构原理）。
> 本文只讲**决策**：切哪里、裁几层、参数怎么填。

---

## 1. PP 切点

`PPProxyTensors` 已传 `hidden_states` + `residual`（block residual），跨 stage 功能上可切。
但为少踩 **`attn_res_block_size=12`** 的边界坑，**切点优先落在 12 的倍数**
（0-based exclusive `end_layer`）。

| `pp_size` | 推荐 `SGLANG_PP_LAYER_PARTITION` | 说明 |
|---|---|---|
| 2（**满血**） | `48,45` | 在满血 MLA/block 边界切开；大段放后 rank 减 bubble |
| 4（**满血**） | `24,24,24,21` | 前三段整 block |
| 2（**裁 24 层**） | `12,12` | ✅ **推荐**：两个完整 attn_res block |
| 2（**裁 12 层**） | **不推荐** | 12 只能切成 `6+6`，落在 block 正中 → **keep=12 时用 `pp=1`** |

> **记不住就记这一条**：`keep=12 → pp=1`；`keep=24 → pp=2, PARTITION=12,12`。

---

## 2. 裁几层

### 为什么必须裁

```
满血 colocated:  4 机 × 16 die 扛 ~1.5TB 权重 + KV  →  已近满
PD 2P+2D:        每侧只剩 2 机，同权重复载两份角色   →  必须裁层或加机器
```

| `KEEP_LAYERS` | 约占层数 | 用途 |
|---|---|---|
| **12** | ~13% | **PD / 单机冒烟**（1 个 attn_res block）；`pp` 保持 **1** |
| **24** | ~26% | ✅ **首选**：PD + Prefill `pp=2`（`PARTITION=12,12`，两个完整 block） |
| 48+ | ≥50% | 接近半模，4 机 PD 仍可能吃紧 |

### 裁层必须同步改（缺一不可）

- `text_config.num_hidden_layers = KEEP`
- `linear_attn_config.kda_layers` 只保留 `<= KEEP` 的 1-based 编号
- `linear_attn_config.full_attn_layers` 同上

实际用 `prepare_k3_cut_override.py` 生成，别手写（手写容易漏后两项，且 override
会整体替换 `text_config`）→ 见 [`SKILL.md §1`](./SKILL.md)。

### 用 override，不要先物理切权重

**用满血 safetensors + override 即可**：loader 按层号加载，更高层不建模块。
不必先做物理 cut 权重。先例：`/home/weights/Kimi-K2.5-w4a8-cutlayers`（6 层）。

---

## 3. 首版拓扑（定死）

**裁 24 层 + PD 2P+2D + Prefill `pp=2`；Decode `pp=1`；关 DSPARK。**
显存仍紧时先 **keep=12 + 全 `pp=1`** 只通 PD，再升到 24 + pp。

| 角色 | 关键参数 |
|---|---|
| Prefill | `--disaggregation-mode prefill --nnodes 2 --tp-size 16 --pp-size 2 --dp-size 1` |
| Decode | `--disaggregation-mode decode --nnodes 2 --tp-size 32 --pp-size 1` |

> ⚠️ **开 PP 会连带关掉 overlap schedule**（Prefill 侧 `pp=2` 即触发，框架强制，无法用参数拦）。
> 启动日志会打 `Pipeline parallelism is incompatible with overlap schedule.`
> 影响：**吞吐预期会低于不开 PP 的基线**，对比 TTFT 时这是已知代价，不是配置错误。
> 机理与完整条件见 [`knowledge/inference/overlap-schedule.md`](../../knowledge/inference/overlap-schedule.md)。

说明：

- Prefill：`tp16 × pp2` 用满 2 机 × 16 die = 32 die；`SGLANG_PP_LAYER_PARTITION=12,12`
- Decode：先 **`pp=1`** —— CPP 收益主要在 **Prefill TTFT**，且 `pp=1` 不触发上面这条
- 端口避开同事：`HTTP 30001`；`dist-init` Prefill `5001` / Decode `5002`；bootstrap 另开
- `chunked-prefill-size` 先固定 `4096` 或 `8192`（**仅 Prefill**）

**降级路径**：2 机 Prefill 仍 OOM → 改 **1P+1D** + `keep=12` + `pp=1` 只验证 PD；
再 `keep=24` + Prefill `pp=2`。

> 具体机器 / node-rank 属于本机信息，不写进公开文档，以 `cpp_pd.sh` 与 `ROLE` / `NODE_RANK` 环境变量为准。

---

## 4. 一页参数速查

```text
MODEL=/home/weights/Kimi-K3-w4a8-int-moe
KEEP_LAYERS=24
OVERRIDE=.../k3_cut_l24.json          # prepare_k3_cut_override.py 生成

# ---- Prefill (2 nodes) ----
--disaggregation-mode prefill
--nnodes 2 --tp-size 16 --pp-size 2
--chunked-prefill-size 4096
--json-model-override-args "$(cat $OVERRIDE)"
export SGLANG_PP_LAYER_PARTITION=12,12
# 关：DSPARK / SGLANG_NPU_USE_MULTI_STREAM=0 / enable_pdmux
# 先不开：--enable-dynamic-chunking

# ---- Decode (2 nodes) ----
--disaggregation-mode decode
--nnodes 2 --tp-size 32 --pp-size 1
--json-model-override-args "$(cat $OVERRIDE)"
```

> 为什么关这些，见
> [`knowledge/models/kimi-k3-layer-structure.md §2.2`](../../knowledge/models/kimi-k3-layer-structure.md)。

---

## 5. 冒烟与验收

### 冒烟（不看性能）

目标：验证 **调度 / P2P / PD bootstrap**。**不看** NotifyDispatch、双流、满血 TTFT。

1. 生成 `keep=12` override
2. 单机（可选）：`--nnodes 1 --tp-size 8 --pp-size 2` + override，短 prompt generate
3. 两机 PD：1P+1D，`pp=1`，确认 KV 传输
4. 再上 Prefill `pp=2`

小模型替代（不依赖 K3 权重）：`test/registered/pp/test_pp_single_node_extra.py`。

### 验收（在裁层模型上，先 12 再 24）

1. **基线**：同裁层、同 PD 拓扑、`pp=1`、固定 chunk，测 TTFT（8K / 32K）
2. **CPP**：Prefill `pp=2` + 同 chunk，比基线 TTFT
3. Log/trace 确认：**多 chunk 在多个 PP stage 上重叠**（不是只有单 stage chunked prefill）
4. 再开 `--enable-dynamic-chunking`，`SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65~0.8`，
   初始 chunk ≈ 固定最优值的 **2–3×**
5. ⚠️ 满血回到「加机器或接受极紧显存」之前，**不要**把裁层数字当成生产 TTFT

## Living doc

切点/裁层/拓扑有变化 → 改本文（决策）；层结构或代码就绪度变了 →
改 [`knowledge/models/kimi-k3-layer-structure.md`](../../knowledge/models/kimi-k3-layer-structure.md)（事实）。
