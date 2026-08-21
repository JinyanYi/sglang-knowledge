# Kimi-K3 Chunked PP（CPP）起步结论

> 约束更新（2026-08-06）：目标是 **PD 分离**；集群只有 **4×A3（4×16×64G ≈ 4TB HBM）**；满血 w4a8 权重磁盘约 **1486GB**，colocated 四机已近打满 → **PD 拆机后必须裁层** 才能起步。
>
> **两机验证（推荐）：** Prefill=`80.5.17.34`，Decode=`80.5.17.38`。脚本见 `[scripts/kimi-k3-cpp/run_prefill_34.sh](./scripts/kimi-k3-cpp/run_prefill_34.sh)` / `[run_decode_38.sh](./scripts/kimi-k3-cpp/run_decode_38.sh)` / `[run_router.sh](./scripts/kimi-k3-cpp/run_router.sh)`。
>
> **代码结论（209 `0728_dspark` @ e0263a9）：** K3 已接 PP；PD（含 Ascend transfer + Prefill `pp_size`）框架已在；**缺的是裁层配置 + 启动参数 + 冒烟，不是从零写功能。**

远程实现仓：`/home/y00970600/KimiK3/sglang-kimiK3`（分支 `**0728_dspark`**）  
满血权重：`/home/weights/Kimi-K3-w4a8-int-moe`（209）/ `/home/weights/Kimi-K3-W4A8`（80.5.17 集群）  
脚本目录：`[scripts/kimi-k3-cpp/](./scripts/kimi-k3-cpp/)`

---

## 0. 代码 walkthrough（0728_dspark）— 冒烟还是缺件？


| 能力                                                                      | 状态       | 证据                                                                                               |
| ----------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------ |
| K3 `make_layers` / `PPMissingLayer` / `PPProxyTensors(hidden+residual)` | **已有**   | `kimi_k3.py`                                                                                     |
| `--pp-size` + `SGLANG_PP_LAYER_PARTITION`                               | **已有**   | 通用 PP                                                                                            |
| `--disaggregation-mode prefill/decode`                                  | **已有**   | `server_args` + `scheduler.init_disaggregation`                                                  |
| Ascend KV transfer                                                      | **已有**   | `disaggregation/ascend/`，文档要求 `ASCEND_MF_STORE_URL` + `--disaggregation-transfer-backend ascend` |
| Prefill 侧 `pp_size>1` 传进 bootstrap                                      | **已有**   | `PrefillBootstrapQueue(..., pp_size=...)`；ascend `conn.py` 有 `pp_size>1` KV 切片                   |
| `--enable-dynamic-chunking`                                             | **已有**   | 需 `pp_size>1`，第二步再开                                                                              |
| `--json-model-override-args` 裁层                                         | **已有**   | 改 `num_hidden_layers` + trim `linear_attn_config`                                                |
| DSPARK + PD/PP 同开                                                       | **首版别开** | 变量太多                                                                                             |
| 同机/跨机 router                                                            | **已有**   | `sglang_router.launch_router --pd-disaggregation`                                                |


**结论：功能基本齐，按文档冒烟即可。** 你们要补的是运维面：裁层 KEEP、34/38 角色脚本、`ASCEND_MF_STORE_URL`、先关 spec、先同 TP 再开 Prefill PP。

注意：`init_zbal_on_npu` 在 `pp_size>1` 时打 error log（要求 zbal mix mode），不一定硬退出；若 PP 起不来再查 `SGLANG_ZBAL_LOCAL_MEM_SIZE` / mix alloc。

---

## 1. 断点分析（breakpoint-audit）

### 1.1 层结构（满血 93 层）


| 项                     | 值                                                                 |
| --------------------- | ----------------------------------------------------------------- |
| `num_hidden_layers`   | **93**                                                            |
| 注意力模式                 | 每 4 层：`KDA×3 + MLA×1`（config 里层号 **1-based**）                     |
| `full_attn_layers`    | 4,8,12,…,92,93（24 个 MLA）                                          |
| `kda_layers`          | 其余 69 个                                                           |
| MoE                   | `first_k_dense_replace=1`，`moe_layer_freq=1` → 几乎每层 MoE           |
| Experts               | `num_experts=896`，`num_experts_per_tok=16`，`num_shared_experts=2` |
| `attn_res_block_size` | **12**（K3 特有 block residual，不是普通 Transformer residual）            |
| DSA / index topk      | **无**（不像 DeepSeek-V3 DSA 需要跨 PP stage 传 `topk_indices`）           |


0-based 前 12 层示例：`KDA,KDA,KDA,MLA` 重复 3 次，正好一个 `attn_res` block。

### 1.2 PP 切点建议

`PPProxyTensors` 已传 `hidden_states` + `residual`（即 block residual），跨 stage **功能上可切**。  
为少踩 `attn_res_block_size=12` 的边界坑，**切点优先落在 12 的倍数**（0-based exclusive `end_layer`）：


| pp_size   | 推荐 `SGLANG_PP_LAYER_PARTITION` | 说明                                          |
| --------- | ------------------------------ | ------------------------------------------- |
| 2（满血）     | `48,45`                        | 在满血 MLA/block 边界切开；大段放后 rank 减 bubble       |
| 4（满血）     | `24,24,24,21`                  | 前三段整 block                                  |
| 2（裁 24 层） | `12,12`                        | **推荐**：两个完整 attn_res block                  |
| 2（裁 12 层） | 不推荐                            | 12 只能切成 6+6，落在 block 正中；**keep=12 时用 pp=1** |


**结论（呼应 supervisor）：** 没有 DSA 跨 stage 元数据；KDA cache / MoE 都是 per-layer 本地状态；跨 stage 只需 activation+residual。断点**没有特别怪的结构障碍**，特殊点只有 `**attn_res_block_size=12` 建议对齐**。

### 1.3 裁层建议（PD + 仅 4 机）

满血 ~1.49TB 权重；PD 若 **2P+2D**，每侧约 2TB HBM，还要 KV/DeepEP/图 → **满血 PD 极紧或直接 OOM**。


| `KEEP_LAYERS` | 约占层数 | 用途                                                                           |
| ------------- | ---- | ---------------------------------------------------------------------------- |
| **12**        | ~13% | **PD / 单机冒烟**（1 个 attn_res block）；`pp` 先保持 **1**（12 无法在 block 边界均分到 2 stage） |
| **24**        | ~26% | **首选 PD + Prefill `pp=2**`（`PARTITION=12,12`，两个完整 block）                     |
| 48+           | ≥50% | 接近半模，4 机 PD 仍可能吃紧                                                            |


裁层必须同步改：

- `text_config.num_hidden_layers = KEEP`
- `linear_attn_config.kda_layers` / `full_attn_layers` 只保留 `<= KEEP` 的 1-based 编号

用满血 safetensors + override 即可（loader 按层号加载，更高层不建模块）；不必先做物理 cut 权重。参考先例：`/home/weights/Kimi-K2.5-w4a8-cutlayers`（6 层）。

---

## 2. 代码 PP 就绪审计（pp-code-ready）

### 2.1 已具备（`kimi_k3.py`）


| 能力                                                                | 位置                                     |
| ----------------------------------------------------------------- | -------------------------------------- |
| `get_pp_group` / `make_layers` / `start_layer`/`end_layer`        | `KimiLinearModel.__init_`_ ~L1104–1129 |
| 非本 rank `PPMissingLayer`（embed / norm / lm_head）                  | 首末 rank 分支                             |
| `PPProxyTensors({hidden_states, residual})`                       | 非 last rank return ~L1226              |
| 外层 `KimiK3ForCausalLM` / VLM wrapper 透传 `start_layer`/`end_layer` | ~L1327+                                |


对照 `kimi_linear` / `deepseek_v2`：**语言主干 PP 骨架已齐**，不是从零接 PP。

### 2.2 冲突与首版必须关掉的项


| 项                                    | 结论                                                                                              |
| ------------------------------------ | ----------------------------------------------------------------------------------------------- |
| **DSPARK / speculative**             | 现网开着；PP+spec 变量多，且部分路径强制 draft 在 pp0。**首版 CPP 关 spec**                                          |
| **PD-Multiplexing (`enable_pdmux`)** | 断言 `pp_size==1` 且不能 disagg。**不要开**；用真正的 `--disaggregation-mode prefill/decode`                  |
| **PD disaggregation + PP**           | Ascend/Mooncake KV 传输路径有 `pp_size > 1` 分支（按当前 PP stage 的层切 KV）。**框架声称支持，但仍是高风险组合 → 先裁层 + 小 pp** |
| **DeepEP**                           | 与 TP/EP 同开；PP 只传 activation，EP 仍在 stage 内。可保留，OOM 时再降 `HCCL_BUFFSIZE`                           |
| **双流 multi-stream**                  | 与 CPP 正交；首版建议关，减少噪声                                                                             |
| **dynamic chunking**                 | 需 `pp_size>1`；**第二步**再开，先固定 `--chunked-prefill-size`                                            |


---

## 3. 四机 PD + 裁层拓扑（launch-draft）

### 3.1 为什么必须 PD + 裁层

```
满血 colocated: 4 机 × 16 die 扛 ~1.5TB 权重 + KV  → 已近满
PD 2P+2D:       每侧只剩 2 机 → 同权重复载两份角色 → 必须裁层或加机器
```

### 3.2 首版拓扑（定死）

**裁 24 层 + PD 2P+2D + Prefill `pp=2`；Decode `pp=1`；关 DSPARK。**  
（显存仍紧时先 **keep=12 + 全 pp=1** 只通 PD，再升到 24+pp。）


| 角色      | 机器                         | 关键参数                                                                            |
| ------- | -------------------------- | ------------------------------------------------------------------------------- |
| Prefill | 209, 212（示例 node-rank 0,1） | `--disaggregation-mode prefill --nnodes 2 --tp-size 16 --pp-size 2 --dp-size 1` |
| Decode  | 216, 217（node-rank 0,1）    | `--disaggregation-mode decode --nnodes 2 --tp-size 32 --pp-size 1`              |


说明：

- Prefill：`tp16×pp2` 用满 2 机×16 die=32 die；`SGLANG_PP_LAYER_PARTITION=12,12`
- Decode：先 **pp=1**，CPP 收益主要在 Prefill TTFT
- 端口避开同事：`HTTP 30001`，`dist-init` Prefill `5001` / Decode `5002`，bootstrap 另开
- `chunked-prefill-size` 先固定 `4096` 或 `8192`（仅 Prefill）

若 2 机 Prefill 仍 OOM：改 **1P+1D** + keep=12 + `pp=1` 只验证 PD；再 keep=24 + Prefill `pp=2`。

### 3.3 脚本

远程：

- `prepare_k3_cut_override.py` → 生成 override JSON
- `cpp_pd.sh` → 按 `ROLE=prefill|decode` / `NODE_RANK` 拉起
- `smoke_pp_cut.sh` → 单角色或单机 pp 冒烟

---

## 4. 冒烟（smoke-optional）

目标：调度 / P2P / PD bootstrap，**不看** NotifyDispatch、双流、满血 TTFT。

1. 生成 keep=12 override
2. 单机（可选）：`--nnodes 1 --tp-size 8 --pp-size 2` + override，短 prompt generate
3. 两机 PD：1P+1D，`pp=1`，确认 KV 传输
4. 再上 Prefill `pp=2`

小模型替代：SGLang `test/registered/pp/test_pp_single_node_extra.py`（不依赖 K3 权重）。

---

## 5. 验收（full-bringup）

在裁层模型上（先 12，再 24）：

1. **基线**：同裁层、同 PD 拓扑、`pp=1`、固定 chunk，测 TTFT（8K/32K）
2. **CPP**：Prefill `pp=2` + 同 chunk，比基线 TTFT
3. Log/trace：多 chunk 在多 PP stage 重叠（不是只有单 stage chunked prefill）
4. 再开 `--enable-dynamic-chunking`，`SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=0.65~0.8`，初始 chunk ≈ 固定最优的 2–3×
5. 满血回到「加机器或接受极紧显存」之前，**不要**把裁层数字当成生产 TTFT

---

## 6. 一页参数速查

```text
MODEL=/home/weights/Kimi-K3-w4a8-int-moe
KEEP_LAYERS=24
OVERRIDE=.../k3_cut_l24.json   # prepare_k3_cut_override.py 生成

# Prefill (2 nodes)
--disaggregation-mode prefill
--nnodes 2 --tp-size 16 --pp-size 2
--chunked-prefill-size 4096
--json-model-override-args "$(cat $OVERRIDE)"
export SGLANG_PP_LAYER_PARTITION=12,12
# 关：DSPARK / SGLANG_NPU_USE_MULTI_STREAM=0 / enable_pdmux
# 先不开：--enable-dynamic-chunking

# Decode (2 nodes)
--disaggregation-mode decode
--nnodes 2 --tp-size 32 --pp-size 1
--json-model-override-args "$(cat $OVERRIDE)"
```

