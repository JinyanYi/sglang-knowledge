# Qwen3.8 Day-0 适配 PR 分析（sgl-project/sglang#34585）

## 日期
2026-08-13（PR 提交于 2026-08-12）

## 背景
- PR 链接：<https://github.com/sgl-project/sglang/pull/34585>（"support qwen 3.8"，**尚未合入 main**，Open 状态，等待 review/CI）
- 内容：Qwen3.8-2.4T-A95B 的 Day-0 支持（Qwen 迄今最大的开源模型）
- 规模：1 个 squash commit（`d0551b6`，17 位 co-author），**113 个文件，+18,129/-363**，主体是 kernel 代码
- 合作方：SGLang/Miles 团队（RadixArk）、Qwen、阿里百炼、NVIDIA、AMD
- 本地分析仓库：`c:\Users\Administrator\Desktop\work\sglang`（`origin/qwen38` 分支，merge-base `9aadacfc`）
- 配套镜像（Day-0 发布）：
  - CUDA 13 / GB300：`lmsysorg/sglang:qwen38`；CUDA 12：`lmsysorg/sglang:qwen38-cu12`
  - ROCm：MI355 `v0.5.17-rocm720-mi35x-20260812`，MI300 `v0.5.17-rocm700-mi30x-20260812`
  - NVFP4 checkpoint：`RadixArk/Qwen3.8-2.4T-A95B-NVFP4`；DSpark 投机解码 checkpoint：`RadixArk/Qwen3.8-2.4T-A95B-DSpark`
- 官方博客（性能数据出处）：<https://www.lmsys.org/blog/2026-08-12-qwen3-8-day0-support>

## 模型本身：创新点

### 架构
- **2.4T 总参数 / 95B 每 token 激活**，92 层
- **混合注意力（延续 Qwen3.5/3.6 路线）**：69 层 GDN 线性注意力 + 23 层 GQA 全注意力，3:1 交错
  - GDN（Gated Delta Network）：SSM + 因果卷积（CausalConv1d），定长循环状态替代不断增长的 KV cache → 每层显存 O(1)、计算 O(N)
- **稀疏 MoE**：每层 512 个 routed expert + 1 个 shared expert，top-10 路由，hidden 8192
- **三种服务状态**并存：全注意力 KV cache、GDN 循环状态（ssm state）、GDN 卷积窗口（conv state）

### 服务侧创新点（也是 SGLang 适配的核心难点）
1. **ReplaySSM**（GDN 状态回放）：MTP 验证时 GDN 层原地更新循环状态，但只有被接受的 prefix 状态需要提交。验证期间不逐位置快照状态，而是记录"循环输入"（raw-input replay），采样器确定 accept length 后用一个 fold kernel 从 checkpoint 回放被接受的 prefix。与 prefix cache、overlap scheduling、PD disagg 可组合。
2. **PD 分离下三种状态全部跨 worker 迁移**：通过 typed state registry 传输，卷积窗口的 q/k/v 子块按 TP 独立分片；MTP 草稿 KV + hidden states + top-k 元数据同包传输；prefill/decode 布局不同时用 staging buffer 合并为每 chunk 一次 bulk RDMA。
3. **Prefill 端 Chunked Pipeline-Parallel（纯 PP，无 MoE dispatch/combine）+ Decode 端 Wide-EP**：PD 分离使两侧可各用各的并行布局（见下方 staging buffer）。实测 8K prefill：FP8/16卡 5,231 vs 3,421 tok/s/GPU（1.53×）；NVFP4/8卡 8,363 vs 5,151（1.62×）。
4. **PP prefill 与 MTP 投机解码不再互斥**：embed 在首 stage、lm_head 在末 stage，draft head 放末 stage 并自持缺失的另一半；draft KV 跨 PD 边界与 target KV 一起 staged。
5. **Staging Buffer 解耦 prefill/decode 布局**：prefill 写 chunk 进 staging buffer 并发布 per-peer watermark，decode 按 chunk 抓取并改写成自己的布局。prefill:decode 配比、PP 深度、decode EP 宽度可独立调。
6. **NVFP4 原生 checkpoint**：RadixArk 量化发布，支持 Miles 共置 LoRA 训练（BF16 Megatron trainer + NVFP4 SGLang rollout 同机 64 张 GB300）。

### 性能（8K/1K，GB300）
- NVFP4 PD disagg 峰值 **5,126 tok/s/GPU @ 36 tok/s/user**（2×PP6 prefill → DP2-attn/TP4/EP8 decode）
- 低延迟端：NVFP4 TP16 单请求 **334 tok/s/user**；FP8 aggregate CC1 362 tok/s/user
- 投机解码（TP8/B300，NVFP4）：MTP **346 tok/s**（accept 3.3）、DSpark **378 tok/s**（accept 4），含 bonus token

## SGLang 做了哪些适配（代码改动逐模块）

### 1. MoE：deferred finalize + FlashInfer MNNVL CuTe DSL AllReduce 融合（PR 最大亮点，E2E +10%+）
- [`qwen35_flashinfer_fusion.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/layers/moe/qwen35_flashinfer_fusion.py)：`Qwen35FlashInferFusionService`（进程内 workspace，`moe_finalize_all_reduce_rms_norm` / `all_reduce_residual_rms_norm`）、`Qwen35MoeFinalizeHandoff`（未 finalize 的 routed 输出 + 分路 gated shared 输出）、`Qwen35FlashInferLayerCommunicator`
- 机制：MoE MLP 返回 handoff（GEMM2 不落 finalize），由下一层 `prepare_attn` 或最后一层的 final norm 用融合 kernel 一次性完成 **MoE finalize + AllReduce + RMSNorm**（含 PDL chaining 与 persistent execution）。hidden 8192 × top-10 的 finalize 输入在 8K prefill 时高达 1.25 GiB，原本占 prefill 10%
- 前置条件：PP=1、每层 block-FP8 MoE 权重 + FlashInfer TRTLLM deferred-finalize、bypassed TopK；开关 `SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION`
- [`flashinfer_trtllm.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py)：支持 `trtllm_fp8_block_scale_moe(do_finalize=False)`，并做 ABI 适配（某些版本 buffer 按 routing_logits dtype 分配但写入 BF16）
- [`flashinfer_fallback/comm/mnnvl_cutedsl/`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/layers/flashinfer_fallback/comm/mnnvl_cutedsl)：**约 5,700 行** —— SGLang 自带的 FlashInfer API 兼容 fallback 实现（LL/BT/HT 三档 device kernels、protocol、config、presets、`mnnvl_cutedsl_ar.py`）。`flashinfer_provider.py` 先探测上游 flashinfer 是否带该 API，没有就用自己的拷贝
- [`elementwise.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/kernels/ops/elementwise/elementwise.py)：新增 `fused_gate_sigmoid_mul`（只算 gated shared 输出、不 add，DO_ADD=False）

### 2. DeepEP v2 通信（`--moe-a2a-backend deepep_v2`）
- [`token_dispatcher/deepep_v2.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py)（+845 行）：新 dispatcher，基于 `deep_ep.ElasticBuffer`；`direct`（单机 NVLink）/ `hybrid`（跨机）拓扑；输出 dtype auto/bf16/fp8
- 环境变量：`SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK`（per-rank 通信 buffer 容量，非模型语义上限）、`SGLANG_DEEPEP_V2_NUM_SMS`
- CUDA graph 策略：**只有 deep_gemm runner + fp8 masked decode 路径可捕获**（静态形状无 host readback）；prefill/extend 的 contig 路径需 host readback，prefill graph 一律关闭
- [`ep_moe_kernels.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/kernels/ops/moe/ep_moe_kernels.py)：新增 `ep_scatter_from_psum`、`_fwd_kernel_ep_scatter_psum_init`、`_fwd_kernel_ep_expand_m_indices_init` 等（DeepEP v2 的 psum 展开布局）；`ep_scatter` 增加 `expert_start` 参数（FlashInfer A2A → DeepGEMM 时做 global→local expert 重映射）
- [`moe_runner/deep_gemm.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py)：新增 `pre_permute_flashinfer_to_deep_gemm` / `post_permute_deep_gemm_to_flashinfer`（FlashInfer 单向 A2A BF16 → DeepGEMM 无重复 dispatch）；contig（eager）/ masked（decode graph）双路径
- Docker 里 `apply_deepep_v2_patch.sh`：从 commit `01dc3aa` 源码编译 DeepEP v2，替换基础镜像的旧版 DeepEP；**调大跨节点超时宏**（CPU 100s→1000s、cycles 2e11→2e12，GB300 多节点 init 用）；CUDA 13 cccl 头路径修复；NCCL 固定到 2.30.7（DeepEP v2 NCCL backend 需要 GIN API，`ncclGinRequest_t` 等，2.30 才引入）

### 3. GDN 线性注意力 kernel
- [`triton_gdn_fused_proj.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py)（+315 行）：融合 decode 路径 —— 读 interleaved `qkvz/ba` 投影输出，一次 kernel 完成 split/reshape/cat + indexed causal conv1d update（`fused_qkvzba_causal_conv1d_update_contiguous`），E2E decode +2~3%
- [`gdn_backend.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/layers/attention/linear/gdn_backend.py)：融合路径 + 真实 tensor 的 oracle 校验（诊断开关 `SGLANG_GDN_DECODE_FUSION_VERIFY_REAL_TENSORS`，逐位比对、失败即 abort）；把 gate `z` 返回给模型
- [`gdn_flashinfer.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py)：**FlashInfer GDN kernel 要求 32 字节对齐指针**——新增 aligned buffer 修复（只读入参可用 scratch 拷贝修复；可写 buffer 对齐不了则回退 Triton，因为 FlashInfer 会写穿指针）；参数级缓存；ReplaySSM ring buffer 记录路径

### 4. ReplaySSM 校验器
- [`replay_state_indices_validator.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/layers/mamba/replay_state_indices_validator.py)：新增 validator（含 `test_replayssm_ring_accounting` 单测）；记录路径已并入 FlashInfer CuTe DSL GDN MTP kernel（BF16 state），verify 结果逐位不变、无吞吐回退

### 5. PD 分离 + PP 组合
- [`disaggregation/common/staging_handler.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/disaggregation/common/staging_handler.py)：`num_writers` 计入 PP（TP 因子 × PP 因子）；`STAGING_REQ` 报文携带 `pp_rank`，bootstrap 连接按 pp_rank 过滤（`conn.py` 里回填 `bootstrap_info["pp_rank"]`）；`slot_layer_ids` 元数据 —— staging slot 顺序是 [所有 K 层, 所有 V 层]，一旦追加 draft KV buffer 就与 `kv_data_ptrs` 顺序不一致
- [`disaggregation/utils.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/disaggregation/utils.py)：`build_kv_layer_ids` / `build_staging_slot_metadata` 辅助函数
- [`prefill.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/disaggregation/prefill.py)：staging buffer + pp_size>1 仅 Mooncake 支持；`_transfer_start_layer` 处理 `HybridLinearKVPool`（`start_layer` 是含线性层的全局索引，要换算成"全注意力层相对偏移"，因为 decode 侧的指针表是全注意力密集的）；MTP draft hidden states 批量拷到 CPU 再分配（避免逐请求 clone）
- [`disaggregation/mooncake/conn.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/disaggregation/mooncake/conn.py)：PP 下按 staging slot 配对（`build_transfer_entry_pairs`），不再按 kv_data_ptrs 条目

### 6. PP + 投机解码（MTP）组合
- server_args：PP 不再禁止 spec decoding —— 允许 `disaggregation_mode == "prefill"`（prefill 引擎把投机当单次 extend 步骤：target forward + 一次 draft extend，无 accept length 循环；decode 侧仍不支持）
- [`parallel_state.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/distributed/parallel_state.py)：新增 `get_self_pp_group()`（每 rank 单成员 PP group）+ `patch_pipeline_parallel_group` 上下文 —— draft 模型只有一层、永远不跨 PP stage，不能读 target 的 PP 拓扑
- [`qwen3_5_mtp.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/models/qwen3_5_mtp.py)：`set_embed_and_head` 支持 None（PP 下 embed/lm_head 只到其一）；captured prefill graph 的静态 token slot 高度与真实 chunk 高度不一致时的 padding 修复
- [`qwen3_5_text.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/models/qwen3_5_text.py)：**权重加载改成惰性 streaming generator**（`load_weights` 逐 tensor 消费，避免 Oakhaven checkpoint 全量物化导致每 Grace 节点 4 进程的 host RSS 峰值超过 Slurm 配额）；`get_embed_and_head` 对 `PPMissingLayer` 返回 None

### 7. BF16 GEMM：FlashInfer PR #4266 Split-K 低延迟路径
- [`unquant.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/srt/layers/quantization/unquant.py)：新增 `--bf16-gemm-backend flashinfer_pr4266`；`SGLANG_ENABLE_BF16_SPLITK_GEMM`；**`_FLASHINFER_PR4266_TUNED_TACTICS` allowlist**（GB300 TP16 实测，每个 (M,N,K) → (blockM, n_splitk, k_splitk, cta_sched)，都通过严格正确性 gate 且比原 dispatch 快 ≥1.26×）；`prefer_direct` 时走单 GEMM direct dense 路径（小 GEMM 避免 split-k 的归约 kernel 开销，E2E ~4%）
- [`flashinfer_pr4266_dense_bf16_gemm_sm100_splitk.py`](file:///c:/Users/Administrator/Desktop/work/sglang/python/sglang/kernels/ops/gemm/flashinfer_pr4266_dense_bf16_gemm_sm100_splitk.py)（+1,041 行）：Split-K GEMM 实现；现该 kernel 已并入 flashinfer 本体（废弃了 `SGLANG_FLASHINFER_PR4266_SOURCE`）

### 8. FlashInfer A2A 放宽 + 互操作
- server_args：`flashinfer` A2A 从"必须 dp==tp"放宽为 `dp_size > 1 && tp_size % dp_size == 0`；runner 支持 `flashinfer_trtllm` / `deep_gemm`

### 9. Docker（Day-0 镜像工程）
- [`qwen38_cu13.Dockerfile`](file:///c:/Users/Administrator/Desktop/work/sglang/docker/qwen38/qwen38_cu13.Dockerfile)：GB300 aarch64/CUDA13，基础镜像是 `lmsysorg/sglang:dev`（nightly，torch 2.13.0+cu130）；DeepEP v2 源码构建 + NCCL 2.30.7 staged 到 `/opt/nccl-2.30.7/lib` 供 `LD_PRELOAD`；FlashInfer 从 pinned commit `906181e3`（PR#4358 merge commit，防 branch 被删导致 git 拉不到）+ nightly cubin/jit-cache；`nvidia-cutlass-dsl>=4.7.0`
- [`cuda_pins.sh`](file:///c:/Users/Administrator/Desktop/work/sglang/docker/qwen38/cuda_pins.sh)：`check-torch` / `reconcile` / `verify` / `import-if-gpu` 四个子命令，把"装了错的 libtorch 组合"从启动期 `assert_pkg_version` 提前到构建期报错
- `.dockerignore`：从指向 `.gitignore` 的 symlink 改为显式 allowlist（原来整个仓库进 build context，`COPY . /sgl-workspace/sglang` 会把测试、文档、CI、`.claude` agent 目录等专有内容打进镜像）

### 10. 测试
- 新增约 20 个测试：`test_gdn_flashinfer_alignment`、`test_replayssm_ring_accounting`、`test_deepep_v2_masked_slab`、`test_flashinfer_a2a_wide_ep`、`test_qwen35_flashinfer_fusion`、`test_pp_hybrid_kv_transfer`、`test_staging_draft_kv_slots`、`test_routed_experts_dp_readback` 等

## 关键洞察 / 值得记住的点

1. **"新模型支持"在 SGLang 里 = 新 kernel + 新 serving 特性，而不是新模型文件**。Qwen3.8 复用 `qwen3_5_moe_text` 架构（代码里没有任何 "qwen38" 字符串），改动集中在：MoE 通信/融合、GDN 线性注意力、量化（NVFP4/fp8/ue8m0）、PD/PP 调度。
2. **三种状态（KV + GDN 循环状态 + 卷积窗口）是贯穿所有特性的主线**：prefix cache（Radix cache 的 `FULL` 与 `MAMBA` 两个 component）、投机验证（ReplaySSM）、PD 传输（typed state registry）、staging slot 顺序（K/V 分块）。
3. **GB300 时代的服务形态预演**：native NVFP4、DeepEP v2 ElasticBuffer、PDL chaining 的通信-计算融合、纯 PP prefill + wide-EP decode 的异构并行，全部出现在这一个 PR 里。
4. 与本地知识库已有条目的关联：`kimi-k3-chunked-pp.md` / `kimi-k3-latent-moe-ep.md`（ReplaySSM、chunked PP、staging buffer 从 Kimi K3 延续过来）、`DeepEP-Ascend-NPU 适配.md`（同一批 `--moe-a2a-backend` 语义扩展）。
5. 工程细节值得借鉴：FlashInfer API 用 **inspect.signature 探测兼容性**再决定用上游还是 fallback 拷贝；`sed -i` 无匹配也 exit 0，所以 patch 后用 grep 断言；容器镜像用 `--no-deps` 安装以避免 CUDA-tagged wheel 被 PyPI untagged 版本覆盖。
6. PR 状态：**未合入、无 review 对话**；CI 因缺 `run-ci` label 未跑。代码质量极高（注释详尽、错误路径完备），但合入前大概率会被拆分/压减 —— 大量 kernel 代码本应进 flashinfer/DeepEP 上游。

## 相关文件 / 区域
- 模型层：`python/sglang/srt/models/qwen3_5.py`、`qwen3_5_text.py`、`qwen3_5_mtp.py`、`qwen2_moe.py`
- MoE：`layers/moe/qwen35_flashinfer_fusion.py`、`moe_runner/{deep_gemm,flashinfer_trtllm,triton}.py`、`token_dispatcher/deepep_v2.py`
- 融合 kernel：`flashinfer_fallback/comm/mnnvl_cutedsl/*`、`kernels/ops/gemm/flashinfer_pr4266_dense_bf16_gemm_sm100_splitk.py`
- 注意力：`layers/attention/linear/{gdn_backend.py,kernels/gdn_flashinfer.py}`、`kernels/ops/attention/triton_gdn_fused_proj.py`、`layers/mamba/replay_state_indices_validator.py`
- 调度/分离：`disaggregation/*`、`managers/scheduler_pp_mixin.py`、`distributed/parallel_state.py`
- 构建：`docker/qwen38/{qwen38_cu12,qwen38_cu13}.Dockerfile`、`cuda_pins.sh`、`apply_deepep_v2_patch.sh`
- 配置：`srt/server_args.py`、`srt/environ.py`、`layers/quantization/unquant.py`

## 标签
`sglang` `qwen3.8` `day0` `混合注意力` `GDN` `ReplaySSM` `DeepEP v2` `NVFP4` `FlashInfer` `CuTe DSL` `PD分离` `PP流水并行` `MTP` `staging buffer` `MoE` `GB300` `PR分析`
