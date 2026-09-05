# Prefill 首轮 dispatch 崩溃 `507015` / MPU address access is invalid

- **日期**：2026-08-10 排查
- **环境**：Ascend 910 **A3**；Kimi-K3 W4A8；PD 分离 **prefill 侧**；
  `deep_ep` wheel `1.0.0+48942e37.cann.9.0.0.b250`（构建 commit `48942e37`，2026-07-27）；
  DeepEP OPP `custom_opp_compiler_version=9.0.0`；容器 `sglang-yjy`
- **结论**：**不是「A3 不支持 896 专家」**。A3 显式校验上限是 **1024**，896 合法。
  真正崩点是 `notify_dispatch.h` 的**多轮路径 UB 分配**在 `numExperts=896` 下越界。
- **状态**：单轮路径**已修（未提交）**；`realRound≥2` 的 tiling 校验失败**未决**。

> 相关：[`../deep-ep/SKILL.md`](../deep-ep/SKILL.md)（hang / 507014 取证链）、
> [`../../knowledge/npu/deepep-ascend-npu.md`](../../knowledge/npu/deepep-ascend-npu.md)（`--moe-a2a-backend` 概念与专家数上限）。

---

## 症状

prefill 首轮 dispatch 就崩，签名固定：

```
RuntimeError: ... error code is 507015
fftsplus aivector error ... errorStr: The MPU address access is invalid
```

栈：

```
buffer.dispatch
  → normal_strategy.dispatch:116
    → _intranode_dispatch
      → runtime.intranode_dispatch
```

---

## 排查过程

### 第一步：排除「专家数上限 512」这个流行说法

去 tiling 代码里查**真实的**校验上限：

| 位置 | 常量 | 值 |
|---|---|---|
| `csrc/deepep/ops/op_host/cam_moe_dispatch_normal_tiling.cc` | `MOE_EXPERT_MAX_NUM` | **1024** |
| `csrc/deepep/ops/op_host/dispatch_layout_tiling.cc` | `MAX_MOE_EXPERTS_NUM` | **1024** |

→ 896 专家在显式校验下**全部合法**。
**「512」是 `ops2/`（A2 / 910B）路径的上限，不是 A3 的。** 照抄这个结论会找错方向。

### 第二步：二分定位到专家数，与 topk 无关

| 配置 | 结果 |
|---|---|
| 384 专家 | 通过 |
| 896 专家，topk=8 | **必崩** |
| 896 专家，topk=16 | **必崩** |

→ 判别变量是**专家数**，不是 topk。

### 第三步：排除 INT8/BF16 这个干扰项

`DEEP_NORMAL_MODE_USE_INT8_QUANT=1` **是干扰项**：
安装版 `deep_ep` 的 normal dispatch **根本不读它**（`quant_mode=None` 直接走 BF16），
节点实际跑的就是 BF16。**INT8/BF16 不是判别变量**，别在这上面做 A/B。

### 第四步：定位到 `notify_dispatch.h` 的多轮路径

多轮时 `batchRounds` 按专家数分流，但原实现把 `numLocalExperts >= 128` 的分支**放在最前**：

- ep ≤ 4 且 `numExperts > 512` 时**恒命中**该分支 → `batchRounds=16`
- UB 按 `16 × 896` 分配，`sendDataAlignLen ≈ 172KB`
- 加上其他 buffer → 超过 **192KB 上限**
- → 内核写越界 → AIVector 非法访问（`507015`）

---

## 根因

`notify_dispatch.h` 多轮路径的 **UB 分配**在 `numExperts=896` 下越界，
不是专家数超限、不是 topk、不是量化类型。

---

## 修复 & 验证

文件：`csrc/deepep/ops/op_kernel/notify_dispatch.h`

1. 新增 `round == 1 → batchRounds = 1` 特判（单轮不需要多轮批处理，UB 最小）
2. 调整分支顺序：把 `numExperts > 512` 的「降半批」分支**提到最前**，消除死代码

**验证**：896 专家 + `round=64` + 128 token（`realRound=1`）通过；修复前同配置必崩 `507015`。

### 本地复现

```bash
docker exec sglang-yjy bash -lc '
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export DEEPEP_NORMAL_LONG_SEQ_ROUND=64
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512
export HCCL_BUFFSIZE=800
cd /home/y00970600/personal/sgl-kernel-npu/tests/python/deepep
python3 test_intranode.py --num-processes 4 --num-tokens 128 \
  --hidden 7168 --num-topk 16 --num-experts 896 --quant-type bf16'
```

- 修复前：`507015` + MPU invalid（与节点一致）
- 384 专家：PASSED
- 带 notify 修复后：128 token（`realRound=1`）通过
- `realRound≥2`（`--num-tokens 600/1024`）→ tiling `ret=-1`（未决问题）

> ⚠️ `507015` 之后该 NPU **进程上下文不可复用，需重启进程**。
> 复现用的空闲卡（如 4-7）不会影响其他占用。

---

## 未决问题

`realRound ≥ 2`（单次 dispatch token > `per_round_tokens=512`）时：

```
CamMoeDispatchNormal do tiling failed, ret=-1
```

host 侧 `OP_TILING_CHECK` 拒绝执行。疑似**窗口校验**问题
（`cam_moe_dispatch_normal_tiling.cc` 的 `actualSize > maxWindowSize`；
round>1 时 combine 走 double buffer 使需求增大）。**尚未定性**。

与 decode 侧 [`HCCL_BUFFSIZE too SMALL`](./2026-08-deepep-decode-hccl-buffsize-too-small.md)
是**同一条链路**（都是窗口/buffer 尺寸估算）。

---

## 一句话给下一个 agent

**单轮（`round=1`）+ 小 chunk 端到端可用。** 多轮路径有两处问题：
① notify UB 越界（已修，未提交）；② `realRound≥2` tiling 校验失败（未修）。
