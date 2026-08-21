# Kimi-K3 PD + CPP 验收清单

在裁层模型上执行（PD 通路径 KEEP=12；CPP 对比 KEEP=24），满血数字不作生产结论。

## A. 准备

- [ ] 四机容器 host 网络、网卡/`/etc/hosts`/`HCCL_IF_IP` 正常
- [ ] 生成 override：`python3 prepare_k3_cut_override.py --keep 24 -o overrides/k3_cut_l24.json`
- [ ] **关** DSPARK、multi-stream、`enable_pdmux`
- [ ] 端口避开同事：HTTP 30001；Prefill dist 5001；Decode dist 5002

## B. 冒烟顺序

1. [ ] `MODE=pp_only KEEP=24 TP=8 PP=2` 单机 ready + 短 generate（`PARTITION=12,12`）
2. [ ] `1P+1D` PD（各 16 die）`KEEP=12`，`pp=1`，端到端一条请求
3. [ ] 升 `KEEP=24`，Prefill `pp=2`（`PARTITION=12,12`），Decode `pp=1`

## C. 四机 PD 拓扑（cpp_pd.sh）

- [ ] Prefill：209/212，`ROLE=prefill NODE_RANK=0/1 KEEP_LAYERS=24`
- [ ] Decode：216/217，`ROLE=decode NODE_RANK=0/1 KEEP_LAYERS=24`
- [ ] ready 后长输入 8K → 再 32K

## D. 对比（同 KEEP=24、同 PD）

| 实验 | Prefill | 记录 |
|------|---------|------|
| 基线 | `pp=1`，固定 chunk 4096 | TTFT |
| CPP | `pp=2`，同 chunk，`PARTITION=12,12` | TTFT |
| Dynamic | `--enable-dynamic-chunking`，init chunk≈2–3×，smooth 0.65–0.8 | TTFT |

- [ ] Log/trace 可见多 chunk 跨 PP stage 重叠
- [ ] 若 stage 不均，调 `SGLANG_PP_LAYER_PARTITION`（满血时优先 `48,45`）

## E. 加压

- [ ] 仍 OOM → 降 `HCCL_BUFFSIZE` / `mem-fraction-static` / running requests；或退回 1P+1D + KEEP=12
- [ ] 满血：需更多机器或接受非 PD / 极紧显存；**不要**把裁层 TTFT 直接当满血结论
