# sglang-knowledge

SGLang / Ascend NPU 上的知识、操作手册与故障排查记录。

## 三类内容，先分清再写

写任何东西之前先问：**这是「为什么」还是「怎么做」还是「怎么修」**。三类分开存放，
混在一起是这类文档腐烂最快的方式。

| 目录 | 回答什么 | 判据 | 例子 |
|---|---|---|---|
| **[`knowledge/`](./knowledge/README.md)** | **为什么** | 半年后重读仍成立；换机型/镜像/任务也还有用 | PD 分离的原理、`--moe-a2a-backend` 的语义 |
| **[`runbooks/`](./runbooks/README.md)** | **怎么做** | 正常路径下的标准动作，不需要先遇到 bug | 起 K3 PD 服务、选空闲 NPU、看护 nightly |
| **[`debugging/`](./debugging/README.md)** | **怎么修** | 由一次真实失败驱动；读完该能修好正在报错的东西 | `507014`、HCCL 端口耗尽、精度掉点定位 |

**边界约定**（两边都遵守，避免重复维护）：

- runbook 只写「照做」，遇到坑就一行链过去，不解释原理。
- debug 只写「怎么定位 + 根因 + 修复」，命令细节链回 runbook。
- knowledge 只留原理性解释（「为什么必须这样」），具体报错串和修复命令一律挪去 debugging。

---

## 快速入口

**挂了 / 报错了** → 先搜 [`debugging/README.md`](./debugging/README.md) 的「报错串 → 去哪」表。

| 我在做什么 | 去哪 |
|---|---|
| 疑惑「overlap 怎么被关了」 | [`knowledge/inference/overlap-schedule.md`](./knowledge/inference/overlap-schedule.md) |
| 起 K3 PD / Chunked PP 服务 | [`runbooks/k3-pd-pp/`](./runbooks/k3-pd-pp/SKILL.md) |
| 容器内环境 / PYTHONPATH 报错 | [`runbooks/npu-docker-env/`](./runbooks/npu-docker-env/SKILL.md) |
| 选哪张 NPU / 端口 | [`runbooks/npu-device-selection/`](./runbooks/npu-device-selection/SKILL.md) |
| 早上看护 nightly | [`runbooks/nightly-watch/`](./runbooks/nightly-watch/SKILL.md) |
| 复现 nightly 失败 case | [`runbooks/nightly-local-repro/`](./runbooks/nightly-local-repro/SKILL.md) |
| git / PR 该注意什么 | [`runbooks/git-workflow.md`](./runbooks/git-workflow.md) |
| HCCL / 多机通信报错 | [`debugging/hccl/`](./debugging/hccl/SKILL.md) |
| DeepEP hang / `507014` | [`debugging/deep-ep/`](./debugging/deep-ep/SKILL.md) |
| 精度掉了 / 要定位分叉层 | [`debugging/precision/`](./debugging/precision/SKILL.md) |
| PP1 和 PP4 结果不一致 | [`debugging/pp-divergence/`](./debugging/pp-divergence/SKILL.md) |
| 某次事故的完整复盘 | [`debugging/cases/`](./debugging/cases/) |

## 关联脚本

| 脚本 | 配套 |
|---|---|
| [`scripts/kimi-k3-cpp/`](./scripts/kimi-k3-cpp/) | [`runbooks/k3-pd-pp/`](./runbooks/k3-pd-pp/SKILL.md) |
| `runbooks/nightly-watch/scripts/check_nightly.py` | [`runbooks/nightly-watch/`](./runbooks/nightly-watch/SKILL.md) |
| `runbooks/nightly-local-repro/scripts/run_repeat.sh` | [`runbooks/nightly-local-repro/`](./runbooks/nightly-local-repro/SKILL.md) |

---

## 维护约定（下一个 agent / 下一个自己必读）

1. **一个坑只写一处**。同类根因出现 ≥2 次且有完整时间线，才从踩坑账本抽成独立 case。
2. **报错串要原文**。别人靠 grep 报错找到这里，写「通信错误」这种转述等于没写。
3. **写清「不是什么」**。「看起来像 bug 但其实不是」的判断最省时间
   （例：`max_running_requests was reduced` 不是错误，但会毁掉性能数字）。
4. **改了脚本就改对应 runbook**，反之亦然。两者不一致是最常见的腐烂方式。
5. **脱敏**：公开内容用 `$HOME` / `<PLACEHOLDER>`，不写真实用户名、内网 IP、容器真名。
6. 真实镜像、模型路径、proxy 端口以**本机 `docker.sh` / 用户当场指定**为准，不写死跨机路径。

---

## 去重约定（重要）

**一个坑只写一处。** 之前 `nightly-local-repro` 的踩坑账本变成了一个什么都往里塞的桶，
与 `npu-docker-env` / `debugging/hccl` / `debugging/deep-ep` / `npu-device-selection` /
`git-workflow` 大量重复。已清理：

- 该账本**顶部加了归口表**，明确 5 类坑各自该去哪写，本表只收「复现流程 / 脚本 / 测试框架独有」的坑
- 迁出 9 条重复条目（tbe / `EI0020` / `max_running_requests` / 脏树 stash / HBM 残留 / deepep hang …）
- §9 的 plog / py-spy / coredump 取证链整节迁入 [`debugging/deep-ep/`](./debugging/deep-ep/SKILL.md)，
  原处只留一张指路表

**新增坑时先问：这是「流程独有」还是「通用」？** 通用的一律写到归口文档，不要就近塞。

---

## 待办

- [ ] **删掉遗留的 `_check_links.py`**（一次性校验脚本，命令审批超时未能自动清理）
- [ ] 根目录 `image.png` / `image-1.png` 是 `knowledge/inference/sampling-params.md` 的插图，
      目前用 `../../image.png` 引用；如需自包含可移入 `knowledge/inference/`
- [ ] 各 `SKILL.md` 的 frontmatter `name` 仍为旧目录名（刻意保留，避免破坏按名引用），
      若确认无外部引用可统一为目录名
- [ ] `~/Desktop/work/` 根目录还有若干散落文档（`适配.md`、`ppdebug.md`、
      `tool_call问题定位.md`、`招行问题定位.md`、`kernel_fixes_without_them.md`、
      `calculate.md`）**未收编**，看着像同类知识，可按三分原则再捞一遍

## 重构说明（2026-09-02）

仓库原本是「13 篇文档平铺在根目录」，知识 / 操作 / 排查三类混装 ——
例如 `DeepEP-Ascend-NPU 适配.md` 里既有概念也有三个故障复盘，
`kimi-k3-chunked-pp.md` 里既有层结构也有切点决策。

已按上面三类重排，`git mv` 保留历史。混装的两篇已拆：

| 原文件 | 拆成 |
|---|---|
| `DeepEP-Ascend-NPU 适配.md` | `knowledge/npu/deepep-ascend-npu.md`（概念 + 三条机制性约束）<br>+ `debugging/cases/` 下 3 个 case |
| `Hunyuan3-Preview 适配.md` | `debugging/cases/2026-hunyuan3-topk-scaling-factor.md` |
| `kimi-k3-chunked-pp.md` | `knowledge/models/kimi-k3-layer-structure.md`（层结构 + 代码就绪度）<br>+ `runbooks/k3-pd-pp/pp-partition-and-cut.md`（切点 / 裁层 / 拓扑） |

另外从 `~/Desktop/work/` 根目录收编了一篇散落的知识（原本不在任何仓库里）：
`介绍 NPU Graph Runner 文件.md` → `knowledge/npu/npu-graph-runner.md`，
并据此补出 case `debugging/cases/2026-08-vit-npu-graph-cu-seqlens-tnd-mismatch.md`。
