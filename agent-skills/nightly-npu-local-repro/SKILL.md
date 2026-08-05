---
name: nightly-npu-local-repro
description: >-
  Reproduce failed SGLang Nightly Test NPU Qwen3.6-35B-A3B jobs on internal
  Ascend NPU hosts. Use after triage finds a failing qwen3_6_35b_a3b job, or
  when the user asks to run the failed nightly case locally via docker.sh /
  a dedicated repro container (not the user's interactive/dev container).
  Living doc: update this skill when you hit new pitfalls.
---

# Nightly NPU 失败 case：本地 docker 复现

配合 `nightly-npu-qwen36-35b`：先定位失败 job，再在 NPU 机器上 docker 复现并回报
output tok/s / TTFT / TPOT / 是否过门槛。

日更任务（本机私有，不进公开仓）：`$HOME/daily task for agent.md`

## Living doc（下一个 agent 必读）

**这是活文档。** 跑复现时如果踩到任何新坑、发现步骤过时、或有更好做法：

1. **立刻更新本 `SKILL.md`**（以及相关的 `npu-docker-pythonpath` / `nightly-npu-qwen36-35b` / 本机 `daily task for agent.md`）
2. 在文末「踩坑账本」追加一条：日期、症状、根因、修复
3. 不要只写在聊天里——下一个 cron agent 只能靠这些文件

> 公开仓里一律用 `$HOME` / `<PLACEHOLDER>`。真实镜像、模型路径、proxy 端口以本机
> `docker.sh`、日更文件或用户当场指定为准。

## 端到端流程（零背景 agent）

```text
1. triage nightly (skill: nightly-npu-qwen36-35b / scripts/check_nightly.py)
2. 若有 qwen3_6_35b_a3b failure → 本 skill 本地复现
3. 回报用户：CI 数 vs 本地数 / PASS|FAIL / 日志路径
4. 若过程有新问题 → 更新 skill（见上）
```

## 0. 网络（本机常需要）

机器常无直连 GitHub。若用户已在本机开好 HTTP(S) 代理转发，先：

```bash
export http_proxy=http://127.0.0.1:<PROXY_PORT>
export https_proxy=http://127.0.0.1:<PROXY_PORT>
export HTTP_PROXY=http://127.0.0.1:<PROXY_PORT>
export HTTPS_PROXY=http://127.0.0.1:<PROXY_PORT>
# 若需从笔记本 SSH 反向转发到 NPU 机，由用户提供具体命令；勿把内网主机名/账号写进公开仓
```

- 无 proxy 时 `git pull` / `curl github.com` 常 **超时**
- 跑 NPU 测试时在容器内 **unset** 这些 proxy，避免干扰

## 1. 仓库角色（为什么要两套）

| 路径 | 角色 |
|------|------|
| `$HOME/community/sglang` | **运行时**（社区仓 latest `main`）。`docker.sh` 的 `SGLANG_PYTHON` 指向这里 |
| `$HOME/ascend/sglang` | **测试用例 + Ascend e2e harness**。分支 **`testcases`**。若无则 `git clone https://github.com/Ascend/sglang` 到 `$HOME/ascend/sglang` |

`PYTHONPATH=$SGLANG_PYTHON` 时，`import sglang.test...` 走的是 **community 下的**
`python/sglang/test`。所以必须把 Ascend 的 `python/sglang/test` **覆盖拷贝**进去
（`docker.sh` 启动时也会做一次）。

> 别的机器可能把 community 放在别的目录；**以该机 `docker.sh` 的 `SGLANG_PYTHON` 为准**，不要写死跨机路径。

## 2. 拉最新

```bash
export http_proxy=http://127.0.0.1:<PROXY_PORT> https_proxy=http://127.0.0.1:<PROXY_PORT>

# community runtime
cd "$HOME/community/sglang"
# 若有本地脏改动：先 stash，再 pull（不要随便丢用户改动）
git stash push -m "agent-temp-before-nightly-repro" -- <dirty-files...>
git pull --ff-only origin main
# stash 默认先留着；不要盲目 stash pop 到新 main 上制造冲突

# ascend testcases
cd "$HOME/ascend/sglang"
git fetch origin
git checkout testcases
# 用户可能有本地改动的 performance case：stash → pull → stash pop
git stash push -m "agent-temp-ascend-before-nightly-repro" -- test/registered/ascend/performance/qwen3_6_35b_a3b/ || true
git pull --ff-only origin testcases
git stash pop || true
```

**坑**：不 stash 时 `git pull --ff-only` 会直接 Abort。

## 3. 同步 test 包到 PYTHONPATH

```bash
SGLANG_PYTHON="${SGLANG_PYTHON:-$HOME/community/sglang/python}"   # 或读 docker.sh
\cp -rf "$HOME/ascend/sglang/python/sglang/test" "${SGLANG_PYTHON}/sglang/"
```

**坑**：交互 shell 里 `cp` 常是 `alias cp='cp -i'`，会对每个文件问 `overwrite?`，脚本会卡死。
**必须用 `\cp -rf`**。

### 优化：`base_url` 不要改错 setUpClass

父类原先会在 `setUpClass` **无条件**执行：

```python
cls.base_url = DEFAULT_URL_FOR_TEST
```

这会盖掉子类设的 port。用户旧笔记里“删掉这行 / 改成固定 port”能跑，但更干净的做法是
**只改同步后的副本**（不要改坏 ascend git 工作区）：

```python
cls.base_url = getattr(cls, "base_url", None) or DEFAULT_URL_FOR_TEST
```

然后在本地 repro 文件里设 `base_url = "http://127.0.0.1:<free-port>"` 即可。

## 4. 起容器（禁止占用用户交互容器）

参考 `$HOME/docker.sh`，**另起专用容器**（例如 `sglang-nightly-repro`），不要占用用户日常开发容器
（记作 `<USER_DEV_CONTAINER>`）。
先 `docker pull` 确认 `main-cann9.0.0-a3` 是最新再跑（本地 tag 可能过旧）。

```bash
SGLANG_PYTHON="${SGLANG_PYTHON:-$HOME/community/sglang/python}"
# 优先用用户/组织提供的镜像；公开等价 tag 示例：
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:main-cann9.0.0-a3}"
CONTAINER=sglang-nightly-repro   # NEVER <USER_DEV_CONTAINER>

docker pull "$IMAGE"

docker run -itd --shm-size=64g --privileged=true --name "$CONTAINER" \
  --net=host -w "$HOME/" \
  -v "$HOME/pip.conf:/root/.config/pip/pip.conf" \
  -v /mnt:/mnt -v /home:/home -v /data:/data \
  -v /var/queue_schedule:/var/queue_schedule \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -e PYTHONPATH=${SGLANG_PYTHON}: \
  --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 \
  --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 \
  --device=/dev/davinci_manager --device=/dev/hisi_hdc \
  --entrypoint=bash "$IMAGE" -c "exec bash"
```

**坑**：部分非交互环境里 `--device=/dev/davinci{0..15}` brace 可能不展开；上面显式列出更稳。

若机器上还没有 `docker.sh`，按上面模板新建一份（改 `SGLANG_PYTHON` / 容器名 / `SGLANG_IMAGE`）。

## 5. 选空闲 NPU + port

```bash
npu-smi info   # 找 "No running processes found in NPU X"
ss -lnt | grep -E ':(9930|9999)\b' || true
```

- 看 **Phy-ID**（不是 NPU card 序号）：TP2 需要两个连续 Phy-ID
- Phy-ID `8,9` → `--base-gpu-id 8`
- Phy-ID `10,11` → `--base-gpu-id 10`
- Phy-ID `6,7` → `--base-gpu-id 6`
- port 自选空闲，例如 `9930` / `9931`（勿撞现有 serve）

## 6. 本地 repro 文件

失败 case 通常在：

`ascend/sglang/test/registered/ascend/performance/qwen3_6_35b_a3b/<case>.py`

推荐复制到 `$HOME/nightly_repro/` 再改：

1. `model = "<MODEL_PATH>"`  
   （`ls` 本机权重目录哪个存在用哪个；CI 默认 ModelScope 路径本地常没有）
2. `other_args` 末尾加 `"--base-gpu-id", <free>`
3. `base_url = "http://127.0.0.1:<free-port>"`（依赖第 3 节的 setUpClass 补丁）
4. 需要 ShareGPT 的 case：  
   `dataset_path = "<DATASET_PATH>"`  
   （`generated-shared-prefix` 可仍带上作兜底）
5. **不要改** `output_token_throughput` / 业务参数（除本地路径/gpu/port），否则对比失真
6. 若要连跑多次做对比：可设 `max_attempts = 1`，避免框架默认 retry=2 把一次变成两次

## 7. 容器内跑（PYTHONPATH 追加，勿覆盖）

见 skill `npu-docker-pythonpath`（`tbe` / 500001 坑）：

```bash
docker exec -d sglang-nightly-repro bash -lc '
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH=$HOME/community/sglang/python:${PYTHONPATH}
cd "$HOME"
python3 -u "$HOME/nightly_repro/<repro>.py" 2>&1 | tee "$HOME/nightly_repro/run.log"
'
```

多次重复可复用：

```bash
CONTAINER=sglang-nightly-repro TEST=/path/to/repro.py RUNS=3 \
  bash personal/sglang-knowledge/agent-skills/nightly-npu-local-repro/scripts/run_repeat.sh
```

### 盯日志时注意

长 case（如 128k prefix）常要 **~1h+**；框架默认 **retry 2 次**，总墙钟可到 **~70min**。
cron 要预留足够时间；后台跑 + 正确轮询。

| 信号 | 含义 |
|------|------|
| `port=` / `base_gpu_id=` | 确认没撞车 |
| `max_running_requests was reduced from X to Y` | **重要**：KV/显存估算导致并发被砍；TTFT 会系统性很高、吞吐偏低；**回报必须写明** |
| `health_generate ... 503` | 启动中，继续等 |
| `The server is fired up and ready to roll!` | 可以开始/已开始 bench |
| `Generating shared-prefix prompts` | GSP 还在造数据，很慢，正常 |
| `Output token throughput` + `AssertionError` / `Ran 1 test` + `OK`/`FAILED` | 真正结束 |

### 轮询结束条件（别误判）

**不要**用裸 `OK` / `200 OK` 当结束标志（会误报）。

用：

```bash
grep -E 'AssertionError:|FAILED \(|^OK$|Ran [0-9]+ test in' "$LOG"
```

或看 unittest 进程是否还在：`pgrep -f <repro_file>`。

## 8. 回报格式

```markdown
Local repro: <case>
- container: sglang-nightly-repro
- base-gpu-id: N, port: P
- image: <digest / Created>
- model: <MODEL_PATH>
- result: PASS/FAIL
- output tok/s: ... (各次都写)
- Mean TTFT / Mean TPOT: ...
- vs CI: <CI actual> / threshold <baseline*0.98>
- caveats: e.g. max_running_requests 103→17
- log: $HOME/nightly_repro/run.log
```

CI 完整 Serving Benchmark 若公开页没有：metrics artifact 无鉴权常 **401**；可搜 job HTML 的 `AssertionError`，或请用户贴 benchmark 块。

## 可复用脚本

| 脚本 | 用途 |
|------|------|
| `nightly-npu-qwen36-35b/scripts/check_nightly.py` | 查 schedule nightly + 筛 35B-A3B job |
| `nightly-npu-local-repro/scripts/run_repeat.sh` | 容器内同一 case 连跑 N 次并抽指标 |

## 踩坑账本（追加在此）

| 坑 | 症状 | 处理 |
|----|------|------|
| 无 proxy | `git pull` / GitHub 连接超时 | `export http(s)_proxy=http://127.0.0.1:<PROXY_PORT>` |
| community/ascend 脏树 | `pull --ff-only` Abort | stash 指定文件后再 pull；community stash 先别盲目 pop |
| `cp -rf` 交互询问 | 脚本卡在 `overwrite?` | 用 `\cp -rf` |
| 占 `<USER_DEV_CONTAINER>` | 用户自己要用 | 容器名用 `sglang-nightly-repro` 等 |
| PYTHONPATH 覆盖 | `ModuleNotFoundError: tbe` | source set_env 后 **追加** PYTHONPATH |
| 父类盖 `base_url` | 子类 port 无效 | synced utils：`getattr(cls,"base_url",None) or DEFAULT_URL_FOR_TEST` |
| 轮询误报 | 匹配到 HTTP `200 OK` 以为测完 | 结束条件用 `Ran N test` / `AssertionError` / `FAILED (` |
| 本地镜像过旧 | 同 tag 但 Created 很旧 | 跑前 `docker pull` |
| `max_running_requests` 被降 | 日志 `103 to 17`；TTFT 极高 | 回报必须注明 |
| 长 case 耗时 | ~1h/次，retry 共 ~70min | 后台跑 + 正确轮询；对比用 `max_attempts=1` |
| CI metrics artifact | 下载 401 | 公开 HTML / 用户粘贴 Serving Benchmark |
| stash 盖掉 `environ.py` | `AttributeError: Envs has no attribute SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK` | 从 `origin/main` 补回缺失 Env；勿用过旧 stash 整文件覆盖 |
| 同机多 NPU job | `Communication_Error_Bind_IP_Port(EI0020)` port 16666 | 本地 repro 设 `HCCL_NPU_SOCKET_PORT_RANGE`（如 `36100-36150`）并换空闲 Phy 对 |
| 本机 `127.0.0.1:1082` proxy 未起来 | `Connection refused`；`check_nightly.py` 失败 | 先直连试 `api.github.com`（有时通）；本机也可能是 `1080`；git 遇 HTTP2 framing 可加 `GIT_HTTP_VERSION=1.1`；容器内仍 **unset** proxy |
| 1080p MM case 首跑冷启动 | 同参数 RUN1 TPOT 极差（如 130ms）、吞吐腰斩；RUN2/3 明显好转 | 至少 `RUNS=3`；分析时勿把首跑当稳态；CI 也可能落在差尾 |
| Full Test vs Nightly 门禁 | Full Test 里 `tpot=50` → `mean_tpot <= 51.0`；吞吐门 `baseline*0.98` | 报失败时写明是 **TPOT 上限** 还是 **吞吐下限**（AssertionError 文案不同） |
| `pgrep -f run_repeat.sh` 自匹配 | wait 循环永不结束 | 用 `pgrep -f 'scripts/run_repeat.sh'` 或看 OUTDIR `finished=` / PID 文件 |
| Phy 看似空闲但 HBM 残留/邻卡忙 | 首跑异常差、后续恢复 | 选 `npu-smi` 明确 “No running processes” 的成对 Phy；避开刚释放的卡再观察一轮 |
| 1024² MM case 与 1080p 同族冷启动 | RUN1 常 TPOT 炸（>51）或吞吐腰斩；RUN2 可能卡在吞吐门（如 1515&lt;1587.6 / 313&lt;352.8）；RUN3 才过 | `RUNS=3`；并行时用**不同 Phy 对 + 不同 port + 不同 HCCL_NPU_SOCKET_PORT_RANGE**；勿把首跑当稳态回归 |
| smoke 用 `inspect.getsource(setUpClass)` | 看到的是 `@retry` 包装的 `safe_setUpClass`，不是真实 `getattr` 行 | 直接 `rg` 同步后的 `test_npu_performance_utils.py`，或 `print(open(...).read())` |
| GitHub API rate limit（proxy IP） | `API rate limit exceeded`；Ascend Full Test 查 job 失败 | 等 reset / 换直连；或 scrape job HTML + annotations；优先复用既有 triage 里的 job URL |
