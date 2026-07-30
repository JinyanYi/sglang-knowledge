---
name: npu-docker-pythonpath
description: >-
  Fixes Ascend/CANN PYTHONPATH overwrite when running SGLang NPU tests inside
  docker (e.g. a dedicated or user-dev container). Use when docker exec python
  tests fail with ModuleNotFoundError: tbe, AclSetCompileopt / NPU compile
  500001, or setUpClass dies in ~50s after exporting PYTHONPATH to
  community/ascend sglang.
---

# NPU Docker PYTHONPATH（勿覆盖 CANN）

在容器里跑 SGLang NPU 性能/功能测试时，**必须先 source Ascend 环境，再追加** sglang 的 `python` 路径。用 `PYTHONPATH=...:` 覆盖式赋值会丢掉 CANN 的 `tbe` 等路径。

## 正确写法

```bash
docker exec <container> bash -lc '
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH=$HOME/community/sglang/python:${PYTHONPATH}
cd "$HOME"
python3 /path/to/test.py
'
```

要点：

1. 先 `source` `ascend-toolkit/set_env.sh` 和 `nnal/atb/set_env.sh`
2. 再用 `${PYTHONPATH}` **追加** sglang 路径（community / ascend / personal 按实际仓库改；以 `docker.sh` 的 `SGLANG_PYTHON` 为准）
3. 可选：按需 `unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING`

## 错误写法（会踩坑）

```bash
# BAD — 覆盖 set_env.sh 写入的 CANN 路径
export PYTHONPATH=$HOME/community/sglang/python:
```

## 症状

- `ModuleNotFoundError: tbe`
- NPU compile / `AclSetCompileopt` / error `500001`
- 测试在 `setUpClass` / server 启动阶段约几十秒失败（不是 bench 阶段）
- 看起来像 GPU/port 冲突，实际是环境被盖掉

## 排查

1. 失败日志里搜 `tbe` / `AclSetCompileopt` / `500001`
2. 在同一 `bash -lc` 里打印：`echo "$PYTHONPATH"`，确认含 Ascend/CANN 相关目录且含 sglang `python`
3. 改成「source 后追加」再跑；不要为这个错误去改 `HCCL_BUFFSIZE` / port / `base-gpu-id`

## 并行压测注意

多 agent / 多实例同机跑时：

- `port` 与 `--base-gpu-id` 不要撞车
- 各自仍用上面的 PYTHONPATH 配方；与 port/GPU 无关
- 复现用专用容器，不要占用用户交互容器 `<USER_DEV_CONTAINER>`

## Living doc

若再遇到环境/PYTHONPATH/CANN 相关新坑，**更新本文件**，并在
`nightly-npu-local-repro/SKILL.md` 的「踩坑账本」加一条交叉引用。
