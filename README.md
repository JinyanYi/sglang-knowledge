# 知识库索引

本文档集记录了 SGLang 代码库开发过程中积累的见解、bug 解决方案和架构理解。

---

## 快速导航

### 推理加速

| 文档 | 描述 |
|------|------|
| [投机推理方法详解](./speculative-decoding.md) | N-gram、EAGLE/EAGLE-3、DFlash 的原理、对比和 SGLang 使用指南 |
| [Multi-Token Prediction (MTP)](./multi-token-prediction.md) | MTP 的训练方式、推理加速原理和 SGLang 实现 |

### 计算优化

| 文档 | 描述 |
|------|------|
| [Prefill vs Decode](./prefill-decode.md) | LLM 推理的两个阶段：计算特性差异、Arithmetic Intensity、PD 分离 |
| [CUDA Graphs](./cuda-graphs.md) | CUDA Graph、Piecewise CUDA Graph 详解，以及 SGLang 实现分析 |
| [显存预算与 Prefill 调度](./memory-budget-and-scheduling.md) | mem-fraction-static、KV 池 profiling、PrefillAdder、Static/Dynamic 分布与实测 case |

### 模型推理

| 文档 | 描述 |
|------|------|
| [模型推理参数设置](./model-temperature-sampling-params.md) | Temperature 参数对模型输出的影响机制，以及 SGLang 中的实现 |

### 架构分析

| 文档 | 描述 |
|------|------|
| [SGLang OpenAI API 服务器流程](./architecture/sglang-openai-api-server-flow.md) | SGLang 如何实现 OpenAI 兼容 API，包括服务器启动流程和请求处理路径 |

---

## 主题分类

### 推理加速
- 投机推理（Speculative Decoding）
- Multi-Token Prediction (MTP)
- N-gram / EAGLE / DFlash 方法对比

### 计算优化
- Prefill vs Decode 阶段分析
- Arithmetic Intensity 与 Roofline 模型
- CUDA Graph 与 Piecewise CUDA Graph
- PD 分离架构
- 显存预算（mem-fraction-static、KV 池、PrefillAdder）

### 模型推理
- Temperature 采样参数
- Sampling Params 配置

### 核心架构
- OpenAI API 服务器实现
- 引擎组件交互（TokenizerManager, Scheduler, DetokenizerManager）
- 进程间通信机制

---

## 标签索引

- `sglang` - SGLang 核心框架
- `架构` - 系统架构设计
- `openai-api` - OpenAI 兼容性
- `服务器` - HTTP/gRPC 服务器
- `推理引擎` - LLM 推理相关
- `推理加速` - Speculative Decoding, MTP, N-gram, EAGLE, DFlash
- `计算优化` - CUDA Graph, PD 分离, Arithmetic Intensity
- `模型推理` - Temperature, Sampling Params
- `temperature` - 温度参数设置
- `sampling-params` - 采样参数配置
- `概率分布` - 模型输出概率分布
