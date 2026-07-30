# 多模态请求端到端流程（Qwen3.6-35B-A3B）

本文档以 **Qwen3.6-35B-A3B**（模型类 `Qwen3VLMoeForConditionalGeneration`）为例，梳理单条 curl 请求传入一张图片时，从 HTTP 请求到响应返回的完整端到端链路。

---

## 总体流程概览

```
HTTP 请求（curl 传入图片）
  │
  ├─ TokenizerManager (tokenizer_manager.py)
  │   ├─ _validate_mm_limits()            # 校验图片数量上限
  │   └─ mm_processor.process_mm_data_async()  # 调用 QwenVLImageProcessor
  │       ├─ 加载图片（PIL Image）
  │       ├─ smart_resize()               # 智能缩放（factor=28）
  │       ├─ HuggingFace processor.__call__()
  │       │   ├─ rescale: [0,255] → [0,1]
  │       │   └─ normalize: (x - mean) / std
  │       └─ 返回 pixel_values（展平后的 2D tensor）
  │
  ├─ MultimodalInputs.from_processor_output()
  │   ├─ 构造 MultimodalDataItem（每个 item = 一张图）
  │   └─ set_pad_value() → hash_feature() # SHA256 哈希
  │
  ├─ Scheduler.handle_generate_request()
  │   ├─ _get_multimodal_inputs()         # 构造 MultimodalInputs
  │   └─ pad_input_ids / extend_image_inputs  # 对齐 token 序列
  │
  ├─ Model Forward
  │   ├─ general_mm_embed_routine()
  │   │   ├─ embed_mm_inputs()
  │   │   │   ├─ 查缓存（MultiModalStaticCache，LRU）
  │   │   │   ├─ 缓存未命中 → ViT 编码
  │   │   │   │   ├─ Patch Embedding (3D Conv)
  │   │   │   │   ├─ Position Embedding (2D RoPE)
  │   │   │   │   ├─ Vision Transformer Blocks (self-attention + MLP)
  │   │   │   │   └─ Patch Merger (spatial merge + projection)
  │   │   │   └─ 写缓存
  │   │   ├─ 注入 vision embedding 到 text embedding
  │   │   └─ language_model.forward()     # LLM 推理
  │   └─ 逐 token decode，生成回复
  │
  └─ 请求完成后
      ├─ Scheduler._maybe_clear_mm_inputs()
      └─ release_features() → req.multimodal_inputs = None
```

---

## 阶段一：图片预处理

### 1.1 请求校验

文件：[`tokenizer_manager.py`](https://github.com/sgl-project/sglang/blob/874fc07d9bbbb714a71e5d4cbe5e005a885168ef/python/sglang/srt/managers/tokenizer_manager.py)

HTTP 请求到达后，`GenerateReqInput` 中携带 `image_data` 字段。首先进行数量校验：

```python
# tokenizer_manager.py#L1050-L1061
def _validate_mm_limits(
    self, obj: Union[GenerateReqInput, EmbeddingReqInput]
) -> None:
    if not self.server_args.limit_mm_data_per_request:
        return
    for modality, limit in self.server_args.limit_mm_data_per_request.items():
        data = getattr(obj, f"{modality}_data", None)
        if data:
            count = len(data) if isinstance(data, list) else 1
            if count > limit:
                raise ValueError(
                    f"{modality.capitalize()} count {count} exceeds limit {limit} per request."
                )
```

通过 `--limit-mm-data-per-request` 启动参数可限制单请求的图片数量。超限直接抛出 `ValueError`，返回 400 错误。

### 1.2 图片加载与 smart_resize

校验通过后，调用 `QwenVLImageProcessor.process_mm_data_async()`：

```python
# tokenizer_manager.py#L891-L896
mm_inputs = await self.mm_processor.process_mm_data_async(
    image_data=obj.image_data,
    audio_data=obj.audio_data,
    input_text=(input_text or input_ids),
    request_obj=obj,
    max_req_input_len=self.max_req_input_len,
)
```

`QwenVLImageProcessor` 注册在 [`qwen_vl.py`](https://github.com/sgl-project/sglang/blob/874fc07d9bbbb714a71e5d4cbe5e005a885168ef/python/sglang/srt/multimodal/processors/qwen_vl.py) 中，适配 Qwen3-VL / Qwen3-VL-MoE 全系列模型：

```python
# qwen_vl.py#L260-L268
class QwenVLImageProcessor(SGLangBaseProcessor):
    models = [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,   # Qwen3.6-35B-A3B 属于此类
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
        ...
    ]
```

`process_mm_data_async` 中首先调用 `load_mm_data`（base_processor.py 中定义），后者调用 HuggingFace 的 AutoProcessor 对图片进行预处理。其中最关键的是 **smart_resize**：

```python
# qwen_vl.py#L84-L112
def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,   # 28
    min_pixels: int = MIN_PIXELS,  # 4 * 28 * 28 = 3136
    max_pixels: int = MAX_PIXELS,  # 由 SGLANG_IMAGE_MAX_PIXELS 环境变量控制
) -> tuple[int, int]:
    """
    Rescales the image so that:
    1. Both dimensions are divisible by 'factor' (28).
    2. Total pixels is within [min_pixels, max_pixels].
    3. Aspect ratio is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar
```

`MAX_PIXELS` 由 `SGLANG_IMAGE_MAX_PIXELS` 环境变量控制。如果设置了该变量，则超限图片会被等比缩小到目标像素数以内。未设置时使用 HuggingFace processor 配置中的默认值。

### 1.3 HuggingFace Processor 的像素级处理

`smart_resize` 只确定了目标尺寸，**真正的像素插值、归一化**由 HuggingFace 的 `Qwen2VLImageProcessor` 完成。在 `base_processor.py` 中调用：

```python
# base_processor.py#L478-L486
result = processor.__call__(
    text=[input_text],
    padding=True,
    return_tensors="pt",
    **kwargs,   # 包含 images, images_kwargs 等
)
```

HuggingFace `Qwen2VLImageProcessor` 内部对每张图依次执行：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1. Resize | `torchvision.transforms.functional.resize` | 缩放至 `smart_resize` 计算出的 `(h_bar, w_bar)`，使用 bicubic 插值 |
| 2. Rescale | `pixel_values = pixel_values / 255.0` | 将 uint8 [0, 255] 转为 float [0.0, 1.0] |
| 3. Normalize | `(pixel_values - mean) / std` | 减均值除标准差，使用 CLIP 标准值 |
| 4. Flatten | 将 `(C, H, W)` 展平为 `(num_patches, C)` | 为后续 patch embedding 准备 |

Normalize 参数（CLIP 标准值）：

```
image_mean = [0.48145466, 0.4578275, 0.40821073]
image_std  = [0.26862954, 0.26130258, 0.27577711]
```

处理完成后，`pixel_values` 被移到 CPU 上（除非 `--keep-mm-feature-on-device`）：

```python
# base_processor.py#L488-L497
if not self.keep_mm_feature_on_device:
    for feature_name in self.FEATURE_NAMES:
        if feature_name in result and isinstance(
            result[feature_name], torch.Tensor
        ):
            result[feature_name] = result[feature_name].to("cpu")
```

---

## 阶段二：MultimodalDataItem 与 Hash 计算

### 2.1 `from_processor_output` 构造 MultimodalDataItem

文件：[`schedule_batch.py`](https://github.com/sgl-project/sglang/blob/874fc07d9bbbb714a71e5d4cbe5e005a885168ef/python/sglang/srt/managers/schedule_batch.py)

`MultimodalInputs.from_processor_output()` 将 Processor 的输出转为 `MultimodalDataItem` 列表：

```python
# schedule_batch.py#L505-L510
@staticmethod
def from_processor_output(obj: MultimodalProcessorOutput):
    mm_items = obj.mm_items
    assert isinstance(mm_items, list)
    mm_items = [item for item in mm_items if item.is_valid()]
```

### 2.2 MultimodalDataItem 结构

每个 `MultimodalDataItem` 代表一张图的全部数据：

```python
# schedule_batch.py#L246-L272
@dataclasses.dataclass
class MultimodalDataItem:
    modality: Modality
    hash: int = None
    pad_value: int = None
    offsets: Optional[list] = None

    # processor 返回的原始像素特征（归一化后的 pixel_values）
    feature: Union[torch.Tensor, np.ndarray] = None
    # 预计算的 embedding（外部编码器产出），与 feature 互斥
    precomputed_embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None

    model_specific_data: dict[str, Any] = dataclasses.field(default_factory=dict)
```

### 2.3 Hash 计算：`set_pad_value()`

每个 item 的 `hash` 通过对 `feature` 做 SHA256 哈希得到，用于缓存 key 和 RadixAttention 前缀匹配：

```python
# schedule_batch.py#L300-L322
def set_pad_value(self):
    if self.pad_value is not None:
        return
    from sglang.srt.managers.mm_utils import hash_feature

    if self.hash is None:
        if self.feature is not None:
            hashed_feature = self.feature
        else:
            hashed_feature = self.precomputed_embeddings
        self.hash = hash_feature(hashed_feature)
    self.pad_value = _compute_pad_value(self.hash)
```

`hash_feature()` 根据数据类型分派哈希策略：

```python
# mm_utils.py#L1257-L1287
def hash_feature(f):
    if isinstance(f, list):
        if len(f) > 0 and isinstance(f[0], torch.Tensor):
            return tensor_hash(f)
        return data_hash(tuple(flatten_nested_list(f)))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        hasher = hashlib.sha256()
        hasher.update(memoryview(arr))
        hash_bytes = hasher.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder="big", signed=False)
    elif isinstance(f, torch.Tensor):
        return tensor_hash([f])
    return data_hash(f)
```

---

## 阶段三：Scheduler 接收请求

文件：[`scheduler.py`](https://github.com/sgl-project/sglang/blob/874fc07d9bbbb714a71e5d4cbe5e005a885168ef/python/sglang/srt/managers/scheduler.py)

### 3.1 `handle_generate_request`

`TokenizerManager` 将处理后的请求发送给 `Scheduler`：

```python
# scheduler.py#L2190-L2228
if recv_req.mm_inputs is not None:
    image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)
    SessionController.adjust_mm_offsets(recv_req, req, image_inputs)

    # 将单个图片占位符 token 展开为多个 dummy token
    if (
        not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
        and self.pad_input_ids_func
    ):
        req.origin_input_ids = array(
            "q", self.pad_input_ids_func(req.origin_input_ids, image_inputs)
        )
    req.extend_image_inputs(image_inputs)
```

### 3.2 `_get_multimodal_inputs`

```python
# scheduler.py#L2004-L2007
def _get_multimodal_inputs(self, mm_inputs_dict):
    if self.server_args.enable_broadcast_mm_inputs_process:
        return self._process_and_broadcast_mm_inputs(mm_inputs_dict)
    else:
        return MultimodalInputs.from_processor_output(mm_inputs_dict)
```

---

## 阶段四：ViT 编码 — Vision Transformer 前向

这是整个流程的核心：**将预处理后的像素值转化为语义 embedding**。

### 4.1 整体架构

文件：[`qwen3_vl.py`](https://github.com/sgl-project/sglang/blob/874fc07d9bbbb714a71e5d4cbe5e005a885168ef/python/sglang/srt/models/qwen3_vl.py)

Qwen3-VL 的视觉编码器 `Qwen3VLMoeVisionModel` 是一个 **Vision Transformer (ViT)**，不是传统的 CNN 特征提取器。它不使用 SIFT/SURF 等手工设计的 kernel，而是用 Transformer 的 self-attention 机制自动学习图像特征。整个 ViT 由以下组件构成：

```python
# qwen3_vl.py#L309-L398
class Qwen3VLMoeVisionModel(nn.Module, RotaryPosMixin):
    def __init__(self, vision_config, ...):
        self.patch_size = vision_config.patch_size           # 14
        self.spatial_merge_size = vision_config.spatial_merge_size  # 2
        self.temporal_patch_size = vision_config.temporal_patch_size  # 2

        # 1. Patch Embedding: 3D 卷积将图像切分为 patch
        self.patch_embed = Qwen3VLVisionPatchEmbed(config=vision_config)

        # 2. Position Embedding: 可学习的 2D 位置编码
        self.pos_embed = VocabParallelEmbedding(...)

        # 3. 多层 Transformer Block
        self.blocks = nn.ModuleList([
            Qwen3_VisionBlock(...) for _ in range(vision_config.depth)
        ])

        # 4. Patch Merger: 将空间上相邻的 patch 合并
        self.merger = Qwen3VLMoeVisionPatchMerger(...)
```

### 4.2 ViT forward 完整流程

入口函数 `get_image_feature` 将 `pixel_values` 送入 ViT：

```python
# qwen3_vl.py#L1330-L1347
def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
    pixel_values = torch.cat([item.feature for item in items], dim=0).type(
        self.visual.dtype
    )
    image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
    assert pixel_values.dim() == 2, pixel_values.dim()
    return self.visual(pixel_values, grid_thw=image_grid_thw)
```

ViT 的 `forward` 方法完整流程：

```python
# qwen3_vl.py#L876-L986
def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:

    # 第1步：Patch Embedding — 3D 卷积切分图像
    x = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
    x = self.patch_embed(x)

    # 第2步：Position Embedding — 加入 2D RoPE 位置编码
    if self._use_vectorized_pos_embed(len(grid_thw_list)):
        pos_embeds = self.fast_pos_embed_interpolate_vectorized(grid_thw_list)
    else:
        pos_embeds = self.fast_pos_embed_interpolate_from_list(grid_thw_list)
    x += pos_embeds

    rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

    # 第3步：构建 cu_seqlens（每个图像的 token 偏移量，用于 flash attention）
    token_cu_seqlens = np.repeat(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(axis=0, dtype=np.int32)
    ...

    # 第4步：逐层 Transformer Block
    x = x.unsqueeze(1)
    for layer_num, blk in enumerate(self.blocks):
        x = blk(
            x,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            ...
        )

    # 第5步：Patch Merger — 空间合并 + 投影到 LLM 的 hidden_size
    x = self.merger(x)
    return x
```

下面逐层拆解每个组件。

### 4.2.1 Patch Embedding（3D 卷积）

```python
# qwen3_vl.py#L143-L167
class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size              # 14
        self.temporal_patch_size = config.temporal_patch_size  # 2
        self.in_channels = config.in_channels            # 3 (RGB)

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = Conv3dLayer(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states shape: (total_patches, 3 * temporal_patch_size * patch_size * patch_size)
        # 先 reshape 为 (T, C, H, W)，再做 3D 卷积
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.temporal_patch_size, self.in_channels, self.patch_size, self.patch_size
        )
        hidden_states = hidden_states.to(target_dtype).transpose(1, 2)
        hidden_states = self.proj(hidden_states)  # 3D Conv: (T, C, D, H, W) → (T, embed_dim, 1, 1, 1)
        return hidden_states.flatten(2).transpose(1, 2)  # → (num_patches, embed_dim)
```

**作用**：将预处理后的 `pixel_values`（展平后的归一化像素）通过 3D 卷积投影到 ViT 的 hidden_size 维度。对于静态图片，`temporal_patch_size=2` 意味着输入被当作 2 帧处理（patch 重复），适配 Qwen 对图片和视频的统一处理方式。

### 4.2.2 Position Embedding（2D RoPE 位置编码）

位置编码通过**双线性插值**从预训练的固定网格（如 70×70）映射到任意分辨率：

```python
# qwen3_vl.py#L537-L596
def fast_pos_embed_interpolate_from_list(self, grid_thw):
    num_grid_per_side = self.num_grid_per_side  # 预训练位置编码的网格大小
    m_size = self.spatial_merge_size             # 2

    for t, h, w in grid_thw:
        # 在 [0, num_grid_per_side-1] 范围内做双线性插值
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h, ...)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w, ...)

        # 四个角点的双线性插值权重
        h_floor = h_idxs.to(torch.long)
        w_floor = w_idxs.to(torch.long)
        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        w11 = dh_grid * dw_grid
        w10 = dh_grid - w11
        w01 = dw_grid - w11
        w00 = 1 - dh_grid - w01

        # 加权求和得到位置编码
        embeds = self.pos_embed(indices)
        embeds *= weights
        combined = embeds.sum(dim=0)

        # 按 spatial_merge_size 重组
        combined = combined.reshape(
            h // m_size, m_size, w // m_size, m_size, hidden_dim
        )
        combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
```

**作用**：让 ViT 知道每个 patch 在原始图像中的空间位置。使用双线性插值使得位置编码可以适配任意分辨率输入，而不局限于预训练时的固定分辨率。

### 4.2.3 Vision Transformer Block（Self-Attention + MLP）

```python
# qwen3_vl.py#L175-L249
class Qwen3_VisionBlock(nn.Module):
    def __init__(self, dim, num_heads, intermediate_dim, ...):
        self.norm1 = norm_layer(dim)     # LayerNorm
        self.norm2 = norm_layer(dim)     # LayerNorm
        self.attn = VisionAttention(     # Multi-head Self-Attention + RoPE
            embed_dim=dim,
            num_heads=num_heads,
            ...
        )
        self.mlp = Qwen3_VisionMLP(      # SiLU + Linear
            dim, intermediate_dim, hidden_act="silu", ...
        )

    def forward(self, x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, ...):
        # Pre-Norm + Self-Attention
        hidden_states = self.norm1(x)
        hidden_states = rearrange(hidden_states, "s b ... -> b s ...")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            ...
        )
        attn = rearrange(attn, "b s ... -> s b ...")
        x += attn   # 残差连接

        # Pre-Norm + MLP
        norm2 = self.norm2(x)
        mlp = self.mlp(norm2)
        x += mlp    # 残差连接
        return x
```

每个 Block 就是一个标准的 Pre-Norm Transformer 层：
- **Self-Attention**：所有 patch 之间互相做注意力，学习全局空间关系。RoPE 提供位置信息。
- **MLP**：`SiLU` 激活 + 两层线性投影，做逐 token 的非线性变换。

这与传统 CV 方法（如用 Sobel/Canny 等 kernel 手工提取边缘/纹理，再用 CNN 逐层抽象）完全不同。ViT 通过 self-attention 自动学习哪些 patch 之间有关联，不需要手工设计 kernel。

### 4.2.4 Patch Merger（空间合并 + 投影）

```python
# qwen3_vl.py#L249-L297
class Qwen3VLMoeVisionPatchMerger(nn.Module):
    def __init__(self, dim, context_dim, spatial_merge_size=2, ...):
        # merge_size=2, 将 2×2=4 个相邻 patch 的 context 拼接
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.norm = norm_layer(context_dim)
        self.linear_fc1 = ColumnParallelLinear(self.hidden_size, ...)
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(..., dim)

    def forward(self, x):
        x = self.norm(x)
        # 将 2×2 的 spatial grid 合并为一个 token
        x = rearrange(x, "(h w) b d -> b d h w", h=..., w=...)
        x = rearrange(x, "b d (h ph) (w pw) -> b (h w) (ph pw d)", ph=2, pw=2)
        # 投影到 LLM 的 hidden_size
        x = self.linear_fc1(x)
        x = self.act_fn(x)
        x = self.linear_fc2(x)
        return x
```

**作用**：ViT 输出的每个 patch embedding 是 `hidden_size` 维。Patch Merger 将空间上 2×2 的相邻 patch 合并为一个 token，再通过两层线性投影 + GELU 映射到 LLM 的 `hidden_size`。这样做的目的是减少视觉 token 数量（每 4 个 patch 变为 1 个 token），同时将维度对齐到 LLM 的 embedding 空间。

### 4.3 Deepstack（多尺度特征堆叠）

Qwen3-VL 支持 deepstack：在 ViT 的特定中间层（由 `deepstack_visual_indexes` 配置）捕获输出，经过独立的 merger 后与最终输出拼接：

```python
# qwen3_vl.py#L969-L984
for layer_num, blk in enumerate(self.blocks):
    x = blk(x, ...)
    if layer_num in self.deepstack_visual_indexes:
        deepstack_feature = self.deepstack_merger_list[num_deepstack_captured](x)
        deepstack_feature_lists.append(deepstack_feature)
        num_deepstack_captured += 1

x = self.merger(x)
# 多尺度特征拼接：最终输出维度 = hidden_size * (1 + len(deepstack_visual_indexes))
hidden_states = torch.cat([x] + deepstack_feature_lists, dim=1)
```

**作用**：提供多尺度视觉特征，类似于传统 CV 中的 FPN（Feature Pyramid Network），但这里是通过 Transformer 中间层来获取不同抽象级别的特征。

---

## 阶段五：Embedding 缓存

### 5.1 `MultiModalStaticCache` 初始化

文件：[`multimodal_cache.py`](https://github.com/sgl-project/sglang/blob/874fc07d9bbbb714a71e5d4cbe5e005a885168ef/python/sglang/srt/mem_cache/multimodal_cache.py)

缓存大小由 `SGLANG_VLM_CACHE_SIZE_MB` 环境变量控制（默认 100MB），在 KV cache 构建时初始化：

```python
# kv_cache_builder.py#L248-L249
embedding_cache_size = envs.SGLANG_VLM_CACHE_SIZE_MB.get()
init_mm_embedding_cache(embedding_cache_size * 1024 * 1024)

# mm_utils.py#L363-L365
def init_mm_embedding_cache(max_size: int = 0):
    global embedding_cache
    embedding_cache = MultiModalStaticCache(max_size)
```

### 5.2 LRU 淘汰策略

基于 `OrderedDict` 的 LRU 缓存，以**字节数**为限制：

```python
# multimodal_cache.py#L80-L135
class MultiModalStaticCache(MultimodalCache):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.mm_cache: OrderedDict[int, EmbeddingResult] = OrderedDict()
        self.current_size = 0

    def set(self, mm_hash: int, embedding: EmbeddingResult, loc=None) -> bool:
        if mm_hash in self.mm_cache:
            self.mm_cache.move_to_end(mm_hash)     # 命中则更新 LRU 位置
            return True
        data_size = _get_tensor_size(embedding.embedding)
        # 淘汰最旧的 entry，直到有足够空间
        while self.current_size + data_size > self.max_size:
            if not self.mm_cache:                   # 缓存空但放不下 → 返回 False
                return False
            lru_hash, lru_embedding = self.mm_cache.popitem(last=False)
            self.current_size -= _get_tensor_size(lru_embedding.embedding)
        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        return True
```

**关键点**：如果单个 embedding 本身就大于 `max_size`（例如一张 4K 图的 embedding ≈ 344MB，而缓存默认仅 100MB），即使缓存完全为空，`set()` 也会返回 `False`，该 embedding 不会被缓存，每次请求都会重新编码。

### 5.3 Per-Image 路径的缓存使用

在 `_get_chunked_embedding_by_item` 中，每张图单独查询和写入缓存：

```python
# mm_utils.py#L560-L620
def _get_chunked_embedding_by_item(...):
    # 1. 对每张图查缓存
    for idx, item, start, end in overlapping:
        cached = embedding_cache.get_single(item.hash)
        if cached is not None:
            cached_embeddings[idx] = cached.embedding      # 缓存命中
        else:
            miss_items.append((idx, item, start, end))      # 缓存未命中

    # 2. 未命中的批量编码（一次 ViT 调用）
    if miss_items:
        miss_item_list = [item for _, item, _, _ in miss_items]
        _move_items_to_device(miss_item_list, device)
        all_miss_embedding = data_embedding_func(miss_item_list)

        # 3. 按每张图的 token 数拆分，分别写入缓存
        token_counts = [end - start + 1 for _, _, start, end in miss_items]
        split_embeddings = torch.split(all_miss_embedding, token_counts, dim=0)
        for (idx, item, _, _), emb in zip(miss_items, split_embeddings):
            cached_embeddings[idx] = emb
            embedding_cache.set(item.hash, EmbeddingResult(embedding=emb))

    # 4. 拼装当前 chunk 的 embedding
    chunk_slices = []
    for idx, _, start, end in overlapping:
        emb = cached_embeddings[idx]
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end - 1)
        local_start = overlap_start - start
        local_end = overlap_end - start + 1
        chunk_slices.append(emb[local_start:local_end])
    return torch.cat(chunk_slices, dim=0)
```

---

## 阶段六：Embedding 注入与 LLM 推理

### 6.1 `general_mm_embed_routine`

文件：[`mm_utils.py`](https://github.com/sgl-project/sglang/blob/874fc07d9bbbb714a71e5d4cbe5e005a885168ef/python/sglang/srt/managers/mm_utils.py)

模型 forward 的入口，负责编排 ViT 编码和 LLM 推理：

```python
# mm_utils.py#L1030-L1110
def general_mm_embed_routine(
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    language_model: nn.Module,
    multimodal_model: Optional[nn.Module] = None,
    ...
) -> torch.Tensor:
    # 1. 文本 token → text embedding
    input_embeds = embed_tokens(input_ids)

    # 2. ViT 编码 → vision embedding，注入到 text embedding 中
    with torch.profiler.record_function("sglang.vlm.mm_embedding"):
        input_embeds, other_info = embed_mm_inputs(
            mm_inputs_list=mm_inputs_list,
            input_ids=input_ids,
            input_embedding=input_embeds,
            multimodal_model=multimodal_model,
            ...
        )

    # 3. LLM 推理
    return language_model(input_embeds=input_embeds, ...)
```

### 6.2 `embed_mm_inputs`：按 modality 分派

```python
# mm_utils.py#L789-L870
for modality in Modality.all():
    items = [item for item in item_flatten_list if item.is_modality(modality=modality)]
    embedder = getattr(multimodal_model, f"get_{modality.name.lower()}_feature", None)
    if len(items) != 0:
        embedding, mask, input_ids = get_embedding_and_mask(
            data_embedding_func=embedder,   # 即 self.get_image_feature
            embedding_items=items,
            ...
        )
```

对于图片，`embedder` 就是 `Qwen3VLForConditionalGeneration.get_image_feature()`（见 4.2 节），内部调用 `self.visual(pixel_values, grid_thw)` 触发完整 ViT 编码。

### 6.3 注入机制

ViT 输出的每个 embedding 替换 input_ids 中对应位置的 `image_token_id` placeholder。通过 `mask` 标记哪些位置是图像 token，`scatter` 操作将 vision embedding 写入对应位置：

```python
# mm_utils.py#L835-L870
input_embeds[mask] = embedding.to(input_embeds.dtype)
```

---

## 阶段七：请求完成后释放

请求 decode 完成后，Scheduler 调用 `_maybe_clear_mm_inputs` 释放多模态数据：

```python
# scheduler.py#L2050-L2059
def _maybe_clear_mm_inputs(self, batch: ScheduleBatch) -> None:
    for req in batch.reqs:
        if not req.finished() or not (mm_inputs := req.multimodal_inputs):
            continue
        if req.session:
            continue    # session 请求保留 mm_inputs 供后续复用
        mm_inputs.release_features()
        req.multimodal_inputs = None
```

`release_features` 释放 processor 产出的原始 feature tensor（CPU 上的 `pixel_values`）：

```python
# schedule_batch.py#L501-L503
def release_features(self):
    for item in self.mm_items:
        item.feature = None
```

> **注意**：`release_features` 仅释放 `item.feature`（CPU 上的原始像素数据），未显式释放 `item.precomputed_embeddings`（NPU 上的 ViT 编码输出）。`precomputed_embeddings` 的释放依赖 `req.multimodal_inputs = None` 后 `MultimodalDataItem` 的引用计数归零，由 Python 引用计数机制触发析构。

---

## 阶段八：NPU 上的 Feature 生命周期

ViT 编码完成后，`general_mm_embed_routine` 中有一段将 device 上的 feature 移到 CPU 的逻辑：

```python
# mm_utils.py#L1119-L1135
if mm_inputs_list:
    for mm_input_obj in mm_inputs_list:
        if mm_input_obj and hasattr(mm_input_obj, "mm_items"):
            for mm_item in mm_input_obj.mm_items:
                feature = getattr(mm_item, "feature", None)
                if isinstance(feature, torch.Tensor) and feature.is_cuda:
                    mm_item.feature = feature.to("cpu", non_blocking=True)
```

在 NPU 上，`feature.is_cuda` 始终为 `False`（因为 `torch_npu` 的 tensor 不是 CUDA tensor），因此这段卸载逻辑在 NPU 上**不会触发**。`feature` 一直保留在 NPU 设备上，直到请求完成后 `req.multimodal_inputs = None` 触发 Python 引用计数归零，由 NPU 的缓存分配器回收显存。

---

## 关键环境变量总结

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `SGLANG_VLM_CACHE_SIZE_MB` | 100 | `MultiModalStaticCache` 最大字节数，控制可缓存的 ViT embedding 总量 |
| `SGLANG_IMAGE_MAX_PIXELS` | 无 | 图片最大像素数，超限自动缩小（在 `smart_resize` 中生效） |


## 启动参数总结

| 参数 | 作用 |
|------|------|
| `--limit-mm-data-per-request` | 限制单请求的图片/视频/音频数量，超限返回 400 |
| `--max-running-requests` | 最大并发请求数，间接限制同时持有的多模态 embedding 总量 |
