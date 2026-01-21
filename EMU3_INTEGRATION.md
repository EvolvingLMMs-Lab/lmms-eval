# Emu3 Integration for lmms-eval

Emu3 是一个统一的多模态模型，使用 next-token prediction 实现文本生成和图像生成。

## 模型文件

- `lmms_eval/models/simple/emu3.py` - 基础 Emu3 模型（支持理解和生成模式）
- `lmms_eval/models/simple/emu3_visual_cot.py` - Emu3 Visual Chain-of-Thought 模型（两阶段推理）

## 使用方法

### 1. 直接理解模式（Understanding Mode）

用于图像理解任务（如 MMBench, MathVista 等）：

```bash
python -m lmms_eval \
  --model emu3 \
  --model_args pretrained=BAAI/Emu3-Chat-hf \
  --tasks mmbench \
  --batch_size 1 \
  --output_path ./logs/emu3/mmbench
```

### 2. 图像生成模式（Generation Mode）

用于文本到图像生成任务：

```bash
python -m lmms_eval \
  --model emu3 \
  --model_args pretrained=BAAI/Emu3-Gen-hf,mode=generation \
  --tasks ueval \
  --batch_size 1 \
  --output_path ./logs/emu3/ueval
```

### 3. Visual Chain-of-Thought 模式

用于需要辅助可视化的推理任务（如 AuxSolidMath）：

```bash
python -m lmms_eval \
  --model emu3_visual_cot \
  --model_args pretrained=BAAI/Emu3-Chat-hf,gen_pretrained=BAAI/Emu3-Gen-hf,save_intermediate=true \
  --tasks auxsolidmath_easy_visual_cot \
  --batch_size 1 \
  --output_path ./logs/emu3_visual_cot/auxsolidmath_easy
```

## 模型参数

### Emu3 基础模型参数

**理解模式参数：**
- `pretrained`: 模型路径（默认: `BAAI/Emu3-Chat-hf`）
- `mode`: 模式选择 `understanding` 或 `generation`（默认: `understanding`）
- `max_new_tokens`: 最大生成 token 数（默认: 512）
- `do_sample`: 是否采样（默认: False）
- `temperature`: 采样温度（默认: 0.0）
- `device_map`: 设备映射（默认: `auto`）

**生成模式参数：**
- `image_height`: 生成图像高度（默认: 1024）
- `image_width`: 生成图像宽度（默认: 1024）
- `cfg_scale`: Classifier-free guidance scale（默认: 4.0）
- `negative_prompt`: 负面提示词
- `output_image_dir`: 图像输出目录

### Emu3VisualCoT 参数

**Stage 1（图像生成）参数：**
- `gen_pretrained`: 生成模型路径（默认: `BAAI/Emu3-Gen-hf`）
- `stage1_image_height`: 生成图像高度（默认: 1024）
- `stage1_image_width`: 生成图像宽度（默认: 1024）
- `stage1_cfg_scale`: CFG scale（默认: 4.0）
- `stage1_max_new_tokens`: 最大生成步数（默认: 50000）

**Stage 2（理解）参数：**
- `pretrained`: 理解模型路径（默认: `BAAI/Emu3-Chat-hf`）
- `stage2_max_new_tokens`: 最大生成 token 数（默认: 16384）
- `stage2_temperature`: 采样温度（默认: 0.0）
- `stage2_do_sample`: 是否采样（默认: False）

**其他参数：**
- `save_intermediate`: 是否保存中间结果（默认: False）
- `intermediate_dir`: 中间结果保存目录
- `fail_gracefully`: 遇到错误时是否优雅失败（默认: True）

## 架构对比

### Emu3 vs Bagel

| 特性 | Emu3 | Bagel |
|------|------|-------|
| 架构 | Next-token prediction | Diffusion model |
| 图像编码 | VQ-VAE (discrete tokens) | VAE (continuous latent) |
| 生成方式 | Autoregressive | Iterative denoising |
| 速度 | 快（单次前向传播） | 慢（多步迭代） |
| 模型大小 | 8B | 7B |

### Visual CoT 工作流程

**Stage 1: 生成辅助可视化**
1. 输入：原始问题 + 原始图像（可选）
2. 使用 Emu3-Gen 生成辅助图像
3. 保存生成的图像

**Stage 2: 使用辅助图像回答**
1. 输入：原始问题 + 生成的辅助图像
2. 使用 Emu3-Chat 理解并回答
3. 返回最终答案

## 任务配置示例

在 YAML 配置文件中添加 Emu3 特定参数：

```yaml
lmms_eval_specific_kwargs:
  emu3_visual_cot:
    stage1_image_height: 1024
    stage1_image_width: 1024
    stage1_cfg_scale: 4.0
    stage2_max_new_tokens: 16384
    stage2_temperature: 0.0
    stage2_do_sample: false
    save_intermediate: true
```

## 注意事项

1. **内存需求**：Emu3 需要较大的 GPU 内存，建议使用 40GB+ 显存的 GPU
2. **模型下载**：首次运行会自动从 Hugging Face 下载模型
3. **Visual CoT**：需要同时加载 Chat 和 Gen 两个模型，内存需求更大
4. **图像生成**：生成高质量图像需要较长时间（约 1-2 分钟/张）

## 故障排除

**问题：CUDA OOM**
- 解决：使用 `device_map="auto"` 启用模型并行
- 或降低图像分辨率：`stage1_image_height=512,stage1_image_width=512`

**问题：生成图像质量差**
- 解决：调整 `stage1_cfg_scale`（增大可提高质量但降低多样性）
- 或修改 `negative_prompt` 添加更多负面提示

**问题：Flash Attention 不可用**
- 解决：安装 `pip install flash-attn` 或设置 `use_flash_attention_2=false`
