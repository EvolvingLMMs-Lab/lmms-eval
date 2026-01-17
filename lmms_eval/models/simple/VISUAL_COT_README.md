# Visual Chain-of-Thought (Visual CoT) 模型架构说明

## 1. 概述

Visual CoT 是一种两阶段推理架构，通过生成辅助可视化图像来增强视觉推理能力：

```
┌─────────────────────────────────────────────────────────────────┐
│                      Visual CoT Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐     ┌──────────────────┐     ┌──────────────┐   │
│   │ Original │────▶│     Stage 1      │────▶│  Auxiliary   │   │
│   │  Image   │     │ Image Generation │     │    Image     │   │
│   └──────────┘     │   (with prompt)  │     └──────────────┘   │
│        │           └──────────────────┘            │           │
│        │                                           │           │
│        │           ┌──────────────────┐            │           │
│        └──────────▶│     Stage 2      │◀───────────┘           │
│                    │ Question Answer  │                        │
│                    │  (both images)   │                        │
│                    └────────┬─────────┘                        │
│                             │                                  │
│                             ▼                                  │
│                    ┌──────────────────┐                        │
│                    │   Final Answer   │                        │
│                    └──────────────────┘                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Stage 1 原图传入确认 ✅

**所有 3 个 Visual CoT 模型都在 Stage 1 传入了原图：**

### 2.1 bagel_visual_cot

```python
# generate_until 方法中获取原图 (line 327-338)
original_image = None
if doc_to_visual is not None:
    doc = self.task_dict[task][split][doc_id]
    original_visuals = doc_to_visual(doc)
    if original_visuals and len(original_visuals) > 0:
        original_image = original_visuals[0]

# Stage 1 传入原图 (line 363-368)
stage1_text, generated_images = self._stage1_generate_image(
    generation_prompt=generation_prompt,
    doc_id=doc_id,
    task=task,
    original_image=original_image,  # ← 原图传入
)

# _stage1_generate_image 内部 (line 195-200)
text, images = self.bagel.generate_text_and_image(
    prompt=generation_prompt,
    doc_id=f"{doc_id}_stage1",
    task=task,
    image=original_image,  # ← 传给 Bagel 模型
)
```

### 2.2 azure_trapi_visual_cot

```python
# generate_until 方法中获取原图 (line 352-356)
original_image = None
visuals = doc_to_visual(self.task_dict[task][split][doc_id])
if visuals and len(visuals) > 0:
    original_image = visuals[0]

# Stage 1 传入原图 (line 377-382)
auxiliary_image = self._stage1_generate_image(
    prompt=generation_prompt,
    original_image=original_image,  # ← 原图传入
    doc_id=doc_id,
    task=task,
)

# _stage1_generate_image 内部 (line 205-223)
if original_image is not None:
    # Use images.edit API for image-to-image generation
    img_buffer = BytesIO()
    original_image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    response = self.client.images.edit(  # ← 使用 images.edit API
        model=self.image_deployment,
        image=img_buffer,              # ← 原图作为输入
        prompt=prompt,
        size=self.stage1_image_size,
        n=1,
    )
```

### 2.3 nano_banana_visual_cot

```python
# generate_until 方法中获取原图 (line 416-420)
original_image = None
visuals = doc_to_visual(self.task_dict[task][split][doc_id])
if visuals and len(visuals) > 0:
    original_image = visuals[0]

# Stage 1 传入原图 (line 443-448)
auxiliary_image = self._stage1_generate_image(
    prompt=generation_prompt,
    original_image=original_image,  # ← 原图传入
    doc_id=doc_id,
    task=task,
)

# _stage1_generate_image 内部
# DMXAPI 模式 (line 214-231):
if original_image is not None:
    response = self.client.images.edit(  # ← 使用 images.edit API
        model=self.image_model_name,
        image=img_buffer,                # ← 原图作为输入
        prompt=prompt,
        ...
    )

# Google API 模式 (line 257-270):
content = []
if original_image is not None:
    content.append(original_image)      # ← 原图加入 content
content.append(prompt)
response = self.client.models.generate_content(
    model=self.model_name,
    contents=content,                   # ← 包含原图的 content
    ...
)
```

## 3. API 使用方式对比

| 模型 | API | 有原图时 | 无原图时 |
|------|-----|----------|----------|
| bagel_visual_cot | Bagel Inferencer | `image=original_image` | `image=None` |
| azure_trapi_visual_cot | Azure OpenAI | `images.edit(image=...)` | `images.generate(...)` |
| nano_banana_visual_cot (DMXAPI) | OpenAI-compatible | `images.edit(image=...)` | `images.generate(...)` |
| nano_banana_visual_cot (Google) | Google Gemini | `contents=[image, prompt]` | `contents=[prompt]` |

## 4. Prompt 格式

Visual CoT 任务使用特殊的 prompt 格式：

```
[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question}[/QUESTION]
```

- **`[GEN_PROMPT]`**: Stage 1 图像生成的 prompt（描述要生成什么样的辅助图）
- **`[QUESTION]`**: Stage 2 问答的 prompt（原始问题 + 指令）

### 4.1 解析逻辑

```python
gen_prompt_match = re.search(
    r"\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]", contexts, re.DOTALL
)
question_match = re.search(
    r"\[QUESTION\](.*?)\[/QUESTION\]", contexts, re.DOTALL
)

if gen_prompt_match and question_match:
    generation_prompt = gen_prompt_match.group(1).strip()
    question = question_match.group(1).strip()
```

## 5. 两阶段详细流程

### Stage 1: 生成辅助图像

**输入**:
- 原始图像 (original_image)
- 生成 prompt (generation_prompt)

**输出**:
- 辅助图像 (auxiliary_image)

**作用**: 根据原图和 prompt，生成一张辅助理解的图像（如标注、辅助线、可视化等）

### Stage 2: 回答问题

**输入**:
- 原始图像 (original_image)
- 辅助图像 (auxiliary_image)
- 问题 prompt (question)

**输出**:
- 最终答案 (answer)

**作用**: 结合两张图像理解问题并给出答案

## 6. 关键设计原则

### 6.1 Stage 1 必须传入原图

Stage 1 **必须**传入原图，原因：
1. 辅助图像需要基于原图内容生成（如在原图上添加辅助线）
2. Image-to-Image 生成比纯文本生成更准确
3. 保持原图的结构和元素一致性

### 6.2 防止信息泄漏

Generation Prompt **不应该**包含答案候选：

❌ 错误示例（泄漏）：
```
The hidden shape represents a category like: animal, vehicle, sports equipment...
```

✅ 正确示例：
```
This image shows a scene where elements are carefully arranged to form a hidden shape.
```

### 6.3 Stage 2 Prompt 对齐

Stage 2 的 Question Prompt 应与直接生成版本对齐，仅增加说明有两张图像：

```
You are given TWO images: the original image and an auxiliary visualization.
{与直接生成版本相同的问题和指令}
```

## 7. 运行示例

```bash
# Azure TRAPI Visual CoT
python -m lmms_eval \
    --model azure_trapi_visual_cot \
    --model_args save_intermediate=true \
    --tasks geometry3k_visual_cot \
    --batch_size 1 \
    --output_path ./logs/

# Bagel Visual CoT
python -m lmms_eval \
    --model bagel_visual_cot \
    --model_args pretrained=/path/to/BAGEL-7B-MoT,save_intermediate=true \
    --tasks geometry3k_visual_cot \
    --batch_size 1 \
    --output_path ./logs/

# Nano Banana Visual CoT
export DMXAPI_API_KEY="your-key"
python -m lmms_eval \
    --model nano_banana_visual_cot \
    --model_args save_intermediate=true \
    --tasks geometry3k_visual_cot \
    --batch_size 1 \
    --output_path ./logs/
```

## 8. 中间结果保存

设置 `save_intermediate=true` 后，Stage 1 生成的辅助图像会保存到：
```
{output_path}/{task_name}/{doc_id}_stage1.png
```

这有助于调试和分析 Visual CoT 的效果。
