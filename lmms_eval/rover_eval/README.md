# ROVER Evaluation Module

评估 **Interleaved Reasoning** 和 **Reasoning Alignment** 的独立模块。

基于 [ROVER Benchmark](https://github.com/xxx/ROVER) 实现，使用 GPT-4o 通过 Azure TRAPI 进行评估。

## 概述

ROVER Evaluation 评估视觉语言模型的推理能力，包含三个核心指标：

| Metric | 评估内容 | 输入 |
|--------|---------|------|
| `reasoning_process` | 推理过程文本质量 | 原图 + think_output |
| `reasoning_visual` | 视觉推理结果质量 | 原图 + 生成图 + (target图) |
| `reasoning_alignment` | 推理文本与生成图的对齐度 | 原图 + 生成图 + think_output |

## 安装

确保已安装依赖：

```bash
pip install azure-identity openai pillow
```

配置 Azure CLI 登录（用于 TRAPI 认证）：

```bash
az login
```

## 环境变量（可选）

```bash
export TRAPI_INSTANCE="gcr/shared"          # Azure endpoint instance
export TRAPI_DEPLOYMENT="gpt-4o_2024-11-20" # GPT-4o deployment name
export TRAPI_API_VERSION="2024-10-21"       # API version
export TRAPI_SCOPE="api://trapi/.default"   # OAuth scope
```

## 快速使用

### 单样本评估

```python
from lmms_eval.rover_eval import ROVEREvaluator

evaluator = ROVEREvaluator(
    metrics=["reasoning_process", "reasoning_visual", "reasoning_alignment"]
)

result = evaluator.evaluate(
    sample_id="sample_001",
    original_image="path/to/original.png",
    generated_image="path/to/generated.png",
    prompt="Show what this plant looks like after 3 months",
    think_output="The plant will grow taller through photosynthesis...",
    reasoning_type="temporal",
    dimension="science",
    keywords="plant development, photosynthesis, growth",
    target_description="leaves expanded and more numerous; stem visibly longer",
)

print(f"Process Score: {result['reasoning_process_score']}")
print(f"Visual Score: {result['reasoning_visual_score']}")
print(f"Alignment Score: {result['reasoning_alignment_score']}")
```

### 批量评估

```python
from lmms_eval.rover_eval import ROVEREvaluator, EvaluationSample

evaluator = ROVEREvaluator()

samples = [
    EvaluationSample(
        sample_id="sample_001",
        original_image="path/to/original_1.png",
        generated_image="path/to/generated_1.png",
        prompt="Task instruction 1",
        think_output="Model reasoning 1...",
        reasoning_type="temporal",
        dimension="science",
    ),
    EvaluationSample(
        sample_id="sample_002",
        original_image="path/to/original_2.png",
        generated_image="path/to/generated_2.png",
        prompt="Task instruction 2",
        think_output="Model reasoning 2...",
        reasoning_type="spatial",
        dimension="common_sense",
    ),
]

results = evaluator.evaluate_batch(
    samples,
    num_workers=10,
    output_path="results.jsonl",
)

# 计算平均分
averages = evaluator.compute_average_scores(results)
print(averages)
```

### 使用函数式 API

```python
from lmms_eval.rover_eval import evaluate_single_sample, evaluate_batch, EvaluationSample

sample = EvaluationSample(
    sample_id="sample_001",
    original_image="path/to/original.png",
    generated_image="path/to/generated.png",
    prompt="Show what this plant looks like after 3 months",
    think_output="The plant will grow taller...",
    reasoning_type="temporal",
    dimension="science",
)

result = evaluate_single_sample(sample)
```

## 评估指标详解

### 1. Reasoning Process (推理过程)

评估模型生成的**推理文本**（think_output）的质量。

**评估维度**：
- **Logical Structure**: 推理是否有组织、有逻辑
- **Domain Knowledge**: 是否正确应用领域知识
- **Reasoning Logic**: 推理是否遵循正确的因果关系
- **Completeness**: 是否包含所有必要的推理步骤

**评分标准** (1-5分)：
- 5: 完美的推理过程，逻辑严密，领域知识正确
- 4: 高质量推理，达到80-90%要求，有轻微不足
- 3: 基本合格，达到60-70%要求，有明显缺陷
- 4: 质量差，30-50%达标，存在重大逻辑错误
- 1: 失败，推理根本错误或缺失

**注意**: 如果 think_output 为空，直接给1分。

### 2. Reasoning Visual (视觉推理)

评估模型生成的**图像**是否符合目标要求。

**评估维度**：
- **Target Match**: 生成图是否匹配目标描述
- **Visual Changes**: 从原图到生成图的变化是否合理
- **Domain Knowledge**: 变化是否符合领域原理
- **Logic Validation**: 变化是否符合逻辑

**评分标准** (1-5分)：
- 5: 完美匹配目标，视觉逻辑无误
- 4: 高质量匹配，80-90%目标达成
- 3: 基本匹配，60-70%目标达成，有明显缺陷
- 2: 匹配差，30-50%目标达成
- 1: 完全不匹配或存在根本性错误

### 3. Reasoning Alignment (推理对齐)

评估**推理文本**和**生成图像**之间的**一致性**。

**评估维度**：
- **Process-Visual Consistency**: 文本描述是否与视觉变化匹配
- **Conclusion Coherence**: 文本结论是否与视觉结果一致
- **Step-by-Step Alignment**: 每个推理步骤是否有对应的视觉证据
- **Logical Consistency**: 是否存在矛盾

**评分约束**: Alignment分数不能比Visual分数高超过1分
- Visual = 1 → Alignment ≤ 2
- Visual = 2 → Alignment ≤ 3
- Visual = 3 → Alignment ≤ 4

## Reasoning Types (推理类型)

支持的推理类型：

| Type | 描述 |
|------|------|
| `temporal` | 时序推理（时间变化） |
| `spatial` | 空间推理（位置关系） |
| `causal` | 因果推理（因果关系） |
| `quantitative` | 数量推理（数量变化） |
| `imaginative` | 想象推理（创意生成） |
| `logical` | 逻辑推理（逻辑关系） |

## Dimensions (知识领域)

支持的知识领域：

| Dimension | 描述 |
|-----------|------|
| `science` | 科学：遵循科学原理和自然规律 |
| `humanity` | 人文：考虑文化、历史和社会背景 |
| `common_sense` | 常识：使用日常知识和实践理解 |
| `logic` | 逻辑：遵循形式推理和数学原则 |

## 输出格式

每个样本的评估结果：

```json
{
  "sample_id": "sample_001",
  "reasoning_process_score": 4,
  "reasoning_process_reasoning": "The reasoning is well-structured...",
  "reasoning_visual_score": 5,
  "reasoning_visual_reasoning": "The generated image perfectly matches...",
  "reasoning_alignment_score": 4,
  "reasoning_alignment_reasoning": "Process and visual are well aligned..."
}
```

## 数据流

```
模型输入: 原图 + prompt
         ↓
模型输出: think_output (推理文本) + generated_image (生成图像)
         ↓
ROVER评估:
  ├── reasoning_process: GPT-4o 评估 think_output 质量
  ├── reasoning_visual: GPT-4o 评估 generated_image 质量
  └── reasoning_alignment: GPT-4o 评估 think_output ↔ generated_image 对齐度
         ↓
输出: 3个分数 (1-5) + 详细reasoning
```

## 文件结构

```
lmms_eval/rover_eval/
├── __init__.py      # 模块入口
├── api.py           # Azure GPT-4o API 客户端
├── prompts.py       # 评估提示词模板
├── evaluator.py     # 主评估逻辑
└── README.md        # 本文档
```

## 参考

- [ROVER Benchmark](https://github.com/xxx/ROVER)
- 原始评估代码: `evaluate_rover.py`, `evaluator.py`, `base_metric.py`
