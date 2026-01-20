# Visual CoT ROVER 评测指南

## 概述

Visual CoT ROVER 专门用于评测 **Visual Chain-of-Thought** 任务，支持从 JSON 元数据文件加载评测数据。

支持七大任务类型的定制化评测标准：
1. **Real-world Applications** (真实应用)
2. **Mathematical Reasoning** (数学推理)
3. **STEM** (科学技术工程数学)
4. **Puzzles and Games** (谜题游戏)
5. **Chart & Table Reasoning** (图表推理)
6. **Spatial Intelligence** (空间智能)
7. **Perception Reasoning** (感知推理)

## JSON 格式

```json
{
  "doc_id": 0,
  "task": "illusionbench_arshia_icon_scene_visual_cot",
  "generation_prompt": "Analyze and enhance the scene characteristics...",
  "stage1_text": null,
  "generated_images": [
    "./logs/bagel_visual_cot/illusionbench_arshia_icon_scene_visual_cot_0_stage1.png"
  ],
  "question": "This image contains an icon integrated into a background...",
  "stage2_answer": "Medieval_Village"
}
```

## 两个评测指标

### RA (Reasoning-to-Visual Alignment)
**评测内容**: 生成的图像是否符合 `generation_prompt` 的要求

**评分依据**:
- 指令遵循度 (40%): 是否按照 prompt 生成
- 视觉质量 (30%): 图像清晰度、完整性
- 任务相关性 (30%): 是否有助于回答问题

**输入**:
- 原始问题图像
- 生成的可视化图像
- `generation_prompt`
- `question`

### AL (Answer-to-Visual Alignment)
**评测内容**: 最终答案是否与生成的图像和问题一致

**评分依据**:
- 视觉-答案一致性 (50%): 答案是否有视觉证据支持
- 问题-答案对齐 (30%): 答案是否回答了问题
- 推理连贯性 (20%): 推理链是否合理

**输入**:
- 原始问题图像
- 生成的可视化图像
- `stage2_answer`
- `question`

## 安装

```bash
pip install openai pillow loguru tqdm
```

## 配置 API

```bash
export AZURE_API_KEY="your-key"
export AZURE_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_DEPLOYMENT_NAME="gpt-4o"
export AZURE_API_VERSION="2024-02-15-preview"
```

## 使用方法

### 1. 单个样本评测

```python
from lmms_eval.rover_eval import VisualCoTEvaluator
from PIL import Image

# 初始化评测器
evaluator = VisualCoTEvaluator(
    metrics=["ra", "al"],  # 两个指标都评测
    max_retries=3
)

# 评测单个 JSON
result = evaluator.evaluate_from_json(
    json_path="0_metadata.json",
    original_image="path/to/original_question_image.png"
)

print(f"RA Score: {result['ra_score']}/5")
print(f"RA Reasoning: {result['ra_reasoning']}")
print(f"\nAL Score: {result['al_score']}/5")
print(f"AL Reasoning: {result['al_reasoning']}")
```

### 2. 批量评测

```python
from lmms_eval.rover_eval import VisualCoTEvaluator
import json
from pathlib import Path

# 准备数据
json_dir = Path("./logs/bagel_visual_cot/")
json_files = list(json_dir.glob("*_metadata.json"))

# 加载原始图像路径
original_images = []
for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)
        # 假设你有原始图像的映射关系
        doc_id = data["doc_id"]
        original_images.append(f"./dataset/images/{doc_id}.png")

# 批量评测
evaluator = VisualCoTEvaluator()
results = evaluator.evaluate_batch(
    json_paths=[str(f) for f in json_files],
    original_images=original_images,
    max_workers=10  # 并行评测
)

# 保存结果
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("visual_cot_evaluation_results.csv", index=False)

# 统计
print(f"Average RA Score: {df['ra_score'].mean():.2f}")
print(f"Average AL Score: {df['al_score'].mean():.2f}")
```

### 3. 只评测某一个指标

```python
# 只评测 RA
evaluator = VisualCoTEvaluator(metrics=["ra"])
result = evaluator.evaluate_from_json(
    json_path="0_metadata.json",
    original_image="original.png"
)
# result 中只有 ra_score 和 ra_reasoning

# 只评测 AL
evaluator = VisualCoTEvaluator(metrics=["al"])
result = evaluator.evaluate_from_json(
    json_path="0_metadata.json",
    original_image="original.png"
)
# result 中只有 al_score 和 al_reasoning
```

### 4. 使用函数式 API

```python
from lmms_eval.rover_eval import evaluate_from_json

result = evaluate_from_json(
    json_path="0_metadata.json",
    original_image="original.png",
    metrics=["ra", "al"],
    task_category=None,  # 可选：任务类型
    max_retries=3
)
```

## 评分标准

| 分数 | 含义 | 完成度 |
|------|------|--------|
| 5 | 完美 | 100% 符合要求 |
| 4 | 优秀 | 80-90% 符合 |
| 3 | 及格 | 60-70% 符合 |
| 2 | 较差 | 30-50% 符合 |
| 1 | 失败 | <30% 符合 |

## 七大任务类型定制评测

### 使用任务类型

```python
# 方式 1: 初始化时指定
evaluator = VisualCoTEvaluator(task_category="mathematical")
result = evaluator.evaluate_from_json(...)

# 方式 2: 评测时指定
evaluator = VisualCoTEvaluator()
result = evaluator.evaluate_from_json(
    json_path="0_metadata.json",
    original_image="img.png",
    task_category="mathematical"  # 数学推理专用标准
)
```

### 任务类型详解

| task_category | 说明 | RA 评测重点 | AL 评测重点 |
|---------------|------|------------|------------|
| `real_world` | 真实应用 | 实用性、安全性、可行性 | 答案的实际可执行性 |
| `mathematical` | 数学推理 | 公式可视化、计算步骤 | 数学正确性、公式应用 |
| `stem` | 科学技术 | 科学准确性、技术精度 | 科学有效性、技术准确性 |
| `puzzles` | 谜题游戏 | 规则遵守、策略清晰度 | 规则符合性、解法有效性 |
| `chart_table` | 图表推理 | 数据标注、趋势可视化 | 数据准确性、无幻觉 |
| `spatial` | 空间智能 | 3D理解、变换准确性 | 空间正确性、几何一致性 |
| `perception` | 感知推理 | 物体突出、场景理解 | 识别准确性、属性正确性 |

### 自动检测（向后兼容）

如果不指定 `task_category`，系统会自动从任务名检测：
- `illusionbench` → `perception`
- `chartqa` → `chart_table`
- `mathvista` → `mathematical`

## 完整示例脚本

```python
#!/usr/bin/env python3
"""
Visual CoT ROVER 评测脚本
"""
import json
import sys
from pathlib import Path
from lmms_eval.rover_eval import VisualCoTEvaluator

def main():
    # 配置
    log_dir = Path("./logs/bagel_visual_cot/")
    image_dir = Path("./dataset/illusionbench/images/")
    output_csv = "visual_cot_results.csv"
    
    # 收集所有 JSON 文件
    json_files = sorted(log_dir.glob("*_metadata.json"))
    print(f"Found {len(json_files)} JSON files")
    
    # 准备评测数据
    json_paths = []
    original_images = []
    
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
            doc_id = data["doc_id"]
            
            # 根据你的数据集结构找到原始图像
            original_img = image_dir / f"{doc_id}.png"
            if not original_img.exists():
                print(f"Warning: {original_img} not found, skipping")
                continue
            
            json_paths.append(str(json_file))
            original_images.append(str(original_img))
    
    print(f"Evaluating {len(json_paths)} samples")
    
    # 初始化评测器
    evaluator = VisualCoTEvaluator(
        metrics=["ra", "al"],
        max_retries=3
    )
    
    # 批量评测
    results = evaluator.evaluate_batch(
        json_paths=json_paths,
        original_images=original_images,
        max_workers=10
    )
    
    # 保存结果
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    # 统计信息
    print("\n=== Evaluation Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Average RA Score: {df['ra_score'].mean():.2f}")
    print(f"Average AL Score: {df['al_score'].mean():.2f}")
    print(f"\nRA Score Distribution:")
    print(df['ra_score'].value_counts().sort_index())
    print(f"\nAL Score Distribution:")
    print(df['al_score'].value_counts().sort_index())

if __name__ == "__main__":
    main()
```

## 任务类型示例

### 数学推理任务
```python
evaluator = VisualCoTEvaluator(task_category="mathematical")
result = evaluator.evaluate_from_json(
    json_path="math_cot_metadata.json",
    original_image="equation.png"
)
# RA 会检查：公式可视化、计算步骤清晰度
# AL 会检查：答案数学正确性、公式应用正确性
```

### 图表推理任务
```python
evaluator = VisualCoTEvaluator(task_category="chart_table")
result = evaluator.evaluate_from_json(
    json_path="chartqa_cot_metadata.json",
    original_image="chart.png"
)
# RA 会检查：数据点标注、趋势可视化
# AL 会检查：数据准确性、无数据幻觉
```

### 空间推理任务
```python
evaluator = VisualCoTEvaluator(task_category="spatial")
result = evaluator.evaluate_from_json(
    json_path="spatial_cot_metadata.json",
    original_image="3d_object.png"
)
# RA 会检查：3D 理解、几何变换准确性
# AL 会检查：空间描述正确性、方向准确性
```

## 输出示例

```json
{
  "doc_id": "0",
  "task": "illusionbench_arshia_icon_scene_visual_cot",
  "json_path": "0_metadata.json",
  "ra_score": 4,
  "ra_reasoning": "Generated image successfully emphasizes scene characteristics...",
  "al_score": 5,
  "al_reasoning": "Answer 'Medieval_Village' is perfectly supported by visual evidence..."
}
```

## 注意事项

1. **原始图像路径**: JSON 中只包含生成的图像，需要你提供原始问题图像的路径
2. **并行评测**: 使用 `max_workers` 控制并行度，避免 API rate limit
3. **错误处理**: 评测失败的样本 score 为 None，reasoning 包含错误信息
4. **API 成本**: 每个样本会调用 GPT-4o 两次（RA + AL），注意成本控制
