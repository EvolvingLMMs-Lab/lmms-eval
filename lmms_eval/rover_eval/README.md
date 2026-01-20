# ROVER Evaluation Module - Visual CoT Evaluator

专门用于评测 **Visual Chain-of-Thought** 任务的 ROVER 评测模块。

## 核心功能

从 JSON 元数据文件评测 Visual CoT 生成质量，支持两个关键指标：

### RA (Reasoning-to-Visual Alignment)
评测生成的图像是否符合 `generation_prompt` 的要求。

### AL (Answer-to-Visual Alignment)
评测最终答案是否与生成图像和原问题一致。

## 七大任务类型支持

每个任务类型都有定制化的评测标准：

| 类型 | task_category | 应用场景 |
|------|---------------|---------|
| 真实应用 | `real_world` | 日常任务、操作指南、实践应用 |
| 数学推理 | `mathematical` | 数学问题、公式推导、计算 |
| 科学技术 | `stem` | 科学实验、技术过程、工程问题 |
| 谜题游戏 | `puzzles` | 棋类、拼图、逻辑谜题 |
| 图表推理 | `chart_table` | 数据分析、图表解读、趋势预测 |
| 空间智能 | `spatial` | 3D 理解、几何变换、方向判断 |
| 感知推理 | `perception` | 物体识别、场景理解、视觉特征 |

## 快速开始

### 1. 安装依赖

```bash
pip install openai pillow loguru tqdm pandas
```

### 2. 配置 Azure OpenAI API

```bash
export AZURE_API_KEY="your-key"
export AZURE_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_DEPLOYMENT_NAME="gpt-4o"
export AZURE_API_VERSION="2024-02-15-preview"
```

### 3. 准备 JSON 元数据

```json
{
  "doc_id": 0,
  "task": "illusionbench_arshia_icon_scene_visual_cot",
  "generation_prompt": "Analyze and enhance the scene characteristics...",
  "generated_images": ["./logs/stage1.png"],
  "question": "This image contains an icon...",
  "stage2_answer": "Medieval_Village"
}
```

### 4. 评测单个样本

```python
from lmms_eval.rover_eval import VisualCoTEvaluator

# 初始化评测器（可指定任务类型）
evaluator = VisualCoTEvaluator(
    metrics=["ra", "al"],
    task_category="perception"  # 使用感知推理专用标准
)

# 评测
result = evaluator.evaluate_from_json(
    json_path="0_metadata.json",
    original_image="path/to/original_image.png"
)

print(f"RA Score: {result['ra_score']}/5")
print(f"AL Score: {result['al_score']}/5")
```

### 5. 批量评测

```python
from lmms_eval.rover_eval import VisualCoTEvaluator
import pandas as pd

evaluator = VisualCoTEvaluator(task_category="chart_table")

results = evaluator.evaluate_batch(
    json_paths=["0_metadata.json", "1_metadata.json", ...],
    original_images=["img0.png", "img1.png", ...],
    max_workers=10
)

# 保存结果
df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)

print(f"Average RA: {df['ra_score'].mean():.2f}")
print(f"Average AL: {df['al_score'].mean():.2f}")
```

### 6. 使用命令行脚本

```bash
python examples/rover_eval/evaluate_visual_cot.py \
    --log_dir ./logs/bagel_visual_cot/ \
    --image_dir ./dataset/illusionbench/images/ \
    --output results.csv \
    --task_category perception \
    --max_workers 10
```

## 评分标准

| 分数 | 含义 | 完成度 |
|------|------|--------|
| 5 | 完美 | 100% 符合要求 |
| 4 | 优秀 | 80-90% 符合 |
| 3 | 及格 | 60-70% 符合 |
| 2 | 较差 | 30-50% 符合 |
| 1 | 失败 | <30% 符合 |

## RA 评测标准

- **指令遵循度 (40%)**: 是否按照 generation_prompt 生成
- **视觉质量 (30%)**: 图像清晰度、完整性
- **任务相关性 (30%)**: 是否有助于回答原问题

## AL 评测标准

- **视觉-答案一致性 (50%)**: 答案是否有视觉证据支持
- **问题-答案对齐 (30%)**: 答案是否正确回答问题
- **推理连贯性 (20%)**: 推理链是否合理（原图→生成图→答案）

## 任务类型定制示例

### 数学推理
```python
evaluator = VisualCoTEvaluator(task_category="mathematical")
# RA 会检查：公式可视化、计算步骤清晰度
# AL 会检查：数学正确性、公式应用正确性
```

### 图表推理
```python
evaluator = VisualCoTEvaluator(task_category="chart_table")
# RA 会检查：数据点标注、趋势可视化
# AL 会检查：数据准确性、无数据幻觉
```

### 空间智能
```python
evaluator = VisualCoTEvaluator(task_category="spatial")
# RA 会检查：3D 理解、几何变换准确性
# AL 会检查：空间描述正确性、方向准确性
```

## API 参考

### VisualCoTEvaluator

```python
class VisualCoTEvaluator:
    def __init__(
        self,
        metrics: List[str] = None,  # ["ra", "al"]
        task_category: Optional[str] = None,  # 任务类型
        max_retries: int = 3,
    )
    
    def evaluate_from_json(
        self,
        json_path: str,
        original_image: Union[str, Image.Image],
        task_category: Optional[str] = None,  # 可覆盖初始化设置
    ) -> Dict
    
    def evaluate_batch(
        self,
        json_paths: List[str],
        original_images: List[Union[str, Image.Image]],
        max_workers: int = 10,
    ) -> List[Dict]
```

### 函数式 API

```python
from lmms_eval.rover_eval import evaluate_from_json, evaluate_batch_from_jsons

result = evaluate_from_json(
    json_path="0_metadata.json",
    original_image="img.png",
    metrics=["ra", "al"],
    task_category="mathematical",
    max_retries=3
)
```

## 输出格式

```json
{
  "doc_id": "0",
  "task": "illusionbench_arshia_icon_scene_visual_cot",
  "json_path": "0_metadata.json",
  "ra_score": 4,
  "ra_reasoning": "Generated image successfully emphasizes...",
  "al_score": 5,
  "al_reasoning": "Answer 'Medieval_Village' is perfectly supported..."
}
```

## 文档

- **详细指南**: `VISUAL_COT_GUIDE.md`
- **示例脚本**: `examples/rover_eval/evaluate_visual_cot.py`

## 注意事项

1. **原始图像**: JSON 中只包含生成的图像，需要提供原始问题图像路径
2. **API 成本**: 每个样本调用 GPT-4o 两次（RA + AL），注意成本
3. **并行控制**: 使用 `max_workers` 控制并行度，避免 API rate limit
4. **任务类型**: 优先级为 `task_category` 参数 > 自动检测任务名

## 模块结构

```
lmms_eval/rover_eval/
├── __init__.py                   # 主入口
├── README.md                     # 本文档
├── VISUAL_COT_GUIDE.md          # 详细使用指南
├── visual_cot_evaluator.py      # 核心评测逻辑
├── visual_cot_prompts.py        # RA/AL prompt + 七大任务定制
└── api.py                       # GPT-4o API 调用

examples/rover_eval/
└── evaluate_visual_cot.py       # 批量评测脚本
```
