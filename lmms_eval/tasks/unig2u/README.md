# UniG2U Benchmark

UniG2U（Unified Generation-to-Understanding）是一个评估多模态模型**统一理解与生成能力**的 benchmark。它覆盖 11 个子任务，横跨图表理解、几何推理、物理分析、视觉空间规划等多种场景，旨在衡量模型能否在"先生成辅助可视化、再回答问题"的两阶段流程中获得增益。

## 任务概览

UniG2U 提供两种评测模式：

| 模式 | Task 名 | 含义 |
|------|---------|------|
| **标准理解** | `unig2u` | 直接看图回答问题（单阶段） |
| **GtA（Generation-to-Answer）** | `unig2u_GtA` | 先生成辅助图像，再结合辅助图像回答（两阶段） |

### 子任务列表

| 子任务 | 领域 | 样本数 |
|--------|------|--------|
| ChartQA100 | 图表理解 | 100 |
| Geometry3K | 平面几何 | — |
| AuxSolidMath | 立体几何（辅助线） | — |
| BabyVision | 细粒度视觉判别 + 追踪 | 246 |
| IllusionBench | 错觉艺术识别 | — |
| MMSI-Bench | 多模态空间智能（5 子项） | 500 |
| PhyX | 物理（光学 + 力学） | 200 |
| RealUnify | 认知心理学（追踪/重建/聚焦） | 300 |
| Uni-MMMU | 交错生成（拼图/迷宫/滑块） | 254 |
| VSP | 视觉空间规划（碰撞/导航） | 100 |
| VisualPuzzles | 视觉推理（5 种推理类型） | 500 |

## 快速测试

```bash
# 标准理解 — 跑全部 11 个子任务
python -m lmms_eval \
    --model ovis_u1 \
    --model_args pretrained=AIDC-AI/Ovis-U1-3B \
    --tasks unig2u \
    --batch_size 1

# GtA（两阶段）— 跑全部 11 个子任务的 Visual CoT 版本
python -m lmms_eval \
    --model ovis_u1 \
    --model_args pretrained=AIDC-AI/Ovis-U1-3B \
    --tasks unig2u_GtA \
    --batch_size 1

# 只跑单个子任务
python -m lmms_eval \
    --model bagel \
    --model_args pretrained=/path/to/BAGEL-7B-MoT,mode=understanding \
    --tasks unig2u_chartqa100 \
    --batch_size 1

# 快速验证（限制样本数）
python -m lmms_eval \
    --model ovis_u1 \
    --model_args pretrained=AIDC-AI/Ovis-U1-3B \
    --tasks unig2u_chartqa100 \
    --batch_size 1 \
    --limit 2
```

## GtA 两阶段流程说明

GtA 模式下，模型执行两步推理：

```
Stage 1: 原始图像 + 生成提示词 → 生成辅助可视化图像
Stage 2: 原始图像 + 辅助图像 + 原始问题 → 回答
```

**触发方式**：GtA 的 yaml 配置中 `generation_kwargs` 包含 `visual_cot: true`，模型的 `generate_until()` 检测到该标志后，显式调用两阶段生成流程，而非普通的单阶段理解。

```yaml
# 标准版 yaml（单阶段）
generation_kwargs:
  max_new_tokens: 16
  temperature: 0

# GtA 版 yaml（两阶段，多了 visual_cot: true）
generation_kwargs:
  visual_cot: true        # ← 这个标志触发两阶段
  max_new_tokens: 16
  temperature: 0
```

## 已支持的模型

| 模型 | model 名 | 标准理解 | GtA | 备注 |
|------|----------|:---:|:---:|------|
| Ovis-U1 | `ovis_u1` | ✓ | ✓ | 条件生成 + 理解 |
| Bagel | `bagel` | ✓ | ✓ | mode=understanding/generation |
| ILLUME+ | `illume_plus` | ✓ | ✓ | enable_visual_cot 参数 |
| MMaDa | `mmada` | ✓ | — | 仅理解 + 生成，无 GtA |
| Qwen-Image-Edit | `qwen_image_edit` | ✓ | ✓ | 编辑模型做 Stage1 |

## 如何接入新模型

要让一个新模型支持 UniG2U，需要实现以下接口：

### 1. 基础要求（标准理解，必须）

模型类继承 `lmms` 基类，实现 `generate_until()` 方法：

```python
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

@register_model("my_model")
class MyModel(lmms):
    def generate_until(self, requests):
        """
        输入：List[Instance]，每个 instance.args 包含：
            (context, gen_kwargs, doc_to_visual, doc_id, task, split)
        输出：List[str]，每个元素是生成的文本回答
        """
        res = []
        for request in requests:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            # 获取图像
            doc = self.task_dict[task][split][doc_id]
            images = doc_to_visual(doc)  # List[PIL.Image]
            # 生成回答
            answer = self._inference(context, images, gen_kwargs)
            res.append(answer)
        return res
```

然后在 `lmms_eval/models/__init__.py` 的 `AVAILABLE_SIMPLE_MODELS` 中注册：

```python
AVAILABLE_SIMPLE_MODELS = {
    ...
    "my_model": "MyModel",
}
```

这样就可以跑 `--tasks unig2u --model my_model` 了。

### 2. 支持 GtA（两阶段生成，可选）

如果你的模型支持图像生成能力，可以增加 GtA 支持。在 `generate_until()` 中检查 `gen_kwargs` 的 `visual_cot` 标志：

```python
def generate_until(self, requests):
    res = []
    for request in requests:
        context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

        if gen_kwargs.pop("visual_cot", False):
            # ── GtA 两阶段流程 ──
            answer = self._visual_cot_pipeline(context, doc_to_visual, doc_id, task, split)
        else:
            # ── 标准理解 ──
            answer = self._standard_inference(context, doc_to_visual, doc_id, task, split)

        res.append(answer)
    return res
```

其中 `_visual_cot_pipeline` 需要实现两个阶段：

```python
def _visual_cot_pipeline(self, context, doc_to_visual, doc_id, task, split):
    """
    Stage 1: 从 context 中解析 [GEN_PROMPT] 提示词，
             结合原始图像生成一张辅助可视化图像。
    Stage 2: 把原始图像 + 辅助图像 + 原始问题交给理解模块，生成回答。
    """
    import re

    doc = self.task_dict[task][split][doc_id]
    original_images = doc_to_visual(doc)

    # 从 prompt 中提取生成指令和问题
    gen_match = re.search(r"\[GEN_PROMPT\](.*?)\[/GEN_PROMPT\]", context, re.DOTALL)
    q_match   = re.search(r"\[QUESTION\](.*?)\[/QUESTION\]", context, re.DOTALL)

    gen_prompt = gen_match.group(1).strip() if gen_match else context
    question   = q_match.group(1).strip() if q_match else context

    # Stage 1: 生成辅助图像
    auxiliary_image = self._generate_image(gen_prompt, original_images)

    # Stage 2: 结合辅助图像回答
    all_images = original_images + [auxiliary_image]
    answer = self._understand(question, all_images)

    return answer
```

### 3. 注册后验证

```bash
# 验证标准理解
python -m lmms_eval --model my_model \
    --model_args pretrained=... \
    --tasks unig2u_chartqa100 --batch_size 1 --limit 2

# 验证 GtA（如果支持）
python -m lmms_eval --model my_model \
    --model_args pretrained=... \
    --tasks unig2u_chartqa100_visual_cot --batch_size 1 --limit 2
```

## 文件结构

```
tasks/unig2u/
├── unig2u.yaml                  # 标准理解 group（11 个子任务）
├── unig2u_GtA.yaml              # GtA group（11 个子任务的 visual_cot 版）
├── chartqa100.yaml              # 子任务 yaml（标准）
├── chartqa100_visual_cot.yaml   # 子任务 yaml（GtA，含 visual_cot: true）
├── mmsi_*.yaml                  # MMSI-Bench 系列
├── ...                          # 其他子任务 yaml
├── utils.py                     # 所有子任务的处理函数（合并）
├── arshia_utils.py              # IllusionBench 专用处理函数
└── README.md                    # 本文件
```

## 数据集

所有数据托管在 HuggingFace: `hf://datasets/kkv233/unig2u_dataset/`
