# IllusionBench Prompt 和评分标准对比

## 对比总结

经过对比 `illusionbench/IllusionBench_codebase` 和 `lmms_eval/tasks/illusionbench` 两个实现，发现 **Prompt 和评分标准存在显著差异**。

---

## 1. Prompt 对比

### 1.1 原始代码库 Prompt（Evaluator.py）

#### Shape 任务 Prompt：

```
This image contains a icon integrated into a background, where elements of the background contribute to forming the icon.
Identify the icon that is represented in the image by choosing exclusively among the following options:{icon_string},{simple_string}, {complex_string}
Provide your response by stating only the single, most accurate class name that represents the icon.
You have to respond with a single word.
```

**特点：**
- 描述更详细："icon integrated into a background, where elements of the background contribute to forming the icon"
- 选项列表中同时包含 shape 候选和 scene 候选（简单+复杂场景）
- 要求："respond with a single word"

#### Scene 任务 Prompt：

```
This image contains an icon integrated into a background, where elements of the background contribute to forming the icon.
Identify the background that is represented in the image by choosing exclusively among the following options:{icon_string},{simple_string}, {complex_string}.
Provide your response by stating only the single, most accurate class name that represents the background.
You have to respond with a single word.
```

**特点：**
- 同样描述详细
- 选项列表同样混合了 shape 和 scene 候选

---

### 1.2 lmms-eval 实现 Prompt（arshia_utils.py）

#### Shape 任务 Prompt：

```python
def _build_shape_prompt(candidates: List[str]) -> str:
    options = ", ".join(candidates)
    return (
        "You are given an image where scene elements form an abstract SHAPE.\n"
        "Task: Identify what shape is hidden in this image.\n\n"
        f"Options: [{options}]\n\n"
        "Reply in this exact format:\nAnswer: <your choice>\n"
    )
```

**实际输出示例（ICON）：**
```
You are given an image where scene elements form an abstract SHAPE.
Task: Identify what shape is hidden in this image.

Options: [animal, vehicle, stationary, sport, music, face_emoji]

Reply in this exact format:
Answer: <your choice>
```

**特点：**
- 描述更简洁："scene elements form an abstract SHAPE"
- 选项列表**只包含 shape 候选**（不包含 scene 候选）
- 要求特定格式："Answer: <your choice>"

#### Scene 任务 Prompt：

```python
def _build_scene_prompt(candidates: List[str]) -> str:
    options = ", ".join(candidates)
    return (
        "You are given an image depicting a SCENE.\n"
        "Task: Identify what scene is shown in this image.\n\n"
        f"Options: [{options}]\n\n"
        "Reply in this exact format:\nAnswer: <your choice>\n"
    )
```

**特点：**
- 描述："depicting a SCENE"
- 选项列表**只包含 scene 候选**

---

### 1.3 Prompt 差异总结

| 维度 | 原始代码库 | lmms-eval 实现 | 是否一致 |
|------|-----------|---------------|---------|
| **描述语言** | "icon integrated into a background, where elements contribute to forming the icon" | "scene elements form an abstract SHAPE" / "depicting a SCENE" | ❌ **不一致** |
| **Shape 选项** | 包含 shape + scene 候选（混合） | 只包含 shape 候选 | ❌ **不一致** |
| **Scene 选项** | 包含 shape + scene 候选（混合） | 只包含 scene 候选 | ❌ **不一致** |
| **输出格式要求** | "respond with a single word" | "Answer: <your choice>" | ❌ **不一致** |
| **任务描述** | 强调 "background contribute to forming" | 强调 "hidden shape" / "scene" | ❌ **不一致** |

**关键差异：**
1. 原始代码库的选项列表**混合了 shape 和 scene 候选**，增加了任务难度（需要从混合列表中区分）
2. lmms-eval 实现使用**分离的选项列表**，降低了任务难度
3. 描述语言完全不同，可能导致模型理解偏差

---

## 2. 评分标准对比

### 2.1 原始代码库评分方式（Evaluator.py:399-404）

```python
if class_name.lower() in prediction:
    normal_results[class_name]['shape'] += 1
elif neg_class.lower() in prediction or neg_class_white_space in prediction:
    normal_results[class_name]['texture'] += 1
else:
    normal_results[class_name]['rest'] += 1
```

**特点：**
- 使用**子串匹配**：`class_name.lower() in prediction`
- **Recall-style 评分**：只要预测中包含正确答案就算正确
- 不进行文本归一化（只转小写）
- 不提取特定格式，直接检查整个预测文本

### 2.2 lmms-eval 实现评分方式（arshia_utils.py）

#### Shape 任务评分：

```python
def illusionbench_arshia_process_results_shape(doc, results):
    pred = str(results[0]) if results else ""
    answer = _extract_answer(pred)  # 提取 "Answer: XX" 格式
    return {
        "shape_recall": _strict_match(answer, doc.get("shape_gt", "")),
    }

def _extract_answer(pred: str) -> str:
    """提取 Answer: XX 格式"""
    match = re.search(r"answer\s*:\s*([^\n,\.]+)", pred, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def _strict_match(pred: str, gt: str) -> int:
    """严格匹配：提取的答案必须等于 ground truth（归一化后）"""
    p = _normalize_text(pred)
    g = _normalize_text(gt)
    if not g:
        return 0
    return int(p == g)

def _normalize_text(s: str) -> str:
    """文本归一化：转小写、统一分隔符、去除标点"""
    s = (s or "").strip().lower()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
```

**特点：**
- 使用**格式提取**：先提取 "Answer: XX" 部分
- **严格匹配**：提取的答案必须完全等于 ground truth（归一化后）
- 进行**文本归一化**：转小写、统一分隔符、去除标点
- 如果格式不匹配，返回空字符串，得分为 0

#### Scene 任务评分：

```python
def illusionbench_arshia_process_results_scene(doc, results):
    pred = str(results[0]) if results else ""
    answer = _extract_answer(pred)
    return {
        "scene_recall": _strict_match(answer, doc.get("scene_gt", "")),
    }
```

**与 Shape 任务相同，使用严格匹配。**

### 2.3 评分标准差异总结

| 维度 | 原始代码库 | lmms-eval 实现 | 是否一致 |
|------|-----------|---------------|---------|
| **匹配方式** | 子串包含检查 (`in`) | 严格相等匹配 (`==`) | ❌ **不一致** |
| **文本提取** | 直接使用整个预测 | 提取 "Answer: XX" 格式 | ❌ **不一致** |
| **文本归一化** | 仅转小写 | 转小写 + 统一分隔符 + 去除标点 | ❌ **不一致** |
| **容错性** | 高（只要包含就算） | 低（必须精确匹配） | ❌ **不一致** |
| **格式要求** | 无特定格式要求 | 要求 "Answer: XX" 格式 | ❌ **不一致** |

**关键差异：**
1. **原始代码库**：Recall-style，容错性高，只要预测中包含正确答案就算正确
2. **lmms-eval**：严格匹配，容错性低，必须精确匹配且符合格式要求
3. **评分严格程度**：lmms-eval 更严格，可能导致评分低于原始代码库

---

## 3. 候选类别对比

### 3.1 Shape 候选

#### 原始代码库（Evaluator.py）：
- **ICON**: `['Animal', 'Face_Emoji', 'Music', 'Sport', 'Stationery', 'Vehicle']`
- **LOGO**: `['Adidas', 'Amazon','Apple', 'Audi', 'BMW', 'Mercedes Benz', 'Facebook', 'Google', 'Instagram', 'Mcdonalds', 'Nasa', 'Nike', 'Olympics', 'Playstation', 'Puma', 'Reebok', 'Spotify', 'Starbucks', 'Tesla', 'Telegram', 'Ubuntu']`
- **IN (sin)**: `['Airplane', 'Bicycle', 'Bird', 'Bottle', 'Car', 'Cat', 'Dog', 'Dolphin', 'Fork', 'Guitar', 'Mug', 'Panda', 'Paper_clip', 'Sailboat', 'Scooter', 'Teapot']`

#### lmms-eval 实现（arshia_utils.py）：
- **ICON**: `["animal", "vehicle", "stationary", "sport", "music", "face_emoji"]`
  - 注意：`Stationery` → `stationary`（可能是拼写错误）
- **LOGO**: `["tesla", "starbucks", "mcdonalds", "adidas", "reebok", "bmw", "ubuntu", "benz", "telegram", "nike", "apple", "puma", "facebook", "playstation", "instagram", "audi", "olympics", "google", "spotify", "amazon", "nasa"]`
  - 注意：`Mercedes Benz` → `benz`，且顺序不同
- **IN**: `["guitar", "teapot", "cat", "paper_clip", "bird", "dolphin", "mug", "bicycle", "bottle", "panda", "dog", "sailboat", "car", "fork", "scooter", "airplane"]`
  - 注意：`Airplane` → `airplane`（大小写），顺序不同

### 3.2 Scene 候选

#### 原始代码库和 lmms-eval 实现：

两者都使用相同的 11 个场景：
- `["Underwater_ruins", "Time_square", "Medieval_Village", "City", "Museum", "Cloud", "Ocean", "Sand_dune", "Bazaar_market", "Forest", "Origami"]`

✅ **Scene 候选一致**

---

## 4. 综合对比结论

### 4.1 一致性检查结果

| 项目 | 原始代码库 | lmms-eval 实现 | 一致性 |
|------|-----------|---------------|--------|
| **Shape Prompt 描述** | "icon integrated into background..." | "scene elements form abstract SHAPE" | ❌ 不一致 |
| **Scene Prompt 描述** | "icon integrated into background..." | "depicting a SCENE" | ❌ 不一致 |
| **Shape 选项列表** | shape + scene 混合 | 仅 shape | ❌ 不一致 |
| **Scene 选项列表** | shape + scene 混合 | 仅 scene | ❌ 不一致 |
| **输出格式** | "single word" | "Answer: <choice>" | ❌ 不一致 |
| **评分方式** | 子串包含（recall） | 严格匹配（exact） | ❌ 不一致 |
| **文本归一化** | 仅小写 | 小写+分隔符+标点 | ❌ 不一致 |
| **Shape 候选类别** | 部分差异（拼写、顺序） | 略有不同 | ⚠️ 基本一致 |
| **Scene 候选类别** | 相同 | 相同 | ✅ 一致 |

### 4.2 主要问题

1. **Prompt 不一致**：描述语言、选项列表内容、格式要求都不同
2. **评分标准不一致**：原始代码库使用 recall-style（宽松），lmms-eval 使用严格匹配（严格）
3. **任务难度差异**：原始代码库选项混合了 shape 和 scene，难度更高

### 4.3 影响

- **评分结果可能不可比**：由于评分标准不同，直接对比两个实现的分数可能没有意义
- **模型表现可能不同**：不同的 prompt 可能导致模型理解偏差，产生不同的输出
- **需要统一标准**：如果要复现原始论文结果，需要对齐 prompt 和评分标准

---

## 5. 建议

1. **统一 Prompt**：决定使用哪个版本的 prompt，或创建一个与原始论文对齐的版本
2. **统一评分标准**：建议使用原始代码库的 recall-style 评分，或者同时支持两种评分方式
3. **文档说明**：在 README 中明确说明与原始实现的差异，避免混淆

---

## 6. 参考文件

- **原始代码库 Prompt**: `illusionbench/IllusionBench_codebase/Zero-shot_experiments/Evaluator.py:285-313`
- **lmms-eval Prompt**: `lmms_eval/tasks/illusionbench/arshia_utils.py:198-232`
- **原始代码库评分**: `illusionbench/IllusionBench_codebase/Zero-shot_experiments/Evaluator.py:399-404`
- **lmms-eval 评分**: `lmms_eval/tasks/illusionbench/arshia_utils.py:267-280`
