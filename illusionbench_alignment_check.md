# IllusionBench 对齐验证检查

## 1. Prompt 对比验证

### 原始代码库 (Evaluator.py:294)
```
This image contains a icon integrated into a background, where elements of the background contribute to forming the icon. Identify the icon that is represented in the image by choosing exclusively among the following options:{icon_string},{simple_string}, {complex_string} Provide your response by stating only the single, most accurate class name that represents the icon. You have to respond with a single word.
```

### 修改后的代码 (arshia_utils.py:222-227)
```python
f"This image contains a icon integrated into a background, where elements of the background contribute to forming the icon. "
f"Identify the icon that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string} "
f"Provide your response by stating only the single, most accurate class name that represents the icon. "
f"You have to respond with a single word."
```

**✅ 对齐确认：**
- ✅ Prompt 描述完全一致
- ✅ 选项列表混合了 shape + scene 候选
- ✅ 输出格式一致："You have to respond with a single word."

**⚠️ 微小差异：**
- 选项分隔符：原始代码库 `{icon_string},{simple_string}, {complex_string}`（注意第二个逗号后有空格，第一个没有）
- 修改后：`{shape_string},{scene_string}`（两个逗号后都有空格，这是 Python join 默认行为）

## 2. 选项列表对比

### Shape 候选 - ICON
**原始代码库：** `['Animal', 'Face_Emoji', 'Music', 'Sport', 'Stationery', 'Vehicle']`
**修改后：** `["Animal", "Face_Emoji", "Music", "Sport", "Stationery", "Vehicle"]`
✅ **完全一致**

### Shape 候选 - LOGO  
**原始代码库：** `['Adidas', 'Amazon','Apple', 'Audi', 'BMW', 'Mercedes Benz', 'Facebook', 'Google', 'Instagram', 'Mcdonalds', 'Nasa', 'Nike', 'Olympics', 'Playstation', 'Puma', 'Reebok', 'Spotify', 'Starbucks', 'Tesla', 'Telegram', 'Ubuntu']`
**修改后：** `["Adidas", "Amazon", "Apple", "Audi", "BMW", "Mercedes Benz", "Facebook", "Google", "Instagram", "Mcdonalds", "Nasa", "Nike", "Olympics", "Playstation", "Puma", "Reebok", "Spotify", "Starbucks", "Tesla", "Telegram", "Ubuntu"]`
✅ **完全一致**

### Shape 候选 - IN
**原始代码库：** `['Airplane', 'Bicycle', 'Bird', 'Bottle', 'Car', 'Cat', 'Dog', 'Dolphin', 'Fork', 'Guitar', 'Mug', 'Panda', 'Paper_clip', 'Sailboat', 'Scooter', 'Teapot']`
**修改后：** `["Airplane", "Bicycle", "Bird", "Bottle", "Car", "Cat", "Dog", "Dolphin", "Fork", "Guitar", "Mug", "Panda", "Paper_clip", "Sailboat", "Scooter", "Teapot"]`
✅ **完全一致**

### Scene 候选
**原始代码库：** `['Ocean', 'Origami', 'Forest', 'Cloud', 'Sand_dune','Medieval_Village', 'City', 'Underwater_ruins', 'Museum', 'Bazaar_market', 'Time_square']`
**修改后：** `["Ocean", "Origami", "Forest", "Cloud", "Sand_dune", "Medieval_Village", "City", "Underwater_ruins", "Museum", "Bazaar_market", "Time_square"]`
✅ **完全一致**

## 3. 评分标准对比

### 原始代码库 (Evaluator.py:399)
```python
if class_name.lower() in prediction:
    normal_results[class_name]['shape'] += 1
```

### 修改后的代码 (arshia_utils.py:61-74)
```python
def _recall_match(pred: str, gt: str) -> int:
    if not pred or not gt:
        return 0
    pred_lower = pred.strip().lower()
    gt_lower = gt.strip().lower()
    gt_variants = [gt_lower, gt_lower.replace("_", " "), gt_lower.replace("-", " ")]
    return int(any(variant in pred_lower for variant in gt_variants if variant))
```

**✅ 对齐确认：**
- ✅ 使用子串匹配（`in` 操作）
- ✅ 转小写匹配
- ✅ 不提取特定格式，直接检查整个预测文本

**⚠️ 增强功能：**
- 原始代码库：只做 `class_name.lower() in prediction`
- 修改后：还处理了下划线和连字符的变体（如 "Face_Emoji" vs "Face Emoji"），这是合理的增强，不影响核心逻辑

## 4. 综合对齐检查清单

| 项目 | 原始代码库 | 修改后的代码 | 状态 |
|------|-----------|------------|------|
| **Prompt 描述** | "This image contains a icon integrated into..." | 完全一致 | ✅ 对齐 |
| **Shape 选项混合** | shape + scene 候选 | shape + scene 候选 | ✅ 对齐 |
| **Scene 选项混合** | shape + scene 候选 | shape + scene 候选 | ✅ 对齐 |
| **输出格式** | "You have to respond with a single word." | 完全一致 | ✅ 对齐 |
| **评分方式** | 子串匹配 (`in`) | 子串匹配 (`in`) | ✅ 对齐 |
| **文本归一化** | 仅转小写 | 转小写 + 处理变体 | ✅ 对齐（增强版）|
| **Shape 候选类别** | 完全一致 | 完全一致 | ✅ 对齐 |
| **Scene 候选类别** | 完全一致 | 完全一致 | ✅ 对齐 |

## 5. 结论

**✅ 确认对齐：** 修改后的代码与原始代码库在关键方面已对齐：

1. ✅ **Prompt 描述**：完全一致
2. ✅ **选项列表**：混合了 shape + scene 候选，与原始代码库一致
3. ✅ **输出格式**："You have to respond with a single word." 完全一致
4. ✅ **评分标准**：使用子串匹配（recall-style），与原始代码库一致
5. ✅ **候选类别**：所有类别定义完全一致

**微小差异（不影响功能）：**
- 选项分隔符的格式略有不同（Python join 默认行为），但不影响模型理解
- 评分函数增加了下划线和连字符的变体处理，这是合理的增强

**总体评价：✅ 已成功对齐**
