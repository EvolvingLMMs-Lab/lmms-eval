# Visual CoT 对齐检查

## 问题分析

原始 IllusionBench 代码库**没有 Visual CoT 实现**，只有：
- Zero-shot 评估
- In-context learning (ICL) 评估

**Visual CoT 是 lmms-eval 框架扩展的功能**，使用两阶段推理：
1. **Stage 1 (GEN_PROMPT)**: 生成辅助图像
2. **Stage 2 (QUESTION)**: 结合原图和辅助图回答问题

## 当前实现检查

### Stage 2 Question Prompt 应该与直接生成版本对齐

**直接生成版本的 Prompt (已对齐原始代码库):**
```
This image contains a icon integrated into a background, where elements of the background contribute to forming the icon. 
Identify the icon that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string} 
Provide your response by stating only the single, most accurate class name that represents the icon. 
You have to respond with a single word.
```

**当前 Visual CoT Stage 2 Prompt:**
```python
f"This image contains a icon integrated into a background, where elements of the background contribute to forming the icon. "
f"Identify the icon that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string} "
f"Provide your response by stating only the single, most accurate class name that represents the icon. "
f"You have to respond with a single word."
```

✅ **已对齐！** Visual CoT 的 Stage 2 prompt 与直接生成版本完全一致。

## 对齐确认

| 项目 | 直接生成版本 | Visual CoT Stage 2 | 状态 |
|------|------------|-------------------|------|
| **Prompt 描述** | "This image contains a icon..." | 完全一致 | ✅ 对齐 |
| **选项列表** | shape + scene 混合 | shape + scene 混合 | ✅ 对齐 |
| **输出格式** | "You have to respond with a single word." | 完全一致 | ✅ 对齐 |
| **评分标准** | 子串匹配 (recall) | 子串匹配 (recall) | ✅ 对齐 |

## 关键点

1. ✅ **Stage 2 Prompt 已对齐**: 与直接生成版本完全一致，使用原始代码库的格式
2. ✅ **选项列表正确**: 混合了 shape + scene 候选
3. ✅ **输出格式正确**: "You have to respond with a single word."
4. ✅ **评分标准正确**: 使用 `_recall_match` (子串匹配)

**结论**: Visual CoT 的 Stage 2 已正确对齐。Stage 1 的 Generation Prompt 是 lmms-eval 的设计，没有原始代码库对应，这是合理的。
