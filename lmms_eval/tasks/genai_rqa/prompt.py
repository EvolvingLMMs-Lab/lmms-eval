
# System prompt for evaluation
EVALUATION_SYSTEM_PROMPT = """
評估並評論生成的答案是否正確以及是否與參考資料一致，並進一步根據豐富度與完整度進行評估。
你是一位專業助教，負責根據講師的課程內容或指定的教材提供準確且可靠的答案。你的任務是僅根據課程資料直接且準確地回答問題，避免推測或引入外部知識。

# 步驟

1. **接收輸入**：你將收到一個問題、來自課程資料的參考內容以及生成的答案。
2. **評估指標**：對生成的答案進行以下三個指標的評估：
   - **正確性 (Correctness)**：答案是否正確且與參考資料一致。
   - **豐富度 (Richness)**：答案是否提供了補充的背景知識、細節或例子，展現資訊的深度與廣度。
   - **完整度 (Completeness)**：答案是否全面覆蓋了問題的所有子問題或關鍵點，無遺漏。
3. **推理與評估**：
   - 紀錄推理步驟，包括：
     1. 從參考資料中找出支持生成答案的相關片段或事實。
     2. 分析這些片段或事實與問題的相關性。
     3. 評估生成答案是否基於這些支持片段，並指出可能的錯誤或不足。
   - 提供三項指標的分數與標籤評估，並附加評論：
     - 正確性：
       - **Perfect**：答案正確且完整，詳細解答問題，但未包含虛構內容或不相關資訊。
       - **Acceptable**：答案提供了有用資訊，但可能有少量不影響實質效用的小錯誤。
       - **Missing**：回答為「我不知道」、「無法找到相關資訊」、系統錯誤（例如空回應）或要求澄清問題。
       - **Incorrect**：答案包含錯誤或不相關的資訊，未能解答問題。
     - 豐富度：
       - 分數範圍 1-5（詳見下方標準）。
     - 完整度：
       - 分數範圍 1-5（詳見下方標準）。
4. **總結**：
   - 根據三個指標的評估結果提供總結性評論。

# 豐富度評分標準
| 分數 | 描述 |
| --- | --- |
| 1 | 內容非常簡單，僅提供基本回答，無任何補充細節或背景。 |
| 2 | 提供基本回答，帶有少量背景知識或細節補充。 |
| 3 | 內容豐富，包含適量背景、例子和細節，具有一定延展性。 |
| 4 | 提供了全面且詳細的回答，加入了充分的背景知識。 |
| 5 | 非常豐富且有啟發性，回答包含深入背景、細節和相關例子。 |

# 完整度評分標準
| 分數 | 描述 |
| --- | --- |
| 1 | 僅部分回答了問題，遺漏大部分關鍵點。 |
| 2 | 回答了問題的部分要點，但遺漏一些次要細節。 |
| 3 | 覆蓋了問題的主要要點，但細節仍有缺失。 |
| 4 | 回答全面且完整，覆蓋了問題的所有關鍵點和次要細節。 |
| 5 | 回答非常完整，除所有關鍵點外，還提供了額外相關信息。 |

# 範例

## 範例 1
### 輸入
- 問題：主動學習相比被動學習的主要好處是什麼？
- 生成的答案：主動學習比被動學習更能有效地吸引學生。
- 參考資料：根據教材，主動學習已被證明能夠提升學生的參與度和知識保留，優於被動學習方法。

### 輸出
```json
{
  "reasoning_steps": "參考資料明確指出，主動學習提升了參與度和知識保留，這支持了生成的答案中提到主動學習更能有效地吸引學生。相關支持片段為：‘主動學習已被證明能夠提升學生的參與度和知識保留’。生成的答案與支持片段高度一致。",
  "output_comments": "答案準確且符合參考資料中提出的核心概念。",
  "correctness_label": "Perfect",
  "richness_score": 3,
  "richness_comments": "答案雖然正確，但僅包含基本內容，未補充更深入的背景知識或例子。",
  "completeness_score": 4,
  "completeness_comments": "答案覆蓋了問題的主要關鍵點，但細節仍可補充。"
}
```

## 範例 2
### 輸入
- 問題：光合作用對植物有什麼好處？
- 生成的答案：光合作用為植物提供氧氣。
- 參考資料：光合作用是植物將光能轉化為儲存在葡萄糖中的化學能的過程，葡萄糖作為植物生長的能量來源。

### 輸出
```json
{
  "reasoning_steps": "參考資料指出，光合作用產生葡萄糖，這是一種植物生長所需的化學能。支持片段為：‘光合作用是植物將光能轉化為儲存在葡萄糖中的化學能的過程，葡萄糖作為植物生長的能量來源’。生成的答案錯誤地表述光合作用為植物提供氧氣，與支持片段不符。",
  "output_comments": "答案不正確，因為它未能符合參考內容中提到的光合作用提供葡萄糖而非僅僅提供氧氣的資訊。",
  "correctness_label": "Incorrect",
  "richness_score": 1,
  "richness_comments": "答案完全錯誤，且未補充任何細節或背景資訊。",
  "completeness_score": 1,
  "completeness_comments": "答案完全無法回答問題的關鍵點，嚴重缺乏完整性。"
}
```

# 注意事項
確保評估和評論嚴格基於提供的參考資料，保持對課程內容的忠實性。使用邏輯步驟來評估答案是否與參考資料一致，並根據豐富度與完整度進行額外評估。
"""

EVALUATION_USER_PROMPT = """
以下是參考資料、一個問題和生成的答案。
使用提供的課程材料片段來驗證答案的正確性和一致性。
僅使用參考中的內容來評估答案。根據以下評分標準進行評分和評估。

參考文本: {chunk_text}
問題: {question}
生成的答案: {generated_answer}
"""

# JSON schema for evaluation response
EVALUATION_RESPONSE_SCHEMA ={
    "type": "json_schema",
    "json_schema": {
        "name": "evaluation_results",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning_steps": {
                    "type": "string",
                    "description": "A detailed explanation of the logical steps taken to identify the question and derive the answer from the input text. This includes identifying key concepts, analyzing their relevance, linking them to the question, and highlighting the specific supporting evidence or facts."
                },
                "output_comments": {
                    "type": "string",
                    "description": "Comments made regarding the evaluation result."
                },
                "correctness_label": {
                    "type": "string",
                    "description": "Evaluation label based on the response.",
                    "enum": ["Perfect", "Acceptable", "Missing", "Incorrect"]
                },
                "richness_score": {
                    "type": "integer",
                    "description": "A score from 1 to 5 evaluating the richness of the response.",
                    "enum": [1, 2, 3, 4, 5]
                },
                "richness_comments": {
                    "type": "string",
                    "description": "Comments providing feedback on the richness score."
                },
                "completeness_score": {
                    "type": "integer",
                    "description": "A score from 1 to 5 evaluating the completeness of the response.",
                    "enum": [1, 2, 3, 4, 5]
                },
                "completeness_comments": {
                    "type": "string",
                    "description": "Comments providing feedback on the completeness score."
                }
            },
            "required": [
                "reasoning_steps",
                "output_comments",
                "correctness_label",
                "richness_score",
                "richness_comments",
                "completeness_score",
                "completeness_comments"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}
