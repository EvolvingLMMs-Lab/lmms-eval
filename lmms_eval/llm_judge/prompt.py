# Default prompts for different judge types
BINARY_JUDGE_PROMPT = """You are a strict evaluator assessing answer correctness. You must output {positive} for fully correct answers and {negative} for any other case.

# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{prediction}
```

# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- For multiple-choice questions: Score {positive} if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score {positive} if the prediction matches the answer semantically, it can be in different format.
  * Score {negative} for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
{positive} or {negative}"""


COMPARATIVE_JUDGE_PROMPT = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of {min_score} to {max_score}, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

[Question]
{question}

{context_section}

[Assistant 1]
{response1}
[End of Assistant 1]

[Assistant 2]
{response2}
[End of Assistant 2]

[System]
{evaluation_instruction}"""


CORRECTNESS_JUDGE_PROMPT = """You are given a question, the solution and the correct answer. Please determine if the solution matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \\boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the solution and the correct answer.
The process or reasoning leading to the Solution is irrelevant, ONLY the correctness of the result matters.
Return only "{positive}" if the solution is correct or "{negative}" if it is incorrect.
Only return "{positive}" or "{negative}" with no additional text or formatting.

Question: 
{question}
--------------------------------
Correct Answer:
{answer}
--------------------------------
Solution: 
{prediction}
--------------------------------"""
