import os
import re

from math_verify import parse, verify
from openai import OpenAI

API_KEY = os.getenv("JUDGE_API_KEY", "YOUR_API_KEY")
BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "gpt-4o-mini")
USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "False")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

JUDGE_PROMPT = """You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case.

# Input
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{prediction}
```

# Evaluation Rules
- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score 1 if the prediction matches the answer semantically, it can be in different format.
  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
1 or 0"""

JUDGE_PROMPT_WITH_ANSWER = """
You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case. You will receive the question, the ground truth answer, and the model prediction.

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
- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score 1 if the prediction matches the answer semantically, it can be in different format.
  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
1 or 0
"""


def extract_boxed_answer(predict_str: str) -> str:
    """Extract the answer from \boxed{} format.

    Args:
        predict_str (str): The prediction string containing the boxed answer.

    Returns:
        str: The extracted answer from \boxed{}, or an empty string if not found.
    """
    # Find all occurrences of \boxed{
    boxed_start = "\\boxed{"
    start_indices = []

    # Find all positions where \boxed{ starts
    pos = 0
    while True:
        pos = predict_str.find(boxed_start, pos)
        if pos == -1:
            break
        start_indices.append(pos)
        pos += 1

    if not start_indices:
        return ""

    # For each \boxed{ occurrence, find the matching closing brace
    results = []
    for start_pos in start_indices:
        brace_count = 0
        pos = start_pos + len(boxed_start) - 1  # Position at the opening brace of \boxed{

        while pos < len(predict_str):
            char = predict_str[pos]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    content_start = start_pos + len(boxed_start)
                    content = predict_str[content_start:pos]
                    results.append(content)
                    break
            pos += 1

    # Return the last (rightmost) match if multiple found
    return results[-1] if results else ""


def extract_anwser_tag(predict_str: str) -> str:
    """Extract the answer tag from the prediction string.

    This function now handles both <answer> tags and \boxed{} format.

    Args:
        predict_str (str): The prediction string containing the answer tag.

    Returns:
        str: The extracted answer tag, or an empty string if not found.
    """
    # First try to extract from <answer> tags
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match_result = re.search(pattern, predict_str)
    if match_result:
        return match_result.group(1)

    # If no <answer> tag found, try to extract from \boxed{} format
    boxed_answer = extract_boxed_answer(predict_str)
    if boxed_answer:
        return boxed_answer

    # If neither format found, try to extract the last number or expression
    # This is a fallback for cases where the answer is just stated without formatting
    lines = predict_str.strip().split("\n")
    for line in reversed(lines):
        # Look for patterns like "The answer is 204" or just "204"
        if line.strip():
            # Try to find numbers at the end of the line
            number_match = re.search(r"\b(\d+(?:\.\d+)?)\b(?:\s*\.?\s*$)", line)
            if number_match:
                return number_match.group(1)

    return ""


def format_reward(predict_str: str) -> float:
    """Check if the prediction string follows the expected format.

    Now handles both <think><answer> format and \boxed{} format.
    """
    # Check for <think>.*</think>.*<answer>.*</answer> pattern
    think_answer_pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>", re.DOTALL)
    if re.fullmatch(think_answer_pattern, predict_str):
        return 1.0

    # Check for \boxed{} format (common in mathematical solutions)
    if extract_boxed_answer(predict_str):
        return 1.0

    # Check for basic answer format (contains some mathematical content and ends with a number)
    if len(predict_str.strip()) > 50:  # Reasonable solution length
        # Look for mathematical expressions or reasoning
        has_math = bool(re.search(r"[=\+\-\*/\(\)\[\]\\]", predict_str))
        # Look for final answer
        has_answer = bool(extract_anwser_tag(predict_str))

        if has_math and has_answer:
            return 0.8  # Partial credit for reasonable format

    return 0.0


def simple_parse(predict_str: str) -> str:
    """Parse the prediction string to extract the answer.

    Args:
        predict_str (str): The prediction string to be parsed.

    Returns:
        str: The parsed answer from the prediction string.
    """
    if predict_str.endswith("."):
        predict_str = predict_str[:-1]

    return predict_str.strip()


def parse_mcq(predict_str: str) -> str:
    """
    Parse multiple choice answers from various formats.
    Handles formats like: "A", "A.", "A)", "(A)", "The answer is A", "A: xxx", etc.
    """
    if not predict_str or predict_str.strip() == "":
        return ""

    # Clean up the response
    response = predict_str.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.strip(char)

    # Add spaces to avoid partial matches
    response = " " + response + " "

    # All possible choice letters (extend if needed)
    all_choices = ["A", "B", "C", "D", "E", "F", "G", "H"]

    candidates = []

    # Pattern 1: Look for choices with parentheses e.g., (A), (B), (C), (D)
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append((choice, response.rfind(f"({choice})"), "parentheses"))

    # Pattern 2: Look for choices with periods e.g., A., B., C., D.
    for choice in all_choices:
        if f"{choice}." in response:
            candidates.append((choice, response.rfind(f"{choice}."), "period"))

    # Pattern 3: Look for choices with colons e.g., A:, B:, C:, D:
    for choice in all_choices:
        if f"{choice}:" in response:
            candidates.append((choice, response.rfind(f"{choice}:"), "colon"))

    # Pattern 4: Look for choices with right parentheses e.g., A), B), C), D)
    for choice in all_choices:
        if f"{choice})" in response:
            candidates.append((choice, response.rfind(f"{choice})"), "right_paren"))

    # Pattern 5: Look for choices with spaces after e.g., A B C D
    for choice in all_choices:
        if f"{choice} " in response:
            candidates.append((choice, response.rfind(f"{choice} "), "space"))

    # Pattern 6: Look for choices with dashes e.g., A- B- C- D-
    for choice in all_choices:
        if f"{choice}-" in response:
            candidates.append((choice, response.rfind(f"{choice}-"), "dash"))

    # Pattern 7: Look for choices with underscores e.g., A_ B_ C_ D_
    for choice in all_choices:
        if f"{choice}_" in response:
            candidates.append((choice, response.rfind(f"{choice}_"), "underscore"))

    # Pattern 8: Look for choices with equal signs e.g., A= B= C= D=
    for choice in all_choices:
        if f"{choice}=" in response:
            candidates.append((choice, response.rfind(f"{choice}="), "equals"))

    # Pattern 9: Look for common answer phrases followed by choices
    answer_phrases = [
        "the answer is",
        "answer is",
        "the correct answer is",
        "correct answer is",
        "the answer",
        "answer",
        "correct answer",
        "the correct answer",
        "the best answer is",
        "best answer is",
        "the best answer",
        "best answer",
        "the option is",
        "option is",
        "the correct option is",
        "correct option is",
        "the choice is",
        "choice is",
        "the correct choice is",
        "correct choice is",
        "i choose",
        "i select",
        "i pick",
        "my answer is",
        "my choice is",
    ]

    for phrase in answer_phrases:
        if phrase in response.lower():
            phrase_start = response.lower().find(phrase)
            # Look for choices after the phrase
            for choice in all_choices:
                choice_pos = response.find(choice, phrase_start)
                if choice_pos != -1:
                    candidates.append((choice, choice_pos, "phrase"))

    # Pattern 10: Look for choices at the very beginning of the response
    for choice in all_choices:
        if response.strip().startswith(choice):
            candidates.append((choice, 0, "start"))

    # Pattern 11: Look for choices at the very end of the response
    for choice in all_choices:
        if response.strip().endswith(choice):
            candidates.append((choice, len(response) - 1, "end"))

    # Pattern 12: Look for choices with numbers (e.g., "1. A", "2. B")
    for i, choice in enumerate(all_choices):
        if f"{i+1}. {choice}" in response:
            candidates.append((choice, response.rfind(f"{i+1}. {choice}"), "numbered"))

    # If no candidates found, try to extract from the entire response
    if not candidates:
        # Look for any choice letter in the response
        for choice in all_choices:
            if choice in response:
                candidates.append((choice, response.rfind(choice), "fallback"))

    # Return the best candidate
    if candidates:
        # Sort by position (later in text) and priority of format
        format_priority = {"start": 10, "end": 9, "numbered": 8, "phrase": 7, "parentheses": 6, "period": 5, "colon": 4, "right_paren": 3, "space": 2, "dash": 1, "underscore": 1, "equals": 1, "fallback": 0}

        # Sort by format priority first, then by position
        candidates.sort(key=lambda x: (format_priority[x[2]], -x[1]), reverse=True)
        return candidates[0][0]

    return ""


def relax_exact_match(predict_str: str, ground_truth: str, relax_portion: float = 0.9) -> float:
    """Check if the prediction string matches the ground truth exactly.

    Args:
        predict_str (str): The prediction string to be checked.
        ground_truth (str): The ground truth string for comparison.
        relax_portion (float): The minimum portion of length required for partial matches.

    Returns:
        float: 1.0 if the prediction matches the ground truth, otherwise 0.0.
    """
    # If the question is an mcq
    if ground_truth in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        predict_str = parse_mcq(predict_str)
        if predict_str == ground_truth:
            return 1.0
        return 0.0
    if predict_str in ground_truth and len(predict_str) >= relax_portion * len(ground_truth):
        return 1.0
    if ground_truth in predict_str and len(ground_truth) >= relax_portion * len(predict_str):
        return 1.0
    return 1.0 if predict_str.strip() == ground_truth.strip() else 0.0


def llm_as_judge_sync(predict_str, ground_truth, extra_info):
    if extra_info is not None and "question" in extra_info:
        prompt = JUDGE_PROMPT_WITH_ANSWER.format(question=extra_info["question"], answer=ground_truth, prediction=predict_str)
    else:
        prompt = JUDGE_PROMPT.format(answer=ground_truth, prediction=predict_str)
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        "max_tokens": 5,
        "model": MODEL_NAME,
    }
    response = client.chat.completions.create(**payload)
    try:
        score = int(response.choices[0].message.content)
    except Exception:
        score = 0
    return score


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.
        sandbox_fusion_url: Not used in this implementation.
        concurrent_semaphore: Not used in this implementation.

    Returns:
        dict: A dictionary containing the computed score and other metrics.
    """
    format_score = 0.1
    format_reward_score = format_reward(solution_str)

    extracted_answer = extract_anwser_tag(solution_str).strip()
    predict_str = simple_parse(extracted_answer)
    gt = simple_parse(ground_truth)

    acc_score = relax_exact_match(predict_str, gt)
    if acc_score == 0.0:
        try:
            gold = parse(gt)
            pred = parse(predict_str)
            acc_score = int(verify(gold, pred))
        except Exception:
            acc_score = 0.0

    if acc_score == 0.0 and USE_LLM_JUDGE == "True":
        acc_score = llm_as_judge_sync(predict_str, ground_truth, extra_info)

    # When the format reward score is 0.0, we directly judge the solution string with the ground truth
    # and the solution string is not too long
    if acc_score == 0.0 and USE_LLM_JUDGE == "True" and format_reward_score == 0.0 and len(solution_str) < 500:
        # Direct judge the solution string with the ground truth
        acc_score = llm_as_judge_sync(solution_str, ground_truth, extra_info)

    score = (1.0 - format_score) * acc_score + format_score * format_reward_score
    score_dict = {
        "score": score,
        "acc_score": acc_score,
        "format_reward_score": format_reward_score,
        "predict_str": predict_str,
        "ground_truth": gt,
    }

    return score_dict
