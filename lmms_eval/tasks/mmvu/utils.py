import os
import re
import string
import sys
from pathlib import Path

import yaml
from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")

base_cache_dir = os.path.expanduser(hf_home)


with open(Path(__file__).parent / "mmvu_val.yaml", "r") as f:
    raw_data_val = f.readlines()
    safe_data_val = []
    for i, line in enumerate(raw_data_val):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_val.append(line)
cache_name_val = yaml.safe_load("".join(safe_data_val))["dataset_kwargs"]["cache_dir"]
cache_dir_val = os.path.join(base_cache_dir, cache_name_val)

# Initialize the LLM judge server (lazy initialization)
_server = None


def get_llm_judge_server():
    """Lazy initialization of LLM judge server"""
    global _server
    if _server is None:
        API_TYPE = os.getenv("API_TYPE", "openai")
        MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")
        server_config = ServerConfig(model_name=MODEL_VERSION)
        _server = get_server(server_name=API_TYPE, config=server_config)
    return _server


def mmvu_doc_to_visual_val(doc):
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir_val, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


multiple_choice_prompt = """
Question:{question}
A: {a}
B: {b}
C: {c}
D: {d}
E: {e}
Visual Information: processed video
Do not generate any intermediate reasoning process. Answer directly with the option letter from the
given choices.
"""

open_ended_prompt = """
Question:{question}
Visual Information: processed video
Do not generate any intermediate reasoning process. Directly output the final answer.
"""

multiple_choice_prompt_cot = """
Question:{question}
A: {a}
B: {b}
C: {c}
D: {d}
E: {e}
Visual Information: processed video
Answer the given multiple-choice question step by step. Begin by explaining your reasoning process
clearly. Conclude by stating the final answer using the following format: "Therefore, the final answer
is: $LETTER" (without quotes), where $LETTER is one of the options. Think step by step before
answering.
"""

open_ended_prompt_cot = """
Question:{question}
Visual Information: processed video
Answer the given question step by step. Begin by explaining your reasoning process clearly. Conclude
by stating the final answer using the following format: "Therefore, the final answer is: "Answer:
$ANSWER" (without quotes), where $ANSWER is the final answer of the question. Think step by
step before answering.
"""


def mmvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question_type = doc["question_type"]
    if question_type == "multiple-choice":
        question = doc["question"]
        choices = doc["choices"]
        full_prompt = multiple_choice_prompt.format(question=question, a=choices["A"], b=choices["B"], c=choices["C"], d=choices["D"], e=choices["E"])
    else:
        question = doc["question"]
        full_prompt = open_ended_prompt.format(question=question)
    return full_prompt


def mmvu_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    question_type = doc["question_type"]
    if question_type == "multiple-choice":
        question = doc["question"]
        choices = doc["choices"]
        full_prompt = multiple_choice_prompt_cot.format(question=question, a=choices["A"], b=choices["B"], c=choices["C"], d=choices["D"], e=choices["E"])
    else:
        question = doc["question"]
        full_prompt = open_ended_prompt_cot.format(question=question)
    return full_prompt


def construct_question_prompt(doc):
    """Construct the question prompt for evaluation"""
    question_type = doc["question_type"]
    if question_type == "multiple-choice":
        choices = doc["choices"]
        return f"""Question: {doc["question"]}
A: {choices["A"]}
B: {choices["B"]}
C: {choices["C"]}
D: {choices["D"]}
E: {choices["E"]}"""
    else:
        return f"Question: {doc['question']}"


def normalize_math_notation(text):
    """Normalize mathematical notation for comparison (e.g., n² -> n^2, n³ -> n^3)"""
    # Convert superscript numbers to caret notation
    superscript_map = {"²": "^2", "³": "^3", "¹": "^1", "⁰": "^0", "⁴": "^4", "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9"}
    normalized = text
    for sup, caret in superscript_map.items():
        normalized = normalized.replace(sup, caret)
    return normalized


def evaluate_with_rule_based(doc, prediction):
    """Rule-based evaluation - returns True if correct, False otherwise"""
    answer = doc["answer"]
    question_type = doc["question_type"]

    if question_type == "multiple-choice":
        # Rule-based evaluation for multiple-choice questions
        pred_str = str(prediction).strip()
        answer_str = str(answer).strip()

        # Method 1: Extract letter from prediction and compare
        letter_match = re.search(r"\b([A-E])\b", pred_str, re.IGNORECASE)
        if letter_match:
            extracted_letter = letter_match.group(1).upper()
            if extracted_letter == answer_str.upper():
                return True

        # Method 2: Check if answer letter appears in prediction (case-insensitive)
        if answer_str.upper() in pred_str.upper():
            # Make sure it's a standalone letter, not part of another word
            if re.search(rf"\b{re.escape(answer_str)}\b", pred_str, re.IGNORECASE):
                return True

        # Method 3: Check if the full answer text (letter + content) matches
        if answer_str in doc.get("choices", {}):
            choice_text = doc["choices"][answer_str].strip().lower()
            pred_lower = pred_str.lower()
            # Check if choice text appears in prediction
            if choice_text in pred_lower:
                return True
            # Check if prediction contains both the letter and key words from choice
            if answer_str.upper() in pred_str.upper():
                # Extract key words from choice (remove common words)
                words = [w.strip(string.punctuation) for w in choice_text.split() if len(w.strip(string.punctuation)) > 2]
                if len(words) > 0:
                    # Check if at least one key word appears in prediction
                    if any(word in pred_lower for word in words):
                        return True

        return False
    else:
        # Rule-based evaluation for open-ended questions
        pred_normalized = str(prediction).strip().lower()
        answer_normalized = str(answer).strip().lower()

        # Normalize mathematical notation (e.g., n² -> n^2)
        pred_normalized = normalize_math_notation(pred_normalized)
        answer_normalized = normalize_math_notation(answer_normalized)

        # Remove common punctuation and extra whitespace for comparison
        pred_clean = " ".join(pred_normalized.split())
        answer_clean = " ".join(answer_normalized.split())

        # Method 1: Exact match (after normalization)
        if pred_clean == answer_clean:
            return True

        # Method 2: Check if answer appears as a substring in prediction
        # This handles cases like: answer="Depth-First Search (DFS)", prediction="The algorithm is depth-first search (DFS)."
        if answer_clean in pred_clean:
            return True

        # Method 3: Check if prediction appears as a substring in answer (for shorter predictions)
        if pred_clean in answer_clean:
            return True

        # Method 4: For numerical answers, try to extract and compare numbers
        # Extract numbers from both strings
        pred_numbers = re.findall(r"\d+\.?\d*", pred_normalized)
        answer_numbers = re.findall(r"\d+\.?\d*", answer_normalized)

        if len(answer_numbers) > 0 and len(pred_numbers) > 0:
            # Try to match numbers (allowing for floating point differences)
            try:
                answer_num = float(answer_numbers[0])
                pred_num = float(pred_numbers[0])
                # Allow small floating point differences (0.01 tolerance)
                if abs(answer_num - pred_num) < 0.01:
                    return True
            except ValueError:
                pass

        # Method 5: Word-level matching for short answers (2-5 words)
        answer_words = [w.strip(string.punctuation) for w in answer_clean.split() if w.strip(string.punctuation)]
        pred_words = [w.strip(string.punctuation) for w in pred_clean.split() if w.strip(string.punctuation)]

        if 2 <= len(answer_words) <= 5:
            # Check if all answer words appear in prediction (order-independent)
            if all(word in pred_words for word in answer_words if len(word) > 2):
                return True

        # Method 6: Special handling for mathematical complexity notation (O(n²) vs O(n^2))
        # Extract O() notation patterns
        pred_o_match = re.search(r"O\s*\([^)]+\)", pred_clean, re.IGNORECASE)
        answer_o_match = re.search(r"O\s*\([^)]+\)", answer_clean, re.IGNORECASE)
        if pred_o_match and answer_o_match:
            pred_o_content = normalize_math_notation(pred_o_match.group(0).lower())
            answer_o_content = normalize_math_notation(answer_o_match.group(0).lower())
            if pred_o_content == answer_o_content:
                return True

        return False


def evaluate_with_llm_judge(doc, prediction):
    """
    Hybrid evaluation: first try rule-based, if rule-based returns False, use GPT judge only for open-ended questions.
    For multiple-choice questions, only use rule-based evaluation.
    Returns: (is_correct, evaluation_method) where evaluation_method is "rule-based" or "gpt-based"
    """
    # First try rule-based evaluation
    rule_correct = evaluate_with_rule_based(doc, prediction)

    # If rule-based says it's correct, return immediately
    if rule_correct:
        return True, "rule-based"

    # Get question type
    question_type = doc["question_type"]

    # For multiple-choice questions, only use rule-based evaluation
    # Don't use GPT judge for multiple-choice questions
    if question_type == "multiple-choice":
        return False, "rule-based"

    # For open-ended questions, if rule-based says it's incorrect, try GPT judge for semantic equivalence
    # This handles cases like "O(n²)" vs "O(n^2)" where rule-based might be too strict
    try:
        server = get_llm_judge_server()
        formatted_question = construct_question_prompt(doc)
        answer = doc["answer"]

        full_answer = str(answer)

        custom_prompt = """You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case.

# Evaluation Rules for Open-Ended Questions
- The model prediction may contain reasoning, focus on extracting the final answer.
- Score 1 if the prediction matches the answer semantically, even if in different format.
- For mathematical notation, treat equivalent forms as correct (e.g., O(n²) = O(n^2), n² = n^2).
- Score 0 for partially correct answers or answers with extra incorrect information.
- Ignore minor differences in formatting, capitalization, or spacing.
- Treat numerical answers as correct if they match within reasonable precision.
- For questions requiring units, both value and unit must be correct.

Return only "1" or "0" with no additional text or formatting."""

        result = server.evaluate_binary(question=formatted_question, answer=full_answer, prediction=prediction, output_format="0/1", custom_prompt=custom_prompt)

        if result["success"]:
            judge_response = result["result"]
            judge_score = str(judge_response).strip()
            is_correct = judge_score == "1"
            return is_correct, "gpt-based"
        else:
            eval_logger.error(f"GPT judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
            # Fall back to rule-based result if GPT fails
            return rule_correct, "rule-based"

    except Exception as e:
        eval_logger.error(f"Error getting GPT judge response: {e}")
        # Fall back to rule-based result if GPT fails
        return rule_correct, "rule-based"


def extract_category(doc):
    category = doc["video_path"].split("/")[-2]
    return category


def mmvu_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    # Handle the case where results[0] might be a list or a string
    pred = results[0]
    if isinstance(pred, list):
        pred_ans = pred[0] if pred else ""
    else:
        pred_ans = pred

    # Ensure pred_ans is a string
    pred_ans = str(pred_ans)

    category = extract_category(doc)

    # Use hybrid evaluation: rule-based first, then GPT if needed
    correct, eval_method = evaluate_with_llm_judge(doc, pred_ans)

    # Extract predicted answer for logging (best effort)
    if doc["question_type"] == "multiple-choice":
        # Try to extract the letter choice from the prediction
        letter_match = re.search(r"\b([A-E])\b", pred_ans)
        extracted_answer = letter_match.group(1) if letter_match else "N/A"
    else:
        # For open-ended, just use the prediction as-is (truncated for logging)
        extracted_answer = pred_ans[:100] + "..." if len(pred_ans) > 100 else pred_ans

    data_dict = {"question_id": doc["id"], "category": category, "pred_answer": extracted_answer, "answer": doc["answer"], "correct": int(correct), "eval_method": eval_method}  # "rule-based" or "gpt-based"

    return {f"accuracy": data_dict}


def mmvu_aggregate_results_val(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """

    TASK_MAP = {
        "Biology": "Science",
        "Chemistry": "Science",
        "Modern_Physics": "Science",
        "Astronomy": "Science",
        "Geography": "Science",
        "Materials_Science": "Science",
        "Neurobiology": "Science",
        "Electromagnetism": "Science",
        "Thermodynamics": "Science",
        "Mechanics": "Science",
        "Civil_Engineering": "Engineering",
        "Electrical_Engineering": "Engineering",
        "Mechanical_Engineering": "Engineering",
        "Biomedical_Engineering": "Engineering",
        "Electronics_and_Communication": "Engineering",
        "Computer_Science": "Engineering",
        "Clinical_Medicine": "Healthcare",
        "Basic_Medicine": "Healthcare",
        "Preventive_Medicine": "Healthcare",
        "Pharmacy": "Healthcare",
        "Dentistry": "Healthcare",
        "Art": "Humanities_and_Social_Science",
        "Literature": "Humanities_and_Social_Science",
        "History": "Humanities_and_Social_Science",
        "Law": "Humanities_and_Social_Science",
        "Economics": "Humanities_and_Social_Science",
        "Management": "Humanities_and_Social_Science",
    }

    TASK_TYPES = list(set(TASK_MAP.values()))

    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        category = result["category"]
        if category in TASK_MAP:
            category = TASK_MAP[category]
            category2score[category]["answered"] += 1
            category2score[category]["correct"] += result.get("correct", False)
    category_scores = {}

    for category in TASK_TYPES:
        total_correct = category2score[category]["correct"]
        total_answered = category2score[category]["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        category_scores[category] = accuracy

    total_correct = sum(category2score[category]["correct"] for category in TASK_TYPES)
    total_answered = sum(category2score[category]["answered"] for category in TASK_TYPES)
    accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
    eval_logger.info("=" * 50)
    eval_logger.info(f"Average Accuracy: {accuracy:.2f}%")
    eval_logger.info("Categorical accuracy: ")
    for key, value in category_scores.items():
        eval_logger.info(f"{key} accuracy: {value:.2f}%")
    eval_logger.info("=" * 50)
    return accuracy
