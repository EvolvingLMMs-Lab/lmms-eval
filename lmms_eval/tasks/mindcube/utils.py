import re
from typing import Optional


def mindcube_doc_to_text(doc):
    """Extracts the text prompt from a MindCube dataset sample.

    Args:
        doc (dict): A MindCube dataset sample dictionary.

    Returns:
        str: The input prompt text.
    """
    return doc["input_prompt"]


def mindcube_doc_to_visual(doc):
    """Extracts and converts visual data from a MindCube dataset sample.

    This function iterates over the 'images' key in the dataset sample, calling
    the .convert("RGB") method on each item (presumed to be a PIL/Pillow Image).

    Args:
        doc (dict): A MindCube dataset sample dictionary.

    Returns:
        list: A list of visual elements (e.g., PIL Images) converted to 'RGB' format.
    """
    return [visual.convert("RGB") for visual in doc["images"]]


# This is taken directly from the official mindcube codebase
def extract_answer(text: str) -> Optional[str]:
    """
    Extract the answer from model response text using regular expressions.
    Returns the last occurrence of the letter of the answer (A, B, C, D, or E)
    based on pattern priority - tries higher priority patterns first.

    Args:
        text: The model response text

    Returns:
        The last answer letter found by the highest priority matching pattern,
        or None if not found
    """
    if not text:
        return None

    # First, try to match simple answer format: A., B., C., D., E. with highest priority
    simple_pattern_matches = list(re.finditer(r"([A-E])\.", text))
    if simple_pattern_matches:
        return simple_pattern_matches[-1].group(1)

    # Then check if <Answer> tag exists and extract content after it
    answer_section_match = re.search(r"<Answer>(.*?)(?:<|$)", text, re.DOTALL)
    if answer_section_match:
        answer_section = answer_section_match.group(1)
        # Check for specific patterns in the answer section
        for pattern in [r"[Mm]y answer is ([A-E])", r"[Mm]y answer is ([A-E])\.", r"[Tt]he answer is ([A-E])", r"(?:Answer: )?([A-E])\.", r"\b([A-E])\b"]:
            matches = list(re.finditer(pattern, answer_section))
            if matches:
                return matches[-1].group(1)

    # If no matches found after <Answer> tag, proceed with regular priority patterns
    patterns = [
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'",]+(?=(?:\n|$|\.|"))',  # Full answer with description
        r'(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'"]+',  # Answer with partial description
        r"(?:^|\n)(?:Answer: )?([A-E])(?:\.|$|\s)",  # Answer at line beginning
        r"[\*\"]([A-E])[\*\"]",  # Answer in quotes or asterisks
        r"\bAnswer:?\s*([A-E])\b",  # Answer following "Answer:"
        r"[Mm]y answer is ([A-E])",  # Added pattern for "My answer is X"
        r"[Mm]y answer is ([A-E])\.",  # Added pattern for "My answer is X."
        r"answer is ([A-E])",  # Added pattern for phrases like "The answer is X"
    ]

    # Try each pattern in order of priority
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            # Return the last match found by this pattern
            return matches[-1].group(1)

    # If none of the priority patterns match, try line-by-line parsing
    # First, try the more specific pattern on each line
    lines = text.split("\n")
    line_matches = []

    for i, line in enumerate(lines):
        # Look for full answer pattern in each line
        match = re.search(r'([A-E])\. [A-Za-z0-9 \-\(\)\'",]+', line)
        if match:
            line_matches.append((i, match.group(1)))

    if line_matches:
        # Return the answer from the last line that matched
        return line_matches[-1][1]

    # Finally, try the most general pattern on each line
    for i in reversed(range(len(lines))):  # Start from bottom
        line = lines[i]
        match = re.search(r"\b([A-E])\b", line)
        if match:
            return match.group(1)

    return None  # No answer found


def mindcube_process_results(doc, results):
    """Processes the model's output for a single MindCube document.

    Compares the predicted answer (extracted from the model's text result)
    with the ground truth answer from the document. It also determines the
    question type based on the document ID for category-specific aggregation.

    Args:
        doc (dict): The original MindCube document, containing 'gt_answer'
                    and 'id'.
        results (list[str]): A list of model output strings. Only the first
                             element (results[0]) is used.

    Returns:
        dict: A dictionary structured for aggregation, containing:
            - 'overall_accuracy': {'score': 1.0 or 0.0}
            - 'around_accuracy': {'score': 1.0 or 0.0, 'type': str}
            - 'among_accuracy': {'score': 1.0 or 0.0, 'type': str}
            - 'rotation_accuracy': {'score': 1.0 or 0.0, 'type': str}
    """
    # extract grounded answer
    grounded_output = doc["gt_answer"]

    # extract predicted answer
    pred = results[0].strip()
    pred_answer = extract_answer(pred)

    score = 1.0 if pred_answer == grounded_output else 0.0

    # get type
    row_id = doc["id"]
    first = row_id.split("_", 1)[0]
    type = None
    if first == "among":
        type = "among"
    elif first == "rotation":
        type = "rotation"
    elif first in ("around", "aroundnew"):
        type = "around"
    else:
        type = "other"

    return {
        "overall_accuracy": {"score": score},
        "around_accuracy": {
            "score": score,
            "type": type,
        },
        "among_accuracy": {
            "score": score,
            "type": type,
        },
        "rotation_accuracy": {
            "score": score,
            "type": type,
        },
    }


def mindcube_aggregate_results(results):
    """Aggregates the 'overall_accuracy' results.

    Calculates the mean score from a list of result dictionaries.

    Args:
        results (list[dict]): A list of dictionaries, where each dict has
                              a 'score' key (e.g., [{'score': 1.0}, ...]).

    Returns:
        float: The average score (mean accuracy).
    """
    # --- Compute the total score across all results ---
    total_score = 0.0
    for res in results:
        total_score += res["score"]

    # --- Compute average score safely ---
    avg_score = total_score / len(results) if results else 0.0
    return avg_score


def mindcube_aggregate_among_results(results):
    """Aggregates the 'among_accuracy' results.

    Calculates the mean score *only* for results where the 'type' is 'among'.

    Args:
        results (list[dict]): A list of dictionaries, where each dict has
                              a 'score' key and a 'type' key
                              (e.g., [{'score': 1.0, 'type': 'among'}, ...]).

    Returns:
        float: The average score (mean accuracy) for 'among' type questions.
    """
    total_score = 0.0
    count = 0
    for res in results:
        if res["type"] == "among":
            total_score += res["score"]
            count += 1
    avg_score = total_score / count if count > 0 else 0.0
    return avg_score


def mindcube_aggregate_rotation_results(results):
    """Aggregates the 'rotation_accuracy' results.

    Calculates the mean score *only* for results where the 'type' is 'rotation'.

    Args:
        results (list[dict]): A list of dictionaries, where each dict has
                              a 'score' key and a 'type' key
                              (e.g., [{'score': 0.0, 'type': 'rotation'}, ...]).

    Returns:
        float: The average score (mean accuracy) for 'rotation' type questions.
    """
    total_score = 0.0
    count = 0
    for res in results:
        if res["type"] == "rotation":
            total_score += res["score"]
            count += 1
    avg_score = total_score / count if count > 0 else 0.0
    return avg_score


def mindcube_aggregate_around_results(results):
    """Aggregates the 'around_accuracy' results.

    Calculates the mean score *only* for results where the 'type' is 'around'.

    Args:
        results (list[dict]): A list of dictionaries, where each dict has
                              a 'score' key and a 'type' key
                              (e.g., [{'score': 1.0, 'type': 'around'}, ...]).

    Returns:
        float: The average score (mean accuracy) for 'around' type questions.
    """
    total_score = 0.0
    count = 0
    for res in results:
        if res["type"] == "around":
            total_score += res["score"]
            count += 1
    avg_score = total_score / count if count > 0 else 0.0
    return avg_score
