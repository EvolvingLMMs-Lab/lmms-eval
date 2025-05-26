import re
from typing import Dict, List


def vlmsareblind_doc_to_visual(doc: Dict) -> List:
    """Extract image from the document."""
    if "image" in doc:
        return [doc["image"]]
    return []


def vlmsareblind_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict) -> str:
    """Format the prompt for the model."""
    prompt = doc.get("prompt", "")
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{prompt}{post_prompt}"


def vlmsareblind_doc_to_target(doc: Dict) -> str:
    """Extract the expected answer from the document."""
    # The answer should be a number in curly brackets like {3}
    answer = doc.get("answer", "")
    return str(answer)


def extract_answer(response: str) -> str:
    """Extract the number from a response containing {N} format."""
    # Look for pattern {number}
    match = re.search(r"\{(\d+)\}", response)
    if match:
        return match.group(0)  # Return the full match including brackets

    # If no brackets found, try to find just a number
    match = re.search(r"\b(\d+)\b", response)
    if match:
        return f"{{{match.group(1)}}}"  # Add brackets

    return response.strip()


def vlmsareblind_process_result(doc: Dict, result) -> Dict:
    """Process the model's response and compare with ground truth."""
    # Handle case where result is a list (common with generate_until)
    if isinstance(result, list):
        result = result[0] if result else ""

    pred = extract_answer(str(result))
    target = doc.get("answer", "")

    # Exact match comparison
    correct = pred == target

    return {
        "exact_match": correct,
        "pred": pred,
        "target": target,
    }
