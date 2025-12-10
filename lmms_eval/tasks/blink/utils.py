import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


def _extract_answer_letter(text: str) -> str:
    """
    Extract the answer choice letter from a string.

    Examples:
    'A answer1' -> 'A'
    'A) answer2' -> 'A'
    '(B) answer' -> 'B'
    'C' -> 'C'
    '(C)' -> 'C'
    'A.' -> 'A'

    Return an empty string if no letter is found.
    """
    text = text.strip()
    match = re.match(r'[\(\s]*([A-Z])[\)\.\s]*', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def blink_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    options_labels = ["A", "B", "C", "D", "E"]
    num_options = len(doc["choices"])
    options_current_task = ", ".join(options_labels[:num_options])
    prompt = lmms_eval_specific_kwargs.get("pre_prompt", "").format(options_current_task) + doc["prompt"]
    return prompt


def blink_doc_to_visual(doc: dict) -> list:
    keys = doc.keys()
    image_keys = [item for item in keys if re.match(r'^image_\d+$', item)]
    image_list = []
    for image_key in image_keys:
        image = doc[image_key]
        if image is not None:
            image_list.append(image.convert("RGB"))
    return image_list


def blink_process_results(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    key_name = "blink_acc"
    # extract grounded answer
    grounded_output = doc["answer"].strip("()")
    response = result[0]

    # extract predicted answer
    pred_letter = _extract_answer_letter(response)
    flag = pred_letter == grounded_output

    omnispatial_submission = {"id": doc["idx"], "gt_content": grounded_output, "pred_parsed": pred_letter, "pred": response, "sub_task": doc["sub_task"], "is_correct": flag}
    return {key_name: omnispatial_submission}


def blink_aggregate_results(results: List[Dict]):
    total_samples = len(results)
    total_correct = 0

    for sample in results:
        if sample["is_correct"]:
            total_correct += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy
