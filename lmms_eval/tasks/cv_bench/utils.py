import re
from collections import defaultdict
from typing import Any, Dict, List, Optional


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
    match = re.match(r"[\(\s]*([A-Z])[\)\.\s]*", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def cv_bench_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    options_labels = ["A", "B", "C", "D", "E"]
    num_options = len(doc["choices"])
    options_current_task = ", ".join(options_labels[:num_options])
    prompt = lmms_eval_specific_kwargs.get("pre_prompt", "").format(options_current_task) + "\n" + doc["prompt"]
    return prompt


def cv_bench_doc_to_visual(doc: dict) -> list:
    return [doc["image"].convert("RGB")]


def cv_bench_process_results(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    key_name = "cv_bench_acc"
    # extract grounded answer
    grounded_output = doc["answer"].strip("()")
    response = result[0]

    # extract predicted answer
    pred_letter = _extract_answer_letter(response)
    flag = pred_letter == grounded_output

    cv_bench_submission = {"id": doc["idx"], "gt_content": grounded_output, "pred_parsed": pred_letter, "pred": response, "type": doc["type"], "task": doc["task"], "source": doc["source"], "is_correct": flag}
    return {key_name: cv_bench_submission}


def cv_bench_aggregate_results(results: List[Dict]):
    total_samples = len(results)
    total_correct = 0

    for sample in results:
        if sample["is_correct"]:
            total_correct += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy


def cv_bench_default_aggregate_results(results: List[Dict]):
    source_samples = defaultdict(list)
    for elem in results:
        source = elem["source"]
        source_samples[source].append(elem["is_correct"])
    source_accuracies = {source: sum(scores) / len(scores) for source, scores in source_samples.items()}
    ade20k_2d = source_accuracies["ADE20K"]
    coco_2d = source_accuracies["COCO"]
    omni_3d = source_accuracies["Omni3D"]

    # original formula
    cv_bench_accuracy = 1 / 2 * ((ade20k_2d + coco_2d) / 2 + omni_3d)
    return cv_bench_accuracy
