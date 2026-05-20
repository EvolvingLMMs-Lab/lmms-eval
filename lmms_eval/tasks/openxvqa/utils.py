import ast
import io
import re

from loguru import logger as eval_logger
from PIL import Image

LETTERS = "ABCDEFG"


def _parse_choices(choices_raw):
    """Parse choices field which may be a string repr of a list or an actual list."""
    if isinstance(choices_raw, str):
        return ast.literal_eval(choices_raw)
    return list(choices_raw)


def _load_image(image_data):
    """Load image from various formats (bytes, dict with bytes, PIL Image)."""
    if isinstance(image_data, Image.Image):
        return image_data
    if isinstance(image_data, bytes):
        return Image.open(io.BytesIO(image_data))
    if isinstance(image_data, dict) and "bytes" in image_data:
        return Image.open(io.BytesIO(image_data["bytes"]))
    return image_data


def openxvqa_doc_to_visual(doc):
    return [_load_image(doc["image"])]


def openxvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    choices = _parse_choices(doc["choices"])
    opts_text = "\n".join(f"({LETTERS[i]}) {opt}" for i, opt in enumerate(choices))

    question = f"Select the best answer to the following multiple-choice question based on the image.\n{doc['question']}\nOptions:\n{opts_text}"

    return f"{pre_prompt}{question}{post_prompt}"


def openxvqa_doc_to_target(doc):
    choices = _parse_choices(doc["choices"])
    idx = int(doc["correct_answer"])
    return LETTERS[idx]


def _extract_answer(response, choices):
    """Extract answer letter from model response."""
    num_choices = len(choices)
    valid_letters = list(LETTERS[:num_choices])
    last_letter = valid_letters[-1]

    response = response.replace("answer", "").replace("Answer", "")
    pred_answer = re.findall(rf"[\(\ ]*([{valid_letters[0]}-{last_letter}])[\)\ ]*", response)

    if pred_answer:
        pred_letter = pred_answer[0].strip()
        if pred_letter in valid_letters:
            return valid_letters.index(pred_letter)

    # Fallback: match option text
    for idx, opt in enumerate(choices):
        opt = opt.strip().strip(".")
        if opt.lower() in response.lower():
            return idx

    eval_logger.warning(f"Cannot extract answer from response: {response}")
    return -1


def openxvqa_process_results(doc, results):
    pred = results[0]
    choices = _parse_choices(doc["choices"])
    pred_idx = _extract_answer(pred, choices)
    gt_idx = int(doc["correct_answer"])

    return {
        "openxvqa_accuracy": {
            "pred_answer": pred_idx,
            "ground_truth": gt_idx,
        }
    }


def openxvqa_aggregate_results(results):
    correct = sum(1 for r in results if r["pred_answer"] == r["ground_truth"])
    return correct / len(results) if results else 0
