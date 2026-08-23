import re
import string
from typing import Any, Dict, List, Optional

from lmms_eval.tasks._task_utils.default_template_yaml import load_default_template_yaml

config = load_default_template_yaml(__file__)


def _extract_answer_letter(text: str, valid_letters: str = string.ascii_uppercase) -> str:
    """
    Extract the answer choice letter from a model response.

    Handles terse answers as well as verbose responses that state the answer
    later in the text. Only letters in ``valid_letters`` are accepted, so a
    prose opener like "The correct answer is (B)." is parsed as 'B', never as
    the 'T' of "The".

    Examples:
    'A answer1' -> 'A'
    'A) answer2' -> 'A'
    '(B) answer' -> 'B'
    'C' -> 'C'
    '(C)' -> 'C'
    'A.' -> 'A'
    'The correct answer is (B).' -> 'B'
    'I think the answer is C because ...' -> 'C'

    Return an empty string if no answer letter is found.
    """
    text = text.strip()
    letters = re.escape(valid_letters.upper())

    # 1) Terse response: the first line is (almost) nothing but the letter.
    first_line = text.split("\n", 1)[0].strip()
    if len(first_line) <= 4:
        match = re.match(rf"^\(?([{letters}])\)?[\.\s:,)]*$", first_line, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # 2) Explicit statement, e.g. "the correct answer is (B)"; use the last one.
    matches = re.findall(rf"answers?(?:\s+is)?\s*[:\-]?\s*\(?([{letters}])\)?(?![A-Za-z])", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # 3) Parenthesized option anywhere, e.g. "... so I pick (C)"; use the last one.
    matches = re.findall(rf"\(([{letters}])\)", text)
    if matches:
        return matches[-1].upper()

    # 4) Leading letter, restricted to the valid choices.
    match = re.match(rf"[\(\s]*([{letters}])[\)\.\s]*", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def blink_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    num_choices = len(doc["choices"])
    choice_letters = ", ".join([chr(65 + i) for i in range(num_choices)])
    prompt = lmms_eval_specific_kwargs.get("pre_prompt", "").format(choice_letters) + doc["prompt"]
    return prompt


def blink_doc_to_visual(doc: dict) -> list:
    keys = doc.keys()
    image_keys = [item for item in keys if re.match(r"^image_\d+$", item)]
    image_list = []
    for image_key in image_keys:
        image = doc[image_key]
        if image is not None:
            image_list.append(image.convert("RGB"))
    return image_list


def _format_neo_ov_content(images: list, prompt: str) -> list:
    if len(images) == 1:
        return [{"type": "image", "url": images[0]}, {"type": "text", "text": prompt}]

    content = []
    for idx, image in enumerate(images, start=1):
        content.append({"type": "text", "text": f"Image-{idx}: "})
        content.append({"type": "image", "url": image})
    content.append({"type": "text", "text": prompt})
    return content


def _blink_neo_ov_prompt(doc: dict[str, Any]) -> str:
    prompt = doc.get("prompt")
    if prompt:
        prompt = prompt.replace("<image>", "").strip()
        return prompt + "\nAnswer with the option's letter from the given choices directly."

    question = doc["question"].replace("<image>", "").strip()
    hint = doc.get("hint")
    if hint is not None and hint == hint:
        question = f"{hint}\n{question}"

    choices = doc.get("choices", [])
    for idx, item in enumerate(choices):
        question += f"\n{string.ascii_uppercase[idx]}. {item}"

    if choices:
        question += "\nAnswer with the option's letter from the given choices directly."
    else:
        question += "\nAnswer the question directly."
    return question


def blink_doc_to_messages(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> list:
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    images = blink_doc_to_visual(doc)

    if lmms_eval_specific_kwargs.get("prompt_format") == "neo_ov":
        prompt = _blink_neo_ov_prompt(doc)
        return [{"role": "user", "content": _format_neo_ov_content(images, prompt)}]

    prompt = blink_doc_to_text(doc, lmms_eval_specific_kwargs)
    content = [{"type": "image", "url": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def blink_process_results(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    key_name = "blink_acc"
    # extract grounded answer
    grounded_output = doc["answer"].strip("()")
    response = result[0]

    # extract predicted answer, accepting only this doc's option letters
    valid_letters = string.ascii_uppercase[: len(doc.get("choices") or [])] or string.ascii_uppercase
    pred_letter = _extract_answer_letter(response, valid_letters)
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
