import io
import re
from typing import Any

from PIL import Image


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().split())


def _extract_question(doc: dict) -> str:
    for key in ["question", "query", "prompt", "instruction", "text"]:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_answers(doc: dict) -> list[str]:
    answers = doc.get("answers", doc.get("answer", doc.get("target")))
    if answers is None:
        return []
    if isinstance(answers, list):
        return [str(item) for item in answers if str(item).strip()]
    return [str(answers)]


def _extract_options(doc: dict) -> list[str]:
    options = doc.get("options", doc.get("choices"))
    if isinstance(options, list):
        values = []
        for item in options:
            if isinstance(item, dict):
                values.append(str(item.get("text", item.get("option", ""))))
            else:
                values.append(str(item))
        return [value for value in values if value.strip()]
    return []


def _extract_option_letter(prediction: str) -> str:
    match = re.search(r"\b([A-Z])\b", prediction.strip().upper())
    if match:
        return match.group(1)
    return ""


def _to_rgb(image_obj: Any):
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, bytes):
        return Image.open(io.BytesIO(image_obj)).convert("RGB")
    return None


def omnidocbench_doc_to_visual(doc):
    visuals = []

    for key in ["image", "page_image", "document_image"]:
        if key in doc:
            img = _to_rgb(doc[key])
            if img is not None:
                visuals.append(img)

    for key in ["images", "page_images", "document_images", "pages"]:
        value = doc.get(key)
        if isinstance(value, list):
            for item in value:
                img = _to_rgb(item)
                if img is not None:
                    visuals.append(img)

    return visuals


def omnidocbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")

    question = _extract_question(doc)
    options = _extract_options(doc)
    if not options:
        return f"{pre_prompt}{question}{post_prompt}"

    option_labels = [chr(ord("A") + idx) for idx in range(len(options))]
    option_lines = "\n".join(f"{label}. {choice}" for label, choice in zip(option_labels, options))
    return f"{pre_prompt}{question}\n{option_lines}{post_prompt}"


def omnidocbench_doc_to_target(doc):
    answers = _extract_answers(doc)
    return answers[0] if answers else ""


def omnidocbench_process_results(doc, results):
    prediction = _normalize_text(results[0])
    answers = _extract_answers(doc)
    if not answers:
        return {"omnidocbench_exact_match": 0.0}

    answer_set = {_normalize_text(answer) for answer in answers}
    score = float(prediction in answer_set)

    options = _extract_options(doc)
    if options:
        pred_letter = _extract_option_letter(str(results[0]))
        if pred_letter:
            for answer in answers:
                if pred_letter == answer.strip().upper()[:1]:
                    score = max(score, 1.0)

    return {"omnidocbench_exact_match": score}
