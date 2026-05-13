import io
import re
from typing import Any

from PIL import Image


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().split())


def _to_rgb(image_obj: Any):
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, bytes):
        return Image.open(io.BytesIO(image_obj)).convert("RGB")
    return None


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


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    if len(left) > len(right):
        left, right = right, left

    previous = list(range(len(left) + 1))
    for i, right_ch in enumerate(right, start=1):
        current = [i]
        for j, left_ch in enumerate(left, start=1):
            insertion = previous[j] + 1
            deletion = current[j - 1] + 1
            substitution = previous[j - 1] + (left_ch != right_ch)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _anls_score(prediction: str, answer: str, threshold: float = 0.5) -> float:
    pred = _normalize_text(prediction)
    target = _normalize_text(answer)
    if not pred and not target:
        return 1.0
    if not pred or not target:
        return 0.0

    distance = _levenshtein_distance(pred, target)
    normalized_distance = distance / max(len(pred), len(target))
    score = 1.0 - normalized_distance
    if score < threshold:
        return 0.0
    return score


def dude_doc_to_visual(doc):
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


def dude_doc_to_text(doc, lmms_eval_specific_kwargs=None):
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


def dude_doc_to_target(doc):
    answers = _extract_answers(doc)
    return answers[0] if answers else ""


def dude_process_results(doc, results):
    prediction = str(results[0]).strip()
    answers = _extract_answers(doc)

    if not answers:
        return {"dude_anls": 0.0}

    score = max(_anls_score(prediction, answer) for answer in answers)

    options = _extract_options(doc)
    if options:
        pred_letter = _extract_option_letter(prediction)
        if pred_letter:
            for answer in answers:
                if pred_letter == answer.strip().upper()[:1]:
                    score = max(score, 1.0)

    return {"dude_anls": score}


def dude_aggregate_anls(items):
    if not items:
        return 0.0
    return sum(float(item) for item in items) / len(items)
