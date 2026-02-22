import io
import re
from typing import Any

from PIL import Image


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().split())


def _to_float(text: str):
    candidate = text.strip().replace(",", "")
    if candidate.endswith("%"):
        try:
            return float(candidate[:-1]) / 100.0
        except ValueError:
            return None

    match = re.match(r"^[-+]?\d*\.?\d+$", candidate)
    if not match:
        return None

    try:
        return float(candidate)
    except ValueError:
        return None


def _relaxed_match(prediction: str, target: str, max_relative_change: float = 0.05) -> float:
    pred = _normalize_text(prediction)
    tgt = _normalize_text(target)
    if not pred or not tgt:
        return 0.0

    pred_float = _to_float(pred)
    tgt_float = _to_float(tgt)

    if pred_float is not None and tgt_float is not None:
        if tgt_float == 0:
            return float(pred_float == 0)
        return float(abs(pred_float - tgt_float) / abs(tgt_float) <= max_relative_change)

    return float(pred == tgt)


def _to_rgb(image_obj: Any):
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")

    if isinstance(image_obj, bytes):
        return Image.open(io.BytesIO(image_obj)).convert("RGB")

    return None


def _extract_question(doc: dict) -> str:
    for key in ["question", "query", "prompt", "instruction"]:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_options(doc: dict) -> list[str]:
    options = doc.get("options", doc.get("choices"))
    if isinstance(options, list):
        normalized = []
        for item in options:
            if isinstance(item, dict):
                text = item.get("text", item.get("option", ""))
                normalized.append(str(text))
            else:
                normalized.append(str(item))
        return [x for x in normalized if x.strip()]
    return []


def _extract_answers(doc: dict) -> list[str]:
    answers = doc.get("answers", doc.get("answer", doc.get("target")))
    if answers is None:
        return []
    if isinstance(answers, list):
        return [str(item) for item in answers if str(item).strip()]
    return [str(answers)]


def _extract_option_letter(prediction: str) -> str:
    normalized = prediction.strip().upper()
    match = re.search(r"\b([A-Z])\b", normalized)
    if match:
        return match.group(1)
    return ""


def officeqa_doc_to_visual(doc):
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


def officeqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
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


def officeqa_doc_to_target(doc):
    answers = _extract_answers(doc)
    return answers[0] if answers else ""


def officeqa_process_results(doc, results):
    prediction = str(results[0]).strip()
    answers = _extract_answers(doc)

    if not answers:
        return {"officeqa_relaxed_accuracy": 0.0}

    best_score = max(_relaxed_match(prediction, answer) for answer in answers)

    options = _extract_options(doc)
    if options:
        pred_letter = _extract_option_letter(prediction)
        if pred_letter:
            for answer in answers:
                answer_norm = answer.strip().upper()
                if pred_letter == answer_norm[:1]:
                    best_score = max(best_score, 1.0)

    return {"officeqa_relaxed_accuracy": best_score}
