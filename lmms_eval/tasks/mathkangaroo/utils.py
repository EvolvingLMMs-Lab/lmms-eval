import re
from typing import Any


def mathkangaroo_doc_to_visual(doc):
    image = doc.get("image")
    if image is not None and hasattr(image, "convert"):
        return [image.convert("RGB")]
    return []


def mathkangaroo_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "\nAnswer with the option letter (A, B, C, D, or E) only.")
    question = str(doc.get("question", "")).strip()
    return f"{pre_prompt}{question}{post_prompt}"


def _normalize_targets(answer: Any) -> set[str]:
    if answer is None:
        return set()
    return set(re.findall(r"[A-E]", str(answer).upper()))


def _extract_prediction(response: str) -> str:
    if not response:
        return ""

    text = str(response).strip()
    direct_match = re.search(r"(?i)(?:final\s+answer|answer|option)\s*(?:is|:)?\s*\(?([A-E])\)?", text)
    if direct_match:
        return direct_match.group(1).upper()

    for line in reversed(text.splitlines()):
        line = line.strip().upper()
        if not line:
            continue
        line_match = re.fullmatch(r"\(?([A-E])\)?[\.)]?", line)
        if line_match:
            return line_match.group(1)

    candidates = re.findall(r"\b([A-E])\b", text.upper())
    if candidates:
        return candidates[-1]
    return ""


def mathkangaroo_process_results(doc, results):
    prediction = _extract_prediction(results[0] if results else "")
    targets = _normalize_targets(doc.get("ground_truth"))
    score = 1.0 if prediction and prediction in targets else 0.0
    return {"mathkangaroo_accuracy": score}
