import re
from decimal import Decimal, InvalidOperation
from typing import Any


def zerobench_doc_to_visual(doc: dict[str, Any]) -> list:
    images = doc.get("question_images_decoded", [])
    return [image.convert("RGB") for image in images if image is not None]


def zerobench_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "\nAnswer with only the final answer.")
    question = doc["question_text"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


def _normalize_text(text: str) -> str:
    normalized = text.strip().lower().strip("\"'`")
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized.endswith("."):
        normalized = normalized[:-1].rstrip()
    return normalized


def _to_decimal(text: str) -> Decimal | None:
    candidate = text.strip().replace(",", "")
    if not re.fullmatch(r"-?(?:\d+\.\d+|\d+|\.\d+)", candidate):
        return None

    try:
        return Decimal(candidate)
    except InvalidOperation:
        return None


def _is_exact_match(prediction: str, target: str) -> bool:
    normalized_prediction = _normalize_text(prediction)
    normalized_target = _normalize_text(target)

    prediction_decimal = _to_decimal(normalized_prediction)
    target_decimal = _to_decimal(normalized_target)
    if prediction_decimal is not None and target_decimal is not None:
        return prediction_decimal == target_decimal

    return normalized_prediction == normalized_target


def zerobench_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, Any]]:
    prediction = results[0].strip() if results else ""
    target = str(doc["question_answer"]).strip()
    score = 1.0 if _is_exact_match(prediction, target) else 0.0

    return {
        "zerobench_exact_match": {
            "question_id": doc.get("question_id", ""),
            "prediction": prediction,
            "target": target,
            "score": score,
        }
    }


def zerobench_aggregate_results(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0

    total_score = sum(result["score"] for result in results)
    return total_score / len(results)
