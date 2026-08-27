import ast
import json
import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional

from loguru import logger as eval_logger
from PIL import Image


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False

    return None


def _parse_bool_from_serialized(candidate: str) -> Optional[bool]:
    candidate = candidate.strip()
    if not candidate:
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(candidate)
        except Exception:
            continue

        if isinstance(parsed, dict):
            for key in ("answer", "Answer", "ANSWER"):
                if key in parsed:
                    return _coerce_bool(parsed[key])
        else:
            parsed_bool = _coerce_bool(parsed)
            if parsed_bool is not None:
                return parsed_bool

    return None


def _parse_bool_from_response(response: str) -> Optional[bool]:
    if not response:
        return None

    cleaned = response.strip()

    direct = _coerce_bool(cleaned.strip("`\"' "))
    if direct is not None:
        return direct

    serialized_candidates = [cleaned]
    serialized_candidates.extend(re.findall(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL))
    serialized_candidates.extend(re.findall(r"\{[\s\S]*?\}", cleaned))

    for candidate in serialized_candidates:
        parsed = _parse_bool_from_serialized(candidate)
        if parsed is not None:
            return parsed

    lowered = cleaned.lower()

    answer_match = re.search(r'"answer"\s*:\s*(true|false)', lowered)
    if answer_match:
        return answer_match.group(1) == "true"

    answer_match = re.search(r"\banswer\s*[:=]\s*(true|false)\b", lowered)
    if answer_match:
        return answer_match.group(1) == "true"

    token_match = re.search(r"\b(true|false|yes|no)\b", lowered)
    if token_match:
        return _coerce_bool(token_match.group(1))

    return None


def viverbench_doc_to_visual(doc: dict[str, Any]) -> list:
    visuals = []
    for image_bytes in doc.get("img", []):
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        visuals.append(image)
    return visuals


def viverbench_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["question"]

    text_parts = [part for part in (pre_prompt, question, post_prompt) if part]
    return "\n".join(text_parts)


def viverbench_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, Any]]:
    response = results[0] if results else ""
    pred_answer = _parse_bool_from_response(response)
    target_answer = bool(doc["answer"])

    submission = {
        "prompt_id": doc.get("prompt_id", ""),
        "task": doc.get("task", "unknown"),
        "target_answer": target_answer,
        "pred_answer": pred_answer,
        "raw_response": response,
        "is_correct": pred_answer is not None and pred_answer == target_answer,
    }

    return {"viverbench_acc": submission}


def viverbench_aggregate_results(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0

    by_task = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0

    for result in results:
        task = result.get("task", "unknown")
        is_correct = bool(result.get("is_correct", False))

        by_task[task]["total"] += 1
        if is_correct:
            by_task[task]["correct"] += 1
            total_correct += 1

    for task in sorted(by_task.keys()):
        stats = by_task[task]
        task_acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        eval_logger.info(f"ViVerBench - {task}: {task_acc:.4f} ({stats['correct']}/{stats['total']})")

    overall_acc = total_correct / len(results)
    eval_logger.info(f"ViVerBench - overall: {overall_acc:.4f} ({total_correct}/{len(results)})")
    return overall_acc
