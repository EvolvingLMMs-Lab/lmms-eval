import ast
import os
import re
import string
from pathlib import Path
from typing import Any

from loguru import logger as eval_logger

IMAGE_ROOT_ENV_VAR = "MMLONGBENCH_IMAGE_ROOT"


def _normalize_text(text: Any) -> str:
    normalized = "" if text is None else str(text)
    normalized = normalized.lower()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    return " ".join(normalized.split())


def _extract_answer_text(prediction: Any) -> str:
    text = "" if prediction is None else str(prediction).strip()
    if not text:
        return ""

    match = re.search(r"answer\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if extracted:
            return extracted
    return text


def _looks_like_visual_haystack(doc: dict) -> bool:
    ctxs = doc.get("ctxs")
    return isinstance(ctxs, list) and len(ctxs) > 0 and isinstance(ctxs[0], str)


def _resolve_image_path(relative_path: str) -> str | None:
    direct_path = Path(relative_path)
    if direct_path.exists():
        return str(direct_path)

    image_root = os.getenv(IMAGE_ROOT_ENV_VAR, "").strip()
    if image_root:
        rooted_path = Path(image_root).expanduser() / relative_path
        if rooted_path.exists():
            return str(rooted_path)

    return None


def _collect_visual_paths(doc: dict) -> list[str]:
    if _looks_like_visual_haystack(doc):
        candidates = list(doc.get("ctxs", []))
    else:
        candidates = list(doc.get("image_list", []))
        for optional_key in ("needle_image_list", "choices_image"):
            optional_values = doc.get(optional_key)
            if isinstance(optional_values, list):
                candidates.extend(optional_values)

    deduplicated = []
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        resolved_path = _resolve_image_path(candidate)
        if resolved_path is None or resolved_path in seen:
            continue
        seen.add(resolved_path)
        deduplicated.append(resolved_path)
    return deduplicated


def _ctx_item_to_prompt_text(item: Any) -> str:
    if isinstance(item, dict):
        if item.get("type") == "image":
            return "<image>"
        text_value = item.get("text", "")
        return str(text_value).strip()
    if isinstance(item, str):
        return "<image>"
    return ""


def _build_context_block(doc: dict) -> str:
    ctxs = doc.get("ctxs", [])
    if not isinstance(ctxs, list):
        return ""

    prompt_chunks = []
    for item in ctxs:
        chunk = _ctx_item_to_prompt_text(item)
        if chunk:
            prompt_chunks.append(chunk)
    return "\n\n".join(prompt_chunks)


def _build_question_block(doc: dict) -> str:
    question = str(doc.get("question", "")).strip()
    choice_images = doc.get("choices_image")
    if not isinstance(choice_images, list) or not choice_images:
        return question

    option_lines = [f"{chr(ord('A') + idx)}. <image>" for idx in range(len(choice_images))]
    return "\n".join([question] + option_lines)


def _build_instruction(doc: dict) -> str:
    if _looks_like_visual_haystack(doc):
        return "You are given a set of images. Answer the question with Yes or No."

    if isinstance(doc.get("choices_image"), list) and doc.get("choices_image"):
        return "You are given interleaved text and images. Answer with the option letter only."

    answer = doc.get("answer")
    if isinstance(answer, list):
        return "You are given interleaved text and images. Answer with a list of integers in brackets, like [1, 2, 3]."

    return "You are given interleaved text and images. Answer the question concisely."


def _to_int_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value if isinstance(item, int)]
    return []


def _extract_predicted_int_list(prediction: str) -> list[int]:
    bracketed_candidates = re.findall(r"\[[^\]]*\]", prediction)
    for candidate in reversed(bracketed_candidates):
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, list):
            values = []
            for item in parsed:
                try:
                    values.append(int(item))
                except (TypeError, ValueError):
                    continue
            if values:
                return values

    return [int(item) for item in re.findall(r"-?\d+", prediction)]


def _extract_binary_label(prediction: str) -> str:
    matches = re.findall(r"\b(yes|no)\b", prediction.lower())
    return matches[-1] if matches else ""


def _extract_choice_index(prediction: str) -> int:
    normalized = prediction.strip().upper()

    explicit_match = re.search(r"ANSWER\s*[:\-]?\s*\(?([A-J])\)?", normalized)
    if explicit_match:
        return ord(explicit_match.group(1)) - ord("A")

    letter_match = re.search(r"\b([A-J])\b(?!.*\b[A-J]\b)", normalized)
    if letter_match:
        return ord(letter_match.group(1)) - ord("A")

    digit_match = re.search(r"\b(\d+)\b(?!.*\b\d+\b)", normalized)
    if digit_match:
        return int(digit_match.group(1))

    return -1


def _substring_exact_match(prediction: str, answers: list[str]) -> float:
    normalized_prediction = _normalize_text(prediction)
    if not normalized_prediction:
        return 0.0

    for answer in answers:
        normalized_answer = _normalize_text(answer)
        if normalized_answer and normalized_answer in normalized_prediction:
            return 1.0
    return 0.0


def _infer_category(doc: dict) -> str:
    category = doc.get("category")
    if isinstance(category, str) and category:
        return category
    if _looks_like_visual_haystack(doc):
        return "visual-haystack"
    return "unknown"


def _score_prediction(doc: dict, parsed_prediction: str) -> float:
    answer = doc.get("answer")

    if _looks_like_visual_haystack(doc):
        return float(_extract_binary_label(parsed_prediction) == _extract_binary_label(str(answer)))

    if isinstance(answer, list):
        gold_list = _to_int_list(answer)
        pred_list = _extract_predicted_int_list(parsed_prediction)
        return float(sum(gold_list) == sum(pred_list))

    if isinstance(answer, int):
        return float(_extract_choice_index(parsed_prediction) == answer)

    if isinstance(answer, str):
        return _substring_exact_match(parsed_prediction, [answer])

    return 0.0


def mmlongbench_doc_to_visual(doc):
    return _collect_visual_paths(doc)


def mmlongbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "Answer:")

    instruction = _build_instruction(doc)
    context = _build_context_block(doc)
    question_block = _build_question_block(doc)

    prompt_parts = []
    prefix_block = f"{pre_prompt}{instruction}".strip()
    if prefix_block:
        prompt_parts.append(prefix_block)
    if context:
        prompt_parts.append(context)
    prompt_parts.append(f"Question: {question_block}")
    if post_prompt:
        prompt_parts.append(post_prompt)

    return "\n\n".join(prompt_parts).strip()


def mmlongbench_process_results(doc, results):
    prediction = _extract_answer_text(results[0] if results else "")
    score = _score_prediction(doc, prediction)
    return {
        "mmlongbench_acc": {
            "score": score,
            "category": _infer_category(doc),
        }
    }


def mmlongbench_aggregate_results(results):
    if not results:
        return 0.0

    total_score = 0.0
    per_category: dict[str, dict[str, float]] = {}

    for result in results:
        score = float(result.get("score", 0.0))
        category = str(result.get("category", "unknown"))
        total_score += score

        if category not in per_category:
            per_category[category] = {"score": 0.0, "count": 0.0}
        per_category[category]["score"] += score
        per_category[category]["count"] += 1

    for category, stats in sorted(per_category.items()):
        count = stats["count"]
        category_score = stats["score"] / count if count > 0 else 0.0
        eval_logger.info("MMLongBench category={} samples={} acc={:.4f}", category, int(count), category_score)

    return total_score / len(results)
