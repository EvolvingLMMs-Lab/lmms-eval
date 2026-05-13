import ast
import json
import re
from collections import defaultdict
from typing import Any, Sequence

from loguru import logger as eval_logger

_BOXED_PREFIX = r"\boxed{"
_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in {"none", "null"}:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
                continue
        return [text]
    return [value]


def _extract_boxed_answers(text: str) -> list[str]:
    answers: list[str] = []
    search_start = 0
    while True:
        start = text.find(_BOXED_PREFIX, search_start)
        if start < 0:
            break

        idx = start + len(_BOXED_PREFIX)
        depth = 1
        while idx < len(text) and depth > 0:
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
            idx += 1

        if depth == 0:
            answers.append(text[start + len(_BOXED_PREFIX) : idx - 1].strip())
            search_start = idx
            continue

        break

    return answers


def _split_top_level_answers(text: str) -> list[str]:
    candidates: list[str] = []
    depth = 0
    start = 0
    for idx, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
        elif char in {",", ";"} and depth == 0:
            chunk = text[start:idx].strip()
            if chunk:
                candidates.append(chunk)
            start = idx + 1

    tail = text[start:].strip()
    if tail:
        candidates.append(tail)
    return candidates


def _extract_predicted_answers(prediction: str) -> list[str]:
    boxed_answers = _extract_boxed_answers(prediction)
    if boxed_answers:
        split_answers: list[str] = []
        for boxed in boxed_answers:
            split_answers.extend(_split_top_level_answers(boxed))
        return split_answers if split_answers else boxed_answers

    final_answer_match = re.search(r"final\s+answer\s*[:：]\s*(.+)", prediction, flags=re.IGNORECASE | re.DOTALL)
    if final_answer_match:
        guessed = final_answer_match.group(1).strip().splitlines()[0].strip()
        guessed_parts = _split_top_level_answers(guessed)
        return guessed_parts if guessed_parts else [guessed]

    non_empty_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    if non_empty_lines:
        tail = non_empty_lines[-1]
        tail_parts = _split_top_level_answers(tail)
        return tail_parts if tail_parts else [tail]

    return []


def _strip_latex_wrappers(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.strip("$")
    cleaned = cleaned.replace("\\left", "")
    cleaned = cleaned.replace("\\right", "")
    cleaned = cleaned.replace("\\,", "")
    cleaned = cleaned.replace("\\!", "")
    cleaned = cleaned.replace("−", "-")
    cleaned = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", r"\1", cleaned)
    return cleaned.strip()


def _normalize_answer(text: str) -> str:
    cleaned = _strip_latex_wrappers(text)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.strip(".:;,。!?")
    return cleaned


def _answer_variants(text: str) -> set[str]:
    normalized = _normalize_answer(text)
    if normalized == "":
        return set()

    variants = {normalized}
    if "=" in normalized:
        left, right = normalized.split("=", maxsplit=1)
        if left:
            variants.add(left)
        if right:
            variants.add(right)
    return {item for item in variants if item}


def _numeric_value(text: str) -> float | None:
    candidate = text.replace(",", "")
    if re.fullmatch(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", candidate):
        try:
            return float(candidate)
        except ValueError:
            return None

    if re.fullmatch(r"[-+]?\d+/\d+", candidate):
        try:
            numerator, denominator = candidate.split("/")
            return float(numerator) / float(denominator)
        except ValueError:
            return None

    matches = _NUMBER_PATTERN.findall(candidate)
    if len(matches) == 1:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None


def _answers_match(prediction: str, answer: str) -> bool:
    pred_variants = _answer_variants(prediction)
    answer_variants = _answer_variants(answer)
    if not pred_variants or not answer_variants:
        return False

    if pred_variants & answer_variants:
        return True

    for pred_variant in pred_variants:
        pred_numeric = _numeric_value(pred_variant)
        for answer_variant in answer_variants:
            answer_numeric = _numeric_value(answer_variant)
            if pred_numeric is not None and answer_numeric is not None:
                if abs(pred_numeric - answer_numeric) <= 1e-3:
                    return True
                if abs(answer_numeric) > 1e-6 and abs(pred_numeric - answer_numeric) / abs(answer_numeric) <= 1e-3:
                    return True

            if pred_variant in answer_variant or answer_variant in pred_variant:
                return True

    return False


def _normalize_points(points: Sequence[Any], expected_count: int) -> list[float]:
    if expected_count <= 0:
        return []

    parsed: list[float] = []
    for value in points:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            parsed.append(1.0)

    if len(parsed) != expected_count:
        return [1.0 / expected_count for _ in range(expected_count)]

    total = sum(parsed)
    if total <= 0:
        return [1.0 / expected_count for _ in range(expected_count)]

    return [value / total for value in parsed]


def _score_prediction(prediction: str, answers: list[str], points: list[Any]) -> tuple[float, list[str]]:
    if not answers:
        return 0.0, []

    predicted_answers = _extract_predicted_answers(prediction)
    if not predicted_answers and prediction.strip() != "":
        predicted_answers = [prediction.strip()]

    weights = _normalize_points(points, len(answers))
    remaining_predictions = predicted_answers.copy()
    total_score = 0.0

    for answer, weight in zip(answers, weights):
        matched_index = -1
        for idx, predicted in enumerate(remaining_predictions):
            if _answers_match(predicted, answer):
                matched_index = idx
                break

        if matched_index >= 0:
            total_score += weight
            remaining_predictions.pop(matched_index)

    return total_score, predicted_answers


def hipho_doc_to_visual(doc: dict[str, Any]) -> list[Any]:
    images = doc.get("image", [])
    if images is None:
        return []
    if not isinstance(images, list):
        images = [images]

    visuals = []
    for image in images:
        if image is None:
            continue
        if hasattr(image, "convert"):
            visuals.append(image.convert("RGB"))
        else:
            visuals.append(image)
    return visuals


def hipho_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")

    context = str(doc.get("context", "")).strip()
    question = str(doc.get("question", "")).strip()
    marking = doc.get("marking")
    information = doc.get("information")

    sections = [
        str(pre_prompt).strip(),
        "You are solving a physics olympiad question.",
        "If the problem has multiple sub-answers, provide them in order.",
        "End with a single line in this format: Final Answer: \\boxed{...}.",
    ]

    if context:
        sections.append(f"Context:\n{context}")
    if question:
        sections.append(f"Question: {question}")
    if marking not in (None, "", "None", "null"):
        sections.append(f"Marking guideline: {marking}")
    if information not in (None, "", "None", "null"):
        sections.append(f"Additional information: {information}")

    post_prompt = str(post_prompt).strip()
    if post_prompt:
        sections.append(post_prompt)

    return "\n\n".join(section for section in sections if section)


def hipho_doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc.get("answer", ""))


def hipho_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, Any]]:
    prediction = ""
    if results:
        prediction = str(results[0]).strip()

    answers = [str(answer) for answer in _as_list(doc.get("answer"))]
    points = _as_list(doc.get("points"))
    score, parsed_prediction = _score_prediction(prediction, answers, points)

    return {
        "hipho_score": {
            "id": str(doc.get("id", doc.get("index", ""))),
            "source": str(doc.get("source", "unknown")),
            "field": str(doc.get("field", "unknown")),
            "score": score,
            "parsed_prediction": parsed_prediction,
            "prediction": prediction,
            "answers": answers,
        }
    }


def hipho_aggregate_results(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0

    source_scores: dict[str, list[float]] = defaultdict(list)
    field_scores: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []

    for item in results:
        score = float(item.get("score", 0.0))
        source = str(item.get("source", "unknown"))
        field = str(item.get("field", "unknown"))

        all_scores.append(score)
        source_scores[source].append(score)
        field_scores[field].append(score)

    for source, source_values in sorted(source_scores.items()):
        eval_logger.info("HiPhO source {}: {:.2f}", source, 100.0 * sum(source_values) / len(source_values))

    for field, field_values in sorted(field_scores.items()):
        eval_logger.info("HiPhO field {}: {:.2f}", field, 100.0 * sum(field_values) / len(field_values))

    return 100.0 * sum(all_scores) / len(all_scores)
