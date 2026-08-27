import re
from typing import Any

try:
    from lmms_eval.tasks._task_utils.math_verify_utils import MathVerifyFn
except Exception:
    MathVerifyFn = None


SUB_QUESTION_WEIGHTS = {
    2: [0.4348, 0.5652],
    3: [0.2506, 0.3258, 0.4236],
    4: [0.1616, 0.2101, 0.2732, 0.3551],
}

TAGGED_ANSWER_PATTERN = re.compile(r"<(\d+)>(.*?)</\1>", re.DOTALL)
ANSWER_CUE_PATTERN = re.compile(r"(?is)(?:final answer|answer)\s*(?:is|:)\s*(.+)")

math_verify_fn = MathVerifyFn() if MathVerifyFn is not None else None


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    normalized = str(text).strip().lower()
    normalized = normalized.replace("\n", " ")
    return " ".join(normalized.split())


def _to_rgb(image_obj):
    if image_obj is None:
        return None
    if hasattr(image_obj, "convert"):
        return image_obj.convert("RGB")
    return image_obj


def _extract_question_chunks(doc):
    interleave = doc.get("question_interleave")
    if not isinstance(interleave, list):
        interleave = []

    raw_images = doc.get("question_images")
    if not isinstance(raw_images, list):
        raw_images = []
    images = [img for img in (_to_rgb(image) for image in raw_images) if img is not None]

    chunks = []
    image_cursor = 0
    for item in interleave:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).lower()
        if item_type == "text":
            text = str(item.get("content", "")).strip()
            if text:
                chunks.append(("text", text))
        elif item_type == "image":
            if image_cursor < len(images):
                chunks.append(("image", images[image_cursor]))
                image_cursor += 1

    while image_cursor < len(images):
        chunks.append(("image", images[image_cursor]))
        image_cursor += 1

    if not chunks:
        fallback_question = str(doc.get("question", "")).strip()
        if fallback_question:
            chunks.append(("text", fallback_question))

    return chunks


def _clean_answer_text(answer: str) -> str:
    answer = str(answer).strip()
    answer = answer.removesuffix(".").strip()
    return answer


def _extract_tagged_answers(text: str) -> list[str]:
    matches = TAGGED_ANSWER_PATTERN.findall(text)
    if not matches:
        return []

    ordered = sorted(matches, key=lambda item: int(item[0]))
    return [_clean_answer_text(content) for _, content in ordered]


def _extract_last_boxed_content(text: str) -> str:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return ""

    cursor = start + len(marker)
    brace_depth = 1
    content_chars = []
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            brace_depth += 1
            content_chars.append(char)
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0:
                return _clean_answer_text("".join(content_chars))
            content_chars.append(char)
        else:
            content_chars.append(char)
        cursor += 1
    return ""


def _extract_answer_candidate(text: str) -> str:
    matches = list(ANSWER_CUE_PATTERN.finditer(text))
    if matches:
        candidate = matches[-1].group(1)
        candidate = candidate.splitlines()[0]
        return _clean_answer_text(candidate)

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if non_empty_lines:
        return _clean_answer_text(non_empty_lines[-1])

    return ""


def _extract_predicted_answers(prediction: str, expected_parts: int) -> list[str]:
    if not prediction:
        return []

    tagged_answers = _extract_tagged_answers(prediction)
    if tagged_answers:
        return tagged_answers

    boxed_content = _extract_last_boxed_content(prediction)
    if boxed_content:
        tagged_in_box = _extract_tagged_answers(boxed_content)
        if tagged_in_box:
            return tagged_in_box
        if expected_parts <= 1:
            return [boxed_content]
        prediction = boxed_content

    if expected_parts <= 1:
        candidate = _extract_answer_candidate(prediction)
        return [candidate] if candidate else []

    enumerated_parts = []
    for line in prediction.splitlines():
        line = line.strip()
        if not line:
            continue
        line_match = re.match(r"^\(?([1-9])\)?[\.:)]\s*(.+)$", line)
        if line_match:
            enumerated_parts.append(_clean_answer_text(line_match.group(2)))
    if enumerated_parts:
        return enumerated_parts

    candidate = _extract_answer_candidate(prediction)
    if candidate:
        return [candidate]
    return []


def _extract_ground_truth_answers(answer: str) -> list[str]:
    tagged_answers = _extract_tagged_answers(answer)
    if tagged_answers:
        return tagged_answers

    cleaned = _clean_answer_text(answer)
    return [cleaned] if cleaned else []


def _judge_answer_match(prediction_part: str, answer_part: str) -> bool:
    if not prediction_part or not answer_part:
        return False

    if math_verify_fn is not None:
        try:
            raw_score = math_verify_fn(prediction_part, answer_part)
            if isinstance(raw_score, tuple):
                score = raw_score[0]
            else:
                score = raw_score
        except Exception:
            score = 0.0

        if float(score) >= 1.0:
            return True

    return _normalize_text(prediction_part) == _normalize_text(answer_part)


def _compute_weighted_score(correctness: list[bool]) -> float:
    if not correctness:
        return 0.0
    if len(correctness) == 1:
        return 1.0 if correctness[0] else 0.0

    if len(correctness) in SUB_QUESTION_WEIGHTS:
        weights = SUB_QUESTION_WEIGHTS[len(correctness)]
        return float(sum(weight for weight, is_correct in zip(weights, correctness) if is_correct))

    return float(sum(1 for is_correct in correctness if is_correct) / len(correctness))


def mathcanvas_doc_to_visual(doc):
    chunks = _extract_question_chunks(doc)
    return [value for chunk_type, value in chunks if chunk_type == "image"]


def mathcanvas_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")

    text_chunks = []
    for chunk_type, value in _extract_question_chunks(doc):
        if chunk_type == "text":
            text_chunks.append(value)
        elif chunk_type == "image":
            text_chunks.append("<image>")

    question = "\n".join(text_chunks).strip()

    prompt_parts = []
    if pre_prompt:
        prompt_parts.append(pre_prompt.strip())
    if question:
        prompt_parts.append(question)
    if post_prompt:
        prompt_parts.append(post_prompt.strip())
    return "\n".join(prompt_parts)


def mathcanvas_process_results(doc, results):
    prediction = ""
    if results:
        prediction = str(results[0]).strip()

    ground_truth_answers = _extract_ground_truth_answers(str(doc.get("answer", "")))
    predicted_answers = _extract_predicted_answers(prediction, expected_parts=len(ground_truth_answers))

    correctness = []
    for index, answer_part in enumerate(ground_truth_answers):
        prediction_part = predicted_answers[index] if index < len(predicted_answers) else ""
        correctness.append(_judge_answer_match(prediction_part, answer_part))

    weighted_score = _compute_weighted_score(correctness)
    complete_score = 1.0 if correctness and all(correctness) else 0.0

    return {
        "mathcanvas_weighted_accuracy": weighted_score,
        "mathcanvas_complete_accuracy": complete_score,
    }
