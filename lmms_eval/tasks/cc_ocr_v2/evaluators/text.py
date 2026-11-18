"""Text recognition, structured field matching and document QA."""

import ast
import json
import re
from collections import Counter

from Levenshtein import distance

from lmms_eval.tasks.cc_ocr.evaluators.ocr_metric import _text_normalize_and_tokenize

_INVALID_JSON = object()
_WIDTH_TRANSLATION = {i: i - 0xFEE0 for i in range(0xFF01, 0xFF5F)}
_WIDTH_TRANSLATION.update({0x3000: " ", 0xFFE5: "¥", 0x2014: "-", 0x2103: "°C"})
_CJK = re.compile(r"[\u4e00-\u9fff]")


def strip_fence(text):
    text = text.strip()
    matched = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return matched[1].strip() if matched else text


def recognition_f1(matches, predicted, expected):
    recall = matches / (expected + 1e-9)
    precision = matches / (predicted + 1e-9)
    return 2 * recall * precision / (recall + precision + 1e-9)


def recognition(doc, reference, prediction):
    word_level = not _CJK.search(reference)
    multi_scene = doc["scenario"].startswith("multi_scene_ocr") or doc["sub_task"] == "multi_scene_recognition"
    gt = _text_normalize_and_tokenize(reference.strip(), word_level, True, multi_scene and word_level)
    pred = _text_normalize_and_tokenize(strip_fence(prediction), word_level, True, multi_scene and word_level)
    matches = sum((Counter(gt) & Counter(pred)).values())
    return {
        "row_score": recognition_f1(matches, len(pred), len(gt)),
        "matched": matches,
        "predicted": len(pred),
        "expected": len(gt),
    }


def _kie_unfence(text):
    pattern = r"```json\s*(.*?)\s*```" if "```json" in text else r"```\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    return match[1].strip() if match else text


def _read_json(text):
    try:
        return json.loads(_kie_unfence(text.strip()))
    except ValueError:
        return _INVALID_JSON


def _kie_value(value):
    value = value.translate(_WIDTH_TRANSLATION)
    value = value.replace("、", ",").replace("-", "").replace("–", "").replace("’", "'").rstrip("。.")
    return re.sub(r"\s+", "", _kie_unfence(value))


def _normalize_values(value):
    if isinstance(value, dict):
        return {k: _normalize_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_values(v) if isinstance(v, (dict, str)) else v for v in value]
    if isinstance(value, str):
        return _kie_value(value)
    return value


def _fields(value, key=""):
    if isinstance(value, dict):
        for name, child in value.items():
            yield from _fields(child, f"{key}.{name}" if key else name)
    elif isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            for item in value:
                yield from _fields(item, key)
        else:
            for item in value:
                if type(item) in {str, int, float} and str(item).strip():
                    yield key, str(item).strip()
    else:
        yield key, str(value).strip()


def extraction_f1(matches, predicted, expected):
    return matches / (matches + (predicted + expected - 2 * matches) / 2 + 1e-6)


def extraction(reference, prediction):
    gt, pred = _read_json(reference), _read_json(prediction)
    pred_valid = pred is not _INVALID_JSON
    if gt is _INVALID_JSON:
        return {"row_score": None, "valid_prediction": pred_valid}
    expected = Counter(_fields(_normalize_values(gt)))
    predicted = Counter(_fields(_normalize_values(pred))) if pred_valid else Counter()
    matches = sum((expected & predicted).values())
    n_pred, n_gt = sum(predicted.values()), sum(expected.values())
    return {
        "row_score": extraction_f1(matches, n_pred, n_gt),
        "matched": matches,
        "predicted": n_pred,
        "expected": n_gt,
        "valid_prediction": pred_valid,
    }


def _qa_answers(reference):
    raw = reference.strip()
    if raw.startswith("["):
        try:
            answers = json.loads(raw)
        except ValueError:
            try:
                answers = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                answers = None
        if isinstance(answers, list) and answers:
            return answers, True
    return [raw], False


def _qa_match(prediction, answer, chinese, list_reference):
    answer = str(answer).lower().strip().replace("\n", " ")
    if chinese:
        answer = answer.replace(" ", "")
        short = len(answer.split(",")) < 4
        # The scalar Chinese branch only handles strings with fewer than four
        # comma-separated parts. Its ANLS fallback is nested under this check.
        if not short and not list_reference:
            return 0.0
    else:
        short = len(answer.split()) < 5
    if short:
        if answer in prediction:
            return 1.0
        if not chinese or list_reference:
            return 0.0
    similarity = 1.0 - distance(prediction, answer) / max(len(prediction), len(answer))
    return similarity if similarity >= 0.5 else 0.0


def qa(reference, prediction):
    if not reference.strip():
        return None
    pred = strip_fence(prediction).lower().replace("\n", " ")
    if not pred:
        return 0.0
    answers, list_reference = _qa_answers(reference)
    chinese = bool(_CJK.search(" ".join(map(str, answers))))
    if chinese:
        pred = pred.replace(" ", "")
    return max(_qa_match(pred, answer, chinese, list_reference) for answer in answers)
