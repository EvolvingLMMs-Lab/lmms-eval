"""Answer scoring compatible with the author-linked Doc2SoarGraph evaluator.

Reference: fengbinzhu/Doc2SoarGraph, commit
71a02715849942b79c1f74934a139e4b56fc6826, tatqa_metric.py and tatqa_utils.py.
The reference joins sorted answer spans before computing token-set F1; it does
not call the alternative percent-prediction helper found in TAT-QA.
"""

import math
import re
import string
from typing import Any

import numpy as np

_NUM_EXCLUDE = str.maketrans("", "", "'\"\\$€£¥%(),[]")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def _scale_value(scale: str) -> float:
    for name, value in (
        ("hundred", 100),
        ("thousand", 1000),
        ("million", 1000000),
        ("billion", 1000000000),
        ("percent", 0.01),
    ):
        if name in scale.lower():
            return value
    return 1


def _is_number(text: str) -> bool:
    words = " ".join(word.translate(_NUM_EXCLUDE) for word in text.split()).split()
    try:
        return bool(words) and not math.isnan(float(words[0])) and (len(words) < 2 or _scale_value(words[1]) != 1)
    except ValueError:
        return False


def _to_number(text: str) -> float | int | None:
    matches = re.findall(r"([+-]?\d+(\.\d+)?)|([+-]?\.\d+)", text.translate(_NUM_EXCLUDE))
    if not matches or not matches[0][0]:
        return None
    first = matches[0][0]
    value = float(first) if "." in first else int(first)
    word_scale = re.search(r"([\d.]+\s?[a-zA-Z]+)", text)
    scale = _scale_value(word_scale.group(0)) if word_scale else 1
    negative = -1 if re.search(r"(\([\d.\s%]+\))", text.strip()) else 1
    percent = 0.01 if re.search(r"([\d.\s]+%)", text.strip()) else 1
    return round(value * scale * negative * percent, 4)


def _normalize(text: str) -> str:
    parts = []
    for token in text.split(" "):
        token = token.lower()
        if not _is_number(token):
            token = token.translate(_PUNCTUATION)
        if _is_number(token):
            token = str(_to_number(token))
        token = " ".join(re.sub(r"\b(a|an|the)\b", " ", token).split())
        if token.strip():
            parts.append(token)
    return " ".join(parts).strip()


def _answer_string(answers: list[Any], scale: str) -> str:
    parts = []
    for answer in sorted(answers, key=str):
        text = str(answer)
        number = _to_number(text) if _is_number(text) else None
        if number is not None:
            number = number if "%" in text else round(number, 2) * _scale_value(scale)
            text = "%.4f" % number  # noqa: UP031 -- Preserve reference numeric formatting.
        elif scale:
            text += " " + scale
        parts.append(text)
    return _normalize(" ".join(parts))


def score_answer(gold: dict[str, Any], prediction: Any, pred_scale: str | None) -> tuple[float, float, float]:
    """Return per-question EM, F1 and scale accuracy, including missing answers."""
    scale_accuracy = float(pred_scale == gold["scale"])
    if prediction is None or prediction == "":
        return 0.0, 0.0, scale_accuracy
    answer_type = gold["answer_type"]
    if answer_type in {"span", "multi-span"}:
        answers = gold["answer"]
        if not isinstance(answers, list):
            raise ValueError("TAT-DQA span gold must be a list")
    elif answer_type == "count":
        answers = [str(int(gold["answer"]))]
    else:
        answers = [str(gold["answer"])]
    expected = _answer_string(answers, gold["scale"])
    predicted_answers = prediction if isinstance(prediction, list) else [prediction]
    try:
        actual = _answer_string(predicted_answers, pred_scale or "")
    except (ValueError, TypeError, OverflowError):
        # The reference get_answer_str falls back to an empty string here.
        actual = ""
    exact = float(actual == expected)
    if answer_type in {"arithmetic", "count"}:
        return exact, exact, scale_accuracy
    actual_tokens, expected_tokens = set(actual.split()), set(expected.split())
    common = len(actual_tokens & expected_tokens)
    precision = common / len(actual_tokens) if actual_tokens else 1.0
    recall = common / len(expected_tokens) if expected_tokens else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # The reference rounds a NumPy float64 produced by its single-bag mean.
    return exact, float(round(np.float64(f1), 2)), scale_accuracy
