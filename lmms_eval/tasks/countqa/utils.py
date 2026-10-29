"""CountQA: object counting on hand-captured, cluttered real-world photos.

Each image of ``Jayant-Sravan/CountQA`` carries one or more counting questions;
``countqa_process_docs`` expands them into one document per question. Scoring is
exact match of the predicted integer against the in-situ annotated count.
"""

import re
from typing import Any, Dict, List, Optional

import datasets
from PIL import Image

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_INT_RE = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d,])|(?<![\d.])\d+(?!\.\d)")
_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)


def countqa_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Expand every image row into one document per (question, answer) pair."""

    def _expand(batch: Dict[str, List[Any]], indices: List[int]) -> Dict[str, List[Any]]:
        out: Dict[str, List[Any]] = {key: [] for key in ("image", "question", "answer", "image_index", "qa_index", "is_focused", "categories", "objects")}
        for row, image_index in enumerate(indices):
            for qa_index, (question, answer) in enumerate(zip(batch["questions"][row], batch["answers"][row])):
                out["image"].append(batch["image"][row])
                out["question"].append(str(question).strip())
                out["answer"].append(str(answer).strip())
                out["image_index"].append(image_index)
                out["qa_index"].append(qa_index)
                out["is_focused"].append(bool(batch["is_focused"][row]))
                out["categories"].append(list(batch["categories"][row] or []))
                out["objects"].append([str(obj).strip() for obj in (batch["objects"][row] or [])])
        return out

    return dataset.map(_expand, batched=True, batch_size=16, with_indices=True, remove_columns=dataset.column_names)


def countqa_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    """Return the image of the document as RGB."""
    return [doc["image"].convert("RGB")]


def countqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Build the counting prompt from the question and the configured pre/post prompts."""
    kwargs = lmms_eval_specific_kwargs or {}
    return f"{kwargs.get('pre_prompt', '')}{doc['question']}{kwargs.get('post_prompt', '')}"


def parse_count(response: str) -> Optional[int]:
    """Extract the first integer of the final answer, or ``None`` if there is none.

    Text up to the last ``</think>`` is dropped so that reasoning models are scored
    on their final answer. Thousands separators (``1,200``) and the number words
    zero-twenty are accepted; decimals are not counts and are ignored.
    """
    text = response.rsplit("</think>", 1)[-1]
    match = _INT_RE.search(text)
    if match:
        return int(match.group().replace(",", ""))
    word = _WORD_RE.search(text)
    if word:
        return _NUMBER_WORDS[word.group(1).lower()]
    return None


def countqa_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    """Score one prediction; every metric receives the same per-sample record."""
    pred = parse_count(results[0] if results else "")
    target = int(doc["answer"])
    record = {
        "image_index": doc["image_index"],
        "qa_index": doc["qa_index"],
        "is_focused": bool(doc["is_focused"]),
        "target": target,
        "pred": pred,
        "correct": pred == target,
        "off_by_one": pred is not None and abs(pred - target) <= 1,
    }
    return {metric: record for metric in ("countqa_acc", "countqa_off_by_one_acc", "countqa_focused_acc", "countqa_unfocused_acc")}


def _mean(values: List[bool]) -> float:
    return sum(values) / len(values) if values else float("nan")


def countqa_aggregate_acc(results: List[Dict[str, Any]]) -> float:
    """Exact-match accuracy over all questions."""
    return _mean([r["correct"] for r in results])


def countqa_aggregate_off_by_one_acc(results: List[Dict[str, Any]]) -> float:
    """Share of answers within one of the true count."""
    return _mean([r["off_by_one"] for r in results])


def countqa_aggregate_focused_acc(results: List[Dict[str, Any]]) -> float:
    """Exact-match accuracy on images annotated as close-up (``is_focused``)."""
    return _mean([r["correct"] for r in results if r["is_focused"]])


def countqa_aggregate_unfocused_acc(results: List[Dict[str, Any]]) -> float:
    """Exact-match accuracy on wide-shot images where the target is one of many objects."""
    return _mean([r["correct"] for r in results if not r["is_focused"]])
