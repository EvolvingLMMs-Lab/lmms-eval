"""Spatial Scene Bench: counting and relational reasoning over procedural 3D scenes.

Scoring follows the official evaluator (https://github.com/selfishout/spatial-scene-bench):
exact match for count / yes-no / multiple-choice answers and VSI-Bench Mean Relative
Accuracy (MRA) for metric distances. The headline score is the macro average over
question types.
"""

import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np

QUESTION_TYPES = [
    "count",
    "count_relation",
    "relation",
    "compare_count",
    "closest",
    "distance",
    "ego_direction",
    "video_count",
    "video_closest",
    "video_distance",
    "video_ego_direction",
]

ANSWER_FORMATS = {
    "count": "Answer with a single integer.",
    "yesno": "Answer with yes or no.",
    "choice": "Answer with the letter of the correct option.",
    "numeric": "Answer with a single number.",
}

NUMBER_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}


def ssb_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    """Return the view (image questions) or the ordered orbit frames (video questions) as RGB images."""
    return [image.convert("RGB") for image in doc["images"]]


def ssb_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Build the question prompt with an answer-format instruction for the answer type."""
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") or ANSWER_FORMATS[doc["answer_type"]]
    return f"{pre_prompt}{doc['question']}\n{post_prompt}"


def ssb_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Build a chat message with all images first, followed by the prompt."""
    content = [{"type": "image", "url": image} for image in ssb_doc_to_visual(doc)]
    content.append({"type": "text", "text": ssb_doc_to_text(doc, lmms_eval_specific_kwargs)})
    return [{"role": "user", "content": content}]


def _final_segment(text: str) -> str:
    tagged = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if tagged:
        return tagged[-1].strip()
    m = re.search(r"(?:final answer|answer)\s*(?:is|:)\s*(.+)", text, flags=re.I)
    return m.group(1).strip() if m else text.strip()


def parse_answer(text: str, answer_type: str, options: Optional[List[str]] = None) -> Optional[str]:
    """Extract a normalized answer string, or None if the response contains no valid answer."""
    seg = _final_segment(text or "")
    low = seg.lower()
    if answer_type in ("count", "numeric"):
        m = re.search(r"-?\d+(?:\.\d+)?", seg)
        if m:
            value = float(m.group())
            return str(int(round(value))) if answer_type == "count" else f"{value:g}"
        for word, value in NUMBER_WORDS.items():
            if re.search(rf"\b{word}\b", low):
                return str(value)
        return None
    if answer_type == "yesno":
        m = re.search(r"\b(yes|no)\b", low)
        return m.group(1) if m else None
    if answer_type == "choice":
        m = re.search(r"^\(?([A-D])\)?(?:[\s.:)]|$)", seg.strip())
        if m:
            return m.group(1)
        m = re.search(r"\(([A-D])\)", seg)
        if m:
            return m.group(1)
        if options:
            hits = [i for i, option in enumerate(options) if option.lower() in low]
            if len(hits) == 1:
                return "ABCD"[hits[0]]
        m = re.search(r"\b([A-D])\b", seg)
        return m.group(1) if m else None
    raise ValueError(f"Unknown answer type: {answer_type}")


def mean_relative_accuracy(pred: float, target: float) -> float:
    """VSI-Bench MRA: mean over theta in {0.50, ..., 0.95} of 1[|pred - target| / target < 1 - theta]."""
    if target == 0:
        return float(pred == 0)
    rel = abs(pred - target) / abs(target)
    return float(np.mean([rel < 1 - t for t in np.arange(0.50, 0.951, 0.05)]))


def score(doc: Dict[str, Any], prediction: Optional[str]) -> float:
    """Score a parsed prediction: MRA for metric distances, exact match otherwise."""
    if prediction is None:
        return 0.0
    if doc["answer_type"] == "numeric":
        return mean_relative_accuracy(float(prediction), float(doc["answer"]))
    if doc["answer_type"] == "count":
        return float(int(prediction) == int(doc["answer"]))
    return float(prediction == doc["answer"])


def ssb_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    """Parse and score one response; the same payload feeds every aggregate metric."""
    pred = parse_answer(results[0] if results else "", doc["answer_type"], doc.get("options"))
    payload = {"id": doc["id"], "question_type": doc["question_type"], "score": score(doc, pred)}
    if doc["answer_type"] == "count":
        payload["mra"] = 0.0 if pred is None else mean_relative_accuracy(float(pred), float(doc["answer"]))
    metrics = {"ssb_overall": payload, "ssb_image_overall": payload, "ssb_video_overall": payload, "ssb_count_mra": payload}
    metrics.update({f"{t}_accuracy": payload for t in QUESTION_TYPES})
    return metrics


def _per_type(results: List[Dict[str, Any]]) -> Dict[str, float]:
    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r["score"])
    return {t: 100.0 * float(np.mean(v)) for t, v in by_type.items()}


def _macro(results: List[Dict[str, Any]], keep: Callable[[str], bool] = lambda t: True) -> float:
    scores = [v for t, v in _per_type(results).items() if keep(t)]
    return float(np.mean(scores)) if scores else float("nan")


def ssb_aggregate_overall(results: List[Dict[str, Any]]) -> float:
    """Macro average of per-type scores over all question types present."""
    return _macro(results)


def ssb_aggregate_image_overall(results: List[Dict[str, Any]]) -> float:
    """Macro average over single-image question types."""
    return _macro(results, lambda t: not t.startswith("video_"))


def ssb_aggregate_video_overall(results: List[Dict[str, Any]]) -> float:
    """Macro average over video question types."""
    return _macro(results, lambda t: t.startswith("video_"))


def ssb_aggregate_count_mra(results: List[Dict[str, Any]]) -> float:
    """Mean Relative Accuracy over all counting questions."""
    mras = [r["mra"] for r in results if "mra" in r]
    return 100.0 * float(np.mean(mras)) if mras else float("nan")


def _type_aggregator(question_type: str) -> Callable[[List[Dict[str, Any]]], float]:
    def aggregate(results: List[Dict[str, Any]]) -> float:
        return _per_type(results).get(question_type, float("nan"))

    aggregate.__name__ = f"ssb_aggregate_{question_type}"
    return aggregate


ssb_aggregate_count = _type_aggregator("count")
ssb_aggregate_count_relation = _type_aggregator("count_relation")
ssb_aggregate_relation = _type_aggregator("relation")
ssb_aggregate_compare_count = _type_aggregator("compare_count")
ssb_aggregate_closest = _type_aggregator("closest")
ssb_aggregate_distance = _type_aggregator("distance")
ssb_aggregate_ego_direction = _type_aggregator("ego_direction")
ssb_aggregate_video_count = _type_aggregator("video_count")
ssb_aggregate_video_closest = _type_aggregator("video_closest")
ssb_aggregate_video_distance = _type_aggregator("video_distance")
ssb_aggregate_video_ego_direction = _type_aggregator("video_ego_direction")
