import re

import numpy as np

import datasets
from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference

CACHE_DIR = "vsisuper_count"
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def doc_to_visual(doc):
    video_path = resolve_media_reference(doc["video_path"], media_type="video", cache_dir=CACHE_DIR, env_vars=("VSISUPER_VIDEO_DIR",))
    return [video_path]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = str(doc["question"]).strip()
    return "These are frames of a video.\n" + question + "\nPlease answer the question using a single word or phrase."


def process_docs_10mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "10mins")


def process_docs_30mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "30mins")


def process_docs_60mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "60mins")


def process_docs_120mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "120mins")


def _extract_number(text):
    if text is None:
        return 0.0
    match = _NUMBER_RE.search(str(text))
    if not match:
        return 0.0
    return float(match.group(0))


def _mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    target = float(target)
    pred = float(pred)

    if target == 0.0:
        return 1.0 if pred == 0.0 else 0.0

    relative_error = abs(pred - target) / abs(target)
    num_pts = int((end - start) / interval) + 2
    conf_intervals = np.linspace(start, end, num_pts)
    return float((relative_error <= (1.0 - conf_intervals)).mean())


def process_results(doc, results):
    prediction = results[0] if results else ""
    target = float(doc.get("answer", 0.0) or 0.0)
    pred = _extract_number(prediction)
    return {"mra": _mean_relative_accuracy(pred, target)}
