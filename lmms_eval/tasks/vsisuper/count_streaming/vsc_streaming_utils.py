import json

import numpy as np

import datasets
from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference

CACHE_DIR = "vsisuper_count"


def doc_to_visual(doc):
    video_path = resolve_media_reference(doc["video_path"], media_type="video", cache_dir=CACHE_DIR, env_vars=("VSISUPER_VIDEO_DIR",))
    return [video_path]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = str(doc["question"]).strip()
    return "These are frames of a video.\n" + question + "\nPlease answer the question using a single word or phrase."


def process_docs_streaming_10mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "10mins_streaming")


def process_docs_streaming_30mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "30mins_streaming")


def process_docs_streaming_60mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "60mins_streaming")


def process_docs_streaming_120mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["split"] == "120mins_streaming")


def _abs_dist_norm(pred, target):
    try:
        return abs(pred - target) / target
    except BaseException:
        return 0.0


def _mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = _abs_dist_norm(pred, target) <= 1 - conf_intervs
    return float(np.mean(accuracy))


def _parse_prediction_list(text):
    return json.loads(text)


def process_results(doc, results):
    prediction = results[0] if results else "[]"
    parsed_results = _parse_prediction_list(prediction)

    mra_scores = []
    for streaming_output, answer in zip(parsed_results, doc["answers"]):
        mra_scores.append(_mean_relative_accuracy(streaming_output, answer))

    return {"mra": float(np.mean(mra_scores)) if mra_scores else 0.0}
