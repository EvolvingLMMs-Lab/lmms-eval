import re
from loguru import logger as eval_logger

import datasets
from collections import OrderedDict

from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference

CACHE_DIR = "vsisuper_recall"
_OPTION_RE = re.compile(r"[A-D]")


def doc_to_visual(doc):
    video_path = resolve_media_reference(doc["video_path"], media_type="video", cache_dir=CACHE_DIR, env_vars=("VSISUPER_VIDEO_DIR",))
    return [video_path]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = str(doc["question"]).strip()
    options = doc.get("options") or []
    options_text = "\n".join(str(option) for option in options)

    return question + "\nOptions:\n" + options_text + "\nAnswer with the option's letter from the given choices directly."


def process_docs_10mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["type"] == "10mins")


def process_docs_30mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["type"] == "30mins")


def process_docs_60mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["type"] == "60mins")


def process_docs_120mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["type"] == "120mins")


def process_docs_240mins(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["type"] == "240mins")


def _normalize_option(text):
    if text is None:
        return ""
    match = _OPTION_RE.search(str(text).upper())
    return match.group(0) if match else ""


def process_results(doc, results):
    prediction = _normalize_option(results[0] if results else "")
    target = _normalize_option(doc.get("answer", ""))
    return {"accuracy": 1.0 if prediction == target else 0.0}
