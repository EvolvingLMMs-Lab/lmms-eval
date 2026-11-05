import os
from collections import defaultdict
from typing import Optional

from loguru import logger as eval_logger

from lmms_eval.utils import resolve_cache_dir

# Must match `dataset_kwargs.cache_dir` in the task yaml files.
CACHE_NAME = "aotbench"


def aotbench_doc_to_visual(doc: dict) -> list[str]:
    """Return the path of the extracted video for a document."""
    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
    video_path = os.path.join(resolve_cache_dir(CACHE_NAME, base_dir=hf_home), doc["video_name"])
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video path: {video_path} does not exist, please check")
    return [video_path]


def aotbench_doc_to_text(doc: dict, lmms_eval_specific_kwargs: Optional[dict] = None) -> str:
    """Return the question, which already holds the options and the answer instruction."""
    return doc["question"]


def aotbench_process_results(doc: dict, results: list[str]) -> dict:
    """Score a reply with the rule of eval/read_qa.py in the official repo.

    The first character of the reply is the prediction. Surrounding whitespace
    is ignored and an empty reply counts as wrong.
    """
    prediction = results[0].strip()[:1]
    return {"aotbench_acc": {"source": doc.get("question_type"), "correct": prediction == doc["ans"]}}


def aotbench_aggregate(results: list[dict]) -> float:
    """Return overall accuracy and log accuracy per source benchmark."""
    by_source = defaultdict(list)
    for result in results:
        if result["source"] is not None:
            by_source[result["source"]].append(result["correct"])
    for source, correct in sorted(by_source.items()):
        eval_logger.info(f"AoTBench {source}: {sum(correct) / len(correct):.4f} ({len(correct)} questions)")
    return sum(result["correct"] for result in results) / len(results)
