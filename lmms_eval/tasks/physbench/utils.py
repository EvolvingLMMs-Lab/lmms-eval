import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

import datasets
import yaml
from loguru import logger as eval_logger

from lmms_eval import utils as lmms_utils
from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

# PhysBench category breakdowns
TASK_TYPES = ["property", "relationships", "scene", "dynamics"]
ABILITY_TYPES = ["identify", "comparison", "static", "dynamic", "perception", "prediction", "judgment", "reasoning"]

# URL for val split answers (answers not included in HF dataset)
VAL_ANSWER_URL = "https://raw.githubusercontent.com/USC-GVL/PhysBench/main/eval/physbench/val_answer.json"

# Cached answer map: idx -> {answer, task_type, sub_type, ability_type}
_val_answers = None


def _fetch_val_answers():
    """Download and cache the val answer file from the PhysBench GitHub repo."""
    global _val_answers
    if _val_answers is not None:
        return _val_answers

    eval_logger.info(f"Fetching PhysBench val answers from {VAL_ANSWER_URL}")
    try:
        with urllib.request.urlopen(VAL_ANSWER_URL, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        _val_answers = {item["idx"]: item for item in data}
        eval_logger.info(f"Loaded {len(_val_answers)} val answers")
    except Exception as e:
        eval_logger.warning(f"Failed to fetch val answers: {e}. Accuracy will not be computed.")
        _val_answers = {}
    return _val_answers


def _load_task_config():
    with open(Path(__file__).parent / "physbench.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = [line for line in raw_data if "!function" not in line]
    return yaml.safe_load("".join(safe_data))


def _get_cache_dir():
    config = _load_task_config()
    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
    return lmms_utils.resolve_cache_dir(config["dataset_kwargs"]["cache_dir"], base_dir=hf_home)


def physbench_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Filter to val split only (test answers are hidden) and merge answers."""
    # Filter to val entries only
    dataset = dataset.filter(lambda x: x["split"] == "val")

    # Fetch and merge answer metadata
    answers = _fetch_val_answers()
    if not answers:
        return dataset

    def _merge_answer(example):
        idx = example["idx"]
        ans_info = answers.get(idx, {})
        example["answer"] = ans_info.get("answer", "")
        example["task_type"] = ans_info.get("task_type", "")
        example["sub_type"] = ans_info.get("sub_type", "")
        example["ability_type"] = ans_info.get("ability_type", "")
        return example

    dataset = dataset.map(_merge_answer)
    return dataset


def _resolve_media_path(cache_dir, fname):
    """Resolve media file path, checking subdirectory and flat layouts.

    The PhysBench README instructs users to ``unzip image.zip -d image`` which
    produces ``{cache_dir}/image/foo.png``.  The lmms-eval framework's zip
    extraction, however, extracts flat into ``cache_dir`` giving
    ``{cache_dir}/foo.png``.  We check both locations.
    """
    ext = os.path.splitext(fname)[1].lower()
    subdir = "video" if ext == ".mp4" else "image"

    # Prefer the subdirectory layout (manual unzip / README instructions)
    subdir_path = os.path.join(cache_dir, subdir, fname)
    if os.path.exists(subdir_path):
        return subdir_path

    # Fall back to flat layout (framework zip extraction)
    flat_path = os.path.join(cache_dir, fname)
    if os.path.exists(flat_path):
        return flat_path

    # Neither found; return the subdirectory path for the warning message
    eval_logger.warning(f"PhysBench media file not found: {subdir_path} (also checked {flat_path})")
    return subdir_path


def physbench_doc_to_visual(doc):
    """Return list of media paths (images and videos) for a document."""
    cache_dir = _get_cache_dir()
    # Clean dataset uses media_path (string), original uses file_name (list)
    media = doc.get("file_name") or doc.get("media_path", "")
    if isinstance(media, str):
        media = [media] if media else []
    return [_resolve_media_path(cache_dir, fname) for fname in media]


def physbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Format the question text.

    The question field already contains inline options (A-D) with <video>/<image>
    placeholders. The lmms-eval framework handles visual injection, so we keep
    the placeholders as-is and append the post_prompt.
    """
    question = doc["question"].strip()

    post_prompt = ""
    if lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return question + post_prompt


def physbench_process_results(doc, results):
    """Extract predicted answer and compare to ground truth."""
    pred = results[0]
    pred_ans = extract_mcq_answer(pred, choices=["A", "B", "C", "D"])
    gt_ans = doc.get("answer", "")

    score = 1.0 if pred_ans.upper() == gt_ans.upper() and gt_ans else 0.0

    data_dict = {
        "idx": doc.get("idx", doc.get("index", "")),
        "task_type": doc.get("task_type", ""),
        "sub_type": doc.get("sub_type", ""),
        "ability_type": doc.get("ability_type", ""),
        "pred_answer": pred_ans,
        "answer": gt_ans,
        "score": score,
    }

    return {"physbench_accuracy": data_dict}


def physbench_aggregate_results(results):
    """Compute overall and per-category accuracy."""
    # Per task_type breakdown
    task_type_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    # Per ability_type breakdown
    ability_type_scores = defaultdict(lambda: {"correct": 0, "total": 0})

    total_correct = 0
    total_answered = 0

    for result in results:
        total_answered += 1
        correct = result["score"]
        total_correct += correct

        task_type = result.get("task_type", "")
        if task_type:
            task_type_scores[task_type]["total"] += 1
            task_type_scores[task_type]["correct"] += correct

        ability_type = result.get("ability_type", "")
        if ability_type:
            ability_type_scores[ability_type]["total"] += 1
            ability_type_scores[ability_type]["correct"] += correct

    # Log per-category results
    for task_type in TASK_TYPES:
        stats = task_type_scores.get(task_type, {"correct": 0, "total": 0})
        if stats["total"] > 0:
            acc = 100 * stats["correct"] / stats["total"]
            eval_logger.info(f"PhysBench task_type={task_type}: {acc:.1f}% ({int(stats['correct'])}/{stats['total']})")

    for ability_type in ABILITY_TYPES:
        stats = ability_type_scores.get(ability_type, {"correct": 0, "total": 0})
        if stats["total"] > 0:
            acc = 100 * stats["correct"] / stats["total"]
            eval_logger.info(f"PhysBench ability_type={ability_type}: {acc:.1f}% ({int(stats['correct'])}/{stats['total']})")

    overall_acc = 100 * total_correct / total_answered if total_answered > 0 else 0
    eval_logger.info(f"PhysBench Overall: {overall_acc:.1f}% ({int(total_correct)}/{total_answered})")
    return overall_acc
