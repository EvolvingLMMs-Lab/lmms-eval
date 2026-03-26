"""COVER benchmark -- counterfactual video reasoning (ACL Findings 2025).

Dataset: lmms-lab-eval/COVER (clean QA data, parsed from PeterPanonly/COVER)
Paper:   https://arxiv.org/abs/2503.10691

The clean dataset has pre-parsed QA pairs with columns:
  src_dataset, video_name, question, choices (JSON), answer, qa_type, aspect.

Videos: VIDEO.zip must be extracted to $HF_HOME/cover/VIDEO/ or set
COVER_DATA_DIR to the video root directory.
"""

import json
import os
import zipfile
from collections import defaultdict

from huggingface_hub import snapshot_download
from loguru import logger as eval_logger

# ---------------------------------------------------------------------------
# Video directory resolution
# ---------------------------------------------------------------------------
_hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
_CACHE_DIR = os.path.join(_hf_home, "cover")


def _get_cache_dir():
    explicit = os.getenv("COVER_DATA_DIR", "").strip()
    if explicit:
        return os.path.expanduser(explicit)
    return _CACHE_DIR


_videos_ready = False


def _ensure_videos():
    """Download VIDEO.zip from PeterPanonly/COVER and extract if needed."""
    global _videos_ready
    if _videos_ready:
        return

    cache_dir = _get_cache_dir()
    video_dir = os.path.join(cache_dir, "VIDEO")
    if os.path.exists(video_dir):
        _videos_ready = True
        return

    eval_logger.info("COVER: downloading VIDEO.zip from PeterPanonly/COVER ...")
    repo_dir = snapshot_download(
        repo_id="PeterPanonly/COVER",
        repo_type="dataset",
        etag_timeout=60,
    )

    video_zip = os.path.join(repo_dir, "VIDEO.zip")
    if os.path.exists(video_zip):
        eval_logger.info(f"COVER: extracting VIDEO.zip to {cache_dir} ...")
        os.makedirs(cache_dir, exist_ok=True)
        with zipfile.ZipFile(video_zip) as zf:
            zf.extractall(cache_dir)
        eval_logger.info("COVER: extraction complete.")
    else:
        eval_logger.warning(f"COVER: VIDEO.zip not found at {video_zip}. " f"Videos must be placed manually in {video_dir}.")
    _videos_ready = True


# ---------------------------------------------------------------------------
# doc_to_visual / doc_to_text
# ---------------------------------------------------------------------------
def cover_doc_to_visual(doc):
    _ensure_videos()
    cache_dir = _get_cache_dir()
    video_path = os.path.join(cache_dir, "VIDEO", doc["src_dataset"], doc["video_name"])

    if os.path.exists(video_path):
        return [video_path]

    # Try extension variants
    base, ext = os.path.splitext(video_path)
    for alt_ext in [".mp4", ".MP4", ".avi", ".AVI", ".mkv"]:
        if alt_ext != ext:
            alt = base + alt_ext
            if os.path.exists(alt):
                return [alt]

    eval_logger.warning(f"COVER video not found: {video_path}")
    return [video_path]


def cover_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    choices = doc["choices"]
    if isinstance(choices, str):
        choices = json.loads(choices)

    sorted_keys = sorted(choices.keys())
    choices_str = "\n".join(f"{k}. {choices[k]}" for k in sorted_keys)

    post_prompt = ""
    if lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs.get("default", {}).get("post_prompt", "")

    return f"{question}\n{choices_str}{post_prompt}"


# ---------------------------------------------------------------------------
# process_results / aggregate_results
# ---------------------------------------------------------------------------
def _extract_answer(response):
    import re

    response = response.strip()
    if not response:
        return ""
    # Direct single letter
    if len(response) == 1 and response.upper() in "ABCDEF":
        return response.upper()
    # Pattern: (A), A., A:
    m = re.match(r"[\(\s]*([A-Fa-f])[\)\.\:\s]", response)
    if m:
        return m.group(1).upper()
    # First letter
    m = re.match(r"^([A-Fa-f])\b", response)
    if m:
        return m.group(1).upper()
    # Search in short response
    if len(response) < 50:
        m = re.search(r"\b([A-Da-d])\b", response)
        if m:
            return m.group(1).upper()
    return response[:1].upper()


def cover_process_results(doc, results):
    pred = _extract_answer(results[0])
    gt = doc["answer"].strip().upper()

    return {
        "cover_accuracy": {
            "pred_answer": pred,
            "answer": gt,
            "qa_type": doc.get("qa_type", ""),
            "aspect": doc.get("aspect", ""),
            "score": 1.0 if pred == gt else 0.0,
        }
    }


def cover_aggregate_results(results):
    total = len(results)
    correct = sum(r["score"] for r in results)

    # Per qa_type
    for qt in ("original", "counterfactual"):
        subset = [r for r in results if r["qa_type"] == qt]
        if subset:
            acc = 100.0 * sum(r["score"] for r in subset) / len(subset)
            eval_logger.info(f"COVER {qt}: {acc:.1f}% ({len(subset)} samples)")

    # Per aspect
    aspect_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        aspect_scores[r["aspect"]]["total"] += 1
        aspect_scores[r["aspect"]]["correct"] += r["score"]
    for aspect in sorted(aspect_scores):
        s = aspect_scores[aspect]
        acc = 100.0 * s["correct"] / s["total"] if s["total"] > 0 else 0.0
        eval_logger.info(f"COVER {aspect}: {acc:.1f}% ({s['total']} samples)")

    overall = 100.0 * correct / total if total > 0 else 0.0
    eval_logger.info(f"COVER Overall: {overall:.1f}% ({total} samples)")
    return overall
