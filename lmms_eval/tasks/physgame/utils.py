import os
import re
import sys
from pathlib import Path

import yaml
from loguru import logger as eval_logger

# Physics categories from the PhysGame benchmark (4 domains, 12 fine-grained)
PHYSICS_DOMAINS = ["Mechanics", "Kinematics", "Optics", "Material Properties"]

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "physgame.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def physgame_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_id = str(doc["question_id"])
    video_path = os.path.join(cache_dir, video_id + ".mp4")

    if os.path.exists(video_path):
        return [video_path]
    # Try alternate extensions
    for ext in ["MP4", "mkv", "webm", "avi"]:
        alt = video_path.rsplit(".", 1)[0] + "." + ext
        if os.path.exists(alt):
            return [alt]
    sys.exit(f"video path:{video_path} does not exist, please check")


def physgame_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = doc["question"]
    options = doc["options"]
    # options is a dict like {"A": "...", "B": "...", "C": "...", "D": "..."}
    if isinstance(options, dict):
        option_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
    elif isinstance(options, list):
        option_str = "\n".join(options)
    else:
        option_str = str(options)

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    full_prompt = f"{pre_prompt}{question}\n{option_str}{post_prompt}"
    return full_prompt


def _extract_answer(s):
    """Extract a single letter answer [A-D] from model output."""
    s = s.strip()
    # Remove common prefixes
    for prefix in [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "Best answer:",
        "Answer:",
    ]:
        s = s.replace(prefix, "")
    s = s.strip().lstrip(":(").strip()

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    match = re.search(r"[ABCD]", s)
    if match is None:
        return ""
    return match[0]


def physgame_process_results(doc, results):
    pred = results[0]
    pred_ans = _extract_answer(pred)
    gt_ans = str(doc["answer"]).strip().upper()

    return {
        "physgame_accuracy": {
            "question_id": doc["question_id"],
            "pred_answer": pred_ans,
            "answer": gt_ans,
        }
    }


def physgame_aggregate_results(results):
    correct = 0
    total = 0
    for result in results:
        total += 1
        if result["pred_answer"] == result["answer"]:
            correct += 1

    accuracy = 100.0 * correct / total if total > 0 else 0
    eval_logger.info(f"PhysGame Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return accuracy
