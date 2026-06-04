import os
import sys
from pathlib import Path

import yaml
from loguru import logger as eval_logger

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "videoeval_pro.yaml", "r") as f:
    raw_data_test = f.readlines()
    safe_data_test = []
    for i, line in enumerate(raw_data_test):
        if "!function" not in line:
            safe_data_test.append(line)
cache_name = yaml.safe_load("".join(safe_data_test))["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(base_cache_dir, cache_name)


def videoeval_pro_doc_to_visual(doc):
    video_path = doc["video"]
    video_path = os.path.join(os.path.join(cache_dir, "videos_filtered"), video_path)
    if os.path.exists(video_path):
        return [video_path]
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")


def videoeval_pro_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    options = doc.get("options", [])
    option_str = "\n".join(options)
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    full_prompt = f"{pre_prompt}{question}\n{option_str}{post_prompt}"
    return full_prompt


def videoeval_pro_process_results(doc, results):
    """Process results for MCQ evaluation (letter matching)."""
    pred_ans = results[0].strip()
    answer = doc.get("answer", "").strip()

    # Extract first letter from prediction
    pred_letter = ""
    for ch in pred_ans:
        if ch.upper() in "ABCDE":
            pred_letter = ch.upper()
            break

    score = 1.0 if pred_letter == answer.upper() else 0.0

    data_dict = {
        "video": doc.get("video", ""),
        "question": doc.get("question", ""),
        "source": doc.get("source", ""),
        "qa_type": doc.get("qa_type", ""),
        "qa_subtype": doc.get("qa_subtype", ""),
        "pred_answer": pred_ans,
        "answer": answer,
        "answer_text": doc.get("answer_text", ""),
        "score": score,
    }

    return {"videoeval_pro_score": data_dict}


def videoeval_pro_aggregate_results(results):
    """Aggregate results by category and overall."""
    TASK_TYPES = {
        "Local Perception",
        "Local Reasoning",
        "Holistic Perception",
        "Holistic Reasoning",
    }
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}
    category2score["overall"] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result.get("qa_type", "")
        if task_type in category2score:
            category2score[task_type]["answered"] += 1
            category2score[task_type]["correct"] += result["score"]
        category2score["overall"]["answered"] += 1
        category2score["overall"]["correct"] += result["score"]

    final_scores = {}
    for task_type, score in category2score.items():
        if score["answered"] > 0:
            final_scores[task_type] = score["correct"] / score["answered"]
        else:
            final_scores[task_type] = 0.0

    eval_logger.info(f"Final scores: {final_scores}")

    return final_scores
