import os
from collections import defaultdict

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

# Physics categories from the PhysGame benchmark (4 domains, 12 fine-grained)
PHYSICS_DOMAINS = ["Mechanics", "Kinematics", "Optics", "Material Properties"]


def _get_cache_dir():
    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
    return os.path.join(hf_home, "physgame")


def physgame_doc_to_visual(doc):
    cache_dir = _get_cache_dir()
    video_path = os.path.join(cache_dir, doc["video_path"])

    if os.path.exists(video_path):
        return [video_path]

    raise FileNotFoundError(f"PhysGame video not found: {video_path}")


def physgame_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = doc["question"]
    options = doc["options"]
    # options is a dict like {"A": "...", "B": "...", "C": "...", "D": "..."}
    if isinstance(options, dict):
        option_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
    elif isinstance(options, list):
        option_str = "\n".join(options)
    else:
        option_str = str(options)

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    full_prompt = f"{pre_prompt}{question}\n{option_str}{post_prompt}"
    return full_prompt


def physgame_process_results(doc, results):
    pred = results[0]
    pred_ans = extract_mcq_answer(pred, choices=["A", "B", "C", "D"])
    gt_ans = str(doc["answer"]).strip().upper()

    return {
        "physgame_accuracy": {
            "question_id": doc["question_id"],
            "class_anno": doc.get("class_anno", "Unknown"),
            "subclass_anno": doc.get("subclass_anno", "Unknown"),
            "pred_answer": pred_ans,
            "answer": gt_ans,
        }
    }


def physgame_aggregate_results(results):
    correct = 0
    total = 0
    domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in results:
        total += 1
        domain = result.get("class_anno", "Unknown")
        domain_stats[domain]["total"] += 1
        is_correct = result["pred_answer"] == result["answer"]
        if is_correct:
            correct += 1
            domain_stats[domain]["correct"] += 1

    for domain in PHYSICS_DOMAINS:
        stats = domain_stats[domain]
        if stats["total"]:
            accuracy = 100.0 * stats["correct"] / stats["total"]
            eval_logger.info("PhysGame [{}]: {:.1f}% ({}/{})", domain, accuracy, stats["correct"], stats["total"])

    accuracy = 100.0 * correct / total if total > 0 else 0
    eval_logger.info(f"PhysGame Overall Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return accuracy
