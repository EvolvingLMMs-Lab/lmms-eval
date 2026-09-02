import os
import re

from loguru import logger as eval_logger

DOMAINS = ["Electromagnetism", "Mechanics", "Optics", "Thermodynamics"]


def _get_cache_dir():
    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
    return os.path.join(hf_home, "physics_rw")


def physics_rw_doc_to_visual(doc):
    cache_dir = _get_cache_dir()
    video_path = os.path.join(cache_dir, doc.get("video_path", ""))

    if os.path.exists(video_path):
        return [video_path]

    raise FileNotFoundError(f"Physics-RW video not found: {video_path}")


def physics_rw_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    instruction = doc.get("instruction", "")
    return f"{pre_prompt}{instruction}{post_prompt}"


def _extract_yes_no(text):
    """Extract yes/no only when it is the first output word, as in the paper."""
    text = text.strip().lower()
    if re.match(r"^yes\b", text):
        return "yes"
    if re.match(r"^no\b", text):
        return "no"
    return ""


def physics_rw_process_results(doc, results):
    pred = results[0]
    pred_ans = _extract_yes_no(pred)
    gt_ans = doc.get("label", "").strip().lower()
    domain = doc.get("domain", "Unknown")

    result = {
        "id": doc.get("id", ""),
        "domain": domain,
        "pred_answer": pred_ans,
        "answer": gt_ans,
    }
    return {"physics_rw_accuracy": result, "physics_rw_macro_f1": result}


def physics_rw_aggregate_results(results):
    domain_stats = {}
    for domain in DOMAINS:
        domain_stats[domain] = {"correct": 0, "total": 0}

    for result in results:
        domain = result["domain"]
        if domain not in domain_stats:
            domain_stats[domain] = {"correct": 0, "total": 0}
        domain_stats[domain]["total"] += 1
        if result["pred_answer"] == result["answer"]:
            domain_stats[domain]["correct"] += 1

    for domain in DOMAINS:
        stats = domain_stats.get(domain, {"correct": 0, "total": 0})
        if stats["total"] > 0:
            acc = 100 * stats["correct"] / stats["total"]
            eval_logger.info("Physics-RW [{}]: {:.1f}% ({}/{})", domain, acc, stats["correct"], stats["total"])

    total_correct = sum(s["correct"] for s in domain_stats.values())
    total = sum(s["total"] for s in domain_stats.values())

    if total == 0:
        return 0.0

    overall = 100 * total_correct / total
    eval_logger.info("Physics-RW overall: {:.1f}% ({}/{})", overall, total_correct, total)
    return overall


def _f1_for_label(results, label):
    true_positive = sum(result["pred_answer"] == label and result["answer"] == label for result in results)
    false_positive = sum(result["pred_answer"] == label and result["answer"] != label for result in results)
    false_negative = sum(result["pred_answer"] != label and result["answer"] == label for result in results)
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def physics_rw_aggregate_macro_f1(results):
    """Compute the paper's binary macro-F1 metric on yes/no predictions."""
    if not results:
        return 0.0
    macro_f1 = 100 * sum(_f1_for_label(results, label) for label in ("yes", "no")) / 2
    eval_logger.info("Physics-RW macro F1: {:.1f}%", macro_f1)
    return macro_f1
