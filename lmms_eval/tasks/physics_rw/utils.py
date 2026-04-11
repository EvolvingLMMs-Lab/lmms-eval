import os
import re

import datasets
from loguru import logger as eval_logger

# Maps video filename prefix to the HuggingFace subdirectory for that domain.
_PREFIX_TO_DOMAIN = {
    "elec": "Electromagnetism",
    "elasticity": "Mechanics",
    "gravity": "Mechanics",
    "dominoes": "Mechanics",
    "optics": "Optics",
    "thermology": "Thermodynamics",
    "buoyancy": "Mechanics",
    "pressure": "Mechanics",
}

_HF_BASE = "https://huggingface.co/datasets/zhaopengyu/Physics-RW/resolve/main/Physics-RW"

DOMAINS = ["Electromagnetism", "Mechanics", "Optics", "Thermodynamics"]


def _get_cache_dir():
    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
    return os.path.join(hf_home, "physics_rw")


def _domain_from_video_path(video_path):
    """Infer domain from video filename prefix."""
    basename = os.path.basename(video_path).replace(".mp4", "")
    for prefix, domain in _PREFIX_TO_DOMAIN.items():
        if basename.startswith(prefix):
            return domain
    # Fallback: try to guess from the full path
    for domain in DOMAINS:
        if domain.lower() in video_path.lower():
            return domain
    return "Unknown"


def _download_video(video_path, domain, cache_dir):
    """Download a video from HuggingFace if not already cached."""
    # video_path is like "video/elec_ori_video_26_0.mp4"
    filename = os.path.basename(video_path)
    local_path = os.path.join(cache_dir, domain, filename)

    if os.path.exists(local_path):
        return local_path

    url = f"{_HF_BASE}/{domain}/classification/{video_path}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        import urllib.request

        eval_logger.info("Physics-RW: downloading {} -> {}", url, local_path)
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except Exception as e:
        eval_logger.warning("Physics-RW: failed to download {}: {}", url, e)
        return None


def physics_rw_process_docs(dataset):
    """Add domain field to each doc based on the video filename prefix."""
    processed = []
    for doc in dataset:
        doc = dict(doc)
        doc["domain"] = _domain_from_video_path(doc.get("video_path", ""))
        processed.append(doc)
    return datasets.Dataset.from_list(processed)


def physics_rw_doc_to_visual(doc):
    cache_dir = _get_cache_dir()
    video_path = doc.get("video_path", "")
    domain = doc.get("domain", _domain_from_video_path(video_path))

    # Try local cache first
    filename = os.path.basename(video_path)
    local_path = os.path.join(cache_dir, domain, filename)

    if not os.path.exists(local_path):
        local_path = _download_video(video_path, domain, cache_dir)

    if local_path and os.path.exists(local_path):
        return [local_path]

    eval_logger.warning("Physics-RW: video not found for {}", video_path)
    return []


def physics_rw_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    instruction = doc.get("instruction", "")
    return f"{pre_prompt}{instruction}{post_prompt}"


def _extract_yes_no(text):
    """Extract yes/no answer from model response."""
    text = text.strip().lower()
    # Direct match at start
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    # Search for yes/no in the response
    yes_match = re.search(r"\byes\b", text)
    no_match = re.search(r"\bno\b", text)
    if yes_match and not no_match:
        return "yes"
    if no_match and not yes_match:
        return "no"
    # Both present: take whichever appears first
    if yes_match and no_match:
        return "yes" if yes_match.start() < no_match.start() else "no"
    return ""


def physics_rw_process_results(doc, results):
    pred = results[0]
    pred_ans = _extract_yes_no(pred)
    gt_ans = doc.get("label", "").strip().lower()
    domain = doc.get("domain", "Unknown")

    return {
        "physics_rw_accuracy": {
            "idx": doc.get("idx", -1),
            "domain": domain,
            "pred_answer": pred_ans,
            "answer": gt_ans,
        }
    }


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
