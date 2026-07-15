"""MMOU: Massive Multi-Task Omni Understanding benchmark.

Repackaged self-contained dataset: ``kcz358/MMOU``
  data/{train,test}-*.parquet                  annotations for available videos
  data_missing/{train,test}_missing-*.parquet  annotations whose upstream video is missing
  videos_chunked_*.zip                          videos/<video_id>.mp4 for main split and
                                                videos/test/<video_id>.mp4 for Test Mini

``dataset_kwargs.video=True`` in the yaml lets lmms-eval auto-extract every
zip into ``$HF_HOME/mmou/`` on first run, so runtime just does a path lookup.
"""

import json
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml
from loguru import logger as eval_logger

DOMAINS = [
    "Sports",
    "Travel",
    "Video Games",
    "Daily Life",
    "Academic Lectures",
    "Film",
    "Pranks",
    "Music",
    "Animation",
    "News",
]

DURATION_BUCKETS = ["< 5", "5-10", "10-20", "20-30", "> 30"]

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = [line for line in raw_data if "!function" not in line]
_config = yaml.safe_load("".join(safe_data))
cache_name = _config["dataset_kwargs"]["cache_dir"]
video_cache_dir = os.path.join(base_cache_dir, cache_name, "videos")


def extract_youtube_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats (fallback for pre-packed docs)."""
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [""])[0]
        if parsed.path.startswith("/embed/") or parsed.path.startswith("/v/"):
            return parsed.path.split("/")[2]
    match = re.search(r"(?:v=|/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    return url


def duration_bucket(seconds: float) -> str:
    minutes = seconds / 60.0
    if minutes < 5:
        return "< 5"
    if minutes < 10:
        return "5-10"
    if minutes < 20:
        return "10-20"
    if minutes < 30:
        return "20-30"
    return "> 30"


def _video_id(doc) -> str:
    vid = doc.get("video_id")
    if vid:
        return str(vid)
    return extract_youtube_id(doc["video_url"])


def _resolve_video_path(doc) -> str:
    vid = _video_id(doc)
    is_test = bool(doc.get("is_test"))
    # Test Mini videos land under videos/test/, main split at videos root.
    for prefix in (("test",) if is_test else ("", "test")):
        candidate = os.path.join(video_cache_dir, prefix, f"{vid}.mp4") if prefix else os.path.join(video_cache_dir, f"{vid}.mp4")
        if os.path.exists(candidate):
            return candidate
    return ""


def mmou_doc_to_visual(doc):
    path = _resolve_video_path(doc)
    if path:
        return [path]
    eval_logger.warning(f"[mmou] Video not found for {_video_id(doc)} ({doc.get('video_url')}).")
    return []


def _format_options(options) -> str:
    if isinstance(options, str):
        options = json.loads(options)
    lines = [f"{k}. {options[k]}" for k in sorted(options.keys())]
    return "\n".join(lines)


def mmou_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get(
        "post_prompt",
        "\nAnswer with the option's letter from the given choices directly.",
    )
    question = doc["question"]
    option_text = _format_options(doc.get("options", {}))
    return f"{pre_prompt}{question}\n{option_text}{post_prompt}"


def mmou_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    prompt = mmou_doc_to_text(doc, lmms_eval_specific_kwargs)
    content = []
    for video_path in mmou_doc_to_visual(doc):
        content.append({"type": "video", "url": video_path})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


_VALID_LETTERS = "ABCDEFGHIJ"


def _extract_answer(response: str) -> str:
    """Extract answer letter (A-J) from model response."""
    if not response:
        return ""
    text = response.strip()
    for pattern in ("the answer is", "answer:", "the option is", "final answer:"):
        idx = text.lower().rfind(pattern)
        if idx >= 0:
            text = text[idx + len(pattern):].strip()
            break
    m = re.search(r"\(([A-J])\)", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-J])\b", text)
    if m:
        return m.group(1)
    for ch in text:
        if ch.upper() in _VALID_LETTERS:
            return ch.upper()
    return ""


def _build_result_dict(doc, pred: str, pred_ans: str) -> dict:
    video_id = _video_id(doc)
    dur = doc.get("video_duration", 0) or 0
    qtypes = doc.get("question_type", [])
    if isinstance(qtypes, str):
        try:
            qtypes = json.loads(qtypes)
        except Exception:
            qtypes = [qtypes]
    return {
        "question_id": doc["question_id"],
        "question": doc["question"],
        "answer": pred_ans,
        "raw_answer": pred,
        "domain": doc.get("domain", "Unknown"),
        "subdomain": doc.get("subdomain", "Unknown"),
        "question_type": qtypes,
        "video_url": doc.get("video_url", ""),
        "video_id": video_id,
        "video_duration": dur,
        "duration_bucket": duration_bucket(dur) if dur else "Unknown",
        "start_time": doc.get("start_time", ""),
        "end_time": doc.get("end_time", ""),
    }


def mmou_process_results_submission(doc, results):
    """Process results for the main (unlabeled) MMOU split.

    Ground truth is not available; results are aggregated into a submission
    file for the MMOU-Eval HuggingFace Space.
    """
    pred = results[0]
    pred_ans = _extract_answer(pred)
    return {"mmou_submission": _build_result_dict(doc, pred, pred_ans)}


def mmou_process_results_scored(doc, results):
    """Process results for the labeled MMOU Test Mini split."""
    pred = results[0]
    pred_ans = _extract_answer(pred)
    gt = str(doc.get("correct_option_letter", "")).strip().upper()
    data = _build_result_dict(doc, pred, pred_ans)
    data["gt"] = gt
    data["score"] = int(pred_ans == gt and gt != "")
    return {"mmou_accuracy": data}


def _write_submission(results, filename: str) -> str:
    output_dir = os.environ.get("MMOU_OUTPUT_DIR", "mmou_output")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    submission = [{"question_id": r["question_id"], "answer": r["answer"]} for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)
    return path


def _log_breakdown(results):
    domain_counts, duration_counts, skill_counts = {}, {}, {}
    for r in results:
        domain_counts[r.get("domain", "Unknown")] = domain_counts.get(r.get("domain", "Unknown"), 0) + 1
        bucket = r.get("duration_bucket", "Unknown")
        duration_counts[bucket] = duration_counts.get(bucket, 0) + 1
        for skill in r.get("question_type", []) or []:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
    eval_logger.info("=== MMOU Domain Breakdown ===")
    for k, v in sorted(domain_counts.items()):
        eval_logger.info(f"  {k}: {v}")
    eval_logger.info("=== MMOU Duration Breakdown ===")
    for k in DURATION_BUCKETS:
        eval_logger.info(f"  {k} min: {duration_counts.get(k, 0)}")
    eval_logger.info("=== MMOU Skill Breakdown ===")
    for k, v in sorted(skill_counts.items()):
        eval_logger.info(f"  {k}: {v}")


def mmou_aggregate_submission(results):
    path = _write_submission(results, "mmou_submission.json")
    eval_logger.info(
        f"MMOU submission saved to {path} ({len(results)} predictions). "
        "Upload to https://huggingface.co/spaces/nvidia/MMOU-Eval for scoring."
    )
    _log_breakdown(results)
    return len(results)


def mmou_aggregate_accuracy(results):
    path = _write_submission(results, "mmou_test_mini_submission.json")
    total = len(results)
    if total == 0:
        return 0.0
    correct = sum(r["score"] for r in results)
    acc = correct / total * 100
    eval_logger.info(f"MMOU Test Mini Accuracy: {acc:.2f}% [{correct}/{total}]")
    eval_logger.info(f"MMOU predictions saved to {path}")

    per_skill = {}
    for r in results:
        for skill in r.get("question_type", []) or []:
            per_skill.setdefault(skill, []).append(r["score"])
    if per_skill:
        eval_logger.info("=== MMOU Per-Skill Accuracy ===")
        for skill, scores in sorted(per_skill.items()):
            skill_acc = sum(scores) / len(scores) * 100
            eval_logger.info(f"  {skill}: {skill_acc:.2f}% [{len(scores)} samples]")

    per_bucket = {}
    for r in results:
        per_bucket.setdefault(r.get("duration_bucket", "Unknown"), []).append(r["score"])
    if per_bucket:
        eval_logger.info("=== MMOU Per-Duration Accuracy ===")
        for k in DURATION_BUCKETS + ["Unknown"]:
            scores = per_bucket.get(k, [])
            if scores:
                eval_logger.info(f"  {k} min: {sum(scores)/len(scores)*100:.2f}% [{len(scores)}]")

    return acc
