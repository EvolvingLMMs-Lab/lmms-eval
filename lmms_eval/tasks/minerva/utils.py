import os
import random
import re
import threading
from collections import defaultdict
from pathlib import Path

from lmms_eval.tasks._task_utils.lance_video_resolver import LanceVideoBlobResolver
from lmms_eval.tasks._task_utils.video_loader import get_video
from lmms_eval.utils import eval_logger

OPTIONS = ["A", "B", "C", "D", "E"]
_VIDEO_EXTENSIONS = ("mp4", "webm", "mkv", "mov")
_LANCE_RESOLVER = None
_LANCE_RESOLVER_LOCK = threading.Lock()


def _get_lance_resolver() -> LanceVideoBlobResolver | None:
    global _LANCE_RESOLVER

    dataset_uri = os.getenv("MINERVA_LANCE_VIDEO_URI", "").strip()
    if dataset_uri == "":
        return None

    if _LANCE_RESOLVER is None:
        with _LANCE_RESOLVER_LOCK:
            if _LANCE_RESOLVER is None:
                id_column = os.getenv("MINERVA_LANCE_VIDEO_ID_COLUMN", "video_id").strip()
                blob_column = os.getenv("MINERVA_LANCE_VIDEO_BLOB_COLUMN", "video_blob").strip()
                cache_dir = Path(os.path.expanduser(os.getenv("MINERVA_LANCE_CACHE_DIR", "~/.cache/lmms_eval/minerva_lance_videos")))
                _LANCE_RESOLVER = LanceVideoBlobResolver(
                    dataset_uri=dataset_uri,
                    id_column=id_column,
                    blob_column=blob_column,
                    cache_dir=cache_dir,
                    ext_column="video_ext",
                    source_name="MINERVA Lance",
                    video_extensions=_VIDEO_EXTENSIONS,
                )

    return _LANCE_RESOLVER


def _resolve_local_video(video_id: str) -> str | None:
    video_dir = os.getenv("MINERVA_VIDEO_DIR", "").strip()
    if video_dir == "":
        return None

    for ext in _VIDEO_EXTENSIONS:
        try:
            return get_video(video_dir, video_id, suffix=ext)
        except FileNotFoundError:
            continue
    return None


def _build_choices(doc):
    choices = []
    for i in range(5):
        choices.append(str(doc.get(f"answer_choice_{i}", "")).strip())
    return choices


def _gold_letter(doc):
    answer_id = doc.get("answer_id")
    if isinstance(answer_id, int) and 0 <= answer_id < len(OPTIONS):
        return OPTIONS[answer_id]

    if isinstance(answer_id, str) and answer_id.isdigit():
        idx = int(answer_id)
        if 0 <= idx < len(OPTIONS):
            return OPTIONS[idx]

    gold_answer = str(doc.get("answer", "")).strip()
    choices = _build_choices(doc)
    for idx, choice in enumerate(choices):
        if choice == gold_answer:
            return OPTIONS[idx]
    return ""


def _extract_choice_letter(text):
    cleaned = str(text).strip()

    explicit_pattern = re.compile(r"(?:answer|option)\s*(?:is|:)\s*([ABCDE])", re.IGNORECASE)
    explicit_matches = list(explicit_pattern.finditer(cleaned))
    if explicit_matches:
        return explicit_matches[-1].group(1).upper()

    prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "Answer:",
        "Option:",
        "Therefore, the final answer is:",
    ]
    for prefix in prefixes:
        cleaned = cleaned.replace(prefix, "")

    matches = list(re.finditer(r"[ABCDE]", cleaned))
    if not matches:
        return ""
    return matches[-1].group(0)


def minerva_doc_to_visual(doc):
    video_id = str(doc.get("video_id", "")).strip()
    if video_id == "":
        eval_logger.warning("minerva_doc_to_visual: document is missing valid 'video_id'; returning empty visual list")
        return []

    local_video = _resolve_local_video(video_id)
    if local_video is not None:
        return [local_video]

    resolver = _get_lance_resolver()
    if resolver is not None:
        return [resolver.resolve(video_id)]

    # Fallback for compatibility if users still rely on direct URL style.
    return [f"https://www.youtube.com/watch?v={video_id}"]


def minerva_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = str(doc.get("question", "")).strip()
    choices = _build_choices(doc)
    choice_text = "\n".join(f"{OPTIONS[idx]}. {choice}" for idx, choice in enumerate(choices))

    return f"{pre_prompt}{question}\n{choice_text}{post_prompt}"


def minerva_doc_to_target(doc):
    return _gold_letter(doc)


def minerva_process_results(doc, results):
    raw_pred = results[0] if isinstance(results, list) and len(results) > 0 else ""
    pred_letter = _extract_choice_letter(raw_pred)
    if pred_letter == "":
        pred_letter = random.choice(OPTIONS)

    gold_letter = _gold_letter(doc)
    score = 1.0 if pred_letter == gold_letter else 0.0

    question_type = doc.get("question_type")
    if question_type is None and "question type" in doc:
        eval_logger.warning("MINERVA doc uses deprecated key 'question type'; prefer 'question_type'")
        question_type = doc.get("question type")

    data_dict = {
        "id": doc.get("key", ""),
        "question_type": question_type if question_type is not None else "UNKNOWN",
        "split": doc.get("split", "UNKNOWN"),
        "category": doc.get("category", "UNKNOWN"),
        "pred_answer": pred_letter,
        "answer": gold_letter,
        "score": score,
    }
    return {"minerva_acc": data_dict}


def minerva_aggregate_results(results):
    by_question_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_split = defaultdict(lambda: {"correct": 0, "total": 0})
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})

    total_correct = 0
    total_count = 0

    for result in results:
        qtype = result.get("question_type", "UNKNOWN")
        split = result.get("split", "UNKNOWN")
        category = result.get("category", "UNKNOWN")
        correct = 1 if result.get("score", 0.0) == 1.0 else 0

        by_question_type[qtype]["correct"] += correct
        by_question_type[qtype]["total"] += 1

        by_split[split]["correct"] += correct
        by_split[split]["total"] += 1

        by_category[category]["correct"] += correct
        by_category[category]["total"] += 1

        total_correct += correct
        total_count += 1

    for qtype, stats in by_question_type.items():
        acc = 100.0 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        eval_logger.info(f"MINERVA - question type [{qtype}] accuracy: {acc:.2f}%")

    for split, stats in by_split.items():
        acc = 100.0 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        eval_logger.info(f"MINERVA - split [{split}] accuracy: {acc:.2f}%")

    for category, stats in by_category.items():
        acc = 100.0 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        eval_logger.info(f"MINERVA - category [{category}] accuracy: {acc:.2f}%")

    overall = 100.0 * total_correct / total_count if total_count > 0 else 0.0
    eval_logger.info(f"MINERVA - overall accuracy: {overall:.2f}%")
    return overall
