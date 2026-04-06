import json
import os
import re
from collections import defaultdict
from functools import lru_cache
from shutil import copy2
from typing import Any, Dict, List, Optional, Tuple

import datasets
from loguru import logger as eval_logger

QUESTION_TYPES = [
    "Action Order",
    "Motion-related Objects",
    "Motion Recognition",
    "Repetition Count",
    "Location-related Motion",
    "Camera Motion",
]

_OPTION_BLOCK_PATTERN = re.compile(r"(^|\n)\s*([A-F])[.):]\s*(.*?)\s*(?=(\n\s*[A-F][.):]\s)|$)", re.DOTALL)
_MISSING_VIDEO_WARNING_EMITTED = False


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _split_question_and_options(question_with_options: str) -> Tuple[str, List[Dict[str, str]]]:
    raw_text = (question_with_options or "").strip()
    if not raw_text:
        return "", []

    matches = list(_OPTION_BLOCK_PATTERN.finditer(raw_text))
    if not matches:
        return _normalize_whitespace(raw_text), []

    first_match = matches[0]
    question_stem = _normalize_whitespace(raw_text[: first_match.start()])
    options: List[Dict[str, str]] = []

    for match in matches:
        label = match.group(2).upper()
        option_text = _normalize_whitespace(match.group(3))
        if option_text:
            options.append({"label": label, "text": option_text})

    if not question_stem:
        question_stem = _normalize_whitespace(raw_text)

    return question_stem, options


def _coerce_options(options_raw: Any) -> List[Dict[str, str]]:
    if not isinstance(options_raw, list):
        return []

    options: List[Dict[str, str]] = []
    for option in options_raw:
        if isinstance(option, dict):
            label = str(option.get("label", "")).strip().upper()
            text = _normalize_whitespace(str(option.get("text", "")))
            if label and text:
                options.append({"label": label, "text": text})

    return options


def _to_metadata_row(raw_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "text" in raw_row and isinstance(raw_row["text"], str):
        line = raw_row["text"].strip()
        if not line:
            return None
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None
    return raw_row


def _append_flat_doc(
    flattened_docs: List[Dict[str, Any]],
    *,
    uid: str,
    question_type: str,
    video_type: str,
    video_path: str,
    question: str,
    options: List[Dict[str, str]],
    answer: str,
    key: str,
) -> None:
    option_labels = {opt["label"] for opt in options}
    if not options or answer not in option_labels:
        return

    flattened_docs.append(
        {
            "id": uid,
            "key": key,
            "video_path": video_path,
            "video_type": video_type,
            "question_type": question_type,
            "question": question,
            "options": options,
            "answer": answer,
        }
    )


def _flatten_motionbench_dataset(dataset: datasets.Dataset) -> datasets.Dataset:
    flattened_docs: List[Dict[str, Any]] = []

    for raw_row in dataset:
        row = _to_metadata_row(raw_row)
        if not row:
            continue

        question_type = str(row.get("question_type", "Unknown"))
        video_type = str(row.get("video_type", "Unknown"))
        video_path = str(row.get("video_path", "")).strip()
        video_key = str(row.get("key", "")).strip()

        if isinstance(row.get("question"), str) and row.get("answer"):
            answer = str(row.get("answer", "")).strip().upper()
            if answer and answer != "NA":
                question = _normalize_whitespace(str(row.get("question", "")).strip())
                options = _coerce_options(row.get("options"))
                if not options:
                    question, options = _split_question_and_options(question)

                _append_flat_doc(
                    flattened_docs,
                    uid=str(row.get("id") or f"{video_key}_{len(flattened_docs)}"),
                    question_type=question_type,
                    video_type=video_type,
                    video_path=video_path,
                    question=question,
                    options=options,
                    answer=answer,
                    key=video_key,
                )
            continue

        for qa in row.get("qa", []) or []:
            answer = str(qa.get("answer", "")).strip().upper()
            if not answer or answer == "NA":
                continue

            raw_question = str(qa.get("question", "")).strip()
            if not raw_question:
                continue

            question, options = _split_question_and_options(raw_question)
            if not options:
                continue

            _append_flat_doc(
                flattened_docs,
                uid=str(qa.get("uid") or f"{video_key}_{len(flattened_docs)}"),
                question_type=question_type,
                video_type=video_type,
                video_path=video_path,
                question=question,
                options=options,
                answer=answer,
                key=video_key,
            )

    if not flattened_docs:
        eval_logger.warning("[motionbench] No scored QA samples found after flattening dataset.")
        return dataset.select(range(0))

    return datasets.Dataset.from_list(flattened_docs)


def motionbench_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    processed = _flatten_motionbench_dataset(dataset)
    eval_logger.info("[motionbench] Loaded {} scored QA samples across all categories", len(processed))
    return processed


def _candidate_video_roots() -> List[str]:
    roots: List[str] = []

    configured = os.getenv("MOTIONBENCH_VIDEO_DIR")
    if configured:
        roots.append(os.path.expanduser(configured))

    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))
    roots.extend(
        [
            os.path.expanduser(os.getenv("MOTIONBENCH_CACHE_DIR", os.path.join(hf_home, "motionbench", "videos"))),
            os.path.join(hf_home, "motionbench", "videos"),
            os.path.join(hf_home, "motionbench"),
            os.path.join(hf_home, "datasets", "motionbench", "videos"),
            os.path.join(hf_home, "datasets", "THUDM___MotionBench"),
        ]
    )
    return roots


def _motionbench_repo_id() -> str:
    return os.getenv("MOTIONBENCH_HF_REPO", "zai-org/MotionBench")


def _is_auto_download_enabled() -> bool:
    return os.getenv("MOTIONBENCH_AUTO_DOWNLOAD", "1").strip().lower() not in {"0", "false", "no"}


@lru_cache(maxsize=1)
def _video_repo_index() -> Dict[str, str]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        eval_logger.warning("[motionbench] huggingface_hub unavailable for auto-download: {}", exc)
        return {}

    api = HfApi()
    repo_id = _motionbench_repo_id()
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as exc:
        eval_logger.warning("[motionbench] failed listing files for {}: {}", repo_id, exc)
        return {}

    index: Dict[str, str] = {}
    for path in files:
        if path.lower().endswith(".mp4"):
            basename = os.path.basename(path)
            if basename not in index:
                index[basename] = path
    return index


def _download_video(video_name: str) -> Optional[str]:
    if not _is_auto_download_enabled():
        return None

    index = _video_repo_index()
    remote_path = index.get(video_name)
    if not remote_path:
        return None

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        eval_logger.warning("[motionbench] huggingface_hub unavailable for download: {}", exc)
        return None

    try:
        downloaded = hf_hub_download(repo_id=_motionbench_repo_id(), repo_type="dataset", filename=remote_path)
    except Exception as exc:
        eval_logger.warning("[motionbench] failed downloading {}: {}", video_name, exc)
        return None

    cache_dir = os.path.expanduser(os.getenv("MOTIONBENCH_CACHE_DIR", os.path.join(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")), "motionbench", "videos")))
    os.makedirs(cache_dir, exist_ok=True)
    flat_path = os.path.join(cache_dir, video_name)
    if not os.path.exists(flat_path):
        try:
            copy2(downloaded, flat_path)
        except Exception:
            return downloaded
    return flat_path


def _resolve_video_path(doc: Dict[str, Any]) -> Optional[str]:
    video_name = str(doc.get("video_path", "")).strip()
    if not video_name:
        return None

    for root in _candidate_video_roots():
        candidate = os.path.join(root, video_name)
        if os.path.exists(candidate):
            return candidate

        if video_name.lower().endswith(".mp4"):
            alt_candidate = os.path.join(root, video_name[:-4] + ".MP4")
            if os.path.exists(alt_candidate):
                return alt_candidate

    downloaded = _download_video(video_name)
    if downloaded and os.path.exists(downloaded):
        return downloaded

    return None


def motionbench_doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict] = None):
    global _MISSING_VIDEO_WARNING_EMITTED

    video_path = _resolve_video_path(doc)
    if video_path:
        return [video_path]

    if not _MISSING_VIDEO_WARNING_EMITTED:
        eval_logger.warning("[motionbench] Video not found for sample. Set MOTIONBENCH_VIDEO_DIR to local dataset or keep MOTIONBENCH_AUTO_DOWNLOAD=1 for best-effort fetch. Continuing with text-only fallback.")
        _MISSING_VIDEO_WARNING_EMITTED = True
    return []


def _resolve_prompt_kwargs(lmms_eval_specific_kwargs: Optional[dict]) -> Dict[str, str]:
    kwargs = lmms_eval_specific_kwargs or {}
    if isinstance(kwargs.get("default"), dict):
        merged = dict(kwargs["default"])
        for key, value in kwargs.items():
            if key != "default":
                merged[key] = value
        return merged
    return kwargs


def motionbench_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict] = None) -> str:
    prompt_kwargs = _resolve_prompt_kwargs(lmms_eval_specific_kwargs)
    pre_prompt = prompt_kwargs.get("pre_prompt", "")
    post_prompt = prompt_kwargs.get("post_prompt", "Answer with only the option letter (A, B, C, D, E, or F).")

    option_lines = []
    for option in doc.get("options", []) or []:
        label = option.get("label", "")
        text = option.get("text", "")
        if label and text:
            option_lines.append(f"{label}. {text}")

    option_block = "\n".join(option_lines)
    question = str(doc.get("question", "")).strip()
    if option_block:
        body = f"{question}\n{option_block}"
    else:
        body = question

    return f"{pre_prompt}{body}\n{post_prompt}".strip()


def motionbench_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict] = None):
    prompt = motionbench_doc_to_text(doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)
    content: List[Dict[str, Any]] = []

    for video_path in motionbench_doc_to_visual(doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs):
        content.append({"type": "video", "url": video_path})

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _extract_choice_letter(text: str) -> str:
    if not isinstance(text, str):
        return ""

    candidate = text
    if "</think>" in candidate:
        candidate = candidate.split("</think>")[-1]

    candidate = candidate.strip().upper()
    if not candidate:
        return ""

    answer_prefixes = [
        "THE BEST ANSWER IS",
        "THE CORRECT ANSWER IS",
        "THE ANSWER IS",
        "BEST ANSWER:",
        "ANSWER:",
    ]
    for prefix in answer_prefixes:
        candidate = candidate.replace(prefix, "")

    match = re.search(r"\b([A-F])\b", candidate)
    if match:
        return match.group(1)

    match = re.search(r"([A-F])[.):]", candidate)
    if match:
        return match.group(1)

    match = re.search(r"[A-F]", candidate)
    if match:
        return match.group(0)

    return ""


def motionbench_process_results(doc: Dict[str, Any], results):
    response = results[0] if results else ""
    pred = _extract_choice_letter(response)
    target = str(doc.get("answer", "")).strip().upper()

    correct = 1.0 if pred and pred == target else 0.0
    answered = 1.0 if pred else 0.0

    return {
        "motionbench_acc": {"score": correct, "total": 1.0},
        "motionbench_answered_rate": {"score": answered, "total": 1.0},
        "motionbench_tracking": {
            "question_type": str(doc.get("question_type", "Unknown")),
            "score": correct,
            "answered": answered,
            "total": 1.0,
        },
    }


def _aggregate_score_total(results, score_key: str = "score") -> float:
    total_score = 0.0
    total_count = 0.0

    for result in results:
        if isinstance(result, dict):
            total_score += float(result.get(score_key, 0.0))
            total_count += float(result.get("total", 1.0))
        else:
            total_score += float(result)
            total_count += 1.0

    if total_count == 0.0:
        return 0.0
    return total_score / total_count


def motionbench_aggregate_acc(results):
    return _aggregate_score_total(results, score_key="score")


def motionbench_aggregate_answered_rate(results):
    return _aggregate_score_total(results, score_key="score")


def motionbench_aggregate_tracking(results):
    bucket = defaultdict(lambda: {"correct": 0.0, "answered": 0.0, "total": 0.0})
    total_correct = 0.0
    total_count = 0.0

    for result in results:
        if not isinstance(result, dict):
            continue

        question_type = str(result.get("question_type", "Unknown"))
        score = float(result.get("score", 0.0))
        answered = float(result.get("answered", 0.0))
        total = float(result.get("total", 1.0))

        bucket[question_type]["correct"] += score
        bucket[question_type]["answered"] += answered
        bucket[question_type]["total"] += total

        total_correct += score
        total_count += total

    for question_type in QUESTION_TYPES:
        stats = bucket.get(question_type)
        if not stats or stats["total"] == 0:
            continue
        acc = stats["correct"] / stats["total"]
        answered_rate = stats["answered"] / stats["total"]
        eval_logger.info(
            "[motionbench] {} - acc: {:.3f}, answered_rate: {:.3f}, n: {}",
            question_type,
            acc,
            answered_rate,
            int(stats["total"]),
        )

    overall_acc = 0.0 if total_count == 0.0 else total_correct / total_count
    eval_logger.info("[motionbench] overall_acc: {:.3f}, samples: {}", overall_acc, int(total_count))
    return overall_acc
