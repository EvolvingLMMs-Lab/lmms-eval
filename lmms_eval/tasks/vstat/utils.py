"""VSTAT benchmark task helpers.

VSTAT evaluates visual state tracking in long-form videos. The official QA
annotations are stored as a nested JSON file on Hugging Face, while only the
synthetic and self-recorded videos are redistributed. YouTube clips must be
downloaded and redacted with the scripts bundled in the dataset repository.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger

from lmms_eval import utils as lmms_utils
from lmms_eval.api.task import ConfigurableTask
from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

_DEFAULT_REPO_ID = "nyu-visionx/vstat"
_DEFAULT_QA_FILENAME = "vstat_qa_clean.json"
_DEFAULT_CACHE_DIR = "vstat"
_INTEGER_PATTERN = re.compile(r"-?\d+")
_CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_SNAPSHOT_DOWNLOAD_KWARGS = {
    "allow_patterns",
    "etag_timeout",
    "force_download",
    "ignore_patterns",
    "local_files_only",
    "max_workers",
    "revision",
    "token",
}

_downloaded_data_root: Path | None = None

_NUMERICAL_INSTRUCTIONS = {
    "book": "Return only a single integer (positive, negative, or 0).\nAnswer:",
    "packing_order_green": "Return only the final count as a single integer.\nAnswer:",
    "packing_order_blue": "Return only the final count as a single integer.\nAnswer:",
    "packing_order_yellow": "Return only the final count as a single integer.\nAnswer:",
    "packing_order_chopsticks": "Return only the final count as a single integer.\nAnswer:",
    "showing_card_count_diamond": "Return only the final count as a single integer.\nAnswer:",
    "showing_card_count_heart": "Return only the final count as a single integer.\nAnswer:",
    "showing_card_count_club": "Return only the final count as a single integer.\nAnswer:",
    "showing_card_count_spade": "Return only the final count as a single integer.\nAnswer:",
}

_SETUP_HINT = (
    "VSTAT downloads the Hugging Face dataset into $HF_HOME/vstat by default. "
    "If this missing file is a YouTube clip, install yt-dlp and ffmpeg so the "
    "task can download and redact YouTube clips during setup. "
    "Set VSTAT_VIDEO_ROOT=/path/to/prepared/vstat to use a different prepared dataset root. "
    "If the QA JSON is elsewhere, set VSTAT_QA_PATH=/path/to/vstat_qa_clean.json."
)


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _hf_home() -> Path:
    return Path(os.path.expanduser(os.path.expandvars(os.getenv("HF_HOME", "~/.cache/huggingface/"))))


def _cache_root_from_config(dataset_kwargs: dict[str, Any] | None) -> Path:
    kwargs = dataset_kwargs or {}
    cache_dir = str(kwargs.get("cache_dir") or _DEFAULT_CACHE_DIR)
    return Path(lmms_utils.resolve_cache_dir(cache_dir, base_dir=str(_hf_home())))


def _snapshot_kwargs(dataset_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: dataset_kwargs[key] for key in _SNAPSHOT_DOWNLOAD_KWARGS if key in dataset_kwargs}


def _load_vstat_payload(qa_path: Path) -> Any:
    with qa_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_vstat_data(payload: Any) -> dict[str, Any]:
    data = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    if not isinstance(data, dict):
        raise ValueError("VSTAT QA JSON should contain a `data` object mapping categories to examples.")
    return data


def _missing_youtube_paths(cache_root: Path, qa_path: Path) -> list[Path]:
    data = _extract_vstat_data(_load_vstat_payload(qa_path))
    missing: list[Path] = []
    seen: set[str] = set()
    for docs in data.values():
        for doc in docs:
            video_path = str(doc.get("video_path", ""))
            if not video_path.startswith("videos/youtube/"):
                continue
            candidate = cache_root / video_path
            key = str(candidate)
            if key not in seen and not candidate.exists():
                missing.append(candidate)
                seen.add(key)
    return missing


def _run_vstat_command(command: list[str], cwd: Path, step_name: str) -> None:
    eval_logger.info(f"VSTAT: running {step_name}: {' '.join(command)}")
    try:
        subprocess.run(command, cwd=str(cwd), check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"VSTAT {step_name} failed with exit code {exc.returncode}.") from exc


def _ensure_youtube_clips(cache_root: Path, qa_path: Path, accelerator: Accelerator) -> None:
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return

    missing = _missing_youtube_paths(cache_root, qa_path)
    if not missing:
        eval_logger.info("VSTAT YouTube clips are already present.")
        accelerator.wait_for_everyone()
        return

    if shutil.which("yt-dlp") is None:
        raise RuntimeError("VSTAT needs yt-dlp to prepare YouTube clips. Run `pip install -U yt-dlp`.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("VSTAT needs ffmpeg to prepare YouTube clips.")

    downloader = cache_root / "scripts" / "download_youtube.py"
    redactor = cache_root / "scripts" / "redact.sh"
    resolution_map = cache_root / "youtube_resolutions.json"
    for required_path in (downloader, redactor, resolution_map, cache_root / "youtube_metadata.json"):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing VSTAT YouTube helper file: {required_path}")

    eval_logger.info(f"VSTAT: preparing {len(missing)} missing YouTube clips in {cache_root}.")
    _run_vstat_command(
        [
            sys.executable,
            str(downloader.relative_to(cache_root)),
            "--resolution-map",
            str(resolution_map.relative_to(cache_root)),
        ],
        cache_root,
        "YouTube clip download",
    )
    _run_vstat_command(["bash", str(redactor.relative_to(cache_root))], cache_root, "YouTube redaction")
    accelerator.wait_for_everyone()


def _download_dataset_to_cache(dataset_path: str | None, dataset_kwargs: dict[str, Any]) -> Path:
    repo_id = dataset_kwargs.get("repo_id") or dataset_path or _DEFAULT_REPO_ID
    qa_filename = dataset_kwargs.get("qa_filename", _DEFAULT_QA_FILENAME)
    cache_root = _cache_root_from_config(dataset_kwargs)
    qa_path = cache_root / qa_filename
    force_download = bool(dataset_kwargs.get("force_download", False))

    accelerator = Accelerator()
    if accelerator.is_main_process:
        if force_download or not qa_path.exists() or not (cache_root / "videos").exists():
            cache_root.mkdir(parents=True, exist_ok=True)
            eval_logger.info(f"Downloading VSTAT dataset from {repo_id} to {cache_root}.")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(cache_root),
                **_snapshot_kwargs(dataset_kwargs),
            )
    accelerator.wait_for_everyone()

    if not qa_path.exists():
        raise FileNotFoundError(f"Missing VSTAT QA file after dataset download: {qa_path}")
    _ensure_youtube_clips(cache_root, qa_path, accelerator)
    return cache_root


def _qa_path_from_config(dataset_path: str | None, dataset_kwargs: dict[str, Any] | None) -> Path:
    override = os.environ.get("VSTAT_QA_PATH")
    if override:
        return _resolve_path(override)

    kwargs = dataset_kwargs or {}
    data_files = kwargs.get("data_files")
    if data_files:
        if isinstance(data_files, dict):
            data_file = data_files.get("test") or next(iter(data_files.values()))
        else:
            data_file = data_files
        return _resolve_path(data_file)

    filename = kwargs.get("qa_filename", _DEFAULT_QA_FILENAME)
    return _download_dataset_to_cache(dataset_path, kwargs) / filename


def _candidate_video_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("VSTAT_VIDEO_ROOT", "VSTAT_DATA_ROOT"):
        value = os.environ.get(env_name)
        if value:
            roots.append(_resolve_path(value))

    if _downloaded_data_root is not None:
        roots.append(_downloaded_data_root)

    for root in (_cache_root_from_config(None), Path.cwd() / "data" / "vstat", Path(__file__).resolve().parents[3] / "data" / "vstat"):
        roots.append(root)

    deduped: list[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def _resolve_video_path(video_path: str) -> Path:
    raw_path = Path(video_path).expanduser()
    if raw_path.is_absolute():
        return raw_path

    for root in _candidate_video_roots():
        candidate = root / raw_path
        if candidate.exists():
            return candidate

    default_root = _candidate_video_roots()[0] if _candidate_video_roots() else Path.cwd()
    return default_root / raw_path


def _normalize_doc_for_arrow(doc: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(doc)
    normalized["answer"] = str(normalized.get("answer", ""))
    normalized["choices"] = normalized.get("choices") or []
    normalized["perceptual_complexity"] = normalized.get("perceptual_complexity") or []
    normalized["answer_index"] = normalized.get("answer_index")
    return normalized


class VSTATTask(ConfigurableTask):
    """ConfigurableTask that expands the official nested VSTAT QA JSON."""

    def __init__(self, *args, config: dict[str, Any] | None = None, **kwargs) -> None:
        if config is not None:
            config = dict(config)
            config.pop("class", None)
        super().__init__(*args, config=config, **kwargs)

    def download(self, dataset_kwargs: dict[str, Any] | None = None) -> None:
        global _downloaded_data_root

        qa_path = _qa_path_from_config(self.config.dataset_path, dict(dataset_kwargs or {}))
        _downloaded_data_root = qa_path.parent

        data = _extract_vstat_data(_load_vstat_payload(qa_path))

        rows = [_normalize_doc_for_arrow(doc) for docs in data.values() for doc in docs]
        split = self.config.test_split
        self.dataset = DatasetDict({split: Dataset.from_list(rows)})
        if self.config.process_docs is not None:
            self.dataset[split] = self.config.process_docs(self.dataset[split])
        self.dataset_no_image = self.dataset.copy()
        eval_logger.info(f"Loaded VSTAT annotations from {qa_path} ({len(rows)} examples).")


def vstat_process_docs(dataset: Dataset) -> Dataset:
    flat_docs: list[dict[str, Any]] = []
    for doc in dataset:
        question = str(doc["question"]).strip()
        answer = str(doc["answer"]).strip()
        choices = doc.get("choices") or []
        answer_type = str(doc.get("answer_type") or ("mcq" if choices else "numeric")).lower()
        is_mcq = answer_type == "mcq" or bool(choices)

        flat_doc: dict[str, Any] = {
            "video_id": str(doc["video_id"]),
            "video_path": str(doc["video_path"]),
            "video_source": str(doc.get("video_source", "")),
            "source_task": str(doc.get("source_task", "")),
            "question": question,
            "answer_type": answer_type,
            "is_mcq": is_mcq,
            "choices": list(choices),
            "answer_index": doc.get("answer_index"),
            "perceptual_complexity": list(doc.get("perceptual_complexity") or []),
            "state_element_type": str(doc.get("state_element_type", "")),
            "state_structure": str(doc.get("state_structure", "")),
        }

        if is_mcq:
            flat_doc["answer_text"] = _normalize_mcq_answer(answer, flat_doc["answer_index"])
            flat_doc["target_value"] = None
        else:
            flat_doc["answer_text"] = answer
            try:
                flat_doc["target_value"] = int(answer)
            except ValueError:
                flat_doc["target_value"] = None

        flat_docs.append(flat_doc)

    return Dataset.from_list(flat_docs)


def _normalize_mcq_answer(answer: str, answer_index: Any) -> str:
    answer = answer.strip().upper()
    if len(answer) == 1 and answer in _CHOICE_LETTERS:
        return answer
    try:
        index = int(answer_index)
    except (TypeError, ValueError):
        return answer
    return _CHOICE_LETTERS[index]


def vstat_doc_to_visual(doc: dict[str, Any]) -> list[str]:
    path = _resolve_video_path(doc["video_path"])
    if not path.exists():
        raise FileNotFoundError(f"Missing VSTAT video file: {path}\n{_SETUP_HINT}")
    return [str(path)]


def _mcq_instruction(num_choices: int) -> str:
    letters = list(_CHOICE_LETTERS[:num_choices])
    if len(letters) <= 1:
        return ""
    if len(letters) == 2:
        return f"Please answer with the letter ({letters[0]} or {letters[1]})."
    return f"Please answer with the letter ({', '.join(letters[:-1])}, or {letters[-1]})."


def vstat_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    body = f"Watch the full video carefully before answering.\n\nQuestion: {doc['question']}"

    if doc["is_mcq"]:
        instruction = _mcq_instruction(len(doc.get("choices") or []))
        post_prompt = kwargs.get("mcq_post_prompt", "")
        if instruction:
            body = f"{body}\n\n{instruction}"
    else:
        instruction = _NUMERICAL_INSTRUCTIONS.get(str(doc.get("source_task", "")))
        if instruction:
            body = f"{body}\n\n{instruction}"
        post_prompt = kwargs.get("numeric_post_prompt", "")

    return f"{pre_prompt}{body}\n\n{post_prompt}".strip()


def vstat_doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc["answer_text"])


def _extract_last_integer(text: str) -> int | None:
    matches = _INTEGER_PATTERN.findall(str(text))
    return int(matches[-1]) if matches else None


def _parse_prediction(doc: dict[str, Any], prediction: str) -> str | int | None:
    if doc["is_mcq"]:
        choices = list(_CHOICE_LETTERS[: max(2, len(doc.get("choices") or []))])
        return extract_mcq_answer(prediction, choices=choices) or None
    if doc.get("target_value") is not None:
        return _extract_last_integer(prediction)
    return str(prediction).strip().lower()


def _exact_match_score(doc: dict[str, Any], parsed_prediction: str | int | None) -> float:
    if parsed_prediction is None:
        return 0.0
    if doc["is_mcq"]:
        return float(str(parsed_prediction).upper() == str(doc["answer_text"]).upper())
    return float(str(parsed_prediction).strip().lower() == str(doc["answer_text"]).strip().lower())


def _numeric_mra(parsed_prediction: str | int | None, target_value: int) -> float:
    if parsed_prediction is None:
        return 0.0
    try:
        pred = int(parsed_prediction)
    except (TypeError, ValueError):
        return 0.0

    if target_value == 0:
        relative_error = 0.0 if pred == 0 else float("inf")
    else:
        relative_error = abs(pred - target_value) / abs(target_value)

    thresholds = [0.5 + 0.05 * i for i in range(10)]
    return sum(relative_error <= 1 - threshold for threshold in thresholds) / len(thresholds)


def vstat_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction = str(results[0]).strip() if results else ""
    parsed_prediction = _parse_prediction(doc, prediction)
    is_numeric = (not doc["is_mcq"]) and doc.get("target_value") is not None

    if is_numeric:
        score = _numeric_mra(parsed_prediction, int(doc["target_value"]))
        return {"Numeric_MRA": score, "ALL_Score_avg": score}

    score = _exact_match_score(doc, parsed_prediction)
    return {"MCQ_ACC": score, "ALL_Score_avg": score}


def vstat_aggregate_mean(results: list[float]) -> float:
    return sum(float(result) for result in results) / len(results) if results else 0.0
