"""Custom Task class for Full-Duplex-Bench v1.5.

The official data release (`Ssshangfu/Full-Duplex-Bench-Data` on HuggingFace)
ships per-sample folders with conversation audio inside versioned zip archives
rather than a loadable parquet/jsonl table, so the stock
``ConfigurableTask.download`` cannot index it. This subclass resolves the data
root (environment variable, explicit kwarg, or a fresh snapshot + unzip of the
official repo) and builds the document list in memory.
"""

import json
import os
import zipfile
from typing import Optional

import datasets
from loguru import logger as eval_logger

from lmms_eval.api.task import ConfigurableTask
from lmms_eval.utils import resolve_cache_dir

FDB_REPO_ID = "Ssshangfu/Full-Duplex-Bench-Data"

# task -> expected judge label (v1.5 correctness mapping from the official
# evaluation repository docs)
CORRECT_LABEL = {
    "user_interruption": "C_RESPOND",
    "user_backchannel": "C_RESUME",
    "talking_to_other": "C_RESUME",
    "background_speech": "C_RESUME",
}


def _unzip_once(zip_path: str, cache_dir: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir():
                continue
            target_path = os.path.join(cache_dir, file_info.filename)
            if not os.path.exists(target_path):
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zip_ref.open(file_info) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())
    eval_logger.info(f"Extracted {zip_path} to {cache_dir}")


def _find_task_dir(root: str, fdb_task: str) -> Optional[str]:
    """Locate `<fdb_task>/<sample>/metadata.json` under a data root."""
    candidates = [
        os.path.join(root, fdb_task),
        os.path.join(root, "v1.5", fdb_task),
        os.path.join(root, "v1_5", fdb_task),
        os.path.join(root, "data-full-duplex-bench", "v1.5", fdb_task),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and any(f == "metadata.json" for _, _, files in os.walk(candidate) for f in files):
            return candidate
    # last resort: recursive search (snapshot layouts may nest)
    for dirpath, dirnames, _filenames in os.walk(root):
        if os.path.basename(dirpath) == fdb_task and any(os.path.exists(os.path.join(dirpath, d, "metadata.json")) for d in dirnames):
            return dirpath
    return None


def _resolve_fdb_root(dataset_kwargs: dict) -> str:
    explicit = dataset_kwargs.get("data_root") or os.environ.get("FDB_DATA_ROOT")
    if explicit:
        return explicit

    from huggingface_hub import snapshot_download

    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
    cache_dir = resolve_cache_dir("full_duplex_bench", base_dir=hf_home)
    snapshot_path = snapshot_download(repo_id=FDB_REPO_ID, repo_type="dataset", allow_patterns=["v1.5/*"])
    zip_files = [os.path.join(snapshot_path, name) for name in ("v1.5/user_interruption.zip", "v1.5/user_backchannel.zip", "v1.5/talking_to_other.zip", "v1.5/background_speech.zip") if os.path.exists(os.path.join(snapshot_path, name))]
    if not zip_files:
        raise FileNotFoundError(f"No v1.5 task zips found in the {FDB_REPO_ID} snapshot at {snapshot_path}. Pass dataset_kwargs.data_root or set FDB_DATA_ROOT to a directory containing the extracted v1.5 data.")
    for zip_file in zip_files:
        _unzip_once(zip_file, cache_dir)
    return cache_dir


def _read_words(json_path: str):
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    chunks = payload.get("chunks", []) if isinstance(payload, dict) else payload
    return chunks or None


def _sort_numeric(samples):
    def key(name: str):
        try:
            return (0, int(name), "")
        except ValueError:
            return (1, 0, name)

    return sorted(samples, key=key)


def _build_rows(root: str, fdb_task: str) -> list:
    task_dir = _find_task_dir(root, fdb_task)
    if task_dir is None:
        raise FileNotFoundError(f"Could not locate the '{fdb_task}' sample folders under {root!r}. Expected either `<root>/{fdb_task}/<sample>/metadata.json` or `<root>/v1.5/{fdb_task}/...`.")
    rows = []
    for sample in _sort_numeric(os.listdir(task_dir)):
        sample_dir = os.path.join(task_dir, sample)
        if not os.path.isdir(sample_dir) or not os.path.exists(os.path.join(sample_dir, "metadata.json")):
            continue
        with open(os.path.join(sample_dir, "metadata.json"), "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        # v1.5 metadata names the overlap speech differently per task
        # (current_turn_text / backchannel_text / background_text).
        overlap_text = next(
            (metadata[key] for key in ("current_turn_text", "backchannel_text", "background_text") if metadata.get(key)),
            "",
        )
        rows.append(
            {
                "sample_id": f"{fdb_task}/{sample}",
                "fdb_task": fdb_task,
                "correct_label": CORRECT_LABEL[fdb_task],
                "context_text": metadata.get("context_text", ""),
                "overlap_text": overlap_text,
                "timestamps": metadata.get("timestamps", []),
                "clean_input_wav": os.path.join(sample_dir, "clean_input.wav"),
                "noisy_input_wav": os.path.join(sample_dir, "input.wav"),
                "input_clean_words": _read_words(os.path.join(sample_dir, "clean_input.json")) or [],
                "input_noisy_words": _read_words(os.path.join(sample_dir, "input.json")) or [],
            }
        )
    if not rows:
        raise FileNotFoundError(f"No Full-Duplex-Bench samples found under {task_dir!r}.")
    eval_logger.info(f"Full-Duplex-Bench task '{fdb_task}': indexed {len(rows)} samples from {task_dir}")
    return rows


class FullDuplexBenchTask(ConfigurableTask):
    """Full-Duplex-Bench v1.5 task with zip-based data preparation."""

    def __init__(self, *args, config=None, **kwargs):
        config = dict(config or {})
        # The `class:` key selects this class; it must not reach TaskConfig.
        config.pop("class", None)
        super().__init__(*args, config=config, **kwargs)

    def download(self, dataset_kwargs=None) -> None:
        dataset_kwargs = dict(dataset_kwargs or {})
        fdb_task = dataset_kwargs.pop("fdb_task", None)
        if fdb_task is None:
            task_name = getattr(self.config, "task", "") or ""
            fdb_task = task_name[len("fdb_") :] if task_name.startswith("fdb_") else ""
        if fdb_task not in CORRECT_LABEL:
            raise ValueError(f"Unknown Full-Duplex-Bench task {fdb_task!r}; expected one of {sorted(CORRECT_LABEL)}")

        root = _resolve_fdb_root(dataset_kwargs)
        rows = _build_rows(root, fdb_task)
        self.dataset = datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})
        # Rows carry only paths and small transcripts; nothing to strip.
        self.dataset_no_image = self.dataset
