import io
import os
import os.path as osp
import re
import tarfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import datasets
import numpy as np
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger
from PIL import Image

REPO_ID = "TACPS-liv/Spatial-DISE"


def spatial_dise_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset_root = _dataset_root()
    tar_index = _tar_index(dataset_root)

    def _process_doc(doc, idx):
        clean_doc = {str(key).strip(): _strip(value) for key, value in doc.items()}
        image_path = _csv_path_to_tar_member(clean_doc["image"])
        shard = tar_index.get(image_path)
        if shard is None:
            raise FileNotFoundError(f"Spatial-DISE image {clean_doc['image']} not found in tar shards under {dataset_root}")

        return {
            "id": f"benchmark_{idx}",
            "question": clean_doc["question"],
            "answer": clean_doc["answer"].upper(),
            "image_path": image_path,
            "image_shard": shard,
            "category": clean_doc.get("category", ""),
            "difficulty": clean_doc.get("difficulty", ""),
            "source": clean_doc.get("source", ""),
            "dise_category": clean_doc.get("dise_category", ""),
        }

    return dataset.map(_process_doc, with_indices=True)


def spatial_dise_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    with tarfile.open(doc["image_shard"]) as tf:
        image_file = tf.extractfile(doc["image_path"])
        if image_file is None:
            raise FileNotFoundError(f"{doc['image_path']} not found in {doc['image_shard']}")
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    return [image]


def spatial_dise_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question'].strip()}{post_prompt}".strip()


def spatial_dise_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    response = results[0]
    target = doc["answer"].strip().upper()
    pred = _extract_answer(response)
    is_correct = pred == target

    return {
        "spatial_dise_acc": {
            "id": doc["id"],
            "gt": target,
            "pred": response,
            "pred_parsed": pred,
            "category": doc["category"],
            "difficulty": doc["difficulty"],
            "dise_category": doc["dise_category"],
            "is_correct": is_correct,
        }
    }


def spatial_dise_aggregate_results(results: List[Dict[str, Any]]) -> float:
    if len(results) == 0:
        return 0.0

    scores = [sample["is_correct"] for sample in results]
    _log_breakdown("category", results)
    _log_breakdown("difficulty", results)
    _log_breakdown("dise_category", results)
    return float(np.mean(scores))


def _extract_answer(response: str) -> str:
    response = str(response).strip()
    try:
        from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

        answer = extract_mcq_answer(response, choices=["A", "B", "C", "D"])
        if answer:
            return answer.strip().upper()
    except Exception:
        pass

    patterns = [
        r"(?:answer|final answer|correct answer)\s*[:：]?\s*\(?([A-D])\)?",
        r"^\s*\(?([A-D])\)?(?:[\.\):\s]|$)",
        r"\b([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def _log_breakdown(key: str, results: List[Dict[str, Any]]) -> None:
    grouped = defaultdict(list)
    for sample in results:
        grouped[sample[key]].append(sample["is_correct"])

    eval_logger.info(f"Spatial-DISE {key} breakdown:")
    for name in sorted(grouped):
        score = float(np.mean(grouped[name]))
        eval_logger.info(f"  {name}: {score:.4f} ({sum(grouped[name])}/{len(grouped[name])})")


def _dataset_root() -> str:
    local_root = os.environ.get("SPATIAL_DISE_ROOT")
    if local_root:
        local_root = osp.expanduser(osp.expandvars(local_root))
        if osp.isdir(local_root):
            return local_root

    return snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision="main",
        allow_patterns=["DISE-bench/DISE-benchmark.csv", "image/*.tar"],
    )


def _csv_path_to_tar_member(path: str) -> str:
    path = str(path).strip()
    if path.startswith("images/"):
        path = path[len("images/") :]
    return path.lstrip("/\\")


@lru_cache(maxsize=4)
def _tar_index(dataset_root: str) -> Dict[str, str]:
    image_dir = osp.join(dataset_root, "image")
    tar_paths = sorted(Path(image_dir).glob("*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"No Spatial-DISE tar shards found under {image_dir}")

    tar_index = {}
    for tar_path in tar_paths:
        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                if member.isfile():
                    tar_index[member.name] = str(tar_path)
    return tar_index


def _strip(value):
    return value.strip() if isinstance(value, str) else value
