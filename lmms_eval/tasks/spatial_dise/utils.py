import io
import os
import os.path as osp
import re
import string
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
MERGE_IMAGE_COLUMNS = [
    ("image", "merged image"),
]
SEPARATE_IMAGE_COLUMNS = [
    ("question_image_path", "separate question image"),
    ("question_image_1_path", "separate question image 1"),
    ("question_image_2_path", "separate question image 2"),
    ("option_a_image_path", "separate option A image"),
    ("option_b_image_path", "separate option B image"),
    ("option_c_image_path", "separate option C image"),
    ("option_d_image_path", "separate option D image"),
]


def spatial_dise_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return _process_docs(dataset, image_mode="merge")


def spatial_dise_process_docs_separate(dataset: datasets.Dataset) -> datasets.Dataset:
    return _process_docs(dataset, image_mode="separate")


def _process_docs(dataset: datasets.Dataset, image_mode: str) -> datasets.Dataset:
    dataset_root = _dataset_root()
    tar_index = _tar_index(dataset_root)

    def _process_doc(doc, idx):
        clean_doc = {str(key).strip(): _strip(value) for key, value in doc.items()}
        image_refs = _image_refs(clean_doc, tar_index, image_mode)
        if len(image_refs) == 0:
            raise FileNotFoundError(f"Spatial-DISE image {clean_doc['image']} not found in tar shards under {dataset_root}")
        option_letters = _option_letters(clean_doc.get("options", ""))

        return {
            "id": f"benchmark_{idx}",
            "question": clean_doc["question"],
            "answer": clean_doc["answer"].upper(),
            "option_letters": option_letters,
            "image_path": image_refs[0]["path"],
            "image_shard": image_refs[0]["shard"],
            "image_paths": [ref["path"] for ref in image_refs],
            "image_shards": [ref["shard"] for ref in image_refs],
            "image_roles": [ref["role"] for ref in image_refs],
            "image_mode": image_mode,
            "category": clean_doc.get("category", ""),
            "difficulty": clean_doc.get("difficulty", ""),
            "source": clean_doc.get("source", ""),
            "dise_category": clean_doc.get("dise_category", ""),
        }

    return dataset.map(_process_doc, with_indices=True)


def spatial_dise_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    images = []
    for image_path, image_shard in zip(doc["image_paths"], doc["image_shards"]):
        images.append(_open_tar_image(image_shard, image_path))
    return images


def spatial_dise_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    option_text = ", ".join(doc.get("option_letters") or ["A", "B", "C", "D"])
    if doc.get("image_mode") == "separate":
        image_context = "Images are provided as separate question/view/option images from the original sample. " f"Use all images together. The answer choices are labeled {option_text}.\n"
    else:
        image_context = f"The image contains answer choices labeled {option_text}.\n"
    return f"{pre_prompt}{image_context}{doc['question'].strip()}{post_prompt}".strip()


def spatial_dise_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    response = results[0]
    target = doc["answer"].strip().upper()
    pred = _extract_answer(response, doc.get("option_letters"))
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


def _extract_answer(response: str, choices=None) -> str:
    response = str(response).strip()
    choices = _normalize_choices(choices)
    letters = "".join(re.escape(choice) for choice in choices)
    try:
        from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

        answer = extract_mcq_answer(response, choices=choices)
        if answer:
            return answer.strip().upper()
    except Exception:
        pass

    patterns = [
        rf"(?:answer|final answer|correct answer)\s*[:：]?\s*\(?([{letters}])\)?",
        rf"^\s*\(?([{letters}])\)?(?:[\.\):\s]|$)",
        rf"\b([{letters}])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def _option_letters(value) -> List[str]:
    if value is None:
        return list("ABCD")
    letters = []
    for option in str(value).replace("，", ",").split(","):
        option = option.strip().upper()
        if option and option[0] in string.ascii_uppercase and option[0] not in letters:
            letters.append(option[0])
    return letters or list("ABCD")


def _normalize_choices(choices) -> List[str]:
    if not choices:
        return list("ABCD")
    normalized = []
    for choice in choices:
        choice = str(choice).strip().upper()
        if choice and choice[0] in string.ascii_uppercase and choice[0] not in normalized:
            normalized.append(choice[0])
    return normalized or list("ABCD")


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


def _image_refs(doc: Dict[str, Any], tar_index: Dict[str, str], image_mode: str) -> List[Dict[str, str]]:
    refs = []
    seen = set()
    columns = SEPARATE_IMAGE_COLUMNS if image_mode == "separate" else MERGE_IMAGE_COLUMNS
    for column, role in columns:
        value = doc.get(column, "")
        if value is None:
            continue
        value = str(value).strip()
        if not value or value.lower() == "nan":
            continue
        member = _csv_path_to_tar_member(value)
        if member in seen:
            continue
        shard = tar_index.get(member)
        if shard is None:
            raise FileNotFoundError(f"Spatial-DISE image {column}={value} not found in tar shards")
        refs.append({"role": role, "path": member, "shard": shard})
        seen.add(member)
    return refs


def _open_tar_image(shard: str, member: str) -> Image.Image:
    with tarfile.open(shard) as tf:
        image_file = tf.extractfile(member)
        if image_file is None:
            raise FileNotFoundError(f"{member} not found in {shard}")
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    return image


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
