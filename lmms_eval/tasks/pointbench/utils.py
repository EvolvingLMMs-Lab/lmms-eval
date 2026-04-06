import re
import zipfile
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List

import datasets
import numpy as np
import requests
from PIL import Image

from lmms_eval.tasks._task_utils.default_template_yaml import load_default_template_yaml
from lmms_eval.utils import eval_logger

POINTARENA_REPO = "PointArena/pointarena-data"
POINTARENA_ROWS_API = "https://datasets-server.huggingface.co/rows"

PROMPT_SUFFIX_0_999 = "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be integers between 0 and 999, representing the pixel locations scaled to a 1000x1000 grid."
PROMPT_SUFFIX_ORIGINAL = "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
FORMAT = "Return only list of tuples, do not add anything else."

config = load_default_template_yaml(__file__)


def pointbench_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(lambda _, idx: {"question_id": idx, "row_idx": idx}, with_indices=True)


def pointbench_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict[str, Any] | None = None) -> str:
    prompt_suffix_type = config.get("metadata", {}).get("prompt_suffix_type", "0_999")
    suffix = PROMPT_SUFFIX_0_999 if prompt_suffix_type == "0_999" else PROMPT_SUFFIX_ORIGINAL

    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    user_input = str(doc.get("user_input", "")).strip()
    return f"{pre_prompt}{user_input} {suffix} {FORMAT}{post_prompt}".strip()


@lru_cache(maxsize=4096)
def _get_image_url(row_idx: int) -> str:
    response = requests.get(
        POINTARENA_ROWS_API,
        params={"dataset": POINTARENA_REPO, "config": "default", "split": "train", "offset": int(row_idx), "length": 1},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError(f"No rows found for row_idx={row_idx}")
    return rows[0]["row"]["image"]["src"]


def _load_image(row_idx: int) -> Image.Image:
    image_url = _get_image_url(row_idx)
    response = requests.get(image_url, timeout=60)
    if response.status_code == 403:
        _get_image_url.cache_clear()
        image_url = _get_image_url(row_idx)
        response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def pointbench_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    row_idx = doc.get("row_idx", doc.get("question_id"))
    if row_idx is None:
        eval_logger.warning("pointbench: missing row_idx for doc={}", doc.get("image_filename", "unknown"))
        return []

    try:
        image = _load_image(int(row_idx))
    except Exception as exc:
        eval_logger.warning("pointbench: failed to load image for row_idx={} ({})", row_idx, exc)
        return []
    return [image]


@lru_cache(maxsize=1)
def _mask_zip_path() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=POINTARENA_REPO, repo_type="dataset", filename="selected_masks.zip")


@lru_cache(maxsize=1)
def _mask_member_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with zipfile.ZipFile(_mask_zip_path()) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".png"):
                continue
            mapping.setdefault(member.rsplit("/", 1)[-1], member)
    return mapping


@lru_cache(maxsize=4096)
def _load_mask(mask_filename: str) -> np.ndarray | None:
    member = _mask_member_map().get(mask_filename)
    if not member:
        return None

    with zipfile.ZipFile(_mask_zip_path()) as archive:
        with archive.open(member) as stream:
            mask = Image.open(BytesIO(stream.read())).convert("L")

    return (np.array(mask) > 127).astype(np.int32)


def _text_to_points(text: str, width: int, height: int) -> np.ndarray:
    pattern = r"\(([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\)"
    matches = re.findall(pattern, text)

    points = []
    for x_raw, y_raw in matches:
        x = float(x_raw)
        y = float(y_raw)

        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            px = int(round(x * width))
            py = int(round(y * height))
        elif 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
            px = int(round((x / 1000.0) * width))
            py = int(round((y / 1000.0) * height))
        else:
            px = int(round(x))
            py = int(round(y))

        points.append((px, py))

    return np.array(points, dtype=np.int32)


def pointbench_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, Dict[str, Any]]:
    key_name = "pointbench_acc"
    mask_filename = str(doc.get("mask_filename", ""))
    mask = _load_mask(mask_filename)
    response = result[0] if result else ""

    if mask is None:
        eval_logger.warning("pointbench: failed to find mask for file={}", mask_filename)
        submission = {
            "id": doc.get("question_id", doc.get("image_filename", "unknown")),
            "pred": response,
            "parsed_points": [],
            "accuracy": 0.0,
            "category": doc.get("category", "unknown"),
        }
        return {key_name: submission}

    points = _text_to_points(response, mask.shape[1], mask.shape[0])
    acc = 0.0
    if len(points) > 0:
        in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
        acc = np.concatenate([mask[points[in_range, 1], points[in_range, 0]], np.zeros(points.shape[0] - in_range.sum())]).mean()

    submission = {
        "id": doc.get("question_id", doc.get("image_filename", "unknown")),
        "pred": response,
        "parsed_points": list(map(tuple, points.tolist())),
        "accuracy": float(acc),
        "category": doc.get("category", "unknown"),
    }
    return {key_name: submission}


def pointbench_aggregate_results(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return float(np.mean([sample.get("accuracy", 0.0) for sample in results]))
