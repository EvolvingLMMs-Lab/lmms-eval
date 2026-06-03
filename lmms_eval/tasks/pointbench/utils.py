import re
import unicodedata
import zipfile
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List

import datasets
import numpy as np
from PIL import Image

from lmms_eval.tasks._task_utils.default_template_yaml import load_default_template_yaml
from lmms_eval.utils import eval_logger

POINTARENA_REPO = "PointArena/pointarena-data"

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


def _zip_basename_key(name: str) -> str:
    """NFC-normalized basename so accented filenames match across data.json/zip."""
    return unicodedata.normalize("NFC", name.rsplit("/", 1)[-1])


def _lookup_member(member_map: Dict[str, str], filename: str) -> str | None:
    """Look up a zip member by basename, tolerating Unicode (NFC/NFD) differences."""
    if not filename:
        return None
    if filename in member_map:
        return member_map[filename]
    return member_map.get(_zip_basename_key(filename))


@lru_cache(maxsize=4)
def _zip_path(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=POINTARENA_REPO, repo_type="dataset", filename=filename)


def _mask_zip_path() -> str:
    return _zip_path("selected_masks.zip")


def _image_zip_path() -> str:
    return _zip_path("selected_images.zip")


def _build_member_map(zip_path: str, suffixes: tuple) -> Dict[str, str]:
    # Key on the raw basename, an NFC-normalized basename, and -- for archives
    # written without the UTF-8 flag -- a cp437->utf-8 recovered basename, so
    # lookups succeed regardless of how the filename was encoded upstream.
    mapping: Dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            member = info.filename
            if not member.lower().endswith(suffixes):
                continue
            base = member.rsplit("/", 1)[-1]
            mapping.setdefault(base, member)
            mapping.setdefault(_zip_basename_key(base), member)
            if not (info.flag_bits & 0x800):  # zipfile decoded the name as cp437
                try:
                    recovered = base.encode("cp437").decode("utf-8")
                    mapping.setdefault(_zip_basename_key(recovered), member)
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
    return mapping


@lru_cache(maxsize=1)
def _mask_member_map() -> Dict[str, str]:
    return _build_member_map(_mask_zip_path(), (".png",))


@lru_cache(maxsize=1)
def _image_member_map() -> Dict[str, str]:
    return _build_member_map(_image_zip_path(), (".jpg", ".jpeg", ".png", ".webp"))


def pointbench_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    # Load the image from selected_images.zip keyed by image_filename. The HF
    # datasets-server rows API serves an image-only ImageFolder in zip order,
    # which does NOT match data.json's question order -- fetching by row index
    # pairs each question with the wrong image. Keying by filename keeps the
    # (question, image) pair aligned and avoids the flaky rows API entirely.
    image_filename = str(doc.get("image_filename", ""))
    member = _lookup_member(_image_member_map(), image_filename)
    if not member:
        eval_logger.warning("pointbench: image not found in selected_images.zip for file={}", image_filename)
        return []
    try:
        with zipfile.ZipFile(_image_zip_path()) as archive:
            with archive.open(member) as stream:
                image = Image.open(BytesIO(stream.read())).convert("RGB")
    except Exception as exc:
        eval_logger.warning("pointbench: failed to load image for file={} ({})", image_filename, exc)
        return []
    return [image]


@lru_cache(maxsize=4096)
def _load_mask(mask_filename: str) -> np.ndarray | None:
    member = _lookup_member(_mask_member_map(), mask_filename)
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


def _binary_score(points: np.ndarray, mask: np.ndarray, category: str, expected_count: int) -> int:
    """Fork-style strict score: 1 if the prediction hits the mask, else 0.

    For non-counting categories only the FIRST predicted point must fall in the
    mask. For "counting" ALL predicted points must fall in the mask AND the
    number of points must equal the expected count. Mirrors the fork's
    pointbench `evaluate_answer`. ``points`` are already pixel coordinates.
    """
    if len(points) == 0:
        return 0
    height, width = mask.shape
    to_check = points if category == "counting" else points[:1]
    for x, y in to_check:
        if not (0 <= x < width and 0 <= y < height and mask[y, x]):
            return 0
    if category == "counting" and len(points) != expected_count:
        return 0
    return 1


def pointbench_process_results(doc: Dict[str, Any], result: List[str]) -> Dict[str, Dict[str, Any]]:
    mask_filename = str(doc.get("mask_filename", ""))
    mask = _load_mask(mask_filename)
    response = result[0] if result else ""
    sample_id = doc.get("question_id", doc.get("image_filename", "unknown"))
    category = doc.get("category", "unknown")

    if mask is None:
        eval_logger.warning("pointbench: failed to find mask for file={}", mask_filename)
        frac = {"id": sample_id, "pred": response, "parsed_points": [], "accuracy": 0.0, "category": category}
        binary = {"id": sample_id, "score": 0, "category": category}
        return {"pointbench_acc": frac, "pointbench_binary_acc": binary}

    points = _text_to_points(response, mask.shape[1], mask.shape[0])

    # Fraction metric (OSS native): mean mask value over all predicted points.
    acc = 0.0
    if len(points) > 0:
        in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
        acc = np.concatenate([mask[points[in_range, 1], points[in_range, 0]], np.zeros(points.shape[0] - in_range.sum())]).mean()

    # Binary metric (fork-style): first point in mask (all + count for counting).
    expected_count = int(doc.get("count", 0) or 0)
    binary = _binary_score(points, mask, category, expected_count)

    frac_submission = {
        "id": sample_id,
        "pred": response,
        "parsed_points": list(map(tuple, points.tolist())),
        "accuracy": float(acc),
        "category": category,
    }
    binary_submission = {"id": sample_id, "score": int(binary), "category": category}
    return {"pointbench_acc": frac_submission, "pointbench_binary_acc": binary_submission}


def pointbench_aggregate_results(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return float(np.mean([sample.get("accuracy", 0.0) for sample in results]))


def pointbench_binary_aggregate_results(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return float(np.mean([sample.get("score", 0) for sample in results]))
