"""CrossPoint-Bench task for lmms-eval.

Cross-view point correspondence benchmark covering four sub-tasks:

- Fine-grained Grounding (coordinate output, in-mask hit)
- Visibility Reasoning (binary MCQ)
- Correspondence-Judgement (MCQ)
- Correspondence-Pointing (coordinate output, in-mask hit)

The HF dataset stores only the JSONL; image files live alongside under
``image/`` and are fetched once with ``snapshot_download`` and cached on
disk for subsequent ``doc_to_visual`` lookups.

Reference: https://arxiv.org/abs/2512.04686
"""

from __future__ import annotations

import base64
import io
import os
import os.path as osp
import re
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

REPO_ID = "WangYipu2002/CrossPoint-Bench"

COORDINATE_TASK_TYPES = {"Fine-grained Grounding", "Correspondence-Pointing"}
MCQ_TASK_TYPES = {"Visibility Reasoning", "Correspondence-Judgement"}

COORDINATE_PROMPT_SUFFIX = " Output the point coordinates in JSON format."


# ---------------------------------------------------------------------------
# Image resolution
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _image_root() -> str:
    """Download (once) and return the local path to the ``image/`` tree."""
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["image/**", "CrossPoint-Bench.jsonl"],
    )
    return osp.join(local_dir, "image")


def crosspoint_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    root = _image_root()
    out = []
    for rel in doc["images"]:
        path = osp.join(root, rel)
        with Image.open(path) as im:
            out.append(im.convert("RGB"))
    return out


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


def crosspoint_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Dict[str, Any] | None = None,
) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre = kwargs.get("pre_prompt", "")
    post = kwargs.get("post_prompt", "")
    question = str(doc["question"])
    if doc.get("type") in COORDINATE_TASK_TYPES:
        question = question + COORDINATE_PROMPT_SUFFIX
    return f"{pre}{question}{post}"


def crosspoint_doc_to_target(doc: Dict[str, Any]) -> str:
    return str(doc.get("answer", ""))


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def _extract_coordinates_from_json(text: str):
    patterns = [
        r'\{[^{}]*"(?:point_2d|point|coordinates?)"\s*:\s*\[([0-9.]+)\s*,\s*([0-9.]+)\]',
        r'\{[^{}]*"x"\s*:\s*([0-9.]+)[^{}]*"y"\s*:\s*([0-9.]+)',
        r'\{[^{}]*"y"\s*:\s*([0-9.]+)[^{}]*"x"\s*:\s*([0-9.]+)',
    ]
    for i, pat in enumerate(patterns):
        m = re.search(pat, text, re.DOTALL)
        if m:
            if i == 2:
                return float(m.group(2)), float(m.group(1))
            return float(m.group(1)), float(m.group(2))
    return None


def _extract_coordinates_from_text(text: str):
    patterns = [
        (r'<point\s+x="([0-9.]+)"\s+y="([0-9.]+)"[^>]*>', False),
        (r"(?:coordinates?|position|location).*?(?:are|is)\s+([0-9.]+),\s*([0-9.]+)", False),
        (r"[xX]\s*:\s*([0-9.]+).*?[yY]\s*:\s*([0-9.]+)", False),
        (r"[yY]\s*:\s*([0-9.]+).*?[xX]\s*:\s*([0-9.]+)", True),
        (r'["\']?x["\']?\s*:\s*([0-9.]+).*?["\']?y["\']?\s*:\s*([0-9.]+)', False),
        (r"\[([0-9.]+),\s*([0-9.]+)\]", False),
        (r"\(([0-9.]+),\s*([0-9.]+)\)", False),
        (r"([0-9.]+),\s*([0-9.]+)", False),
    ]
    for pat, swapped in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            x, y = float(m.group(1)), float(m.group(2))
            if swapped:
                x, y = y, x
            return x, y
    return None


def _extract_coordinates(text: str):
    return _extract_coordinates_from_json(text) or _extract_coordinates_from_text(text)


def _extract_mcq_letter(text: str):
    s = text.strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()

    patterns = [
        r"\\boxed\{(?:\\text\{)?([ABCD])(?:\..*?)?\}",
        r"\((?:Choice\s+)?([ABCD])\)",
        r"\*\*Answer:\s*([ABCD])\*\*",
        r"\*\*([ABCD])\.\s*(?:Yes|No)",
        r"(?:answer is|correct answer is|choose)\s*[:\s]*\*?\*?\(?([ABCD])\)?",
        r"\*\*([ABCD])\*\*",
        r"(?:^|\s|Answer:\s*)\(?([ABCD])\)?\s*[\.:,]",
        r"^\s*\(?([ABCD])\)?\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, s, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    final_patterns = [
        r"(?:Final Answer|Therefore|So|Hence|Thus|The correct answer is|The answer is)" r".*?(?:\\boxed\{([ABCD])\}|\*\*([ABCD])\*\*|\(?([ABCD])\)?)",
        r"Answer:\s*\(?([ABCD])\)?",
    ]
    for pat in final_patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            for g in m.groups():
                if g:
                    return g.upper()

    letters = re.findall(r"\b([ABCD])\b", s)
    if len(letters) == 1:
        return letters[0].upper()
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _decode_base64_mask(b64: str):
    try:
        return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("L"))
    except Exception:
        return None


def _point_in_mask(x: int, y: int, mask: np.ndarray, threshold: int = 128) -> bool:
    h, w = mask.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return False
    return bool(mask[y, x] > threshold)


def _image_size(path: str):
    try:
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return None


def _score(pred: str, doc: Dict[str, Any]) -> int:
    task_type = str(doc.get("type", ""))
    answer = str(doc.get("answer", ""))

    if task_type in COORDINATE_TASK_TYPES:
        coords = _extract_coordinates(pred)
        if coords is None:
            return 0
        x_raw, y_raw = coords
        coord_format = os.environ.get("CROSSPOINT_COORD_FORMAT", "absolute")
        img_paths = list(doc.get("images") or [])
        dims = None
        if img_paths:
            dims = _image_size(osp.join(_image_root(), img_paths[0]))
        if dims is not None:
            w, h = dims
            if coord_format == "relative_1":
                x_abs, y_abs = int(x_raw * w), int(y_raw * h)
            elif coord_format == "relative_1000":
                x_abs, y_abs = int((x_raw / 1000.0) * w), int((y_raw / 1000.0) * h)
            else:
                x_abs, y_abs = int(x_raw), int(y_raw)
        else:
            x_abs, y_abs = int(x_raw), int(y_raw)
        mask = _decode_base64_mask(answer)
        if mask is None:
            return 0
        return int(_point_in_mask(x_abs, y_abs, mask))

    if task_type in MCQ_TASK_TYPES:
        letter = _extract_mcq_letter(pred)
        if letter is None:
            return 0
        return int(letter == answer.strip().upper())

    return 0


# All per-doc metric values share the same {hit, type, level} structure so any
# aggregator can filter on type or level.
_METRIC_KEYS = (
    "crosspoint_accuracy",
    "fine_grained_grounding",
    "visibility_reasoning",
    "correspondence_judgement",
    "correspondence_pointing",
    "level_object",
    "level_part",
)


def crosspoint_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    pred = results[0] if results else ""
    item = {
        "hit": _score(pred, doc),
        "type": str(doc.get("type", "")),
        "level": str(doc.get("level", "")),
    }
    return {k: item for k in _METRIC_KEYS}


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def crosspoint_aggregate_overall(results: List[Dict[str, Any]]) -> float:
    return _mean([r["hit"] for r in results])


def _aggregate_by_type(results, target_type):
    return _mean([r["hit"] for r in results if r["type"] == target_type])


def _aggregate_by_level(results, target_level):
    return _mean([r["hit"] for r in results if r["level"] == target_level])


def crosspoint_aggregate_fine_grained_grounding(results):
    return _aggregate_by_type(results, "Fine-grained Grounding")


def crosspoint_aggregate_visibility_reasoning(results):
    return _aggregate_by_type(results, "Visibility Reasoning")


def crosspoint_aggregate_correspondence_judgement(results):
    return _aggregate_by_type(results, "Correspondence-Judgement")


def crosspoint_aggregate_correspondence_pointing(results):
    return _aggregate_by_type(results, "Correspondence-Pointing")


def crosspoint_aggregate_level_object(results):
    return _aggregate_by_level(results, "object")


def crosspoint_aggregate_level_part(results):
    return _aggregate_by_level(results, "part")
