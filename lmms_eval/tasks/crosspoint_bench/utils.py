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
        (r'(?:coordinates?|position|location).*?(?:are|is)\s+([0-9.]+),\s*([0-9.]+)', False),
        (r'[xX]\s*:\s*([0-9.]+).*?[yY]\s*:\s*([0-9.]+)', False),
        (r'[yY]\s*:\s*([0-9.]+).*?[xX]\s*:\s*([0-9.]+)', True),
        (r'["\']?x["\']?\s*:\s*([0-9.]+).*?["\']?y["\']?\s*:\s*([0-9.]+)', False),
        (r'\[([0-9.]+),\s*([0-9.]+)\]', False),
        (r'\(([0-9.]+),\s*([0-9.]+)\)', False),
        (r'([0-9.]+),\s*([0-9.]+)', False),
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
        r'\\boxed\{(?:\\text\{)?([ABCD])(?:\..*?)?\}',
        r'\((?:Choice\s+)?([ABCD])\)',
        r'\*\*Answer:\s*([ABCD])\*\*',
        r'\*\*([ABCD])\.\s*(?:Yes|No)',
        r'(?:answer is|correct answer is|choose)\s*[:\s]*\*?\*?\(?([ABCD])\)?',
        r'\*\*([ABCD])\*\*',
        r'(?:^|\s|Answer:\s*)\(?([ABCD])\)?\s*[\.:,]',
        r'^\s*\(?([ABCD])\)?\s*$',
    ]
    for pat in patterns:
        m = re.search(pat, s, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    final_patterns = [
        r'(?:Final Answer|Therefore|So|Hence|Thus|The correct answer is|The answer is)'
        r'.*?(?:\\boxed\{([ABCD])\}|\*\*([ABCD])\*\*|\(?([ABCD])\)?)',
        r'Answer:\s*\(?([ABCD])\)?',
    ]
    for pat in final_patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            for g in m.groups():
                if g:
                    return g.upper()

    letters = re.findall(r'\b([ABCD])\b', s)
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


def crosspoint_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    pred = results[0] if results else ""
    task_type = str(doc.get("type", ""))
    level = str(doc.get("level", ""))
    answer = str(doc.get("answer", ""))
    hit = 0

    if task_type in COORDINATE_TASK_TYPES:
        coords = _extract_coordinates(pred)
        if coords is not None:
            x_raw, y_raw = coords
            # Resolve coordinate format. Heuristic + env override (matches the
            # vlmevalkit version): default to absolute pixel coordinates.
            coord_format = os.environ.get("CROSSPOINT_COORD_FORMAT", "absolute")
            img_paths = list(doc.get("images") or [])
            dims = None
            if img_paths:
                root = _image_root()
                dims = _image_size(osp.join(root, img_paths[0]))
            if dims is not None:
                w, h = dims
                if coord_format == "relative_1":
                    x_abs = int(x_raw * w)
                    y_abs = int(y_raw * h)
                elif coord_format == "relative_1000":
                    x_abs = int((x_raw / 1000.0) * w)
                    y_abs = int((y_raw / 1000.0) * h)
                else:
                    x_abs, y_abs = int(x_raw), int(y_raw)
            else:
                x_abs, y_abs = int(x_raw), int(y_raw)
            mask = _decode_base64_mask(answer)
            if mask is not None:
                hit = int(_point_in_mask(x_abs, y_abs, mask))
    elif task_type in MCQ_TASK_TYPES:
        letter = _extract_mcq_letter(pred)
        if letter is not None:
            hit = int(letter == answer.strip().upper())

    return {
        "crosspoint_accuracy": {
            "hit": hit,
            "type": task_type,
            "level": level,
        }
    }


def crosspoint_aggregate_results(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0

    # Overall
    overall = sum(r["hit"] for r in results) / len(results) * 100

    # By type
    from collections import defaultdict
    by_type = defaultdict(list)
    by_level = defaultdict(list)
    by_type_level = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r["hit"])
        by_level[r["level"]].append(r["hit"])
        by_type_level[(r["type"], r["level"])].append(r["hit"])

    lines = ["", "=" * 70, "CrossPoint-Bench Evaluation Results", "=" * 70,
             f"  {'Overall':<45} {overall:5.1f}%  ({sum(r['hit'] for r in results)}/{len(results)})"]
    for k in sorted(by_type):
        hits = by_type[k]
        lines.append(f"  type/{k:<40} {sum(hits)/len(hits)*100:5.1f}%  ({sum(hits)}/{len(hits)})")
    for k in sorted(by_level):
        hits = by_level[k]
        lines.append(f"  level/{k:<39} {sum(hits)/len(hits)*100:5.1f}%  ({sum(hits)}/{len(hits)})")
    for (t, lv) in sorted(by_type_level):
        hits = by_type_level[(t, lv)]
        lines.append(f"  {t}/{lv:<45} {sum(hits)/len(hits)*100:5.1f}%  ({sum(hits)}/{len(hits)})")
    lines.append("=" * 70)
    print("\n".join(lines))

    return overall / 100.0
