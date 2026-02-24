from __future__ import annotations

import json
import os
from typing import Any

from PIL import Image


def ice_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    src = doc.get("source_image", "")
    if isinstance(src, str) and src and os.path.exists(src):
        return [Image.open(src).convert("RGB")]
    return []


def ice_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None) -> str:
    instruction = str(doc.get("instruction", "")).strip()
    if lmms_eval_specific_kwargs:
        pre_prompt = str(lmms_eval_specific_kwargs.get("pre_prompt", ""))
        post_prompt = str(lmms_eval_specific_kwargs.get("post_prompt", ""))
        return f"{pre_prompt}{instruction}{post_prompt}"
    return instruction


def ice_doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc.get("instruction", ""))


def ice_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    if not results:
        return {"artifact_saved": 0.0}

    raw = results[0]
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"artifact_saved": 0.0}

    images = parsed.get("images", []) if isinstance(parsed, dict) else []
    if not isinstance(images, list) or not images:
        return {"artifact_saved": 0.0}

    first = images[0]
    if isinstance(first, str) and os.path.exists(first):
        return {"artifact_saved": 1.0}
    return {"artifact_saved": 0.0}
