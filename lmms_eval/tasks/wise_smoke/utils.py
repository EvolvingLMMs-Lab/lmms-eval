import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import requests
from PIL import Image


def wise_smoke_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    visuals: List[Any] = []
    source_images = doc.get("source_images", [])
    if not isinstance(source_images, list):
        return visuals

    for image_url in source_images:
        if not isinstance(image_url, str) or not image_url.strip():
            continue
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        visuals.append(image)

    return visuals


def wise_smoke_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    prompt = doc.get("prompt", "").strip()
    hint = doc.get("hint", "").strip()
    if hint:
        prompt = f"{prompt}\nHint: {hint}"
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post = lmms_eval_specific_kwargs.get("post_prompt", "")
        prompt = f"{pre}{prompt}{post}"
    return prompt


def wise_smoke_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    pred = results[0] if results else "{}"
    try:
        payload = json.loads(pred)
    except Exception:
        payload = {}
    images = payload.get("images", []) if isinstance(payload, dict) else []
    ok = 0.0
    for p in images:
        if Path(p).exists():
            ok = 1.0
            break
    return {"wise_smoke_success": ok}
