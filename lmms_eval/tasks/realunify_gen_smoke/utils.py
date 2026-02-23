import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


def realunify_gen_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    for key in ["ref_image", "initial_image", "image", "cand0_image", "cand1_image"]:
        img_data = doc.get(key)
        if not img_data:
            continue
        if isinstance(img_data, Image.Image):
            return [img_data.convert("RGB")]
        if isinstance(img_data, bytes):
            return [Image.open(BytesIO(img_data)).convert("RGB")]
        if isinstance(img_data, dict) and "bytes" in img_data:
            return [Image.open(BytesIO(img_data["bytes"])).convert("RGB")]
    fallback = Image.new("RGB", (512, 512), color=(245, 245, 245))
    return [fallback]


def realunify_gen_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    question = doc.get("question", "").strip()
    prompt = "Create one edited image that helps solve this unified reasoning puzzle. " "Keep visual style consistent with the input image and avoid adding text overlays.\n\n" f"Problem: {question}"
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post = lmms_eval_specific_kwargs.get("post_prompt", "")
        prompt = f"{pre}{prompt}{post}"
    return prompt


def realunify_gen_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
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
    return {"realunify_gen_smoke_success": ok}
