import base64
import re
from io import BytesIO
from typing import Dict, List, Any
from PIL import Image

def doc_to_visual(sample: dict) -> list:
    visual_list = []

    if "image" in sample:
        img_val = sample.get("image")
        if img_val:
            if img_val.startswith("data:image"):
                img_val = re.sub(r"^data:image/[^;]+;base64,", "", img_val)
            img = Image.open(BytesIO(base64.b64decode(img_val)))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            visual_list.append(img)

    question = sample.get("question", "")
    image_tag_nums = re.findall(r"<image_(\d+)>", question)
    for num in image_tag_nums:
        img_col = f"image_{num}"
        img_val = sample.get(img_col)
        if img_val:
            if img_val.startswith("data:image"):
                img_val = re.sub(r"^data:image/[^;]+;base64,", "", img_val)
            img = Image.open(BytesIO(base64.b64decode(img_val)))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            visual_list.append(img)

    return visual_list


def pil_to_base64_url(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"


def doc_to_text(sample: Dict[str, Any], lmms_kwargs: Dict[str, Any] = None) -> str:
    pre_prompt = lmms_kwargs.get("pre_prompt", "") if lmms_kwargs else ""
    post_prompt = lmms_kwargs.get("post_prompt", "") if lmms_kwargs else ""
    question = str(sample.get("question", "")).strip()

    options = sample.get("options", [])
    if isinstance(options, dict):
        options = list(options.values())
    elif not isinstance(options, list):
        options = [str(options)]

    options_text = ""
    if options:
        letters = ["A", "B", "C", "D"]
        options_text = "\n".join(f"{letters[i]}: {opt}" for i, opt in enumerate(options) if i < len(letters))

    return f"{pre_prompt}\n{question}\n{options_text}\n{post_prompt}".strip()


def doc_to_messages(sample: Dict[str, Any], lmms_kwargs: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    text_content = doc_to_text(sample, lmms_kwargs)
    image_list = doc_to_visual(sample)

    content = [{"type": "text", "text": text_content}]

    for img in image_list:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        content.append({
            "type": "image",
            "url": f"data:image/png;base64,{img_b64}"
        })

    return [{"role": "user", "content": content}]


def process_results(sample, outputs, *args, **kwargs):
    target = sample.get("answer", "").strip()
    return {"target": target, "sample_id": sample["id"]}


def mmesci_agg(results: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0}

    correct = sum(1 for r in results if r["prediction"] == r["target"])
    return {
        "accuracy": round(correct / total, 4),
        "total_samples": total,
        "correct_samples": correct,
    }
    