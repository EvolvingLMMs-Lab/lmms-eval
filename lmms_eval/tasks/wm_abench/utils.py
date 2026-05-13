import re
from typing import Any, Dict, List, Optional

from PIL import Image

# Map 0-indexed int to letter
INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}
# Map 1-indexed int to letter
ONEIDX_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"}

IMAGE_CHOICE_PROMPT_SUFFIX = "\nPlease answer with only the letter (A, B, C, or D) of the correct option."


def _ensure_rgb(img) -> Image.Image:
    """Convert a PIL Image to RGB, handling None and dict (lazy HF Image) gracefully."""
    if img is None:
        return Image.new("RGB", (224, 224))
    if isinstance(img, dict):
        # HF datasets Image feature may be a dict with 'bytes' or 'path'
        if "bytes" in img and img["bytes"] is not None:
            import io

            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        if "path" in img and img["path"] is not None:
            return Image.open(img["path"]).convert("RGB")
        return Image.new("RGB", (224, 224))
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return img


def _has_image_choices(doc: dict) -> bool:
    return "image_choices" in doc and doc["image_choices"] is not None and len(doc["image_choices"]) > 0


def _has_letter_choices(doc: dict) -> bool:
    return "choices_str_letter" in doc and doc["choices_str_letter"]


def _normalize_answer(doc: dict) -> str:
    """Normalize the ground-truth answer to a single uppercase letter (A-D).

    Image-choice configs: answer is 0-indexed int -> map to A-D.
    Text-choice configs with letter choices: answer may be 1-indexed int or letter string.
    Open-ended configs: answer is a free-form string (e.g., 'yes'/'no').
    """
    answer = doc["answer"]

    if _has_image_choices(doc):
        # Image-choice: 0-indexed int
        if isinstance(answer, int):
            return INDEX_TO_LETTER.get(answer, str(answer))
        try:
            return INDEX_TO_LETTER.get(int(answer), str(answer))
        except (ValueError, TypeError):
            return str(answer).upper().strip()

    if _has_letter_choices(doc):
        # Text-choice with letter options available
        if isinstance(answer, int):
            return ONEIDX_TO_LETTER.get(answer, str(answer))
        answer_str = str(answer).strip().upper()
        if len(answer_str) == 1 and answer_str in "ABCDEF":
            return answer_str
        # Might be a number as string
        try:
            return ONEIDX_TO_LETTER.get(int(answer_str), answer_str)
        except (ValueError, TypeError):
            pass

    # Open-ended or other: return as-is (lowered for comparison)
    return str(doc["answer"]).strip().lower()


def _make_montage(source_imgs: list, choice_imgs: list) -> Image.Image:
    """Tile source images (top) and labeled choice images (bottom) into one image.

    Layout: source images in top row, choice images A/B/C/D in bottom row.
    Each cell is resized to a common size for uniformity.
    """
    from PIL import ImageDraw, ImageFont

    CELL = 384  # px per cell
    PAD = 4
    LABEL_H = 28

    all_sources = [img.resize((CELL, CELL)) for img in source_imgs]
    all_choices = [img.resize((CELL, CELL)) for img in choice_imgs]

    n_src = len(all_sources)
    n_ch = len(all_choices)
    cols = max(n_src, n_ch)

    row_h = CELL + LABEL_H + PAD
    width = cols * (CELL + PAD) + PAD
    height = PAD + row_h + row_h + PAD  # 2 rows

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Row 0: source images with "Source" label
    y0 = PAD
    for i, img in enumerate(all_sources):
        x = PAD + i * (CELL + PAD)
        draw.text((x + 2, y0), f"Source {i+1}" if n_src > 1 else "Source", fill=(0, 0, 0))
        canvas.paste(img, (x, y0 + LABEL_H))

    # Row 1: choice images with A/B/C/D labels
    y1 = PAD + row_h
    for i, img in enumerate(all_choices):
        x = PAD + i * (CELL + PAD)
        label = chr(65 + i)
        draw.text((x + 2, y1), f"Option {label}", fill=(0, 0, 0))
        canvas.paste(img, (x, y1 + LABEL_H))

    return canvas


def wm_abench_doc_to_visual(doc: dict) -> list:
    """Return list of PIL Images for the model to see.

    For image-choice configs: creates a labeled montage (source + choices)
    as a SINGLE image to avoid multi-image → video_fps issues in vllm.
    For text-choice / open-ended configs: only source images.
    """
    # Source images
    source_imgs = []
    source = doc.get("source")
    if source is not None:
        if isinstance(source, list):
            for img in source:
                source_imgs.append(_ensure_rgb(img))
        else:
            source_imgs.append(_ensure_rgb(source))

    # Image choices → montage
    if _has_image_choices(doc):
        choice_imgs = [_ensure_rgb(img) for img in doc["image_choices"]]
        montage = _make_montage(source_imgs, choice_imgs)
        return [montage]

    return source_imgs if source_imgs else [Image.new("RGB", (224, 224))]


def wm_abench_doc_to_text(doc: dict, lmms_eval_specific_kwargs: Optional[dict] = None) -> str:
    """Build the text prompt for the model.

    Image-choice configs: append instructions referencing choice images by letter.
    Text-choice configs with choices_str_letter: replace numeric choices with letter choices.
    Open-ended configs: use prompt as-is.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    prompt = doc.get("prompt", "")

    if _has_image_choices(doc):
        n_choices = len(doc["image_choices"])
        choice_labels = ", ".join(chr(65 + i) for i in range(n_choices))

        text = f"{prompt}\n\n" f"The image shows the source (top row) and answer options {choice_labels} (bottom row). " f"Which option correctly shows what happens next?" f"{IMAGE_CHOICE_PROMPT_SUFFIX}"
        return text

    if _has_letter_choices(doc):
        # Strip embedded numeric choices from the original prompt and use
        # letter-labeled choices instead for consistent answer extraction.
        choices_letter = doc["choices_str_letter"]

        # Try to extract the base question before numeric choice list.
        # Patterns: "... 1. choice1\n2. choice2..." or "... 1. choice1 2. choice2..."
        base_prompt = re.split(r"\s*(?:1\.|A\.)\s", prompt, maxsplit=1)[0].strip()
        if not base_prompt:
            base_prompt = prompt

        text = f"{base_prompt}\n{choices_letter}\n" f"Answer with the option's letter from the given choices directly."
        return text

    # Open-ended: use prompt as-is
    return prompt


def _extract_answer_letter(text: str) -> str:
    """Extract a single answer letter (A-D) from model response."""
    text = text.strip()
    if not text:
        return ""

    # Direct single letter
    if len(text) == 1 and text.upper() in "ABCDEF":
        return text.upper()

    # Pattern: (A), A., A), A:, etc.
    match = re.match(r"[\(\s]*([A-Fa-f])[\)\.\:\s]", text)
    if match:
        return match.group(1).upper()

    # Just starts with a letter followed by word boundary
    match = re.match(r"^([A-Fa-f])\b", text)
    if match:
        return match.group(1).upper()

    # Search anywhere in short responses
    if len(text) < 30:
        match = re.search(r"\b([A-Da-d])\b", text)
        if match:
            return match.group(1).upper()

    return text.strip()


def wm_abench_doc_to_target(doc: dict) -> str:
    """Return the ground truth answer as a normalized string."""
    return _normalize_answer(doc)


def _is_safety_blocked(text: str) -> bool:
    return text.startswith("[SAFETY_BLOCKED:")


def wm_abench_process_results(doc: dict, results: List[str]) -> Dict[str, Any]:
    """Process model output and compare to ground truth."""
    pred_raw = results[0].strip()
    gt = _normalize_answer(doc)
    safety_blocked = _is_safety_blocked(pred_raw)

    # For MC tasks (image-choice or letter-choice), extract letter
    if safety_blocked:
        pred = ""
    elif _has_image_choices(doc) or _has_letter_choices(doc):
        pred = _extract_answer_letter(pred_raw)
    else:
        # Open-ended: compare lowercase
        pred = pred_raw.lower().strip()

    is_correct = pred == gt

    result_entry = {
        "pred": pred,
        "pred_raw": pred_raw,
        "answer": gt,
        "is_correct": is_correct,
        "safety_blocked": safety_blocked,
    }

    return {
        "wm_abench_acc": result_entry,
        "wm_abench_acc_clean": result_entry,
        "wm_abench_blocked_rate": result_entry,
    }


def wm_abench_aggregate_results(results: List[Dict]) -> float:
    """Compute overall accuracy."""
    if not results:
        return 0.0
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    blocked = sum(1 for r in results if r.get("safety_blocked", False))
    if blocked:
        from loguru import logger

        logger.info(f"wm_abench_acc: {correct}/{total} correct, {blocked}/{total} safety_blocked")
    return correct / total


def wm_abench_aggregate_results_clean(results: List[Dict]) -> float:
    """Compute accuracy excluding safety-blocked samples."""
    if not results:
        return 0.0
    clean = [r for r in results if not r.get("safety_blocked", False)]
    if not clean:
        return 0.0
    correct = sum(1 for r in clean if r["is_correct"])
    return correct / len(clean)


def wm_abench_aggregate_blocked_rate(results: List[Dict]) -> float:
    """Compute fraction of samples that were safety-blocked."""
    if not results:
        return 0.0
    blocked = sum(1 for r in results if r.get("safety_blocked", False))
    return blocked / len(results)
