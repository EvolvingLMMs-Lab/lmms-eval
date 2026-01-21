import io
import os
import re
from typing import Dict, List, Optional, Tuple

from datasets import Dataset
from PIL import Image

# IllusionBench (Arshia Hemmat et al.) HF dataset has fields:
# - image_name: str, e.g. "animal-1-Medieval_Village-Easy-1.5-90.png"
# - image: datasets.Image(decode=False) -> {"bytes": ..., "path": ...}
#
# Official evaluation reports shape recall and scene recall across three subsets:
# Illusion-IN / Illusion-LOGO / Illusion-ICON:
# - https://arshiahemmat.github.io/illusionbench/

_DIFF_TOKENS = {"easy", "medium", "hard"}

# ============ 候选类别定义 ============
# Shape candidates (原始大小写，与原始代码库对齐)
SHAPE_CANDIDATES_ICON = [
    "Animal", "Face_Emoji", "Music", "Sport", "Stationery", "Vehicle"
]

SHAPE_CANDIDATES_LOGO = [
    "Adidas", "Amazon", "Apple", "Audi", "BMW", "Mercedes Benz", "Facebook", 
    "Google", "Instagram", "Mcdonalds", "Nasa", "Nike", "Olympics", 
    "Playstation", "Puma", "Reebok", "Spotify", "Starbucks", "Tesla", 
    "Telegram", "Ubuntu"
]

SHAPE_CANDIDATES_IN = [
    "Airplane", "Bicycle", "Bird", "Bottle", "Car", "Cat", "Dog", "Dolphin",
    "Fork", "Guitar", "Mug", "Panda", "Paper_clip", "Sailboat", "Scooter", "Teapot"
]

# Scene candidates (与原始代码库一致)
SIMPLE_SCENE_CANDIDATES = [
    "Ocean", "Origami", "Forest", "Cloud", "Sand_dune"
]

COMPLEX_SCENE_CANDIDATES = [
    "Medieval_Village", "City", "Underwater_ruins", "Museum", 
    "Bazaar_market", "Time_square"
]

SCENE_CANDIDATES = SIMPLE_SCENE_CANDIDATES + COMPLEX_SCENE_CANDIDATES


def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    # unify separators
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    # remove most punctuation
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _recall_match(pred: str, gt: str) -> int:
    """
    "Recall" here means: does the model mention the correct label somewhere in its output.
    This matches the original codebase evaluation style (substring matching).
    Uses simple lowercase matching like the original codebase: class_name.lower() in prediction
    """
    if not pred or not gt:
        return 0
    # Simple lowercase substring matching (like original codebase)
    pred_lower = pred.strip().lower()
    gt_lower = gt.strip().lower()
    # Also handle underscore/space variations
    gt_variants = [gt_lower, gt_lower.replace("_", " "), gt_lower.replace("-", " ")]
    return int(any(variant in pred_lower for variant in gt_variants if variant))


def _extract_answer(pred: str) -> str:
    """
    Extract answer from 'Answer: XX' format.
    Returns the extracted answer or empty string if not found.
    """
    # Try to match "Answer: XX" pattern (case insensitive)
    match = re.search(r"answer\s*:\s*([^\n,\.]+)", pred, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # No match found, return empty string (will result in 0 score)
    return ""


def _strict_match(pred: str, gt: str) -> int:
    """
    Strict match: extracted answer must equal ground truth (after normalization).
    """
    p = _normalize_text(pred)
    g = _normalize_text(gt)
    if not g:
        return 0
    return int(p == g)


def _strip_trailing_index(shape_raw: str) -> str:
    """
    Many filenames look like 'animal-1-...'. We treat 'animal' as the shape label and
    drop trailing purely-numeric suffix if present.
    """
    parts = (shape_raw or "").split("-")
    if len(parts) >= 2 and re.fullmatch(r"\d+", parts[-1] or ""):
        return "-".join(parts[:-1])
    return shape_raw


def parse_image_name(image_name: str) -> Tuple[str, str]:
    """
    Parse image_name -> (shape, scene)
    Example:
      animal-1-Medieval_Village-Easy-1.5-90.png -> ("animal", "Medieval_Village")
    """
    base = os.path.basename(image_name or "")
    base = re.sub(r"\.(png|jpg|jpeg|webp)$", "", base, flags=re.IGNORECASE)
    parts = [p for p in base.split("-") if p != ""]

    diff_idx: Optional[int] = None
    for i, p in enumerate(parts):
        if p.lower() in _DIFF_TOKENS:
            diff_idx = i
            break
    if diff_idx is None or diff_idx < 1:
        raise ValueError(f"Unrecognized image_name format (no difficulty token): {image_name}")

    scene = parts[diff_idx - 1]
    shape_raw = "-".join(parts[: diff_idx - 1])
    shape = _strip_trailing_index(shape_raw)

    if not shape or not scene:
        raise ValueError(f"Unrecognized image_name format (empty shape/scene): {image_name}")
    return shape, scene


def illusionbench_arshia_process_docs(dataset: Dataset) -> Dataset:
    """
    Add parsed GT fields. Keep schema simple for lmms-eval:
      - image_name
      - image
      - shape_gt
      - scene_gt
    """
    rows: List[Dict] = []
    for ex in dataset:
        image_name = ex.get("image_name")
        if not image_name:
            continue
        try:
            shape_gt, scene_gt = parse_image_name(image_name)
        except Exception:
            continue
        rows.append(
            {
                "image_name": image_name,
                "image": ex.get("image"),
                "shape_gt": shape_gt,
                "scene_gt": scene_gt,
            }
        )
    return Dataset.from_list(rows)


def _decode_image_field(image_field) -> Image.Image:
    """
    Supports:
    - PIL.Image.Image (already decoded)
    - dict with {'bytes': ..., 'path': ...}
    """
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    if isinstance(image_field, dict):
        b = image_field.get("bytes")
        if b is None:
            raise ValueError("image dict missing bytes; cannot decode")
        with Image.open(io.BytesIO(b)) as im:
            return im.convert("RGB")
    raise TypeError(f"Unsupported image field type: {type(image_field)}")


def illusionbench_arshia_doc_to_visual(doc):
    return [_decode_image_field(doc["image"])]


def illusionbench_arshia_doc_to_text(doc):
    # Keep it strict for easy parsing; allow models to think, but enforce output format.
    return (
        "You are given an image where scene elements form an abstract SHAPE.\n"
        "Task:\n"
        "1) Identify the abstract shape.\n"
        "2) Identify the scene.\n\n"
        "Reply in exactly TWO lines using this format:\n"
        "Shape: <shape>\n"
        "Scene: <scene>\n"
    )


def illusionbench_arshia_doc_to_text_shape(doc):
    return (
        "You are given an image where scene elements form an abstract SHAPE.\n"
        "Task: Identify the abstract shape.\n\n"
        "Reply in ONE line using this format:\n"
        "Shape: <shape>\n"
    )


def _build_shape_prompt(shape_candidates: List[str], scene_candidates: List[str], task_type: str = "icon") -> str:
    """
    Build shape prompt matching original codebase format.
    Options list mixes shape candidates + scene candidates (simple + complex).
    task_type: "icon", "logo", or "in"
    """
    # Combine shape and scene candidates as in original codebase
    all_options = shape_candidates + scene_candidates
    shape_string = ", ".join(shape_candidates)
    scene_string = ", ".join(scene_candidates)
    
    if task_type == "icon":
        return (
            f"This image contains a icon integrated into a background, where elements of the background contribute to forming the icon. "
            f"Identify the icon that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string} "
            f"Provide your response by stating only the single, most accurate class name that represents the icon. "
            f"You have to respond with a single word."
        )
    elif task_type == "logo":
        return (
            f"This image contains a icon integrated into a background, where elements of the background contribute to forming the logo. "
            f"Identify the logo that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string} "
            f"Provide your response by stating only the single, most accurate class name that represents the logo. "
            f"You have to respond with a single word."
        )
    else:  # "in" or "sin"
        return (
            f"This image contains a icon integrated into a background, where elements of the background contribute to forming the icon. "
            f"Identify the shape that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string} "
            f"Provide your response by stating only the single, most accurate class name that represents the icon. "
            f"You have to respond with a single word."
    )


def illusionbench_arshia_doc_to_text_shape_icon(doc):
    return _build_shape_prompt(SHAPE_CANDIDATES_ICON, SCENE_CANDIDATES, "icon")


def illusionbench_arshia_doc_to_text_shape_logo(doc):
    return _build_shape_prompt(SHAPE_CANDIDATES_LOGO, SCENE_CANDIDATES, "logo")


def illusionbench_arshia_doc_to_text_shape_in(doc):
    return _build_shape_prompt(SHAPE_CANDIDATES_IN, SCENE_CANDIDATES, "in")


def _build_scene_prompt(shape_candidates: List[str], scene_candidates: List[str], task_type: str = "icon") -> str:
    """
    Build scene prompt matching original codebase format.
    Options list mixes shape candidates + scene candidates (simple + complex).
    task_type: "icon", "logo", or "in"
    """
    shape_string = ", ".join(shape_candidates)
    scene_string = ", ".join(scene_candidates)
    
    if task_type == "icon":
        return (
            f"This image contains an icon integrated into a background, where elements of the background contribute to forming the icon. "
            f"Identify the background that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string}. "
            f"Provide your response by stating only the single, most accurate class name that represents the background. "
            f"You have to respond with a single word."
        )
    elif task_type == "logo":
        return (
            f"This image contains an icon integrated into a background, where elements of the background contribute to forming the logo. "
            f"Identify the background that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string}. "
            f"Provide your response by stating only the single, most accurate class name that represents the background. "
            f"You have to respond with a single word."
        )
    else:  # "in" or "sin"
        return (
            f"This image contains an icon integrated into a background, where elements of the background contribute to forming the icon. "
            f"Identify the background that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string}. "
            f"Provide your response by stating only the single, most accurate class name that represents the background. "
            f"You have to respond with a single word."
    )


def illusionbench_arshia_doc_to_text_scene(doc):
    # Default to icon for backward compatibility
    # In practice, different YAML files call different functions
    return _build_scene_prompt(SHAPE_CANDIDATES_ICON, SCENE_CANDIDATES, "icon")


def illusionbench_arshia_doc_to_text_scene_icon(doc):
    """Scene task for ICON subset"""
    return _build_scene_prompt(SHAPE_CANDIDATES_ICON, SCENE_CANDIDATES, "icon")


def illusionbench_arshia_doc_to_text_scene_logo(doc):
    """Scene task for LOGO subset"""
    return _build_scene_prompt(SHAPE_CANDIDATES_LOGO, SCENE_CANDIDATES, "logo")


def illusionbench_arshia_doc_to_text_scene_in(doc):
    """Scene task for IN subset"""
    return _build_scene_prompt(SHAPE_CANDIDATES_IN, SCENE_CANDIDATES, "in")


_LINE_SHAPE = re.compile(r"^\s*shape\s*:\s*(?P<v>.+?)\s*$", re.IGNORECASE)
_LINE_SCENE = re.compile(r"^\s*scene\s*:\s*(?P<v>.+?)\s*$", re.IGNORECASE)


def _extract_fields(pred: str) -> Tuple[str, str]:
    shape_pred = ""
    scene_pred = ""
    for line in (pred or "").splitlines():
        m1 = _LINE_SHAPE.match(line)
        if m1:
            shape_pred = m1.group("v").strip()
            continue
        m2 = _LINE_SCENE.match(line)
        if m2:
            scene_pred = m2.group("v").strip()
            continue
    # Fallback: if model didn't follow format, use full text for recall matching
    if not shape_pred:
        shape_pred = pred or ""
    if not scene_pred:
        scene_pred = pred or ""
    return shape_pred, scene_pred


def illusionbench_arshia_process_results(doc, results):
    pred = str(results[0]) if results else ""
    shape_pred, scene_pred = _extract_fields(pred)
    return {
        "shape_recall": _recall_match(shape_pred, doc.get("shape_gt", "")),
        "scene_recall": _recall_match(scene_pred, doc.get("scene_gt", "")),
    }


def illusionbench_arshia_process_results_shape(doc, results):
    """
    Process shape results using recall-style matching (substring matching) 
    to match original codebase evaluation style.
    """
    pred = str(results[0]) if results else ""
    # Use recall_match instead of strict_match to match original codebase
    # Original codebase: class_name.lower() in prediction
    return {
        "shape_recall": _recall_match(pred, doc.get("shape_gt", "")),
    }


def illusionbench_arshia_process_results_scene(doc, results):
    """
    Process scene results using recall-style matching (substring matching)
    to match original codebase evaluation style.
    """
    pred = str(results[0]) if results else ""
    # Use recall_match instead of strict_match to match original codebase
    # Original codebase: class_name.lower() in prediction
    return {
        "scene_recall": _recall_match(pred, doc.get("scene_gt", "")),
    }


def illusionbench_arshia_aggregate(results: List[Optional[int]]) -> float:
    vals = [v for v in results if v is not None]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def illusionbench_arshia_doc_to_target(doc):
    # Metrics are recall-only; target text is unused.
    return ""


# ============ Split shape/scene visual_cot prompts (for separate tasks) ============

def illusionbench_arshia_doc_to_text_visual_cot_icon_shape(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-ICON subset - SHAPE task only"""
    # generation_prompt: NO candidate leakage - only describe the task
    generation_prompt = (
        "This image shows a scene where elements are carefully arranged to form a hidden shape. "
        "Your task: Extract and visualize this hidden shape. "
        "Generate a clear image that highlights the shape's outline, contours, and structure. "
        "Make the hidden shape prominent and easily recognizable."
    )

    # question_prompt: Add auxiliary image explanation + use original codebase format with mixed options
    shape_string = ", ".join(SHAPE_CANDIDATES_ICON)
    scene_string = ", ".join(SCENE_CANDIDATES)
    question_prompt = (
        f"You are given TWO images: the original image and an auxiliary visualization. "
        f"This image contains a icon integrated into a background, where elements of the background contribute to forming the icon. "
        f"Identify the icon that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string} "
        f"Provide your response by stating only the single, most accurate class name that represents the icon. "
        f"You have to respond with a single word."
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


def illusionbench_arshia_doc_to_text_visual_cot_icon_scene(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-ICON subset - SCENE task only"""
    # generation_prompt: NO candidate leakage
    generation_prompt = (
        "This image depicts a specific scene or environment. "
        "Your task: Analyze and enhance the scene characteristics. "
        "Generate a clear visualization that emphasizes the environmental features and setting."
    )

    # question_prompt: Add auxiliary image explanation + use original codebase format with mixed options
    shape_string = ", ".join(SHAPE_CANDIDATES_ICON)
    scene_string = ", ".join(SCENE_CANDIDATES)
    question_prompt = (
        f"You are given TWO images: the original image and an auxiliary visualization. "
        f"This image contains an icon integrated into a background, where elements of the background contribute to forming the icon. "
        f"Identify the background that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string}. "
        f"Provide your response by stating only the single, most accurate class name that represents the background. "
        f"You have to respond with a single word."
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


def illusionbench_arshia_doc_to_text_visual_cot_logo_shape(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-LOGO subset - SHAPE task only"""
    # generation_prompt: NO candidate leakage
    generation_prompt = (
        "This image shows a scene where elements are carefully arranged to form a hidden shape. "
        "Your task: Extract and visualize this hidden shape. "
        "Generate a clear image that highlights the shape's distinctive outline and design elements."
    )

    # question_prompt: Add auxiliary image explanation + use original codebase format with mixed options
    shape_string = ", ".join(SHAPE_CANDIDATES_LOGO)
    scene_string = ", ".join(SCENE_CANDIDATES)
    question_prompt = (
        f"You are given TWO images: the original image and an auxiliary visualization. "
        f"This image contains a icon integrated into a background, where elements of the background contribute to forming the logo. "
        f"Identify the logo that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string} "
        f"Provide your response by stating only the single, most accurate class name that represents the logo. "
        f"You have to respond with a single word."
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


def illusionbench_arshia_doc_to_text_visual_cot_logo_scene(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-LOGO subset - SCENE task only"""
    # generation_prompt: NO candidate leakage
    generation_prompt = (
        "This image depicts a specific scene or environment. "
        "Your task: Analyze and enhance the scene characteristics. "
        "Generate a clear visualization that emphasizes the environmental features and setting."
    )

    # question_prompt: Add auxiliary image explanation + use original codebase format with mixed options
    shape_string = ", ".join(SHAPE_CANDIDATES_LOGO)
    scene_string = ", ".join(SCENE_CANDIDATES)
    question_prompt = (
        f"You are given TWO images: the original image and an auxiliary visualization. "
        f"This image contains an icon integrated into a background, where elements of the background contribute to forming the logo. "
        f"Identify the background that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string}. "
        f"Provide your response by stating only the single, most accurate class name that represents the background. "
        f"You have to respond with a single word."
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


def illusionbench_arshia_doc_to_text_visual_cot_in_shape(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-IN subset - SHAPE task only"""
    # generation_prompt: NO candidate leakage
    generation_prompt = (
        "This image shows a scene where elements are carefully arranged to form a hidden shape. "
        "Your task: Extract and visualize this hidden shape. "
        "Generate a clear image that highlights the shape's outline and recognizable features."
    )

    # question_prompt: Add auxiliary image explanation + use original codebase format with mixed options
    shape_string = ", ".join(SHAPE_CANDIDATES_IN)
    scene_string = ", ".join(SCENE_CANDIDATES)
    question_prompt = (
        f"You are given TWO images: the original image and an auxiliary visualization. "
        f"This image contains a icon integrated into a background, where elements of the background contribute to forming the icon. "
        f"Identify the shape that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string} "
        f"Provide your response by stating only the single, most accurate class name that represents the icon. "
        f"You have to respond with a single word."
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


def illusionbench_arshia_doc_to_text_visual_cot_in_scene(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-IN subset - SCENE task only"""
    # generation_prompt: NO candidate leakage
    generation_prompt = (
        "This image depicts a specific scene or environment. "
        "Your task: Analyze and enhance the scene characteristics. "
        "Generate a clear visualization that emphasizes the environmental features and setting."
    )

    # question_prompt: Add auxiliary image explanation + use original codebase format with mixed options
    shape_string = ", ".join(SHAPE_CANDIDATES_IN)
    scene_string = ", ".join(SCENE_CANDIDATES)
    question_prompt = (
        f"You are given TWO images: the original image and an auxiliary visualization. "
        f"This image contains an icon integrated into a background, where elements of the background contribute to forming the icon. "
        f"Identify the background that is represented in the image by choosing exclusively among the following options:{shape_string},{scene_string}. "
        f"Provide your response by stating only the single, most accurate class name that represents the background. "
        f"You have to respond with a single word."
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


