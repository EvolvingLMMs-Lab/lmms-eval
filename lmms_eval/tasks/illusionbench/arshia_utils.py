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
SHAPE_CANDIDATES_ICON = [
    "animal", "vehicle", "stationary", "sport", "music", "face_emoji"
]

SHAPE_CANDIDATES_LOGO = [
    "tesla", "starbucks", "mcdonalds", "adidas", "reebok", "bmw", "ubuntu",
    "benz", "telegram", "nike", "apple", "puma", "facebook", "playstation",
    "instagram", "audi", "olympics", "google", "spotify", "amazon", "nasa"
]

SHAPE_CANDIDATES_IN = [
    "guitar", "teapot", "cat", "paper_clip", "bird", "dolphin", "mug",
    "bicycle", "bottle", "panda", "dog", "sailboat", "car", "fork",
    "scooter", "airplane"
]

SCENE_CANDIDATES = [
    "Underwater_ruins", "Time_square", "Medieval_Village", "City", "Museum",
    "Cloud", "Ocean", "Sand_dune", "Bazaar_market", "Forest", "Origami"
]


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
    This is intentionally more tolerant than strict exact-match because the website reports recall.
    """
    p = _normalize_text(pred)
    g = _normalize_text(gt)
    if not g:
        return 0
    return int(g in p)


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


def _build_shape_prompt(candidates: List[str]) -> str:
    options = ", ".join(candidates)
    return (
        "You are given an image where scene elements form an abstract SHAPE.\n"
        "Task: Identify what shape is hidden in this image.\n\n"
        f"Options: [{options}]\n\n"
        "Reply with ONLY ONE word from the options above.\n"
    )


def illusionbench_arshia_doc_to_text_shape_icon(doc):
    return _build_shape_prompt(SHAPE_CANDIDATES_ICON)


def illusionbench_arshia_doc_to_text_shape_logo(doc):
    return _build_shape_prompt(SHAPE_CANDIDATES_LOGO)


def illusionbench_arshia_doc_to_text_shape_in(doc):
    return _build_shape_prompt(SHAPE_CANDIDATES_IN)


def illusionbench_arshia_doc_to_text_scene(doc):
    return (
        "You are given an image depicting a SCENE.\n"
        "Task: Identify the scene.\n\n"
        "Reply in ONE line using this format:\n"
        "Scene: <scene>\n"
    )


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
    pred = str(results[0]) if results else ""
    shape_pred, _ = _extract_fields(pred)
    return {
        "shape_recall": _recall_match(shape_pred, doc.get("shape_gt", "")),
    }


def illusionbench_arshia_process_results_scene(doc, results):
    pred = str(results[0]) if results else ""
    _, scene_pred = _extract_fields(pred)
    return {
        "scene_recall": _recall_match(scene_pred, doc.get("scene_gt", "")),
    }


def illusionbench_arshia_aggregate(results: List[Optional[int]]) -> float:
    vals = [v for v in results if v is not None]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def illusionbench_arshia_doc_to_target(doc):
    # Metrics are recall-only; target text is unused.
    return ""


def illusionbench_arshia_doc_to_text_visual_cot_icon(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-ICON subset (abstract category icons)"""
    generation_prompt = (
        "This image shows a scene where elements are carefully arranged to form a hidden abstract icon or category shape. "
        "The hidden shape represents a category like: animal, vehicle, sports equipment, musical instrument, face emoji, or stationary item. "
        "Your task: Extract and visualize this hidden icon/category shape. "
        "Generate a clear image that highlights the icon's outline, contours, and structure. "
        "Make the hidden icon prominent and easily recognizable as a specific category."
    )

    # Stage 2: Use fixed candidate list, emphasize using both images
    shape_options = ", ".join(SHAPE_CANDIDATES_ICON)
    scene_options = ", ".join(SCENE_CANDIDATES)
    question_prompt = (
        "You are given TWO images in sequence:\n"
        "1) ORIGINAL IMAGE - shows the complete scene where elements form a hidden icon\n"
        "2) AUXILIARY IMAGE - extracts and highlights only the hidden icon outline\n\n"
        "Your task:\n"
        "1) Identify the hidden ICON/SHAPE by looking at the auxiliary visualization\n"
        f"   Shape options: [{shape_options}]\n"
        "2) Identify the SCENE by looking at the ORIGINAL IMAGE (the first image shown)\n"
        f"   Scene options: [{scene_options}]\n\n"
        "IMPORTANT: The auxiliary image only shows the shape outline. "
        "You MUST look at the ORIGINAL image to identify the scene.\n\n"
        "Reply in exactly TWO lines:\n"
        "Shape: <select one from shape options>\n"
        "Scene: <select one from scene options>\n"
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


def illusionbench_arshia_doc_to_text_visual_cot_logo(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-LOGO subset (brand logos)"""
    generation_prompt = (
        "This image shows a scene where elements are carefully arranged to form a hidden brand logo or trademark. "
        "The hidden shape represents a well-known brand logo like: Tesla, Nike, Starbucks, McDonald's, Apple, Google, BMW, etc. "
        "Your task: Extract and visualize this hidden brand logo. "
        "Generate a clear image that highlights the logo's distinctive outline, contours, and iconic design elements. "
        "Make the hidden brand logo prominent and easily recognizable."
    )

    # Stage 2: Use fixed candidate list, emphasize using both images
    shape_options = ", ".join(SHAPE_CANDIDATES_LOGO)
    scene_options = ", ".join(SCENE_CANDIDATES)
    question_prompt = (
        "You are given TWO images in sequence:\n"
        "1) ORIGINAL IMAGE - shows the complete scene where elements form a hidden brand logo\n"
        "2) AUXILIARY IMAGE - extracts and highlights only the hidden logo outline\n\n"
        "Your task:\n"
        "1) Identify the hidden BRAND LOGO by looking at the auxiliary visualization\n"
        f"   Shape options: [{shape_options}]\n"
        "2) Identify the SCENE by looking at the ORIGINAL IMAGE (the first image shown)\n"
        f"   Scene options: [{scene_options}]\n\n"
        "IMPORTANT: The auxiliary image only shows the logo outline. "
        "You MUST look at the ORIGINAL image to identify the scene.\n\n"
        "Reply in exactly TWO lines:\n"
        "Shape: <select one from shape options>\n"
        "Scene: <select one from scene options>\n"
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


def illusionbench_arshia_doc_to_text_visual_cot_in(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-IN subset (ImageNet objects)"""
    generation_prompt = (
        "This image shows a scene where elements are carefully arranged to form a hidden object or thing. "
        "The hidden shape represents a concrete object like: guitar, cat, bird, bicycle, bottle, airplane, dog, teapot, etc. "
        "Your task: Extract and visualize this hidden object. "
        "Generate a clear image that highlights the object's outline, contours, and recognizable features. "
        "Make the hidden object prominent and easily identifiable."
    )

    # Stage 2: Use fixed candidate list, emphasize using both images
    shape_options = ", ".join(SHAPE_CANDIDATES_IN)
    scene_options = ", ".join(SCENE_CANDIDATES)
    question_prompt = (
        "You are given TWO images in sequence:\n"
        "1) ORIGINAL IMAGE - shows the complete scene where elements form a hidden object\n"
        "2) AUXILIARY IMAGE - extracts and highlights only the hidden object outline\n\n"
        "Your task:\n"
        "1) Identify the hidden OBJECT by looking at the auxiliary visualization\n"
        f"   Shape options: [{shape_options}]\n"
        "2) Identify the SCENE by looking at the ORIGINAL IMAGE (the first image shown)\n"
        f"   Scene options: [{scene_options}]\n\n"
        "IMPORTANT: The auxiliary image only shows the object outline. "
        "You MUST look at the ORIGINAL image to identify the scene.\n\n"
        "Reply in exactly TWO lines:\n"
        "Shape: <select one from shape options>\n"
        "Scene: <select one from scene options>\n"
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


# ============ Split shape/scene visual_cot prompts (for separate tasks) ============

def illusionbench_arshia_doc_to_text_visual_cot_icon_shape(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-ICON subset - SHAPE task only"""
    generation_prompt = (
        "This image shows a scene where elements are carefully arranged to form a hidden abstract icon or category shape. "
        "The hidden shape represents a category like: animal, vehicle, sports equipment, musical instrument, face emoji, or stationary item. "
        "Your task: Extract and visualize this hidden icon/category shape. "
        "Generate a clear image that highlights the icon's outline, contours, and structure. "
        "Make the hidden icon prominent and easily recognizable as a specific category."
    )

    shape_options = ", ".join(SHAPE_CANDIDATES_ICON)
    question_prompt = (
        "You are given TWO images in sequence:\n"
        "1) ORIGINAL IMAGE - shows the complete scene where elements form a hidden icon\n"
        "2) AUXILIARY IMAGE - extracts and highlights only the hidden icon outline\n\n"
        "Your task: Identify the hidden ICON/SHAPE by looking at BOTH images.\n"
        f"Shape options: [{shape_options}]\n\n"
        "Reply in ONE line using this format:\n"
        "Shape: <select one from shape options>\n"
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


def illusionbench_arshia_doc_to_text_visual_cot_icon_scene(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-ICON subset - SCENE task only"""
    generation_prompt = (
        "This image depicts a specific scene or environment. "
        "Your task: Analyze and enhance the scene characteristics. "
        "Generate a clear visualization that emphasizes the environmental features, atmospheric elements, "
        "architectural details, landscape patterns, and overall setting. "
        "Make the scene type clearly recognizable by highlighting key environmental indicators."
    )

    scene_options = ", ".join(SCENE_CANDIDATES)
    question_prompt = (
        "You are given TWO images in sequence:\n"
        "1) ORIGINAL IMAGE - shows the scene/environment\n"
        "2) AUXILIARY IMAGE - emphasizes and enhances the scene characteristics\n\n"
        "Your task: Identify the SCENE by analyzing BOTH images.\n"
        "The auxiliary image helps you recognize environmental features.\n"
        f"Scene options: [{scene_options}]\n\n"
        "Reply in ONE line using this format:\n"
        "Scene: <select one from scene options>\n"
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


def illusionbench_arshia_doc_to_text_visual_cot_logo_shape(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-LOGO subset - SHAPE task only"""
    generation_prompt = (
        "This image shows a scene where elements are carefully arranged to form a hidden brand logo or trademark. "
        "The hidden shape represents a well-known brand logo like: Tesla, Nike, Starbucks, McDonald's, Apple, Google, BMW, etc. "
        "Your task: Extract and visualize this hidden brand logo. "
        "Generate a clear image that highlights the logo's distinctive outline, contours, and iconic design elements. "
        "Make the hidden brand logo prominent and easily recognizable."
    )

    shape_options = ", ".join(SHAPE_CANDIDATES_LOGO)
    question_prompt = (
        "You are given TWO images in sequence:\n"
        "1) ORIGINAL IMAGE - shows the complete scene where elements form a hidden brand logo\n"
        "2) AUXILIARY IMAGE - extracts and highlights only the hidden logo outline\n\n"
        "Your task: Identify the hidden BRAND LOGO by looking at BOTH images.\n"
        f"Shape options: [{shape_options}]\n\n"
        "Reply in ONE line using this format:\n"
        "Shape: <select one from shape options>\n"
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


def illusionbench_arshia_doc_to_text_visual_cot_logo_scene(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-LOGO subset - SCENE task only"""
    generation_prompt = (
        "This image depicts a specific scene or environment. "
        "Your task: Analyze and enhance the scene characteristics. "
        "Generate a clear visualization that emphasizes the environmental features, atmospheric elements, "
        "architectural details, landscape patterns, and overall setting. "
        "Make the scene type clearly recognizable by highlighting key environmental indicators."
    )

    scene_options = ", ".join(SCENE_CANDIDATES)
    question_prompt = (
        "You are given TWO images in sequence:\n"
        "1) ORIGINAL IMAGE - shows the scene/environment\n"
        "2) AUXILIARY IMAGE - emphasizes and enhances the scene characteristics\n\n"
        "Your task: Identify the SCENE by analyzing BOTH images.\n"
        "The auxiliary image helps you recognize environmental features.\n"
        f"Scene options: [{scene_options}]\n\n"
        "Reply in ONE line using this format:\n"
        "Scene: <select one from scene options>\n"
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


def illusionbench_arshia_doc_to_text_visual_cot_in_shape(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-IN subset - SHAPE task only"""
    generation_prompt = (
        "This image shows a scene where elements are carefully arranged to form a hidden object or thing. "
        "The hidden shape represents a concrete object like: guitar, cat, bird, bicycle, bottle, airplane, dog, teapot, etc. "
        "Your task: Extract and visualize this hidden object. "
        "Generate a clear image that highlights the object's outline, contours, and recognizable features. "
        "Make the hidden object prominent and easily identifiable."
    )

    shape_options = ", ".join(SHAPE_CANDIDATES_IN)
    question_prompt = (
        "You are given TWO images in sequence:\n"
        "1) ORIGINAL IMAGE - shows the complete scene where elements form a hidden object\n"
        "2) AUXILIARY IMAGE - extracts and highlights only the hidden object outline\n\n"
        "Your task: Identify the hidden OBJECT by looking at BOTH images.\n"
        f"Shape options: [{shape_options}]\n\n"
        "Reply in ONE line using this format:\n"
        "Shape: <select one from shape options>\n"
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


def illusionbench_arshia_doc_to_text_visual_cot_in_scene(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for Illusion-IN subset - SCENE task only"""
    generation_prompt = (
        "This image depicts a specific scene or environment. "
        "Your task: Analyze and enhance the scene characteristics. "
        "Generate a clear visualization that emphasizes the environmental features, atmospheric elements, "
        "architectural details, landscape patterns, and overall setting. "
        "Make the scene type clearly recognizable by highlighting key environmental indicators."
    )

    scene_options = ", ".join(SCENE_CANDIDATES)
    question_prompt = (
        "You are given TWO images in sequence:\n"
        "1) ORIGINAL IMAGE - shows the scene/environment\n"
        "2) AUXILIARY IMAGE - emphasizes and enhances the scene characteristics\n\n"
        "Your task: Identify the SCENE by analyzing BOTH images.\n"
        "The auxiliary image helps you recognize environmental features.\n"
        f"Scene options: [{scene_options}]\n\n"
        "Reply in ONE line using this format:\n"
        "Scene: <select one from scene options>\n"
    )
    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT][QUESTION]{question_prompt}[/QUESTION]"


