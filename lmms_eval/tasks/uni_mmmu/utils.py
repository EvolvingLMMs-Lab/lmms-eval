"""
Uni-MMMU Task Utilities
Complete evaluation logic with image parsing, DreamSim, and VLM judge support.
Migrated from eval_ummmu.py.
"""

import base64
import colorsys
import glob as glob_module
import json
import mimetypes
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Optional imports for advanced evaluation
try:
    from dreamsim import dreamsim as dreamsim_fn
    HAS_DREAMSIM = True
except ImportError:
    dreamsim_fn = None
    HAS_DREAMSIM = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# Common Utilities
# ============================================================================

def read_json(path: Union[str, Path]) -> Any:
    """Safely read JSON file"""
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def image_to_base64(image_path: Union[str, Path]) -> str:
    """Convert image to base64 data URL"""
    path = Path(image_path)
    if not path.is_file():
        return ""

    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        ext = path.suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"
        }.get(ext, "image/png")

    try:
        with Image.open(path) as img:
            # Convert to RGB if needed
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                if mime_type == "image/jpeg":
                    img = img.convert("RGB")

            buffer = BytesIO()
            save_format = "PNG" if mime_type == "image/png" else "JPEG"
            img.save(buffer, format=save_format)
            image_bytes = buffer.getvalue()
    except Exception:
        image_bytes = path.read_bytes()

    b64_string = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64_string}"


def find_first_json_substring(text: str) -> Optional[str]:
    """Extract first JSON object from text"""
    if not text:
        return None
    start_index = text.find("{")
    if start_index == -1:
        return None

    brace_depth, in_string, is_escaped = 0, False, False
    for i in range(start_index, len(text)):
        char = text[i]
        if char == '"' and not is_escaped:
            in_string = not in_string
        if not in_string:
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    return text[start_index : i + 1]
        is_escaped = char == "\\" and not is_escaped
    return None


# ============================================================================
# Jigsaw Task Functions
# ============================================================================

def jigsaw_doc_to_visual(doc: Dict) -> List[str]:
    """
    Get visual inputs for jigsaw task.
    Returns paths to: [reference_2x2, candidate_0, candidate_1]
    """
    base_path = Path("G:/Uni-MMMU/data/jigsaw_dataset_2x2ref")

    # Get image paths from metadata
    ref_path = doc.get("ref_2x2_path", "")
    cand0_path = doc.get("candidate_0_path", "")
    cand1_path = doc.get("candidate_1_path", "")

    return [
        str(base_path / ref_path) if ref_path else "",
        str(base_path / cand0_path) if cand0_path else "",
        str(base_path / cand1_path) if cand1_path else "",
    ]


def jigsaw_doc_to_text(doc: Dict) -> str:
    """Get text prompt for jigsaw task"""
    # Prompt is defined in YAML, this just returns context if needed
    return "Analyze the reference and candidates to complete the puzzle."


def jigsaw_doc_to_messages(doc: Dict) -> List[Dict]:
    """
    Build multimodal messages for jigsaw task.
    Format compatible with vision-language models.
    """
    images = jigsaw_doc_to_visual(doc)

    # Get prompt template from config
    prompt = doc.get("prompt_template", jigsaw_doc_to_text(doc))

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": images[0]},  # Reference 2x2
                {"type": "image", "image": images[1]},  # Candidate 0
                {"type": "image", "image": images[2]},  # Candidate 1
            ],
        }
    ]


def jigsaw_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """
    Process jigsaw task results with complete evaluation logic.

    Evaluates:
    1. Text accuracy: Parse choice from JSON and compare with GT
    2. Image quality: DreamSim distance to ground truth completions
    """
    result_text = results[0] if results else ""

    # Parse choice from <FINAL_ANSWER_JSON>
    choice = None
    match = re.search(
        r"<FINAL_ANSWER_JSON>\s*(\{.*?\})\s*</FINAL_ANSWER_JSON>",
        result_text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        json_str = match.group(1)
        parsed_json = find_first_json_substring(json_str)
        if parsed_json:
            data = json.loads(parsed_json)
            choice = data.get("choice")

    # Text accuracy
    gt_label = doc.get("label", -1)
    text_correct = 1 if choice == gt_label else 0

    # Image evaluation with DreamSim
    image_score = 0.0

    doc_id = doc.get("id", "unknown")
    output_dir = Path("./eval_results/images")
    pred_imgs = sorted(glob_module.glob(str(output_dir / f"*{doc_id}*.png")))

    if len(pred_imgs) >= 2 and HAS_DREAMSIM:
        base_path = Path("G:/Uni-MMMU/data/jigsaw_dataset_2x2ref")
        label = doc.get("label", 0)
        gt_ok = doc.get("gt_completed_2x2_path", "")
        gt_bad = doc.get("gt_wrong_2x2_path", "")

        gt_c0 = str(base_path / (gt_ok if label == 0 else gt_bad))
        gt_c1 = str(base_path / (gt_bad if label == 0 else gt_ok))

        if Path(gt_c0).exists() and Path(gt_c1).exists():
            im0 = Image.open(pred_imgs[0])
            im1 = Image.open(pred_imgs[1])
            gt0 = Image.open(gt_c0)
            gt1 = Image.open(gt_c1)

            d0 = compute_dreamsim_distance(im0, gt0)
            d1 = compute_dreamsim_distance(im1, gt1)
            image_score = 1.0 - (d0 + d1) / 2.0

    return {
        "jigsaw_text_acc": text_correct,
        "jigsaw_image_score": image_score,
    }


# ============================================================================
# Maze Task Functions
# ============================================================================

def maze_doc_to_visual(doc: Dict) -> List[str]:
    """Get visual input for maze task (initial maze image)"""
    base_path = Path("G:/Uni-MMMU/data/maze_dataset")
    step0_path = doc.get("step0", "")

    return [str(base_path / step0_path)] if step0_path else []


def maze_doc_to_text(doc: Dict) -> str:
    """Get text prompt for maze task"""
    return "Find the shortest path from S (start) to G (goal) in this maze."


def maze_doc_to_messages(doc: Dict) -> List[Dict]:
    """Build multimodal messages for maze task"""
    images = maze_doc_to_visual(doc)
    prompt = doc.get("prompt_template", maze_doc_to_text(doc))

    content = [{"type": "text", "text": prompt}]
    if images:
        content.append({"type": "image", "image": images[0]})

    return [{"role": "user", "content": content}]


def maze_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """
    Process maze task results with complete evaluation logic.

    Evaluates:
    1. Text moves: Parse and compare with ground truth
    2. Image sequence: Parse generated maze images and validate states
    """
    result_text = results[0] if results else ""

    # Parse moves from last <ANSWER_JSON> block
    pred_moves = []
    matches = list(re.finditer(
        r"<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>",
        result_text,
        re.DOTALL | re.IGNORECASE
    ))
    if matches:
        last_match = matches[-1].group(1)
        moves_data = json.loads(last_match)
        pred_moves = [str(m).strip().lower() for m in moves_data]

    # Ground truth moves
    gt_moves = [str(m).lower() for m in doc.get("steps_long", [])]

    # Text evaluation
    text_exact = 1 if pred_moves == gt_moves else 0
    text_frame_acc = (
        sum(1 for p, g in zip(pred_moves, gt_moves) if p == g) / len(gt_moves)
        if gt_moves else 0.0
    )

    # Image evaluation with parsing
    img_exact, img_frame_acc = 0, 0.0

    doc_id = doc.get("id", "unknown")
    output_dir = Path("./eval_results/images")
    img_dirs = sorted(output_dir.glob(f"*{doc_id}_cand_*"))

    if img_dirs:
        img_files = sorted(img_dirs[0].glob("*.png"))
        parser_params = {"grid_h": 6, "grid_w": 6, "tolerance": 0.75}

        parsed_grids = []
        for img_file in img_files:
            parsed = parse_maze_image(str(img_file), **parser_params)
            parsed_grids.append(parsed.get("grid"))

        if parsed_grids:
            parse_ok = sum(
                1 for g in parsed_grids
                if g and len(g) == parser_params["grid_h"]
            )
            img_frame_acc = parse_ok / len(parsed_grids)
            img_exact = 1 if parse_ok == len(parsed_grids) else 0

    return {
        "maze_text_exact": text_exact,
        "maze_text_frame_acc": text_frame_acc,
        "maze_img_exact": img_exact,
        "maze_img_frame_acc": img_frame_acc,
    }


# ============================================================================
# Sliding Puzzle Task Functions
# ============================================================================

def sliding_doc_to_visual(doc: Dict) -> List[str]:
    """Get visual input for sliding puzzle task"""
    base_path = Path("G:/Uni-MMMU/data/sliding_puzzle_dataset")
    init_path = doc.get("init_png", "")

    return [str(base_path / init_path)] if init_path else []


def sliding_doc_to_text(doc: Dict) -> str:
    """Get text prompt for sliding puzzle task"""
    return "Solve this 3×3 sliding puzzle by moving tiles into the empty space."


def sliding_doc_to_messages(doc: Dict) -> List[Dict]:
    """Build multimodal messages for sliding puzzle task"""
    images = sliding_doc_to_visual(doc)
    prompt = doc.get("prompt_template", sliding_doc_to_text(doc))

    content = [{"type": "text", "text": prompt}]
    if images:
        content.append({"type": "image", "image": images[0]})

    return [{"role": "user", "content": content}]


def sliding_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """Process sliding puzzle results with complete evaluation logic"""
    result_text = results[0] if results else ""

    # Parse moves from last <ANSWER_JSON>
    pred_moves = []
    matches = list(re.finditer(
        r"<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>",
        result_text,
        re.DOTALL | re.IGNORECASE
    ))
    if matches:
        moves_data = json.loads(matches[-1].group(1))
        pred_moves = [str(m).strip() for m in moves_data]

    gt_moves = doc.get("steps_words", [])

    # Text evaluation
    text_exact = 1 if pred_moves == gt_moves else 0
    text_frame_acc = (
        sum(1 for p, g in zip(pred_moves, gt_moves) if p == g) / len(gt_moves)
        if gt_moves else 0.0
    )

    # Image evaluation with parsing
    img_exact, img_frame_acc = 0, 0.0

    doc_id = doc.get("id", "unknown")
    output_dir = Path("./eval_results/images")
    img_dirs = sorted(output_dir.glob(f"*{doc_id}_cand_*"))

    if img_dirs:
        img_files = sorted(img_dirs[0].glob("*.png"))
        parser_params = {
            "grid_h": 3,
            "grid_w": 3,
            "num_categories": 9,
            "tolerance": 0.80
        }

        parsed_ascii = []
        for img_file in img_files:
            parsed = parse_sliding_puzzle_image(str(img_file), **parser_params)
            parsed_ascii.append(parsed.get("ascii"))

        if parsed_ascii:
            parse_ok = sum(1 for a in parsed_ascii if a and "?" not in a)
            img_frame_acc = parse_ok / len(parsed_ascii)
            img_exact = 1 if parse_ok == len(parsed_ascii) else 0

    return {
        "sliding_text_exact": text_exact,
        "sliding_text_frame_acc": text_frame_acc,
        "sliding_img_exact": img_exact,
        "sliding_img_frame_acc": img_frame_acc,
    }


# ============================================================================
# Geometry Task Functions
# ============================================================================

def geometry_doc_to_visual(doc: Dict) -> List[str]:
    """Get visual input for geometry task (original diagram)"""
    base_path = Path("G:/Uni-MMMU/data/math_data")

    # Extract original image path from nested structure
    original_image = doc.get("original_image", "")

    return [str(base_path / original_image)] if original_image else []


def geometry_doc_to_text(doc: Dict) -> str:
    """Get text prompt for geometry task"""
    problem = doc.get("problem_text_en", doc.get("problem_text", ""))
    return f"Problem: {problem}\n\nDraw the necessary auxiliary lines and solve."


def geometry_doc_to_messages(doc: Dict) -> List[Dict]:
    """Build multimodal messages for geometry task"""
    images = geometry_doc_to_visual(doc)
    prompt = doc.get("prompt_template", geometry_doc_to_text(doc))

    content = [{"type": "text", "text": prompt}]
    if images:
        content.append({"type": "image", "image": images[0]})

    return [{"role": "user", "content": content}]


def geometry_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """
    Process geometry results with VLM Judge evaluation.

    Evaluates:
    1. Overlay accuracy: VLM Judge compares generated overlay with GT
    2. Text accuracy: LLM Judge evaluates reasoning and conclusion
    """
    import os

    result_text = results[0] if results else ""

    # Default scores
    overlay_acc = 0.0
    text_acc = 0.0

    # Check if llm_judge is available
    if not HAS_TORCH:
        return {
            "geometry_overlay_acc": overlay_acc,
            "geometry_text_acc": text_acc,
        }

    # Import judge only when needed
    from lmms_eval.llm_judge import ServerConfig, get_server, Request

    # Initialize judges
    overlay_judge_config = ServerConfig(
        model_name=os.getenv("OVERLAY_JUDGE_MODEL", "gpt-4o"),
        temperature=0.0,
        max_tokens=512,
        timeout=60,
        num_retries=2
    )
    overlay_judge = get_server(
        server_name=os.getenv("OVERLAY_JUDGE_API", "openai"),
        config=overlay_judge_config
    )

    text_judge_config = ServerConfig(
        model_name=os.getenv("TEXT_JUDGE_MODEL", "gpt-4o"),
        temperature=0.0,
        max_tokens=1024,
        timeout=60,
        num_retries=2
    )
    text_judge = get_server(
        server_name=os.getenv("TEXT_JUDGE_API", "openai"),
        config=text_judge_config
    )

    # === Overlay Evaluation ===
    doc_id = doc.get("id", doc.get("problem_index", "unknown"))
    output_dir = Path("./eval_results/images")
    pred_imgs = sorted(output_dir.glob(f"*{doc_id}_overlay*.png"))

    if pred_imgs and len(pred_imgs) > 0:
        base_img_path = doc.get("original_image", "")
        gt_overlay_path = doc.get("auxiliary_image", "")

        if base_img_path and gt_overlay_path:
            base_path = Path("G:/Uni-MMMU/data/math_data")
            base_img = str(base_path / base_img_path)
            gt_overlay = str(base_path / gt_overlay_path)
            pred_overlay = str(pred_imgs[0])

            if Path(base_img).exists() and Path(gt_overlay).exists():
                # Prepare overlay judge prompt
                aux_en = doc.get("auxiliary_text_en", "")
                overlay_system = """You are a strict yet fair judge for geometry diagram overlays.
Your job: compare images to decide if the candidate overlay correctly draws the REQUIRED auxiliary lines.

Input images:
  (1) base: original (no overlays), the clean figure.
  (2) gt  : ground-truth overlay (what SHOULD be drawn).
  (3) pred: ONLY ONE candidate overlay to judge.

Also given the English text listing auxiliary lines to draw.

Rules:
  - Focus on whether the REQUIRED lines exist (correct endpoints/relations) in the candidate.
  - Ignore extra small ticks/labels/harmless marks if they do not distort the meaning.
  - Tolerate minor pixel/position noise if geometry intent is clear (parallel/perpendicular/through-point).
  - ALL specified lines must be present; otherwise mark 0.
  - Output MUST be a compact JSON: {"overlay_ok":0|1,"overlay_reason":"<short>"}
"""

                overlay_user = f"""Auxiliary lines to draw (English):
{aux_en}

Compare the pred overlay with the gt overlay. Does pred correctly draw all required lines?

Output JSON only."""

                # Build multimodal content
                overlay_content = [
                    {"type": "text", "text": overlay_user},
                    {"type": "image_url", "image_url": {"url": image_to_base64(base_img)}},
                    {"type": "image_url", "image_url": {"url": image_to_base64(gt_overlay)}},
                    {"type": "image_url", "image_url": {"url": image_to_base64(pred_overlay)}},
                ]

                # Call VLM Judge
                request = Request(
                    messages=[
                        {"role": "system", "content": overlay_system},
                        {"role": "user", "content": overlay_content},
                    ],
                    config=overlay_judge_config
                )

                response = overlay_judge.evaluate(request)
                if response.success:
                    # Parse JSON from response
                    result_json = find_first_json_substring(response.content)
                    if result_json:
                        data = json.loads(result_json)
                        overlay_acc = 1.0 if int(data.get("overlay_ok", 0)) == 1 else 0.0

    # === Text Evaluation ===
    problem_text = doc.get("problem_text_en", doc.get("problem_text", ""))
    gt_solution = doc.get("solution_en", doc.get("solution", ""))
    task_type = doc.get("type", "calculation")

    if problem_text and gt_solution and result_text:
        text_system = """You are a rigorous grader for geometry reasoning.
Given: problem statement (text), ground-truth solution text (reference), and a candidate solution text.

Decide two things:
  (i) is the reasoning rigorous (no major logical gaps or false claims),
  (ii) is the final conclusion correct.

For CALCULATION problems: conclusion correctness means the final NUMERIC result matches the ground-truth
(ignore formatting/units; radicals/π are okay if numerically equivalent), even if choices differ.

For PROVING problems: conclusion correctness means the claim is indeed established (may differ in steps but must be valid).

Only if (rigorous AND correct) → text_ok=1, else 0.

Output MUST be a compact JSON: {"reasoning_rigorous":0|1,"conclusion_correct":0|1,"text_ok":0|1,"text_reason":"<short>"}
"""

        text_user = f"""Task type: {task_type}

Problem:
{problem_text}

Ground truth solution:
{gt_solution}

Candidate solution:
{result_text}

Evaluate the candidate. Output JSON only."""

        request = Request(
            messages=[
                {"role": "system", "content": text_system},
                {"role": "user", "content": text_user},
            ],
            config=text_judge_config
        )

        response = text_judge.evaluate(request)
        if response.success:
            result_json = find_first_json_substring(response.content)
            if result_json:
                data = json.loads(result_json)
                text_acc = 1.0 if int(data.get("text_ok", 0)) == 1 else 0.0

    return {
        "geometry_overlay_acc": overlay_acc,
        "geometry_text_acc": text_acc,
    }


# ============================================================================
# Dataset Loading (for lmms-eval framework)
# ============================================================================

def load_jigsaw_dataset():
    """Load jigsaw dataset in HuggingFace format"""
    import datasets

    base_path = Path("G:/Uni-MMMU/data/jigsaw_dataset_2x2ref")
    metadata = read_json(base_path / "metadata.json")

    items = metadata.get("items", [])

    # Convert to HF dataset format
    data = {
        "id": [item["id"] for item in items],
        "ref_2x2_path": [item["ref_2x2_path"] for item in items],
        "candidate_0_path": [item["candidate_0_path"] for item in items],
        "candidate_1_path": [item["candidate_1_path"] for item in items],
        "label": [item["label"] for item in items],
    }

    return datasets.Dataset.from_dict(data)


# ============================================================================
# Image Parsing Functions (Migrated from eval_ummmu.py)
# ============================================================================

def parse_maze_image(img_path: str, **kwargs) -> Dict[str, Any]:
    """
    Parse maze image to extract grid structure, start, and goal positions.

    Args:
        img_path: Path to maze image
        grid_h: Grid height (default: 6)
        grid_w: Grid width (default: 6)
        tolerance: Color matching tolerance (default: 0.60)

    Returns:
        {"ascii": str, "grid": List[List[str]], "start": (row,col), "goal": (row,col)}
    """
    def _hex_to_rgb01(h: str) -> np.ndarray:
        h = h.lstrip("#")
        return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0

    PALETTE = {
        "floor": _hex_to_rgb01("#f4efe6"),
        "wall": _hex_to_rgb01("#1f2937"),
        "start": _hex_to_rgb01("#2563eb"),
        "goal": _hex_to_rgb01("#22c55e"),
        "white": _hex_to_rgb01("#ffffff"),
        "path": _hex_to_rgb01("#ef4444"),
    }

    def _rgb_img_to01(img: Image.Image) -> np.ndarray:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.asarray(img).astype(np.float32) / 255.0

    def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a - b, axis=-1)

    def _closest_label(rgb01: np.ndarray, labels=("floor", "wall", "start", "goal")):
        ds = np.stack([_dist(rgb01, PALETTE[lab]) for lab in labels], axis=-1)
        return np.argmin(ds, axis=-1), ds.min(axis=-1)

    def _find_board_square(rgb01, content_labels=("wall", "start", "goal", "path"),
                          whiteish_labels=("white", "floor"), row_frac_thresh=0.003, margin_pixels=2):
        H, W, _ = rgb01.shape
        d_content = np.stack([_dist(rgb01, PALETTE[k]) for k in content_labels], axis=-1).min(axis=-1)
        d_white = np.stack([_dist(rgb01, PALETTE[k]) for k in whiteish_labels], axis=-1).min(axis=-1)
        content_mask = (d_content + 0.01) < d_white

        def fr():
            for i in range(H):
                if content_mask[i].mean() > row_frac_thresh:
                    return i
            return 0

        def lr():
            for i in range(H-1, -1, -1):
                if content_mask[i].mean() > row_frac_thresh:
                    return i
            return H-1

        def fc():
            for j in range(W):
                if content_mask[:,j].mean() > row_frac_thresh:
                    return j
            return 0

        def lc():
            for j in range(W-1, -1, -1):
                if content_mask[:,j].mean() > row_frac_thresh:
                    return j
            return W-1

        top, bottom = fr(), lr()
        left, right = fc(), lc()
        top = max(0, top - margin_pixels)
        left = max(0, left - margin_pixels)
        bottom = min(H-1, bottom + margin_pixels)
        right = min(W-1, right + margin_pixels)

        h, w = bottom - top + 1, right - left + 1
        side = max(h, w)
        cy, cx = (top + bottom)//2, (left + right)//2
        half = side//2
        t = max(0, cy - half)
        b = min(H, t + side)
        t = b - side
        l = max(0, cx - half)
        r = min(W, l + side)
        l = r - side
        return t, l, b, r

    grid_h = kwargs.get("grid_h", 6)
    grid_w = kwargs.get("grid_w", 6)
    tolerance = kwargs.get("tolerance", 0.60)
    heuristic_min_start = kwargs.get("heuristic_min_start", 0.08)
    heuristic_min_goal = kwargs.get("heuristic_min_goal", 0.25)

    img = Image.open(img_path)
    rgb = _rgb_img_to01(img)
    t, l, b, r = _find_board_square(rgb)
    board = rgb[t:b, l:r, :]
    H, W = board.shape[:2]
    cell_h, cell_w = H/grid_h, W/grid_w

    label_names = ("floor", "wall", "start", "goal")
    L_FLOOR, L_WALL, L_START, L_GOAL = 0, 1, 2, 3
    s_thresh, g_thresh = tolerance * 0.25, tolerance * 0.75
    grid_labels = []
    start_pos = None
    goal_pos = None
    frac_map = np.zeros((grid_h, grid_w, len(label_names)), dtype=np.float32)

    for gr in range(grid_h):
        row_syms = []
        y0, y1 = int(round(gr * cell_h)), int(round((gr+1) * cell_h))
        for gc in range(grid_w):
            x0, x1 = int(round(gc * cell_w)), int(round((gc+1) * cell_w))
            tile = board[y0:y1, x0:x1, :]
            if tile.size == 0:
                row_syms.append("?")
                continue

            idx_map, _ = _closest_label(tile, labels=label_names)
            fracs = np.array([(idx_map == k).mean() for k in range(len(label_names))], dtype=np.float32)
            frac_map[gr, gc, :] = fracs
            f_floor, f_wall, f_start, f_goal = fracs[L_FLOOR], fracs[L_WALL], fracs[L_START], fracs[L_GOAL]

            if f_start >= s_thresh and f_goal >= g_thresh:
                start_pos, goal_pos, sym = (gr,gc), (gr,gc), "SG"
            elif f_start >= s_thresh:
                start_pos, sym = (gr,gc), "S"
            elif f_goal >= g_thresh:
                goal_pos, sym = (gr,gc), "G"
            else:
                sym = "#" if f_wall >= tolerance else " " if f_floor >= tolerance else "?"
            row_syms.append(sym)
        grid_labels.append(row_syms)

    if start_pos is None:
        best = np.unravel_index(np.argmax(frac_map[:,:,L_START]), (grid_h,grid_w))
        if frac_map[best[0],best[1],L_START] >= heuristic_min_start:
            start_pos = (int(best[0]), int(best[1]))
            grid_labels[start_pos[0]][start_pos[1]] = "S"

    if goal_pos is None:
        best = np.unravel_index(np.argmax(frac_map[:,:,L_GOAL]), (grid_h,grid_w))
        if frac_map[best[0],best[1],L_GOAL] >= heuristic_min_goal:
            goal_pos = (int(best[0]), int(best[1]))
            grid_labels[goal_pos[0]][goal_pos[1]] = "SG" if grid_labels[goal_pos[0]][goal_pos[1]]=="S" else "G"

    if start_pos and start_pos == goal_pos:
        r0, c0 = start_pos
        grid_labels[r0][c0] = "SG"

    ascii_rows = ["".join(row) for row in grid_labels]
    return {"ascii": "\n".join(ascii_rows), "grid": grid_labels, "start": start_pos, "goal": goal_pos}


def parse_sliding_puzzle_image(img_path: str, **kwargs) -> Dict[str, Any]:
    """
    Parse sliding puzzle image to extract tile positions.

    Args:
        img_path: Path to puzzle image
        grid_h: Grid height (default: 3)
        grid_w: Grid width (default: 3)
        num_categories: Number of tile colors (default: 9)
        tolerance: Color matching tolerance (default: 0.80)

    Returns:
        {"ascii": str, "grid": List[List[str]]}
    """
    grid_h = kwargs.get("grid_h", 3)
    grid_w = kwargs.get("grid_w", 3)
    num_categories = kwargs.get("num_categories", 9)
    tolerance = kwargs.get("tolerance", 0.80)

    def _hex_to_rgb01(h: str) -> np.ndarray:
        h = h.lstrip("#")
        return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0

    SET1_9 = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
              "#ffff33", "#a65628", "#f781bf", "#999999"]

    def _distinct_palette_k(k):
        if k <= 9:
            return [_hex_to_rgb01(SET1_9[i % 9]) for i in range(k)]
        return [np.array(colorsys.hsv_to_rgb(i/float(k), 0.85, 0.95), dtype=np.float32) for i in range(k)]

    def _rgb_img_to01(img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.asarray(img).astype(np.float32) / 255.0

    def _edist(a, b):
        return np.linalg.norm(a - b, axis=-1)

    def _find_square_board(rgb, content_colors, row_frac_thresh=0.002, margin_px=2):
        H, W, _ = rgb.shape
        d_white = _edist(rgb, np.array([1.0, 1.0, 1.0]))
        d_content = np.stack([_edist(rgb, c) for c in content_colors], axis=-1).min(axis=-1)
        content_mask = (d_content + 0.01) < d_white

        fr = lambda: next((i for i in range(H) if content_mask[i].mean() > row_frac_thresh), 0)
        lr = lambda: next((i for i in range(H-1,-1,-1) if content_mask[i].mean() > row_frac_thresh), H-1)
        fc = lambda: next((j for j in range(W) if content_mask[:,j].mean() > row_frac_thresh), 0)
        lc = lambda: next((j for j in range(W-1,-1,-1) if content_mask[:,j].mean() > row_frac_thresh), W-1)

        top, bottom, left, right = fr(), lr(), fc(), lc()
        top, left = max(0, top - margin_px), max(0, left - margin_px)
        bottom, right = min(H-1, bottom + margin_px), min(W-1, right + margin_px)

        h, w, side = bottom - top + 1, right - left + 1, max(bottom - top + 1, right - left + 1)
        cy, cx, half = (top + bottom)//2, (left + right)//2, side//2
        t, l = max(0, cy - half), max(0, cx - half)
        b, r = min(H, t + side), min(W, l + side)
        t, l = b - side, r - side
        return t, l, b, r

    img = Image.open(img_path)
    rgb = _rgb_img_to01(img)
    palette = _distinct_palette_k(num_categories)
    t, l, b, r = _find_square_board(rgb, content_colors=palette)
    board = rgb[t:b, l:r, :]

    H, W, _ = board.shape
    cell_h, cell_w = H/grid_h, W/grid_w
    labels_grid = []

    for gr in range(grid_h):
        y0, y1 = int(round(gr*cell_h)), int(round((gr+1)*cell_h))
        row_syms = []
        for gc in range(grid_w):
            x0, x1 = int(round(gc*cell_w)), int(round((gc+1)*cell_w))
            tile = board[y0:y1, x0:x1, :]
            if tile.size == 0:
                row_syms.append("?")
                continue

            dstack = np.stack([_edist(tile, c) for c in palette], axis=-1)
            argmin = np.argmin(dstack, axis=-1)
            counts = np.array([(argmin == k).mean() for k in range(len(palette))], dtype=np.float32)
            k_star = int(counts.argmax())

            row_syms.append(str(k_star+1) if float(counts[k_star]) >= tolerance else "?")
        labels_grid.append(row_syms)

    ascii_str = "\n".join([" ".join(row) for row in labels_grid])
    return {"ascii": ascii_str, "grid": labels_grid}


# ============================================================================
# DreamSim Integration for Jigsaw Image Evaluation
# ============================================================================

_DREAMSIM_MODEL = None
_DREAMSIM_PREPROCESS = None


def init_dreamsim(device="cuda", cache_dir=None):
    """Initialize DreamSim model (lazy loading)"""
    global _DREAMSIM_MODEL, _DREAMSIM_PREPROCESS

    if not HAS_DREAMSIM or not HAS_TORCH:
        return None, None

    if _DREAMSIM_MODEL is None and dreamsim_fn is not None:
        _DREAMSIM_MODEL, _DREAMSIM_PREPROCESS = dreamsim_fn(
            pretrained=True, device=device, cache_dir=cache_dir
        )

    return _DREAMSIM_MODEL, _DREAMSIM_PREPROCESS


def compute_dreamsim_distance(img_a: Image.Image, img_b: Image.Image) -> float:
    """
    Compute DreamSim perceptual distance between two images.
    Returns distance in [0, 1], where 0 = identical, 1 = completely different.
    """
    if not HAS_DREAMSIM or not HAS_TORCH:
        return 1.0

    model, preprocess = init_dreamsim()
    if model is None:
        return 1.0

    img_a_rgb = img_a.convert("RGB")
    img_b_rgb = img_b.convert("RGB")

    device = next(model.parameters()).device
    tensor_a = preprocess(img_a_rgb).to(device)
    tensor_b = preprocess(img_b_rgb).to(device)

    with torch.no_grad():
        dist = model(tensor_a, tensor_b)

    return float(dist.item())
