import ast
import json
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image

JIGSAW_PROMPT = """You are given:
(1) a 2x2 reference image with the bottom-right cell hidden
(2) two candidate patches ("Candidate 0" and "Candidate 1")

Compare which candidate correctly completes the puzzle based on seam continuity,
color/texture gradient, structural alignment, and global semantics.

Output your analysis followed by your decision in this exact format:
<FINAL_ANSWER_JSON>
{"choice": 0 or 1, "rationale": "brief explanation"}
</FINAL_ANSWER_JSON>"""


MAZE_PROMPT = """You are a maze solver.

Semantics:
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start position
- Green frame: goal area

Find the path from start to goal. Legal moves: up, down, left, right only.

Output the move sequence as:
<ANSWER_JSON>["right","down","left"]</ANSWER_JSON>"""


SLIDING_PROMPT = """You are a sliding puzzle solver.

The puzzle is a 3x3 grid with 8 colored tiles and one empty space (red).
A move slides an adjacent tile into the empty space.
Moves are named by the direction the tile moves (not the empty space).

Find the solution sequence. Legal moves: up, down, left, right.

Output the move sequence as:
<ANSWER_JSON>["down","right","up"]</ANSWER_JSON>"""


GEOMETRY_PROMPT = """You are given a geometry problem with a diagram.

Analyze the problem and provide:
1. What auxiliary lines need to be drawn (if any)
2. A step-by-step solution with reasoning
3. The final answer

Be rigorous and show all calculations."""


def _extract_image_bytes(img_data: Any) -> Optional[Image.Image]:
    if img_data is None:
        return None
    if isinstance(img_data, Image.Image):
        return img_data.convert("RGB")
    if isinstance(img_data, dict) and "bytes" in img_data:
        return Image.open(BytesIO(img_data["bytes"])).convert("RGB")
    if isinstance(img_data, bytes):
        return Image.open(BytesIO(img_data)).convert("RGB")
    return None


def _find_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None

    depth, in_str, escaped = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if c == '"' and not escaped:
            in_str = not in_str
        if not in_str:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        escaped = c == "\\" and not escaped
    return None


def _parse_json_list(raw: str) -> List[Any]:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass
    return []


def _find_last_json_list(text: str) -> Optional[str]:
    matches = list(re.finditer(r"\[.*?\]", text, re.DOTALL))
    for match in reversed(matches):
        candidate = match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return None


def _normalize_geometry_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[°\$\\]", "", text)
    text = re.sub(r"\b(cm|mm|m|degrees?|units?)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_final_answer(text: str) -> str:
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+([^\n.]+)",
        r"(?:answer|solution)[:\s]+([^\n.]+)$",
        r"=\s*([^\n=]+)$",
        r"\\boxed\{([^}]+)\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    return lines[-1] if lines else ""


def jigsaw_doc_to_visual(doc: Dict) -> List[Image.Image]:
    images = []
    for key in ["ref_image", "cand0_image", "cand1_image"]:
        if key in doc and doc[key]:
            img = _extract_image_bytes(doc[key])
            if img:
                images.append(img)
    return images


def jigsaw_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    return JIGSAW_PROMPT


def jigsaw_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    result_text = results[0] if results else ""

    if isinstance(result_text, str):
        try:
            parsed = json.loads(result_text)
            if isinstance(parsed, dict) and "text" in parsed:
                result_text = parsed["text"]
        except (json.JSONDecodeError, TypeError):
            pass

    choice = None
    match = re.search(
        r"<FINAL_ANSWER_JSON>\s*(\{.*?\})\s*</FINAL_ANSWER_JSON>",
        result_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"<FINAL_ANSWER\s+JSON>\s*(\{.*?\})\s*</FINAL_ANSWER\s+JSON>",
            result_text,
            re.DOTALL | re.IGNORECASE,
        )

    if match:
        json_str = _find_json_object(match.group(1))
        if json_str:
            try:
                data = json.loads(json_str)
                choice = data.get("choice")
            except json.JSONDecodeError:
                pass

    if choice is None:
        choice_match = re.search(r'"choice"\s*:\s*(\d)', result_text)
        if choice_match:
            choice = int(choice_match.group(1))

    gt_label = doc.get("label", -1)
    if isinstance(gt_label, str):
        try:
            gt_label = int(gt_label)
        except ValueError:
            gt_label = -1

    if choice is not None and not isinstance(choice, int):
        try:
            choice = int(choice)
        except (ValueError, TypeError):
            choice = None

    correct = 1 if choice is not None and choice == gt_label else 0
    return {"exact_match": correct}


def maze_doc_to_visual(doc: Dict) -> List[Image.Image]:
    if "initial_image" in doc and doc["initial_image"]:
        img = _extract_image_bytes(doc["initial_image"])
        if img:
            return [img]
    return []


def maze_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    return MAZE_PROMPT


def maze_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    result_text = results[0] if results else ""

    if isinstance(result_text, str):
        try:
            parsed = json.loads(result_text)
            if isinstance(parsed, dict) and "text" in parsed:
                result_text = parsed["text"]
        except (json.JSONDecodeError, TypeError):
            pass

    pred_moves = []
    matches = list(
        re.finditer(
            r"<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>",
            result_text,
            re.DOTALL | re.IGNORECASE,
        )
    )
    if matches:
        moves_data = _parse_json_list(matches[-1].group(1))
        pred_moves = [str(m).strip().lower() for m in moves_data]
    else:
        fallback = _find_last_json_list(result_text)
        if fallback:
            moves_data = _parse_json_list(fallback)
            pred_moves = [str(m).strip().lower() for m in moves_data]

    gt_moves_str = doc.get("steps", "[]")
    gt_moves = _parse_json_list(gt_moves_str) if isinstance(gt_moves_str, str) else gt_moves_str
    gt_moves = [str(m).lower() for m in gt_moves]

    exact = 1 if pred_moves == gt_moves else 0
    frame_acc = sum(1 for p, g in zip(pred_moves, gt_moves) if p == g) / len(gt_moves) if gt_moves else 0.0

    return {"exact_match": exact, "frame_accuracy": frame_acc}


def sliding_doc_to_visual(doc: Dict) -> List[Image.Image]:
    if "initial_image" in doc and doc["initial_image"]:
        img = _extract_image_bytes(doc["initial_image"])
        if img:
            return [img]
    return []


def sliding_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    return SLIDING_PROMPT


def sliding_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    result_text = results[0] if results else ""

    if isinstance(result_text, str):
        try:
            parsed = json.loads(result_text)
            if isinstance(parsed, dict) and "text" in parsed:
                result_text = parsed["text"]
        except (json.JSONDecodeError, TypeError):
            pass

    pred_moves = []
    matches = list(
        re.finditer(
            r"<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>",
            result_text,
            re.DOTALL | re.IGNORECASE,
        )
    )
    if matches:
        moves_data = _parse_json_list(matches[-1].group(1))
        pred_moves = [str(m).strip().lower() for m in moves_data]
    else:
        fallback = _find_last_json_list(result_text)
        if fallback:
            moves_data = _parse_json_list(fallback)
            pred_moves = [str(m).strip().lower() for m in moves_data]

    gt_moves_str = doc.get("steps_words", "[]")
    gt_moves = _parse_json_list(gt_moves_str) if isinstance(gt_moves_str, str) else gt_moves_str
    gt_moves = [str(m).lower() for m in gt_moves]

    exact = 1 if pred_moves == gt_moves else 0
    frame_acc = sum(1 for p, g in zip(pred_moves, gt_moves) if p == g) / len(gt_moves) if gt_moves else 0.0

    return {"exact_match": exact, "frame_accuracy": frame_acc}


def geometry_doc_to_visual(doc: Dict) -> List[Image.Image]:
    if "image" in doc and doc["image"]:
        img = _extract_image_bytes(doc["image"])
        if img:
            return [img]
    return []


def geometry_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    question = doc.get("question", doc.get("problem", ""))
    return f"{GEOMETRY_PROMPT}\n\nProblem: {question}"


def geometry_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    result_text = results[0] if results else ""

    if isinstance(result_text, str):
        try:
            parsed = json.loads(result_text)
            if isinstance(parsed, dict) and "text" in parsed:
                result_text = parsed["text"]
        except (json.JSONDecodeError, TypeError):
            pass

    gt_answer = str(doc.get("answer", doc.get("solution_en", ""))).strip()
    gt_normalized = _normalize_geometry_answer(gt_answer)

    pred_answer = _extract_final_answer(result_text)
    pred_normalized = _normalize_geometry_answer(pred_answer)

    correct = 1 if gt_normalized and gt_normalized == pred_normalized else 0
    return {"exact_match": correct}
