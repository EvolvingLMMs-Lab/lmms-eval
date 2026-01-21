"""
Uni-MMMU Task Utilities
Text-only evaluation using GPT-4o API via Azure TRAPI.
"""

import json
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image

# Azure OpenAI imports
from azure.identity import (
    ChainedTokenCredential,
    AzureCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI


# ============================================================================
# GPT-4o API Client (from api.py)
# ============================================================================

_CLIENT = None
_DEPLOYMENT = None


def get_gpt4o_client():
    """Get or create Azure OpenAI client for GPT-4o."""
    global _CLIENT, _DEPLOYMENT

    if _CLIENT is None:
        scope = os.getenv("TRAPI_SCOPE", "api://trapi/.default")
        api_version = os.getenv("TRAPI_API_VERSION", "2024-10-21")
        _DEPLOYMENT = os.getenv("TRAPI_DEPLOYMENT", "gpt-4o_2024-11-20")
        instance = os.getenv("TRAPI_INSTANCE", "gcr/shared")
        endpoint = f"https://trapi.research.microsoft.com/{instance}"

        chained = ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        )
        credential_provider = get_bearer_token_provider(chained, scope)

        _CLIENT = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=credential_provider,
            api_version=api_version,
        )

    return _CLIENT, _DEPLOYMENT


def call_gpt4o(prompt: str, max_tokens: int = 512) -> str:
    """Call GPT-4o API with a text prompt."""
    client, deployment = get_gpt4o_client()

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[GPT-4o Error] {e}")
        return ""


# ============================================================================
# Common Utilities
# ============================================================================

def find_first_json_substring(text: str) -> Optional[str]:
    """Extract first JSON object from text."""
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

def jigsaw_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual inputs for jigsaw task."""
    images = []
    for key in ["ref_image", "cand0_image", "cand1_image"]:
        if key in doc and doc[key]:
            img_bytes = doc[key]["bytes"] if isinstance(doc[key], dict) else doc[key]
            images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))
    return images


def jigsaw_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Get text prompt for jigsaw task - text-only evaluation."""
    prompt = """You are a unified vision-language model. You will be given:
(1) a 2×2 reference image with the bottom-right cell hidden, and
(2) two candidate patch images ("Candidate 0" and "Candidate 1").

Your job:
- Mentally visualize placing each candidate into the bottom-right cell.
- Compare which candidate yields the correct completion based on seam continuity, color/texture gradient, structural alignment, and global semantics.

Output EXACTLY the following:

1) Brief analysis comparing the two candidates

2) One strict JSON object with your decision, wrapped as:
<FINAL_ANSWER_JSON>
{"choice": 0 or 1, "rationale": "≤30 words decisive cue"}
</FINAL_ANSWER_JSON>

Hard constraints:
- Deterministic, single outcome. No hedging, no multiple possibilities.
- Do not restate the task as the answer; reason from visual evidence.

Inputs:"""
    return prompt


def jigsaw_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """Process jigsaw results - text evaluation only."""
    result_raw = results[0] if results else ""
    
    # Handle case where result is a JSON string (from bagel format_output)
    result_text = ""
    if isinstance(result_raw, str):
        # Try to parse as JSON first (bagel outputs JSON string)
        try:
            parsed_result = json.loads(result_raw)
            if isinstance(parsed_result, dict) and "text" in parsed_result:
                result_text = parsed_result["text"]
            else:
                result_text = result_raw
        except (json.JSONDecodeError, TypeError):
            # Not JSON, use as-is
            result_text = result_raw
    else:
        result_text = str(result_raw)

    # Parse choice from <FINAL_ANSWER_JSON> or <FINAL_ANSWER JSON> (flexible matching)
    choice = None
    
    # Try exact match first: <FINAL_ANSWER_JSON>
    match = re.search(
        r"<FINAL_ANSWER_JSON>\s*(\{.*?\})\s*</FINAL_ANSWER_JSON>",
        result_text,
        re.DOTALL | re.IGNORECASE,
    )
    
    # If not found, try with space: <FINAL_ANSWER JSON>
    if not match:
        match = re.search(
            r"<FINAL_ANSWER\s+JSON>\s*(\{.*?\})\s*</FINAL_ANSWER\s+JSON>",
            result_text,
            re.DOTALL | re.IGNORECASE,
        )
    
    if match:
        json_str = match.group(1)
        parsed = find_first_json_substring(json_str)
        if parsed:
            try:
                data = json.loads(parsed)
                choice = data.get("choice")
            except:
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
    
    text_correct = 1 if choice is not None and choice == gt_label else 0

    return {"jigsaw_text_acc": text_correct}


# ============================================================================
# Maze Task Functions
# ============================================================================

def maze_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual input for maze task."""
    if "initial_image" in doc and doc["initial_image"]:
        img_bytes = doc["initial_image"]["bytes"] if isinstance(doc["initial_image"], dict) else doc["initial_image"]
        return [Image.open(BytesIO(img_bytes)).convert("RGB")]
    return []


def maze_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Get text prompt for maze task - text-only evaluation."""
    prompt = """You are a precise maze solver.

SEMANTICS
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- Green rectangular frame: goal (reaching any white cell inside the green frame counts as success)
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.

OUTPUT FORMAT
1) Briefly describe your reasoning for the path.

2) Output the final move list as a JSON array of lowercase strings, wrapped as:
<ANSWER_JSON>["right","down","left"]</ANSWER_JSON>

NO EXTRAS
- No extra explanations beyond the brief reasoning and the JSON answer."""
    return prompt


def maze_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """Process maze results - text evaluation only using GPT-4o."""
    result_raw = results[0] if results else ""
    
    # Handle case where result is a JSON string (from bagel format_output)
    result_text = ""
    if isinstance(result_raw, str):
        # Try to parse as JSON first (bagel outputs JSON string)
        try:
            parsed_result = json.loads(result_raw)
            if isinstance(parsed_result, dict) and "text" in parsed_result:
                result_text = parsed_result["text"]
            else:
                result_text = result_raw
        except (json.JSONDecodeError, TypeError):
            # Not JSON, use as-is
            result_text = result_raw
    else:
        result_text = str(result_raw)

    # Parse predicted moves from <ANSWER_JSON>
    pred_moves = []
    matches = list(re.finditer(
        r"<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>",
        result_text,
        re.DOTALL | re.IGNORECASE
    ))
    if matches:
        try:
            moves_data = json.loads(matches[-1].group(1))
            pred_moves = [str(m).strip().lower() for m in moves_data]
        except:
            pass

    # Ground truth moves
    gt_moves_str = doc.get("steps", "[]")
    gt_moves = json.loads(gt_moves_str) if isinstance(gt_moves_str, str) else gt_moves_str
    gt_moves = [str(m).lower() for m in gt_moves]

    # Text evaluation
    text_exact = 1 if pred_moves == gt_moves else 0
    text_frame_acc = (
        sum(1 for p, g in zip(pred_moves, gt_moves) if p == g) / len(gt_moves)
        if gt_moves else 0.0
    )

    return {
        "maze_text_exact": text_exact,
        "maze_text_frame_acc": text_frame_acc,
    }


# ============================================================================
# Sliding Puzzle Task Functions
# ============================================================================

def sliding_doc_to_visual(doc: Dict) -> List[Image.Image]:
    """Get visual input for sliding puzzle task."""
    if "initial_image" in doc and doc["initial_image"]:
        img_bytes = doc["initial_image"]["bytes"] if isinstance(doc["initial_image"], dict) else doc["initial_image"]
        return [Image.open(BytesIO(img_bytes)).convert("RGB")]
    return []


def sliding_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Get text prompt for sliding puzzle task - text-only evaluation."""
    prompt = """You are a precise sliding puzzle solver.

TASK
- You will be given an INITIAL state of a 3x3 sliding puzzle.
- The goal is to find the sequence of moves to solve the puzzle.

SEMANTICS
- The puzzle is a 3x3 grid with 8 colored tiles and one empty space.
- The RED square represents the EMPTY space.
- A "move" consists of sliding an adjacent colored tile INTO the empty (red) space.
- Moves are named by the direction the COLORED TILE moves. For example, if the blue tile is directly above the red space, moving the blue tile down into the red space's position is a "down" move.
- Legal moves: up, down, left, right only. One tile per step.

OUTPUT FORMAT
1) Briefly describe your reasoning for the solution.

2) Output the final move list as a JSON array of lowercase strings, wrapped as:
<ANSWER_JSON>["down","right","up"]</ANSWER_JSON>

NO EXTRAS
- No extra explanations beyond the brief reasoning and the JSON answer."""
    return prompt


def sliding_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """Process sliding puzzle results - text evaluation only."""
    result_raw = results[0] if results else ""
    
    # Handle case where result is a JSON string (from bagel format_output)
    result_text = ""
    if isinstance(result_raw, str):
        # Try to parse as JSON first (bagel outputs JSON string)
        try:
            parsed_result = json.loads(result_raw)
            if isinstance(parsed_result, dict) and "text" in parsed_result:
                result_text = parsed_result["text"]
            else:
                result_text = result_raw
        except (json.JSONDecodeError, TypeError):
            # Not JSON, use as-is
            result_text = result_raw
    else:
        result_text = str(result_raw)

    # Parse predicted moves from <ANSWER_JSON>
    pred_moves = []
    matches = list(re.finditer(
        r"<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>",
        result_text,
        re.DOTALL | re.IGNORECASE
    ))
    if matches:
        try:
            moves_data = json.loads(matches[-1].group(1))
            pred_moves = [str(m).strip().lower() for m in moves_data]
        except:
            pass

    # Ground truth moves
    gt_moves_str = doc.get("steps_words", "[]")
    gt_moves = json.loads(gt_moves_str) if isinstance(gt_moves_str, str) else gt_moves_str
    gt_moves = [str(m).lower() for m in gt_moves]

    # Text evaluation
    text_exact = 1 if pred_moves == gt_moves else 0
    text_frame_acc = (
        sum(1 for p, g in zip(pred_moves, gt_moves) if p == g) / len(gt_moves)
        if gt_moves else 0.0
    )

    return {
        "sliding_text_exact": text_exact,
        "sliding_text_frame_acc": text_frame_acc,
    }


# ============================================================================
# Visual CoT Versions - Aligned with Original Uni-MMMU
# ============================================================================
# These prompts match the exact prompts used in the original Uni-MMMU benchmark


# Jigsaw prompt (aligned with original jigsaw.py)
JIGSAW_PROMPT = """You are a unified vision-language model. You will be given:
(1) a 2×2 reference image with the bottom-right cell hidden, and
(2) two candidate patch images ("Candidate 0" and "Candidate 1").

Your job:
- For each candidate, synthesize a completed 2×2 image by placing that candidate EXACTLY into the bottom-right cell. Keep the other three cells pixel-identical to the reference (no filtering, no re-rendering). If sizes differ, only scale the candidate to fit that quadrant; do NOT rotate, mirror, or alter colors.
- Compare the two completed results and decide which candidate yields the correct completion.

Output EXACTLY the following, in order:

1) A single image with Candidate 0 placed in the bottom-right cell

2) A single image with Candidate 1 placed in the bottom-right cell


3) analysis comparing seam continuity, color/texture gradient, structural alignment, and global semantics

4) One strict JSON object with your decision, wrapped as:
<FINAL_ANSWER_JSON>
{"choice": 0 or 1, "rationale": "≤30 words decisive cue"}
</FINAL_ANSWER_JSON>

Hard constraints:
- Deterministic, single outcome. No hedging, no multiple possibilities.
- No meta talk about prompts, models, or pipelines.
- Do not restate the task as the answer; reason from visual evidence.
- The only edits allowed are pasting the candidate into the bottom-right cell and necessary size matching for that cell. All other pixels must remain identical to the reference.

Inputs:"""


# Maze prompt (aligned with original maze.py)
MAZE_PROMPT = """You are a precise maze solver.

SEMANTICS (for all mazes)
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- Green rectangular frame: goal (reaching any white cell inside the green frame counts as success)
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.

OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE — generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the maze state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep palette/layout/scale identical to the input; only the blue dot moves.
   - The number of returned images MUST equal the number of moves in the final answer (see step 2).
   - Absolutely FORBIDDEN: any collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images; no arrows, captions, or overlays; no GIFs/animations/video.

2) After all step images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of lowercase strings, wrapped as:
   <ANSWER_JSON>["right","down","left"]</ANSWER_JSON>


NO EXTRAS
- No tools, no OCR, no explanations, and no text other than the single <ANSWER_JSON>…</ANSWER_JSON> line.
- Do not restate the instructions or the condition.

REMINDERS
- Decide the full path first, then emit the image sequence (one image per move), then the single <ANSWER_JSON> line.
- One move per image; images must be separate files/parts, not stitched together in any way."""


# Sliding puzzle prompt (aligned with original sliding.py)
SLIDING_PROMPT = """You are a precise sliding puzzle solver.

SEMANTICS
- The puzzle is a 3×3 grid with 8 colored tiles and one empty space (shown as red).
- A "move" slides an adjacent tile INTO the empty space.
- Moves are named by the direction the COLORED TILE moves (not the empty space).
- Legal moves: up, down, left, right only. One tile per step.

OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE — generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the puzzle state AFTER applying exactly one move.
   - Do NOT include the initial (pre-move) state.
   - Keep tile colors identical; only positions change.
   - The number of returned images MUST equal the number of moves in the final answer.
   - Absolutely FORBIDDEN: any collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images.

2) After all step images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of lowercase strings, wrapped as:
   <ANSWER_JSON>["down","right","up"]</ANSWER_JSON>

NO EXTRAS
- No explanations beyond the single <ANSWER_JSON>…</ANSWER_JSON> line.
- Do not restate the instructions."""


def jigsaw_doc_to_text_visual_cot(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Visual CoT prompt for jigsaw task - aligned with original Uni-MMMU."""
    return JIGSAW_PROMPT


def maze_doc_to_text_visual_cot(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Visual CoT prompt for maze task - aligned with original Uni-MMMU."""
    return MAZE_PROMPT


def sliding_doc_to_text_visual_cot(doc: Dict, lmms_eval_specific_kwargs: Dict = None) -> str:
    """Visual CoT prompt for sliding puzzle - aligned with original Uni-MMMU."""
    return SLIDING_PROMPT
