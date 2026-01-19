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
        parsed = find_first_json_substring(json_str)
        if parsed:
            try:
                data = json.loads(parsed)
                choice = data.get("choice")
            except:
                pass

    # If no JSON found, try to find choice directly
    if choice is None:
        # Try to find "choice": 0 or "choice": 1
        choice_match = re.search(r'"choice"\s*:\s*(\d)', result_text)
        if choice_match:
            choice = int(choice_match.group(1))

    gt_label = doc.get("label", -1)
    text_correct = 1 if choice == gt_label else 0

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
    result_text = results[0] if results else ""

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
    result_text = results[0] if results else ""

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
