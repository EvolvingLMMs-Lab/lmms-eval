"""SPaRC visual: 2D pathfinding puzzles from "The Witness", given as an image.

The dataset (``lkaesberg/SPaRC``) has no image column, so each puzzle is drawn
with the ``sparc-visualization`` package (``plot_type="original"``), as the
reference ``sparc --visual`` command does.

The prompt builder and the path validator below are copied from the reference
implementation, so the task does not depend on the ``sparc-puzzle`` package and
its command-line dependencies:

* prompt:    https://github.com/lkaesberg/SPaRC/blob/v0.7.0/sparc/prompts/single_shot_visual.py
* validator: https://github.com/lkaesberg/SPaRC/blob/v0.7.0/sparc/validation.py
* messages:  https://github.com/lkaesberg/SPaRC/blob/v0.7.0/sparc/process_puzzle.py (``process_puzzle_visual``)

The logic is unchanged; only type hints and formatting were adapted to the
repository's linters. A path counts as correct only if it exactly matches one
of the puzzle's stored solutions (``validate_solution``). ``analyze_path``
gives the structural checks behind the paper's "invalid path" rate.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional

import datasets
from PIL import Image

from lmms_eval.imports import MissingOptionalDependencyError, optional_import

get_puzzle_image, _HAS_SPARC_VISUALIZATION = optional_import("sparc_visualization.plot", "get_puzzle_image")

PLOT_TYPE = "original"
SYSTEM_PROMPT = "You are an expert at solving visual puzzles from 'The Witness' game."
LEVELS = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Prompt builder (copied from sparc/prompts/single_shot_visual.py)
# ---------------------------------------------------------------------------


def _get_polyshapes_str(puzzle_data: Dict) -> str:
    """Extract polyshapes string from puzzle data."""
    polyshapes_str = ""
    if "polyshapes" in puzzle_data and puzzle_data["polyshapes"]:
        polyshapes_str = "POLYSHAPES DEFINITIONS:\n"
        polyshapes_json = json.loads(puzzle_data["polyshapes"])
        for shape_id, shape_def in polyshapes_json.items():
            shape_def_str = "\n".join(map(str, shape_def))
            polyshapes_str += f"Shape {shape_id}:\n{shape_def_str}\n\n"
    return polyshapes_str


def get_prompt(puzzle_data: Dict) -> Dict:
    """
    Generate the complete prompt dict for single-shot visual puzzle solving.

    Args:
        puzzle_data: The puzzle data dictionary

    Returns:
        Dict with 'system' and 'user' message content
    """
    grid_size = puzzle_data.get("grid_size", {"width": 0, "height": 0})
    polyshapes_str = _get_polyshapes_str(puzzle_data)
    text_visualization = puzzle_data.get("text_visualization", "")

    # The reference prompt's first line ends with a space; \x20 keeps it without trailing whitespace in this file.
    user_content = f"""You are an expert spatial reasoning AI specializing in solving puzzles from the game 'The Witness'.\x20
Your task is to solve the puzzle by finding a valid line from the Start Node to the End Node.

## Input Format
You receive TWO representations of the same puzzle:
1. **An image** showing the visual puzzle grid
2. **A text description** (below) providing the same puzzle in symbolic notation

Use both representations together — the image helps with spatial reasoning and visual pattern recognition, while the text description provides precise cell contents and coordinates.

## Visual Appearance of the Puzzle Image
The image shows a Witness puzzle grid of size {grid_size["width"] * 2}x{grid_size["height"] * 2} with a dark teal/green background. Here is how elements appear visually:

- **Grid structure**: The board has teal/green cells separated by dark gray grid lines (the path network)
- **Start Node**: A large filled circle on the grid edge — this is where the path begins
- **End Node**: A small rounded nub/extension protruding outward from the grid edge — this is where the path must end
- **Dots**: Small black hexagons located on the grid lines — the path MUST pass through these
- **Gaps**: Broken/missing segments on the grid lines — the path CANNOT cross these
- **Colored Squares**: Colored rounded rectangles inside cells (e.g., black, red, blue squares)
- **Colored Stars**: 8-pointed star shapes in their respective color inside cells
- **Triangles**: Small colored upward-pointing triangles inside cells — the count (1-4 triangles) indicates how many edges of that cell the path must touch
- **Polyshapes (positive)**: Tetromino-like colored filled block patterns inside cells, showing the shape the region must match
- **Polyshapes (negative/ylop)**: Same block patterns but drawn as hollow/outlined squares instead of filled — these subtract area from the positive polyshapes in their region

## Grid & Coordinate System
- Nodes are indexed (x, y) where (0,0) is the top-left node
- x increases to the right, y increases downward
- The grid cells have rule symbols located at cells where both x and y are odd
- The line goes AROUND cells containing rules, forming boundaries
- Both line and rule cells are on the same grid. Therefore each intersection has a distance of 2 to the next intersection.

## Solving Rules
1. Draw a continuous line from the START NODE (big circle on the line) to the END NODE (rounded nub) without visiting the same node twice.
2. The line can only be placed on valid path cells.
3. The line can only travel 1 cell per step (no diagonal moves).
4. The line acts as a boundary, potentially dividing the grid cells into one or more distinct regions.
5. All rules associated with symbols on the grid must be satisfied:
   - **Dots** (black hexagons on grid lines): The line MUST pass through each dot.
   - **Colored squares** (filled rounded rectangles in cells): All squares within a single region must be the same color. Different colored squares MUST be separated into different regions.
   - **Colored stars** (8-pointed stars in cells): Each star must be paired with EXACTLY one other element of the same color in its region. Other colors are ignored.
   - **Triangles** (small triangles in cells): The line must touch EXACTLY the number of edges specified by the count of triangles in that cell (edges are top, right, bottom, left of the cell).
   - **Polyshapes** (filled block patterns in cells): The region containing this symbol must be shaped EXACTLY like the defined polyshape.
   - **Negative polyshapes** (hollow block patterns in cells): These subtract area from the regular polyshapes in the same region. The region must have the area of the regular polyshapes minus the area of the negative ones. Their shapes and colors do not have to match.

{polyshapes_str}

## Text Description of the Puzzle
The text below uses symbolic notation where: `S`=Start, `E`=End, `+`=walkable path cell, `N`=empty rule cell, `G`=gap, `.`=dot, `o-X`=square of color X, `*-X`=star of color X, `A/B/C/D-X`=triangle (1-4 edges), `P-X-Y`=polyshape, `Y-X-Y`=negative polyshape. Color codes: R=Red, B=Blue, G=Green, Y=Yellow, W=White, O=Orange, P=Purple, K=Black.

{text_visualization}

## Task
Analyze both the puzzle image and the text description carefully and determine the solution path.
First, explain your reasoning step-by-step, including key deductions and constraint checks made along the way.
Then, provide the final solution as a sequence of node coordinates in (x, y) format, starting with the start node and ending with the end node, after this string: "####". DON'T SKIP ANY intermediate nodes (the distance between each node must be 1).
Example coordinate list: [(0,0), (1,0), (2,0), (2,1), ...]
"""

    return {
        "system": "You are an expert at solving visual puzzles from 'The Witness' game.",
        "user": user_content,
    }


# ---------------------------------------------------------------------------
# Validator (copied from sparc/validation.py)
# ---------------------------------------------------------------------------


def extract_solution_path(solution_text: str, puzzle_data: Optional[Dict] = None) -> Optional[List[Dict[str, int]]]:
    """Extract solution path from LLM's response

    Args:
        solution_text: Text response from the LLM
        puzzle_data: Optional puzzle data dict to extract end point from

    Returns:
        List of coordinate dicts or None if no path found
    """
    # First, check if "Solution" appears in the text
    solution_marker = "####"
    if solution_marker in solution_text:
        # Only process text after "####"
        solution_part = solution_text.split(solution_marker)[-1]
    else:
        # If no solution marker, use the full text
        solution_part = solution_text

    # Look for coordinate patterns like (0,0) -> (0,1) or similar
    # Pattern for (x,y) or (x, y) coordinates
    coord_pattern = r"\((\d+),\s*(\d+)\)"
    coords = re.findall(coord_pattern, solution_part)

    if coords:
        # Extract end point from puzzle data if provided
        end_point = None
        if puzzle_data:
            puzzle_array = puzzle_data.get("puzzle_array", [])
            for y, row in enumerate(puzzle_array):
                for x, cell in enumerate(row):
                    if cell == "E":
                        end_point = {"x": x, "y": y}
                        break
                if end_point:
                    break

        # Convert string coordinates to integer dicts
        path = []
        for x, y in coords:
            point = {"x": int(x), "y": int(y)}
            path.append(point)

            # Stop extracting if we've reached the end point
            if end_point and point["x"] == end_point["x"] and point["y"] == end_point["y"]:
                break

        return path

    # If no coordinates found, return None
    return None


def validate_solution(extracted_path: Optional[List[Dict[str, int]]], puzzle_data: Dict) -> bool:
    """Validate if the solution is valid by comparing with known solutions

    Args:
        extracted_path: Optional list of coordinate dicts (can be None if no path extracted)
        puzzle_data: Dictionary containing puzzle data including solutions

    Returns:
        Boolean indicating if the solution is valid. Returns False if no valid path is provided.
    """
    # If no path was extracted, validation fails immediately.
    if not extracted_path:
        return False

    extracted_path = [(p["x"], p["y"]) for p in extracted_path]
    if len(extracted_path) < 2:
        return False

    # Check against all valid solutions in the database
    all_solutions = puzzle_data.get("solutions", [])
    if not all_solutions:
        return False

    # For each solution in the database, check if our path matches
    for solution in all_solutions:
        solution_path = [(p["x"], p["y"]) for p in solution["path"]]

        # Check if the paths match exactly
        if extracted_path == solution_path:
            return True

    return False


def analyze_path(solution_path: Optional[List[Dict[str, int]]], puzzle: Dict) -> Dict:
    """Analyze the solution path for detailed validation metrics

    Args:
        solution_path: Optional list of coordinate dicts representing the solution path (can be None)
        puzzle: Puzzle dictionary with puzzle array and metadata

    Returns:
        Dictionary with detailed path analysis results
    """
    # Early return with all metrics False if no path was provided
    if not solution_path:
        return {
            "starts_at_start_ends_at_exit": False,
            "connected_line": False,
            "non_intersecting_line": False,
            "start_to_exit_connected": False,
            "no_rule_crossing": False,
            "fully_valid_path": False,
        }

    solution_path = [(p["x"], p["y"]) for p in solution_path]

    # Get puzzle information
    puzzle_array = puzzle.get("puzzle_array", [])
    if not puzzle_array:
        return {
            "starts_at_start_ends_at_exit": False,
            "connected_line": False,
            "non_intersecting_line": False,
            "start_to_exit_connected": False,
            "no_rule_crossing": False,
            "fully_valid_path": False,
        }

    # Find start and end points
    start_point = None
    end_point = None
    for y, row in enumerate(puzzle_array):
        for x, cell in enumerate(row):
            if cell == "S":
                start_point = (x, y)
            elif cell == "E":
                end_point = (x, y)

    if not start_point or not end_point:
        return {
            "starts_at_start_ends_at_exit": False,
            "connected_line": False,
            "non_intersecting_line": False,
            "start_to_exit_connected": False,
            "no_rule_crossing": False,
            "fully_valid_path": False,
        }

    # Check if path starts at start and ends at exit
    starts_at_start = len(solution_path) > 0 and solution_path[0] == start_point
    ends_at_exit = len(solution_path) > 0 and solution_path[-1] == end_point
    starts_at_start_ends_at_exit = starts_at_start and ends_at_exit

    # Check if path is connected (no gaps)
    connected_line = True
    for i in range(1, len(solution_path)):
        prev_x, prev_y = solution_path[i - 1]
        curr_x, curr_y = solution_path[i]
        # Check if adjacent (Manhattan distance of 1)
        if abs(prev_x - curr_x) + abs(prev_y - curr_y) != 1:
            connected_line = False
            break

    # Check if path doesn't intersect with itself
    non_intersecting_line = len(set(solution_path)) == len(solution_path)

    # Check if there's a connected path from start to exit
    start_to_exit_connected = starts_at_start_ends_at_exit and connected_line

    # Identify rule cells as cells where both coordinates are odd
    rule_cells = set()
    for y in range(len(puzzle_array)):
        for x in range(len(puzzle_array[0]) if len(puzzle_array) > 0 else 0):
            if x % 2 == 1 and y % 2 == 1:
                rule_cells.add((x, y))

    # Check if path crosses rule cells
    no_rule_crossing = not any((x, y) in rule_cells for x, y in solution_path[1:-1])

    # Check if path is fully valid
    fully_valid_path = starts_at_start_ends_at_exit and connected_line and non_intersecting_line and no_rule_crossing

    return {
        "starts_at_start_ends_at_exit": starts_at_start_ends_at_exit,
        "connected_line": connected_line,
        "non_intersecting_line": non_intersecting_line,
        "start_to_exit_connected": start_to_exit_connected,
        "no_rule_crossing": no_rule_crossing,
        "fully_valid_path": fully_valid_path,
    }


# ---------------------------------------------------------------------------
# lmms-eval hooks
# ---------------------------------------------------------------------------


def sparc_visual_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    """Render the puzzle as the reference ``sparc --visual`` image (``plot_type="original"``)."""
    if not _HAS_SPARC_VISUALIZATION:
        raise MissingOptionalDependencyError("sparc-visualization", feature="the sparc_visual task")
    return [get_puzzle_image(doc, plot_type=PLOT_TYPE).convert("RGB")]


def sparc_visual_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """User text of the SPaRC single-shot visual prompt (the system prompt is added in ``doc_to_messages``)."""
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{get_prompt(doc)['user']}{post_prompt}"


def sparc_visual_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """System prompt, then one user turn with the image first and the text second, as in ``process_puzzle_visual``."""
    content = [{"type": "image", "url": image} for image in sparc_visual_doc_to_visual(doc)]
    content.append({"type": "text", "text": sparc_visual_doc_to_text(doc, lmms_eval_specific_kwargs)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def sparc_visual_doc_to_target(doc: Dict[str, Any]) -> str:
    """First stored solution, formatted like the expected answer (reference only)."""
    path = doc["solutions"][0]["path"]
    return "#### " + ", ".join(f"({p['x']},{p['y']})" for p in path)


def sparc_visual_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Interleave difficulty levels so ``--limit n`` covers all levels evenly.

    Puzzles keep their dataset order within each level; the output order is
    level 1, 2, 3, 4, 5, level 1, 2, ... until every level is exhausted.
    """
    by_level: Dict[int, List[int]] = {}
    for idx, level in enumerate(dataset["difficulty_level"]):
        by_level.setdefault(level, []).append(idx)
    queues = [by_level[level] for level in sorted(by_level)]
    order = [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]
    return dataset.select(order)


def sparc_visual_process_hard(dataset: datasets.Dataset) -> datasets.Dataset:
    """Keep only difficulty-level-5 puzzles (the paper's "hard" subset)."""
    return dataset.filter(lambda doc: doc["difficulty_level"] == 5)


def sparc_visual_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    """Score one generation: exact-match accuracy and the paper's invalid-path flag."""
    response = (results[0] if results else "") or ""
    path = extract_solution_path(response, doc)
    solved = validate_solution(path, doc)
    analysis = analyze_path(path, doc)
    return {
        "acc": float(solved),
        "invalid_path": float(not analysis["fully_valid_path"]),
    }


def sparc_visual_process_results_by_level(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """``sparc_visual_process_results`` plus one ``acc_level_<k>`` entry per difficulty level.

    Every document fills every ``acc_level_<k>`` key with the same payload, and
    the level aggregators keep only the documents of their level.
    """
    out: Dict[str, Any] = sparc_visual_process_results(doc, results)
    payload = {"difficulty_level": int(doc["difficulty_level"]), "solved": out["acc"]}
    out.update({f"acc_level_{level}": payload for level in LEVELS})
    return out


def _level_aggregator(level: int) -> Callable[[List[Dict[str, Any]]], float]:
    def aggregate(results: List[Dict[str, Any]]) -> float:
        scores = [r["solved"] for r in results if r["difficulty_level"] == level]
        return sum(scores) / len(scores) if scores else float("nan")

    aggregate.__name__ = f"sparc_visual_aggregate_level_{level}"
    aggregate.__doc__ = f"Accuracy over the difficulty-level-{level} puzzles (NaN if none were evaluated)."
    return aggregate


sparc_visual_aggregate_level_1 = _level_aggregator(1)
sparc_visual_aggregate_level_2 = _level_aggregator(2)
sparc_visual_aggregate_level_3 = _level_aggregator(3)
sparc_visual_aggregate_level_4 = _level_aggregator(4)
sparc_visual_aggregate_level_5 = _level_aggregator(5)
