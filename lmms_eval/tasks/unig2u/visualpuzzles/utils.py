import json
import logging
import os
import random
import re
from collections import defaultdict

import numpy as np
from PIL import Image

eval_logger = logging.getLogger("lmms-eval")

MULTI_CHOICE_DIRECT_PROMPT = "Answer the question with the option's letter from the given choices directly."
COT_PROMPT = "Solve the multiple-choice question and then answer with the option letter from the given choices. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering."
PROMPTS = {"MULTI_CHOICE_DIRECT_PROMPT": MULTI_CHOICE_DIRECT_PROMPT, "COT_PROMPT": COT_PROMPT}


def VisualPuzzles_doc_to_visual(doc):
    image = doc["image"]
    # Handle HuggingFace datasets image format (dict with 'bytes' or already PIL Image)
    if isinstance(image, dict):
        # If it's a dict, convert to PIL Image
        if "bytes" in image:
            import io
            image = Image.open(io.BytesIO(image["bytes"]))
        elif "path" in image:
            image = Image.open(image["path"])
        else:
            # Try to use the dict directly as Image (some formats)
            import numpy as np
            image = Image.fromarray(np.array(image))
    return [image]


def VisualPuzzles_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = "Question: " + doc["question"].strip()
    options = doc["options"]
    if options != None:
        question += "\nOptions:\n(A) " + options[0] + "\n(B) " + options[1] + "\n(C) " + options[2] + "\n(D) " + options[3]
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."
    question += "\n" + PROMPTS[lmms_eval_specific_kwargs["prompt"]]
    return question


def parse_response(response, all_choices, index2ans):
    """
    Return the last letter appearing after 'ANSWER:' in the input text.
    If there's no match, return None.
    """
    pattern = r"Answer:\s*\(([A-Za-z])\)"  # Answer: (A)
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"(?<!Final )Answer:\s*([A-Za-z])"  # Answer: A
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"Answer:\s*([A-Za-z])"  # Answer: A
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"\s*\(([A-Za-z])\)"  # e.g., (A) (B) (C) (D)
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    response = " " + response.strip()
    pattern = r"\s*([A-Za-z])\)"  # e.g., A) B) C) D)
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"\s*\{([A-Za-z])\}"  # e.g., {A} {B} {C} {D}
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r"\s*\$([A-Za-z])\$"  # e.g., $A$, $B$, $C$, $D$
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r" ([A-Da-d])\."  # e.g., A. B. C. D.
    matches = re.findall(pattern, response)
    if matches:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    pattern = r" ([A-Da-d])"  # e.g., A B C D
    matches = re.findall(pattern, response)
    if matches and len(response) <= 5:
        for match in matches[::-1]:
            if match in all_choices or match.upper() in all_choices:
                return match
    if index2ans != None:
        for index in all_choices:
            ans = index2ans[index]
            if f"answer: {ans}" in response.lower():
                return index
            if f"answer:{ans}" in response.lower():
                return index
        last_found = None
        last_index = -1
        for index in all_choices:
            ans = index2ans[index]
            idx = response.rfind(ans)
            if idx > last_index:
                last_found = index
                last_index = idx
        if last_found:
            return last_found
    return random.choice(all_choices)


def VisualPuzzles_process_result(doc, results):
    """
    Process results with category-based metrics

    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (category), value: metric dict
    """
    pred = results[0].strip()
    all_choices = ["A", "B", "C", "D"]
    if doc["options"] == None:
        index2ans = None
    else:
        index2ans = {all_choices[i]: doc["options"][i] for i in range(4)}
    pred = parse_response(pred, all_choices, index2ans)
    target = doc["answer"]

    # Calculate score
    score = 1.0 if pred.lower() == target.lower() else 0.0

    # Get category from doc
    category = doc.get("category", "Unknown")

    # Return results for both category-specific and overall metrics
    result_dict = {
        category: {
            "question_id": doc.get("id", doc.get("idx", "unknown")),
            "category": category,
            "score": score
        },
        "average": {
            "question_id": doc.get("id", doc.get("idx", "unknown")),
            "category": category,
            "score": score
        }
    }

    return result_dict


def VisualPuzzles_aggregate_results(results):
    """
    Aggregate results by category

    Args:
        results: a list of values returned by process_results
    Returns:
        Average score for the category
    """
    category_scores = defaultdict(list)

    for result in results:
        score = result["score"]
        category = result["category"]
        category_scores[category].append(score)

    # Calculate average for each category
    category_avg_scores = {}
    for category, scores in category_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0.0
        category_avg_scores[category] = avg_score
        eval_logger.info(f"{category}: {avg_score:.4f} ({len(scores)} samples)")

    # Calculate overall average
    all_scores = [score for scores in category_scores.values() for score in scores]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return avg_score


def VisualPuzzles_process_result_simple(doc, results):
    """
    Simplified process results for single-category tasks.

    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with accuracy score
    """
    pred = results[0].strip()
    all_choices = ["A", "B", "C", "D"]
    if doc["options"] == None:
        index2ans = None
    else:
        index2ans = {all_choices[i]: doc["options"][i] for i in range(4)}
    pred = parse_response(pred, all_choices, index2ans)
    target = doc["answer"]

    # Calculate score
    score = 1.0 if pred.lower() == target.lower() else 0.0

    return {"accuracy": score}


def VisualPuzzles_aggregate_simple(results):
    """
    Simple aggregation for single-category tasks.

    Args:
        results: a list of score values
    Returns:
        Average accuracy
    """
    if not results:
        return 0.0
    return sum(results) / len(results)


# ============================================================
# Visual CoT Prompt Functions (Category-Specific)
# ============================================================

def VisualPuzzles_doc_to_text_visual_cot_algorithmic(doc, lmms_eval_specific_kwargs=None):
    """
    Visual CoT prompt for ALGORITHMIC reasoning puzzles.
    Stage 1: Identify numerical/symbolic patterns and computation rules.
    """
    question = "Question: " + doc["question"].strip()
    options = doc["options"]
    if options is not None:
        question += "\nOptions:\n(A) " + options[0] + "\n(B) " + options[1] + "\n(C) " + options[2] + "\n(D) " + options[3]
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."

    # Stage 1: Generate auxiliary visualization for algorithmic reasoning
    generation_prompt = f"""You are given an algorithmic reasoning puzzle. Analyze the puzzle and create a helpful visualization.

{question}

Your task:
1. Identify any numerical sequences, patterns, or computational rules in the puzzle
2. Create a diagram that clearly shows:
   - The step-by-step computation or transformation process
   - Arrows or annotations showing how numbers/symbols change
   - The mathematical relationship or formula discovered
   - Highlighted patterns (e.g., +2, ×3, alternating, etc.)
3. Label each step of the algorithm clearly

Generate a clear diagram that reveals the underlying algorithmic pattern."""

    # Stage 2: Solve using both images
    question_prompt = f"""{question}

You are given TWO images:
1) ORIGINAL PUZZLE: The algorithmic reasoning puzzle
2) AUXILIARY DIAGRAM: A visualization showing the computational pattern and steps

Use the auxiliary diagram to understand the algorithm, then select the correct answer.
Answer with the option letter (A, B, C, or D) directly."""

    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT]\n[QUESTION]{question_prompt}[/QUESTION]"


def VisualPuzzles_doc_to_text_visual_cot_analogical(doc, lmms_eval_specific_kwargs=None):
    """
    Visual CoT prompt for ANALOGICAL reasoning puzzles.
    Stage 1: Identify transformation relationships between elements.
    """
    question = "Question: " + doc["question"].strip()
    options = doc["options"]
    if options is not None:
        question += "\nOptions:\n(A) " + options[0] + "\n(B) " + options[1] + "\n(C) " + options[2] + "\n(D) " + options[3]
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."

    # Stage 1: Generate auxiliary visualization for analogical reasoning
    generation_prompt = f"""You are given an analogical reasoning puzzle (A is to B as C is to ?).

{question}

Your task:
1. Identify the transformation relationship between the first pair of elements
2. Create a diagram that clearly shows:
   - What changes occur (rotation, reflection, color change, size change, addition/removal of elements)
   - Arrows indicating the direction and type of transformation
   - Labels describing each transformation (e.g., "rotate 90°", "invert colors", "add dot")
   - Apply the same transformation to show what the answer should look like
3. Make the analogy relationship visually explicit

Generate a diagram that reveals the transformation pattern between pairs."""

    # Stage 2: Solve using both images
    question_prompt = f"""{question}

You are given TWO images:
1) ORIGINAL PUZZLE: The analogical reasoning puzzle
2) AUXILIARY DIAGRAM: A visualization showing the transformation relationship

Use the auxiliary diagram to understand how A transforms to B, apply the same rule to C, then select the correct answer.
Answer with the option letter (A, B, C, or D) directly."""

    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT]\n[QUESTION]{question_prompt}[/QUESTION]"


def VisualPuzzles_doc_to_text_visual_cot_deductive(doc, lmms_eval_specific_kwargs=None):
    """
    Visual CoT prompt for DEDUCTIVE reasoning puzzles.
    Stage 1: Map out logical rules and inference chains.
    """
    question = "Question: " + doc["question"].strip()
    options = doc["options"]
    if options is not None:
        question += "\nOptions:\n(A) " + options[0] + "\n(B) " + options[1] + "\n(C) " + options[2] + "\n(D) " + options[3]
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."

    # Stage 1: Generate auxiliary visualization for deductive reasoning
    generation_prompt = f"""You are given a deductive reasoning puzzle that requires logical inference.

{question}

Your task:
1. Identify the given premises, rules, or constraints in the puzzle
2. Create a diagram that clearly shows:
   - All given conditions/rules listed clearly
   - A logical flowchart or inference chain
   - Step-by-step deduction from premises to conclusion
   - Elimination of incorrect possibilities
   - The logical path leading to the answer
3. Use arrows to show the deduction flow

Generate a logical inference diagram that traces the reasoning path."""

    # Stage 2: Solve using both images
    question_prompt = f"""{question}

You are given TWO images:
1) ORIGINAL PUZZLE: The deductive reasoning puzzle
2) AUXILIARY DIAGRAM: A logical inference diagram showing the deduction steps

Follow the logical chain in the auxiliary diagram to reach the conclusion, then select the correct answer.
Answer with the option letter (A, B, C, or D) directly."""

    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT]\n[QUESTION]{question_prompt}[/QUESTION]"


def VisualPuzzles_doc_to_text_visual_cot_inductive(doc, lmms_eval_specific_kwargs=None):
    """
    Visual CoT prompt for INDUCTIVE reasoning puzzles.
    Stage 1: Identify repeating patterns and generalize rules from examples.
    """
    question = "Question: " + doc["question"].strip()
    options = doc["options"]
    if options is not None:
        question += "\nOptions:\n(A) " + options[0] + "\n(B) " + options[1] + "\n(C) " + options[2] + "\n(D) " + options[3]
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."

    # Stage 1: Generate auxiliary visualization for inductive reasoning
    generation_prompt = f"""You are given an inductive reasoning puzzle that requires pattern recognition.

{question}

Your task:
1. Observe the sequence of examples and identify the underlying pattern
2. Create a diagram that clearly shows:
   - The repeating elements or motifs highlighted/circled
   - The progression rule (what changes from one step to the next)
   - Annotations showing the pattern cycle or growth rule
   - A prediction of what comes next based on the pattern
   - Color-coding or numbering to show pattern repetition
3. Make the inductive pattern visually obvious

Generate a diagram that highlights the repeating pattern and predicts the next element."""

    # Stage 2: Solve using both images
    question_prompt = f"""{question}

You are given TWO images:
1) ORIGINAL PUZZLE: The inductive reasoning puzzle
2) AUXILIARY DIAGRAM: A visualization highlighting the pattern and its progression

Use the auxiliary diagram to understand the pattern rule, then select the answer that continues the pattern correctly.
Answer with the option letter (A, B, C, or D) directly."""

    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT]\n[QUESTION]{question_prompt}[/QUESTION]"


def VisualPuzzles_doc_to_text_visual_cot_spatial(doc, lmms_eval_specific_kwargs=None):
    """
    Visual CoT prompt for SPATIAL reasoning puzzles.
    Stage 1: Visualize rotations, folding, or 3D transformations.
    """
    question = "Question: " + doc["question"].strip()
    options = doc["options"]
    if options is not None:
        question += "\nOptions:\n(A) " + options[0] + "\n(B) " + options[1] + "\n(C) " + options[2] + "\n(D) " + options[3]
    else:
        question += "\nOptions: Choose from (A) (B) (C) (D) in the image."

    # Stage 1: Generate auxiliary visualization for spatial reasoning
    generation_prompt = f"""You are given a spatial reasoning puzzle involving 3D visualization or transformations.

{question}

Your task:
1. Analyze the spatial transformation required (rotation, folding, unfolding, different viewpoint)
2. Create a diagram that clearly shows:
   - The object from multiple angles if rotation is involved
   - Step-by-step folding/unfolding process if applicable
   - Arrows indicating rotation direction and degree
   - Reference points or markers to track orientation
   - The resulting shape after transformation
3. Add axis lines or reference frames to clarify spatial orientation

Generate a multi-view or step-by-step transformation diagram."""

    # Stage 2: Solve using both images
    question_prompt = f"""{question}

You are given TWO images:
1) ORIGINAL PUZZLE: The spatial reasoning puzzle
2) AUXILIARY DIAGRAM: A visualization showing the spatial transformation from multiple views

Use the auxiliary diagram to mentally trace the transformation, then select the correct answer.
Answer with the option letter (A, B, C, or D) directly."""

    return f"[GEN_PROMPT]{generation_prompt}[/GEN_PROMPT]\n[QUESTION]{question_prompt}[/QUESTION]"
