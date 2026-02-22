"""
BabyVision Gen Evaluation Prompts
"""


def get_type_criteria(task_type: str, subtype: str) -> str:
    """Get evaluation criteria based on task type."""

    if task_type == "Fine-grained Discrimination":
        if subtype == "Find the different":
            return """
TASK: Find the unique/different element among many similar elements.
CRITERIA: Is the circle on the SAME grid position as ground truth?
- Circle on DIFFERENT element = FALSE
- No circle visible = FALSE"""

        elif subtype == "Find the same":
            return """
TASK: Find identical elements or matching figures.
CRITERIA: Are ALL marked elements the same as in ground truth?
- Circles on DIFFERENT elements = FALSE
- Missing circles = FALSE"""

        elif subtype == "Find the shadow":
            return """
TASK: Find the shadow/silhouette that matches the colored figure.
CRITERIA: Is the SAME option circled?"""

        elif subtype in ["Find the same components", "2D Pattern Completion", "Pattern and Color Completion"]:
            return """
TASK: Find/select the correct option.
CRITERIA: Is the SAME option circled/selected?"""

        elif subtype in ["Count Same Patterns", "Count Clusters"]:
            return """
TASK: Count patterns or fill in numbers.
CRITERIA: Do the markings/numbers match ground truth exactly?"""

    elif task_type == "Visual Tracking":
        if subtype == "Maze":
            return """
TASK: Draw a path through the maze.
CRITERIA: Does the path follow the EXACT SAME route as ground truth?
- Different route = FALSE
- No visible path = FALSE"""

        elif subtype == "Connect the lines":
            return """
TASK: Trace a line following the continuous path.
CRITERIA: Does the traced line follow the SAME path as ground truth?"""

        elif subtype == "Metro map":
            return """
TASK: Draw the shortest path between metro stations.
CRITERIA: Does the path follow the EXACT SAME route as ground truth?"""

        elif subtype == "Recognize numbers and letters":
            return """
TASK: Fill in letters/numbers in blanks.
CRITERIA: Are the EXACT SAME characters filled in each blank?"""

    elif task_type == "Spatial Perception":
        if subtype in ["3D Views", "3D Cube Unfold", "Paper Folding", "3D Pattern Completion"]:
            return """
TASK: Select the correct option for spatial reasoning.
CRITERIA: Is the SAME option circled?"""

        elif subtype == "Count 3D blocks":
            return """
TASK: Count cubes in a 3D structure.
CRITERIA: Is the EXACT SAME number written?"""

    elif task_type == "Visual Pattern Recognition":
        return """
TASK: Identify pattern and select correct option.
CRITERIA: Is the SAME option circled?"""

    return """
CRITERIA: Does the answer match ground truth exactly?
- Different answer = FALSE
- Missing answer = FALSE"""


def build_evaluation_prompt(task_type: str, subtype: str, generation_prompt: str) -> str:
    """Build type-specific evaluation prompt."""
    header = f"""You are evaluating an AI-generated image for a visual reasoning task.

TASK TYPE: {task_type}
SUBTYPE: {subtype}
GENERATION INSTRUCTION: "{generation_prompt}"

You are provided with THREE images:
- **Image 1 (Input)**: The original question/puzzle image
- **Image 2 (Ground Truth)**: The CORRECT answer showing what the result SHOULD look like
- **Image 3 (Generated)**: The AI-generated result to be evaluated

Compare Image 3 (Generated) with Image 2 (Ground Truth) to determine if they show the SAME answer.
"""

    criteria = get_type_criteria(task_type, subtype)

    footer = """

DECISION RULES:
- TRUE: Generated image shows the EXACT SAME answer as Ground Truth
- FALSE: Generated shows a DIFFERENT answer, NO answer, or UNCLEAR answer

IMPORTANT:
- Focus ONLY on whether the ANSWER matches, ignore style differences
- A marking on a DIFFERENT element/option = FALSE
- A path taking a DIFFERENT route = FALSE
- A DIFFERENT number or character = FALSE
- Missing required answer = FALSE

Respond with ONLY one word: "True" or "False"
"""

    return header + criteria + footer
