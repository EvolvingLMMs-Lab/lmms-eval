"""
Prompts for Visual CoT Evaluation

Two main evaluation metrics:
- RA (Reasoning-to-Visual Alignment): Does generated image match generation_prompt?
- AL (Answer-to-Visual Alignment): Is answer consistent with generated images?
"""

# =============================================================================
# RA: Reasoning-to-Visual Alignment Prompt
# =============================================================================

REASONING_VISUAL_PROMPT = """
You are a professional AI evaluation specialist for Visual Chain-of-Thought tasks.

## Task Context
- **Task**: {task}
- **Original Question**: {question}

## Your Objective
Evaluate if the **generated visualization images** correctly follow the **generation prompt**.

## Images Provided
- **Image 1**: Original question image (the input)
- **Image 2+**: Generated visualization image(s) (the auxiliary output created to help answer the question)

## Generation Prompt (What model was instructed to generate)
"{generation_prompt}"

## Evaluation Criteria

### 1. Instruction Following (40%)
- Does the generated image follow the generation prompt instructions?
- Are all requested visual elements present?
- Does it match the specified visualization type?

### 2. Visual Quality (30%)
- Is the generated image clear and well-formed?
- Are visual elements properly rendered?
- Is it helpful as an auxiliary visualization?

### 3. Relevance to Task (30%)
- Does the visualization help answer the original question?
- Is it relevant to the task context?
- Does it provide useful visual cues?

## Scoring Scale (1-5)

- **5 Perfect**: Generated image(s) perfectly match generation prompt; all instructions followed; high quality; highly relevant
- **4 Good**: Generated image(s) mostly match prompt (80-90%); minor issues; good quality; relevant
- **3 Adequate**: Generated image(s) partially match prompt (60-70%); noticeable issues; acceptable quality
- **2 Poor**: Generated image(s) barely match prompt (30-50%); major issues; poor quality or relevance
- **1 Failed**: Generated image(s) completely fail to match prompt; unusable; irrelevant

## Important Notes
- Focus on whether the **generation prompt** was correctly executed
- Do NOT evaluate if the final answer is correct (that's a separate metric)
- Evaluate the **visualization quality** and **instruction adherence**

## Output Format
{{
  "reasoning_visual_score": X,
  "reasoning": "Detailed evaluation of how well generated images match the generation prompt"
}}
"""


# =============================================================================
# AL: Answer-to-Visual Alignment Prompt
# =============================================================================

ANSWER_VISUAL_ALIGNMENT_PROMPT = """
You are a professional AI evaluation specialist for Visual Chain-of-Thought tasks.

## Task Context
- **Task**: {task}
- **Original Question**: {question}

## Your Objective
Evaluate if the **final answer** is consistent with the **generated visualization images** and the **original question**.

## Images Provided
- **Image 1**: Original question image (the input)
- **Image 2+**: Generated visualization image(s) (auxiliary output used to derive the answer)

## Final Answer
"{answer}"

## Evaluation Criteria

### 1. Visual-Answer Consistency (50%)
- Is the answer supported by visual evidence in generated images?
- Can you trace the answer back to visual elements?
- Are there contradictions between answer and visuals?

### 2. Question-Answer Alignment (30%)
- Does the answer correctly address the original question?
- Is the answer format appropriate?
- Is it a valid response to the question?

### 3. Reasoning Coherence (20%)
- Does the reasoning flow make sense (original → generated visual → answer)?
- Is the answer logically derived from the visualization?
- Is the overall chain-of-thought coherent?

## Scoring Scale (1-5)

- **5 Perfect**: Answer perfectly matches generated visuals and question; fully consistent; logically sound; correct reasoning
- **4 Good**: Answer mostly consistent (80-90%); minor inconsistencies; generally correct reasoning
- **3 Adequate**: Answer partially consistent (60-70%); some contradictions; reasoning has gaps
- **2 Poor**: Answer barely consistent (30-50%); major contradictions; flawed reasoning
- **1 Failed**: Answer completely inconsistent with visuals or question; contradictory; nonsensical

## Important Notes
- Focus on **consistency** between answer and generated visuals
- Check if answer logically follows from the visualization
- Verify answer addresses the original question
- Do NOT evaluate if generation prompt was followed (that's RA metric)

## Output Format
{{
  "answer_visual_score": X,
  "reasoning": "Detailed evaluation of answer consistency with visuals and question"
}}
"""


# =============================================================================
# Seven Task Categories - Customized Criteria
# =============================================================================

TASK_CATEGORY_CRITERIA = {
    "real_world": {
        "ra_criteria": """
## Real-world Applications - RA Criteria
- **Safety & Feasibility**: Does visualization show safe, practical steps?
- **Contextual Understanding**: Does it consider real-world constraints (time, resources, safety)?
- **Actionable Guidance**: Are visual steps clear enough to follow in practice?
- **Ethical Considerations**: Does visualization respect safety protocols and ethical guidelines?
""",
        "al_criteria": """
## Real-world Applications - AL Criteria
- **Practical Validity**: Is the answer practically feasible and safe?
- **Evidence-Based**: Is answer supported by visual evidence showing real-world execution?
- **Completeness**: Does answer address all critical real-world aspects (safety, resources)?
- **Ethical Soundness**: Is the answer ethically appropriate and safe?
""",
    },
    "mathematical": {
        "ra_criteria": """
## Mathematical Reasoning - RA Criteria
- **Formula Visualization**: Are mathematical formulas and equations properly visualized?
- **Step-by-Step Clarity**: Does visualization show clear calculation steps?
- **Numerical Accuracy**: Are all numerical values and calculations visually correct?
- **Mathematical Notation**: Is proper mathematical notation used?
""",
        "al_criteria": """
## Mathematical Reasoning - AL Criteria
- **Calculation Correctness**: Is the answer mathematically accurate?
- **Formula Application**: Does answer correctly apply formulas shown in visualization?
- **Logical Consistency**: Is the mathematical logic coherent from visual → answer?
- **Unit Consistency**: Are units and scales properly maintained?
""",
    },
    "stem": {
        "ra_criteria": """
## STEM - RA Criteria
- **Scientific Accuracy**: Does visualization follow scientific principles and laws?
- **Technical Precision**: Are technical details and processes accurately depicted?
- **Domain Knowledge**: Does it demonstrate correct STEM domain knowledge?
- **Experimental/Process Visualization**: Are processes or experiments properly illustrated?
""",
        "al_criteria": """
## STEM - AL Criteria
- **Scientific Validity**: Is answer scientifically correct?
- **Technical Accuracy**: Does answer align with technical/scientific principles shown?
- **Evidence-Based**: Is answer supported by scientific evidence in visualization?
- **Domain Consistency**: Does answer demonstrate proper STEM domain understanding?
""",
    },
    "puzzles": {
        "ra_criteria": """
## Puzzles and Games - RA Criteria
- **Rule Adherence**: Does visualization follow game/puzzle rules correctly?
- **Strategic Clarity**: Are strategic steps or moves clearly visualized?
- **Valid Moves**: Are all shown moves/steps valid according to rules?
- **Solution Path**: Does visualization show a valid path to solution?
""",
        "al_criteria": """
## Puzzles and Games - AL Criteria
- **Rule Compliance**: Does answer follow all game/puzzle rules?
- **Solution Validity**: Is the answer a valid solution to the puzzle?
- **Move Correctness**: Do final moves/steps match what's shown in visualization?
- **Win Condition**: Does answer achieve the stated goal/win condition?
""",
    },
    "chart_table": {
        "ra_criteria": """
## Chart & Table Reasoning - RA Criteria
- **Data Highlighting**: Are relevant data points properly highlighted?
- **Trend Visualization**: Are trends, patterns, or key values emphasized?
- **Label Clarity**: Are axes, labels, and legends clearly marked?
- **Data Accuracy**: Is extracted/highlighted data numerically accurate?
""",
        "al_criteria": """
## Chart & Table Reasoning - AL Criteria
- **Data Accuracy**: Does answer match actual data values in chart/table?
- **Trend Correctness**: If describing trends, is interpretation correct?
- **Numerical Precision**: Are numerical answers precise and accurate?
- **No Hallucination**: Is answer derived from actual data, not hallucinated?
""",
    },
    "spatial": {
        "ra_criteria": """
## Spatial Intelligence - RA Criteria
- **3D Understanding**: Does visualization correctly represent 3D spatial relationships?
- **Transformation Accuracy**: Are rotations, translations, or transformations correct?
- **Perspective Consistency**: Is spatial perspective maintained correctly?
- **Geometric Precision**: Are geometric relationships accurately visualized?
""",
        "al_criteria": """
## Spatial Intelligence - AL Criteria
- **Spatial Correctness**: Is answer spatially accurate?
- **Transformation Validity**: Do described transformations match visualization?
- **Geometric Consistency**: Is answer geometrically consistent with shown spatial relationships?
- **Orientation Accuracy**: Are orientations and positions correctly described?
""",
    },
    "perception": {
        "ra_criteria": """
## Perception Reasoning - RA Criteria
- **Object Highlighting**: Are relevant objects properly highlighted or marked?
- **Scene Understanding**: Does visualization show correct scene interpretation?
- **Feature Emphasis**: Are key perceptual features emphasized?
- **Visual Clarity**: Is the visualization clear for perception tasks?
""",
        "al_criteria": """
## Perception Reasoning - AL Criteria
- **Object Identification**: Are identified objects correct?
- **Scene Accuracy**: Does answer accurately describe the scene?
- **Attribute Correctness**: Are object attributes (color, shape, position) correct?
- **Visual Evidence**: Is answer supported by visible evidence in images?
""",
    },
}

# Legacy task name mappings for backward compatibility
TASK_SPECIFIC_ADDITIONS = {
    "illusionbench": {
        "ra_addition": TASK_CATEGORY_CRITERIA["perception"]["ra_criteria"],
        "al_addition": TASK_CATEGORY_CRITERIA["perception"]["al_criteria"],
    },
    "chartqa": {
        "ra_addition": TASK_CATEGORY_CRITERIA["chart_table"]["ra_criteria"],
        "al_addition": TASK_CATEGORY_CRITERIA["chart_table"]["al_criteria"],
    },
    "mathvista": {
        "ra_addition": TASK_CATEGORY_CRITERIA["mathematical"]["ra_criteria"],
        "al_addition": TASK_CATEGORY_CRITERIA["mathematical"]["al_criteria"],
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_reasoning_visual_prompt(
    generation_prompt: str,
    question: str,
    task: str,
    task_category: str = None,
) -> str:
    """
    Get RA evaluation prompt.
    
    Args:
        generation_prompt: Prompt used to generate images
        question: Original question text
        task: Task name
        task_category: Task category for customized prompts
                      ["real_world", "mathematical", "stem", "puzzles", 
                       "chart_table", "spatial", "perception"]
        
    Returns:
        Formatted prompt string
    """
    prompt = REASONING_VISUAL_PROMPT.format(
        task=task,
        question=question,
        generation_prompt=generation_prompt,
    )
    
    # Priority 1: Use task_category if explicitly provided
    if task_category and task_category in TASK_CATEGORY_CRITERIA:
        prompt += "\n" + TASK_CATEGORY_CRITERIA[task_category]["ra_criteria"]
        return prompt
    
    # Priority 2: Auto-detect from task name (backward compatibility)
    for key in TASK_SPECIFIC_ADDITIONS:
        if key in task.lower():
            prompt += "\n" + TASK_SPECIFIC_ADDITIONS[key]["ra_addition"]
            return prompt
    
    return prompt


def get_answer_visual_alignment_prompt(
    answer: str,
    question: str,
    task: str,
    task_category: str = None,
) -> str:
    """
    Get AL evaluation prompt.
    
    Args:
        answer: Final answer text
        question: Original question text
        task: Task name
        task_category: Task category for customized prompts
                      ["real_world", "mathematical", "stem", "puzzles", 
                       "chart_table", "spatial", "perception"]
        
    Returns:
        Formatted prompt string
    """
    prompt = ANSWER_VISUAL_ALIGNMENT_PROMPT.format(
        task=task,
        question=question,
        answer=answer,
    )
    
    # Priority 1: Use task_category if explicitly provided
    if task_category and task_category in TASK_CATEGORY_CRITERIA:
        prompt += "\n" + TASK_CATEGORY_CRITERIA[task_category]["al_criteria"]
        return prompt
    
    # Priority 2: Auto-detect from task name (backward compatibility)
    for key in TASK_SPECIFIC_ADDITIONS:
        if key in task.lower():
            prompt += "\n" + TASK_SPECIFIC_ADDITIONS[key]["al_addition"]
            return prompt
    
    return prompt
