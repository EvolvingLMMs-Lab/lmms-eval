# Geometry3K

## 1. Task Overview

Geometry3K evaluates plane geometry problem-solving abilities. The task involves:
- Understanding geometry diagrams (triangles, circles, quadrilaterals, etc.)
- Identifying relevant geometric relationships
- Applying theorems to compute answers
- Constructing auxiliary lines when needed

## 2. Task Files

| Task | YAML File | Description |
|------|-----------|-------------|
| Direct | `geometry3k.yaml` | Single-stage geometry problem solving |
| Visual CoT | `geometry3k_visual_cot.yaml` | Two-stage with auxiliary construction generation |

## 3. Prompt Design

### 3.1 Direct Generation

```
{problem}

Instructions:
1. Carefully analyze the geometry diagram shown above.
2. Read the problem statement and identify what needs to be found.
3. Show your step-by-step solution with clear reasoning.
4. Include all intermediate calculations.
5. State the final answer clearly at the end.

Please solve this problem step by step.
```

### 3.2 Visual CoT (Two-Stage)

Uses `[GEN_PROMPT]...[/GEN_PROMPT][QUESTION]...[/QUESTION]` format.

#### Stage 1: Generation Prompt

```
You are given a geometry problem with a diagram. Analyze the problem and create
an enhanced version of the SAME diagram with auxiliary constructions added.

Problem: {problem}

Instructions:
1. KEEP all original elements exactly as they are (all points, lines, labels, and measurements)
2. Analyze what auxiliary constructions would help solve this problem
3. ADD auxiliary lines in a different color (e.g., red or dashed lines):
   - Perpendicular lines from center to chords
   - Extended lines if needed
   - Angle bisectors, midpoints, or other helpful constructions
4. Label any new points you add (use letters not already in the diagram)
5. The final diagram should look like the original with extra auxiliary lines drawn on top

Generate an enhanced diagram that preserves the original and adds helpful auxiliary constructions.
```

#### Stage 2: Question Prompt

```
{problem}

You are given TWO images:
1) ORIGINAL DIAGRAM: The geometry problem as given
2) AUXILIARY DIAGRAM: The same diagram with auxiliary constructions (extra lines) added to help solve the problem

Instructions:
1. Look at the auxiliary diagram to see what constructions were added
2. Use these auxiliary lines to identify key geometric relationships (perpendiculars, congruent segments, etc.)
3. Apply relevant theorems (Pythagorean theorem, chord properties, etc.)
4. Show your step-by-step solution with clear calculations
5. State the final numerical answer

Solve this problem step by step.
```

## 4. Key Functions

| Function | Purpose |
|----------|---------|
| `geometry3k_doc_to_text` | Build prompt for direct generation |
| `geometry3k_doc_to_text_visual_cot` | Build two-stage prompt for Visual CoT |
| `geometry3k_doc_to_visual` | Extract image from document |
| `geometry3k_doc_to_target` | Get ground truth answer |
| `geometry3k_process_results` | LLM Judge evaluation with Azure TRAPI |
| `geometry3k_aggregate` | Aggregate accuracy scores |

## 5. Evaluation Metrics

- **geometry3k_accuracy**: LLM Judge (GPT-5.1) evaluates if the candidate's final answer is mathematically equivalent to the ground truth

The Judge considers:
- Numeric equivalence (allows minor rounding differences)
- Algebraic expressions (e.g., "2√221" = "2*sqrt(221)")
- Fractions (e.g., "1/2" = "0.5")
- LaTeX formatting differences
- Unit differences (if values match)

## 6. Stage 1 Original Image Handling

Visual CoT models receive the original geometry diagram in Stage 1:
- The model analyzes the diagram and problem together
- Generates an enhanced diagram with auxiliary constructions overlaid
- Stage 2 receives both original and auxiliary diagram for solving

## 7. Running Evaluation

```bash
# Direct generation
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=/path/to/model \
    --tasks geometry3k \
    --batch_size 4 \
    --output_path ./logs/

# Visual CoT
python -m lmms_eval \
    --model azure_trapi_visual_cot \
    --model_args save_intermediate=true \
    --tasks geometry3k_visual_cot \
    --batch_size 1 \
    --output_path ./logs/
```

## 8. Environment Variables

For LLM Judge evaluation:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export JUDGE_DEPLOYMENT="gpt-5.1"  # or other deployment name
```
