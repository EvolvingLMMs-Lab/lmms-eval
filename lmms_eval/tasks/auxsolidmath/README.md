# AuxSolidMath (立体几何辅助线)

## 1. Task Overview

AuxSolidMath evaluates solid geometry (立体几何) problem-solving abilities with emphasis on auxiliary line construction. The task involves:
- Understanding 3D geometric figures (prisms, pyramids, spheres, etc.)
- Identifying appropriate auxiliary constructions
- Drawing perpendiculars, connecting points, finding intersections
- Computing distances, angles, volumes in 3D space

## 2. Task Files

| Task | YAML File | Description |
|------|-----------|-------------|
| Direct (Easy) | `auxsolidmath_easy.yaml` | Single-stage solid geometry problems |
| Visual CoT (Easy) | `auxsolidmath_easy_visual_cot.yaml` | Two-stage with auxiliary construction generation |

## 3. Prompt Design

### 3.1 Direct Generation

```
You are given a solid geometry problem with a 3D diagram.

Problem: {question}

Instructions:
1. First, carefully analyze the 3D diagram and identify what auxiliary lines (辅助线)
   need to be drawn to solve this problem. Common auxiliary constructions in solid
   geometry include:
   - Connecting points to form line segments
   - Drawing perpendiculars from a point to a plane or line
   - Finding midpoints and connecting them
   - Extending lines to find intersections
   - Drawing parallel lines through specific points
   - Constructing cross-sections

2. Clearly state which auxiliary lines you will draw and why they are helpful. For example:
   - "Connect point A to point B to form segment AB"
   - "Draw a perpendicular from point P to plane ABC, with foot H"
   - "Take the midpoint M of edge AB, connect M to C"
   - "Extend line DE to intersect plane ABC at point F"

3. After describing the auxiliary lines, provide a step-by-step solution using these
   auxiliary constructions.

4. Show all intermediate calculations and reasoning, including:
   - Distance calculations
   - Angle calculations
   - Volume/area calculations if needed

5. State the final answer clearly.

Please think step by step, starting with the auxiliary line construction.
```

### 3.2 Visual CoT (Two-Stage)

Uses `[GEN_PROMPT]...[/GEN_PROMPT][QUESTION]...[/QUESTION]` format.

#### Stage 1: Generation Prompt

```
You are given a solid geometry problem with a 3D diagram. Analyze the problem
and create an enhanced version of the SAME diagram with auxiliary constructions added.

Problem: {question}

Instructions:
1. KEEP all original elements exactly as they are (all points, edges, faces, labels)
2. Analyze what auxiliary constructions would help solve this problem
3. ADD auxiliary lines in a different color (e.g., red or dashed lines). Common
   auxiliary constructions include:
   - Perpendiculars from a point to a plane or line
   - Line segments connecting specific points
   - Midpoints of edges with connections
   - Extended lines to find intersections
   - Parallel lines through specific points
   - Cross-sections of the solid
4. Label any new points you add (use letters not already in the diagram)
5. The final diagram should look like the original 3D figure with extra auxiliary
   lines drawn on top

Generate an enhanced 3D diagram that preserves the original and adds helpful
auxiliary constructions.
```

#### Stage 2: Question Prompt

```
You are given a solid geometry problem.

Problem: {question}

You are given TWO images:
1) ORIGINAL DIAGRAM: The 3D solid geometry figure as given
2) AUXILIARY DIAGRAM: The same figure with auxiliary constructions (extra lines)
   added to help solve the problem

Instructions:
1. Look at the auxiliary diagram to see what constructions were added
   (perpendiculars, connecting segments, midpoints, etc.)
2. Identify the geometric relationships established by these auxiliary lines
3. Use these constructions to set up your solution approach
4. Apply relevant theorems (Pythagorean theorem in 3D, properties of perpendiculars,
   volume formulas, etc.)
5. Show your step-by-step solution with clear calculations
6. State the final numerical answer

Solve this problem step by step using the auxiliary constructions.
```

## 4. Common Auxiliary Constructions (辅助线)

| Chinese Term | English | Description |
|--------------|---------|-------------|
| 连接两点 | Connect points | Form line segment between two points |
| 过点作垂线 | Draw perpendicular | Perpendicular from point to plane/line |
| 取中点 | Find midpoint | Midpoint of edge with connections |
| 延长线 | Extend line | Extend to find intersection |
| 作平行线 | Draw parallel | Parallel line through specific point |
| 作截面 | Construct cross-section | Plane cutting through solid |

## 5. Key Functions

| Function | Purpose |
|----------|---------|
| `auxsolidmath_doc_to_text` | Build prompt for direct generation |
| `auxsolidmath_doc_to_text_visual_cot` | Build two-stage prompt for Visual CoT |
| `auxsolidmath_doc_to_visual` | Extract original image from document |
| `auxsolidmath_doc_to_target` | Get ground truth answer |
| `auxsolidmath_process_results` | LLM Judge evaluation |
| `auxsolidmath_aggregate` | Aggregate accuracy scores |

## 6. Evaluation Metrics

- **auxsolidmath_text_acc**: LLM Judge (GPT-5.1) evaluates:
  - Reasoning rigorousness (no major logical gaps)
  - Conclusion correctness (final answer matches ground truth)
  - Both must be satisfied for text_ok=1

For **CALCULATION** problems: Numeric result must match (radicals/π allowed)
For **PROVING** problems: The claim must be validly established

## 7. Data Fields

| Field | Description |
|-------|-------------|
| `question` | Problem statement |
| `original_image` | 3D diagram as PIL Image |
| `answer` | Ground truth answer |
| `auxiliary_line_description` | Description of expected auxiliary constructions |

## 8. Running Evaluation

```bash
# Direct generation
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=/path/to/model \
    --tasks auxsolidmath_easy \
    --batch_size 4 \
    --output_path ./logs/

# Visual CoT
python -m lmms_eval \
    --model azure_trapi_visual_cot \
    --model_args save_intermediate=true \
    --tasks auxsolidmath_easy_visual_cot \
    --batch_size 1 \
    --output_path ./logs/
```

## 9. Environment Variables

For LLM Judge evaluation:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export JUDGE_DEPLOYMENT="gpt-5.1"  # or other deployment name
```
