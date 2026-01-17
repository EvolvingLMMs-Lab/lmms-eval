# VisualPuzzles

## 1. Task Overview

VisualPuzzles evaluates visual reasoning abilities across 5 reasoning categories:
- **Algorithmic**: Numerical sequences, patterns, computational rules
- **Analogical**: Transformation relationships (A:B :: C:?)
- **Deductive**: Logical inference from premises
- **Inductive**: Pattern recognition and generalization
- **Spatial**: 3D visualization, rotations, folding

## 2. Task Files

### 2.1 Direct Generation Tasks

| Category | YAML File |
|----------|-----------|
| All Categories | `VisualPuzzles.yaml` |
| Algorithmic | `VisualPuzzles_algorithmic.yaml` |
| Analogical | `VisualPuzzles_analogical.yaml` |
| Deductive | `VisualPuzzles_deductive.yaml` |
| Inductive | `VisualPuzzles_inductive.yaml` |
| Spatial | `VisualPuzzles_spatial.yaml` |

### 2.2 Visual CoT Tasks

| Category | YAML File | doc_to_text Function |
|----------|-----------|----------------------|
| All Categories | `VisualPuzzles_visual_cot.yaml` | - |
| Algorithmic | `VisualPuzzles_algorithmic_visual_cot.yaml` | `VisualPuzzles_doc_to_text_visual_cot_algorithmic` |
| Analogical | `VisualPuzzles_analogical_visual_cot.yaml` | `VisualPuzzles_doc_to_text_visual_cot_analogical` |
| Deductive | `VisualPuzzles_deductive_visual_cot.yaml` | `VisualPuzzles_doc_to_text_visual_cot_deductive` |
| Inductive | `VisualPuzzles_inductive_visual_cot.yaml` | `VisualPuzzles_doc_to_text_visual_cot_inductive` |
| Spatial | `VisualPuzzles_spatial_visual_cot.yaml` | `VisualPuzzles_doc_to_text_visual_cot_spatial` |

## 3. Prompt Design

### 3.1 Direct Generation

```
Question: {question}
Options:
(A) {option_A}
(B) {option_B}
(C) {option_C}
(D) {option_D}
Solve the multiple-choice question and then answer with the option letter from
the given choices. The last line of your response should be of the following
format: 'Answer: $LETTER' (without quotes) where LETTER is one of options.
Think step by step before answering.
```

### 3.2 Visual CoT (Two-Stage)

Uses `[GEN_PROMPT]...[/GEN_PROMPT][QUESTION]...[/QUESTION]` format.

#### Stage 1: Generation Prompts (Category-Specific)

**Algorithmic:**
```
You are given an algorithmic reasoning puzzle. Analyze the puzzle and create a helpful visualization.

{question with options}

Your task:
1. Identify any numerical sequences, patterns, or computational rules in the puzzle
2. Create a diagram that clearly shows:
   - The step-by-step computation or transformation process
   - Arrows or annotations showing how numbers/symbols change
   - The mathematical relationship or formula discovered
   - Highlighted patterns (e.g., +2, ×3, alternating, etc.)
3. Label each step of the algorithm clearly

Generate a clear diagram that reveals the underlying algorithmic pattern.
```

**Analogical:**
```
You are given an analogical reasoning puzzle (A is to B as C is to ?).

{question with options}

Your task:
1. Identify the transformation relationship between the first pair of elements
2. Create a diagram that clearly shows:
   - What changes occur (rotation, reflection, color change, size change, addition/removal of elements)
   - Arrows indicating the direction and type of transformation
   - Labels describing each transformation (e.g., "rotate 90°", "invert colors", "add dot")
   - Apply the same transformation to show what the answer should look like
3. Make the analogy relationship visually explicit

Generate a diagram that reveals the transformation pattern between pairs.
```

**Deductive:**
```
You are given a deductive reasoning puzzle that requires logical inference.

{question with options}

Your task:
1. Identify the given premises, rules, or constraints in the puzzle
2. Create a diagram that clearly shows:
   - All given conditions/rules listed clearly
   - A logical flowchart or inference chain
   - Step-by-step deduction from premises to conclusion
   - Elimination of incorrect possibilities
   - The logical path leading to the answer
3. Use arrows to show the deduction flow

Generate a logical inference diagram that traces the reasoning path.
```

**Inductive:**
```
You are given an inductive reasoning puzzle that requires pattern recognition.

{question with options}

Your task:
1. Observe the sequence of examples and identify the underlying pattern
2. Create a diagram that clearly shows:
   - The repeating elements or motifs highlighted/circled
   - The progression rule (what changes from one step to the next)
   - Annotations showing the pattern cycle or growth rule
   - A prediction of what comes next based on the pattern
   - Color-coding or numbering to show pattern repetition
3. Make the inductive pattern visually obvious

Generate a diagram that highlights the repeating pattern and predicts the next element.
```

**Spatial:**
```
You are given a spatial reasoning puzzle involving 3D visualization or transformations.

{question with options}

Your task:
1. Analyze the spatial transformation required (rotation, folding, unfolding, different viewpoint)
2. Create a diagram that clearly shows:
   - The object from multiple angles if rotation is involved
   - Step-by-step folding/unfolding process if applicable
   - Arrows indicating rotation direction and degree
   - Reference points or markers to track orientation
   - The resulting shape after transformation
3. Add axis lines or reference frames to clarify spatial orientation

Generate a multi-view or step-by-step transformation diagram.
```

#### Stage 2: Question Prompts

```
{question with options}

You are given TWO images:
1) ORIGINAL PUZZLE: The {category} reasoning puzzle
2) AUXILIARY DIAGRAM: A visualization showing the {category-specific description}

Use the auxiliary diagram to understand the {reasoning type}, then select the correct answer.
Answer with the option letter (A, B, C, or D) directly.
```

## 4. Design Note: Options in Generation Prompt

The options (A, B, C, D) appear in the generation prompt because they are **part of the question itself**, not answer candidates being leaked. The model needs to see the question context to understand what type of puzzle it's solving and generate an appropriate auxiliary visualization.

This is different from information leakage where answer candidates (like category names in classification tasks) are inappropriately revealed.

## 5. Key Functions

| Function | Purpose |
|----------|---------|
| `VisualPuzzles_doc_to_text` | Build prompt for direct generation |
| `VisualPuzzles_doc_to_text_visual_cot_{category}` | Build two-stage prompt for Visual CoT |
| `VisualPuzzles_doc_to_visual` | Extract image from document |
| `VisualPuzzles_process_result` | Process results with category metrics |
| `VisualPuzzles_process_result_simple` | Simplified processing for single-category tasks |
| `VisualPuzzles_aggregate_results` | Aggregate category-wise scores |
| `parse_response` | Extract answer letter from model response |

## 6. Evaluation Metrics

- **Accuracy**: Exact match of predicted option letter (A/B/C/D) with ground truth
- **Category-wise Accuracy**: Accuracy aggregated per reasoning category

## 7. Running Evaluation

```bash
# Direct generation (single category)
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=/path/to/model \
    --tasks VisualPuzzles_algorithmic \
    --batch_size 4 \
    --output_path ./logs/

# Visual CoT (single category)
python -m lmms_eval \
    --model azure_trapi_visual_cot \
    --model_args save_intermediate=true \
    --tasks VisualPuzzles_algorithmic_visual_cot \
    --batch_size 1 \
    --output_path ./logs/
```
