# V*-Bench (Visual Star Benchmark)

## Overview

V*-Bench is a visual question-answering benchmark designed to evaluate multimodal language models' capabilities in visual perception and reasoning. The benchmark focuses on assessing models' ability to accurately identify and reason about visual attributes in images through multiple-choice questions.

## Dataset Details

- **Dataset**: `lmms-lab/vstar-bench`
- **Size**: 191 test samples
- **Format**: Multiple-choice questions with 4 options (A, B, C, D)
- **Modalities**: Image + Text

## Task Categories

The benchmark includes two main categories:

1. **Direct Attributes** (`vstar_bench_direct_attributes`)
   - Questions about direct visual properties such as colors, objects, counts, and characteristics
   - Examples: "What is the color of the glove?", "What is the breed of the dog?", "How many people are in the image?"

2. **Relative Position** (`vstar_bench_relative_position`)
   - Questions about spatial relationships and positioning of objects within images
   - Evaluates understanding of spatial concepts and object relationships

## Evaluation

### Metrics
- **Overall Accuracy**: Percentage of correctly answered questions across all categories
- **Category-specific Accuracy**: Accuracy for each individual category (direct_attributes, relative_position)

### Running the Benchmark

To evaluate a model on V*-Bench:

```bash
# Run the full benchmark
lmms-eval --model <model_name> --tasks vstar_bench --output_path ./results

# Run specific categories
lmms-eval --model <model_name> --tasks vstar_bench_direct_attributes --output_path ./results
lmms-eval --model <model_name> --tasks vstar_bench_relative_position --output_path ./results
```

## Configuration

The benchmark uses the following configuration:
- **Generation Settings**:
  - `max_new_tokens`: 16
  - `temperature`: 0
  - `top_p`: 1.0
  - `num_beams`: 1
  - `do_sample`: false

- **Prompt Template**:
  - Post-prompt: "\nAnswer with the option's letter from the given choices directly."

## Implementation Details

### Answer Extraction
The evaluation system extracts answer letters (A, B, C, or D) from model responses using multiple patterns to handle various response formats:
- Direct letter: "A"
- With punctuation: "A.", "A)", "(A)"
- Full answer format: "Answer: A", "The answer is A"

### Aggregation
Results are aggregated both by category and overall, providing detailed performance metrics for different aspects of visual understanding.

## File Structure

```
vstar_bench/
├── __init__.py
├── README.md
├── _default_template_yaml         # Base configuration
├── vstar_bench.yaml              # Main task configuration
├── vstar_bench_direct_attributes.yaml
├── vstar_bench_relative_position.yaml
└── utils.py                      # Processing and evaluation functions
```

## References

- Dataset: https://huggingface.co/datasets/lmms-lab/vstar-bench