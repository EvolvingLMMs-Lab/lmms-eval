# AIME

## Paper
The American Invitational Mathematics Examination (AIME) is a selective and prestigious 15-question 3-hour test given to high school students who qualify based on their AMC 10 or AMC 12 scores. All problems have integer answers between 0 and 999 inclusive. Questions increase in difficulty as the exam progresses.

The AIME dataset evaluates mathematical problem-solving capabilities on competition-level mathematics problems.

Homepage: https://huggingface.co/datasets/simplescaling/aime_nofigures

## Dataset

This implementation includes both:
- `aime_nofigures`: AIME problems without figures/diagrams
- `aime_figures`: AIME problems with figures/diagrams

The dataset uses problems from AIME competitions, formatted for language model evaluation.

## Groups and Tasks

#### Groups

- `math_word_problems`

#### Tasks

- `aime_nofigures`: AIME problems without figures
- `aime_figures`: AIME problems with figures
- `aime24_nofigures`: AIME 2024 problems without figures
- `aime24_figures`: AIME 2024 problems with figures
- `aime25_nofigures`: AIME 2025 problems without figures
- Various aggregated versions (agg8, agg64) for multiple sampling

### Evaluation

The evaluation checks if the model's output matches the correct integer answer (0-999). The implementation includes:
- Answer extraction from model outputs
- Support for boxed answers (e.g., `\boxed{123}`)
- Optional GPT-4o-mini based answer extraction for complex formats
- Coverage and majority voting metrics for aggregated tasks

### Environment Variables

- `PROCESSOR=gpt-4o-mini`: Use GPT-4o-mini for answer extraction
- `PROMPTSTEP`: Add thinking steps prompt
- `PROMPTTOKEN`: Add thinking tokens prompt
- `PROMPTLONG`: Add long thinking prompt
- `PROMPTSHORT`: Add short thinking prompt

### Checklist

- [ ] Is in Eval-harness v1.0?
- [ ] Has been checked for regression from v1.0?
- [ ] Has been checked for equivalence with original paper methodology?
- [ ] "Main" checked variant clearly denoted?