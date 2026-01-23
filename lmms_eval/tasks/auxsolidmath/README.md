# AuxSolidMath

AuxSolidMath is a benchmark for evaluating solid geometry reasoning with auxiliary line construction.

## Overview

- **Paper**: [GeoVLMath: Enhancing Geometry Reasoning via Auxiliary Lines](https://arxiv.org/abs/2510.11020)
- **Dataset**: [shasha/AuxSolidMath](https://huggingface.co/datasets/shasha/AuxSolidMath)
- **Size**: 3,018 real-exam solid geometry problems
- **Difficulty Splits**: `test_easy` (150 problems), `test_hard` (152 problems)

## Tasks

| Task | Description |
|------|-------------|
| `auxsolidmath` | Full benchmark (easy + hard) |
| `auxsolidmath_easy` | Easy difficulty subset |
| `auxsolidmath_hard` | Hard difficulty subset |

## Dataset Fields

| Field | Description |
|-------|-------------|
| `id` | Problem identifier |
| `question` | Problem statement |
| `original_image` | 3D solid geometry diagram |
| `auxiliary_line_image` | Diagram with auxiliary constructions |
| `auxiliary_line_description` | Description of auxiliary lines needed |
| `answer` | Ground truth answer |

## Evaluation

Uses string matching with numerical tolerance for answer comparison. Supports:
- Exact string matching
- Numerical comparison with tolerance
- LaTeX expression normalization
