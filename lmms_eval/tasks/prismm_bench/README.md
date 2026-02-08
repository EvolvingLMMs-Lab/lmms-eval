# PRISMM-Bench

## Task Description

PRISMM-Bench is a comprehensive benchmark for evaluating multimodal language models on their ability to identify and resolve inconsistencies in scientific papers. It aims at assessing the current state of employing LMMs as automated peer-review assistants.

See the original paper: [https://arxiv.org/abs/2510.16505](https://arxiv.org/abs/2510.16505)

## Installation

To run the PRISMM-Bench tasks, you need to install the required dependencies. The benchmark requires `poppler` to be installed on your system for PDF processing.

### System Dependencies

#### Linux

**Ubuntu/Debian**
```bash
sudo apt-get install poppler-utils
```

**Fedora/RHEL**
```bash
sudo dnf install poppler
```

**OpenSUSE**
```bash
sudo zypper install poppler-tools
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/) and add to PATH.

### Python Dependencies

Install the lmms-eval package with the prismmbench extras:

```bash
pip install -e ".[prismmbench]"
```

Or using uv:

```bash
uv pip install -e ".[prismmbench]"
```

## Overview of Tasks

PRISMM-Bench contains 7 different task variants across 3 main task types:

### 1. Identification Tasks (3 variants)

**Task Type:** Identify inconsistencies in scientific papers

#### `prismm_bench_identification`
- **Description:** Given a scientific paper with extracted text parts and corresponding content images, identify which answer option correctly describes an inconsistency.
- **Input:** Text context from paper parts + images of those parts
- **Output:** Single letter (A, B, C, or D) indicating the correct answer
- **Use Case:** Tests model's ability to find inconsistencies when provided with both text and visual context

#### `prismm_bench_identification_whole_page`
- **Description:** Identify inconsistencies using full page images of the PDF instead of extracted parts
- **Input:** Full page screenshots + question + options (no text context)
- **Output:** Single letter (A, B, C, or D) indicating the correct answer
- **Use Case:** Tests model's ability to analyze entire pages without pre-extracted text, relying on visual document understanding

#### `prismm_bench_identification_whole_doc`
- **Description:** Identify inconsistencies using concatenated/tiled images of the entire document
- **Input:** Up to 5 concatenated/tiled pages + question + options (no text context)
- **Output:** Single letter (A, B, C, or D) indicating the correct answer
- **Use Case:** Tests model's ability to find document-level inconsistencies across multiple pages viewed simultaneously

### 2. Remedy Tasks (3 variants)

**Task Type:** Determine actions to resolve inconsistencies

#### `prismm_bench_remedy`
- **Description:** Given an inconsistency in a scientific paper, identify which action needs to be taken to resolve it
- **Input:** Text context from paper parts + images of those parts
- **Output:** Single letter (A, B, C, or D) indicating the correct remedy action
- **Use Case:** Tests model's ability to suggest corrections when provided with both text and visual context

#### `prismm_bench_remedy_whole_page`
- **Description:** Determine remedy actions using full page images instead of extracted parts
- **Input:** Full page screenshots + question + options (no text context)
- **Output:** Single letter (A, B, C, or D) indicating the correct remedy action
- **Use Case:** Tests model's ability to suggest fixes based on entire page analysis

#### `prismm_bench_remedy_whole_doc`
- **Description:** Determine remedy actions using concatenated/tiled images of the entire document
- **Input:** Up to 5 concatenated/tiled pages + question + options (no text context)
- **Output:** Single letter (A, B, C, or D) indicating the correct remedy action
- **Use Case:** Tests model's ability to suggest document-level fixes across multiple pages

### 3. Pair-Match Task (1 variant)

#### `prismm_bench_pair_match`
- **Description:** Given one part of a paper (text or image), find which other part creates an inconsistency when combined
- **Input:** Query (text or image) + multiple candidate parts (images)
- **Output:** Single letter (A, B, C, or D) indicating which candidate creates the inconsistency
- **Use Case:** Tests model's ability to match related content and identify which combinations produce inconsistencies
- **Note:** This task includes a filter to only use samples where pair_match is available

## Dataset

The benchmark uses the PRISMM-Bench dataset from Hugging Face:
- **Dataset ID:** `daluggas/PRISMM-Bench`
- **Splits:** test

## Metrics

All tasks use **Exact Match (EM)** as the evaluation metric:
- **Metric:** exact_match
- **Aggregation:** mean
- **Higher is Better:** true

The model's output is compared against the ground truth answer, with flexible matching:
1. Exact match of uppercase letters (e.g., "A" vs "A")
2. First character match if output is longer (e.g., "A)" vs "A")