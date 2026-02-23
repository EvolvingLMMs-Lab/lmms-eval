# LMMs-Eval v0.8

## Overview

v0.8 focuses on one core goal: **unified model evaluation** across both understanding and generation tasks for image, video, and audio modalities.

Previous releases established a strong foundation for multimodal understanding benchmarks. In v0.8, we extend the center of gravity toward a unified evaluation protocol that can compare:

- modality-specific generation models (for example, image-only or video-only generators), and
- unified multimodal models that must handle understanding + generation in one system.

This document is a version roadmap and survey plan. It defines the scope, benchmark priorities, model tracks, and implementation milestones for the v0.8 cycle.

---

## Table of Contents

- [1. v0.8 Theme and Scope](#1-v08-theme-and-scope)
- [2. Evaluation Matrix for Unified Models](#2-evaluation-matrix-for-unified-models)
- [3. Model Tracks to Survey](#3-model-tracks-to-survey)
- [4. Core Benchmarks for Image Generation](#4-core-benchmarks-for-image-generation)
- [5. Core Benchmarks for Video Generation](#5-core-benchmarks-for-video-generation)
- [6. Unified-Eval Benchmarks (Cross-Modal)](#6-unified-eval-benchmarks-cross-modal)
- [7. v0.8 Deliverables in lmms-eval](#7-v08-deliverables-in-lmms-eval)
- [8. Milestones](#8-milestones)
- [9. Risks and Mitigations](#9-risks-and-mitigations)
- [10. Success Criteria](#10-success-criteria)

---

## 1. v0.8 Theme and Scope

### 1.1 Primary Theme

**Unified model evaluation** means we evaluate a model not only on "can it understand" or "can it generate," but on whether it can do both reliably, consistently, and measurably under one evaluation harness.

### 1.2 Scope in v0.8

v0.8 prioritizes:

1. Survey and benchmark alignment for **image generation** and **video generation**.
2. Explicit split between:
   - **Specialized models** (single-track dominant), and
   - **Unified models** (understanding + generation, often cross-modal).
3. A benchmark set with approximately 5 core benchmarks per generation track.
4. A first unified scorecard design that can combine understanding and generation metrics in one report.

### 1.3 Out of Scope (v0.8)

- Full production leaderboard with public hosting.
- Perfect metric unification across all generation modalities.
- Complete coverage for every available generation model family.

---

## 2. Evaluation Matrix for Unified Models

We define a 2x2 matrix as the base planning abstraction.

| Capability | Image | Video |
|---|---|---|
| Understanding | Existing lmms-eval strengths (MME/MMMU/MMVet/etc.) | Existing lmms-eval strengths (VideoMME/Video-MMMU/etc.) |
| Generation | v0.8 survey + benchmark integration target | v0.8 survey + benchmark integration target |

v0.8 objective is to make this matrix operational, so one report can answer:

- Is this model strong only on generation?
- Is this model strong only on understanding?
- Is this model balanced as a unified system?

---

## 3. Model Tracks to Survey

In each track, we survey both specialized and unified families.

### 3.1 Image Generation Track

#### Specialized image-generation model families

- `FLUX` family (recommend pinning concrete variants, for example FLUX.2).
- `Z-Image` family.
- `Qwen-Image` family.

#### Unified model candidates with generation ability

- `BAGEL` (ByteDance Seed).
- `Nano Banana` / Gemini image generation line.

### 3.2 Video Generation Track

#### Specialized video-generation model families

- `Wan` family (recommend pinning concrete variants, for example Wan2.1).
- `LTX` family (recommend pinning concrete variants, for example LTX-2).
- `HunyuanVideo` family (recommend pinning concrete variants, for example HunyuanVideo-1.5).

#### Unified model candidates with video generation ability

- `BAGEL` (as unified baseline candidate across image/video understanding and generation).

Notes for naming discipline in v0.8 docs and scripts:

- Prefer canonical names with explicit version tags when available.
- Avoid ambiguous family-only names in benchmark reports.

---

## 4. Core Benchmarks for Image Generation

v0.8 baseline recommendation is to track the following five benchmark families:

1. **HEIM**
   - Holistic text-to-image evaluation across broad dimensions.
   - Ref: https://crfm.stanford.edu/helm/heim/latest/

2. **T2I-CompBench++**
   - Compositional consistency (attributes, relations, counting, structure).
   - Ref: https://karine-h.github.io/T2I-CompBench/

3. **GenAI-Bench (Gecko)**
   - Prompt-to-rating alignment and metric reliability analysis.
   - Ref: https://arxiv.org/abs/2404.16820

4. **ImagenWorld**
   - Open-ended and real-world task coverage (generation/editing/composition).
   - Ref: https://openreview.net/forum?id=bld9g6jFh9

5. **GenArena**
   - Pairwise evaluation protocol for robust comparative ranking.
   - Ref: https://arxiv.org/html/2602.06013v1

Selection rationale in v0.8:

- Cover both quality and compositional faithfulness.
- Include both absolute and pairwise evaluation paradigms.
- Include at least one benchmark focused on evaluation methodology itself.

---

## 5. Core Benchmarks for Video Generation

v0.8 baseline recommendation is to track the following five benchmark families:

1. **VBench / VBench++**
   - De facto multi-dimension video generation evaluation stack.
   - Ref: https://vchitect.github.io/VBench-project/

2. **VBench-2.0**
   - Stronger focus on intrinsic faithfulness and world consistency.
   - Ref: https://arxiv.org/pdf/2503.21755

3. **EvalCrafter**
   - Objective metric suite for visual quality, alignment, motion, and consistency.
   - Ref: https://evalcrafter.github.io/

4. **ChronoMagic-Bench**
   - Temporal metamorphosis and long-horizon consistency focus.
   - Ref: https://arxiv.org/abs/2406.18522

5. **Video-Bench**
   - Human preference aligned evaluation reference.
   - Ref: https://github.com/Video-Bench/Video-Bench

Selection rationale in v0.8:

- Balance automated scores with human preference alignment.
- Emphasize temporal consistency and physical plausibility.
- Keep one benchmark with broad adoption as an anchor.

---

## 6. Unified-Eval Benchmarks (Cross-Modal)

To avoid evaluating unified models with only modality-specific criteria, v0.8 also prioritizes unified evaluation benchmarks:

1. **MME-Unify**
   - Unified MLLM benchmark for understanding + generation + mixed tasks.
   - Ref: https://github.com/MME-Benchmarks/MME-Unify

2. **RealUnify**
   - Measures synergy between understanding and generation instead of isolated capability.
   - Ref: https://arxiv.org/abs/2509.24897

3. **UEval**
   - Interleaved multimodal output evaluation with detailed rubric criteria.
   - Ref: https://arxiv.org/abs/2601.22155

Unified-eval role in v0.8:

- Specialized model benchmarks tell us peak single-task performance.
- Unified benchmarks tell us whether one model can coordinate multiple capabilities coherently.

Both are required for a fair frontier-model evaluation story.

---

## 7. v0.8 Deliverables in lmms-eval

### 7.1 Documentation Deliverables

- v0.8 version documentation (this file).
- Survey tables for model families and benchmark families.
- Metric glossary for understanding vs. generation reporting.

### 7.2 Evaluation Pipeline Deliverables

1. **Task Inventory Refresh**
   - Tag existing tasks by: modality, understanding/generation, and benchmark family.

2. **Generation Track Onboarding Plan**
   - Define integration sequence for image generation benchmarks first, then video generation benchmarks.

3. **Unified Scorecard Draft**
   - Report sections:
     - Understanding score block
     - Generation score block
     - Unified capability block

4. **Model Naming and Versioning Convention**
   - Require explicit model variant naming in reports and scripts.

### 7.3 Suggested Result Schema Additions

Potential metadata fields for v0.8 experiments:

- `capability_type`: `understanding` | `generation` | `unified`
- `modality_track`: `image` | `video` | `audio` | `cross_modal`
- `benchmark_family`: canonical benchmark family name
- `model_variant`: explicit evaluated model variant

---

## 8. Milestones

### Milestone 1 - Survey Freeze

- Finalize model families per track.
- Finalize 5 core benchmarks for image generation.
- Finalize 5 core benchmarks for video generation.

### Milestone 2 - Integration Draft

- Add/standardize benchmark task definitions where needed.
- Define initial unified scorecard output format.

### Milestone 3 - Pilot Runs

- Run pilot evaluations on representative specialized models.
- Run pilot evaluations on representative unified models.
- Compare score stability and reporting clarity.

### Milestone 4 - v0.8 Wrap-Up

- Publish consolidated report template.
- Document known gaps and v0.9 carry-over items.

---

## 9. Risks and Mitigations

1. **Risk: Metric mismatch across benchmarks**
   - Mitigation: keep per-benchmark metrics raw, add a normalized summary layer rather than forcing one metric.

2. **Risk: Ambiguous model naming causes irreproducible results**
   - Mitigation: enforce explicit model variant and release tag in configs/results.

3. **Risk: Unified models look weak due to specialized benchmark bias**
   - Mitigation: include dedicated unified benchmarks (MME-Unify/RealUnify/UEval) in the core set.

4. **Risk: Over-expanding scope in one version**
   - Mitigation: keep v0.8 focused on image/video generation survey + unified scorecard draft, defer full leaderboard work.

---

## 10. Success Criteria

v0.8 is successful if we can ship:

1. A reproducible benchmark set for image/video generation with clear rationale.
2. A clean specialized-vs-unified comparison protocol.
3. A first unified evaluation report format that can be executed in lmms-eval workflows.
4. Documentation that lets contributors extend this direction in v0.9 without re-defining the framework.

---

## Appendix A: Initial Model Family Checklist

### Image generation

- FLUX family
- Z-Image family
- Qwen-Image family

### Video generation

- Wan family
- LTX family
- HunyuanVideo family

### Unified model candidates

- BAGEL (ByteDance Seed)
- Nano Banana / Gemini image-generation line

This checklist is intentionally small for v0.8 so integration quality stays high.
