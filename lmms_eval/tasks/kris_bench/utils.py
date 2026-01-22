"""
KRIS-Bench task utils for lmms-eval.

Reference implementation:
  - github_repo/Kris_Bench/metrics_common.py
  - github_repo/Kris_Bench/metrics_view_change.py
  - github_repo/Kris_Bench/metrics_multi_element.py
  - github_repo/Kris_Bench/metrics_temporal_prediction.py
  - github_repo/Kris_Bench/metrics_knowledge.py
  - github_repo/Kris_Bench/utils/prompts.py

This task:
  1) Generates an image per sample (typically edit/generate conditioned on 1+ images).
  2) Scores the generated image with a multimodal judge (OpenAI-compatible endpoint).

Image output (compatible with original Kris_Bench repo convention):
  {KRIS_BENCH_OUTPUT_DIR}/results/{KRIS_BENCH_MODEL_NAME}/{category}/{image_id}.jpg

Environment variables:
  - KRIS_BENCH_DATA_ROOT: path to KRIS_Bench directory (default: "KRIS_Bench")
  - KRIS_BENCH_MODEL_NAME: name used in output path (default: "bagel")
  - KRIS_BENCH_OUTPUT_DIR: base directory for "results/" (default: BAGEL_OUTPUT_IMAGE_DIR or ./logs/kris_bench_results)

Judge server (ImgEdit-style):
  - KRIS_BENCH_API_KEY: API key for judge server (use "EMPTY" for local vLLM)
  - KRIS_BENCH_BASE_URL: base URL (e.g., http://localhost:8000/v1)
  - KRIS_BENCH_EVAL_MODEL_NAME: model used for judging (default: "default" -> auto-detect if /models works)
  - KRIS_BENCH_JUDGE_MODEL_NAME: optional separate judge model
  - KRIS_BENCH_TIMEOUT: API timeout seconds (default: 180)
  - KRIS_BENCH_MAX_RETRIES: retries on transient errors (default: 3)
  - KRIS_BENCH_CALL_DELAY: delay between calls seconds (default: 0.5)
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger as eval_logger
from PIL import Image

# -----------------------------------------------------------------------------
# Score scaling
# -----------------------------------------------------------------------------
#
# The judge returns 1-5 scores (as in the original KRIS-Bench scripts).
# In papers / leaderboards these are typically reported on a 0-100 scale.
# Normalization follows summarize.py: (score - 1) * (100 / (max_score - 1))
# This maps: 1→0, 2→25, 3→50, 4→75, 5→100
#
_MAX_SCORE = 5


def _normalize_score(score: float, max_score: int = _MAX_SCORE) -> float:
    """Normalize a 1..max_score score to a 0..100 scale (matching summarize.py)."""
    if max_score <= 1:
        return 0.0
    return (score - 1) * (100.0 / (max_score - 1))


# -----------------------------------------------------------------------------
# Category groups (derived from the original repo scripts)
# -----------------------------------------------------------------------------

COMMON_CATEGORIES = {
    "count_change",
    "color_change",
    "anomaly_correction",
    "position_movement",
    "size_adjustment",
    "part_completion",
    "multi-instruction_execution",
}
VIEWPOINT_CATEGORY = "viewpoint_change"
MULTI_CATEGORY = "multi-element_composition"
TEMPORAL_CATEGORY = "temporal_prediction"
KNOWLEDGE_CATEGORIES = {
    "abstract_reasoning",
    "mathematics",
    "practical_knowledge",
    "medicine",
    "rule-based_reasoning",
    "biology",
    "geography",
    "chemistry",
    "humanities",
    "physics",
}

# -----------------------------------------------------------------------------
# Knowledge dimension groupings (requested reporting structure)
# -----------------------------------------------------------------------------

FACTUAL_ATTRIBUTE_CATEGORIES = {
    "count_change",
    "color_change",
    "size_adjustment",
    "part_completion",
    "anomaly_correction",
}
FACTUAL_SPATIAL_CATEGORIES = {"position_movement", "viewpoint_change"}

CONCEPTUAL_SOCIAL_CATEGORIES = {"practical_knowledge", "humanities"}
CONCEPTUAL_NATURAL_CATEGORIES = {"biology", "chemistry", "geography", "mathematics", "medicine", "physics"}

PROCEDURAL_LOGICAL_CATEGORIES = {"abstract_reasoning", "rule-based_reasoning"}
PROCEDURAL_INSTR_DECOMP_CATEGORIES = {"multi-instruction_execution", "multi-element_composition"}


def _temporal_target_frame_from_doc(doc) -> Optional[int]:
    """
    For temporal_prediction samples, infer which frame is the target (ground-truth) frame.

    The jsonl uses filenames like: "<id>-<frame>.jpg". We prefer gt_img if present.
    """
    gt = str(doc.get("gt_img") or "").strip()
    if gt:
        m = re.search(r"(\d+)-(\d+)", os.path.basename(gt))
        if m:
            try:
                return int(m.group(2))
            except Exception:
                return None
    ori = doc.get("ori_img") or []
    if isinstance(ori, str):
        ori = [ori]
    frame_numbers: List[int] = []
    for fn in ori:
        m = re.search(r"(\d+)-(\d+)", os.path.basename(str(fn)))
        if m:
            try:
                frame_numbers.append(int(m.group(2)))
            except Exception:
                continue
    missing = {1, 2, 3, 4} - set(frame_numbers)
    if len(missing) == 1:
        return list(missing)[0]
    return None


def _kris_dimension_group(doc) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (dimension_group_id, big_class_id) in snake_case.
    """
    category = str(doc.get("category") or "").strip()
    if not category:
        return None, None

    if category in FACTUAL_ATTRIBUTE_CATEGORIES:
        return "factual_attribute_perception", "factual_knowledge"
    if category in FACTUAL_SPATIAL_CATEGORIES:
        return "factual_spatial_perception", "factual_knowledge"
    if category == TEMPORAL_CATEGORY:
        tgt = _temporal_target_frame_from_doc(doc)
        if tgt == 1:
            return "factual_temporal_reverse_prediction", "factual_knowledge"
        if tgt in (2, 3):
            return "factual_temporal_intermediate_prediction", "factual_knowledge"
        if tgt == 4:
            return "factual_temporal_forward_prediction", "factual_knowledge"
        return "factual_temporal_prediction_unknown", "factual_knowledge"

    if category in CONCEPTUAL_SOCIAL_CATEGORIES:
        return "conceptual_social_science", "conceptual_knowledge"
    if category in CONCEPTUAL_NATURAL_CATEGORIES:
        return "conceptual_natural_science", "conceptual_knowledge"

    if category in PROCEDURAL_LOGICAL_CATEGORIES:
        return "procedural_logical_reasoning", "procedural_knowledge"
    if category in PROCEDURAL_INSTR_DECOMP_CATEGORIES:
        return "procedural_instruction_decomposition", "procedural_knowledge"

    return None, None


# -----------------------------------------------------------------------------
# Prompts (copied from github_repo/Kris_Bench/utils/prompts.py)
# -----------------------------------------------------------------------------

prompt_consist = """
You are a professional digital artist and image evaluation specialist.

You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.

Your Objective:
Your task is to **evaluate the visual consistency between the original and edited images, focusing exclusively on elements that are NOT specified for change in the instruction**. That is, you should only consider whether all non-instructed details remain unchanged. Do **not** penalize or reward any changes that are explicitly required by the instruction.

## Evaluation Scale (1 to 5):
You will assign a **consistency_score** according to the following rules:
- **5 Perfect Consistency**: All non-instruction elements are completely unchanged and visually identical.
- **4 Minor Inconsistency**: Only one very small, non-instruction detail is different (e.g., a tiny accessory, a subtle shadow, or a minor background artifact).
- **3 Noticeable Inconsistency**: One clear non-instruction element is changed (e.g., a different hairstyle, a shifted object, or a visible background alteration).
- **2 Significant Inconsistency**: Two or more non-instruction elements have been noticeably altered.
- **1 Severe Inconsistency**: Most or all major non-instruction details are different (e.g., changed identity, gender, or overall scene layout).

## Guidance:
- First, **identify all elements that the instruction explicitly allows or requires to be changed**. Exclude these from your consistency check.
- For all other elements (e.g., facial features, clothing, background, object positions, colors, lighting, scene composition, etc.), **compare Image B to Image A** and check if they remain visually identical.
- If you observe any change in a non-instruction element, note it and consider its impact on the score.
- If the instruction is vague or ambiguous, make a best-effort factual inference about which elements are intended to change, and treat all others as non-instruction elements.

## Note:
- **Do not penalize changes that are required by the instruction.**
- **Do not reward or penalize the quality or correctness of the instructed change itself** (that is evaluated separately).
- If the edited image introduces new artifacts, objects, or changes to non-instruction elements, this should lower the consistency score.

## Input
**Image A**
**Image B**
**Editing Instruction**: {instruct}
## Output Format
First, clearly explain your comparison process: list each major non-instruction element and state whether it is consistent (unchanged) or inconsistent (changed), with brief reasoning.
Then, provide your evaluation in the following JSON format:
{{
"reasoning": **Compared to original image**, [list of non-instruction elements that changed or remained the same] **in the edited image**. 
"consistency_score": X
}}
"""

prompt_quality = """
You are a professional digital artist and image evaluation specialist.

You will be given:
- **Image A**: a single AI-generated image.

## Objective:
Your task is to **evaluate the perceptual quality** of the image, focusing on:
- **Structural and semantic coherence**
- **Natural appearance**
- **Absence of generation artifacts**

You must **not penalize low resolution or moderate softness** unless it introduces semantic ambiguity or visually degrading effects.

## Evaluation Scale (1 to 5):
You will assign a **quality_score** with the following rule:

- **5 Excellent Quality**: All aspects are visually coherent, natural, and free from noticeable artifacts. Structure, layout, and textures are accurate and consistent.
- **4 Minor Issues**: One small imperfection (e.g., slight texture blending, minor lighting inconsistency).
- **3 Noticeable Artifacts**: One or two clear visual flaws or semantic problems (e.g., extra fingers, minor duplication, slight distortion).
- **2 Structural Degradation**: Multiple distracting errors (e.g., melted hands, warped shapes, unreadable text).
- **1 Severe Errors**: Major structural failures or hallucinations (e.g., broken anatomy, garbled symbols).

## Guidance:
Check the following visual aspects and mark them as ✔ (satisfactory) or ✘ (problematic):
- Structural coherence (e.g., correct anatomy, object shapes, legible text)
- Naturalness (lighting, perspective, shadow logic)
- Artifact-free (no duplication, ghosting, watermarks)
- Texture fidelity (clothing, hair, surfaces not melted or corrupted)
- Optional: Sharpness (only penalize if blur causes semantic loss)
✔ The more checks, the higher the score.

Example
  "reasoning": "Structural coherence: ✔, Natural appearance: ✔, Artifacts: ✔, Texture fidelity: ✘ (fabric partially deformed).",
  "quality_score": 4

## Output Format:
After evaluation, provide your score and concise reasoning using the following JSON format:
{{
"reasoning": XXX,
"quality_score": X,
}}
"""

prompt_instruction_following = """
You are a professional digital artist and image evaluation specialist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules. 

You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.

Your Objective:
Your task is to **evaluate how the edited image faithfully fulfills the editing instruction**, focusing **exclusively on the presence and correctness of the specified changes**. 

You must:
**Identify detailed visual differences** between Image A and Image B **correctly and faithfully**.
Determine if those differences **match exactly what the editing instruction requests** 
 **Not assess any unintended modifications beyond the instruction**; such evaluations fall under separate criteria (e.g., visual consistency).
**Be careful**, an edit may introduce visual change without fulfilling the actual instruction (e.g., replacing the object instead of modifying it)

## Reasoning:
You must follow these reasoning steps before scoring:
**1. Detect Difference**: What has visually changed between Image A and Image B? (e.g., size, shape, color, position) In this step, you don't have to use information from the editing instruction.
**2. Expected Visual Caption**: Write a factual description of how the edited image should look if the instruction were perfectly followed.
**3. Instruction Match**: 
Compare the observed differences in **1** to the expected change in **2**:
- Was the correct object modified (not replaced)?
- Was the requested attribute (e.g., size, color, position) modified as intended?
- Is the degree of modification accurate (e.g., “match size,” “slightly increase,” etc.)?
**4. Decision**: Use the 1–5 scale to assign a final score.

## Evaluation Scale (1 to 5):
You will assign an **instruction_score** with following rule:
- **5 Perfect Compliance**: The edited image **precisely matches** the intended modification; all required changes are present and accurate. 
- **4 Minor Omission**: The core change is made, but **minor detail** is missing or slightly incorrect. 
- **3 Partial Compliance**: The main idea is present, but one or more required aspects are wrong or incomplete. 
- **2 Major Omission**: Most of the required changes are missing or poorly implemented. 
- **1 Non-Compliance**: The instruction is **not followed at all** or is **completely misinterpreted** 

## Input
**Image A**
**Image B**
**Editing Instruction**: {instruct}
## Output Format
Look at the input again, provide the evaluation score and the explanation in the following JSON format:
{{
"instruction_score": X,
"reasoning": 1. Detect Difference 2. Expected Visual Caption 3. Instruction Match 4. Decision
}}
"""

prompt_dual_evaluation = """
You are a professional digital artist and image evaluation specialist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules. 

You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.
4. **Real-World Knowledge Explanation**: a factual rationale describing what the correct result should look like and why, based on domain knowledge (e.g., physics, chemistry, logic).

## Objective
You must provide **two independent scores** for the **edited image**:
- **Instruction Score**: Does the edited image visually and accurately follow the editing instruction?
- **Knowledge Score**: Given the instruction and original image, does the edited image reflect what should realistically happen based on the explanation?

## A. Instruction Compliance
Your Objective:
Your task is to **evaluate how the edited image faithfully fulfills the editing instruction**, focusing **exclusively on the presence and correctness of the specified changes**. 

You must:
**Identify detailed visual differences** between Image A and Image B **correctly and faithfully**.
Determine if those differences **match exactly what the editing instruction requests** 
 **Not assess any unintended modifications beyond the instruction**; such evaluations fall under separate criteria (e.g., visual consistency).
**Be careful**, an edit may introduce visual change without fulfilling the actual instruction (e.g., replacing the object instead of modifying it)

## Reasoning:
You must follow these reasoning steps before scoring:
**1. Detect Difference**: What has visually changed between Image A and Image B? (e.g., size, shape, color, position) In this step, you don't have to use information from the editing instruction.
**2. Expected Visual Caption**: Write a factual description of how the edited image should look if the instruction were perfectly followed.
**3. Instruction Match**: 
Compare the observed differences in **1** to the expected change in **2**:
- Was the correct object modified (not replaced)?
- Was the requested attribute (e.g., size, color, position) modified as intended?
- Is the degree of modification accurate (e.g., “match size,” “slightly increase,” etc.)?
**4. Decision**: Use the 1–5 scale to assign a final score.

## Evaluation Scale (1 to 5):
You will assign an **instruction_score** with following rule:
- **5 Perfect Compliance**: The edited image **precisely matches** the intended modification; all required changes are present and accurate. 
- **4 Minor Omission**: The core change is made, but **minor detail** is missing or slightly incorrect. 
- **3 Partial Compliance**: The main idea is present, but one or more required aspects are wrong or incomplete. 
- **2 Major Omission**: Most of the required changes are missing or poorly implemented. 
- **1 Non-Compliance**: The instruction is **not followed at all** or is **completely misinterpreted** 

## B. Knowledge Plausibility 
Your Objective:
Evaluate whether the edited image, after applying the instruction to the original image, accurately reflects the real-world behavior described in the provided explanation.

If instruction is not followed (score ≤ 2), assign `knowledge_score = 1` and note: *"Instruction failure ⇒ knowledge invalid."*

## Input
**Original Image**
**Edited Image**
**Editing Instruction**: {instruct}
**Real-World Knowledge Explanation**：{explanation}

## Output Format
Provide both scores and clear reasoning in the following JSON format:
{{
  "instruction_score": X,
  "instruction_reasoning": 1. Detect Difference 2. Expected Visual Caption 3. Instruction Match 4. Decision
  "knowledge_score": X,
  "knowledge_reasoning": 1. Detect Difference 2. Expected Knowledge Expectation 3. Knowledge Match 4. Decision
}}
"""

prompt_abnormal_instruction_following = """
You are a professional digital artist and image evaluation specialist. You will evaluate whether the edited image faithfully and accurately follows the editing instruction, with a focus on correcting unreasonable or implausible aspects.

## You will be given:
1. **Original Image**
2. **Edited Image**
3. **Editing Instruction**: {instruct}  (typically a general instruction such as "correct the unreasonable parts in the image")
4. **Explanation**:  {explanation} (What the image should look like if it were reasonable)

## Your Objective:
Your task is to **evaluate how well the edited image corrects the unreasonable or implausible aspects** described or implied by the instruction, using the explanation as the factual reference for what a "reasonable" image should look like.

## Output Format
Provide your evaluation in the following JSON format:
{{
  "instruction_score": X,
  "reasoning": 1. Detect Unreasonable Aspects 2. Expected Visual Caption 3. Correction Match 4. Decision
}}
"""

prompt_view_instruction_following = """
You are a professional digital artist and image-evaluation specialist.

## Inputs
1. **Original Image**
2. **Edited Image**
3. **Ground-Truth Image**
4. **Editing Instruction**: {instruct}

## Objective
Assess whether the edited image alters the **viewpoint / perspective** of the scene exactly as specified, using the ground-truth image as reference.

## Output Format
Then output in JSON:
{{
  "instruction_score": X,
  "reasoning": "1. Detect Viewpoint Change 2. Expected Visual Caption 3. Viewpoint-Change Match 4. Decision"
}}
"""

prompt_consist_temporal = """
You are a professional digital artist and image-evaluation specialist.

## Inputs
1. **Reference Frames**: multiple original images
2. **Predicted Frame**: one modified image
3. **Modification Instruction**: {instruct}

## Objective
Evaluate **visual consistency** of the predicted frame within the temporal context of the reference frames.

## Output Format
Then output:
{{
  "consistency_score": X,
  "reasoning": 1. Detect Consistency 2. Expected Visual Caption 3. Consistency Match 4. Decision
}}
"""

prompt_instruction_temporal = """
You are a professional digital artist and image-evaluation specialist.

## Inputs
1. **Reference Frames**: multiple original images
2. **Predicted Frame**: one modified image
3. **Modification Instruction**: {instruct}

## Objective
Judge whether the predicted frame **faithfully follows the temporal instruction**.

## Output Format
Then output:
{{
  "instruction_score": X,
  "reasoning": 1. Detect Instruction Following 2. Expected Visual Caption 3. Instruction Following Match 4. Decision
}}
"""

prompt_consist_multi = """
You are a professional digital artist and image-evaluation specialist.

## Inputs
1. **Multiple Source Images**
2. **Composite Image**: final output
3. **Modification Instruction**: {instruct}

## Objective
Assess **visual consistency** between the composite image and the chosen **background source**.

## Output Format
Then output:
{{
  "consistency_score": X,
  "reasoning": 1. Detect Consistency 2. Expected Visual Caption 3. Consistency Match 4. Decision
}}
"""

prompt_instruction_multi = """
You are a professional digital artist and image-evaluation specialist.

## Inputs
1. **Multiple Source Images**
2. **Composite Image**: final output
3. **Modification Instruction**: {instruct}

## Objective
Determine whether the composite image **accurately follows the instruction**.

## Output Format
Then output:
{{
  "instruction_score": X,
  "reasoning": 1. Detect Instruction Following 2. Expected Visual Caption 3. Instruction Following Match 4. Decision
}}
"""


# -----------------------------------------------------------------------------
# Dataset -> model inputs
# -----------------------------------------------------------------------------


def _data_root() -> str:
    return os.getenv("KRIS_BENCH_DATA_ROOT", "KRIS_Bench")


def kris_bench_doc_to_visual(doc):
    """
    Return 1+ source images.

    doc["ori_img"] is a list of filenames, relative to {KRIS_BENCH_DATA_ROOT}/{category}/.
    """
    category = str(doc.get("category") or "").strip()
    ori_imgs = doc.get("ori_img") or []
    if isinstance(ori_imgs, str):
        ori_imgs = [ori_imgs]
    if not category or not ori_imgs:
        return []

    root = _data_root()
    out = []
    for fn in ori_imgs:
        p = os.path.join(root, category, fn)
        if not os.path.exists(p):
            continue
        try:
            out.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
    return out


def kris_bench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    instruction = (doc.get("ins_en") or "").strip()
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{instruction}{post_prompt}"


def kris_bench_doc_to_target(doc):
    return (doc.get("ins_en") or "").strip()


# -----------------------------------------------------------------------------
# Judge backend helpers (ImgEdit-style)
# -----------------------------------------------------------------------------


def _get_int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _get_float_env(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _get_openai_client(*, api_key: str, base_url: str, timeout: int):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai package not installed (needed for KRIS-Bench judging).") from e
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _get_eval_config() -> Dict[str, Any]:
    api_key = os.getenv("KRIS_BENCH_API_KEY")
    base_url = os.getenv("KRIS_BENCH_BASE_URL")
    if not api_key:
        raise RuntimeError("KRIS_BENCH_API_KEY is not set.")
    if not base_url:
        raise RuntimeError("KRIS_BENCH_BASE_URL is not set.")
    eval_model = os.getenv("KRIS_BENCH_EVAL_MODEL_NAME", "default")
    judge_model = os.getenv("KRIS_BENCH_JUDGE_MODEL_NAME", eval_model)
    return {
        "api_key": api_key,
        "base_url": base_url,
        "eval_model": eval_model,
        "judge_model": judge_model,
        "timeout": _get_int_env("KRIS_BENCH_TIMEOUT", 180),
        "max_retries": _get_int_env("KRIS_BENCH_MAX_RETRIES", 3),
        "call_delay": _get_float_env("KRIS_BENCH_CALL_DELAY", 0.5),
    }


_OPENAI_CLIENT = None


def _get_or_create_client(cfg: Dict[str, Any]):
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    _OPENAI_CLIENT = _get_openai_client(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=int(cfg["timeout"]))
    eval_logger.info(f"Initialized KRIS-Bench judge client (eval_model={cfg.get('eval_model')}, judge_model={cfg.get('judge_model')})")
    return _OPENAI_CLIENT


def _detect_default_model_name(client) -> str:
    try:
        models = client.models.list()
        if getattr(models, "data", None):
            return models.data[0].id
    except Exception:
        pass
    return "default"


def _call_chat(messages: List[Dict[str, Any]], *, max_tokens: int, temperature: float, model_override: Optional[str] = None) -> str:
    cfg = _get_eval_config()
    client = _get_or_create_client(cfg)
    model_name = model_override or cfg.get("eval_model", "default")
    if model_name == "default":
        model_name = _detect_default_model_name(client)
        cfg["eval_model"] = model_name

    max_retries = int(cfg.get("max_retries", 3))
    call_delay = float(cfg.get("call_delay", 0.5))
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            if attempt > 0 or call_delay > 0:
                time.sleep(call_delay * (2**attempt) if attempt > 0 else call_delay)
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content if resp.choices else ""
        except Exception as e:
            last_error = e
            s = str(e).lower()
            retryable = any(k in s for k in ["timeout", "timed out", "504", "502", "503", "gateway", "rate limit", "overloaded"])
            if retryable and attempt < max_retries - 1:
                wait = (2**attempt) * 2
                eval_logger.warning(f"KRIS judge call failed (attempt {attempt+1}/{max_retries}), retrying in {wait}s: {str(e)[:200]}")
                time.sleep(wait)
                continue
            raise
    raise last_error if last_error else RuntimeError("KRIS judge call failed with unknown error")


def image_to_base64(image: Any) -> Optional[str]:
    """
    Encode image to base64.
    To reduce payload size, always encode as JPEG in-memory (quality=85).
    """
    try:
        if isinstance(image, str):
            if not os.path.exists(image):
                return None
            image = Image.open(image).convert("RGB")
        elif hasattr(image, "convert"):
            image = image.convert("RGB")
        else:
            return None

        buf = BytesIO()
        image.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def _msg_text(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def _msg_image_jpeg(b64: str) -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


# -----------------------------------------------------------------------------
# Response parsing (adapted from github_repo/Kris_Bench/metrics_common.py)
# -----------------------------------------------------------------------------


def _extract_json_field(response: str, score_key: str, reason_key: str) -> Tuple[Optional[int], Optional[str]]:
    pattern = r"\{[^{}]*" + re.escape(score_key) + r"[^{}]*\}"
    m = re.search(pattern, response, re.DOTALL)
    if not m:
        return None, None
    try:
        data = json.loads(m.group(0))
        score = data.get(score_key)
        reason = data.get(reason_key)
        if score is not None:
            score = int(score)
        return score, reason
    except Exception:
        return None, None


_DEFAULT_PATTERNS = [
    r"([1-5])\s*/\s*5",
    r"([1-5])\s+out\s+of\s+5",
    r"\b([1-5])\b",
]


def _extract_score_and_reason(response: str, *, score_key: str, reason_fields: List[str], prefix_patterns: Optional[List[str]] = None) -> Tuple[Optional[int], Optional[str]]:
    for rf in reason_fields:
        score, reason = _extract_json_field(response, score_key, rf)
        if score is not None:
            return score, reason
    patterns = (prefix_patterns or []) + _DEFAULT_PATTERNS
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE | re.DOTALL)
        if m:
            return int(m.group(1)), None
    return None, None


def _extract_consistency_score_and_reason(response: str) -> Tuple[Optional[int], Optional[str]]:
    return _extract_score_and_reason(
        response,
        score_key="consistency_score",
        reason_fields=["reason", "reasoning"],
        prefix_patterns=[r"consistency[_\s]*score\s*[:：]?\s*([1-5])"],
    )


def _extract_instruction_score_and_reason(response: str) -> Tuple[Optional[int], Optional[str]]:
    return _extract_score_and_reason(
        response,
        score_key="instruction_score",
        reason_fields=["reasoning", "reason"],
        prefix_patterns=[r"instruction[_\s]*score\s*[:：]?\s*([1-5])"],
    )


def _extract_quality_score_and_reason(response: str) -> Tuple[Optional[int], Optional[str]]:
    return _extract_score_and_reason(
        response,
        score_key="quality_score",
        reason_fields=["reasoning", "reason"],
        prefix_patterns=[r"quality[_\s]*score\s*[:：]?\s*([1-5])"],
    )


def _extract_dual_scores(response: str) -> Dict[str, Any]:
    m = re.search(r"\{[^{}]*instruction_score[^{}]*\}", response, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return {
                "instruction_score": int(data.get("instruction_score")) if data.get("instruction_score") is not None else None,
                "knowledge_score": int(data.get("knowledge_score")) if data.get("knowledge_score") is not None else None,
                "instruction_reasoning": data.get("instruction_reasoning") or data.get("reasoning"),
                "knowledge_reasoning": data.get("knowledge_reasoning"),
            }
        except Exception:
            pass

    instr = knowl = None
    m1 = re.search(r"instruction[_\s]*score\s*[:：]\s*([1-5])", response, re.IGNORECASE)
    if m1:
        instr = int(m1.group(1))
    m2 = re.search(r"knowledge[_\s]*score\s*[:：]\s*([1-5])", response, re.IGNORECASE)
    if m2:
        knowl = int(m2.group(1))
    return {
        "instruction_score": instr,
        "knowledge_score": knowl,
        "instruction_reasoning": None,
        "knowledge_reasoning": None,
    }


# -----------------------------------------------------------------------------
# Per-sample evaluation
# -----------------------------------------------------------------------------


def _get_paths_from_doc(doc) -> Tuple[str, str, List[str], str, str]:
    category = str(doc.get("category") or "").strip()
    image_id = str(doc.get("image_id") or doc.get("id") or "").strip()
    key = str(doc.get("key") or f"{category}__{image_id}")
    ori_imgs = doc.get("ori_img") or []
    if isinstance(ori_imgs, str):
        ori_imgs = [ori_imgs]
    gt_img = str(doc.get("gt_img") or "").strip()
    return key, category, list(ori_imgs), gt_img, image_id


def _evaluate_one(
    *,
    category: str,
    instruction: str,
    explanation: str,
    ori_img_paths: List[str],
    edited_img_path: str,
    gt_img_path: Optional[str],
) -> Dict[str, Any]:
    ori_pairs: List[Tuple[str, str]] = []
    for p in ori_img_paths:
        b64 = image_to_base64(p)
        if b64:
            ori_pairs.append((p, b64))
    ori_b64_list = [b64 for (_p, b64) in ori_pairs]
    edited_b64 = image_to_base64(edited_img_path)
    gt_b64 = image_to_base64(gt_img_path) if gt_img_path else None

    out: Dict[str, Any] = {
        "consistency_score": None,
        "consistency_reasoning": None,
        "instruction_score": None,
        "instruction_reasoning": None,
        "quality_score": None,
        "quality_reasoning": None,
        "knowledge_score": None,
        "knowledge_reasoning": None,
    }

    if not edited_b64 or not ori_b64_list:
        return out

    # ---- Consistency ----
    if category == MULTI_CATEGORY:
        prompt = prompt_consist_multi.format(instruct=instruction)
        content = [_msg_text(prompt)]
        for i, b64 in enumerate(ori_b64_list, start=1):
            content.append(_msg_text(f"Reference Image {i}:"))
            content.append(_msg_image_jpeg(b64))
        content.append(_msg_text("Predicted Image:"))
        content.append(_msg_image_jpeg(edited_b64))
        resp = _call_chat([{"role": "user", "content": content}], max_tokens=1000, temperature=0.0)
        out["consistency_score"], out["consistency_reasoning"] = _extract_consistency_score_and_reason(resp)

    elif category == TEMPORAL_CATEGORY:
        prompt = prompt_consist_temporal.format(instruct=instruction)
        content = [_msg_text(prompt)]
        frame_numbers: List[int] = []
        ref_frames: List[Tuple[int, str]] = []
        for p, b64 in ori_pairs:
            m = re.search(r"(\d+)-(\d+)", os.path.basename(p))
            frame_num = int(m.group(2)) if m else 0
            if frame_num:
                frame_numbers.append(frame_num)
            ref_frames.append((frame_num, b64))

        all_possible = {1, 2, 3, 4}
        missing = all_possible - set(frame_numbers)
        pred_frame_num = list(missing)[0] if len(missing) == 1 else (max(frame_numbers) + 1 if frame_numbers else 0)

        frames: List[Tuple[int, str, str]] = [(fn, "Reference", b64) for fn, b64 in ref_frames]
        frames.append((pred_frame_num, "Generated", edited_b64))
        frames.sort(key=lambda x: x[0])

        for fn, kind, b64 in frames:
            content.append(_msg_text(f"Frame {fn} ({kind}):" if fn else f"{kind} Frame:"))
            content.append(_msg_image_jpeg(b64))
        resp = _call_chat([{"role": "user", "content": content}], max_tokens=1000, temperature=0.0)
        out["consistency_score"], out["consistency_reasoning"] = _extract_consistency_score_and_reason(resp)

    else:
        prompt = prompt_consist.format(instruct=instruction)
        content = [
            _msg_text(prompt),
            _msg_text("This is the original image:"),
            _msg_image_jpeg(ori_b64_list[0]),
            _msg_text("This is the edited image:"),
            _msg_image_jpeg(edited_b64),
        ]
        resp = _call_chat([{"role": "user", "content": content}], max_tokens=1000, temperature=0.0)
        out["consistency_score"], out["consistency_reasoning"] = _extract_consistency_score_and_reason(resp)

    # ---- Instruction / (Dual) ----
    if category in KNOWLEDGE_CATEGORIES:
        prompt = prompt_dual_evaluation.format(instruct=instruction, explanation=explanation or "")
        content = [
            _msg_text(prompt),
            _msg_text("This is the original image:"),
            _msg_image_jpeg(ori_b64_list[0]),
            _msg_text("This is the edited image:"),
            _msg_image_jpeg(edited_b64),
        ]
        resp = _call_chat([{"role": "user", "content": content}], max_tokens=1200, temperature=0.0)
        dual = _extract_dual_scores(resp)
        out["instruction_score"] = dual.get("instruction_score")
        out["instruction_reasoning"] = dual.get("instruction_reasoning")
        out["knowledge_score"] = dual.get("knowledge_score")
        out["knowledge_reasoning"] = dual.get("knowledge_reasoning")

    elif category == VIEWPOINT_CATEGORY:
        prompt = prompt_view_instruction_following.format(instruct=instruction)
        content = [
            _msg_text(prompt),
            _msg_text("This is the original image:"),
            _msg_image_jpeg(ori_b64_list[0]),
            _msg_text("This is the edited image:"),
            _msg_image_jpeg(edited_b64),
        ]
        if gt_b64:
            content.append(_msg_text("This is the ground truth image:"))
            content.append(_msg_image_jpeg(gt_b64))
        resp = _call_chat([{"role": "user", "content": content}], max_tokens=1000, temperature=0.0)
        out["instruction_score"], out["instruction_reasoning"] = _extract_instruction_score_and_reason(resp)

    elif category == MULTI_CATEGORY:
        prompt = prompt_instruction_multi.format(instruct=instruction)
        content = [_msg_text(prompt)]
        for i, b64 in enumerate(ori_b64_list, start=1):
            content.append(_msg_text(f"Reference Image {i}:"))
            content.append(_msg_image_jpeg(b64))
        content.append(_msg_text("Predicted Image:"))
        content.append(_msg_image_jpeg(edited_b64))
        resp = _call_chat([{"role": "user", "content": content}], max_tokens=1000, temperature=0.0)
        out["instruction_score"], out["instruction_reasoning"] = _extract_instruction_score_and_reason(resp)

    elif category == TEMPORAL_CATEGORY:
        prompt = prompt_instruction_temporal.format(instruct=instruction)
        content = [_msg_text(prompt)]
        frame_numbers = []
        ref_frames = []
        for p, b64 in ori_pairs:
            m = re.search(r"(\d+)-(\d+)", os.path.basename(p))
            frame_num = int(m.group(2)) if m else 0
            if frame_num:
                frame_numbers.append(frame_num)
            ref_frames.append((frame_num, b64))

        all_possible = {1, 2, 3, 4}
        missing = all_possible - set(frame_numbers)
        pred_frame_num = list(missing)[0] if len(missing) == 1 else (max(frame_numbers) + 1 if frame_numbers else 0)

        frames = [(fn, "Reference", b64) for fn, b64 in ref_frames]
        frames.append((pred_frame_num, "Generated", edited_b64))
        frames.sort(key=lambda x: x[0])
        for fn, kind, b64 in frames:
            content.append(_msg_text(f"Frame {fn} ({kind}):" if fn else f"{kind} Frame:"))
            content.append(_msg_image_jpeg(b64))
        resp = _call_chat([{"role": "user", "content": content}], max_tokens=1000, temperature=0.0)
        out["instruction_score"], out["instruction_reasoning"] = _extract_instruction_score_and_reason(resp)

    else:
        if category in {"anomaly_correction", "abnormality_correction"}:
            prompt = prompt_abnormal_instruction_following.format(instruct=instruction, explanation=explanation or "")
        else:
            prompt = prompt_instruction_following.format(instruct=instruction)
        content = [
            _msg_text(prompt),
            _msg_text("This is the original image:"),
            _msg_image_jpeg(ori_b64_list[0]),
            _msg_text("This is the edited image:"),
            _msg_image_jpeg(edited_b64),
        ]
        resp = _call_chat([{"role": "user", "content": content}], max_tokens=1200, temperature=0.0)
        out["instruction_score"], out["instruction_reasoning"] = _extract_instruction_score_and_reason(resp)

    # ---- Quality ----
    q_content = [
        _msg_text(prompt_quality),
        _msg_text("This is the image to evaluate:"),
        _msg_image_jpeg(edited_b64),
    ]
    q_resp = _call_chat([{"role": "user", "content": q_content}], max_tokens=800, temperature=0.0)
    out["quality_score"], out["quality_reasoning"] = _extract_quality_score_and_reason(q_resp)

    return out


# -----------------------------------------------------------------------------
# lmms-eval glue
# -----------------------------------------------------------------------------


def kris_bench_process_results(doc, results, **kwargs):
    """
    Parse model output JSON, reorganize saved images to Kris_Bench required structure,
    then call the judge backend and return metrics.
    """
    model_name = os.getenv("KRIS_BENCH_MODEL_NAME", "bagel")
    output_base_dir = os.getenv("KRIS_BENCH_OUTPUT_DIR") or os.getenv("BAGEL_OUTPUT_IMAGE_DIR") or "./logs/kris_bench_results"

    key, category, ori_imgs, gt_img, image_id = _get_paths_from_doc(doc)
    instruction = (doc.get("ins_en") or "").strip()
    explanation = (doc.get("explain_en") or "").strip()

    pred0 = results[0] if results else "{}"
    if isinstance(pred0, dict):
        pred = pred0
    else:
        try:
            pred = json.loads(pred0)
        except Exception:
            pred = {"text": "", "images": []}

    model_images = pred.get("images", []) or []
    generated_path = None
    if model_images:
        p0 = model_images[0]
        if isinstance(p0, dict):
            p0 = p0.get("path") or p0.get("image") or p0.get("file")
        if isinstance(p0, str) and os.path.exists(p0):
            generated_path = p0

    # Fallback to unified location if needed: {BAGEL_OUTPUT_IMAGE_DIR}/{task}/{key}.png
    if generated_path is None:
        bagel_root = os.getenv("BAGEL_OUTPUT_IMAGE_DIR")
        if bagel_root:
            cand = os.path.join(bagel_root, "kris_bench", f"{key}.png")
            if os.path.exists(cand):
                generated_path = cand

    # Reorganize to: {output_base_dir}/results/{model}/{category}/{image_id}.jpg
    saved_jpg_path = None
    if generated_path and os.path.exists(generated_path):
        target_dir = os.path.join(output_base_dir, "results", model_name, category)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, f"{image_id}.jpg")
        try:
            img = Image.open(generated_path).convert("RGB")
            img.save(target_path, format="JPEG", quality=95)
            saved_jpg_path = target_path
        except Exception as e:
            eval_logger.warning(f"kris_bench: failed to convert/save image for key={key}: {e}")

    root = _data_root()
    ori_img_paths = [os.path.join(root, category, fn) for fn in (ori_imgs or [])]
    gt_img_path = os.path.join(root, category, gt_img) if gt_img else None

    scores: Dict[str, Any] = {}
    if saved_jpg_path and os.path.exists(saved_jpg_path):
        try:
            scores = _evaluate_one(
                category=category,
                instruction=instruction,
                explanation=explanation,
                ori_img_paths=ori_img_paths,
                edited_img_path=saved_jpg_path,
                gt_img_path=gt_img_path,
            )
        except Exception as e:
            eval_logger.error(f"kris_bench judge failed for key={key} category={category}: {str(e)[:300]}")
            scores = {}

    def _pack(score: Optional[float], reasoning: Optional[str]) -> Dict[str, Any]:
        # NOTE: score must be a float (not None) to avoid np.std crash in lmms-eval's
        # calculate_clt_aggregate_metric. If evaluation failed, we return -1.0 as a
        # sentinel value which aggregation functions will filter out.
        return {
            "key": key,
            "category": category,
            "image_id": image_id,
            "edited_image_path": saved_jpg_path,
            "score": float(score) if score is not None else -1.0,
            "reasoning": reasoning,
            "valid": score is not None,  # Flag to indicate if this score should be used
        }

    overall_vals: List[float] = []
    for k_score in ("consistency_score", "instruction_score", "quality_score", "knowledge_score"):
        v = scores.get(k_score)
        if v is None:
            continue
        try:
            overall_vals.append(float(v))
        except Exception:
            continue
    overall_avg = float(np.mean(overall_vals)) if overall_vals else None

    dim_group, big_class = _kris_dimension_group(doc)

    out = {
        "kris_bench_consistency_score": _pack(scores.get("consistency_score"), scores.get("consistency_reasoning")),
        "kris_bench_instruction_score": _pack(scores.get("instruction_score"), scores.get("instruction_reasoning")),
        "kris_bench_quality_score": _pack(scores.get("quality_score"), scores.get("quality_reasoning")),
        "kris_bench_knowledge_score": _pack(scores.get("knowledge_score"), scores.get("knowledge_reasoning")),
        "kris_bench_overall_avg": _pack(overall_avg, None),
    }

    if dim_group:
        out[f"kris_bench_{dim_group}_consistency_score"] = _pack(scores.get("consistency_score"), scores.get("consistency_reasoning"))
        out[f"kris_bench_{dim_group}_instruction_score"] = _pack(scores.get("instruction_score"), scores.get("instruction_reasoning"))
        out[f"kris_bench_{dim_group}_quality_score"] = _pack(scores.get("quality_score"), scores.get("quality_reasoning"))
        if scores.get("knowledge_score") is not None:
            out[f"kris_bench_{dim_group}_knowledge_score"] = _pack(scores.get("knowledge_score"), scores.get("knowledge_reasoning"))
        out[f"kris_bench_{dim_group}_avg"] = _pack(overall_avg, None)

    if big_class:
        out[f"kris_bench_{big_class}_avg"] = _pack(overall_avg, None)

    return out


# -----------------------------------------------------------------------------
# Aggregations
# -----------------------------------------------------------------------------


def _aggregate_metric(results, metric_name: str) -> float:
    if not results:
        return 0.0
    cat_scores: Dict[str, List[float]] = defaultdict(list)
    scores: List[float] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        # Skip invalid results (score=-1.0 is sentinel for failed evaluation)
        if not r.get("valid", True):
            continue
        s = r.get("score")
        if s is None or s < 0:
            continue
        try:
            s = float(s)
        except Exception:
            continue
        scores.append(s)
        cat_scores[str(r.get("category") or "unknown")].append(s)

    overall_raw = float(np.mean(scores)) if scores else 0.0
    overall = _normalize_score(overall_raw)
    eval_logger.info(f"[kris_bench] {metric_name} overall={overall:.2f} (0-100) (n={len(scores)})")
    for cat in sorted(cat_scores.keys()):
        cat_avg = float(np.mean(cat_scores[cat]))
        eval_logger.info(f"[kris_bench] {metric_name} {cat}: {_normalize_score(cat_avg):.2f} (0-100) (n={len(cat_scores[cat])})")
    return overall


def kris_bench_aggregate_mean(results) -> float:
    """
    Quiet mean aggregation (no per-category logging).
    Used for many group metrics to avoid extremely verbose logs.
    """
    if not results:
        return 0.0
    scores: List[float] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        # Skip invalid results (score=-1.0 is sentinel for failed evaluation)
        if not r.get("valid", True):
            continue
        s = r.get("score")
        if s is None or s < 0:
            continue
        try:
            scores.append(float(s))
        except Exception:
            continue
    return _normalize_score(float(np.mean(scores))) if scores else 0.0


def kris_bench_aggregate_consistency(results) -> float:
    return _aggregate_metric(results, "consistency_score")


def kris_bench_aggregate_instruction(results) -> float:
    return _aggregate_metric(results, "instruction_score")


def kris_bench_aggregate_quality(results) -> float:
    return _aggregate_metric(results, "quality_score")


def kris_bench_aggregate_knowledge(results) -> float:
    return _aggregate_metric(results, "knowledge_score")
