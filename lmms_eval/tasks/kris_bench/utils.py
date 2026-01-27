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

Data flow (aligned with gedit_bench/imgedit):
  - Input images: loaded from HF dataset via --process_with_media (doc["ori_images"])
  - Edited images: loaded from model output path (pred["images"][0])

Judge server environment variables:
  - KRIS_BENCH_API_KEY: API key for judge server (use "EMPTY" for local vLLM)
  - KRIS_BENCH_BASE_URL: base URL (e.g., http://localhost:8000/v1)
  - KRIS_BENCH_EVAL_MODEL_NAME: model used for judging (default: "default" -> auto-detect)
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

from lmms_eval.tasks.kris_bench.prompt import (
    prompt_abnormal_instruction_following,
    prompt_consist,
    prompt_consist_multi,
    prompt_consist_temporal,
    prompt_dual_evaluation,
    prompt_instruction_following,
    prompt_instruction_multi,
    prompt_instruction_temporal,
    prompt_quality,
    prompt_view_instruction_following,
)

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
            return int(m.group(2))
    ori = doc.get("ori_img") or []
    if isinstance(ori, str):
        ori = [ori]
    frame_numbers: List[int] = []
    for fn in ori:
        m = re.search(r"(\d+)-(\d+)", os.path.basename(str(fn)))
        if m:
            frame_numbers.append(int(m.group(2)))
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
# Dataset -> model inputs
# -----------------------------------------------------------------------------


def kris_bench_doc_to_visual(doc):
    """Return source images from doc (loaded via --process_with_media)."""
    return [img.convert("RGB") for img in doc["ori_images"]]


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


def image_to_base64(image: Any) -> str:
    """Encode PIL image to base64 JPEG."""
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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
            # Clamp score to valid 1-5 range (judge may hallucinate out-of-range values)
            score = max(1, min(5, score))
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
    def _clamp(v):
        """Clamp score to valid 1-5 range."""
        if v is None:
            return None
        return max(1, min(5, int(v)))

    m = re.search(r"\{[^{}]*instruction_score[^{}]*\}", response, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return {
                "instruction_score": _clamp(data.get("instruction_score")),
                "knowledge_score": _clamp(data.get("knowledge_score")),
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


def _get_doc_metadata(doc) -> Tuple[str, str, str]:
    """Extract key, category, image_id from doc."""
    category = str(doc.get("category") or "").strip()
    image_id = str(doc.get("image_id") or doc.get("id") or "").strip()
    key = str(doc.get("key") or f"{category}__{image_id}")
    return key, category, image_id


def _evaluate_one(
    *,
    category: str,
    instruction: str,
    explanation: str,
    ori_images: List[Any],
    edited_image: Any,
    gt_image: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate generated image with judge."""
    # Encode original images
    ori_b64_list = [image_to_base64(img) for img in ori_images]

    # Encode edited image
    edited_b64 = image_to_base64(edited_image)

    # Encode GT image if provided
    gt_b64 = image_to_base64(gt_image) if gt_image is not None else None

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
        for i, b64 in enumerate(ori_b64_list, start=1):
            content.append(_msg_text(f"Frame {i} (Reference):"))
            content.append(_msg_image_jpeg(b64))
        content.append(_msg_text(f"Frame {len(ori_b64_list) + 1} (Generated):"))
        content.append(_msg_image_jpeg(edited_b64))
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
        for i, b64 in enumerate(ori_b64_list, start=1):
            content.append(_msg_text(f"Frame {i} (Reference):"))
            content.append(_msg_image_jpeg(b64))
        content.append(_msg_text(f"Frame {len(ori_b64_list) + 1} (Generated):"))
        content.append(_msg_image_jpeg(edited_b64))
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
    Process model predictions and evaluate using multimodal judge.

    Similar to gedit_bench and imgedit:
    - Original images: from doc (via --process_with_media)
    - Edited image: from pred["images"][0] path

    Args:
        doc: Document containing ori_images, gt_image, instruction, category, etc.
        results: Model predictions [JSON string with {"text": "...", "images": [...]}]
        **kwargs: Additional arguments

    Returns:
        Dict with metrics for all breakdown categories
    """
    key, category, image_id = _get_doc_metadata(doc)
    instruction = (doc.get("ins_en") or "").strip()
    explanation = (doc.get("explain_en") or "").strip()

    # Parse prediction JSON (same as gedit_bench/imgedit)
    pred = results[0] if results else "{}"
    try:
        pred = json.loads(pred)
    except (json.JSONDecodeError, TypeError):
        eval_logger.warning(f"Failed to parse prediction JSON for key={key}")
        pred = {"text": "", "images": []}

    model_images = pred.get("images", [])

    # Get images from doc (loaded via --process_with_media)
    ori_images_list = [img.convert("RGB") for img in doc["ori_images"]]
    gt_image_obj = doc["gt_image"].convert("RGB") if doc.get("gt_image") else None

    # Load edited image from pred path (same as gedit_bench/imgedit)
    edited_image_pil = Image.open(model_images[0]).convert("RGB") if model_images else None

    # Evaluate using judge
    scores: Dict[str, Any] = {}
    if edited_image_pil and ori_images_list:
        try:
            scores = _evaluate_one(
                category=category,
                instruction=instruction,
                explanation=explanation,
                ori_images=ori_images_list,
                edited_image=edited_image_pil,
                gt_image=gt_image_obj,
            )
        except Exception as e:
            eval_logger.error(f"kris_bench judge failed for key={key} category={category}: {str(e)[:300]}")
            scores = {}

    # Get edited image path for logging
    edited_image_path = model_images[0] if model_images else None

    def _pack(score: Optional[float], reasoning: Optional[str]) -> Dict[str, Any]:
        # NOTE: score must be a float (not None) to avoid np.std crash in lmms-eval's
        # calculate_clt_aggregate_metric. If evaluation failed, we return -1.0 as a
        # sentinel value which aggregation functions will filter out.
        return {
            "key": key,
            "category": category,
            "image_id": image_id,
            "edited_image_path": edited_image_path,
            "score": float(score) if score is not None else -1.0,
            "reasoning": reasoning,
            "valid": score is not None,  # Flag to indicate if this score should be used
        }

    overall_vals = [float(scores[k]) for k in ("consistency_score", "instruction_score", "quality_score", "knowledge_score") if scores.get(k) is not None]
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
        if not isinstance(r, dict) or not r.get("valid", True):
            continue
        s = r.get("score")
        if s is None or s < 0:
            continue
        scores.append(float(s))
        cat_scores[str(r.get("category") or "unknown")].append(float(s))

    overall_raw = float(np.mean(scores)) if scores else 0.0
    overall = _normalize_score(overall_raw)
    eval_logger.info(f"[kris_bench] {metric_name} overall={overall:.2f} (0-100) (n={len(scores)})")
    for cat in sorted(cat_scores.keys()):
        cat_avg = float(np.mean(cat_scores[cat]))
        eval_logger.info(f"[kris_bench] {metric_name} {cat}: {_normalize_score(cat_avg):.2f} (0-100) (n={len(cat_scores[cat])})")
    return overall


def kris_bench_aggregate_mean(results) -> float:
    """Quiet mean aggregation (no per-category logging)."""
    if not results:
        return 0.0
    scores = [float(r["score"]) for r in results if isinstance(r, dict) and r.get("valid", True) and r.get("score") is not None and r["score"] >= 0]
    return _normalize_score(float(np.mean(scores))) if scores else 0.0


def kris_bench_aggregate_consistency(results) -> float:
    return _aggregate_metric(results, "consistency_score")


def kris_bench_aggregate_instruction(results) -> float:
    return _aggregate_metric(results, "instruction_score")


def kris_bench_aggregate_quality(results) -> float:
    return _aggregate_metric(results, "quality_score")


def kris_bench_aggregate_knowledge(results) -> float:
    return _aggregate_metric(results, "knowledge_score")
