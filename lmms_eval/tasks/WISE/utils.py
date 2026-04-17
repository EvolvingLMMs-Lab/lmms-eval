"""WISE text-to-image benchmark utilities.

Evaluation flow:
  1. Send doc["Prompt"] to a text-to-image model.
  2. Model generates images to output_dir (e.g., WISE_raw/bagel_umm/).
  3. Find generated images from output_dir based on doc_id.
  4. Use an OpenAI-compatible multimodal judge to produce three scores:
     consistency, realism, aesthetic_quality (each 0-2).
  5. Calculate WiScore = (0.7*consistency + 0.2*realism + 0.1*aesthetic) / 2.
  6. Aggregate WiScores by WISE category weights.

Environment variables:
  - WISE_RAW_OUTPUT_DIR: model output directory (e.g., /pfs/.../WISE_raw/bagel_umm).
    Required for finding generated images.
  - WISE_API_KEY: judge API key. Falls back to OPENAI_API_KEY.
  - WISE_BASE_URL: optional OpenAI-compatible base URL.
    Falls back to OPENAI_BASE_URL or OPENAI_API_BASE.
  - WISE_MODEL_NAME: judge model name. Default: gpt-4o-2024-05-13.
  - WISE_TIMEOUT: API timeout seconds. Default: 180.
  - WISE_MAX_RETRIES: retry count. Default: 3.
  - WISE_CALL_DELAY: delay between retries/calls in seconds. Default: 0.5.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from loguru import logger as eval_logger
from PIL import Image

CATEGORY_RANGES = {
    "culture": range(1, 401),
    "time": range(401, 568),
    "space": range(568, 701),
    "biology": range(701, 801),
    "physics": range(801, 901),
    "chemistry": range(901, 1001),
}

CATEGORY_LABELS = {
    "culture": "CULTURE",
    "time": "TIME",
    "space": "SPACE",
    "biology": "BIOLOGY",
    "physics": "PHYSICS",
    "chemistry": "CHEMISTRY",
}

GROUPS = {
    "space_time": ("time", "space"),
    "science": ("biology", "physics", "chemistry"),
}

WISE_WEIGHTS = {
    "culture": 0.4,
    "time": 0.167,
    "space": 0.133,
    "biology": 0.1,
    "physics": 0.1,
    "chemistry": 0.1,
}

REQUIRED_SCORE_KEYS = {"consistency", "realism", "aesthetic_quality"}
MAX_EXTRACT_RETRIES = 3

_OPENAI_CLIENT = None


def wise_doc_to_visual(doc):
    """Text-to-image task: no visual input is provided to the model."""
    return []


def wise_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = str(doc.get("Prompt") or doc.get("prompt") or "").strip()
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{prompt}{post_prompt}"


def wise_doc_to_target(doc):
    return str(doc.get("Explanation") or doc.get("Hint") or "")


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _get_eval_config() -> Dict[str, Any]:
    api_key = os.getenv("WISE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("WISE_API_KEY or OPENAI_API_KEY must be set for WISE judging.")

    return {
        "api_key": api_key,
        "base_url": os.getenv("WISE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"),
        "model": os.getenv("WISE_MODEL_NAME", "gpt-4o-2024-05-13"),
        "timeout": _get_int_env("WISE_TIMEOUT", 180),
        "max_retries": _get_int_env("WISE_MAX_RETRIES", 3),
        "call_delay": _get_float_env("WISE_CALL_DELAY", 0.5),
    }


def _get_openai_client(cfg: Dict[str, Any]):
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai package is required for WISE judging.") from e

    kwargs = {"api_key": cfg["api_key"], "timeout": cfg["timeout"]}
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    _OPENAI_CLIENT = OpenAI(**kwargs)
    eval_logger.info(f"Initialized WISE judge client (model={cfg['model']})")
    return _OPENAI_CLIENT


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _msg_text(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def _msg_image_png(image_base64: str) -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}


def _build_evaluation_messages(doc: Dict[str, Any], image_base64: str) -> List[Dict[str, Any]]:
    prompt = str(doc.get("Prompt") or doc.get("prompt") or "")
    explanation = str(doc.get("Explanation") or doc.get("explanation") or "")

    eval_prompt = f"""Please evaluate strictly and return ONLY the three scores as requested.

# Text-to-Image Quality Evaluation Protocol

## System Instruction
You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS. Only images meeting the HIGHEST standards should receive top scores.

**Input Parameters**
- PROMPT: [User's original prompt to]
- EXPLANATION: [Further explanation of the original prompt]
---

## Scoring Criteria

**Consistency (0-2):**  How accurately and completely the image reflects the PROMPT.
* **0 (Rejected):**  Fails to capture key elements of the prompt, or contradicts the prompt.
* **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, or not accurately.  Noticeable deviations from the prompt's intent.
* **2 (Exemplary):**  Perfectly and completely aligns with the PROMPT.  Every single element and nuance of the prompt is flawlessly represented in the image. The image is an ideal, unambiguous visual realization of the given prompt.

**Realism (0-2):**  How realistically the image is rendered.
* **0 (Rejected):**  Physically implausible and clearly artificial. Breaks fundamental laws of physics or visual realism.
* **1 (Conditional):** Contains minor inconsistencies or unrealistic elements.  While somewhat believable, noticeable flaws detract from realism.
* **2 (Exemplary):**  Achieves photorealistic quality, indistinguishable from a real photograph.  Flawless adherence to physical laws, accurate material representation, and coherent spatial relationships. No visual cues betraying AI generation.

**Aesthetic Quality (0-2):**  The overall artistic appeal and visual quality of the image.
* **0 (Rejected):**  Poor aesthetic composition, visually unappealing, and lacks artistic merit.
* **1 (Conditional):**  Demonstrates basic visual appeal, acceptable composition, and color harmony, but lacks distinction or artistic flair.
* **2 (Exemplary):**  Possesses exceptional aesthetic quality, comparable to a masterpiece.  Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating artistic style. Demonstrates a high degree of artistic vision and execution.

---

## Output Format

**Do not include any other text, explanations, or labels.** You must return only three lines of text, each containing a metric and the corresponding score, for example:

**Example Output:**
Consistency: 2
Realism: 1
Aesthetic Quality: 0

---

**IMPORTANT Enforcement:**

Be EXTREMELY strict in your evaluation. A score of '2' should be exceedingly rare and reserved only for images that truly excel and meet the highest possible standards in each metric. If there is any doubt, downgrade the score.

For **Consistency**, a score of '2' requires complete and flawless adherence to every aspect of the prompt, leaving no room for misinterpretation or omission.

For **Realism**, a score of '2' means the image is virtually indistinguishable from a real photograph in terms of detail, lighting, physics, and material properties.

For **Aesthetic Quality**, a score of '2' demands exceptional artistic merit, not just pleasant visuals.

---
Here are the Prompt and EXPLANATION for this evaluation:
PROMPT: "{prompt}"
EXPLANATION: "{explanation}"
Please strictly adhere to the scoring criteria and follow the template format when providing your results."""

    return [
        {
            "role": "system",
            "content": [_msg_text("You are a professional Vincennes image quality audit expert, please evaluate the image quality strictly according to the protocol.")],
        },
        {
            "role": "user",
            "content": [
                _msg_text(eval_prompt),
                _msg_image_png(image_base64),
            ],
        },
    ]


def _extract_scores(text: str) -> Dict[str, float]:
    """Extract three-component scores from judge response.

    Expected format:
        Consistency: 2
        Realism: 1
        Aesthetic Quality: 0
    """
    pat = r"\*{0,2}(Consistency|Realism|Aesthetic Quality)\*{0,2}\s*[::]?\s*(\d)"
    matches = re.findall(pat, text, re.IGNORECASE)
    out = {}
    for k, v in matches:
        out[k.lower().replace(" ", "_")] = float(v)
    if out:
        return out

    # Fallback: handle plain 3-line numeric outputs like:
    # 2
    # 1
    # 0
    nums = re.findall(r"(?m)^\s*([0-2])\s*$", text)
    if len(nums) >= 3:
        return {
            "consistency": float(nums[0]),
            "realism": float(nums[1]),
            "aesthetic_quality": float(nums[2]),
        }
    return {}


def _call_judge(messages: List[Dict[str, Any]]) -> str:
    cfg = _get_eval_config()
    client = _get_openai_client(cfg)
    max_retries = int(cfg.get("max_retries", 3))
    call_delay = float(cfg.get("call_delay", 0.5))
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            if call_delay > 0:
                time.sleep(call_delay * (2**attempt) if attempt > 0 else call_delay)
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                temperature=0.0,
                max_tokens=128,
            )
            return resp.choices[0].message.content if resp.choices else ""
        except Exception as e:
            last_error = e
            retryable = any(k in str(e).lower() for k in ["timeout", "timed out", "504", "502", "503", "gateway", "rate limit", "overloaded"])
            if retryable and attempt < max_retries - 1:
                wait = (2**attempt) * 2
                eval_logger.warning(f"WISE judge call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {str(e)[:200]}")
                time.sleep(wait)
                continue
            raise

    raise last_error if last_error else RuntimeError("WISE judge call failed.")


def _judge_image(doc: Dict[str, Any], image_path: str) -> Dict[str, Any]:
    """Judge image and calculate WiScore from three component scores.

    Component scores are in range 0-2.
    WiScore formula: (0.7 * consistency + 0.2 * realism + 0.1 * aesthetic_quality) / 2
    This normalizes the result to 0.0-1.0 range.
    """
    image_base64 = _encode_image(image_path)
    messages = _build_evaluation_messages(doc, image_base64)

    last_response = ""
    for attempt in range(1, MAX_EXTRACT_RETRIES + 1):
        response = _call_judge(messages)
        last_response = response
        scores = _extract_scores(response)
        if REQUIRED_SCORE_KEYS.issubset(scores.keys()):
            # Calculate WiScore from three components
            consistency = float(scores["consistency"])
            realism = float(scores["realism"])
            aesthetic_quality = float(scores["aesthetic_quality"])
            wiscore = (0.7 * consistency + 0.2 * realism + 0.1 * aesthetic_quality) / 2

            return {
                "consistency": consistency,
                "realism": realism,
                "aesthetic_quality": aesthetic_quality,
                "score": wiscore,
                "evaluation": response,
                "status": "ok",
            }
        eval_logger.warning(f"WISE judge response could not be parsed (attempt {attempt}/{MAX_EXTRACT_RETRIES}): {response[:200]}")

    return {
        "consistency": 0.0,
        "realism": 0.0,
        "aesthetic_quality": 0.0,
        "score": 0.0,
        "evaluation": last_response,
        "status": "parse_failed",
    }


def _prompt_id(doc: Dict[str, Any]) -> int:
    value = doc.get("prompt_id")
    if value is None:
        value = doc.get("id")
    try:
        return int(value)
    except Exception:
        return -1


def _category_from_prompt_id(prompt_id: int) -> str:
    for category, id_range in CATEGORY_RANGES.items():
        if prompt_id in id_range:
            return category
    return "unknown"


def _load_prediction(results: Sequence[str]) -> Dict[str, Any]:
    pred = results[0] if results else "{}"
    if isinstance(pred, dict):
        return pred
    try:
        return json.loads(pred)
    except (json.JSONDecodeError, TypeError):
        eval_logger.warning(f"Failed to parse WISE prediction JSON: {str(pred)[:200]}")
        return {"text": str(pred), "images": []}


def _find_generated_image(output_dir: str, doc_id: int) -> Optional[str]:
    """Find generated image for doc_id in output_dir.

    Supported formats:
    - bagel.py: output_dir/20260417_xxx/WISE_{doc_id}_0.png
    - bagel_unig2u.py: output_dir/WISE_{doc_id}.png
    - mmada.py: output_dir/WISE_{doc_id}.png
    """
    if not os.path.exists(output_dir):
        return None

    # 1. Check for timestamp subdirectories (bagel.py case)
    subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and re.match(r"\d{8}_\d{6}_[a-f0-9]+", d)]

    if subdirs:
        # Use the latest timestamp directory
        latest_subdir = sorted(subdirs, reverse=True)[0]
        search_dir = os.path.join(output_dir, latest_subdir)
    else:
        search_dir = output_dir

    # 2. Try common naming patterns
    possible_names = [
        f"WISE_{doc_id}_0.png",  # bagel.py
        f"WISE_{doc_id}.png",  # bagel_unig2u, mmada
    ]

    for name in possible_names:
        path = os.path.join(search_dir, name)
        if os.path.exists(path):
            return path

    return None


def _save_prompt_image(prompt_id: int, doc_id: int) -> Optional[str]:
    """Find the generated image path for evaluation.

    Args:
        prompt_id: WISE dataset prompt_id
        doc_id: lmms-eval doc index

    Returns:
        Image path if found, None otherwise
    """
    output_dir = os.getenv("WISE_RAW_OUTPUT_DIR")
    if not output_dir:
        eval_logger.warning("WISE_RAW_OUTPUT_DIR environment variable not set")
        return None

    # Find image from output_dir
    image_path = _find_generated_image(output_dir, doc_id)
    if not image_path:
        eval_logger.warning(f"Missing image: prompt_id={prompt_id}, doc_id={doc_id}")
        return None

    return image_path


def _pack_result(
    *,
    doc: Dict[str, Any],
    image_path: Optional[str],
    consistency: float = 0.0,
    realism: float = 0.0,
    aesthetic_quality: float = 0.0,
    score: float,
    evaluation: Optional[str],
    status: str,
) -> Dict[str, Any]:
    prompt_id = _prompt_id(doc)
    category = _category_from_prompt_id(prompt_id)
    return {
        "prompt_id": prompt_id,
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, "UNKNOWN"),
        "raw_category": doc.get("Category"),
        "subcategory": doc.get("Subcategory"),
        "prompt": doc.get("Prompt") or doc.get("prompt"),
        "explanation": doc.get("Explanation") or doc.get("explanation"),
        "hint": doc.get("Hint"),
        "image_path": image_path,
        "consistency": float(consistency),
        "realism": float(realism),
        "aesthetic_quality": float(aesthetic_quality),
        "score": float(score),
        "evaluation": evaluation,
        "status": status,
        "valid": True,
    }


def wise_process_results(doc, results, **kwargs):
    pred = _load_prediction(results)
    prompt_id = _prompt_id(doc)
    doc_id = kwargs.get("doc_id", 0)  # Get doc_id from kwargs
    image_path = _save_prompt_image(prompt_id, doc_id)

    if not image_path:
        packed = _pack_result(doc=doc, image_path=None, consistency=0.0, realism=0.0, aesthetic_quality=0.0, score=0.0, evaluation=None, status="missing_image")
    else:
        try:
            judged = _judge_image(doc, image_path)
            packed = _pack_result(
                doc=doc,
                image_path=image_path,
                consistency=judged.get("consistency", 0.0),
                realism=judged.get("realism", 0.0),
                aesthetic_quality=judged.get("aesthetic_quality", 0.0),
                score=judged["score"],
                evaluation=judged.get("evaluation"),
                status=judged.get("status", "ok"),
            )
        except Exception as e:
            if "API_KEY" in str(e) or "openai package is required" in str(e):
                raise
            eval_logger.error(f"WISE judge failed for prompt_id={prompt_id}: {str(e)[:300]}")
            packed = _pack_result(doc=doc, image_path=image_path, consistency=0.0, realism=0.0, aesthetic_quality=0.0, score=0.0, evaluation=None, status="judge_failed")

    aggregate_entry = {key: packed[key] for key in ("prompt_id", "category", "category_label", "raw_category", "subcategory", "image_path", "consistency", "realism", "aesthetic_quality", "score", "status", "valid")}

    return {
        "WISE_culture_score": aggregate_entry,
        "WISE_time_score": aggregate_entry,
        "WISE_space_score": aggregate_entry,
        "WISE_biology_score": aggregate_entry,
        "WISE_physics_score": aggregate_entry,
        "WISE_chemistry_score": aggregate_entry,
        "WISE_overall_wiscore": aggregate_entry,
    }


def _valid_results(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = []
    for result in results:
        if not isinstance(result, dict) or not result.get("valid", True):
            continue
        score = result.get("score")
        if not isinstance(score, (int, float)):
            continue
        valid.append(result)
    return valid


def _mean_score(results: Iterable[Dict[str, Any]]) -> float:
    valid = _valid_results(results)
    if not valid:
        return 0.0
    return float(np.mean([float(r["score"]) for r in valid]))


def _mean_for_categories(results: Iterable[Dict[str, Any]], categories: Sequence[str], label: str) -> float:
    categories = tuple(categories)
    selected = [r for r in _valid_results(results) if r.get("category") in categories]
    if not selected:
        eval_logger.warning(f"[WISE] No samples for {label}.")
        return 0.0
    score = _mean_score(selected)
    eval_logger.info(f"[WISE] {label}: {score:.4f} (n={len(selected)})")
    return score


def _log_prompt_id_coverage(results: Sequence[Dict[str, Any]]) -> None:
    present = {int(r["prompt_id"]) for r in results if isinstance(r.get("prompt_id"), int) and r["prompt_id"] > 0}
    if not present:
        return
    expected = set(range(1, 1001))
    missing = sorted(expected - present)
    if missing:
        preview = missing[:20]
        suffix = "..." if len(missing) > len(preview) else ""
        eval_logger.warning(f"[WISE] Missing {len(missing)} expected prompt_ids for full WISE scoring: {preview}{suffix}")


def wise_aggregate_mean(results) -> float:
    score = _mean_score(results)
    eval_logger.info(f"[WISE] Overall mean binary score: {score:.4f} (n={len(_valid_results(results))})")
    return score


def wise_aggregate_culture(results) -> float:
    return _mean_for_categories(results, ("culture",), "CULTURE")


def wise_aggregate_time(results) -> float:
    return _mean_for_categories(results, ("time",), "TIME")


def wise_aggregate_space(results) -> float:
    return _mean_for_categories(results, ("space",), "SPACE")


def wise_aggregate_space_time(results) -> float:
    return _mean_for_categories(results, GROUPS["space_time"], "SPACE-TIME")


def wise_aggregate_biology(results) -> float:
    return _mean_for_categories(results, ("biology",), "BIOLOGY")


def wise_aggregate_physics(results) -> float:
    return _mean_for_categories(results, ("physics",), "PHYSICS")


def wise_aggregate_chemistry(results) -> float:
    return _mean_for_categories(results, ("chemistry",), "CHEMISTRY")


def wise_aggregate_science(results) -> float:
    return _mean_for_categories(results, GROUPS["science"], "SCIENCE")


def wise_aggregate_overall_wiscore(results) -> float:
    valid = _valid_results(results)
    if not valid:
        return 0.0

    _log_prompt_id_coverage(valid)

    category_scores: Dict[str, List[float]] = defaultdict(list)
    for result in valid:
        category = str(result.get("category") or "unknown")
        if category in WISE_WEIGHTS:
            category_scores[category].append(float(result["score"]))

    weighted_sum = 0.0
    used_weight = 0.0
    for category, weight in WISE_WEIGHTS.items():
        scores = category_scores.get(category, [])
        if not scores:
            eval_logger.warning(f"[WISE] Missing category for weighted WiScore: {CATEGORY_LABELS[category]}")
            continue
        category_mean = float(np.mean(scores))
        weighted_sum += weight * category_mean
        used_weight += weight
        eval_logger.info(f"[WISE] {CATEGORY_LABELS[category]}: {category_mean:.4f} (n={len(scores)}, weight={weight})")

    if used_weight <= 0:
        return 0.0
    wiscore = weighted_sum / used_weight
    eval_logger.info(f"[WISE] Overall WiScore: {wiscore:.4f} (used_weight={used_weight:.3f})")
    return wiscore
