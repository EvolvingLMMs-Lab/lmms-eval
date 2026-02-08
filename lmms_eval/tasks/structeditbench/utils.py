"""
StructEditBench (Structured-Visuals) task utils for lmms-eval.

Reference implementation:
  - github_repo/Structured-Visuals/qwen_scoring.py

This task evaluates edited images with a 2-stage pipeline:
  1) Vision-QA on the edited image -> "model_response"
  2) LLM judge (text-only) decides Correct/Incorrect given question, GT, response

Per-sample accuracies (following qwen_scoring.py):
  - editing_accuracy (%)
  - maintain_accuracy (%)
  - weighted_accuracy (%) = 0.9 * editing + 0.1 * maintain

Environment variables (ImgEdit-style):
  - STRUCTEDITBENCH_API_KEY: API key for eval server (use "EMPTY" for local vLLM)
  - STRUCTEDITBENCH_BASE_URL: base URL of eval server (e.g., http://localhost:8000/v1)
  - STRUCTEDITBENCH_EVAL_MODEL_NAME: model used for BOTH QA and judge (default: "default" -> auto-detect if server supports /models)
  - STRUCTEDITBENCH_JUDGE_MODEL_NAME: optional separate judge model (stage-2)
  - STRUCTEDITBENCH_TIMEOUT: API timeout seconds (default: 180)
  - STRUCTEDITBENCH_MAX_RETRIES: retries on transient errors (default: 3)
  - STRUCTEDITBENCH_CALL_DELAY: delay between calls seconds (default: 0.5)
  - STRUCTEDITBENCH_MAX_QA: optional int cap for qa_list length
  - STRUCTEDITBENCH_QA_MAX_TOKENS: default 128
  - STRUCTEDITBENCH_JUDGE_MAX_TOKENS: default 16
"""

import base64
import json
import os
import time
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger as eval_logger
from PIL import Image

STRUCTEDITBENCH_CATEGORIES = ["chart", "math", "graph", "puzzle", "science", "table"]
CHART_SUBCATEGORIES = {"add_and_del", "auxiliary", "category", "color", "num"}

QA_PROMPT = (
    "You are a vision QA assistant. Look at the image carefully and answer the question in the simplest words. "
    "Your answer must be the single, definitive, and deterministic response based exclusively on the visual "
    "information in the image. Do not infer or add outside information. "
    'If the question is not about the image, or the mentioned elements are not visible, return "N/A". '
    "Directly output the concise answer with no explanation.\n"
    "Question: "
)

JUDGE_PROMPT = (
    "# Task: Evaluate if the model's response is correct.\n"
    'Based on the "Question", "Ground Truth Answer" and "Model Response" provided below, please determine if the '
    '"Model Response" is acceptable.\n\n'
    "# Evaluation Criteria:\n"
    "1. **Core Meaning Priority**: The model's response should capture the essential entity, action, or relation "
    "described in the Ground Truth Answer.\n"
    "   - Accept answers that correctly identify the main object(s), relation(s), or action(s), even if they omit "
    "secondary descriptors such as size, color, thickness, or position.\n"
    "   - Accept concise answers if they preserve the central meaning of the Ground Truth Answer.\n"
    "   - Accept alternative wording or paraphrases that do not contradict the core content.\n"
    "   - Reject answers that describe a different object, relation, or action than the ground truth.\n"
    '   - Reject answers that are "N/A".\n\n'
    "2. **Numerical Values**: A tolerance of ±10% is allowed for any numerical values compared to the "
    '"Ground Truth Answer".\n\n'
    "3. **Over- or under-specification**:\n"
    "   - Accept answers that are less detailed but still factually consistent with the ground truth.\n"
    "   - Accept answers that provide extra detail if this detail does not contradict the ground truth.\n"
    "   - Reject answers that omit or alter the critical subject/object or core relation.\n\n"
    "# Inputs:\n"
    "[Question]: {question}\n"
    "[Ground Truth Answer]: {ground_truth_answer}\n"
    "[Model Response]: {model_response}\n\n"
    "# Output Requirements:\n"
    'Please judge strictly according to the rules above and output only the single word "Correct" or "Incorrect". '
    "Do not include any other explanations, reasons, or punctuation."
)

_openai_client = None


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


def _normalize_category(category: Any) -> str:
    if category is None:
        return "unknown"
    c = str(category).strip().lower()
    if c in CHART_SUBCATEGORIES or "chart" in c:
        return "chart"
    if c in STRUCTEDITBENCH_CATEGORIES:
        return c
    return c or "unknown"


def image_to_base64(image: Any) -> Optional[str]:
    """Encode image to base64 PNG. Returns None on failure."""
    try:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        eval_logger.warning(f"image_to_base64 failed: {e}")
        return None


def structeditbench_doc_to_visual(doc):
    """Return the source image for editing."""
    img = doc.get("source_image") or doc.get("input_image") or doc.get("image") or doc.get("input_image_raw") or doc.get("source")
    if img is None:
        return []
    try:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        return [img]
    except Exception as e:
        eval_logger.warning(f"structeditbench_doc_to_visual failed: {e}")
        return []


def structeditbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    instruction = doc.get("instruction") or doc.get("edit_prompt") or doc.get("prompt") or doc.get("edit_instruction") or ""
    pre_prompt = ""
    post_prompt = ""
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{instruction}{post_prompt}"


def structeditbench_doc_to_target(doc):
    return doc.get("instruction") or doc.get("edit_prompt") or doc.get("prompt") or doc.get("edit_instruction") or ""


def _get_openai_client(*, api_key: str, base_url: Optional[str], timeout: int):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("openai package not installed (needed for StructEditBench scoring).") from e
    kwargs = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _get_eval_config() -> Dict[str, Any]:
    api_key = os.getenv("STRUCTEDITBENCH_API_KEY")
    base_url = os.getenv("STRUCTEDITBENCH_BASE_URL")
    if not api_key:
        raise RuntimeError("STRUCTEDITBENCH_API_KEY is not set.")
    if not base_url:
        raise RuntimeError("STRUCTEDITBENCH_BASE_URL is not set.")
    eval_model = os.getenv("STRUCTEDITBENCH_EVAL_MODEL_NAME", "default")
    judge_model = os.getenv("STRUCTEDITBENCH_JUDGE_MODEL_NAME", eval_model)
    return {
        "api_key": api_key,
        "base_url": base_url,
        "eval_model": eval_model,
        "judge_model": judge_model,
        "timeout": _get_int_env("STRUCTEDITBENCH_TIMEOUT", 180),
        "max_retries": _get_int_env("STRUCTEDITBENCH_MAX_RETRIES", 3),
        "call_delay": _get_float_env("STRUCTEDITBENCH_CALL_DELAY", 0.5),
        "qa_max_tokens": _get_int_env("STRUCTEDITBENCH_QA_MAX_TOKENS", 128),
        "judge_max_tokens": _get_int_env("STRUCTEDITBENCH_JUDGE_MAX_TOKENS", 16),
    }


def _get_or_create_client(cfg: Dict[str, Any]):
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    _openai_client = _get_openai_client(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=int(cfg["timeout"]))
    eval_logger.info(f"Initialized StructEditBench OpenAI client (eval_model={cfg.get('eval_model')}, judge_model={cfg.get('judge_model')})")
    return _openai_client


def _detect_default_model_name(client) -> str:
    try:
        models = client.models.list()
        if getattr(models, "data", None):
            return models.data[0].id
    except Exception:
        pass
    return "default"


def _call_chat(client, *, model: str, messages: List[Dict[str, Any]], max_tokens: int, temperature: float, max_retries: int, call_delay: float) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(max(1, int(max_retries))):
        try:
            if call_delay and call_delay > 0:
                time.sleep(call_delay if attempt == 0 else call_delay * (2**attempt))
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content if resp.choices else ""
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            transient = any(k in msg for k in ["timeout", "timed out", "504", "502", "503", "gateway", "rate limit", "overloaded"])
            if transient and attempt < max_retries - 1:
                wait_time = (2**attempt) * 2
                eval_logger.warning(f"structeditbench eval API transient error (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise
    if last_error:
        raise last_error
    return ""


def _judge_is_correct(text: str) -> bool:
    if not text:
        return False
    return text.strip().lower().startswith("correct")


def _evaluate_one_image(edited_image: Image.Image, qa_list: List[Dict[str, Any]]) -> Tuple[float, float, float, List[Dict[str, Any]]]:
    max_qa_env = os.getenv("STRUCTEDITBENCH_MAX_QA")
    if max_qa_env:
        qa_list = qa_list[: int(max_qa_env)]

    edited_b64 = image_to_base64(edited_image)
    if not edited_b64:
        return 0.0, 0.0, 0.0, []

    cfg = _get_eval_config()
    client = _get_or_create_client(cfg)

    eval_model = cfg.get("eval_model", "default")
    judge_model = cfg.get("judge_model", eval_model)
    if eval_model == "default":
        eval_model = _detect_default_model_name(client)
        cfg["eval_model"] = eval_model
    if judge_model == "default":
        judge_model = _detect_default_model_name(client)
        cfg["judge_model"] = judge_model

    editing_correct = editing_total = 0
    maintain_correct = maintain_total = 0
    qa_results: List[Dict[str, Any]] = []
    failed_qa_count = 0

    for i, qa in enumerate(qa_list or []):
        question = (qa.get("question") or "").strip()
        gt = (qa.get("ground_truth_answer") or qa.get("answer") or "").strip()
        label = (qa.get("label") or "editing").strip().lower()
        if not question or not gt:
            continue

        try:
            # Stage 1: QA on edited image
            qa_text = QA_PROMPT + question
            qa_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{edited_b64}"}},
                        {"type": "text", "text": qa_text},
                    ],
                }
            ]
            model_response = _call_chat(
                client,
                model=eval_model,
                messages=qa_messages,
                max_tokens=int(cfg.get("qa_max_tokens", 128)),
                temperature=0.0,
                max_retries=int(cfg.get("max_retries", 3)),
                call_delay=float(cfg.get("call_delay", 0.5)),
            ).strip()

            # Stage 2: Judge Correct/Incorrect
            judge_text = JUDGE_PROMPT.format(question=question, ground_truth_answer=gt, model_response=model_response)
            judge_messages = [{"role": "user", "content": [{"type": "text", "text": judge_text}]}]
            judge_out = _call_chat(
                client,
                model=judge_model,
                messages=judge_messages,
                max_tokens=int(cfg.get("judge_max_tokens", 16)),
                temperature=0.0,
                max_retries=int(cfg.get("max_retries", 3)),
                call_delay=float(cfg.get("call_delay", 0.5)),
            )
            is_correct = _judge_is_correct(judge_out)

            if label == "maintain":
                maintain_total += 1
                if is_correct:
                    maintain_correct += 1
            else:
                editing_total += 1
                if is_correct:
                    editing_correct += 1

            qa_results.append(
                {
                    "question": question,
                    "ground_truth_answer": gt,
                    "model_response": model_response,
                    "judge": judge_out,
                    "correct": bool(is_correct),
                    "label": label,
                }
            )
        except Exception as e:
            failed_qa_count += 1
            eval_logger.warning(f"QA {i+1}/{len(qa_list)} failed: {str(e)[:120]}...")
            continue

    if failed_qa_count > 0:
        eval_logger.warning(f"Skipped {failed_qa_count}/{len(qa_list)} QAs due to API errors")

    editing_acc = (editing_correct / editing_total * 100.0) if editing_total > 0 else 0.0
    maintain_acc = (maintain_correct / maintain_total * 100.0) if maintain_total > 0 else 0.0
    weighted_acc = editing_acc if maintain_total <= 0 else (0.9 * editing_acc + 0.1 * maintain_acc)
    return float(editing_acc), float(maintain_acc), float(weighted_acc), qa_results


def structeditbench_process_results(doc, results, **kwargs):
    model_name = os.getenv("STRUCTEDITBENCH_MODEL_NAME", "bagel")
    key = str(doc.get("key") or doc.get("id") or doc.get("uid") or "unknown")
    category = _normalize_category(doc.get("category"))

    qa_list = doc.get("qa_list") or []
    if isinstance(qa_list, str):
        try:
            qa_list = json.loads(qa_list)
        except Exception:
            qa_list = []

    pred0 = results[0] if results else "{}"
    if isinstance(pred0, dict):
        pred = pred0
    else:
        try:
            pred = json.loads(pred0)
        except Exception:
            pred = {"text": "", "images": []}

    model_images = pred.get("images", []) or []
    if model_images and isinstance(model_images[0], dict):
        p = model_images[0].get("path") or model_images[0].get("image") or model_images[0].get("file")
        model_images = [p] if p else []

    edited_image_pil = None
    edited_image_path = None

    if model_images:
        p0 = model_images[0]
        if isinstance(p0, str) and os.path.exists(p0):
            edited_image_path = p0
            try:
                edited_image_pil = Image.open(p0).convert("RGB")
            except Exception as e:
                eval_logger.warning(f"Failed to load edited image for key={key}: {e}")

    if edited_image_pil is None:
        # Fallback: allow evaluation-only datasets that already contain the model image
        cand_fields = [f"output_image_{model_name}", f"output_image_{model_name.lower()}", "output_image", "edited_image"]
        for f in cand_fields:
            v = doc.get(f)
            if v is None:
                continue
            try:
                if hasattr(v, "convert"):
                    edited_image_pil = v.convert("RGB")
                    break
                if isinstance(v, str) and os.path.exists(v):
                    edited_image_pil = Image.open(v).convert("RGB")
                    edited_image_path = v
                    break
            except Exception:
                continue

    if edited_image_pil is None:
        eval_logger.warning(f"structeditbench: no edited image found for key={key}")
        base = {"key": key, "category": category, "score": 0.0}
        return {
            "structeditbench_weighted_accuracy": base,
            "structeditbench_editing_accuracy": base,
            "structeditbench_maintain_accuracy": base,
            "structeditbench_chart_weighted_accuracy": base,
            "structeditbench_math_weighted_accuracy": base,
            "structeditbench_graph_weighted_accuracy": base,
            "structeditbench_puzzle_weighted_accuracy": base,
            "structeditbench_science_weighted_accuracy": base,
            "structeditbench_table_weighted_accuracy": base,
        }

    try:
        editing_acc, maintain_acc, weighted_acc, qa_results = _evaluate_one_image(edited_image_pil, qa_list)
    except Exception as e:
        eval_logger.error(f"structeditbench scoring failed for key={key}: {e}")
        editing_acc, maintain_acc, weighted_acc, qa_results = 0.0, 0.0, 0.0, []

    eval_logger.info(f"[structeditbench] key={key} category={category} weighted={weighted_acc:.2f}% " f"edit={editing_acc:.2f}% maintain={maintain_acc:.2f}% " f"(qa={len(qa_results)}/{len(qa_list) if isinstance(qa_list, list) else 0})")

    base_entry = {
        "key": key,
        "category": category,
        "edited_image_path": edited_image_path,
        "num_qa": len(qa_results),
    }
    weighted_entry = {**base_entry, "score": float(weighted_acc), "qa_results": qa_results}
    weighted_entry_no_qa = {**base_entry, "score": float(weighted_acc)}
    editing_entry = {**base_entry, "score": float(editing_acc)}
    maintain_entry = {**base_entry, "score": float(maintain_acc)}

    return {
        "structeditbench_weighted_accuracy": weighted_entry,
        "structeditbench_editing_accuracy": editing_entry,
        "structeditbench_maintain_accuracy": maintain_entry,
        "structeditbench_chart_weighted_accuracy": weighted_entry_no_qa,
        "structeditbench_math_weighted_accuracy": weighted_entry_no_qa,
        "structeditbench_graph_weighted_accuracy": weighted_entry_no_qa,
        "structeditbench_puzzle_weighted_accuracy": weighted_entry_no_qa,
        "structeditbench_science_weighted_accuracy": weighted_entry_no_qa,
        "structeditbench_table_weighted_accuracy": weighted_entry_no_qa,
    }


def structeditbench_aggregate_score(results):
    if not results:
        return 0.0
    scores = [r["score"] for r in results if isinstance(r, dict) and "score" in r]
    return float(np.mean(scores)) if scores else 0.0


def _aggregate_by_category(results, category: str):
    if not results:
        return 0.0
    cat = category.strip().lower()
    scores = [r["score"] for r in results if isinstance(r, dict) and r.get("category") == cat and "score" in r]
    return float(np.mean(scores)) if scores else 0.0


def structeditbench_aggregate_chart(results):
    return _aggregate_by_category(results, "chart")


def structeditbench_aggregate_math(results):
    return _aggregate_by_category(results, "math")


def structeditbench_aggregate_graph(results):
    return _aggregate_by_category(results, "graph")


def structeditbench_aggregate_puzzle(results):
    return _aggregate_by_category(results, "puzzle")


def structeditbench_aggregate_science(results):
    return _aggregate_by_category(results, "science")


def structeditbench_aggregate_table(results):
    return _aggregate_by_category(results, "table")
