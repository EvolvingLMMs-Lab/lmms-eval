import os
import re
from collections import defaultdict
from pathlib import Path

from loguru import logger as eval_logger
from PIL import Image


def _normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s<>]", "", text)
    return text.strip()


def _resolve_image_path(image_ref):
    if not image_ref:
        return None

    candidate_paths = []

    if isinstance(image_ref, dict):
        image_ref = image_ref.get("path") or image_ref.get("bytes")

    if isinstance(image_ref, Image.Image):
        return image_ref.copy().convert("RGB")

    if isinstance(image_ref, str):
        candidate_paths.extend(
            [
                Path(image_ref),
                Path(image_ref.lstrip("./")),
            ]
        )

        image_root = os.getenv("MMIE_IMAGE_ROOT", "").strip()
        if image_root:
            root = Path(image_root)
            candidate_paths.extend([root / image_ref, root / image_ref.lstrip("./")])

        for candidate in candidate_paths:
            if candidate.exists() and candidate.is_file():
                try:
                    return Image.open(candidate).convert("RGB")
                except Exception as exc:
                    eval_logger.debug("Failed to open MMIE image {}: {}", candidate, exc)
                    return None

    return None


def _conversation_by_role(doc):
    conversations = doc.get("conversations", [])
    human = []
    gpt = []
    for turn in conversations:
        role = turn.get("from")
        content = turn.get("content") or []
        if role == "human":
            human = content
        elif role == "gpt":
            gpt = content
    return human, gpt


def _content_to_text(content):
    pieces = []
    for item in content:
        text = (item.get("text") or "").strip()
        image_ref = item.get("image")
        if text:
            pieces.append(text)
        if image_ref:
            pieces.append("<image>")
    return "\n".join(pieces).strip()


def _content_to_visuals(content):
    visuals = []
    for item in content:
        image_ref = item.get("image")
        if not image_ref:
            continue
        image = _resolve_image_path(image_ref)
        if image is not None:
            visuals.append(image)
    return visuals


def doc_to_visual(doc):
    human_content, _ = _conversation_by_role(doc)
    return _content_to_visuals(human_content)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    human_content, _ = _conversation_by_role(doc)
    prompt = _content_to_text(human_content)

    if pre_prompt:
        prompt = f"{pre_prompt}{prompt}"
    if post_prompt:
        prompt = f"{prompt}{post_prompt}"

    return prompt


def doc_to_target(doc):
    _, gpt_content = _conversation_by_role(doc)
    return _content_to_text(gpt_content)


def _token_f1(reference, prediction):
    ref_tokens = _normalize_text(reference).split()
    pred_tokens = _normalize_text(prediction).split()

    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_set = set(ref_tokens)
    pred_set = set(pred_tokens)
    common = ref_set.intersection(pred_set)
    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(ref_set)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def mmie_process_results(doc, results):
    prediction = (results[0] or "").strip()
    reference = doc_to_target(doc)

    f1 = _token_f1(reference, prediction)
    exact_match = 1.0 if _normalize_text(reference) == _normalize_text(prediction) else 0.0

    result = {
        "id": doc.get("id"),
        "catagory": doc.get("catagory", "unknown"),
        "field": doc.get("field", "unknown"),
    }

    return {
        "mmie_f1": {**result, "score": f1},
        "mmie_exact_match": {**result, "score": exact_match},
    }


def _aggregate_scores(results, metric_name):
    if not results:
        return 0.0

    overall = sum(item["score"] for item in results) / len(results)

    catagory_scores = defaultdict(list)
    field_scores = defaultdict(list)
    for item in results:
        catagory_scores[item.get("catagory", "unknown")].append(item["score"])
        field_scores[item.get("field", "unknown")].append(item["score"])

    eval_logger.info("MMIE {} overall: {:.4f}", metric_name, overall)
    for catagory, values in sorted(catagory_scores.items()):
        eval_logger.info("MMIE {} catagory {}: {:.4f}", metric_name, catagory, sum(values) / len(values))
    for field, values in sorted(field_scores.items()):
        eval_logger.info("MMIE {} field {}: {:.4f}", metric_name, field, sum(values) / len(values))

    return overall


def mmie_aggregate_f1(results):
    return _aggregate_scores(results, "f1")


def mmie_aggregate_exact_match(results):
    return _aggregate_scores(results, "exact_match")
