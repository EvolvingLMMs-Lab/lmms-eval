import re
from collections import defaultdict
from functools import lru_cache

import datasets
from huggingface_hub import hf_hub_download
from loguru import logger as eval_logger
from PIL import Image

MME_CC_DATASET_REPO = "MaxwellWen/MME-CC"

_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?|```", re.IGNORECASE)


def _resolve_prompt_kwargs(lmms_eval_specific_kwargs):
    kwargs = lmms_eval_specific_kwargs or {}
    if isinstance(kwargs.get("default"), dict):
        merged_kwargs = dict(kwargs["default"])
        for key, value in kwargs.items():
            if key != "default":
                merged_kwargs[key] = value
        return merged_kwargs
    return kwargs


def _extract_subtask(doc):
    image_list = doc.get("image_list")
    if isinstance(image_list, list) and image_list:
        first_image_path = str(image_list[0]).strip()
        if "/" in first_image_path:
            return first_image_path.split("/", 1)[0]

    extra = doc.get("extra")
    if isinstance(extra, dict):
        for key in ["Subtask", "subtask"]:
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().replace(" ", "_")

    return "unknown"


def _extract_reference_answer(doc):
    ground_truth = doc.get("ground_truth")
    if not isinstance(ground_truth, dict):
        return ""

    raw_reference = ground_truth.get("answer", "")
    if not isinstance(raw_reference, str):
        return ""

    reference = raw_reference.strip()
    if "## The correct answer is:" in reference:
        reference = reference.split("## The correct answer is:", 1)[1].strip()
    if "## Scoring criteria:" in reference:
        reference = reference.split("## Scoring criteria:", 1)[0].strip()
    return reference


def _normalize_answer(text):
    if not isinstance(text, str):
        return ""

    normalized = text.strip()
    if "</think>" in normalized:
        normalized = normalized.split("</think>")[-1].strip()

    normalized = _CODE_FENCE_PATTERN.sub("", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().casefold()


@lru_cache(maxsize=4096)
def _download_image(image_path):
    return hf_hub_download(
        repo_id=MME_CC_DATASET_REPO,
        repo_type="dataset",
        filename=image_path,
    )


def mme_cc_process_docs(dataset):
    processed_docs = []
    for doc in dataset:
        updated_doc = dict(doc)
        updated_doc["subtask"] = _extract_subtask(updated_doc)
        updated_doc["target_answer"] = _extract_reference_answer(updated_doc)
        processed_docs.append(updated_doc)

    eval_logger.info("[mme_cc] Loaded {} samples", len(processed_docs))
    return datasets.Dataset.from_list(processed_docs)


def mme_cc_doc_to_visual(doc):
    visuals = []
    for image_path in doc.get("image_list", []):
        if not isinstance(image_path, str) or not image_path.strip():
            continue
        local_path = _download_image(image_path)
        with Image.open(local_path) as image:
            visuals.append(image.convert("RGB"))
    return visuals


def mme_cc_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt_kwargs = _resolve_prompt_kwargs(lmms_eval_specific_kwargs)
    pre_prompt = prompt_kwargs.get("pre_prompt", "")
    post_prompt = prompt_kwargs.get("post_prompt", "")
    prompt = str(doc.get("prompt", "")).strip()
    return f"{pre_prompt}{prompt}{post_prompt}".strip()


def mme_cc_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    prompt = mme_cc_doc_to_text(doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)
    content = []
    for image in mme_cc_doc_to_visual(doc):
        content.append({"type": "image", "url": image})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def mme_cc_doc_to_target(doc):
    target_answer = doc.get("target_answer")
    if isinstance(target_answer, str) and target_answer.strip():
        return target_answer
    return _extract_reference_answer(doc)


def mme_cc_process_results(doc, results):
    prediction = results[0] if results else ""
    reference = mme_cc_doc_to_target(doc)

    exact_match = 1.0 if _normalize_answer(prediction) == _normalize_answer(reference) else 0.0
    answered = 1.0 if isinstance(prediction, str) and prediction.strip() else 0.0
    subtask = str(doc.get("subtask", _extract_subtask(doc)))

    return {
        "mme_cc_exact_match": {"score": exact_match, "total": 1.0, "subtask": subtask},
        "mme_cc_answered_rate": {"score": answered, "total": 1.0},
    }


def _aggregate_score(results):
    total_score = 0.0
    total_count = 0.0

    for result in results:
        if isinstance(result, dict):
            total_score += float(result.get("score", 0.0))
            total_count += float(result.get("total", 1.0))
        else:
            total_score += float(result)
            total_count += 1.0

    if total_count == 0.0:
        return 0.0
    return total_score / total_count


def mme_cc_aggregate_exact_match(results):
    subtask_stats = defaultdict(lambda: {"score": 0.0, "total": 0.0})

    for result in results:
        if not isinstance(result, dict):
            continue
        subtask = str(result.get("subtask", "unknown"))
        subtask_stats[subtask]["score"] += float(result.get("score", 0.0))
        subtask_stats[subtask]["total"] += float(result.get("total", 1.0))

    for subtask in sorted(subtask_stats):
        total = subtask_stats[subtask]["total"]
        if total == 0.0:
            continue
        score = subtask_stats[subtask]["score"] / total
        eval_logger.info("[mme_cc] {} exact_match: {:.3f} (n={})", subtask, score, int(total))

    return _aggregate_score(results)


def mme_cc_aggregate_answered_rate(results):
    return _aggregate_score(results)
