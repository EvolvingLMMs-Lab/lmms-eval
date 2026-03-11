import io
import json
import re
from collections import Counter
from typing import Any

import Levenshtein
from PIL import Image


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().split())


def _extract_question(doc: dict) -> str:
    for key in ["question", "query", "prompt", "instruction", "text"]:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_answers(doc: dict) -> list[str]:
    answers = doc.get("answers", doc.get("answer", doc.get("target")))
    if answers is None:
        return []
    if isinstance(answers, list):
        return [str(item) for item in answers if str(item).strip()]
    return [str(answers)]


def _extract_options(doc: dict) -> list[str]:
    options = doc.get("options", doc.get("choices"))
    if isinstance(options, list):
        values = []
        for item in options:
            if isinstance(item, dict):
                values.append(str(item.get("text", item.get("option", ""))))
            else:
                values.append(str(item))
        return [value for value in values if value.strip()]
    return []


def _extract_option_letter(prediction: str) -> str:
    match = re.search(r"\b([A-Z])\b", prediction.strip().upper())
    if match:
        return match.group(1)
    return ""


def _to_rgb(image_obj: Any):
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, bytes):
        return Image.open(io.BytesIO(image_obj)).convert("RGB")
    return None


def _detect_document_language(doc) -> str:
    """Detect whether a document is 'en', 'cn', or 'mixed' from its answer JSON.

    Inspects ``page_info.page_attribute.language`` first (most reliable).
    Falls back to majority vote over ``layout_dets[*].attribute.text_language``
    and ``layout_dets[*].attribute.language`` (for tables).
    """
    answer_raw = doc.get("answer") or doc.get("answers") or doc.get("target")
    if not answer_raw:
        return "mixed"

    if isinstance(answer_raw, list):
        answer_raw = answer_raw[0]
    if not isinstance(answer_raw, str):
        return "mixed"

    try:
        answer = json.loads(answer_raw)
    except (json.JSONDecodeError, TypeError):
        return "mixed"

    # Try page-level language first
    page_lang = None
    page_info = answer.get("page_info") or {}
    page_attr = page_info.get("page_attribute") or {}
    page_lang_raw = page_attr.get("language", "")
    if "chinese" in page_lang_raw:
        page_lang = "cn"
    elif "english" in page_lang_raw:
        page_lang = "en"

    if page_lang:
        return page_lang

    # Fall back to element-level majority vote
    lang_counts = Counter()
    for det in answer.get("layout_dets", []):
        attr = det.get("attribute") or {}
        # Text elements use "text_language", tables use "language"
        lang_val = attr.get("text_language", "") or attr.get("language", "")
        if "chinese" in lang_val:
            lang_counts["cn"] += 1
        elif "english" in lang_val or lang_val.endswith("_en"):
            lang_counts["en"] += 1
        elif "mixed" in lang_val:
            lang_counts["mixed"] += 1

    if not lang_counts:
        return "mixed"

    top = lang_counts.most_common(1)[0][0]
    return top


def omnidocbench_doc_to_visual(doc):
    visuals = []

    for key in ["image", "page_image", "document_image"]:
        if key in doc:
            img = _to_rgb(doc[key])
            if img is not None:
                visuals.append(img)

    for key in ["images", "page_images", "document_images", "pages"]:
        value = doc.get(key)
        if isinstance(value, list):
            for item in value:
                img = _to_rgb(item)
                if img is not None:
                    visuals.append(img)

    return visuals


def omnidocbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")

    question = _extract_question(doc)
    options = _extract_options(doc)
    if not options:
        return f"{pre_prompt}{question}{post_prompt}"

    option_labels = [chr(ord("A") + idx) for idx in range(len(options))]
    option_lines = "\n".join(f"{label}. {choice}" for label, choice in zip(option_labels, options))
    return f"{pre_prompt}{question}\n{option_lines}{post_prompt}"


def omnidocbench_doc_to_target(doc):
    answers = _extract_answers(doc)
    return answers[0] if answers else ""


def _normalized_levenshtein_score(pred: str, ref: str) -> float:
    """Compute (1 - normalized_levenshtein_distance) * 100.

    Following the Kimi K2.5 technical report metric.
    """
    if not pred and not ref:
        return 100.0
    max_len = max(len(pred), len(ref))
    if max_len == 0:
        return 100.0
    dist = Levenshtein.distance(pred, ref)
    return (1.0 - dist / max_len) * 100.0


def omnidocbench_process_results(doc, results):
    prediction = _normalize_text(results[0])
    answers = _extract_answers(doc)
    lang = _detect_document_language(doc)

    if not answers:
        em_score = 0.0
        nld_score = 0.0
    else:
        # Exact match
        answer_set = {_normalize_text(answer) for answer in answers}
        em_score = float(prediction in answer_set)

        options = _extract_options(doc)
        if options:
            pred_letter = _extract_option_letter(str(results[0]))
            if pred_letter:
                for answer in answers:
                    if pred_letter == answer.strip().upper()[:1]:
                        em_score = max(em_score, 1.0)

        # Normalized Levenshtein score: (1 - NLD) * 100, take best across answers
        nld_score = max(_normalized_levenshtein_score(prediction, _normalize_text(answer)) for answer in answers)

    lang_payload_em = {"score": em_score, "lang": lang}
    lang_payload_nld = {"score": nld_score, "lang": lang}

    return {
        "omnidocbench_exact_match": em_score,
        "omnidocbench_nld_score": nld_score,
        "omnidocbench_exact_match_en": lang_payload_em,
        "omnidocbench_exact_match_cn": lang_payload_em,
        "omnidocbench_exact_match_mixed": lang_payload_em,
        "omnidocbench_nld_score_en": lang_payload_nld,
        "omnidocbench_nld_score_cn": lang_payload_nld,
        "omnidocbench_nld_score_mixed": lang_payload_nld,
    }


def _aggregate_by_lang(results, target_langs):
    """Filter results by language and compute mean score."""
    filtered = [r["score"] for r in results if r["lang"] in target_langs]
    if not filtered:
        return 0.0
    return sum(filtered) / len(filtered)


def omnidocbench_aggregate_exact_match_en(results, args):
    return _aggregate_by_lang(results, {"en"})


def omnidocbench_aggregate_exact_match_cn(results, args):
    return _aggregate_by_lang(results, {"cn"})


def omnidocbench_aggregate_exact_match_mixed(results, args):
    return _aggregate_by_lang(results, {"mixed"})


def omnidocbench_aggregate_nld_score_en(results, args):
    return _aggregate_by_lang(results, {"en"})


def omnidocbench_aggregate_nld_score_cn(results, args):
    return _aggregate_by_lang(results, {"cn"})


def omnidocbench_aggregate_nld_score_mixed(results, args):
    return _aggregate_by_lang(results, {"mixed"})
