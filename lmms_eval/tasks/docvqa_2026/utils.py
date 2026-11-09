"""DocVQA 2026 Challenge evaluation utilities.

Implements the official prompt and two-stage hybrid scorer from
``https://github.com/VLR-CVC/DocVQA2026`` (``eval_utils.py``) over the
``VLR-CVC/DocVQA-2026`` dataset. Each multi-page document record in the dataset
carries multiple question-answer pairs; ``docvqa_2026_process_docs`` expands
them into one evaluation sample per question while loading document page images
lazily per ``doc_id``.
"""

import ast
import io
import math
import re
import string
import sys
from collections import defaultdict
from functools import lru_cache
from typing import Any

import datasets
from dateutil import parser as date_parser
from PIL import Image

from lmms_eval.utils import eval_logger

# Required by the official DocVQA 2026 instructions for high-resolution maps,
# engineering drawings, and scientific posters.
Image.MAX_IMAGE_PIXELS = None

CATEGORIES: tuple[str, ...] = (
    "business_report",
    "comics",
    "engineering_drawing",
    "infographics",
    "maps",
    "science_paper",
    "science_poster",
    "slide",
)
ALL_METRICS: tuple[str, ...] = (
    "docvqa_2026_acc",
    *(f"docvqa_2026_{category}" for category in CATEGORIES),
)

MASTER_PROMPT = (
    "ACT AS an expert Document Visual Question Answering (DocVQA) system. "
    "ANALYZE the provided images to extract precise information.\n\n"
    "### MANDATORY RESPONSE RULES:\n"
    "1. SOURCE ADHERENCE: If the question is unanswerable from the document, "
    'respond ONLY with "Unknown".\n'
    "2. LIST FORMATTING: List multiple answers in order of appearance, "
    'separated by a comma and a single space (e.g., "Answer A, Answer B"). '
    'Do NOT use "and".\n'
    "3. NUMBERS & UNITS:\n"
    "   - Convert units to their standardized abbreviation "
    '(e.g., use "kg" not "kilograms", "m" not "meters").\n'
    "   - Place a single space between the number and the unit "
    '(e.g., "50 kg", "10 USD").\n'
    "4. PERCENTAGES: For percentages, attach the '%' symbol directly to the "
    'number with NO space (e.g., "50%", not "50 %").\n'
    "5. DATE FORMATTING: Convert all dates to YYYY-MM-DD format "
    '(e.g., convert "Jan 1st 24" to "2024-01-01").\n'
    "6. DECIMAL FORMATTING: Decimals should be separated by a single period "
    '(e.g., "3.14", not "3,14").\n'
    "7. THOUSANDS SEPARATOR: Do NOT use commas as thousands separators "
    '(e.g., "1000", not "1,000").\n'
    "8. NO FILLER: Output ONLY the result. Do not frame with sentences like "
    '"The answer is...".'
    "\n\n### REASONING PROTOCOL:\n"
    "1. Perform exhaustive step-by-step reasoning to locate and verify the data.\n"
    "2. Verify if the data contains a date, number, or unit.\n"
    "3. Step-by-step, transform the data to match the MANDATORY RESPONSE RULES "
    "(e.g., converting date format).\n"
    "\n\n### OUTPUT FORMAT:\n"
    "After your analysis, you MUST provide the final result in the following format:\n"
    "FINAL ANSWER: [Your exact formatted answer]\n"
    "Ensure the content inside [FINAL ANSWER] strictly follows the "
    "MANDATORY RESPONSE RULES."
)

_MAGNITUDE_UNIT_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*(.*)$")
_VERSION_STRING_RE = re.compile(r"^\d+\.\d+\.\d+$")
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TRANSLATOR = str.maketrans(string.punctuation, " " * len(string.punctuation))
_DOC_VISUAL_SOURCE: dict[str, tuple[datasets.Dataset, int]] = sys.modules.setdefault(  # type: ignore[assignment]
    "_lmms_eval_docvqa_2026_visual_source",
    {},
)


# Dataset expansion & visual decoding


def get_evaluation_prompt() -> str:
    """Return the official DocVQA 2026 master prompt."""
    return MASTER_PROMPT


def _require_non_empty_answer(raw_ans: Any, question_id: str = "") -> str:
    ans_str = "" if raw_ans is None else str(raw_ans).strip()
    if not ans_str or ans_str.upper() == "NULL":
        raise ValueError(
            f"DocVQA 2026: missing or empty ground-truth answer for question_id={question_id!r}",
        )
    return ans_str


def _extract_qa_pairs(
    questions_field: Any,
    answers_field: Any,
) -> list[tuple[str, str, str]]:
    """Pair ``(question_id, question, answer)`` entries from one document record."""
    if isinstance(questions_field, dict):
        q_ids = [str(qid) for qid in questions_field.get("question_id", [])]
        q_texts = [str(q) for q in questions_field.get("question", [])]
    else:
        q_ids = [str(item["question_id"]) for item in (questions_field or [])]
        q_texts = [str(item["question"]) for item in (questions_field or [])]

    if not q_ids or len(q_ids) != len(q_texts) or len(set(q_ids)) != len(q_ids):
        raise ValueError("DocVQA 2026: malformed or duplicate question_id entries")

    ans_map: dict[str, Any] = {}
    if isinstance(answers_field, dict):
        ans_ids = answers_field.get("question_id", [])
        ans_vals = answers_field.get("answer", [])
        for qid, ans in zip(ans_ids, ans_vals, strict=True):
            ans_map[str(qid)] = ans
    elif isinstance(answers_field, list):
        for item in answers_field:
            if isinstance(item, dict) and "question_id" in item and "answer" in item:
                ans_map[str(item["question_id"])] = item["answer"]

    pairs: list[tuple[str, str, str]] = []
    for qid, qtext in zip(q_ids, q_texts, strict=True):
        pairs.append((qid, qtext, _require_non_empty_answer(ans_map.get(qid), qid)))
    return pairs


def docvqa_2026_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Expand document-level rows into one row per question without duplicating page bytes."""
    Image.MAX_IMAGE_PIXELS = None
    visual_source = dataset.remove_columns(["preview"]) if "preview" in dataset.column_names else dataset
    doc_feature = visual_source.features.get("document")
    if isinstance(doc_feature, datasets.Sequence) and isinstance(
        doc_feature.feature,
        datasets.Image,
    ):
        visual_source = visual_source.cast_column(
            "document",
            datasets.Sequence(datasets.Image(decode=False)),
        )

    meta_drop = [col for col in ("preview", "document") if col in dataset.column_names]
    meta_dataset = dataset.remove_columns(meta_drop) if meta_drop else dataset

    if "question" in dataset.column_names and "questions" not in dataset.column_names:
        for row_idx, row in enumerate(meta_dataset):
            _require_non_empty_answer(
                row.get("answer"),
                str(row.get("question_id", row_idx)),
            )
            if "document" in visual_source.column_names and "doc_id" in meta_dataset.column_names:
                _DOC_VISUAL_SOURCE[str(row["doc_id"])] = (visual_source, row_idx)
        return meta_dataset

    flattened: list[dict[str, str]] = []
    for row_idx, row in enumerate(meta_dataset):
        doc_id = str(row["doc_id"])
        doc_category = str(row["doc_category"])
        if "document" in visual_source.column_names:
            _DOC_VISUAL_SOURCE[doc_id] = (visual_source, row_idx)
        qa_pairs = _extract_qa_pairs(row["questions"], row.get("answers"))
        for qid, question, answer in qa_pairs:
            flattened.append(
                {
                    "doc_id": doc_id,
                    "doc_category": doc_category,
                    "question_id": qid,
                    "question": question,
                    "answer": answer,
                },
            )

    eval_logger.info(
        f"DocVQA 2026: expanded {len(dataset)} documents into {len(flattened)} questions.",
    )
    return datasets.Dataset.from_list(flattened)


@lru_cache(maxsize=256)
def _decode_page_bytes(data: bytes) -> Image.Image:
    """Decode raw image bytes into an RGB PIL Image (content-addressed LRU cache)."""
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(io.BytesIO(data)) as img:
        return img.convert("RGB")


def _decode_page(page: Any) -> Image.Image:
    """Convert a single page item (PIL image, HF dict, bytes, or path) to RGB PIL."""
    Image.MAX_IMAGE_PIXELS = None
    if isinstance(page, Image.Image):
        return page.convert("RGB")
    if isinstance(page, dict):
        if page.get("bytes") is not None:
            return _decode_page_bytes(bytes(page["bytes"])).copy()
        if page.get("path"):
            with Image.open(page["path"]) as img:
                return img.convert("RGB")
    if isinstance(page, (bytes, bytearray)):
        return _decode_page_bytes(bytes(page)).copy()
    if isinstance(page, str):
        with Image.open(page) as img:
            return img.convert("RGB")
    raise TypeError(f"Unsupported page visual representation: {type(page)!r}")


def docvqa_2026_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    """Return the ordered RGB page images for a DocVQA 2026 question."""
    Image.MAX_IMAGE_PIXELS = None
    if doc.get("document") is not None:
        return [_decode_page(page) for page in doc["document"]]

    doc_id = str(doc.get("doc_id", ""))
    if doc_id not in _DOC_VISUAL_SOURCE:
        raise KeyError(f"DocVQA 2026 visuals not found for doc_id={doc_id!r}")

    source_ds, row_idx = _DOC_VISUAL_SOURCE[doc_id]
    return [_decode_page(page) for page in source_ds[row_idx]["document"]]


def docvqa_2026_doc_to_text(
    doc: dict[str, Any],
    lmms_eval_specific_kwargs: dict[str, Any] | None = None,
) -> str:
    """Format the official DocVQA 2026 master prompt with the document question."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = str(doc["question"]).strip()
    return f"{pre_prompt}{MASTER_PROMPT}\n\nQuestion: {question}{post_prompt}"


# Official metric (ported from VLR-CVC/DocVQA2026/eval_utils.py)


def parse_magnitude_unit(text: str) -> tuple[float | None, str]:
    """Extract ``(numeric_magnitude, unit_string)`` when ``text`` starts with a number."""
    cleaned = str(text).lower().strip()
    match = _MAGNITUDE_UNIT_RE.match(cleaned)
    if not match:
        return None, cleaned
    number_str, unit_str = match.groups()
    try:
        val = float(number_str)
        if not math.isfinite(val):
            return None, cleaned
        return val, unit_str.strip()
    except (ValueError, OverflowError):
        return None, cleaned


def check_strict_match(pred: str, gt: str) -> bool:
    """Step A of the official scorer: strict numeric+unit match or calendar date match."""
    pred_val, pred_unit = parse_magnitude_unit(pred)
    gt_val, gt_unit = parse_magnitude_unit(gt)

    if pred_val is not None and gt_val is not None and pred_val == gt_val:
        return pred_unit == gt_unit

    p_clean = str(pred).strip()
    g_clean = str(gt).strip()

    if _VERSION_STRING_RE.match(p_clean) or _VERSION_STRING_RE.match(g_clean):
        return p_clean == g_clean

    if len(p_clean) >= 6 and len(g_clean) >= 6:
        try:
            pred_date = date_parser.parse(p_clean, fuzzy=False).date()
            gt_date = date_parser.parse(g_clean, fuzzy=False).date()
            if pred_date == gt_date:
                return True
        except (ValueError, TypeError, OverflowError):
            pass

    return False


def clean_text(text: str) -> str:
    """Normalize punctuation, articles (``a``, ``an``, ``the``), and whitespace for ANLS."""
    translated = str(text).lower().translate(_PUNCT_TRANSLATOR)
    without_articles = _ARTICLE_RE.sub(" ", translated)
    return " ".join(without_articles.split())


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute exact Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def get_anls(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity (1 - distance / max_len)."""
    a = str(s1).lower().strip()
    b = str(s2).lower().strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = _levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    return 1.0 - (float(dist) / float(max_len))


def is_string_correct(
    clean_pred: str,
    clean_gt_candidates: list[str],
    threshold: float = 0.9,
) -> bool:
    """Return whether ``clean_pred`` reaches at least ``threshold`` ANLS against any candidate."""
    best_score = max(
        (get_anls(clean_pred, candidate) for candidate in clean_gt_candidates),
        default=0.0,
    )
    return best_score >= threshold


def _parse_gt_candidates(ground_truth: Any) -> list[str]:
    """Parse ground truth into non-empty candidate strings using ``ast.literal_eval``."""
    if ground_truth is None:
        raise ValueError("DocVQA 2026: ground_truth cannot be None")
    if isinstance(ground_truth, list):
        candidates = [_require_non_empty_answer(x) for x in ground_truth]
    else:
        gt_str = _require_non_empty_answer(ground_truth)
        parsed_gt: Any = None
        try:
            parsed_gt = ast.literal_eval(gt_str)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            parsed_gt = None
        if isinstance(parsed_gt, list):
            candidates = [_require_non_empty_answer(x) for x in parsed_gt]
        else:
            candidates = [gt_str]
    if not candidates:
        raise ValueError("DocVQA 2026: ground_truth candidate list cannot be empty")
    return candidates


def evaluate_docvqa_prediction(raw_prediction: str, ground_truth: Any) -> tuple[bool, str]:
    """Official two-stage hybrid evaluator from ``VLR-CVC/DocVQA2026/eval_utils.py``."""
    gt_candidates = _parse_gt_candidates(ground_truth)
    text = str(raw_prediction or "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]

    if "FINAL ANSWER:" not in text:
        return False, text

    extracted_answer = text.split("FINAL ANSWER:")[-1].strip()
    if not extracted_answer:
        return False, extracted_answer

    gt_is_numeric = False
    for gt in gt_candidates:
        if check_strict_match(extracted_answer, gt):
            return True, extracted_answer
        gt_val, _ = parse_magnitude_unit(gt)
        if gt_val is not None:
            gt_is_numeric = True

    if gt_is_numeric:
        return False, extracted_answer

    clean_pred = clean_text(extracted_answer)
    clean_gt_candidates = [clean_text(gt) for gt in gt_candidates]
    is_ok = is_string_correct(clean_pred, clean_gt_candidates, threshold=0.9)
    return is_ok, extracted_answer


# lmms-eval hooks


def docvqa_2026_process_results(
    doc: dict[str, Any],
    results: list[str],
) -> dict[str, Any]:
    """Score one prediction and emit overall record plus numeric-or-null domain scores."""
    raw_prediction = results[0] if results else ""
    answer = _require_non_empty_answer(
        doc.get("answer"),
        str(doc.get("question_id", "")),
    )
    is_correct, extracted = evaluate_docvqa_prediction(raw_prediction, answer)
    score = float(is_correct)
    doc_category = str(doc.get("doc_category", ""))
    record = {
        "doc_id": str(doc.get("doc_id", "")),
        "question_id": str(doc.get("question_id", "")),
        "doc_category": doc_category,
        "extracted_answer": extracted,
        "score": score,
        "correct": score,
    }
    out: dict[str, Any] = {"docvqa_2026_acc": record}
    for category in CATEGORIES:
        out[f"docvqa_2026_{category}"] = score if doc_category == category else None
    return out


def _aggregate_category(results: list[Any], category: str) -> float:
    scores: list[float] = []
    for item in results:
        if item is None:
            continue
        if isinstance(item, dict):
            if item.get("doc_category") == category:
                scores.append(float(item["correct"]))
        else:
            scores.append(float(item))
    return sum(scores) / len(scores) if scores else float("nan")


def docvqa_2026_aggregate_acc(results: list[dict[str, Any]]) -> float:
    """Overall accuracy across all questions, logging per-category breakdown."""
    if not results:
        return float("nan")
    by_category: dict[str, list[float]] = defaultdict(list)
    for item in results:
        category = str(item.get("doc_category", ""))
        if category:
            by_category[category].append(float(item["correct"]))
    for category, scores in sorted(by_category.items()):
        avg = sum(scores) / len(scores)
        eval_logger.info(f"DocVQA 2026 {category}: {avg:.4f} ({len(scores)} questions)")
    return sum(float(r["correct"]) for r in results) / len(results)


def docvqa_2026_aggregate_business_report(results: list[Any]) -> float:
    return _aggregate_category(results, "business_report")


def docvqa_2026_aggregate_comics(results: list[Any]) -> float:
    return _aggregate_category(results, "comics")


def docvqa_2026_aggregate_engineering_drawing(results: list[Any]) -> float:
    return _aggregate_category(results, "engineering_drawing")


def docvqa_2026_aggregate_infographics(results: list[Any]) -> float:
    return _aggregate_category(results, "infographics")


def docvqa_2026_aggregate_maps(results: list[Any]) -> float:
    return _aggregate_category(results, "maps")


def docvqa_2026_aggregate_science_paper(results: list[Any]) -> float:
    return _aggregate_category(results, "science_paper")


def docvqa_2026_aggregate_science_poster(results: list[Any]) -> float:
    return _aggregate_category(results, "science_poster")


def docvqa_2026_aggregate_slide(results: list[Any]) -> float:
    return _aggregate_category(results, "slide")
