"""SlideVQA task utilities.

The Hugging Face rows contain one question, one answer, and up to 20 slide
images in page_1 ... page_20. We pack the deck into a few grid images so
multi-image models can answer with a short, directly scored response.
Grid packing and scoring follow the public SlideVQA evaluation convention used by VLMEvalKit.
"""

import math
import re
from typing import Any

from PIL import Image

from lmms_eval.api.metrics import levenshtein_distance

PAGE_COLUMNS = tuple(f"page_{index}" for index in range(1, 21))


def slidevqa_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    """Return SlideVQA pages as up to five grid images."""
    pages = [doc[column].convert("RGB") for column in PAGE_COLUMNS if doc.get(column) is not None]
    if not pages:
        raise ValueError(f"Slide deck {doc.get('deck_name')!r} has no images")
    return concat_images(pages)


def concat_images(images: list[Image.Image], max_concat: int = 5, column_num: int = 2) -> list[Image.Image]:
    """Concatenate deck pages into grids."""
    interval = max(math.ceil(len(images) / max_concat), 1)
    grids = []

    for start in range(0, len(images), interval):
        batch = images[start : start + interval]
        rows = math.ceil(len(batch) / column_num)
        grid = Image.new("RGB", (batch[0].width * column_num, batch[0].height * rows), "white")

        for index, image in enumerate(batch):
            grid.paste(image, ((index % column_num) * image.width, (index // column_num) * image.height))
        grids.append(grid)

    return grids


def slidevqa_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, str] | None = None) -> str:
    """Build the short-answer SlideVQA prompt."""
    kwargs = lmms_eval_specific_kwargs or {}
    return f"{kwargs.get('pre_prompt', '')}{doc['question']}{kwargs.get('post_prompt', '')}"


def slidevqa_doc_to_messages(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Build interleaved chat messages for multi-image models."""
    content = [{"type": "image", "url": image} for image in slidevqa_doc_to_visual(doc)]
    content.append({"type": "text", "text": slidevqa_doc_to_text(doc, lmms_eval_specific_kwargs)})
    return [{"role": "user", "content": content}]


def normalize_answer(answer: Any) -> str:
    """Normalize a SlideVQA answer."""
    if answer is None or (isinstance(answer, float) and math.isnan(answer)):
        return "not answerable"
    return re.sub("\n", "", str(answer)).lower()


def anls_score(answer: str, prediction: str, threshold: float = 0.5) -> float:
    """Compute thresholded ANLS for one answer/prediction pair."""
    length = max(len(answer), len(prediction))
    if length == 0:
        return 0.0
    score = 1.0 - levenshtein_distance(answer, prediction) / length
    return score if score > threshold else 0.0


def word_f1(answer: str, prediction: str) -> float:
    """Compute whitespace-token overlap F1."""
    answer_words = answer.strip().split()
    prediction_words = prediction.strip().split()
    if not answer_words or not prediction_words:
        return 0.0

    recall = sum(word in answer_words for word in prediction_words) / len(answer_words)
    precision = sum(word in answer_words for word in prediction_words) / len(prediction_words)
    if recall + precision <= 1e-4:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def slidevqa_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """Score one SlideVQA prediction."""
    answer = normalize_answer(doc.get("answer"))
    prediction = str(results[0]).lower()
    return {
        "anls": anls_score(answer, prediction),
        "em": float(answer.strip() == prediction.strip()),
        "f1": word_f1(answer, prediction),
    }
