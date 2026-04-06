import ast
from collections import defaultdict

import datasets
from loguru import logger as eval_logger
from PIL import Image

MTVQA_PROMPT_SUFFIX = "\nAnswer the question using a word or phrase in the language of the question."


def _parse_qa_pairs(value):
    if isinstance(value, list):
        parsed_pairs = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed_value = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            eval_logger.warning("Failed to parse MTVQA qa_pairs: {}", exc)
            return []
        parsed_pairs = parsed_value if isinstance(parsed_value, list) else []
    else:
        return []

    normalized_pairs = []
    for pair in parsed_pairs:
        if not isinstance(pair, dict):
            continue
        question = str(pair.get("question", "")).strip()
        answer = str(pair.get("answer", "")).strip()
        if question and answer:
            normalized_pairs.append({"question": question, "answer": answer})
    return normalized_pairs


def mtvqa_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    flattened_docs = []
    for idx, doc in enumerate(dataset):
        doc_dict = dict(doc)
        sample_id = str(doc_dict.get("id", "")).strip() or f"mtvqa_{idx}"

        category = str(doc_dict.get("lang", "")).strip() or "unknown"
        qa_pairs = _parse_qa_pairs(doc_dict.get("qa_pairs"))

        for qa_idx, qa in enumerate(qa_pairs):
            flattened_docs.append(
                {
                    "question_id": f"{sample_id}_{qa_idx}",
                    "id": sample_id,
                    "category": category,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "image": doc_dict.get("image"),
                }
            )

    if not flattened_docs:
        eval_logger.warning("[mtvqa] No samples found after flattening qa_pairs.")
        return dataset.select(range(0))

    eval_logger.info("[mtvqa] Loaded {} QA pairs from {} images.", len(flattened_docs), len(dataset))
    return datasets.Dataset.from_list(flattened_docs)


def _to_rgb_image(image_value):
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            from io import BytesIO

            return Image.open(BytesIO(image_value["bytes"])).convert("RGB")
        if image_value.get("path"):
            return Image.open(image_value["path"]).convert("RGB")
    raise TypeError(f"Unsupported MTVQA image payload type: {type(image_value)}")


def mtvqa_doc_to_visual(doc):
    image_value = doc.get("image")
    if image_value is None:
        sample_id = str(doc.get("id", "")).strip()
        raise KeyError(f"Missing MTVQA image payload for sample id: {sample_id}")
    return [_to_rgb_image(image_value)]


def mtvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", MTVQA_PROMPT_SUFFIX)
    question = str(doc["question"]).strip()
    return f"{pre_prompt}{question}{post_prompt}"


def mtvqa_process_results(doc, results):
    prediction = str(results[0]) if results else ""
    pred = prediction.strip().lower().replace(".", "")
    answer = str(doc["answer"]).strip().lower().replace(".", "")
    score = 1.0 if answer in pred else 0.0

    return {
        "mtvqa_score": {
            "category": doc["category"],
            "score": score,
        }
    }


def mtvqa_aggregate_results(results):
    category_scores = defaultdict(list)

    for result in results:
        category = str(result.get("category", "Unknown"))
        score = float(result.get("score", 0.0))
        category_scores[category].append(score)
        category_scores["Average"].append(score)

    for category in sorted(category_scores.keys()):
        scores = category_scores[category]
        if not scores:
            continue
        eval_logger.info("MTVQA {}: {:.2f}", category, (sum(scores) / len(scores)) * 100)

    average_scores = category_scores.get("Average", [])
    if not average_scores:
        return 0.0
    return (sum(average_scores) / len(average_scores)) * 100
