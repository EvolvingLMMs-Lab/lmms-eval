import base64
import io
import os
import re
import string
from collections import defaultdict

from loguru import logger as eval_logger
from PIL import Image

dir_name = os.path.dirname(os.path.abspath(__file__))

eval_type_dict = {
    "coarse perception": ["image scene and topic", "image style & quality", "image emotion"],
    "fine-grained perception": ["object counting", "recognition", "localization"],
    "instance reasoning": ["single-instance reasoning", "cross-instance attribute reasoning", "cross-instance relation reasoning"],
    "logical reasoning": ["code & sequence reasoning", "diagram reasoning", "common reasoning"],
    "science & technology": ["biology & chemistry & physics", "electronics & energy & mechanical eng.", "geography & earth science & agriculture"],
    "math": ["geometry", "numeric commonsense and calculation", "statistical reasoning"],
}


replace_prompt = " Please answer yes or no."


def mmstar_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mmstar_oc_doc_to_visual(doc):
    """
    Opencompass version of MMStar
    https://huggingface.co/datasets/morpheushoc/MMStar_opencompass
    """
    byte_string = doc["image"]
    img_data = base64.b64decode(byte_string)
    image = Image.open(io.BytesIO(img_data))
    return [image]


def mmstar_oc_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Opencompass version of MMStar: https://huggingface.co/datasets/morpheushoc/MMStar_opencompass
    Modified from: https://github.com/open-compass/VLMEvalKit/blob/19c0e386c0967936b5ab4357abdabd670ba5d361/vlmeval/vlm/qwen3_vl/prompt.py#L93
    """
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc["question"]
    question = question.replace("<image 1>", "")
    options = {cand: doc[cand] for cand in string.ascii_uppercase if cand in doc}

    options_prompt = "Options:\n"
    for key, item in options.items():
        options_prompt += f"{key}. {item}\n"

    prompt = f"{pre_prompt}{question}\n"
    prompt += options_prompt
    prompt += f"{post_prompt}"
    prompt = prompt.rstrip()
    return prompt


def mmstar_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def exact_match(pred, gt):
    """Extract MCQ letter from prediction and compare to ground truth."""
    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

    gt_letter = gt.strip().upper()
    pred_letter = extract_mcq_answer(pred, choices=["A", "B", "C", "D"])
    return 1.0 if pred_letter == gt_letter else 0.0


def exact_match_ko(pred, gt):
    """Extract MCQ letter from Korean prediction and compare to ground truth."""
    from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

    gt_letter = gt.strip().upper()
    pred_letter = extract_mcq_answer(pred, choices=["A", "B", "C", "D"])
    return 1.0 if pred_letter == gt_letter else 0.0


def mmstar_process_results_ko(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0]
    gt = doc["answer"]

    score = exact_match_ko(pred, gt)
    category = doc["category"]
    l2_category = doc["l2_category"]
    return {category: {"question_id": doc["index"], "l2_category": l2_category, "score": score}, "average": {"question_id": doc["index"], "l2_category": l2_category, "score": score}}


def mmstar_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = results[0]
    gt = doc["answer"]

    score = exact_match(pred, gt)
    category = doc["category"]
    l2_category = doc["l2_category"]
    return {category: {"question_id": doc["index"], "l2_category": l2_category, "score": score}, "average": {"question_id": doc["index"], "l2_category": l2_category, "score": score}}


def mmstar_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    l2_category_scores = defaultdict(list)
    for result in results:
        score = result["score"]
        l2_category = result["l2_category"]
        l2_category_scores[l2_category].append(score)

    l2_category_avg_score = {}
    for l2_category, scores in l2_category_scores.items():
        avg_score = sum(scores) / len(scores)
        l2_category_avg_score[l2_category] = avg_score
        eval_logger.info(f"{l2_category}: {avg_score:.2f}")

    avg_score = sum(l2_category_avg_score.values()) / len(l2_category_avg_score)
    return avg_score


# --- mmstar_hybrid -----------------------------------------------------------
# Perception categories keep the direct prompt; reasoning ones get a CoT prompt
# ending in "Answer: $LETTER".

HYBRID_COT_CATEGORIES = {"logical reasoning", "science & technology", "math"}

# Last marker wins, so reasoning about options before committing is fine.
_ANSWER_LINE_RE = re.compile(r"(?<!\w)answer\s*(?:is\b|[=:：])\s*\(?([A-D])\)?(?!\w)", flags=re.IGNORECASE)
# Fallback for models that mark the answer but ignore the wording. Kept local:
# reasoning_utils.extract_boxed_answer would pull in math_verify for one letter.
_BOXED_LETTER_RE = re.compile(r"\\boxed\{\s*\(?([A-D])\)?\s*\}", flags=re.IGNORECASE)


def mmstar_hybrid_perception_docs(dataset):
    return dataset.filter(lambda category: category not in HYBRID_COT_CATEGORIES, input_columns="category")


def mmstar_hybrid_reasoning_docs(dataset):
    return dataset.filter(lambda category: category in HYBRID_COT_CATEGORIES, input_columns="category")


def extract_cot_answer(response):
    """Read the letter a CoT response committed to, else "" (scores 0)."""
    text = str(response or "").strip()
    if not text:
        return ""

    matches = _ANSWER_LINE_RE.findall(text)
    if matches:
        return matches[-1].upper()

    boxed = _BOXED_LETTER_RE.findall(text)
    if boxed:
        return boxed[-1].upper()

    return ""


def mmstar_cot_process_results(doc, results):
    """``mmstar_process_results`` on the strictly parsed letter."""
    return mmstar_process_results(doc, [extract_cot_answer(results[0])])
