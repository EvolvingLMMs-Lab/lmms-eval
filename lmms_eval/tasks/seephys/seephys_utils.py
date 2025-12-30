# seephys_utils.py
import ast
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import yaml
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks.seephys.seephys_evals import SeephysEvaluator, load_seephys_config

config = load_seephys_config()
seephys_evaluator = SeephysEvaluator()


def seephys_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    if "images" not in doc or not doc["images"]:
        eval_logger.warning(f"Document index {doc.get('index', 'N/A')} has no 'images' field or it is empty.")
        return []

    image_list = doc["images"]

    if not isinstance(image_list, list):
        raise TypeError(f"Expected 'images' field to be a list, but got {type(image_list)} for index {doc.get('index', 'N/A')}")

    processed_images = []
    for i, image in enumerate(image_list):
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected item {i} in 'images' list to be PIL.Image, but got {type(image)} for index {doc.get('index', 'N/A')}")

        if image.mode != "RGB":
            image = image.convert("RGB")
        processed_images.append(image)

    return processed_images


def seephys_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict = None) -> str:
    question = doc.get("question", "")
    if not isinstance(question, str) or question.lower() == "nan":
        question = ""

    lang = doc.get("language", "English")
    sig_figs = doc.get("sig_figs")

    if lang == "English":
        question += "\nPlease answer this question with reasoning. First output your reasoning process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
    else:
        question += "\n请用推理来回答这个问题。首先在<think></think>标签中输出推理过程，然后在<answer></answer>标签中输入最终答案。"

    try:
        if sig_figs and (isinstance(sig_figs, (str, int, float)) and not np.isnan(float(sig_figs))):
            sf_str = str(int(float(sig_figs)))
            if lang == "English":
                question += f" The final answer should retain {sf_str} significant figures."
            else:
                question += f" 最终答案应保留{sf_str}位有效数字。"
    except (ValueError, TypeError):
        pass

    return question


def seephys_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    results: list of strings returned by the model generation for this doc (usually one item)
    This function will:
      - pick the first result as prediction (strip)
      - insert into doc["prediction"]
      - choose quick_extract path or LLM judger path depending on config
      - return either filtered_resps (if predict_only) or eval_results dict
    """
    if not results:
        eval_logger.error(f"Received empty results list for index {doc.get('index', 'N/A')}.")
        prediction = ""
    else:
        prediction_candidates = [r for r in results if isinstance(r, str) and r.strip() != ""]
        if prediction_candidates:
            prediction = prediction_candidates[0].strip()
        else:
            prediction = (results[0] or "").strip()

    doc["prediction"] = prediction
    doc["answer"] = str(doc.get("answer", ""))

    quick_extract = config.get("metadata", {}).get("quick_extract", False)

    true_false = 0
    if quick_extract:
        tmp = seephys_evaluator.Seephys_process_line(doc)
        true_false = tmp.get("match", 0)
    else:
        llm_tmp = seephys_evaluator.Seephys_auxeval(doc)
        true_false = int(llm_tmp.get("res", 0))

    eval_result = {
        "index": doc.get("index", -1),
        "true_false": true_false,
        "level": doc.get("level"),
        "subject": doc.get("subject"),
        "language": doc.get("language"),
        "img_category": doc.get("img_category"),
        "vision_relevance": doc.get("vision_relevance"),
        "answer": doc.get("answer", ""),
    }

    if config.get("predict_only", False):
        return {"filtered_resps": [prediction]}

    return {"eval_results": eval_result}


def seephys_aggregate_results(results: List[Dict[str, Any]]) -> float:
    if not results:
        eval_logger.warning("Aggregating empty results list. Returning 0.0")
        return 0.0

    hit = [int(x.get("true_false", 0)) for x in results]
    Overall_acc = float(np.mean(hit)) if len(hit) > 0 else 0.0
    return Overall_acc
