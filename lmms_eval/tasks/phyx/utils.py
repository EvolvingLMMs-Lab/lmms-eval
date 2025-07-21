import base64
import io
from pathlib import Path

import numpy as np
import yaml
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks.phyx.phyx_evals import PhyXEvaluator

with open(Path(__file__).parent / "phyx.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


phyx_evaluator = PhyXEvaluator()


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def phyx_doc_to_visual(doc):
    image = decode_base64_to_image(doc["image"])
    return [image]


def phyx_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    query_prompt = doc["question"]
    return query_prompt


def phyx_process_results_mc(doc, results):
    prediction = results[0].strip()
    doc["prediction"] = str(prediction)
    doc["answer"] = str(doc["answer"])
    if config["metadata"]["quick_extract"]:
        tmp = phyx_evaluator.PhyX_process_line_MC(doc)
        true_false = tmp["match"]
    else:
        llm_tmp = phyx_evaluator.PhyX_auxeval_MC(doc)
        true_false = llm_tmp["res"]

    eval_result = {
        "index": doc["index"],
        "true_false": true_false,
        "category": doc["category"],
        "answer": doc["answer"],
    }

    return {
        "eval_results": eval_result,
    }


def phyx_process_results(doc, results):
    prediction = results[0].strip()
    doc["prediction"] = str(prediction)
    doc["answer"] = str(doc["answer"])
    if config["metadata"]["quick_extract"]:
        tmp = phyx_evaluator.PhyX_process_line(doc)
        true_false = tmp["match"]
    else:
        llm_tmp = phyx_evaluator.PhyX_auxeval(doc)
        true_false = llm_tmp["res"]

    eval_result = {
        "index": doc["index"],
        "true_false": true_false,
        "category": doc["category"],
        "answer": doc["answer"],
    }

    return {
        "eval_results": eval_result,
    }


def phyx_aggregate_results(results):
    hit = [x["true_false"] for x in results]
    Overall_acc = np.mean(hit)
    return Overall_acc
