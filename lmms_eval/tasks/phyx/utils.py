import base64
import io
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from lmms_eval.tasks.phyx.phyx_evals import PhyXEvaluator


def load_phyx_config():
    with open(Path(__file__).parent / "phyx.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for line in raw_data:
            if "!function" not in line:
                safe_data.append(line)
        return yaml.safe_load("".join(safe_data))


config = load_phyx_config()
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


# ============================================================================
# Visual CoT Versions
# ============================================================================

# Optics: 根据题目画光路图
OPTICS_GEN_PROMPT = (
    "Based on this optics problem, draw a light ray diagram that helps solve the problem. "
    "Show the paths of light rays, including incident rays, reflected rays, refracted rays, "
    "and any relevant angles or focal points as needed by the problem."
)

# Mechanics: 根据题目画受力分析图
MECHANICS_GEN_PROMPT = (
    "Based on this mechanics problem, draw a free body diagram (force analysis diagram) "
    "that helps solve the problem. "
    "Show all the forces acting on the object(s), including gravity, normal force, friction, "
    "tension, applied forces, etc., with arrows indicating direction and relative magnitude."
)


def phyx_doc_to_text_optics_cot(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for PhyX Optics task."""
    question = (
        "In addition to the original image, you are also given an auxiliary "
        "light ray diagram to help you solve the problem.\n\n"
        + doc["question"]
    )
    return f"[GEN_PROMPT]{OPTICS_GEN_PROMPT}[/GEN_PROMPT][QUESTION]{question}[/QUESTION]"


def phyx_doc_to_text_mechanics_cot(doc, lmms_eval_specific_kwargs=None):
    """Visual CoT prompt for PhyX Mechanics task."""
    question = (
        "In addition to the original image, you are also given an auxiliary "
        "free body diagram (force analysis diagram) to help you solve the problem.\n\n"
        + doc["question"]
    )
    return f"[GEN_PROMPT]{MECHANICS_GEN_PROMPT}[/GEN_PROMPT][QUESTION]{question}[/QUESTION]"
