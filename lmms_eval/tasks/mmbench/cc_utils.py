import logging
import yaml
import os
from pathlib import Path
import pandas as pd
import json

eval_logger = logging.getLogger("lmms-eval")
from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator

with open(Path(__file__).parent / "mmbench_cn.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

mmbench_evaluator = MMBench_Evaluator(sys_prompt=config["metadata"]["sys_prompt"])


def mmbench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mmbench_cn_cc_doc_to_text(doc, model_specific_prompt_kwargs=None):
    option_candidate = ["A", "B", "C", "D", "E"]
    options_prompt, options_dict = mmbench_evaluator.create_options_prompt(doc, option_candidate)

    data = {
        # "img": doc["image"],
        "question": doc["question"],
        "answer": doc.get("answer", None),
        "options": options_prompt,
        "category": doc["category"],
        "options_dict": options_dict,
        "index": doc["index"],
        "source": doc["source"],
    }

    query_prompt = f"{data['question']} {data['options']}"

    if model_specific_prompt_kwargs:
        query_prompt = f"{query_prompt}\n{model_specific_prompt_kwargs['post_prompt']}"

    return query_prompt


def mmbench_cn_cc_process_results(doc, results):
    model_response = results[0].strip()
    return {
        "submission": {
            "index": doc["index"],
            "question": doc["question"],
            "answer": doc["answer"],
            "prediction": model_response,
            "source": doc["source"],
            "category": doc["category"],
        }
    }


def mmbench_cn_cc_aggregate_results(results):
    df = pd.DataFrame(results)
    os.makedirs("./submissions", exist_ok=True)
    with pd.ExcelWriter("./submissions/mmbench_cn_cc_results.xlsx") as writer:
        df.to_excel(writer, index=False)
    eval_logger.info(f"Saved results to mmbench_cn_cc_results.xlsx")
