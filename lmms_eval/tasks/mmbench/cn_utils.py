import yaml
import os
from pathlib import Path
import pandas as pd
import json
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

from loguru import logger as eval_logger
from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

with open(Path(__file__).parent / "mmbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]
API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
else:
    API_URL = "YOUR_API_URL"
    API_KEY = "YOUR_API_KEY"


mmbench_evaluator = MMBench_Evaluator(sys_prompt=config["metadata"]["sys_prompt"], API_KEY=API_KEY, API_URL=API_URL, model_version=GPT_EVAL_MODEL_NAME)


def mmbench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mmbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    option_candidate = ["A", "B", "C", "D", "E"]
    options_prompt, options_dict = mmbench_evaluator.create_options_prompt(doc, option_candidate)

    data = {
        # "img": doc["image"],
        "question": doc["question"],
        "answer": doc.get("answer", None),
        "options": options_prompt,
        "category": doc["category"],
        "L2-category": doc["L2-category"],
        "options_dict": options_dict,
        "index": doc["index"],
        "hint": doc["hint"],
        "source": doc["source"],
        "split": doc["split"],
    }

    query_prompt = f"{data['hint']} {data['question']} {data['options']}" if pd.notna(data["hint"]) else f"{data['question']} {data['options']}"

    if lmms_eval_specific_kwargs:
        query_prompt = f"{query_prompt}\n{lmms_eval_specific_kwargs['post_prompt']}"

    return query_prompt


def mmbench_process_results(doc, results):
    model_response = results[0].strip()
    data = {
        "gpt_eval_score": {
            "index": doc["index"],
            "question": doc["question"],
            "answer": doc["answer"],
            "prediction": model_response,
            "hint": doc["hint"],
            "source": doc["source"],
            "split": doc["split"],
            "category": doc["category"],
            "L2-category": doc["L2-category"],
        },
        "submission": {
            "index": doc["index"],
            "question": doc["question"],
            "answer": doc["answer"],
            "prediction": model_response,
            "hint": doc["hint"],
            "source": doc["source"],
            "split": doc["split"],
            "category": doc["category"],
            "L2-category": doc["L2-category"],
        },
    }
    option_candidate = ["A", "B", "C", "D", "E"]
    for c in option_candidate:
        data["submission"][c] = doc.get(c, "nan")
        data["gpt_eval_score"][c] = doc.get(c, "nan")
    return data


def mmbench_aggregate_dev_results_eval(results, args):
    print(f"============= MMBench-CN(Dev) Detailed Results =============")
    overall_acc, category_acc, l2_category_acc = mmbench_evaluator.eval_result(results, eval_method="openai")
    file = generate_submission_file("mmbench_cn_dev_results.json", args)
    details_info = {
        "overall_acc": overall_acc,
        "category_acc": category_acc,
        "l2_category_acc": l2_category_acc,
    }
    with open(file, "w") as f:
        json.dump(details_info, f)
    return overall_acc * 100


def mmbench_aggregate_dev_results(results, args):
    df = pd.DataFrame(results)
    excel_write_path = generate_submission_file("mmbench_cn_dev_results.xlsx", args)
    with pd.ExcelWriter(excel_write_path) as writer:
        df.to_excel(writer, index=False)
    eval_logger.info(f"Saved results to {excel_write_path}")


def mmbench_aggregate_test_results(results, args):
    df = pd.DataFrame(results)
    excel_write_path = generate_submission_file("mmbench_cn_test_results.xlsx", args)
    with pd.ExcelWriter(excel_write_path) as writer:
        df.to_excel(writer, index=False)
    eval_logger.info(f"Saved results to {excel_write_path}")
