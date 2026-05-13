import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.reasoning_utils import (
    extract_anwser_tag,
    make_reasoning_doc_to_messages,
    make_reasoning_process_results,
)
from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator

GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")
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


@lru_cache(maxsize=1)
def _load_mmbench_sys_prompt():
    with open(Path(__file__).parent.parent / "mmbench.yaml", "r") as f:
        raw_data = f.readlines()
    safe_data = [line for line in raw_data if "!function" not in line]
    config = yaml.safe_load("".join(safe_data))
    return config["metadata"]["sys_prompt"]


def mmbench_cn_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    sys_prompt = _load_mmbench_sys_prompt()
    mmbench_evaluator = MMBench_Evaluator(sys_prompt=sys_prompt, API_KEY=API_KEY, API_URL=API_URL, model_version=GPT_EVAL_MODEL_NAME)
    option_candidate = ["A", "B", "C", "D", "E"]
    options_prompt, _ = mmbench_evaluator.create_options_prompt(doc, option_candidate)

    query_prompt = f"{doc['hint']} {doc['question']} {options_prompt}" if str(doc["hint"]) != "nan" and doc["hint"] else f"{doc['question']} {options_prompt}"

    return query_prompt


def mmbench_en_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    mmbench_evaluator = MMBench_Evaluator(sys_prompt="", API_KEY=API_KEY, API_URL=API_URL, model_version=GPT_EVAL_MODEL_NAME)
    option_candidate = ["A", "B", "C", "D", "E"]
    options_prompt, _ = mmbench_evaluator.create_options_prompt(doc, option_candidate)

    query_prompt = f"{doc['hint']} {doc['question']} {options_prompt}" if str(doc["hint"]) != "nan" and doc["hint"] else f"{doc['question']} {options_prompt}"

    return query_prompt


def mmbench_doc_to_visual(doc):
    num_image = int(os.environ.get("NUM_IMAGE", "1"))

    if num_image == 1:
        return [doc["image"].convert("RGB")]
    elif num_image == 2:
        return [doc["image"].convert("RGB"), doc["image"].convert("RGB")]
    else:
        raise ValueError(f"num_image must be 1 or 2, got {num_image}")


mmbench_cn_doc_to_messages = make_reasoning_doc_to_messages(mmbench_doc_to_visual, mmbench_cn_doc_to_text)
mmbench_en_doc_to_messages = make_reasoning_doc_to_messages(mmbench_doc_to_visual, mmbench_en_doc_to_text)
mmbench_process_results = make_reasoning_process_results("mmbench", mmbench_cn_doc_to_text)


def mmbench_process_results_test(doc, results):
    """Process results for test splits, extracting answer from <answer> tag for submission."""
    model_response = results[0].strip()
    extracted_answer = extract_anwser_tag(model_response).strip()

    data = {
        "submission": {
            "index": doc["index"],
            "question": doc["question"],
            "prediction": model_response,
            "answer": extracted_answer,
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
    return data


def mmbench_aggregate_test_results_cn(results, args):
    df = pd.DataFrame(results)
    excel_write_path = generate_submission_file("mmbench_cn_test_reasoning_results.xlsx", args)
    with pd.ExcelWriter(excel_write_path) as writer:
        df.to_excel(writer, index=False)
    eval_logger.info(f"Saved results to {excel_write_path}")


def mmbench_aggregate_test_results_en(results, args):
    df = pd.DataFrame(results)
    excel_write_path = generate_submission_file("mmbench_en_test_reasoning_results.xlsx", args)
    with pd.ExcelWriter(excel_write_path) as writer:
        df.to_excel(writer, index=False)
    eval_logger.info(f"Saved results to {excel_write_path}")
