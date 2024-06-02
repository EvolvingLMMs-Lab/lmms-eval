import logging
import yaml
import os
from pathlib import Path
import pandas as pd
import json
from PIL import Image
from io import BytesIO
import base64
from lmms_eval.tasks.mmupd.mmupd_evals import MMUPD_Evaluator
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

eval_logger = logging.getLogger("lmms-eval")

with open(Path(__file__).parent / "mmupd.yaml", "r") as f:
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


mmupd_evaluator = MMUPD_Evaluator(sys_prompt=config["metadata"]["sys_prompt"], API_KEY=API_KEY, API_URL=API_URL, model_version=GPT_EVAL_MODEL_NAME)


def mmupd_doc_to_visual(doc):
    return [Image.open(BytesIO(base64.b64decode(doc["image"])))]


def mmupd_doc_to_text(doc, model_specific_prompt_kwargs=None):
    option_candidate = ["A", "B", "C", "D", "E"]
    options_prompt, options_dict = mmupd_evaluator.create_options_prompt(doc, option_candidate)

    data = {
        # "img": doc["image"],
        "question": doc["question"],
        "answer": doc.get("answer", None),
        "options": options_prompt,
        "category": doc["category"],
        "options_dict": options_dict,
        "index": doc["index"],
        "hint": doc["hint"],
        "source": doc["source"],
        "split": doc["split"],
    }

    query_prompt = f"{data['hint']}\n{data['question']}{data['options']}" if pd.notna(data["hint"]) and data["hint"] != "nan" else f"{data['question']}{data['options']}"

    if model_specific_prompt_kwargs:
        query_prompt = f"{query_prompt}{model_specific_prompt_kwargs['post_prompt']}"

    return query_prompt


def mmupd_process_results(doc, results):
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
            "type": doc["type"],
            "masked_answer": doc["masked_answer"]
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
            "type": doc["type"],
            "masked_answer": doc["masked_answer"]
        },
    }
    option_candidate = ["A", "B", "C", "D", "E"]
    for c in option_candidate:
        data["submission"][c] = doc.get(c, "nan")
        data["gpt_eval_score"][c] = doc.get(c, "nan")
    return data


def mmaad_base(results, args):
    return mmupd_results_eval(results, args, upd_type="aad", question_type="base")


def mmaad_option(results, args):
    return mmupd_results_eval(results, args, upd_type="aad", question_type="option")


def mmaad_instruction(results, args):
    return mmupd_results_eval(results, args, upd_type="aad", question_type="inst")


def mmiasd_base(results, args):
    return mmupd_results_eval(results, args, upd_type="iasd", question_type="base")


def mmiasd_option(results, args):
    return mmupd_results_eval(results, args, upd_type="iasd", question_type="option")


def mmiasd_instruction(results, args):
    return mmupd_results_eval(results, args, upd_type="iasd", question_type="inst")


def mmivqd_base(results, args):
    return mmupd_results_eval(results, args, upd_type="ivqd", question_type="base")


def mmivqd_option(results, args):
    return mmupd_results_eval(results, args, upd_type="ivqd", question_type="option")


def mmivqd_instruction(results, args):
    return mmupd_results_eval(results, args, upd_type="ivqd", question_type="inst")


def mmupd_results_eval(results, args, upd_type, question_type):

    print("============= MMUPD Bench Detailed Results =============")

    overall_acc_standard, category_acc_standard, standard_results_df = mmupd_evaluator.eval_result(results, eval_method="openai", upd_type=upd_type, question_type=question_type, eval_type="standard")
    overall_acc_upd, category_acc_upd, upd_results_df = mmupd_evaluator.eval_result(results, eval_method="openai", upd_type=upd_type, question_type=question_type, eval_type=upd_type)

    overall_acc_dual, category_acc_dual, dual_results_df = mmupd_evaluator.calculate_dual_acc(standard_results_df, upd_results_df)

    file_json = generate_submission_file(f"mmupd_results_{upd_type}_{question_type}.json", args)

    details_info = {
        "overall_acc_dual": overall_acc_dual,
        "category_acc_dual": category_acc_dual,
        "overall_acc_standard": overall_acc_standard,
        "category_acc_standard": category_acc_standard,
        "overall_acc_upd": overall_acc_upd,
        "category_acc_upd": category_acc_upd,
    }

    with open(file_json, "w") as f:
        json.dump(details_info, f)

    file_excel = generate_submission_file(f"mmupd_results_{upd_type}_{question_type}_dual.xlsx", args)
    dual_results_df.to_excel(file_excel, index=False)

    return overall_acc_dual * 100
