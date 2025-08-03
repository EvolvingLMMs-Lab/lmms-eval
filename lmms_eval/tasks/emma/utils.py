import datetime
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

eval_logger = logging.getLogger("lmms-eval")

dir_name = os.path.dirname(os.path.abspath(__file__))


emma_config = {
    "Strategy_Instruction": {"CoT": "Please solve the problem step by step.", "Directly": "Please ensure that your output only contains the final answer without any additional content (such as intermediate reasoning steps)."},
    "multi_choice_format": '{context}\n{question}\n{options}\nAnswer with the option\'s letter from the given choices and put the letter in one "\\boxed{{}}". ',
    "open_ended_format": '{context}\n{question}\nAnswer the question using a single word or phrase and put the answer in one "\\boxed{{}}". ',
}

with open(Path(__file__).parent / "emma_all.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
cache_dir = os.path.join(hf_home, config["dataset_kwargs"]["cache_dir"])


print(config)
eval_logger.info(f"Using '{config['metadata']['strategy']}' strategy for EMMA task.")


def build_query(sample):
    """Build the text query by combining the context, question and options. The <image_n> token is still there"""
    context = sample["context"]
    question = sample["question"]
    example = ""
    res_dict = {}
    strategy = config["metadata"]["strategy"]
    if sample["type"].lower() == "multiple choice":
        options = sample["options"]
        start_chr = "A"
        for option in options:
            example += f"{start_chr}: {option}\n"
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt_sample_structure = emma_config["multi_choice_format"]
        empty_prompt = empty_prompt_sample_structure.format(context=context, question=question, options=example)
        if strategy == "CoT":
            res_dict["query"] = empty_prompt + emma_config["Strategy_Instruction"]["CoT"]
        else:
            res_dict["query"] = empty_prompt + emma_config["Strategy_Instruction"]["Directly"]

        res_dict["gt_content"] = options[ord(sample["answer"].upper()) - ord("A")]
    else:
        empty_prompt_sample_structure = emma_config["open_ended_format"]
        empty_prompt = empty_prompt_sample_structure.format(context=context, question=question)
        if strategy == "CoT":
            res_dict["query"] = empty_prompt + emma_config["Strategy_Instruction"]["CoT"]
        else:
            res_dict["query"] = empty_prompt + emma_config["Strategy_Instruction"]["Directly"]
        res_dict["gt_content"] = sample["answer"]

    # append existing key and value in data
    res_dict.update(sample)
    return res_dict


def create_message(sample):
    query = sample["query"]
    all_contents = []
    matches = re.findall(r"<(image_\d+)>", query)
    split_text = re.split(r"<image_\d+>", query)
    for i, fragment in enumerate(split_text):
        if fragment.strip():
            all_contents.extend([{"type": "text", "text": fragment}])
        if i < len(matches):
            if sample[matches[i]]:
                img_base64 = encode_image_to_base64(sample[matches[i]])
                all_contents.extend([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}])
            else:
                eval_logger.error(f"The image token {matches[i]} is in the query, but there is no corresponding image provided by the data")

    messages = [{"role": "user", "content": all_contents}]
    return messages


def emma_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    res_dict = build_query(doc)
    return res_dict["query"]


def emma_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    res_dict = build_query(doc)
    return create_visual(res_dict)


def emma_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    res_dict = build_query(doc)
    return create_message(res_dict)


def emma_process_results(results, lmms_eval_specific_kwargs=None):
    pass


def emma_aggregate_results(results):
    pass
