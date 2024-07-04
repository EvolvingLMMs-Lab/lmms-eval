import os
import re
import sys
import datetime
import lmms_eval.tasks._task_utils.file_utils as file_utils
from lmms_eval.filters.extraction import ExtendedRegexFilter
import json
import yaml
from pathlib import Path
import requests
import time
from loguru import logger as eval_logger

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

NUM_SECONDS_TO_SLEEP = 5
GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]
API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }

eval_prompt = """You are an AI assistant who will help me to evaluate the quality of a model response to a few candidate ground truth answers.

Some criterion
- Response that perfectly reflect the meaning of the ground truth: 1 point
- Response that reflect none of the key points in the ground truth: 0 point
- Some part in the response are correct but some parts in the ground truth are not mentioned in the response: 0.5 point
- Some part in the response are correct but other parts in the response are not mentioned in the ground truth: 0.5 point

Here're some examples about the scoring criterion and format:
model response: Steam Cleaning Services
ground truth: ["steam clean", "steam clean", "cleaning", "car", "steam clean"],
Point: 1

model response: A cowboy action shooter.
ground truth: ["man"]
Point: 1

model response: I'm sorry, but I can't assist with that request.
ground truth: ["quality"]
Point: 0

Let's begin this task:
model response: {model_response}
ground truth: {ground_truth}
Point:"""


def get_eval(model_response: str, ground_truth: str, max_tokens: int, retries: int = 5):
    global headers
    content = eval_prompt.format(model_response=model_response, ground_truth=ground_truth)

    messages = [
        {"role": "user", "content": content},
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
            break  # If successful, break out of the loop

        except Exception as e:
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:  # If we have retries left, sleep and then continue to next attempt
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:  # If this was the last attempt, log and return empty
                eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                return "", ""
    return "", ""


# A bit ugly here
# But the idea is that we will unzip all the zip files
# To HF HOME cache dir
# And load it here
HF_HOME = os.environ["HF_HOME"]
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir)


# Pass in video path here
# Can only work correctly with video llm
def mix_evals_video2text_doc_to_visual(doc):
    video_path = doc["video_path"]
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# This is the place where you format your question
def mix_evals_video2text_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]

    user_prompt = doc["prompt"]

    if "options" in doc:
        option_prompt = "Here are the options:\n"
        for idx, option in enumerate(doc["options"]):
            char_idx = chr(ord("A") + idx)
            option = option.strip()
            option_prompt += f"{char_idx}. {option}\n"

        option_prompt = option_prompt.rstrip("\n")
        user_prompt = f"{user_prompt}\n{option_prompt}"

    if pre_prompt:
        user_prompt = f"{pre_prompt}\n{user_prompt}"

    if post_prompt:
        user_prompt = f"{user_prompt}\n{post_prompt}"
    return user_prompt


OPEN_CONVS_PROMPT = """{PRE}
{FIRST}
{POST}
"""


def mix_evals_video2text_doc_to_text_open_convs(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]

    filtered_first_turn = re.sub(r"<video_[0-9]+>", "", doc["first_turn_user_prompt"])
    return OPEN_CONVS_PROMPT.format(
        PRE=pre_prompt,
        POST=post_prompt,
        FIRST=filtered_first_turn,
    )


MODEL_CONVS_PROMPT = """{FIRST}
{MODEL_RESPONSE}
{PRE}
{SECOND}
{POST}
"""


def mix_evals_video2text_doc_to_text_open_2nd_convs(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]

    return MODEL_CONVS_PROMPT.format(
        PRE=pre_prompt,
        POST=post_prompt,
        FIRST=doc["first_turn_user_prompt"],
        SECOND=doc["second_turn_user_prompt"],
        MODEL_RESPONSE=doc["model_response"],
    )


def mix_evals_video2text_process_results_open_convs(doc, result):
    pred = result[0]
    return {"submission": {"pred": pred, "question_idx": doc["question_index"], "first_turn_video_caption": doc["first_turn_video_caption"], "target": ""}}


def mix_evals_video2text_process_results_freeform(doc, result):
    pred = result[0]
    ground_truth_str = ", ".join([f'"{gt}"' for gt in doc["target"]])
    ground_truth_str = f"[{ground_truth_str}]"
    content = eval_prompt.format(model_response=pred, ground_truth=ground_truth_str)
    eval_answer, model_name = get_eval(model_response=pred, ground_truth=ground_truth_str, max_tokens=1024)
    return {
        "submission": {"pred": pred, "question_idx": doc["question_index"], "target": doc["target"], "eval_answer": eval_answer, "gpt_prompt": content},
        "gpt_eval": {"pred": pred, "question_idx": doc["question_index"], "target": doc["target"], "eval_answer": eval_answer, "gpt_prompt": content},
    }


def mix_evals_video2text_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"mix_evals_video2text_{task}-{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Submission file saved to {path}")


def mix_evals_video2text_gpt_eval(results, args):
    score = 0
    for result in results:
        eval_answer = result["eval_answer"]
        eval_score = re.search(r"([0-9.]+)", eval_answer).group(1)
        try:
            eval_score = float(eval_score)
        except Exception as e:
            eval_logger.error(f"Error parsing eval_score: {e}")
            eval_score = 0.0
        score += eval_score

    return score / len(results)


# Factory into different aggregate
def mix_evals_video2text_aggregate_gen(results, args):
    mix_evals_video2text_aggregate_submissions(results, args, "OpenConvs")


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        filtered_resps = []

        for r, doc in zip(resps, docs):
            # Regex to directly extract the option letter from the model response
            option_letter_regex = re.compile(r"\b([A-Z])\.\s+([^\n]*)")

            # Process each response
            filtered = []
            for resp in r:
                # Try to match the option letter at the start of the response
                match = option_letter_regex.match(resp)
                if match:
                    # If a match is found, append the matched letter
                    filtered.append(match.group(1))
                else:
                    # If no match, return the original response
                    filtered.append(resp)

            # Assuming we need the first response that matches or the original response
            filtered_resps.append(filtered[0])

        return filtered_resps
