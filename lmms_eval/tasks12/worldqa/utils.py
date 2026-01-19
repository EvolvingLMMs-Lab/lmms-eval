import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml

import lmms_eval.tasks._task_utils.file_utils as file_utils
from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.tasks.worldqa.worldqa_mc_evaluator import WorldQA_MC_Evaluator

NUM_SECONDS_TO_SLEEP = 5

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

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

eval_prompt = """You are an AI assistant who will help me to evaluate the quality of the candidate responses belonging to a question. The quality of the responses should be referred to the ground truth response.

Some criterion
- Response that perfectly reflect the key points in the ground truth: 1 point
- Response that reflect none of the key points in the ground truth: 0 point
- Some part in the response are correct but other parts in the response are contrast to the ground truth: 0.3 point
- Some part in the response are correct but some parts in the ground truth are not mentioned in the response: 0.5 point
- Some part in the response are correct but other parts in the response are not mentioned in the ground truth: 0.5 point

Your output should be in the following format:
Keypoint in the ground truth response:
XXX
Rationale:
XXXX
Point:
1/0.5/0.3/0

Let's begin this task:
question: {question}
ground truth: {answer}
candidate: {candidate}
"""


def get_eval(question: str, ground_truth: str, candidate: str, max_tokens: int, retries: int = 5):
    global headers

    content = eval_prompt.format(question=question, answer=ground_truth, candidate=candidate)

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
HF_HOME = os.getenv("HF_HOME", "~/.cache/huggingface/")
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir, "videos")


from loguru import logger as eval_logger


# Pass in video path here
# Can only work correctly with video llm
def worldqa_doc_to_visual(doc):
    video_path = doc["video_idx"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# This is the place where you format your question
def worldqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    question = doc["question"]
    if "option" in doc:
        for op in doc["option"]:
            question += "\n" + op
        # post_prompt = "\nAnswer with the option's letter from the given choices directly."

    return f"{pre_prompt}{question}{post_prompt}"


def worldqa_doc_to_answer(doc):
    return doc["answer"]


# If it is mc, keep the option for exact match
def worldqa_doc_to_answer_mc(doc):
    return doc["answer"].split(".")[0].strip()


# If it is mc ppl, keep the option str for perplexity base matching
def worldqa_doc_to_answer_mc_ppl(doc):
    return doc["answer"].split(".")[1].strip()


# An example of showing how to custom metric
# Your metric name should have the same key name in your return dict
def worldqa_process_results(doc, result):
    pred = result[0]
    content = eval_prompt.format(question=doc["question"], answer=doc["answer"], candidate=pred)
    eval_answer, model_name = get_eval(question=doc["question"], ground_truth=doc["answer"], candidate=pred, max_tokens=1024)
    return {
        "submission": {"pred": pred, "question_idx": doc["question_idx"], "object_description": doc["object_description"], "answer": doc["answer"], "eval_answer": eval_answer, "gpt_prompt": content},
        "gpt_eval": {"pred": pred, "question_idx": doc["question_idx"], "object_description": doc["object_description"], "answer": doc["answer"], "eval_answer": eval_answer, "gpt_prompt": content},
    }


def worldqa_process_results_mc(doc, result):
    pred = result[0]
    data = {
        "gpt_eval": {"pred": pred, "question_idx": doc["question_idx"], "object_description": doc["object_description"], "answer": doc["answer"], "option": doc["option"], "question": doc["question"]},
    }
    return data


def worldqa_aggregate_mc_eval(results):
    score = 0
    evaluator = WorldQA_MC_Evaluator(API_KEY=API_KEY, API_URL=API_URL)
    for result in results:
        score += evaluator.evaluate(result)
    return score / len(results)


def worldqa_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"worldqa-{task}-{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Submission file saved to {path}")


def worldq_gen_gpt_eval(results, args):
    score = 0
    for result in results:
        eval_answer = result["eval_answer"]
        eval_score = eval_answer.split("\n")[-1].strip()
        try:
            eval_score = float(eval_score)
        except (ValueError, TypeError, AttributeError):
            eval_score = 0.0
        score += eval_score

    return score / len(results)


# Factory into different aggregate
def worldqa_aggregate_gen(results, args):
    worldqa_aggregate_submissions(results, args, "Generation")


def worldqa_aggregate_mc(results, args):
    worldqa_aggregate_submissions(results, args, "MC")


def worldqa_aggregate_mc_ppl(results, args):
    worldqa_aggregate_submissions(results, args, "MC_PPL")


def worldqa_doc_to_choice(doc):
    return [op.split(".")[1].strip() for op in doc["option"]]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            question = doc["question"]
            if "option" in doc:
                for op in doc["option"]:
                    question += "\n" + op
            # Regex to extract multiple choice options from the question
            multiple_choices_regex = re.compile(r"\b([A-Z])\.\s+([^\n]*)")
            matches = multiple_choices_regex.findall(question)

            # Build regex patterns and mappings for each choice
            for m in matches:
                choice_text = m[1].strip()
                fallback_regexes.append(f"{re.escape(choice_text)}")
                choice_to_alpha[choice_text] = next_alpha

                next_alpha = chr(ord(next_alpha) + 1)

            # Compile regex to match any of the extracted choices
            fallback_regex = re.compile("|".join(fallback_regexes))

            # Process each response
            filtered = []
            for resp in r:
                # Remove any punctuation and extra spaces
                cleaned_resp = re.sub(r"[^\w\s]", "", resp).strip()
                # Try to match cleaned response with the choice text
                match = fallback_regex.search(cleaned_resp)
                if match and match.group() in choice_to_alpha:
                    # Map the matched choice text back to its corresponding letter
                    filtered.append(choice_to_alpha[match.group()])
                else:
                    # If no match, return the cleaned response
                    filtered.append(cleaned_resp)

            filtered_resps.append(filtered[0])

        return filtered_resps
