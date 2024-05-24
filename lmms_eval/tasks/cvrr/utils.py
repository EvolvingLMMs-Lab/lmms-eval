from decord import VideoReader, cpu
import numpy as np
import os
import sys
import datetime
import lmms_eval.tasks._task_utils.file_utils as file_utils
import json
import logging
import yaml
from pathlib import Path

import requests
import openai
from openai import OpenAI
import time
import ast

eval_logger = logging.getLogger("lmms-eval")

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

# Unzip all the zip files to HF HOME cache dir
HF_HOME = os.environ["HF_HOME"]
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir, "CVRR-ES/continuity_and_object_instance_count")


# Pass in video path here
# Can only work correctly with video llm
def cvrr_doc_to_visual(doc):
    video_path = doc["VideoID"]

    if doc["DimensionName"] == "Continuity and Object Instance Count":

        video_path = os.path.join(cache_dir, video_path)
        if os.path.exists(video_path):
            video_path = video_path
        else:
            sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# format the question
def cvrr_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]

    question = doc["Q"]

    return f"{pre_prompt}{question}{post_prompt}"


# format answer
def cvrr_doc_to_answer(doc):
    return doc["A"]


# Note: we process answer and gpt_eval seperately, in case gpt is not stable
# so we obtained a submission file for answer first
# and then feed the submission file to gpt for scoring


# Process result for evaluation
def cvrr_process_results(doc, result):
    pred = result[0]

    return {"submission": {"VideoID": doc["VideoID"], "Q": doc["Q"], "A": doc["A"], "pred": pred, "DimensionName": doc["DimensionName"]}}


def cvrr_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"cvrr-{task}-{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")

    return path


def get_eval(question, answer, pred, task, max_tokens: int, retries: int = 5):
    global headers

    if task == "continuity_and_object_instance_count":
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs. "
                "Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the correctness and accuracy of the predicted answer with the ground-truth.\n"
                "- Consider predictions with less specific details as correct evaluation, unless such details are explicitly asked in the question.\n",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Ground truth correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness."
                "Please generate the response in the form of a Python dictionary string with keys 'pred', 'score' and 'reason', where value of 'pred' is  a string of 'correct' or 'incorrect', value of 'score' is in INTEGER, not STRING and value of 'reason' should provide the reason behind the decision."
                "Only provide the Python dictionary string."
                "For example, your response should look like this: {'pred': 'correct', 'score': 4.8, 'reason': reason}.",
            },
        ]

    print(messages)

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # Raises HTTPError for bad responses
            try:
                response_data = response.json()  # Attempt to parse JSON
            except requests.exceptions.JSONDecodeError:
                eval_logger.error(f"JSON decode error on attempt {attempt + 1}. Response text: {response.text}")
                continue  # Skip to next retry
            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
        # Handle HTTP errors separately
        except requests.exceptions.HTTPError as e:
            eval_logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
        # Handle other requests-related errors
        except requests.exceptions.RequestException as e:
            eval_logger.error(f"Request exception on attempt {attempt + 1}: {e}")
        except Exception as e:
            eval_logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

        # Handle other unexpected errors
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
        else:  # If this was the last attempt, log and return empty
            eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
            return "", ""

    return "", ""


def parse_score(review):
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        review_dict = ast.literal_eval(review)
        correctness = review_dict.get("pred", "incorrect")
        score = review_dict.get("score", 0)
        reason = review_dict.get("reason", "")
        return correctness, float(score), reason
    except SyntaxError as e:
        eval_logger.error(f"Syntax error parsing the review string: {e}. Review content: {review}")
        return "incorrect", float(0), ""
    except ValueError as e:
        eval_logger.error(f"Value error parsing the review string: {e}. Review content: {review}")
        return "incorrect", float(0), ""
    except Exception as e:
        eval_logger.error(f"Unexpected error parsing the review string: {e}. Review content: {review}")
        return "incorrect", float(0), ""


def cvrr_print_scores(eval_file_path, args):
    # Load the predictions from the result file
    with open(eval_file_path, "r") as file:
        evaluated_list = json.load(file)

    score_file_name = "scores.json"
    path = file_utils.generate_submission_file(score_file_name, args)

    # Compute average score and final accuracy
    # Initialize counters
    yes_count = 0
    no_count = 0
    total_score = 0

    # Iterate over the results to count correctness and sum scores
    for result_list in evaluated_list:
        eval_dict = result_list[0]
        total_score += eval_dict["score"]

        if eval_dict["Correctness"] == "yes":
            yes_count += 1
        else:
            no_count += 1

    # Calculate accuracy and average score
    accuracy = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    average_score = total_score / len(evaluated_list) if evaluated_list else 0

    # Print the results
    print(f"Accuracy: {accuracy}")
    print(f"Average Score: {average_score}")

    # Write the processed data to the scores file
    with open(path, "w") as f:
        json.dump({"accuracy": accuracy, "average_score": average_score}, f, indent=4)

    eval_logger.info(f"Score file saved to {path}")


def cvrr_gpt_eval(result_file_path, args, task):
    """
    Process the result file containing predictions, score them using GPT,
    and save the results with added scores and correctness fields to a new file.

    Args:
        result_file_path: path to the JSON file with results to be evaluated
    """

    eval_file_name = "gpt_eval_result.json"
    eval_file_path = file_utils.generate_submission_file(eval_file_name, args)

    # Load the predictions from the result file
    with open(result_file_path, "r") as file:
        result_list = json.load(file)

    evaluated_results = []

    # Load the predictions from the result file
    with open(result_file_path, "r") as file:
        result_list = json.load(file)

    # Process each result to generate scores
    for data_dict in result_list:
        try:
            question = data_dict.get("Q", "")
            answer = data_dict.get("A", "")
            pred = data_dict.get("pred", "")

            # Assume get_eval returns a review and the model name, and parse_score parses this review
            review, model_name = get_eval(question, answer, pred, task, 64)
            correctness, score, reason = parse_score(review)
        except Exception as e:
            eval_logger.error(f"Error for Video Name: {data_dict.get('VideoID', 'Unknown')}: {e}")
            review = "Failed to Get a Proper Review."
            model_name = "Failed Request"
            score = 0
            correctness = "incorrect"
            reason = ""

        # Update the dictionary with the new entries
        eval_dict = {
            "pred": correctness,
            "score": score,
            "reason": reason,
        }
        result_dict = {
            "Q": question,
            "A": answer,
            "pred": pred,
        }
        updated_list = [eval_dict, result_dict]
        evaluated_results.append(updated_list)

    # Save the evaluated results to a new JSON file
    with open(eval_file_path, "w") as f:
        json.dump(evaluated_results, f, indent=4)

    return eval_file_path


def cvrr_aggregate_results(results, args):
    result_file_path = cvrr_aggregate_submissions(results, args, "continuity_and_object_instance_count")
    eval_file_path = cvrr_gpt_eval(result_file_path, args, "continuity_and_object_instance_count")
    cvrr_print_scores(eval_file_path, args)
