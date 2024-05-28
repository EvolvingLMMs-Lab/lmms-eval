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
cache_dir = os.path.join(cache_dir, "Test_Videos")


# Pass in video path here
# Can only work correctly with video llm
def videochatgpt_doc_to_visual(doc):
    video_path = doc["video_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# format the question
def videochatgpt_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]

    question = doc["question"]

    return f"{pre_prompt}{question}{post_prompt}"


# format the question for consistency
def videochatgpt_doc_to_text_consistency(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]

    if "question_1" in doc:
        question = doc["question_1"]
    else:
        question = doc["question_2"]

    return f"{pre_prompt}{question}{post_prompt}"


# format answer
def videochatgpt_doc_to_answer(doc):
    return doc["answer"]


# Note: we process answer and gpt_eval seperately, in case gpt is not stable
# so we obtained a submission file for answer first
# and then feed the submission file to gpt for scoring


# Process result for evaluation in generic task
def videochatgpt_process_results_generic(doc, result):
    pred = result[0]

    return {
        "correctness": {"video_name": doc["video_name"], "Q": doc["question"], "A": doc["answer"], "pred": pred},
        "detailed_orientation": {"video_name": doc["video_name"], "Q": doc["question"], "A": doc["answer"], "pred": pred},
        "context": {"video_name": doc["video_name"], "Q": doc["question"], "A": doc["answer"], "pred": pred},
    }


# Process result for evaluation in temporal task
def videochatgpt_process_results_temporal(doc, result):
    pred = result[0]

    return {"submission": {"video_name": doc["video_name"], "Q": doc["question"], "A": doc["answer"], "pred": pred}}


# Process result for generation in consistency task
def videochatgpt_process_results_consistency(doc, result):
    pred = result[0]

    # if it is question_1, then assign prediction for the 1st question
    # else assign prediction for the 2nd question
    if doc["question_1"] != "None":
        return {"submission": {"video_name": doc["video_name"], "Q1": doc["question_1"], "A": doc["answer"], "pred1": pred}}
    else:
        return {"submission": {"video_name": doc["video_name"], "Q2": doc["question_2"], "A": doc["answer"], "pred2": pred}}


def videochatgpt_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"inference_results_videochatgpt_{task}_{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")

    return path


def videochatgpt_aggregate_submissions_consistency(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"inference_results_videochatgpt_{task}_{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    combined_results = []

    # Iterate over the results list in steps of 2
    for i in range(0, len(results), 2):
        # Merge the current dict with the next one
        first_dict = results[i]
        second_dict = results[i + 1] if i + 1 < len(results) else {}

        # If 'video_name' is the same in both and is the key we use to match them
        if first_dict.get("video_name") == second_dict.get("video_name"):
            # Combine q2 and pred2 from the even dict into the odd dict
            first_dict["Q2"] = second_dict.get("Q2")
            first_dict["pred2"] = second_dict.get("pred2")
            combined_results.append(first_dict)

    # Save the combined results to a file
    with open(path, "w") as f:
        json.dump(combined_results, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")

    return path


def get_eval_generic(question, answer, pred, task, max_tokens: int, retries: int = 5):
    global headers

    if task == "correctness":
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
                "- The predicted answer must be factually accurate and align with the video content.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the factual accuracy of the prediction compared to the answer.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {''score': 4.8}.",
            },
        ]
    elif task == "detailed_orientation":
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine its level of detail, considering both completeness and specificity. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
                "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {''score': 4.8}.",
            },
        ]
    elif task == "context":
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the contextual understanding of generative outputs for video-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if the generated response aligns with the overall context of the video content. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Evaluate whether the predicted answer aligns with the overall context of the video content. It should not provide information that is out of context or misaligned.\n"
                "- The predicted answer must capture the main themes and sentiments of the video.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Provide your evaluation of the contextual understanding of the prediction compared to the answer.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a contextual understanding score where the contextual understanding score is an integer value between 0 and 5, with 5 indicating the highest level of contextual understanding. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is contextual understanding score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {''score': 4.8}.",
            },
        ]
    elif task == "temporal":
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the temporal understanding of generative outputs for video-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they correctly reflect the temporal sequence of events in the video content. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the temporal consistency between the predicted answer and the correct answer. The predicted answer should correctly reflect the sequence of events or details as they are presented in the video content.\n"
                "- Consider synonyms or paraphrases as valid matches, but only if the temporal order is maintained.\n"
                "- Evaluate the temporal accuracy of the prediction compared to the answer.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a temporal accuracy score where the temporal accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of temporal consistency. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the temporal accuracy score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {''score': 4.8}.",
            },
        ]

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


def get_eval_consistency(question1, question2, answer, pred1, pred2, max_tokens: int, retries: int = 5):
    global headers

    messages = [
        {
            "role": "system",
            "content": "You are an intelligent chatbot designed for evaluating the consistency of generative outputs for similar video-based question-answer pairs. "
            "You will be given two very similar questions, a common answer common to both the questions and predicted answers for the two questions ."
            "Your task is to compare the predicted answers for two very similar question, with a common correct answer and determine if they are consistent. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Focus on the consistency between the two predicted answers and the correct answer. Both predicted answers should correspond to the correct answer and to each other, and should not contain any contradictions or significant differences in the conveyed information.\n"
            "- Both predicted answers must be consistent with each other and the correct answer, in terms of the information they provide about the video content.\n"
            "- Consider synonyms or paraphrases as valid matches, but only if they maintain the consistency in the conveyed information.\n"
            "- Evaluate the consistency of the two predicted answers compared to the correct answer.",
        },
        {
            "role": "user",
            "content": "Please evaluate the following video-based question-answer pair:\n\n"
            f"Question 1: {question1}\n"
            f"Question 2: {question2}\n"
            f"Correct Answer: {answer}\n"
            f"Predicted Answer to Question 1: {pred1}\n"
            f"Predicted Answer to Question 2: {pred2}\n\n"
            "Provide your evaluation only as a consistency score where the consistency score is an integer value between 0 and 5, with 5 indicating the highest level of consistency. "
            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the consistency score in INTEGER, not STRING."
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {''score': 4.8}.",
        },
    ]

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
        score = review_dict.get("score", 0)
        return float(score)
    except SyntaxError as e:
        eval_logger.error(f"Syntax error parsing the review string: {e}. Review content: {review}")
        return 0
    except ValueError as e:
        eval_logger.error(f"Value error parsing the review string: {e}. Review content: {review}")
        return 0
    except Exception as e:
        eval_logger.error(f"Unexpected error parsing the review string: {e}. Review content: {review}")
        return 0


def videochatgpt_print_scores(eval_file_path, args, task):
    # Load the predictions from the result file
    with open(eval_file_path, "r") as file:
        evaluated_list = json.load(file)

    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    score_file_name = f"scores_videochatgpt_{task}_{now_date_time}.json"
    path = file_utils.generate_submission_file(score_file_name, args)

    # Compute average score
    total_score = 0

    # Iterate over the results to sum scores
    for result_dict in evaluated_list:
        total_score += result_dict["score"]

    # Calculate accuracy and average score
    average_score = total_score / len(evaluated_list) if evaluated_list else 0

    # Write the processed data to the scores file
    with open(path, "w") as f:
        json.dump({"average_score": average_score}, f, indent=4)

    eval_logger.info(f"Score file saved to {path}")

    return average_score


def videochatgpt_gpt_eval(result_file_path, args, task):
    """
    Process the result file containing predictions, score them using GPT,
    and save the results with added scores and correctness fields to a new file.

    Args:
        result_file_path: path to the JSON file with results to be evaluated
    """
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    eval_file_name = f"gpt_eval_result_videochatgpt_{task}_{now_date_time}.json"
    eval_file_path = file_utils.generate_submission_file(eval_file_name, args)

    # Load the predictions from the result file
    with open(result_file_path, "r") as file:
        result_list = json.load(file)

    evaluated_results = []

    # Load the predictions from the result file
    with open(result_file_path, "r") as file:
        result_list = json.load(file)

    # Process each result to generate scores
    # If task is consistency (2 questions with 2 answers)
    if task == "consistency":
        for data_dict in result_list:
            try:
                question1 = data_dict.get("Q1", "")
                question2 = data_dict.get("Q2", "")
                answer = data_dict.get("A", "")
                pred1 = data_dict.get("pred1", "")
                pred2 = data_dict.get("pred2", "")

                # Assume get_eval returns a review and the model name, and parse_score parses this review
                review, model_name = get_eval_consistency(question1, question2, answer, pred1, pred2, 64)
                score = parse_score(review)
            except Exception as e:
                eval_logger.error(f"Error for Video Name: {data_dict.get('video_name', 'Unknown')}: {e}")
                review = "Failed to Get a Proper Review."
                model_name = "Failed Request"
                score = 0

            # Update the dictionary with the new entries
            updated_dict = {
                "video_name": data_dict["video_name"],
                "score": score,
                "Q1": question1,
                "Q2": question2,
                "A": answer,
                "pred1": pred1,
                "pred2": pred2,
            }
            evaluated_results.append(updated_dict)
    # If task is correctness, context, detail, temporal (1 question with 1 answer)
    else:
        # Process each result to generate scores
        for data_dict in result_list:
            try:
                question = data_dict.get("Q", "")
                answer = data_dict.get("A", "")
                pred = data_dict.get("pred", "")

                # Assume get_eval returns a review and the model name, and parse_score parses this review
                review, model_name = get_eval_generic(question, answer, pred, task, 64)
                score = parse_score(review)
            except Exception as e:
                eval_logger.error(f"Error for Video Name: {data_dict.get('video_name', 'Unknown')}: {e}")
                review = "Failed to Get a Proper Review."
                model_name = "Failed Request"
                score = 0

            # Update the dictionary with the new entries
            updated_dict = {
                "video_name": data_dict["video_name"],
                "score": score,
                "Q": question,
                "A": answer,
                "pred": pred,
            }
            evaluated_results.append(updated_dict)

    # Save the evaluated results to a new JSON file
    with open(eval_file_path, "w") as f:
        json.dump(evaluated_results, f, indent=4)

    return eval_file_path


# Factory into different aggregate
def videochatgpt_aggregate_correctness(results, args):
    result_file_path = videochatgpt_aggregate_submissions(results, args, "correctness")
    eval_file_path = videochatgpt_gpt_eval(result_file_path, args, "correctness")
    average_score = videochatgpt_print_scores(eval_file_path, args, "correctness")
    return average_score


def videochatgpt_aggregate_detailed_orientation(results, args):
    result_file_path = videochatgpt_aggregate_submissions(results, args, "detailed_orientation")
    eval_file_path = videochatgpt_gpt_eval(result_file_path, args, "detailed_orientation")
    average_score = videochatgpt_print_scores(eval_file_path, args, "detailed_orientation")
    return average_score


def videochatgpt_aggregate_context(results, args):
    result_file_path = videochatgpt_aggregate_submissions(results, args, "context")
    eval_file_path = videochatgpt_gpt_eval(result_file_path, args, "context")
    average_score = videochatgpt_print_scores(eval_file_path, args, "context")
    return average_score


def videochatgpt_aggregate_temporal(results, args):
    result_file_path = videochatgpt_aggregate_submissions(results, args, "temporal")
    eval_file_path = videochatgpt_gpt_eval(result_file_path, args, "temporal")
    average_score = videochatgpt_print_scores(eval_file_path, args, "temporal")
    return average_score


def videochatgpt_aggregate_consistency(results, args):
    result_file_path = videochatgpt_aggregate_submissions_consistency(results, args, "consistency")
    eval_file_path = videochatgpt_gpt_eval(result_file_path, args, "consistency")
    average_score = videochatgpt_print_scores(eval_file_path, args, "consistency")
    return average_score
