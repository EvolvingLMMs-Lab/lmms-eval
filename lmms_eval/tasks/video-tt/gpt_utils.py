import ast
import datetime
import json
import os
import sys
import time

import numpy as np
import openai
import requests
import yaml
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from openai import OpenAI

NUM_SECONDS_TO_SLEEP = 1

GPT_EVAL_MODEL_NAME = os.getenv("GPT_EVAL_MODEL_NAME", "Qwen2.5-72B-Instruct")

API_TYPE = os.getenv("API_TYPE", "openai")

QWEN_HTTP_CHAT_URL = os.getenv("QWEN_API_URL", None)

import ipaddress


def ip_port_to_url(ip, port):
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv4Address):
        url = f"http://{ip}:{port}"
    else:
        url = f"http://[{ip}]:{port}"
    return url


if API_TYPE == "openai":
    if "qwen" in GPT_EVAL_MODEL_NAME.lower():
        GPT_EVAL_MODEL_NAME = GPT_EVAL_MODEL_NAME.replace("Qwen/", "")
        API_URL = QWEN_HTTP_CHAT_URL
    else:
        API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def get_eval(question, answer, pred, max_tokens: int, retries: int = 1):
    global headers

    default_prompt = (
        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
        "------"
        "##INSTRUCTIONS: "
        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
        "- Consider synonyms or paraphrases as valid matches.\n"
        "- Evaluate the correctness of the prediction compared to the answer."
        f"Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {answer}\n"
        f"Predicted Answer: {pred}\n\n"
        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
    )

    if os.getenv("GPT_EVAL_PROMPT"):
        default_prompt = os.getenv("GPT_EVAL_PROMPT")
    messages = [
        {
            "role": "user",
            "content": default_prompt,
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
        pred = review_dict.get("pred", "no")
        score = review_dict.get("score", 0)
        return [pred, float(score)]
    except SyntaxError as e:
        eval_logger.error(f"Syntax error parsing the review string: {e}. Review content: {review}")
        return ["no", 0]
    except ValueError as e:
        eval_logger.error(f"Value error parsing the review string: {e}. Review content: {review}")
        return ["no", 0]
    except Exception as e:
        eval_logger.error(f"Unexpected error parsing the review string: {e}. Review content: {review}")
        return ["no", 0]


def gpt_score_proccess(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary
    """
    try:
        question = doc["question"]
        answer = doc["answer"]
        pred = result[0]

        # Assume get_eval returns a review and the model name, and parse_score parses this review
        review, model_name = get_eval(question, answer, pred, 1)
        scores = parse_score(review)
    except Exception as e:
        eval_logger.error(f"Error for Question ID: {doc.get('question_id', 'Unknown')}: {e}")
        question = doc["question"]
        answer = doc["answer"]
        pred = result[0]
        review = "Failed to Get a Proper Review."
        model_name = "Failed Request"
        scores = ["no", 0]
        # data_dict = {"video_id": doc["video_id"], "capability": capability, "pred_answer": pred_ans, "answer": doc["answer"]}

    data_dict = {"video_id": doc["video_id"], "capability": doc["capability"], "scores": scores, "correctness": scores[1], "answer": answer}

    return {f"videott_open_ended_score": data_dict}


# Factory into different aggregate
def aggregate_score(results, args):
    yes_count = 0
    no_count = 0
    total_score = 0

    # Iterate over the results to count correctness and sum scores
    for result_dict in results:
        if result_dict["Correctness"] == "yes":
            yes_count += 1
        else:
            no_count += 1
        total_score += result_dict["score"]

    # Calculate accuracy and average score
    accuracy = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    average_score = total_score / len(results) if results else 0
    eval_logger.info(f"Accuracy: {accuracy}")
    eval_logger.info(f"Average Score: {average_score}")
    return average_score


def aggregate_accuracy(results, args):
    yes_count = 0
    no_count = 0
    total_score = 0

    # Iterate over the results to count correctness and sum scores
    for result_dict in results:
        if result_dict["Correctness"] == "yes":
            yes_count += 1
        else:
            no_count += 1
        total_score += result_dict["score"]

    # Calculate accuracy and average score
    accuracy = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    average_score = total_score / len(results) if results else 0
    eval_logger.info(f"Accuracy: {accuracy}")
    eval_logger.info(f"Average Score: {average_score}")
    return accuracy * 100


def accuracy_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    # extract the first Character
    pred = results[0]
    if pred:
        pred = pred[0]

    # return {f"videomme_percetion_score": data_dict for metric in matrices}
    return {f"accuracy": pred == doc["answer"]}


def doc_to_text_with_options(doc, lmms_eval_specific_kwargs=None):
    option_prompt = "Please respond with only the letter of the correct answer."
    question = doc["question"]
    options = ast.literal_eval(doc["options"])
    full_prompt = question + "\n" + "\n".join(options) + "\n" + option_prompt
    return full_prompt
