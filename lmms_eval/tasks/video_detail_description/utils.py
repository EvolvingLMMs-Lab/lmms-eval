from decord import VideoReader, cpu
import requests
import time
import ast
import os
import sys
import datetime
import lmms_eval.tasks._task_utils.file_utils as file_utils
import json
import logging
import yaml
from pathlib import Path

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

# A bit ugly here
# But the idea is that we will unzip all the zip files
# To HF HOME cache dir
# And load it here
HF_HOME = os.environ["HF_HOME"]
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, "datasets", cache_dir)
cache_dir = os.path.join(cache_dir, "evaluation/Test_Videos")

eval_logger = logging.getLogger("lmms-eval")


# Pass in video path here
# Can only work correctly with video llm
def video_detail_description_doc_to_visual(doc):
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
def video_detail_description_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]

    question = doc["question"]
    if "question_1" in doc:
        for op in doc["option"]:
            question += "\n" + op
        post_prompt = "\nAnswer with the option's letter from the given choices directly."

    return f"{pre_prompt}{question}{post_prompt}"


def video_detail_description_doc_to_answer(doc):
    return doc["answer"]


# Process result for evaluation in generic task
def video_detail_description_process_results_generic(doc, result):
    pred = result[0]

    return {"gpt_eval": {"video_name": doc["video_name"], "Q": doc["question"], "A": doc["answer"], "pred": pred}}


def video_detail_description_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"video_detail_description-{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")


def get_eval_generic(question, answer, pred, max_tokens: int, retries: int = 5):
    global headers

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
            "For example, your response should look like this: {'score': 4.8}.",
        },
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        # "response_format": {"type": "json_object"},
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

        if "Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt." in json.loads(response.content)["error"]["message"]:
            eval_logger.error(f"Repetitive patterns in prompt. Drop this data.")
            return "", ""

        # Handle other unexpected errors
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
        else:  # If this was the last attempt, log and return empty
            eval_logger.error(f"All {retries} attempts failed.")
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


def gpt_eval(results):
    evaluated_results = []

    for data_dict in results:
        try:
            question = data_dict.get("Q", "")
            answer = data_dict.get("A", "")
            pred = data_dict.get("pred", "")

            # Assume get_eval returns a review and the model name, and parse_score parses this review
            review, model_name = get_eval_generic(question, answer, pred, 64)
            score = parse_score(review)
        except Exception as e:
            eval_logger.error(f"Error for Video Name: {data_dict.get('video_name', 'Unknown')}: {e}")
            review = "Failed to Get a Proper Review."
            model_name = ""
            score = 0

        # Update the dictionary with the new entries
        updated_dict = {
            "video_name": data_dict["video_name"],
            "review": review,
            "score": score,
            # "Q": question,
            # "A": answer,
            # "pred": pred,
        }
        evaluated_results.append(updated_dict)

    return evaluated_results


def video_detail_description_gen_gpt_eval(results, args):
    eval_results = gpt_eval(results)
    score = 0
    for result in eval_results:
        eval_score = result["score"]
        try:
            eval_score = float(eval_score)
        except:
            eval_score = 0.0
        score += eval_score

    return score / len(results)
