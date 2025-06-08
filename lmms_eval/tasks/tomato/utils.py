import base64
import datetime
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI
from PIL import Image

random.seed(42)


from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))


SYSTEM_PROMPT = "You are an expert in understanding dynamics of objects."
NUM_FRAMES = 16
MAX_ITER = 5
USE_GPT_PARSER = False  # whether to use gpt parser from TOMATO's source code, else use lmms_eval parser

if USE_GPT_PARSER:
    eval_logger.info(f"Using GPT parser for TOMATO task. The max iteration is set to {MAX_ITER}. " "If the response is not a valid answer, it will try to use GPT to parse the response.")
    API_TYPE = os.getenv("API_TYPE", "azure")
    if API_TYPE == "openai":
        endpoint = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
        subscription_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
        client = OpenAI(
            api_key=subscription_key,
            api_base=endpoint,
            api_version="2025-01-01-preview",
        )

    elif API_TYPE == "azure":
        endpoint = os.getenv("ENDPOINT_URL", "https://haku-chat.openai.azure.com/")
        deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=subscription_key,
            api_version="2025-01-01-preview",
        )
    else:
        raise ValueError(f"Unsupported API_TYPE: {API_TYPE}. Please set it to 'openai' or 'azure'.")

eval_logger.info(f"Using {NUM_FRAMES} frames for TOMATO task. Please set the max_num_frames=16 in model_args for the result reported in the TOMATO paper: https://arxiv.org/pdf/2410.23266.")


"""
It's important to note that should set the max_num_frames of the model_args to 16, as reported from https://arxiv.org/pdf/2410.23266.
Sample command to run the evaluation task for Qwen2-VL model on TOMATO dataset:
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model qwen2_vl\
    --model_args max_num_frames=16,system_prompt="You are an expert in understanding dynamics of objects." \
    --tasks tomato \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix tomato \
    --output_path ./logs/
"""

with open(Path(__file__).parent / "tomato.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
cache_dir = os.path.join(hf_home, config["dataset_kwargs"]["cache_dir"])


def construct_prompt(question: str, options: list, num_frames: int) -> Tuple:
    """
    Args:
        question (str): question in the dataset
        options (list): list of options
        num_frames (int): number of frames extracted from the video

    Returns:
        prompt (str): well-constructed prompt
        all_choices (list): list of options (A, B, C, ...)
        index2ans (dict): dictionary of option-answer mapping
    """

    all_choices = [f"{chr(65 + i)}" for i in range(len(options))]
    index2ans = {all_choices[i]: options[i] for i in range(len(options))}

    prompt = f"""You will be provided with {num_frames} separate frames uniformly sampled from a video, the frames are provided in chronological order of the video. Analyze these frames and provide the answer to the question about the video content. Answer the multiple-choice question about the video content. 

You must use these frames to answer the multiple-choice question; do not rely on any externel knowledge or commonsense. 

<question> 
{question} 
</question>

<options> 
{index2ans} 
</options>

Even if the information in these separate frames is not enough to answer the question, PLEASE TRY YOUR BEST TO GUESS AN ANSWER WHICH YOU THINK WOULD BE THE MOST POSSIBLE ONE BASED ON THE QUESTION. 

DO NOT GENERATE ANSWER SUCH AS 'NOT POSSIBLE TO DETERMINE.' 
"""

    return prompt, all_choices, index2ans


def read_video(video_path: str, total_frames: int):
    # Create a VideoCapture object
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("Could not open video file.")
    try:
        # Initialize a list to store base64 encoded frames
        base64_frames = []

        # Read frames in a loop
        while True:
            success, frame = video.read()
            if not success:
                break  # No more frames or error occurred

            # Encode the frame as a JPEG
            _, buffer = cv2.imencode(".jpg", frame)

            # Convert the image to base64 string
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            base64_frames.append(frame_base64)

        if total_frames == 1:
            selected_indices = [np.random.choice(range(total_frames))]
        else:
            selected_indices = np.linspace(0, len(base64_frames) - 1, total_frames, dtype=int)

        selected_base64_frames = [base64_frames[index] for index in selected_indices]

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps else 0

        return selected_base64_frames, duration
    finally:
        # Release the video capture object
        video.release()


def tomato_doc_to_visual(doc):
    """
    Return the path to the video only
    """
    video_paths = []
    # Get the video
    abs_video_path = os.path.join(cache_dir, doc["video_path"])
    abs_video_path = os.path.expanduser(abs_video_path)
    if os.path.exists(abs_video_path):
        video_paths.append(abs_video_path)
    else:
        eval_logger.error(f"Video path does not exist: {abs_video_path}")
    return video_paths


def tomato_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Process the document to a prompt for video + audio inputs
    """
    prompt, all_choices, index2ans = construct_prompt(question=doc["question"], options=doc["options"], num_frames=NUM_FRAMES)
    return prompt


def gpt_parser(response, all_choices, index2ans):
    prompt = f"""You are given a response, a list of multiple-choice options, and a index2answer mapping. You are required to extract the letter option from the GPT. 
    
    response: {response}

    all_choices: {all_choices}

    index2answer: {index2ans}

Only output the single parsed letter from the response. No other texts are needed. 

If you think no options can match the index2answer dictionary, randomly select one letter. 

Your extracted letter is: 
"""
    prompt_message = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    params = {
        "model": "gpt-4o",
        "messages": prompt_message,
        "max_tokens": 16,
        "temperature": 0.0,
    }
    response = client.chat.completions.create(**params)
    response = response.choices[0].message.content

    return response


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"{choice}" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice} " in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        # pred_index = random.choice(all_choices)
        pred_index = "A"
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def pre_parser(response, all_choices, index2ans):
    parsed_response = ""
    response = response.strip()

    # preprocess matches
    full_choices = [f"{k}: {v}" for k, v in index2ans.items()]
    pattern = r"^Answer is:?[\(]?([A-Fa-f])[\)]?$"
    match = re.match(pattern, response)

    # exact match single letter
    if len(response) == 1 and response.upper() in all_choices:
        parsed_response = response.upper()

    # exact match of the choice
    elif response.upper() in full_choices:
        parsed_response = response[0].upper()

    # regex match of "Answer is: A", "Answer is (A)", etc
    elif match:
        parsed_response = match.group(1).upper()

    return parsed_response


def tomato_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case av_odyssey score), value: metric value
    """
    respone = results[0]
    _, all_choices, index2ans = construct_prompt(question=doc["question"], options=doc["options"], num_frames=NUM_FRAMES)
    optionized_list = [f"{chr(65 + i)}. {option}" for i, option in enumerate(doc["options"])]
    gt = optionized_list[doc["answer"]]
    if USE_GPT_PARSER:
        parsed_response = pre_parser(response=respone, all_choices=all_choices, index2ans=index2ans)
        if parsed_response not in all_choices:
            curr_iter = 0
            while curr_iter < MAX_ITER:
                response_candidate = gpt_parser(respone, all_choices, index2ans)
                if response_candidate in all_choices:
                    parsed_response = response_candidate
                    break
                curr_iter += 1
            if parsed_response not in all_choices:
                parsed_response = random.choice(all_choices)

    else:
        parsed_response = parse_multi_choice_response(respone, all_choices, index2ans)
    score = 1.0 if parsed_response == gt[0] else 0.0

    reason_type = doc["reason_type"]
    demo_type = doc["demonstration_type"]
    key_name = "tomato_score"
    # Note: the key name here is very important. It decides which aggregation function will receive the results
    # We note down the question id/category to help us aggregate the results later
    return {key_name: {"question_id": doc["id"], "score": score, "reason_type": reason_type, "demonstration_type": demo_type}}


def tomato_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    reason2scores = defaultdict(list)
    demo2scores = defaultdict(list)
    num_corrects = 0
    num_total = 0
    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        reason_type = result["reason_type"]
        demonstration_type = result["demonstration_type"]
        if reason_type not in reason2scores:
            reason2scores[reason_type] = []
        reason2scores[reason_type].append(score)
        if demonstration_type not in demo2scores:
            demo2scores[demonstration_type] = []
        demo2scores[demonstration_type].append(score)
        num_corrects += score
        num_total += 1
    # calculate the average score for each reason type
    reason_scores = {reason: sum(scores) / len(scores) for reason, scores in reason2scores.items()}
    demo_scores = {demo: sum(scores) / len(scores) for demo, scores in demo2scores.items()}
    for reason, score in reason_scores.items():
        eval_logger.info(f"Evaluation on reasoning type: {reason}, Score: {score:.6f}")
    for demo, score in demo_scores.items():
        eval_logger.info(f"Evaluation on demonstration type: {demo}, Score: {score:.6f}")

    overall_score = num_corrects / num_total if num_total > 0 else 0.0
    eval_logger.info(f"Overall performance (across all questions):  {overall_score:.6f}")
    return overall_score
