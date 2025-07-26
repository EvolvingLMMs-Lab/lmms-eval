import base64
import os
import time
from copy import deepcopy
from http import HTTPStatus
from io import BytesIO
from pathlib import Path

import numpy as np
import yaml

# Set up a logger
from loguru import logger as eval_logger
from lmms_eval.llm_judge import Request, ServerConfig, get_server

NUM_SECONDS_TO_SLEEP = 5
dir_path = os.path.dirname(os.path.realpath(__file__))

judge_rules = "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image shown to you. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. Assume assistant 1 always receive a score of 10 and is the correct answer.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."

with open(Path(__file__).parent / "_default_template_wilder_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")
API_TYPE = config["metadata"]["api_type"]

# Initialize the judge server
server_config = ServerConfig(
    model_name=GPT_EVAL_MODEL_NAME,
    temperature=0.0,
    max_tokens=1024
)
server = get_server(server_name=API_TYPE, config=server_config)


def get_chat_response(base64_image, prompt, max_retries=5, wait_time=10):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }
    ]

    # Update server config with specific parameters for this request
    custom_config = ServerConfig(
        model_name=GPT_EVAL_MODEL_NAME,
        temperature=0.0,
        max_tokens=1024
    )

    for attempt in range(max_retries):
        try:
            # Create a Request object for the unified judge API
            request = Request(
                messages=messages,
                images=[base64_image],  # Pass the base64 image
                config=custom_config
            )
            
            # Use the unified judge API
            response = server.evaluate(request)
            
            content = response.content if response.content else ""
            return content, response.model_used
        except Exception as e:
            eval_logger.warning(f"Request failed on attempt {attempt+1}: {e}")
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                eval_logger.error(f"Failed to get response after {max_retries} attempts")
                return "", GPT_EVAL_MODEL_NAME


def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            eval_logger.debug(f"Can not split: {review}. Returning [-1, -1]")
            return [-1, -1]
    except Exception as e:
        eval_logger.debug(f"Error: {e}. Returning [-1, -1]")
        return [-1, -1]


def llava_process_results(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case coco_bleu), value: metric value
    """
    try:
        question = doc.get("Question", "")
        ans1 = doc.get("Answer", "")
        ans2 = result[0] if result else ""
        content = f"[Question]\n{question}\n\n" + f"[Assistant 1]\n{ans1}\n\n[End of Assistant 1]\n\n" + f"[Assistant 2]\n{ans2}\n\n[End of Assistant 2]\n\n" f"[System]\n{judge_rules}\n\n"
        visuals = llava_doc_to_visual(doc)
        image_path = doc["image"]
        base64_image = image_to_base64(image_path)
        review, model_name = get_chat_response(base64_image, content)
        scores = parse_score(review)
    except Exception as e:
        eval_logger.error(f"Error for Question ID: {doc.get('question_id', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = "Failed Request"
        scores = [-1, -1]

    data_dict = {"question": question, "ans1": ans1, "ans2": ans2, "review": review, "scores": scores, "eval_model": model_name, "content": content}
    # return {"gpt_eval_llava_all": review_dict}
    return {"gpt_eval_llava_all": data_dict}


def llava_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def llava_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['Question']}{post_prompt}"


def llava_all_aggregation(results):
    return llava_aggregation(results, "all")


def llava_aggregation(results, category):
    try:
        scores = []
        for result in results:
            if -999 in result["scores"]:
                continue
            scores.append(result["scores"])

        stats = np.asarray(scores).mean(0).tolist()
        stats = [round(x, 3) for x in stats]
        # gpt4_score_percentage = stats[0] * 10
        # model_score_percentage = stats[1] * 10
        # eval_logger.info(f"Category: {category}")
        # eval_logger.info(f"GPT4 Score: {gpt4_score_percentage:.1f}%")
        # eval_logger.info(f"Model Score: {model_score_percentage:.1f}%")
        # eval_logger.info("=========================")
        return round(stats[1] / stats[0] * 100, 1)
    except Exception as e:
        eval_logger.info(f"Error in llava_aggregation: {e}, and in category: {category}")
        return None
