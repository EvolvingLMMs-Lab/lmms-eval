import base64
import yaml
import os
from pathlib import Path
import requests
import logging
import time
from copy import deepcopy
import numpy as np
from http import HTTPStatus
from io import BytesIO

eval_logger = logging.getLogger("lmms-eval")

try:
    import dashscope
except:
    eval_logger.debug("Dashcope not found, make sure you install dashscope to use qwen vl")

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

GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]
API_TYPE = config["metadata"]["api_type"]

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

elif API_TYPE == "qwen_vl":
    API_URL = os.getenv("QWEN_ENDPOINT", "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation")
    API_KEY = os.getenv("DASHSCOPE_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def get_chat_response(base64_image, prompt, max_retries=5, wait_time=10):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": [
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
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"], GPT_EVAL_MODEL_NAME
        except requests.exceptions.RequestException as e:
            eval_logger.warning(f"Request failed on attempt {attempt+1}: {e}")
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                eval_logger.error(f"Failed to get response after {max_retries} attempts")
                return "", GPT_EVAL_MODEL_NAME
        except Exception as e:
            eval_logger.error(f"Error on attempt {attempt+1}: {e}")
            return "", GPT_EVAL_MODEL_NAME


def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def qwen_multimodal_conversation_call(text_content, image_content, retries=5):
    """Simple single round multimodal conversation call."""
    messages = [{"role": "user", "content": [{"image": image_content}, {"text": text_content}]}]
    for attempt in range(retries):
        try:
            response_data = dashscope.MultiModalConversation.call(model=GPT_EVAL_MODEL_NAME, messages=messages)
            # The response status_code is HTTPStatus.OK indicate success,
            # otherwise indicate request is failed, you can get error code
            # and message from code and message.
            content = response_data["output"]["choices"][0]["message"]["content"][0]["text"].strip()
            if content != "":
                return content, GPT_EVAL_MODEL_NAME
            break  # If successful, break out of the loop
        except Exception as e:
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:  # If we have retries left, sleep and then continue to next attempt
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:  # If this was the last attempt, log and return empty
                eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                return "", ""
    return "", ""


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
        question = doc.get("question", "")
        ans1 = doc.get("gpt4v_answer", "")
        ans2 = result[0] if result else ""
        content = f"[Question]\n{question}\n\n" + f"[Assistant 1]\n{ans1}\n\n[End of Assistant 1]\n\n" + f"[Assistant 2]\n{ans2}\n\n[End of Assistant 2]\n\n" f"[System]\n{judge_rules}\n\n"
        visuals = llava_doc_to_visual(doc)
        if API_TYPE == "qwen_vl":
            file_path = os.path.join(dir_path, f"tmp_{doc['question_id']}.jpg")
            visuals[0].save(file_path)
            image_content = "file://" + file_path
            review, model_name = qwen_multimodal_conversation_call(content, image_content=image_content)
            os.remove(file_path)
        elif API_TYPE == "openai":
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


def llava_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = model_specific_prompt_kwargs.get("pre_prompt", "")
    post_prompt = model_specific_prompt_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"


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
