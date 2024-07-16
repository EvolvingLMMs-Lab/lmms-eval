import os
import re
import requests
import time
import json

from collections import defaultdict
from rouge import Rouge
import yaml
from pathlib import Path
from loguru import logger as eval_logger
from PIL import Image

spot_the_diff = ["Spot-the-Diff", "Birds-to-Words", "CLEVR-Change"]
image_edit_instruct = ["IEdit", "HQ-Edit", "MagicBrush"]
visual_story_telling = ["AESOP", "FlintstonesSV", "PororoSV", "VIST"]
visual_cloze = ["COMICS_Dialogue", "RecipeQA_VisualCloze"]
text_rich_vqa = ["WebQA", "TQA", "OCR-VQA", "DocVQA"]
multi_image_vqa = ["MIT-States_StateCoherence", "MIT-States_PropertyCoherence", "VISION", "RecipeQA_ImageCoherence"]

puzzle = ["RAVEN"]
nlrv2 = ["NLVR2_Mantis"]
qbench = ["QBench"]

scan_qa = ["ScanQA"]
alfred = ["ALFRED"]
nuscenes = ["nuscenes"]
scannet_chat = ["ScanNet_chat"]
scannet_task = ["ScanNet_task"]
blink = ["BLINK"]
math_verse = ["MathVerse"]
sci_verse = ["SciVerse"]
mantis = ["Mantis"]


def doc_to_visual(doc):
    max_visual_count = 16
    visuals = []
    for i in range(max_visual_count):
        if f"image_{i}" in doc:
            image = doc[f"image_{i}"]
            if image is None:
                continue  # Skip this image if it's None
            if isinstance(image, Image.Image):
                visuals.append(image.copy().convert("RGB"))
            else:
                try:
                    # If the image is not already a PIL Image, try to open it
                    visuals.append(Image.open(image).convert("RGB"))
                except Exception as e:
                    print(f"Error opening image_{i}: {e}")
                    # Optionally, you can add a placeholder image or just continue
                    continue

    return visuals


# This is the place where you format your question
def doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}

    oe_post_prompt = ""
    if "oe_post_prompt" in model_specific_prompt_kwargs:
        oe_post_prompt = model_specific_prompt_kwargs["oe_post_prompt"]

    mcq_post_prompt = ""
    if "mcq_post_prompt" in model_specific_prompt_kwargs:
        mcq_post_prompt = model_specific_prompt_kwargs["mcq_post_prompt"]

    user_prompt = doc["question"]

    if mcq_post_prompt != "" and doc["question_type"] == "multi-choice":
        user_prompt = user_prompt.split("Your answer is:")[0].split("\n")[0].strip()
        user_prompt = f"{user_prompt}\n{mcq_post_prompt}"

    if oe_post_prompt != "" and doc["question_type"] == "open-ended":
        user_prompt = f"{user_prompt}\n{oe_post_prompt}"

    return user_prompt


def doc_to_text_conversation(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}

    conversations = doc["conversations"]

    if isinstance(conversations, list):
        user_prompt = json.dumps(conversations)
    else:
        user_prompt = conversations

    return user_prompt


def doc_to_text_multi_turn(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}

    return doc["conversations"]


def interleave_process_results(doc, results):
    pred = results[0]
    sample_id = doc["sample_id"]

    if doc["question_type"] == "multi-choice":
        score = mcq_acc(doc["answer"], pred)
        model_response = {"sample_id": sample_id, "sub_task": doc["sub_task"], "question_type": doc["question_type"], "answer": doc["answer"], "parsed_pred": pred, "score": score}
    elif doc["question_type"] == "open-ended":
        score = oe_rogue(doc["answer"], pred)
        model_response = {"sample_id": sample_id, "sub_task": doc["sub_task"], "question_type": doc["question_type"], "answer": doc["answer"], "parsed_pred": pred, "score": score}
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {
        "overall_score": model_response,
    }


def mcq_acc(answer, pred):
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile("(\d)(\,)(\d)")
    punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(answer):
        option_regex = re.compile(r"^([A-E])\.\s*(.+)$", re.IGNORECASE)
        match = option_regex.match(answer.strip())

        if match:
            # If matched, return the option letter in uppercase
            return match.group(1).upper()
        else:
            # If no match, process the answer as before
            answer = answer.replace("\n", " ")
            answer = answer.replace("\t", " ")
            answer = answer.strip()
            answer = processPunctuation(answer)
            answer = answer.strip("'")
            answer = answer.strip('"')
            answer = answer.strip(")")
            answer = answer.strip("(")
            answer = answer.strip().lower()

            # Try to find any single letter (A-E) in the processed answer
            letter_match = re.search(r"\b([A-E])\b", answer, re.IGNORECASE)
            if letter_match:
                return letter_match.group(1).upper()

            return answer

    pred = process(pred)
    answer = process(answer)

    if pred == answer:
        score = 1
    else:
        score = 0

    return score


def oe_rogue(answer, pred):
    rouge = Rouge()
    if pred == "":
        score = 0
    else:
        if len(pred) > 512:
            pred = pred[:512]
        score = rouge.get_scores(pred, answer)[0]["rouge-l"]["f"]

    return score


def overall_score(results):
    categories = {
        "Spot-the-Diff": spot_the_diff,
        "Image-Edit": image_edit_instruct,
        "Visual-Story-Telling": visual_story_telling,
        "Visual-Cloze": visual_cloze,
        "Text-Rich-VQA": text_rich_vqa,
        "Multi-Image-VQA": multi_image_vqa,
        "Puzzle": puzzle,
        "NLVR2": nlrv2,
        "QBench": qbench,
        "ScanQA": scan_qa,
        "ALFRED": alfred,
        "nuscenes": nuscenes,
        "ScanNet_chat": scannet_chat,
        "ScanNet_task": scannet_task,
        "BLINK": blink,
        "MathVerse": math_verse,
        "SciVerse": sci_verse,
        "Mantis": mantis,
    }

    category_scores = {}
    matched_subtasks = set()

    eval_logger.info(f"Evaluation Sub-Task Results:")
    for category, subtasks in categories.items():
        score = 0
        count = 0
        for result in results:
            if result["sub_task"] in subtasks:
                count += 1
                score += result["score"]
                matched_subtasks.add(result["sub_task"])
        if count > 0:
            avg_score = score / count
            category_scores[category] = avg_score
            eval_logger.info(f"{category}: {avg_score:.3f}")

    if not matched_subtasks:
        raise ValueError("No subtasks were matched in the results. Check if the subtask names are correct.")

    # Calculate overall score
    total_score = sum(category_scores.values())
    num_categories = len(category_scores)
    overall_score = total_score / num_categories if num_categories > 0 else 0

    return overall_score


# EVAL_PROMPT = """
# [Question]
# {question}

# [Assistant Response]
# {model_response}

# [Ground Truth Response]
# {ground_truth}

# [System]
# Rate whether the assistant response correctly matches the ground truth, it's about a question towards a sequence of images shared by the user.
# The rating should be 1-5, where 1 is incorrect and 5 is correct.
# Your response should be in the format:
# Explanation: (your explanation)
# Rating: (int)
# """

# NUM_SECONDS_TO_SLEEP = 5
# dir_path = os.path.dirname(os.path.realpath(__file__))
# with open(Path(__file__).parent / "_default_template_interleave_yaml", "r") as f:
#     raw_data = f.readlines()
#     safe_data = []
#     for i, line in enumerate(raw_data):
#         # remove function definition since yaml load cannot handle it
#         if "!function" not in line:
#             safe_data.append(line)

#     config = yaml.safe_load("".join(safe_data))

# GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]
# API_TYPE = config["metadata"]["api_type"]

# if API_TYPE == "openai":
#     API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
#     API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json",
#     }
# elif API_TYPE == "azure":
#     API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
#     API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
#     headers = {
#         "api-key": API_KEY,
#         "Content-Type": "application/json",
#     }
# else:
#     API_URL = ""
#     API_KEY = ""


# def get_chat_response(prompt, max_retries=5, wait_time=10):
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json",
#     }

#     payload = {
#         "model": GPT_EVAL_MODEL_NAME,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                 ],
#             }
#         ],
#         "max_tokens": 1024,
#         "temperature": 0.0,
#     }

#     for attempt in range(max_retries):
#         try:
#             response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
#             response.raise_for_status()
#             response_data = response.json()
#             return response_data["choices"][0]["message"]["content"], GPT_EVAL_MODEL_NAME
#         except requests.exceptions.RequestException as e:
#             eval_logger.warning(f"Request failed on attempt {attempt+1}: {e}")
#             time.sleep(wait_time)
#             if attempt == max_retries - 1:
#                 eval_logger.error(f"Failed to get response after {max_retries} attempts")
#                 return "", GPT_EVAL_MODEL_NAME
#         except Exception as e:
#             eval_logger.error(f"Error on attempt {attempt+1}: {e}")
#             return "", GPT_EVAL_MODEL_NAME


# def in_domain_oe_gpt_eval(results, args):
#     total_score = 0
#     available_count = 0
#     for result in results:
#         if result["question_type"] == "open-ended":
#             question = result["question"]
#             model_response = result["parsed_pred"]
#             ground_truth = result["answer"]
#             content = EVAL_PROMPT.format(question=question, model_response=model_response, ground_truth=ground_truth)
#             result["gpt_eval_input"] = content
#             model_output, model_name = get_chat_response(content)
#             try:
#                 explanation = re.search(r"Explanation: (.*)\n", model_output).group(1)
#                 rating = re.search(r"Rating: (\d+)\n", model_output).group(1)
#                 result["gpt_eval_explanation"] = explanation
#                 result["gpt_eval_rating"] = rating
#                 result["gpt_eval_model_name"] = model_name
#             except:
#                 eval_logger.error(f"Error on evaluating {result['sample_id']}. Results: {results}")
#                 result["gpt_eval_explanation"] = ""
#                 result["gpt_eval_rating"] = 0
#                 result["gpt_eval_model_name"] = model_name

#             total_score += result["gpt_eval_rating"]
#             available_count += 1

#         elif result["question_type"] == "multi-choice":
#             pass

#     return (total_score / available_count) * 20.0 if available_count > 0 else 0


# # class MultiChoiceRegexFilter(ExtendedRegexFilter):
# #     def __init__(self, *args, **kwargs):
# #         super().__init__(*args, **kwargs)

# #     def apply(self, resps, docs):
# #         filtered_resps = []

# #         for r, doc in zip(resps, docs):
# #             # Regex to directly extract the option letter from the model response
# #             option_letter_regex = re.compile(r"\b([A-Z])\.\s+([^\n]*)")

# #             # Process each response
# #             filtered = []
# #             for resp in r:
# #                 # Try to match the option letter at the start of the response
# #                 match = option_letter_regex.match(resp)
# #                 if match:
# #                     # If a match is found, append the matched letter
# #                     filtered.append(match.group(1))
# #                 else:
# #                     # If no match, return the original response
# #                     filtered.append(resp)

# #             # Assuming we need the first response that matches or the original response
# #             filtered_resps.append(filtered[0])

# #         return filtered_resps
