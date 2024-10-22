from calendar import c
import re
import os
import sys
import datetime
from loguru import logger as eval_logger
import lmms_eval.tasks._task_utils.file_utils as file_utils
from lmms_eval.filters.extraction import ExtendedRegexFilter
import json
import yaml
import torch
from pathlib import Path
import requests
import time
import numpy as np
from pycocoevalcap.eval import COCOEvalCap, Bleu, Meteor, Rouge, Cider, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from tqdm import tqdm
# import nltk
# nltk.download('punkt')
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.tokenize import word_tokenize
# from rouge import Rouge


NUM_SECONDS_TO_SLEEP = 5

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

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



system_messages = {
    '2':
        '''
        You will be given two text segments in the following format: [text1][text2]. These two texts will be descriptions of a counterintuitive (humorous, creative, or magical) video. For text2, your task is to provide a score based on the following criteria:
        1. Content: Score out of 20 points. If the content is nearly identical, award 20 points. If the content differs slightly, deduct 5 points. If the content differs significantly, deduct 10 points. If the content differs greatly, deduct 15 points. If the content is completely different, deduct 20 points.
        2. Details: Score out of 50 points. Describe the video's details, including characters, scenes, actions, dialogues, etc. Deduct 5 points for each differing detail. Clearly identify and count the differing details to calculate the final score.
        3. Logic: Score out of 20 points. The description should be logically consistent without any unreasonable situations. If the logic is nearly identical, award 20 points. If the logic is generally consistent but differs in details, award 15 points. If there are some differences in logic but still similar overall, award 10 points. If there are significant differences in logic, award 5 points.
        4. Language Expression: Score out of 10 points. Evaluate the fluency and word usage of the text. If the language expression is at a consistent level, award 10 points. If there are minor differences in language expression, award 5 points. If there are significant differences in language expression, award 0 points.
        Note: If the content differs significantly, multiply the total score by 0.5. If the content differs greatly, multiply the total score by 0.25.
        The output format is (remember not to have any comments, directly output scores) :
        [Content: Score], [Details: Score], [Logic: Score], [Language: Score], [Factor: 1 or 0.5 or 0.25]
        [Final Score]
        ''',
    '3':
        '''
        You will be given two text segments in the following format: [text1][text2]. These two texts will be explanations for a counterintuitive video (humorous, creative, or magical). For text2, your task is to provide a score based on the following criteria:
        1. Language Expression: Score out of 5 points. Evaluate the fluency and word usage of the text. If the language expression is at a consistent level, award 5 points. If there are significant differences in language expression, award 0 points.
        2. Logic: Score out of 10 points. The explanation should be logically sound, preferably with logical words and cause-effect relationships. If the logic is nearly identical, award 10 points. If the logic is generally consistent but differs in details, award 5 points. If there are some differences in logic but still similar overall, award 5 points. If there are significant differences in logic, award 0 points.
        3. Common Sense Errors: Score out of 10 points. The explanation should not contain any obvious common sense errors. Deduct 5 points for each occurrence of a common sense error.
        4. Understanding of Humor, Creativity, or Magic: Score out of 40 points. If the explanation focuses on the same key points as the reference answer, award 35 points or above. If the explanation provides reasons for the counterintuitive phenomenon but differs from the reference answer, award between 15-35 points based on the difference. If the explanation provides reasons for the counterintuitive phenomenon but differs greatly from the reference answer, award between 0-15 points.
        5. Details: Score out of 35 points. While providing the explanation, include video details that contribute to the humor, creativity, or magical effect. Deduct 5 points for each additional or missing detail compared to the reference answer.
        6. If the explanation differs significantly from the reference answer and includes descriptive details not mentioned in the reference answer, multiply the total score by 0.5.
        7. The minimum score is 0, and the maximum score is 100.
        The output format is (remember not to have any comments, directly output scores) :
        [Language: Score], [Logic: Score], [Common Sense Errors: Score], [Understanding: Score], [Details: Score], [Factor: 1 or 0.5 or 0.25]
        [Final Score]
        ''',
    '4':
        '''
        You will be given four text segments in the following format: [Description][Explanation][text1][text2]. The first two texts are descriptions of a video and its explanation, respectively. The third text is a reference title. Your task is to evaluate whether the fourth text is a good title. Note that the fourth text may not be a title but a statement including the video. In that case, extract the actual title and evaluate it. Consider the following points while assigning a score:
        1. The title should mention the content of the video.
        2. A title with a certain level of humor or creativity is preferable.
        Provide a score ranging from 0 to 100, considering the above criteria and tell the reason.
        The output format is:
        [Final Score]
        ('Final Score' are in square brackets remember! Just one line! Remember not to have any comments, directly output scores. Remember DO NOT GIVE ME EXPLANATION!!!!!!!!!) :
        ''',
}


def extract_last_number(string):
    # Use regular expression to match last number
    match = re.search(r'\d+(\.\d*)?(?=[^\d.]*$)', string)

    if match:
        last_integer = float(match.group())
        return last_integer
    else:
        # If there is no integer in the string, return None or other default value
        print('No integers in this string.')
        return 0


# Different Implementations of BLEU-4 and Rouge with nltk and pycocoevalcap

# Use nltk to calculate BLEU-4 score
# def calculate_bleu4(reference, hypothesis):
#     smooth_fn = SmoothingFunction().method1
#     ref_tokens = word_tokenize(reference)
#     hyp_tokens = word_tokenize(hypothesis)
#     blue4_score = sentence_bleu([ref_tokens],
#                                 hyp_tokens,
#                                 weights=(0.25, 0.25, 0.25, 0.25),
#                                 smoothing_function=smooth_fn)
#
#     return blue4_score

# use pycocoevalcap to calculate BLEU-4 score
def calculate_bleu4(reference, hypothesis):
    ref = {0: [{'caption': reference}]}
    hyp = {0: [{'caption': hypothesis}]}

    tokenizer = PTBTokenizer()
    ref_tokenized = tokenizer.tokenize(ref)
    hyp_tokenized = tokenizer.tokenize(hyp)

    scorer = Bleu(4)
    score, _ = scorer.compute_score(ref_tokenized, hyp_tokenized)
    bleu4_score = score[3]
    return bleu4_score


# Use rouge to calculate ROUGE-L score
# def calculate_rouge(reference, hypothesis):
#     rouge = Rouge()
#     rouge_score = rouge.get_scores(hypothesis, reference, avg=True)
#     return rouge_score['rouge-l']['f']

# use pycocoevalcap to calculate ROUGE-L score
def calculate_rouge(reference, hypothesis):
    ref = {0: [{'caption': reference}]}
    hyp = {0: [{'caption': hypothesis}]}


    tokenizer = PTBTokenizer()
    ref_tokenized = tokenizer.tokenize(ref)
    hyp_tokenized = tokenizer.tokenize(hyp)
    # eval_logger.info("computing ROUGE score")
    scorer = Rouge()
    score, _ = scorer.compute_score(ref_tokenized, hyp_tokenized)

    return score


def get_eval(candidate: str, task: str, content: str, max_tokens: int, retries: int = 5):
    # TODO: Delete this line
    # return "This is the evaluation answer of gpt4, Score: 80/100", "gpt_prompt", 100  
    
    global headers
    # Truncate the candidate to the max length
    max_len = {
        'H2': 150,
        'H3': 180,
        'H4': 40,
        'C2': 390,
        'C3': 310,
        'C4': 30,
        'M2': 180,
        'M3': 130
    }
    if len(candidate) > max_len[task]:
        candidate = candidate[:max_len[task]]
    content = content + '[' + candidate + ']'
    gpt_prompt = str(content)
    messages = [
        {"role": "system", "content": system_messages[task[1]]},
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
            gpt_score = extract_last_number(content)
            if content != "" and score != 0:
                return content, gpt_prompt, gpt_score
            break  # If successful, break out of the loop

        except Exception as e:
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:  # If we have retries left, sleep and then continue to next attempt
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:  # If this was the last attempt, log and return empty
                eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                return "", "", 0
    return "", "", 0


# A bit ugly here
# But the idea is that we will unzip all the zip files
# To HF HOME cache dir
# And load it here
HF_HOME = os.getenv("HF_HOME", "~/.cache/huggingface/")
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir, "videos")




# Pass in video path here
# Can only work correctly with video llm
def funqa_doc_to_visual(doc):
    video_path = doc["video_name"]
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# This is the place where you format your question
def funqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    return question


def funqa_doc_to_answer(doc):
    return doc["answer"]


# An example of showing how to custom metric
# Your metric name should have the same key name in your return dict
def funqa_process_results(doc, result):
    pred = result[0]
    # content = eval_prompt.format(question=doc["question"], answer=doc["answer"], candidate=pred)
    content, gpt_prompt, gpt_score= get_eval(candidate=pred, task=doc["task"], content=doc["prompt"],
                                    max_tokens=1024)
    return {
        "submission": {"pred": pred, "answer": doc["answer"], "task": doc["task"], 
                       "eval_answer": content, gpt_prompt: gpt_prompt, "gpt_score": gpt_score},
        "funqa_gpt": {"pred": pred, "answer": doc["answer"], "task": doc["task"], 
                      "eval_answer": content, gpt_prompt: gpt_prompt, "gpt_score": gpt_score},
        "funqa_BLEU": {"pred": pred, "answer": doc["answer"], "task": doc["task"]},
        "funqa_ROUGE": {"pred": pred, "answer": doc["answer"], "task": doc["task"]},
        "funqa_BLEURT": {"pred": pred, "answer": doc["answer"], "task": doc["task"]},
    }


def funqa_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"funqa-{task}-{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Submission file saved to {path}")

def funqa_aggregate_results_bleurt(results, args):
    bleurt_version = 'lucadiliello/BLEURT-20'
    eval_logger.info(f"Loading BLEURT model {bleurt_version}, you can change to the small version BLEURT-20-D12 in tasks/funqa/utils.py")
    config = BleurtConfig.from_pretrained(bleurt_version)
    model = BleurtForSequenceClassification.from_pretrained(bleurt_version)
    tokenizer = BleurtTokenizer.from_pretrained(bleurt_version)

    scores_dict = {'H2': [], 'H3': [], 'H4': [], 'C2': [], 'C3': [], 'C4': [], 'M2': [], 'M3': []}
    eval_logger.info(f"Calculating BLEURT score")
    for result in tqdm(results):
        gt = result["answer"]
        pred = result["pred"]
        task = result["task"]
        model.eval()
        with torch.no_grad():
            inputs = tokenizer([gt], [pred], padding='longest', return_tensors='pt')
            res = model(**inputs).logits.flatten().tolist()
        scores_dict[task].append(res[0])

    mean_scores = {task: np.mean(scores) if scores else 0 for task, scores in scores_dict.items()}
    eval_logger.info(f"BLEURT score: {mean_scores}")
    mean_score = sum(mean_scores.values()) / len(mean_scores)

    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    scores_file_name = f"funqa-BLEURT-{now_date_time}.json"
    path = file_utils.generate_submission_file(scores_file_name, args, subpath="scores")
    with open(path, "w") as f:
        json.dump(mean_scores, f)
    eval_logger.info(f"BLEURT score file saved to {path}")
    return mean_score

def funqa_aggregate_results(results, metric, args):
    # Add more metrics here
    score_functions = {
        "BLEU": calculate_bleu4,
        "ROUGE": calculate_rouge,
    }
    if metric not in score_functions:
        raise ValueError("Unsupported metric")

    scores_dict = {'H2': [], 'H3': [], 'H4': [], 'C2': [], 'C3': [], 'C4': [], 'M2': [], 'M3': []}
    eval_logger.info(f"Calculating {metric} score")
    for result in tqdm(results):
        gt = result["answer"]
        pred = result["pred"]
        task = result["task"]
        scores_dict[task].append(score_functions[metric](gt, pred))

    mean_scores = {task: np.mean(scores) if scores else 0 for task, scores in scores_dict.items()}
    eval_logger.info(f"{metric} score: {mean_scores}")
    mean_score = sum(mean_scores.values()) / len(mean_scores)

    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    scores_file_name = f"funqa-{metric}-{now_date_time}.json"
    path = file_utils.generate_submission_file(scores_file_name, args, subpath="scores")
    with open(path, "w") as f:
        json.dump(mean_scores, f)
    eval_logger.info(f"{metric} score file saved to {path}")
    return mean_score


def funqa_GPT_eval(results, args):
    scores_dict = {'H2': [], 'H3': [], 'H4': [], 'C2': [], 'C3': [], 'C4': [], 'M2': [], 'M3': []}
    for result in results:
        gpt_score = result["gpt_score"]
        scores_dict[result["task"]].append(gpt_score)
    mean_scores = {task: np.mean(scores) if scores else 0 for task, scores in scores_dict.items()}
    eval_logger.info(f"GPT eval score: {mean_scores}")

    mean_score = sum(mean_scores.values()) / len(mean_scores)

    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    scores_file_name = f"funqa-gpteval-{now_date_time}.json"
    path = file_utils.generate_submission_file(scores_file_name, args, subpath="scores")
    with open(path, "w") as f:
        json.dump(mean_scores, f)
    eval_logger.info(f"GPT eval score file saved to {path}")
    return mean_score


# Factory into different aggregate
def funqa_aggregate(results, args):
    funqa_aggregate_submissions(results, args, "all")

# ugly functions
def funqa_BLEU(results, args):
    return funqa_aggregate_results(results, "BLEU", args)


def funqa_ROUGE(results, args):
    return funqa_aggregate_results(results, "ROUGE", args)

def funqa_BLEURT(results, args):
    return funqa_aggregate_results_bleurt(results, args)
