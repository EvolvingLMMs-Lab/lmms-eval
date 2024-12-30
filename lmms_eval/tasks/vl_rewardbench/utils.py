import json
import os
import random
import re
from collections import defaultdict

import requests
from loguru import logger as eval_logger


LLM_PARSE_ANSWER_PROMPT = """
You are given a pairwise judgement for two responses. Please return the better response according to the judgement.
Return the Answer X ONLY. e.g., Answer 1 or Answer 2.

Judgement: {judgement}
"""

API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def get_prompt(data_obj, random_number):
    answers = [data_obj["response"][0], data_obj["response"][1]] if random_number == 0 else [data_obj["response"][1], data_obj["response"][0]]
    prompt_str = f""" You are a highly capable multimodal AI assistant tasked with evaluating answers to visual questions. Please analyze the following image and question, then determine which of the two provided answers is better.

Question: {data_obj["query"]}

Answer 1: {answers[0]}

Answer 2: {answers[1]}

Please evaluate both answers based on the following criteria:
1. Accuracy: How well does the answer align with the visual information in the image?
2. Completeness: Does the answer fully address all aspects of the question?
3. Clarity: Is the answer easy to understand and well-articulated?
4. Relevance: Does the answer directly relate to the question and the image?

After your evaluation, please:
1. Explain your reasoning for each criterion.
2. Provide an overall judgment on which answer is better (Answer 1 or Answer 2). For example: Overall Judgment: Answer X is better.

Your response should be structured and detailed, demonstrating your understanding of both the visual and textual elements of the task."""
    return prompt_str


def vlrewardbench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vlrewardbench_doc_to_text(doc):
    # we randomly choose the order of the answers to avoid positional bias
    random_number = sum(len(res) for res in doc["response"]) % 2  # we use the length sum % 2 as a random number generator to decide the order of the answers
    query_prompt = get_prompt(doc, random_number)
    return query_prompt


def parse_pred_ans(pred_ans):
    pred_ans = pred_ans.strip()
    pattern = r"(?:Overall Judgment|Therefore)\s*.*\s*-*\s*Answer\s*(\d+)\s*is\s*(?:the\s*)?(?:slightly\s*)?better"
    match = re.search(pattern, pred_ans.replace("\n", "").replace("*", ""), re.IGNORECASE)
    flag_choice = -1
    if match:
        answer_number = int(match.group(1))
        flag_choice = answer_number
    else:
        # parse by llm
        parsed_response = parse_by_llm(pred_ans)
        if "Answer 1".lower() in parsed_response.lower():
            flag_choice = 1
        elif "Answer 2".lower() in parsed_response.lower():
            flag_choice = 2
        else:
            eval_logger.warning(f"Cannot parse the answer: {pred_ans}, we randomly choose a choice")
            flag_choice = random.choice([1, 2])

    return flag_choice  # which one is better


def parse_by_llm(response, model="gpt-4o-mini", max_tokens=32):
    # get the judgement from response using gpt-4o
    data = {"max_tokens": max_tokens, "model": model, "temperature": 0.0, "top_p": 1.0, "presence_penalty": 1, "messages": [{"role": "user", "content": LLM_PARSE_ANSWER_PROMPT.format(judgement=response)}]}
    response = requests.post(API_URL, headers=headers, data=json.dumps(data).encode("utf-8"))
    result = response.content.decode("utf-8")
    dict_result = json.loads(result)
    llm_output = dict_result["choices"][0]["message"]["content"].strip()
    return llm_output


def vlrewardbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    pred = results[0]
    pred_ans = parse_pred_ans(pred) # 1 or 2 indicte which one is better 
    random_number = sum(len(res) for res in doc["response"]) % 2  # we use the length sum % 2 as a random number generator to decide the order of the answers
    # Note: human_ranking [0, 1] -> answer 1 is better,  [1, 0] -> answer 2 is better 
    gt_ans = doc["human_ranking"].index(0 if random_number == 0 else 1) + 1

    if pred_ans == gt_ans:
        score = 1.0
    else:
        score = 0.0
    category = doc["id"].split("-")[0].split("_")[0].lower()

    group_mapping = {
        "mathverse": "reasoning",
        "hallucination": "hallucination",
        "mmmu": "reasoning",
        "rlhf": "hallucination",
        "rlaif": "hallucination",
        "wildvision": "general",
        "vlfeedback": "general",
    }
    key_name = "vlreward_score"
    # Note: the key name here is very important. It decides which aggregation function will receive the results
    # We note down the question id/category to help us aggregate the results later
    return {key_name: {"question_id": doc["id"], "category": group_mapping.get(category, "general"), "score": score}}


def vlrewardbench_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = defaultdict(list)
    for result in results:
        score = result["score"]
        category = result["category"]
        category2score[category].append(score)

    category2avg_score = {}
    for category, question2scores in category2score.items():
        category2avg_score[category] = sum(question2scores) / len(question2scores)
    for category, avg_score in category2avg_score.items():
        eval_logger.info(f"{category}: {avg_score:.2f}")
    total_score = sum(category2avg_score.values()) / len(category2avg_score)  # Macro-Avg across tasks
    return total_score
