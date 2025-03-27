import datetime
import json
import os

from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.olympiadbench.olympiadbench_evals import OlympiadBenchEvaluator

dir_name = os.path.dirname(os.path.abspath(__file__))

API_TYPE = os.getenv("API_TYPE", "openai")
if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    client = OpenAI(api_key=API_KEY)
    gpt_model = config["metadata"]["gpt_eval_model_name"]

elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    API_VERSION = os.getenv("AZURE_API_VERSION", "2023-07-01-preview")
    client = AzureOpenAI(azure_endpoint=API_URL, api_version=API_VERSION, api_key=API_KEY)
    gpt_model = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")


def get_chat_response(prompt, max_token=256, retry=5):
    messages = [
        {"role": "user", "content": prompt},
    ]
    for i in range(retry):
        try:
            completion = client.chat.completions.create(model=gpt_model, messages=messages, temperature=0.5 * i, max_tokens=max_token)
            prediction = completion.choices[0].message.content.strip()
            if prediction.lower() == "yes" or prediction.lower() == "no":
                return prediction
        except Exception as e:
            eval_logger.error(e)
    return "no"


def build_olympiadbench_gpt4_prompt(question_data):
    prompt = """You are given a question, a solution and the correct answer. Please determine if the solution matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the solution and the correct answer.
The process or reasoning leading to the Solution is irrelevant, ONLY the correctness of the result matters.
Return only "Yes" if the solution is correct or "No" if it is incorrect.
Only return "Yes" or "No" with no additional text or formatting.

Question: 
{question}
--------------------------------
Correct Answer:
{answer}
--------------------------------
Solution: 
{solution}
--------------------------------
"""
    question = question_data["question"]
    answer = "\n".join(question_data["final_answer"])
    response = str(question_data["response"])
    prompt = prompt.format(question=question, answer=answer, solution=response)
    return prompt


def olympiadbench_doc_to_visual(doc):
    visual_list = []
    for i in range(1, 6):
        if f"image_{i}" in doc and doc[f"image_{i}"] is not None:
            visual_list.append(doc[f"image_{i}"].convert("RGB"))
    return visual_list


def olympiadbench_doc_to_text(doc):
    question = doc["question"]
    subject = doc["subfield"]
    mul_ans = doc["is_multiple_answer"]
    if mul_ans is None:
        mul_ans = False
    ans_type = doc["answer_type"]
    if ans_type == "Need_human_evaluate":
        ans_type = "proof based"

    pre_prompt = f"The following is a question from an International {subject} competition.\n"

    post_prompt = ""
    if not mul_ans:
        post_prompt += f"The answer of the question should be {ans_type}.\n"
    else:
        post_prompt += f"The question has multiple answers, each of them should be {ans_type}.\n"
    post_prompt += (
        "Please calculate the answer according to the given requirements and the information provided. Please use LaTeX format to represent the variables and formulas used in the solution process and results. Please end your solution with "
    )
    if not mul_ans:
        post_prompt += '"So the final answer is \\boxed{answer}."\n'
    else:
        post_prompt += "So the final answer is \\boxed{multiple answers connected with commas}.\n"

    final_question = pre_prompt + question + "\n" + post_prompt
    return final_question


def olympiadbench_process_results(doc, results):
    prediction = results[0].strip()
    # Build the prompt for GPT-4o evaluation
    question_data = {"id": doc.get("id", "unknown"), "question": doc["question"], "final_answer": doc["final_answer"], "response": prediction}

    # Build the prompt and get GPT-4o's judgment
    prompt = build_olympiadbench_gpt4_prompt(question_data)
    try:
        completion = get_chat_response(prompt)
        if completion.lower() == "yes" or completion.lower() == "no":
            judge_result = 1 if completion.lower() == "yes" else 0
        else:
            eval_logger.error(f"Invalid response: {completion}")
            judge_result = 0
    except Exception as e:
        eval_logger.error(f"Error getting chat response: {e}")
        judge_result = 0

    return {"llm_as_judge_eval": judge_result}
