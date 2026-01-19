import datetime
import json
import os

from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.olympiadbench.olympiadbench_evals import OlympiadBenchEvaluator

dir_name = os.path.dirname(os.path.abspath(__file__))

# Initialize the judge server
API_TYPE = os.getenv("API_TYPE", "openai")
GPT_MODEL = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

server_config = ServerConfig(
    model_name=GPT_MODEL,
)
server = get_server(server_name=API_TYPE, config=server_config)


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
    question = doc["question"]
    answer = "\n".join(doc["final_answer"])

    # Define custom prompt for OlympiadBench evaluation
    custom_prompt = """You are given a question, a solution and the correct answer. Please determine if the solution matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the solution and the correct answer.
The process or reasoning leading to the Solution is irrelevant, ONLY the correctness of the result matters.
Return only "Yes" if the solution is correct or "No" if it is incorrect.
Only return "Yes" or "No" with no additional text or formatting.

Question: {question}

Response: {prediction}

Ground Truth: {answer}"""

    try:
        # Use the llm_judge API for binary evaluation
        result = server.evaluate_binary(question=question, answer=answer, prediction=prediction, output_format="yes/no", custom_prompt=custom_prompt)

        # Parse the result
        if result["success"]:
            judge_response = result["result"]
            judge_result = 1 if judge_response else 0
        else:
            eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
            judge_result = 0

    except Exception as e:
        eval_logger.error(f"Error getting judge response: {e}")
        judge_result = 0

    return {"llm_as_judge_eval": judge_result}
