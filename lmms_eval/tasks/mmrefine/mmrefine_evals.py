import os
import time

from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.tasks.mmrefine.prompts import (
    EVAL_PROMPT_CORRECT,
    EVAL_PROMPT_INCORRECT,
    PARSING_PROMPT,
    REFINEMENT_PROMPT,
)


class MMRefineEvaluator:
    """Evaluator for MMRefine mathematical problem refinement tasks.

    Handles evaluation of model-generated refinements for multimodal
    mathematical problem-solving, supporting OpenAI and Azure OpenAI APIs.
    """

    API_TYPE = os.getenv("API_TYPE", "openai")
    if API_TYPE == "openai":
        API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        client = OpenAI(api_key=API_KEY, base_url=API_URL.rstrip("chat/completions"))
        gpt_model = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

    elif API_TYPE == "azure":
        API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
        API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
        API_VERSION = os.getenv("AZURE_API_VERSION", "2023-07-01-preview")
        client = AzureOpenAI(azure_endpoint=API_URL, api_version=API_VERSION, api_key=API_KEY)
        gpt_model = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")
    server_config = ServerConfig(
        model_name=gpt_model,
    )
    server = get_server(server_name=API_TYPE, config=server_config)

    def __init__(self, quick_extract=False):
        self.quick_extract = quick_extract

    def get_chat_response(self, prompt, temperature=0, max_tokens=256, n=1, patience=10000000, sleep_time=0):
        messages = [
            {"role": "user", "content": prompt},
        ]
        payload = {"model": self.gpt_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

        while patience > 0:
            patience -= 1
            try:
                response = self.client.chat.completions.create(**payload)
                if n == 1:
                    prediction = response.choices[0].message.content.strip()
                    if prediction and prediction != "":
                        return prediction
                else:
                    prediction = [choice.message.content.strip() for choice in response.choices]
                    if prediction and prediction[0] != "":
                        return prediction

            except Exception as e:
                if "Rate limit" not in str(e):
                    eval_logger.error(e)

                if "Please reduce the length of the messages" in str(e):
                    eval_logger.error("!!Reduce prompt size")
                    # reduce input prompt and keep the tail
                    new_size = int(len(prompt) * 0.9)
                    new_start = len(prompt) - new_size
                    prompt = prompt[new_start:]
                    payload["messages"] = [
                        {"role": "user", "content": prompt},
                    ]

                if sleep_time > 0:
                    time.sleep(sleep_time)
        return ""

    def evaluate_answer(self, problem, prediction):
        if not prediction:
            return {}

        if problem["solution_label"] == "correct":
            full_prompt = EVAL_PROMPT_CORRECT.format(response=prediction)
            try:
                resp = self.get_chat_response(full_prompt, temperature=0, max_tokens=256, n=1)
            except Exception as e:
                eval_logger.error(e)
                eval_logger.error("Error in evaluating answer for problem")
            try:
                solution_correctness = int(resp.strip())
            except:
                try:
                    solution_correctness = self.get_chat_response(PARSING_PROMPT.format(target="Output", model_response=resp), temperature=0, max_tokens=256, n=1)
                except:
                    solution_correctness = 0
            return {
                "solution_correctness": solution_correctness,
            }
        else:
            full_prompt = EVAL_PROMPT_INCORRECT.format(initial_solution=problem["initial_solution"], feedback=prediction, reference_feedback=problem["reference_feedback"])
            try:
                resp = self.get_chat_response(full_prompt, temperature=0, max_tokens=1024, n=1)
                try:
                    error_detection = int(self.get_chat_response(PARSING_PROMPT.format(target="Error Detection", model_response=resp), temperature=0, max_tokens=256, n=1).strip())
                except:
                    error_detection = 0
                try:
                    error_correction = int(self.get_chat_response(PARSING_PROMPT.format(target="Error Correction", model_response=resp), temperature=0, max_tokens=256, n=1).strip())
                except:
                    error_correction = 0
                try:
                    solution_correctness = int(self.get_chat_response(PARSING_PROMPT.format(target="Effectiveness and Correctness of the Feedback", model_response=resp), temperature=0, max_tokens=256, n=1).strip())
                except:
                    solution_correctness = 0
            except Exception as e:
                eval_logger.error(e)
                eval_logger.error(f"Error in evaluating answer for problem")

            return {
                "error_detection": error_detection,
                "error_correction": error_correction,
                "solution_correctness": solution_correctness,
            }

    def create_one_query(self, problem):
        query = REFINEMENT_PROMPT.format(question=problem["question"], initial_solution=problem["initial_solution"])
        return query

    def eval_result(self, result, config):
        # extract and score for each question
        full_prediction = result["prediction"].strip()
        problem = {
            "answer": result["answer"] if "answer" in result else None,
            "initial_solution": result["initial_solution"],
            "solution_label": result["solution_label"],
            "reference_feedback": result["reference_feedback"] if "reference_feedback" in result else None,
        }
        if config["metadata"].get("trunk_response", -1) > 0:
            prediction = " ".join(full_prediction.split(" ")[-config["metadata"]["trunk_response"] :])
        else:
            prediction = full_prediction
        eval_result = self.evaluate_answer(problem, prediction)

        result["result"] = eval_result

        def classify_results(row):
            result = row["result"]
            if row["solution_label"] == "correct":
                if result["solution_correctness"] == 0:
                    return "False Error Detection"
                return "Validation Success"
            if result["error_detection"] == 0:
                return "Refinement Failure"
            if result["error_correction"] == 0:
                return "Error Detection Success"
            if result["solution_correctness"] == 0:
                return "Error Correction Success"
            if result["solution_correctness"] == 1:
                return "Refinement Success"

        result["eval_result"] = classify_results(result)
        return result
