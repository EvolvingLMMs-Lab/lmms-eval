import os
from pathlib import Path

import yaml
from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server

# Load config from k12.yaml
with open(Path(__file__).parent / "k12.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

# Initialize the judge server
API_TYPE = os.getenv("API_TYPE", "openai")
GPT_MODEL = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

server_config = ServerConfig(
    model_name=GPT_MODEL,
)
server = get_server(server_name=API_TYPE, config=server_config)


def k12_doc_to_visual(doc):
    visual_list = []
    if "image" in doc and doc["image"] is not None:
        visual_list.append(doc["image"].convert("RGB"))
    return visual_list


def k12_doc_to_text(doc):
    question = doc["question"]
    return question


def k12_process_results(doc, results):
    prediction = results[0].strip()
    question = doc["question"]
    answer = doc["answer"]

    try:
        # Use the llm_judge API for binary evaluation
        result = server.evaluate_binary(question=question, answer=answer, prediction=prediction, output_format="0/1")

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
