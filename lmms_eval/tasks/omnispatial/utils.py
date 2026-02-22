import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.llm_judge import get_server
from lmms_eval.llm_judge.protocol import ServerConfig

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))

cache_dir = snapshot_download(
    repo_id=config["dataset_path"],
    repo_type="dataset",
    local_dir_use_symlinks=False,
)

# Initialize the LLM judge server
if config["metadata"]["eval_type"] == "llm":
    eval_logger.info("Using LMMS judge server for OmniSpatial task.")
    API_TYPE = os.getenv("API_TYPE", "openai")
    # Use JUDGE_MODEL_VERSION instead of MODEL_VERSION
    JUDGE_MODEL_VERSION = os.getenv("JUDGE_MODEL_VERSION", "gpt-4.1-mini")

    server_config = ServerConfig(
        model_name=JUDGE_MODEL_VERSION,
    )
    server = get_server(server_name=API_TYPE, config=server_config)


def omnispatial_doc_to_visual(doc: dict) -> list:
    visual = []
    image_path = os.path.join(cache_dir, doc["image_path"])
    if os.path.exists(image_path):
        visual.append(Image.open(image_path).convert("RGB"))
    else:
        raise FileExistsError(f"video path:{image_path} does not exist.")
    return visual


def omnispatial_doc_to_text(doc: dict[str, Any]) -> str:
    prompt = SYS_PROMPTS[config["metadata"]["prompt_type"]] + "\n" + FORMAT_PROMPTS[config["metadata"]["eval_type"]] + "\n\n" + doc["question"]
    options = doc["options"]
    for i in range(len(options)):
        prompt += f"\n{chr(65 + i)}. {options[i]}"
    return prompt


def omnispatial_process_results(doc: Dict, results: List[str]) -> Dict[str, Dict]:
    # extract grounded answer
    grounded_output = doc["gt"]
    response = results[0]

    query = omnispatial_doc_to_text(doc)

    # extract predicted answer
    eval_type = config["metadata"]["eval_type"]
    if eval_type == "json":
        try:
            cleaned = response.strip().removeprefix("```json").removesuffix("```").strip()
            pred_letter = json.loads(cleaned).get("answer", "A").strip().upper()[:1]
        except Exception:
            pred_letter = "A"
        flag = pred_letter == grounded_output
    elif eval_type == "re":
        PATTERN = re.compile(r"Answer\s*:\s*([A-D])\b", re.IGNORECASE)
        pred_letter = PATTERN.findall(response)[-1] if len(PATTERN.findall(response)) > 0 else "A"
        flag = pred_letter == grounded_output
    elif eval_type == "direct":
        pred_letter = response.strip().upper()[:1]
        flag = pred_letter == grounded_output
    elif eval_type == "llm":
        try:
            from lmms_eval.llm_judge.protocol import Request, ServerConfig

            custom_config = ServerConfig(model_name=JUDGE_MODEL_VERSION)
            msgs = [
                {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": f"Question: {query}\nPred: {response}\nGT: {grounded_output}"}]},
            ]
            request = Request(messages=msgs, config=custom_config)

            response = server.evaluate(request)

            if response.success:
                judge_response = response.content.strip().lower()
                flag = "true" in judge_response
            else:
                eval_logger.error("Judge evaluation failed: %s", response.content)
                flag = False

        except Exception as e:
            eval_logger.error("Error getting judge response: %s", e)
            flag = False
    else:
        assert False, f"Unknown eval_type: {eval_type}"
    category = "omnispatial_" + doc["sub_task_type"].lower()
    key_benchmark = "omnispatial"
    omnispatial_submission = {"id": doc["id"], "query": query, "gt_content": grounded_output, "pred": response, "task": doc["task_type"], "sub_task": category, "is_correct": flag}
    return {category: omnispatial_submission, key_benchmark: omnispatial_submission}


def omnispatial_group_aggregate_results(results: List[Dict]) -> float:
    return float(np.mean([sample["is_correct"] for sample in results]))


def omnispatial_aggregate_results(results: List[Dict]) -> float:
    sub_task_to_eval_samples = defaultdict(list)
    task_to_eval_samples = defaultdict(list)
    total_samples = len(results)
    total_correct = 0

    for sample in results:
        sub_task = sample["sub_task"]
        task = sample["task"]
        is_correct = sample["is_correct"]

        if is_correct:
            total_correct += 1
            sub_task_to_eval_samples[sub_task].append(1)
            task_to_eval_samples[task].append(1)
        else:
            sub_task_to_eval_samples[sub_task].append(0)
            task_to_eval_samples[task].append(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    sub_task_accuracies = {sub_task: sum(scores) / len(scores) for sub_task, scores in sub_task_to_eval_samples.items()}
    task_accuracies = {task: sum(scores) / len(scores) for task, scores in task_to_eval_samples.items()}
    eval_logger.info(f"{'Total Samples':<20}: {total_samples}")
    eval_logger.info(f"{'Total Correct':<20}: {total_correct}")
    eval_logger.info(f"{'Overall Accuracy':<20}: {accuracy:.4f}")
    eval_logger.info("\n")

    eval_logger.info(f"{'Per-Sub-Task Accuracy':<40}")
    eval_logger.info("-" * 40)
    for sub_task, acc in sub_task_accuracies.items():
        eval_logger.info(f"{sub_task:<20}: {acc:.4f}")
    eval_logger.info("=" * 40)

    eval_logger.info(f"{'Per-Task Accuracy':<40}")
    eval_logger.info("-" * 40)
    for task, acc in task_accuracies.items():
        eval_logger.info(f"{task:<20}: {acc:.4f}")
    eval_logger.info("=" * 40)
    return accuracy


# from https://github.com/qizekun/OmniSpatial
###############################################################################
#                             Response Formatting                             #
###############################################################################

RE_FORMAT = """
End your answer with a separate line formatted exactly as:

Answer: X
where X ∈ {A, B, C, D}.
"""

JSON_FORMAT = """
You need to respond with the answer in JSON format:

```json
{
    "analysis": "The analysis of the image and question",
    "answer": "A"
}
```
"""

LLM_FORMAT = """
Your answer must be clear and accurate.
"""

DIRECT_FORMAT = """
Note: You only need to respond with A, B, C, or D without providing any additional information.
"""

FORMAT_PROMPTS = {"re": RE_FORMAT, "json": JSON_FORMAT, "llm": LLM_FORMAT, "direct": DIRECT_FORMAT}

###############################################################################
#                             System Prompts                                 #
###############################################################################

DEFAULT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive
1. **Image** - a single RGB frame depicting a scene.
2. **Question** - a natural-language query about spatial relationships between objects in the image.
3. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Based on the image and question, provide your answer.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply “insufficient information.”
"""

ZERO_SHOT_COT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive
1. **Image** - a single RGB frame depicting a scene.
2. **Question** - a natural-language query about spatial relationships between objects in the image.
3. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Think step by step and provide the answer.
Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply “insufficient information.”
"""

MANUAL_COT_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive
1. **Image** - a single RGB frame depicting a scene.
2. **Question** - a natural-language query about spatial relationships between objects in the image.
3. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Guidelines
----------
Please follow these steps to analyze the image and answer the question:
1. First, carefully observe the image and identify all relevant objects and their spatial relationships.
2. Next, break down the question into key components that need to be addressed.
3. Think through the spatial reasoning step-by-step to arrive at your answer. It may be necessary to transfer perspective to better understand the scene.
4. Finally, select the most appropriate option (A, B, C, or D) based on your analysis.

Always ground your answer in the visual evidence; do not hallucinate unseen objects.
If uncertain, pick the most plausible option—never refuse or reply “insufficient information.”
"""

LLM_JUDGE_SYSTEM_PROMPT = """
You are a judge for QA tests.

The user will provide:
Question: The original question.
Pred: The predicted answer.
GT: The ground truth answer.

You need to judge whether the predicted answer is correct or not; just judge the final answer.
If the predicted answer is correct, respond with "True".
If the predicted answer is incorrect, respond with "False".
"""

BLIND_SYSTEM_PROMPT = """
You are a spatial-reasoning assistant.

Task
-----
You will receive
1. **Question** - a natural-language query about spatial relationships.
2. **Options** - ≥2 answer candidates, each tagged by a capital letter (A, B, C, D…).

Based on the question only, provide your answer.
"""

SYS_PROMPTS = {
    "none": DEFAULT_SYSTEM_PROMPT,
    "zeroshot_cot": ZERO_SHOT_COT_SYSTEM_PROMPT,
    "manual_cot": MANUAL_COT_SYSTEM_PROMPT,
}
