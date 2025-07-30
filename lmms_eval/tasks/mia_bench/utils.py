import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def mia_bench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mia_bench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question_text = doc["instruction"]

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""

    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    formatted_question = f"{pre_prompt}{question_text}{post_prompt}"

    return formatted_question


# ============================
# Result Processing Functions
# ============================

with open(Path(__file__).parent / "mia_bench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

NUM_SECONDS_TO_SLEEP = 10
API_TYPE = os.getenv("API_TYPE", "openai")
GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

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
    headers = {"api-key": API_KEY, "Content-Type": "application/json", "api-version": "2023-07-01-preview"}


def get_eval(content: str, max_tokens: int, retries: int = 5):
    global headers

    messages = [
        {"role": "user", "content": content},
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
            break  # If successful, break out of the loop

        except Exception as e:
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:  # If we have retries left, sleep and then continue to next attempt
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:  # If this was the last attempt, log and return empty
                eval_logger.error(f"All {retries} attempts failed. Last error message: {e}")
                return "", ""
    return "", ""


def generate_prompt(d, response):
    instruction = d["instruction"]
    weight = d["component_weight"] * 1
    d["num_of_component"] = len(d["components"])
    for i in range(len(weight)):
        weight[i] = str(weight[i])
    if d["num_of_component"] == 1:
        components = """The first component is:' """ + d["components"][0] + "'"
        score = """The first component is worth """ + weight[0] + " scores."
    elif d["num_of_component"] == 2:
        components = """The first component is:' """ + d["components"][0] + """', and the second component is:' """ + d["components"][1] + "'"
        score = """The first and second component is each worth """ + weight[0] + " and " + weight[1] + " scores."
    elif d["num_of_component"] == 3:
        components = """The first component is:' """ + d["components"][0] + """', and the second component is:' """ + d["components"][1] + """', and the third component is:' """ + d["components"][2] + "'"
        score = """The first second, and third component is each worth """ + weight[0] + ", " + weight[1] + " and " + weight[2] + " scores."
    elif d["num_of_component"] == 4:
        components = (
            """The first component is:' """
            + d["components"][0]
            + """', and the second component is:' """
            + d["components"][1]
            + """', and the third component is:' """
            + d["components"][2]
            + """', and the fourth component is:' """
            + d["components"][3]
            + "'"
        )
        score = """The first second, third, and fourth component is each worth """ + weight[0] + ", " + weight[1] + ", " + weight[2] + " and " + weight[3] + " scores."
    elif d["num_of_component"] == 5:
        components = (
            """The first component is:' """
            + d["components"][0]
            + """', and the second component is:' """
            + d["components"][1]
            + """', and the third component is:' """
            + d["components"][2]
            + """', and the fourth component is:' """
            + d["components"][3]
            + """', and the fifth component is:' """
            + d["components"][4]
            + "'"
        )
        score = """The first second, third, fourth and fifth component is each worth """ + weight[0] + ", " + weight[1] + ", " + weight[2] + ", " + weight[3] + " and " + weight[4] + " scores."
    return (
        """Here is an instruction for a multimodal LLM: ' """
        + instruction
        + """ You need to grade if the response from the model follows each component of the instruction. """
        + components
        + """ The response is:' """
        + response
        + """' You need to score the response and be strict. The total score ranges from 0 to 10, depending on if the response follows the instruction. """
        + score
        + " List scores of each component, and the total score in one sentence in this EXACT format: score of component 1: x/2, score of component 2: y/8, total score: z/10. Use only numbers for x, y, z. Do not use markdown formatting or asterisks. Then explain your reasons."
    )


def process_rawscore(component_type, raw_score):
    import re

    from loguru import logger as eval_logger

    score_dict = {}

    try:
        # Validate inputs
        if not component_type or not isinstance(component_type, list):
            eval_logger.error(f"Invalid component_type: {component_type}")
            return {"total_score": 0}

        if not raw_score or not isinstance(raw_score, str):
            eval_logger.error(f"Invalid raw_score: {raw_score}")
            # Initialize with zeros for all components
            for component in component_type:
                score_dict[component] = 0
            score_dict["total_score"] = 0
            return score_dict

        # More robust regex patterns to extract scores
        # Pattern 1: "score of component X: Y/Z" or "component X: Y/Z"
        component_pattern = r"(?:score\s+of\s+)?component\s+(\d+)\s*:\s*(\d+)\s*/\s*(\d+)"

        # Pattern 2: "total score: Y/Z"
        total_pattern = r"total\s+score\s*:\s*(\d+)\s*/\s*(\d+)"

        # Find all component scores
        try:
            component_matches = re.findall(component_pattern, raw_score, re.IGNORECASE)

            for match in component_matches:
                try:
                    component_num = int(match[0]) - 1  # Convert to 0-based index
                    if 0 <= component_num < len(component_type):
                        numerator = int(match[1].strip())
                        denominator = int(match[2].strip())
                        score = numerator / denominator if denominator != 0 else 0
                        # Clamp score between 0 and 1
                        score = max(0, min(1, score))
                        score_dict[component_type[component_num]] = score
                    else:
                        eval_logger.warning(f"Component number {component_num + 1} out of range for {len(component_type)} components")
                except (ValueError, IndexError) as e:
                    eval_logger.warning(f"Error parsing component match {match}: {e}")
                    continue

        except Exception as e:
            eval_logger.error(f"Error in component pattern matching: {e}")

        # Find total score
        try:
            total_match = re.search(total_pattern, raw_score, re.IGNORECASE)
            if total_match:
                total_numerator = int(total_match.group(1).strip())
                total_denominator = int(total_match.group(2).strip())
                total_score = total_numerator / total_denominator if total_denominator != 0 else 0
                # Clamp total score between 0 and 1
                total_score = max(0, min(1, total_score))
                score_dict["total_score"] = total_score
            else:
                # Fallback: calculate total as average of component scores
                if score_dict:
                    total_score = sum(score_dict.values()) / len(score_dict)
                    score_dict["total_score"] = total_score
                else:
                    score_dict["total_score"] = 0
        except Exception as e:
            eval_logger.error(f"Error parsing total score: {e}")
            score_dict["total_score"] = 0

        # Ensure all components have scores
        for i, component in enumerate(component_type):
            if component not in score_dict:
                eval_logger.warning(f"Missing score for component: {component}")
                score_dict[component] = 0

        # Ensure total_score exists
        if "total_score" not in score_dict:
            if score_dict:
                score_dict["total_score"] = sum(v for k, v in score_dict.items() if k != "total_score") / len([k for k in score_dict.keys() if k != "total_score"])
            else:
                score_dict["total_score"] = 0

    except Exception as e:
        eval_logger.error(f"Unexpected error in process_rawscore: {e}")
        eval_logger.error(f"Raw score content: {raw_score[:500] if raw_score else 'None'}...")
        # Emergency fallback: return zeros
        score_dict = {}
        for component in component_type if component_type else []:
            score_dict[component] = 0
        score_dict["total_score"] = 0

    # Final validation
    try:
        # Ensure all values are numeric and within valid range
        for key, value in score_dict.items():
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                eval_logger.warning(f"Invalid score for {key}: {value}, setting to 0")
                score_dict[key] = 0
    except Exception as e:
        eval_logger.error(f"Error in final validation: {e}")

    return score_dict


def mia_bench_process_results(doc, results):
    from loguru import logger as eval_logger

    try:
        # Validate inputs
        if not results or len(results) == 0:
            eval_logger.error("No results provided")
            return {"gpt_eval_score": {"total_score": 0}}

        if not doc or not isinstance(doc, dict):
            eval_logger.error(f"Invalid doc: {doc}")
            return {"gpt_eval_score": {"total_score": 0}}

        # Extract response safely
        try:
            response = results[0].strip() if results[0] else ""
            if not response:
                eval_logger.warning("Empty response from model")
                response = ""
        except (IndexError, AttributeError) as e:
            eval_logger.error(f"Error extracting response: {e}")
            response = ""

        # Extract components safely
        try:
            components = doc.get("components", [])
            if not components or not isinstance(components, list):
                eval_logger.error(f"Invalid components in doc: {components}")
                return {"gpt_eval_score": {"total_score": 0}}
        except Exception as e:
            eval_logger.error(f"Error extracting components: {e}")
            return {"gpt_eval_score": {"total_score": 0}}

        # Generate evaluation prompt
        try:
            eval_prompt = generate_prompt(doc, response)
            if not eval_prompt:
                eval_logger.error("Failed to generate evaluation prompt")
                # Create fallback score_dict
                score_dict = {"total_score": 0}
                for component in components:
                    score_dict[component] = 0
                return {"gpt_eval_score": score_dict}
        except Exception as e:
            eval_logger.error(f"Error generating prompt: {e}")
            # Create fallback score_dict
            score_dict = {"total_score": 0}
            for component in components:
                score_dict[component] = 0
            return {"gpt_eval_score": score_dict}

        # Get evaluation from LLM
        try:
            eval_score, _ = get_eval(eval_prompt, 1024)
            if not eval_score or not isinstance(eval_score, str):
                eval_logger.warning(f"Invalid eval_score returned: {eval_score}")
                eval_score = ""
        except Exception as e:
            eval_logger.error(f"Error getting evaluation from LLM: {e}")
            eval_score = ""

        # Process the raw score
        try:
            score_dict = process_rawscore(components, eval_score)
            if not score_dict or not isinstance(score_dict, dict):
                eval_logger.error("process_rawscore returned invalid result")
                # Create fallback score_dict
                score_dict = {"total_score": 0}
                for component in components:
                    score_dict[component] = 0
        except Exception as e:
            eval_logger.error(f"Error processing raw score: {e}")
            # Create fallback score_dict
            score_dict = {"total_score": 0}
            for component in components:
                score_dict[component] = 0

        return {"gpt_eval_score": score_dict}

    except Exception as e:
        eval_logger.error(f"Unexpected error in mia_bench_process_results: {e}")
        # Emergency fallback
        return {"gpt_eval_score": {"total_score": 0}}


# ============================
# Aggregation Functions
# ============================


def mia_bench_aggregate_results(results):
    total_score = 0
    for result in results:
        # Overall accuracy
        total_score += result["total_score"]
    return total_score / len(results)
