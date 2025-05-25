"""
Example refactored version of llava-bench-coco utils using unified judge interface
This demonstrates how to migrate existing code to use the new unified interface
"""

import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from loguru import logger as eval_logger

# Import the unified judge interface
from lmms_eval.api.judge import JudgeConfig, JudgeRequest, get_judge
from lmms_eval.api.judge_utils import evaluate_with_judge, parse_score

NUM_SECONDS_TO_SLEEP = 0.5

LLAVA_W_METRICS = ["gpt_eval_llava_conv", "gpt_eval_llava_detail", "gpt_eval_llava_complex"]

rule_dict = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rule.json"), "r"))

with open(Path(__file__).parent / "llava-bench-coco.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

GPT_EVAL_MODEL_NAME = config["metadata"]["gpt_eval_model_name"]


def get_eval(content: str, max_tokens: int, retries: int = 3):
    """
    Refactored get_eval using unified judge interface
    """
    # Use the unified evaluation function
    response_content, model_used = evaluate_with_judge(
        prompt=content,
        model_name=GPT_EVAL_MODEL_NAME,
        system_prompt="You are a helpful and precise assistant for checking the quality of the answer.",
        temperature=0.2,
        max_tokens=max_tokens,
        num_retries=retries,
    )

    return response_content, model_used


def llava_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def llava_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"


def llava_process_results(doc, result):
    """
    Process results using the unified judge interface

    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case coco_bleu), value: metric value
    """
    try:
        question = doc.get("question", "")
        ans1 = doc.get("answer", "")
        ans2 = result[0] if result else ""
        captions = doc.get("caption", [])
        context = "\n".join(captions) if isinstance(captions, list) else captions
        category = "llava_bench_" + doc.get("category", "")
        rule = rule_dict.get(category, {})
        prompt = rule.get("prompt", "")
        role = rule.get("role", "user")

        # Format the evaluation prompt
        content = f"[Context]\n{context}\n\n" f"[Question]\n{question}\n\n" f"[{role} 1]\n{ans1}\n\n[End of {role} 1]\n\n" f"[{role} 2]\n{ans2}\n\n[End of {role} 2]\n\n" f"[System]\n{prompt}\n\n"

        # Get evaluation using unified interface
        review, model_name = get_eval(content, 1024)

        # Parse scores using utility function
        scores = parse_score(review)

    except Exception as e:
        eval_logger.error(f"Error for Question ID: {doc.get('question_id', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = "Failed Request"
        scores = [-1, -1]

    metric = f"gpt_eval_llava_{doc.get('category', 'unknown')}"
    category_review_dict = {"question": question, "ans1": ans1, "ans2": ans2, "context": context, "category": category, "review": review, "scores": scores, "eval_model": model_name, "content": content}

    non_category_review_dict = deepcopy(category_review_dict)
    non_category_review_dict["scores"] = [-999, -999]

    data_dict = {}
    for m in LLAVA_W_METRICS:
        if m == metric:
            data_dict[m] = category_review_dict
        else:
            data_dict[m] = non_category_review_dict
    data_dict["gpt_eval_llava_all"] = category_review_dict

    return data_dict


# Aggregation functions remain the same
def llava_conv_aggregation(results):
    return llava_aggregation(results, "conv")


def llava_complex_aggregation(results):
    return llava_aggregation(results, "complex")


def llava_detail_aggregation(results):
    return llava_aggregation(results, "detail")


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
        return round(stats[1] / stats[0] * 100, 1)
    except Exception as e:
        eval_logger.error(f"Error in llava_aggregation: {e}")
        return None


# Alternative implementation using a Judge instance directly
class LlavaJudgeEvaluator:
    """
    Example of using Judge instance directly for more complex evaluation scenarios
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or GPT_EVAL_MODEL_NAME

        # Create judge configuration
        self.judge_config = JudgeConfig(model_name=self.model_name, temperature=0.2, max_tokens=1024, num_retries=3, system_prompt="You are a helpful and precise assistant for checking the quality of the answer.")

        # Initialize judge
        self.judge = get_judge(config=self.judge_config)

    def evaluate_response(self, doc, result):
        """Evaluate a single response using the judge"""
        question = doc.get("question", "")
        ans1 = doc.get("answer", "")
        ans2 = result[0] if result else ""
        captions = doc.get("caption", [])
        context = "\n".join(captions) if isinstance(captions, list) else captions
        category = "llava_bench_" + doc.get("category", "")
        rule = rule_dict.get(category, {})
        prompt = rule.get("prompt", "")
        role = rule.get("role", "user")

        # Format the evaluation prompt
        content = f"[Context]\n{context}\n\n" f"[Question]\n{question}\n\n" f"[{role} 1]\n{ans1}\n\n[End of {role} 1]\n\n" f"[{role} 2]\n{ans2}\n\n[End of {role} 2]\n\n" f"[System]\n{prompt}\n\n"

        # Create judge request
        request = JudgeRequest(messages=[{"role": "user", "content": content}])

        try:
            # Get evaluation
            response = self.judge.evaluate(request)
            review = response.content
            model_name = response.model_used
            scores = parse_score(review)

        except Exception as e:
            eval_logger.error(f"Error evaluating Question ID {doc.get('question_id', 'Unknown')}: {e}")
            review = "Failed to Get a Proper Review."
            model_name = "Failed Request"
            scores = [-1, -1]

        return {"question": question, "ans1": ans1, "ans2": ans2, "context": context, "category": category, "review": review, "scores": scores, "eval_model": model_name, "content": content}
