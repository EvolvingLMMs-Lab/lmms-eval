# TODO: IMPLEMENT LMM JUDGE
# IMPLEMENT NO LMM JUDGEs
# INTERLEAVE FORMAT
# doc to message, interlvade
import datetime
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from latex2sympy2 import latex2sympy
from PIL import Image
from sympy import simplify
from word2number import w2n

from lmms_eval.llm_judge import get_server
from lmms_eval.llm_judge.protocol import ServerConfig
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional

eval_logger = logging.getLogger("lmms-eval")

dir_name = os.path.dirname(os.path.abspath(__file__))


emma_config = {
    "Strategy_Instruction": {"CoT": "Please solve the problem step by step.", "Directly": "Please ensure that your output only contains the final answer without any additional content (such as intermediate reasoning steps)."},
    "multi_choice_format": '{context}\n{question}\n{options}\nAnswer with the option\'s letter from the given choices and put the letter in one "\\boxed{{}}". ',
    "open_ended_format": '{context}\n{question}\nAnswer the question using a single word or phrase and put the answer in one "\\boxed{{}}". ',
}

with open(Path(__file__).parent / "emma_all.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
cache_dir = os.path.join(hf_home, config["dataset_kwargs"]["cache_dir"])


# Initialize the LLM judge server
if config["metadata"]["use_lmms_judge"]:
    eval_logger.info("Using LMMS judge server for EMMA task.")
    API_TYPE = os.getenv("API_TYPE", "azure")  # Default to azure based on .env
    # For Azure OpenAI, use DEPLOYMENT_NAME as the model_name
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")

    server_config = ServerConfig(
        model_name=DEPLOYMENT_NAME,  # Use deployment name for Azure OpenAI
    )
    server = get_server(server_name=API_TYPE, config=server_config)


def build_query(sample):
    """Build the text query by combining the context, question and options. The <image_n> token is still there
    Return a dictionary with the query and ground truth content.
    """
    context = sample["context"]
    question = sample["question"]
    example = ""
    res_dict = {}
    strategy = config["metadata"]["strategy"]
    if sample["type"].lower() == "multiple choice":
        options = sample["options"]
        start_chr = "A"
        for option in options:
            example += f"{start_chr}: {option}\n"
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt_sample_structure = emma_config["multi_choice_format"]
        empty_prompt = empty_prompt_sample_structure.format(context=context, question=question, options=example)
        if strategy == "CoT":
            res_dict["query"] = empty_prompt + emma_config["Strategy_Instruction"]["CoT"]
        else:
            res_dict["query"] = empty_prompt + emma_config["Strategy_Instruction"]["Directly"]

        res_dict["gt_content"] = options[ord(sample["answer"].upper()) - ord("A")]
    else:
        empty_prompt_sample_structure = emma_config["open_ended_format"]
        empty_prompt = empty_prompt_sample_structure.format(context=context, question=question)
        if strategy == "CoT":
            res_dict["query"] = empty_prompt + emma_config["Strategy_Instruction"]["CoT"]
        else:
            res_dict["query"] = empty_prompt + emma_config["Strategy_Instruction"]["Directly"]
        res_dict["gt_content"] = sample["answer"]

    # append existing key and value in data
    res_dict.update(sample)
    return res_dict


def replace_images_tokens(input_string):
    "Function to replace <image_n> tokens with a single <image> token. Use only for interleaved format."
    for i in range(1, 5):
        question_text = f"<image_{i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def emma_doc_to_text(doc):
    res_dict = build_query(doc)
    return res_dict["query"]


def emma_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    res_dict = build_query(doc)
    if config["metadata"]["interleaved_format"]:  # ON DEVELOPMENT
        pass
    else:
        image_tokens = re.findall(r"<image_\d+>", res_dict["query"])
        image_tokens = sorted(list(set([image_token.strip("<>").replace(" ", "_") for image_token in image_tokens])))
        visual = [doc[image_token].convert("RGB") for image_token in image_tokens]
        return visual


def emma_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    # If you use doc to messages, the interleaved format is always used
    config["metadata"]["interleaved_format"] = True

    # Get the text query and visual data
    res_dict = build_query(doc)
    query = res_dict["query"]

    # Get all images based on image tokens in the query
    image_tokens = re.findall(r"<image_\d+>", query)
    image_tokens = sorted(list(set([image_token.strip("<>").replace(" ", "_") for image_token in image_tokens])))
    visuals = [doc[image_token].convert("RGB") for image_token in image_tokens]

    # Replace EMMA's <image_n> tokens with generic <image> tokens for splitting
    processed_query = replace_images_tokens(query)

    # Initialize message structure
    messages = [{"role": "user", "content": []}]

    # Split text by <image> tokens
    interleaved_content = processed_query.split("<image>")

    # Build interleaved content
    for i, (image, text) in enumerate(zip(visuals, interleaved_content)):
        # Add text part if not empty
        if text.strip() != "":
            messages[0]["content"].append({"type": "text", "text": text.strip()})
        # Add image part
        messages[0]["content"].append({"type": "image", "url": image})

    # There will be one more text part after the last image
    if len(interleaved_content) > len(visuals) and interleaved_content[-1].strip():
        messages[0]["content"].append({"type": "text", "text": interleaved_content[-1].strip()})

    return messages


def emma_process_results(doc, results):
    key_name = "emma_score"
    for pred in results:
        res_dict = build_query(doc)
        gt = doc["answer"]
        query = res_dict["query"]

        if config["metadata"]["use_lmms_judge"]:
            # Use LMM judge to evaluate the prediction
            submit_prompt = create_test_prompt(score_demo_prompt, doc, pred)
            print("Run here")
            try:
                # Create a Request object for the unified judge API
                from lmms_eval.llm_judge.protocol import Request

                request = Request(messages=[{"role": "user", "content": submit_prompt}], config=server_config)

                # Send the request to the LMM judge server
                judge_response_obj = server.evaluate(request)
                judge_response = judge_response_obj.content
                judge_result = judge_response.strip().lower()
                print(f"Judge response: {judge_response}")

                # Parse the judge result to determine correctness
                is_correct = "correct" in judge_result and "incorrect" not in judge_result

                emma_submission = {"id": doc["pid"], "query": query, "gt_content": gt, "pred": pred, "subject": doc["subject"], "category": doc["category"], "judge_response": judge_response, "is_correct": is_correct}

            except Exception as e:
                eval_logger.error(f"Error using LMM judge: {e}")
                # Fallback to fast_extract_answer if judge fails
                pred_extracted = fast_extract_answer(pred)
                is_correct = is_equal(pred_extracted, gt)

                emma_submission = {"id": doc["pid"], "query": query, "gt_content": gt, "pred": pred, "subject": doc["subject"], "category": doc["category"], "judge_error": str(e), "is_correct": is_correct}

        else:
            # for no lmms judge, use fast_extract_answer only
            pred = fast_extract_answer(pred)
            emma_submission = {"id": doc["pid"], "query": query, "gt_content": gt, "pred": pred, "subject": doc["subject"], "category": doc["category"], "is_correct": is_equal(pred, gt)}
            # Note: the key name here is very important. It decides which aggregation function will receive the results
            # We note down the question id/category to help us aggregate the results later
        return {key_name: emma_submission}


def emma_aggregate_results(results):
    subject_to_eval_samples = defaultdict(list)
    category_to_eval_samples = defaultdict(list)
    total_samples = len(results)
    total_correct = 0

    for sample in results:
        subject = sample["subject"]
        category = sample["category"]

        # Check if using LMM judge results or traditional evaluation
        if "is_correct" in sample:
            # Use LMM judge result
            is_correct = sample["is_correct"]
        else:
            # Use traditional evaluation method
            is_correct = is_equal(sample["pred"], sample["gt_content"])

        if is_correct:
            total_correct += 1
            subject_to_eval_samples[subject].append(1)
            category_to_eval_samples[category].append(1)
        else:
            subject_to_eval_samples[subject].append(0)
            category_to_eval_samples[category].append(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    subject_accuracies = {subject: sum(scores) / len(scores) for subject, scores in subject_to_eval_samples.items()}
    category_accuracies = {category: sum(scores) / len(scores) for category, scores in category_to_eval_samples.items()}
    print(f"{'Total Samples':<20}: {total_samples}")
    print(f"{'Total Correct':<20}: {total_correct}")
    print(f"{'Overall Accuracy':<20}: {accuracy:.4f}")
    print()

    print(f"{'Per-Subject Accuracy':<40}")
    print("-" * 40)
    for subject, acc in subject_accuracies.items():
        print(f"{subject:<20}: {acc:.4f}")
    print()

    print(f"{'Per-Category Accuracy':<40}")
    print("-" * 40)
    for category, acc in category_accuracies.items():
        print(f"{category:<20}: {acc:.4f}")
    print("=" * 40)
    return accuracy


#################################################
# Helper functions written by official EMMA repo.
#################################################


def extract_full_boxed_content(s):
    """
    https://github.com/EMMA-Bench/EMMA/blob/main/evaluation/utils.py#L22
    Extract the full content inside \boxed{}, handling nested braces {{}} properly.
    """
    results = []

    i = 0
    while i < len(s):
        if s[i : i + 7] == r"\boxed{":
            brace_stack = []
            start = i + 7
            i = start

            while i < len(s):
                if s[i] == "{":
                    brace_stack.append(i)
                elif s[i] == "}":
                    if brace_stack:
                        brace_stack.pop()
                    else:
                        results.append(s[start:i])
                        break
                i += 1
        i += 1

    return results


def is_number(s):
    """https://github.com/EMMA-Bench/EMMA/blob/main/evaluation/utils.py#L14"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_equal(md_ans, gt_ans):
    # https://github.com/EMMA-Bench/EMMA/blob/main/evaluation/utils.py#L50

    md_ans = md_ans.lower()
    gt_ans = gt_ans.lower()

    if md_ans.strip() == gt_ans.strip():
        return True

    try:
        md_ans_cache = str(w2n.word_to_num(md_ans))
        if md_ans_cache.strip() == gt_ans.strip():
            return True
    except ValueError:
        pass

    # For Math
    try:
        # Parse LaTeX expressions into sympy and compare numerical values
        md_sympy = latex2sympy(md_ans)
        gt_sympy = latex2sympy(gt_ans)

        # Compare evaluated results, rounded to 2 decimal places
        if round(float(md_sympy.evalf()), 2) == round(float(gt_sympy.evalf()), 2):
            return True

        # Additionally, compare simplified symbolic expressions
        if simplify(md_sympy - gt_sympy) == 0:
            return True
    except Exception:
        pass  # Ignore parsing errors or evaluation failures

    return False


def fast_extract_answer(response):
    """
    https://github.com/EMMA-Bench/EMMA/blob/main/evaluation/evaluate.py#L11
    """
    response = response.strip()
    # Direct Strategy Multi-Choice
    # A / A: / A.
    for ch in "ABCDEFGH":
        if response.upper() == ch or response.startswith(f"{ch}:") or response.startswith(f"{ch}."):
            return ch

    # Direct Strategy Open-ended
    # 1
    if is_number(response):
        return response

    # CoT strategy
    if "boxed{" in response:
        try:
            model_answers = extract_full_boxed_content(response)
            if model_answers:
                # for coding
                # \\boxed{\\text{}}
                try:
                    text_content = re.findall(r"\\text{(.*?)}", model_answers[-1])
                    if text_content:
                        return text_content[-1].strip()
                except Exception:
                    pass
                return model_answers[-1].strip()
        except Exception:
            pass

    # for Coding
    # the correct answer is\n D.
    for flag in ["final answer is", "correct answer is", "answer should be", "answer is", "answer:"]:
        if flag in response.lower():
            try:
                model_answer = response.lower().split(flag)[-1].strip()
                return model_answer.split("\n")[0].split(".")[0]
            except Exception:
                pass

    return ""


def create_test_prompt(score_prompt, problem, pred):
    """
    https://github.com/EMMA-Bench/EMMA/blob/main/evaluation/evaluate.py#L54
    """
    score_prompt = score_prompt.strip()
    # response = problem[label]
    response = pred
    answer = problem["answer"]
    full_prompt = f"{score_prompt}\n" + f"Response: {response}\n" + f"Answer: {answer}\n" + "Correct_or_not:"
    return full_prompt


score_demo_prompt = """Please read the following example. Then determine whether the response is correct and type it 
at the end of the prompt. It is worth noting that the final answer in the response is usually in \\boxed{}, 
You only need to compare the final answer in the response with the answer, without considering the logical 
correctness of the response itself.

Response: The correct answer is:\n\nA

Answer: A

Correct_or_not: Correct

Response: The correct option is:\n\n\\[\n\\boxed{E}\n\\]

Answer: C

Correct_or_not: Incorrect
"""


# Alternative implementation using base64 encoding (currently not used, adopted from EMMA repo)
# def create_message(sample):
#     """
#     Alternative implementation for creating messages with base64 encoded images.
#     This would require implementing encode_image_to_base64 function.
#     """
#     query = sample["query"]
#     all_contents = []
#     matches = re.findall(r"<(image_\d+)>", query)
#     split_text = re.split(r"<image_\d+>", query)
#     for i, fragment in enumerate(split_text):
#         if fragment.strip():
#             all_contents.extend([{"type": "text", "text": fragment}])
#         if i < len(matches):
#             if sample[matches[i]]:
#                 # Would need to implement encode_image_to_base64 function
#                 # img_base64 = encode_image_to_base64(sample[matches[i]])
#                 # all_contents.extend([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}])
#                 all_contents.extend([{"type": "image", "url": sample[matches[i]]}])
#             else:
#                 eval_logger.error(f"The image token {matches[i]} is in the query, but there is no corresponding image provided by the data")
#
#     messages = [{"role": "user", "content": all_contents}]
#     return messages
