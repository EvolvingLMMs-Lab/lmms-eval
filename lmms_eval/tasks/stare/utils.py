import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import yaml
from latex2sympy2 import latex2sympy
from sympy import simplify
from word2number import w2n

from lmms_eval.llm_judge import get_server
from lmms_eval.llm_judge.protocol import ServerConfig

eval_logger = logging.getLogger("lmms-eval")

dir_name = os.path.dirname(os.path.abspath(__file__))


stare_config = {
    "Strategy_Instruction": {"CoT": "Please solve the problem step by step.", "Directly": "Please ensure that your output only contains the final answer without any additional content (such as intermediate reasoning steps)."},
    "multi_choice_format": '\n{question}\nAnswer with the option\'s letter from the given choices and put the letter in one "\\boxed{{}}". ',
}

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")

# Initialize the LLM judge server
if config["metadata"]["use_lmms_judge"]:
    eval_logger.info("Using LMMS judge server for STARE task.")
    API_TYPE = os.getenv("API_TYPE", "azure")  # Default to azure based on .env
    # For Azure OpenAI, use DEPLOYMENT_NAME as the model_name
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4o")

    server_config = ServerConfig(
        model_name=DEPLOYMENT_NAME,  # Use deployment name for Azure OpenAI
    )
    server = get_server(server_name=API_TYPE, config=server_config)


# This is taken directly from the official STARE codebase
def build_query(sample):
    """
    Construct a multiple-choice query. Inserts <image> placeholder if missing.
    Appends either chain-of-thought (CoT) or direct instruction from config.
    """
    question = sample["question"]
    images = sample.get("images", [])
    answer = sample["answer"].strip().upper()
    strategy = config["metadata"]["strategy"]

    # Ensure <image> placeholder if images are provided
    if "<image>" not in question and images:
        question += "\n<image>"

    # Fill the template with the question
    prompt_template = stare_config["multi_choice_format"]
    filled_prompt = prompt_template.format(question=question)

    # Append the selected strategy instruction
    if strategy == "CoT":
        query = f"{filled_prompt}\n{stare_config['Strategy_Instruction']['CoT']}"
    else:
        query = f"{filled_prompt}\n{stare_config['Strategy_Instruction']['Directly']}"

    # Build the result dictionary
    res_dict = {"query": query, "gt_content": answer, **sample}
    return res_dict


def stare_doc_to_text(doc):
    """Extracts the text prompt from a STARE dataset sample.

    Args:
        doc (dict): A STARE dataset sample dictionary.

    Returns:
        str: The input prompt text.
    """
    res_dict = build_query(doc)
    return res_dict["query"]


def stare_doc_to_visual(doc):
    """Extracts and converts visual data from a STARE dataset sample.

    This function iterates over the 'images' key in the dataset sample, calling
    the .convert("RGB") method on each item (presumed to be a PIL/Pillow Image).

    Args:
        doc (dict): A STARE dataset sample dictionary.

    Returns:
        list: A list of visual elements (e.g., PIL Images) converted to 'RGB' format.
    """
    try:
        [visual.convert("RGB") for visual in doc["images"]]
    except:
        print("Not successful.")
        print(doc["qid"])
    return [visual.convert("RGB") for visual in doc["images"]]


def stare_process_results(doc, results):
    key_name = "stare_score"
    for pred in results:
        res_dict = build_query(doc)
        gt = doc["answer"]
        query = res_dict["query"]

        if config["metadata"]["use_lmms_judge"]:
            # Use LMM judge to evaluate the prediction
            print("doc", doc)
            print("\n")
            print("pred", pred)
            submit_prompt = create_test_prompt(score_demo_prompt, doc, pred)
            try:
                # Create a Request object for the unified judge API
                from lmms_eval.llm_judge.protocol import Request

                request = Request(messages=[{"role": "user", "content": submit_prompt}], config=server_config)

                # Send the request to the LMM judge server
                judge_response_obj = server.evaluate(request)
                judge_response = judge_response_obj.content
                judge_result = judge_response.strip().lower()

                # Parse the judge result to determine correctness
                is_correct = "correct" in judge_result and "incorrect" not in judge_result

                stare_submission = {"id": doc["qid"], "query": query, "gt_content": gt, "pred": pred, "category": doc["category"], "judge_response": judge_response, "is_correct": is_correct}

            except Exception as e:
                eval_logger.error(f"Error using LMM judge: {e}")
                # Fallback to fast_extract_answer if judge fails
                pred_extracted = fast_extract_answer(pred)
                is_correct = is_equal(pred_extracted, gt)

                stare_submission = {"id": doc["qid"], "query": query, "gt_content": gt, "pred": pred, "category": doc["category"], "judge_error": str(e), "is_correct": is_correct}

        else:
            # for no lmms judge, use fast_extract_answer only
            pred = fast_extract_answer(pred)
            stare_submission = {"id": doc["qid"], "query": query, "gt_content": gt, "pred": pred, "category": doc["category"], "is_correct": is_equal(pred, gt)}
        return {key_name: stare_submission}


def stare_aggregate_results(results):
    category_to_eval_samples = defaultdict(list)
    total_samples = len(results)
    total_correct = 0

    for sample in results:
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
            category_to_eval_samples[category].append(1)
        else:
            category_to_eval_samples[category].append(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    category_accuracies = {category: sum(scores) / len(scores) for category, scores in category_to_eval_samples.items()}
    print(f"{'Total Samples':<20}: {total_samples}")
    print(f"{'Total Correct':<20}: {total_correct}")
    print(f"{'Overall Accuracy':<20}: {accuracy:.4f}")
    print()

    print(f"{'Per-Category Accuracy':<40}")
    print("-" * 40)
    for category, acc in category_accuracies.items():
        print(f"{category:<20}: {acc:.4f}")
    print("=" * 40)
    return accuracy


#################################################
# Helper functions written by official STARE repo.
#################################################


def extract_full_boxed_content(s):
    """
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
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_equal(md_ans, gt_ans):

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
    response = response.strip()
    for ch in "ABCDEFGH":
        if response.upper() == ch or response.startswith(f"{ch}:") or response.startswith(f"{ch}."):
            return ch

    # Direct Strategy Open-ended
    if is_number(response):
        return response

    # CoT strategy
    if "boxed{" in response:
        try:
            model_answers = extract_full_boxed_content(response)
            if model_answers:
                try:
                    text_content = re.findall(r"\\text{(.*?)}", model_answers[-1])
                    if text_content:
                        return text_content[-1].strip()
                except Exception:
                    pass
                return model_answers[-1].strip()
        except Exception:
            pass

    for flag in ["final answer is", "correct answer is", "answer should be", "answer is", "answer:"]:
        if flag in response.lower():
            try:
                model_answer = response.lower().split(flag)[-1].strip()
                return model_answer.split("\n")[0].split(".")[0]
            except Exception:
                pass

    return ""


def create_test_prompt(score_prompt, problem, pred):
    score_prompt = score_prompt.strip()
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
