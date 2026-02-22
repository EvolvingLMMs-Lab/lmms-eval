import re
from contextlib import contextmanager

from loguru import logger as eval_logger
from PIL import Image

Image.MAX_IMAGE_PIXELS = 10_0000_0000

TASK_PAIRs = [
    "Complex reasoning/Anomaly Detection and Interpretation",
    "Complex reasoning/Environmental condition reasoning",
    "Complex reasoning/Route planning",
    "Counting/Counting with changing detection",
    "Counting/Counting with complex reasoning",
    "Counting/Overall counting",
    "Counting/Regional counting",
    "Land use classification/Overall Land use classification",
    "Land use classification/Regional Land use classification",
    "Object properties/Object classification",
    "Object properties/Object color",
    "Object properties/Object motion state",
    "Object spatial relationship/Object spatial relationship",
]


def xlrs_doc_to_visual(doc):
    return [img.convert("RGB") for img in doc["image"]]


def xlrs_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    option_prompt = "The choices are listed below:\n" + "\n".join(doc["multi-choice options"]) + "\n"
    # pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    # post_promt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = ""
    assert doc["category"] in TASK_PAIRs, f"Unknown task: {doc['category']}"
    if doc["category"] == "Land use classification/Overall Land use classification":
        post_promt = "\nSelect the best answer(s) for the multiple-choice question based on the image. There may be more than one correct option. Only respond with the letter(s) corresponding to the correct answer(s) (A, B, C, D), with multiple choices separated by spaces.The answer(s) is(are):"
    else:
        post_promt = "\nSelect the best answer for the multiple-choice question based on the image. Only respond with the letter corresponding to the correct answer (A, B, C, D).\nThe answer is:"
    question += pre_prompt + option_prompt + post_promt
    return question


# [Image] [Question] The choices are listed below:
# (A) [Choice A]
# (B) [Choice B]
# (C) [Choice C]
# (D) [Choice D]
# (E) [Choice E]
# Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.
# The best answer is:


def extract_characters_regex(s, choices=["(A)", "(B)", "(C)", "(D)", "(E)"]):
    if type(s) is dict:
        s = ""
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option isThe correct option is",
        "Best answer:Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if not re.search("[ABCDE]", s):
        return ""
    matches = re.findall(r"\(([a-eA-E])\)", s)
    if len(matches) == 0:
        matches = re.findall(r"(?:^|\s)?([a-eA-E])(?:$|[\s,.])?", s)
    if len(matches) == 0:
        matches = re.findall(r"[a-eA-E]", s)
    if len(matches) == 0:
        return ""
    else:
        matches = set(mat.upper() for mat in matches)
        return "".join(matches)


def xlrs_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case xlrs_micro_score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    category, sub_category = doc["category"].split("/")[:2]
    task_category = doc["l2-category"]
    data_dict = {
        "question_id": doc["index"],
        "category": category,
        "sub_category": sub_category,
        "task_category": task_category,
        "pred_answer": pred_ans,
        "answer": doc["answer"],
    }

    return {"xlrs_micro_score": data_dict, "xlrs_macro_score": data_dict}


def xlrs_aggregate_results(results, macro=False):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """

    metrics = {}
    for task_pair in TASK_PAIRs:
        task, subtask = task_pair.split("/")
        if task not in metrics.keys():
            metrics[task] = {}
        metrics[f"{task}"][f"{subtask}"] = {}
    # for task in TASKS:
    #     metrics[f"{task}"] = {}
    #     for subtask in SUBTASKS:
    #         metrics[f"{task}"][f"{subtask}"] = {}

    for i in range(len(results)):
        result = results[i]
        Task = result["category"]
        Subtask = result["sub_category"]
        Category = result["task_category"].lower()
        if "attribute" in Category.lower():
            Category = Category.split("/")[0] + "/attribute"
        # print(f"pred: {result['pred_answer']}, answer: {result['answer']}")
        # exact match
        cnt = 1 if set(result["pred_answer"]) == set(result["answer"]) else 0
        if Category not in metrics[Task][Subtask].keys():
            metrics[Task][Subtask][f"{Category}"] = {
                "true": cnt,
                "false": 1 - cnt,
            }
        else:
            metrics[Task][Subtask][f"{Category}"]["true"] += cnt
            metrics[Task][Subtask][f"{Category}"]["false"] += 1 - cnt
    macros = []
    sum_all, succ_all = 0, 0
    for task, tasks_values in metrics.items():
        eval_logger.info("*" * 32 + f"{task} (Task Start)")
        cnt_task, sum_task = 0, 0
        for substask, subtask_value in tasks_values.items():
            eval_logger.info("+" * 16 + f"{substask} (Subtask Start)")
            cnt_subtask, sum_subtask, e_subtask = 0, 0, 0
            for category, category_dict in subtask_value.items():
                cnt_subtask += category_dict["true"]
                sum_subtask += category_dict["false"] + category_dict["true"]
                acc = category_dict["true"] / (category_dict["false"] + category_dict["true"])
                eval_logger.info("-" * 4 + "\t" + "Acc " + "{:.4f}".format(acc) + f"\t{category.capitalize()} ({category_dict['false'] + category_dict['true']} items)")

            if sum_subtask == 0:
                acc_subtasks = 0
            else:
                acc_subtasks = cnt_subtask / sum_subtask
            eval_logger.info("+" * 16 + "\t Acc " + "{:.4f}".format(acc_subtasks) + f"\t{substask} ({sum_subtask} items)")
            macros.append(acc_subtasks)
            cnt_task += cnt_subtask
            sum_task += sum_subtask

        if sum_task == 0:
            acc_task = 0
        else:
            acc_task = cnt_task / sum_task
        succ_all += cnt_task
        sum_all += sum_task
        eval_logger.info("*" * 32 + "Acc " + "{:.4f}".format(acc_task) + f"\t{task} ({sum_task} items)\n")
    eval_logger.info("*" * 32 + "Overall Acc " + "{:.4f}".format(succ_all / sum_all))
    if macro is True:
        return sum(macros) / len(macros)
    else:
        return succ_all / sum_all


@contextmanager
def mute_eval_logger():
    eval_logger.disable(__name__)
    try:
        yield
    finally:
        eval_logger.enable(__name__)


def xlrs_aggregate_results_macro_score(results):
    with mute_eval_logger():
        return xlrs_aggregate_results(results, macro=True)
