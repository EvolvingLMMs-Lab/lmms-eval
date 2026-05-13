# Copyright 2025 Xiaomi Corporation.

from lmms_eval.tasks._task_utils.reasoning_utils import (
    compute_score,
    make_reasoning_doc_to_messages,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


def olympiadbench_doc_to_visual(doc):
    res = []
    for i in range(1, 6):
        image_key = f"image_{i}"
        if doc[image_key] is not None:
            res.append(doc[image_key])
    return [image.convert("RGB") for image in res]


def olympiadbench_doc_to_text(doc):
    question = doc["question"]
    subject = doc["subfield"]
    mul_ans = doc["is_multiple_answer"]
    if mul_ans is None:
        mul_ans = False
    ans_type = doc["answer_type"]
    if ans_type == "Need_human_evaluate":
        ans_type = "proof based"

    language = doc["language"]
    if language == "English":
        pre_prompt = f"The following is a question from an International {subject} competition.\n"

        post_prompt = ""
        if not mul_ans:
            post_prompt += f"The answer of question should be {ans_type}.\n"
        else:
            post_prompt += f"The question has multiple answers, each of them should be {ans_type}.\n"

    elif language == "Chinese":
        pre_prompt = f"以下是中国{subject}竞赛中的解答题。\n"

        post_prompt = ""
        if not mul_ans:
            post_prompt += f"答案类型为{ans_type}。\n"
        else:
            post_prompt += f"题目有多个答案，答案类型均为{ans_type}。\n"

    final_question = pre_prompt + question + "\n" + post_prompt
    return final_question


def _olympiadbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return olympiadbench_doc_to_text(doc)


olympiadbench_doc_to_messages = make_reasoning_doc_to_messages(olympiadbench_doc_to_visual, _olympiadbench_doc_to_text, system_prompt=SYSTEM_PROMPT)


def olympiadbench_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = olympiadbench_doc_to_text(doc)
    ground_truth = doc["final_answer"][0] if doc["final_answer"] else ""
    extra_info = {"question": question}
    for pred in results:
        score_dict = compute_score(data_source="olympiadbench_official", solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}
