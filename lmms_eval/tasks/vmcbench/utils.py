import os
import numpy as np
import random


dir_name = os.path.dirname(os.path.abspath(__file__))


datasets_category_map = {
    "SEEDBench": "general",
    "MMStar": "general",
    "A-OKVQA": "general",
    "VizWiz": "general",
    "MMVet": "general",
    "VQAv2": "general",
    "OKVQA": "general",
    "MMMU": "reason",
    "MathVista": "reason",
    "ScienceQA": "reason",
    "RealWorldQA": "reason",
    "GQA": "reason",
    "MathVision": "reason",
    "TextVQA": "ocr",
    "OCRVQA": "ocr",
    "AI2D": "doc",
    "ChartQA": "doc",
    "DocVQA": "doc",
    "InfoVQA": "doc",
    "TableVQABench": "doc",
}


def vmcbench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vmcbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    options = {cand: doc[cand] for cand in "ABCD"}
    options_prompt = "Options:\n"
    for key, item in options.items():
        options_prompt += f"{key}. {item}\n"

    prompt = f"Question: {question}\n{options_prompt}"

    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        prompt = f"{lmms_eval_specific_kwargs['pre_prompt']}{prompt}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        prompt = f"{prompt}{lmms_eval_specific_kwargs['post_prompt']}"

    return prompt


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    response = str(response)
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response or f"{choice}. " in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice} " in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def vmcbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    response = results[0]

    all_choices = []
    for i in range(4):
        all_choices.append(chr(65 + i))
    index2ans = {index: doc[index] for index in all_choices}

    pred_index = parse_multi_choice_response(response, all_choices, index2ans)
    answer = doc["answer"]

    score = int(pred_index == answer)

    category = doc["category"]
    main_category = datasets_category_map[category]
    return {
        main_category: {"question_id": doc["index"], "category": category, "score": score},
        "average": {"question_id": doc["index"], "category": category, "score": score},
    }


def vmcbench_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    scores = [result["score"] for result in results]
    return sum(scores) / len(scores)
