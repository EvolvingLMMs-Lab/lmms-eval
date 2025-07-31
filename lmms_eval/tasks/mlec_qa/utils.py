import re
import numpy as np
from loguru import logger as eval_logger


def mlec_qa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    "qtext": "高血压脑病的眼底表现是（　　）。 ",
    "options": {
      "A": "眼底正常",
      "B": "视网膜动脉变细",
      "C": "眼底出血渗出",
      "D": "视网膜动脉变窄，动静脉交叉压迫",
      "E": "眼底出血伴视乳头水肿"
    },
    "answer": "E"
    """
    question = doc["qtext"]
    options = "\n".join([f"{key}. {value}" for key, value in doc["options"].items()])
    return f"{question}\n{options}\n请只回答选项字母（如：A）。"


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    """
    all_choices = list(options.keys())
    index2ans = options
    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    ans_with_brack = False
    candidates = []

    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0:
        # Fallback to check for the content of the answer in the response
        for choice, content in index2ans.items():
            if content.lower() in response.lower():
                candidates.append(choice)

    if not candidates:
        return random.choice(all_choices)

    if len(candidates) > 1:
        start_indexes = []
        if ans_with_brack:
            for can in candidates:
                start_indexes.append(response.rfind(f"({can})"))
        else:
            # Check for the last occurrence of the choice
            for can in candidates:
                # A simple choice check
                start_indexes.append(response.rfind(f" {can} "))
        return candidates[np.argmax(start_indexes)]

    return candidates[0]


def mlec_qa_process_results(doc, results):
    pred = results[0]
    index2ans, all_choices = get_multi_choice_info(doc["options"])
    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)

    return {
        "mlec_qa_acc": {"qid": doc["qid"], "answer": doc["answer"], "parsed_pred": parsed_pred},
        "submission": {doc["qid"]: parsed_pred},
    }


def mlec_qa_aggregate_results(results):
    correct = sum(1 for res in results if res["answer"] == res["parsed_pred"])
    accuracy = correct / len(results)
    eval_logger.info(f"Total samples: {len(results)}")
    eval_logger.info(f"Accuracy: {accuracy:.4f}")
    return accuracy 