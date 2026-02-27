from lmms_eval.tasks._task_utils.reasoning_utils import (
    compute_score,
    make_reasoning_doc_to_messages,
)


def seed_doc_to_visual(doc):
    return [image.convert("RGB") for image in doc["image"]]


def seed_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    question += "\n" + f"A. {doc['choice_a']}\n"
    question += f"B. {doc['choice_b']}\n"
    question += f"C. {doc['choice_c']}\n"
    question += f"D. {doc['choice_d']}"
    return f"{question}\nAnswer with the option's letter from the given choices directly."


seed_reasoning_doc_to_messages = make_reasoning_doc_to_messages(seed_doc_to_visual, seed_doc_to_text)


def seed_reasoning_process_results(doc, results):
    data_type = doc["data_type"]
    question = seed_doc_to_text(doc, None)
    ground_truth = doc["answer"]
    extra_info = {"question": question}

    acc_score = 0
    format_score = 0
    for pred in results:
        score_dict = compute_score(data_source="seedbench", solution_str=pred.strip(), ground_truth=ground_truth, extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    n = len(results) or 1
    return {
        f"seed_{data_type}_acc_score": acc_score / n,
        f"seed_{data_type}_format_score": format_score / n,
        "seed_all_acc_score": acc_score / n,
        "seed_all_format_score": format_score / n,
    }
