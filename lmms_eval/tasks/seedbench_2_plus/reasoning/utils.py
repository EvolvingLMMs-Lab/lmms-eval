from lmms_eval.tasks._task_utils.reasoning_utils import (
    make_reasoning_doc_to_messages,
    make_reasoning_process_results,
)


def parse_choice_img(choice: str, img_token: str = "<image>"):
    if "jpg" in choice or "png" in choice:
        return img_token
    return choice


def seed_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def seed_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    img_token = lmms_eval_specific_kwargs.get("img_token", "<image>") if lmms_eval_specific_kwargs else "<image>"
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "Answer with the option's letter from the given choices directly.") if lmms_eval_specific_kwargs else "Answer with the option's letter from the given choices directly."

    question.replace("<img>", img_token)
    question += "\n" + f"A. {parse_choice_img(doc['choice_A'], img_token)}\n"
    question += f"B. {parse_choice_img(doc['choice_B'], img_token)}\n"
    question += f"C. {parse_choice_img(doc['choice_C'], img_token)}\n"
    question += f"D. {parse_choice_img(doc['choice_D'], img_token)}"

    return f"{question}\n{post_prompt}"


seed_reasoning_doc_to_messages = make_reasoning_doc_to_messages(seed_doc_to_visual, seed_doc_to_text)

_base_process = make_reasoning_process_results("seedbench_2_plus", seed_doc_to_text)


def seed_reasoning_process_results(doc, results):
    data_type = doc["question_image_type"].capitalize()
    base = _base_process(doc, results)
    return {
        f"seedbench_2_plus_{data_type}_acc_score": base["acc_score"],
        f"seedbench_2_plus_{data_type}_format_score": base["format_score"],
        "seedbench_2_plus_all_acc_score": base["acc_score"],
        "seedbench_2_plus_all_format_score": base["format_score"],
    }
