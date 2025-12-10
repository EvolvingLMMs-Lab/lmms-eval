from datasets import Dataset

from lmms_eval.tasks._task_utils.reasoning_utils import compute_score
from lmms_eval.tasks.charxiv.constant import DESCRIPTIVE_RESP_INST, REASONING_RESP_INST

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


def get_number_instruction(answer):
    base = answer.split(".")
    whole, decimal = base[0], None if len(base) == 1 else base[1]
    # check if it contains decimal places
    if whole is not None and decimal is None:
        inst = "* Your final answer must be an exact integer."
    elif whole is not None and decimal is not None:
        num_decimal = len(decimal)
        inst = f"* Your final answer must be a number with {num_decimal} decimal places."
    else:
        raise ValueError(f"Invalid answer: {answer}")
    return inst


def charxiv_reasoning_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    inst_category = doc["reasoning_q_source"]
    if inst_category in [1, 2, 3]:
        question = REASONING_RESP_INST[inst_category].format(doc["reasoning_q"])
    # 4: number-in-general -> need to specify the number of decimal places
    elif inst_category == 4:
        question = REASONING_RESP_INST[inst_category].format(doc["reasoning_q"], get_number_instruction(doc["reasoning_a"]))
    return question


def descriptive_query_helper(qid, subplot_loc):
    if qid in [18, 19]:
        # skip subplot location when asking about the layout of the subplots
        return DESCRIPTIVE_RESP_INST[qid]
    if isinstance(subplot_loc, list):
        if subplot_loc[0] == 0:
            # when there is only one subplot
            prefix = "For the current plot, "
        else:
            # when there are multiple subplots
            prefix = f"For the subplot at row {subplot_loc[0]} and column {subplot_loc[1]}, "
    # when subplots do not form a grid
    elif isinstance(subplot_loc, str):
        prefix = f"For {subplot_loc}, "
    else:
        raise ValueError(f"Invalid subplot_loc: {subplot_loc}")
    # return the question with the subplot location
    return DESCRIPTIVE_RESP_INST[qid].format(prefix)


def charxiv_descriptive_process_docs(dataset: Dataset) -> Dataset:
    # Four descriptive questions for each reasoning question
    dataset = dataset.repeat(4)

    def _process_row(example, indice):
        q_number = indice % 4 + 1
        descriptive_q = example[f"descriptive_q{q_number}"]
        subplot_loc = example["subplot_loc"]
        if subplot_loc is None:
            subplot_row = example["subplot_row"]
            subplot_col = example["subplot_col"]
            subplot_loc = [subplot_row, subplot_col]
        descriptive_q = descriptive_query_helper(descriptive_q, subplot_loc)
        example[f"descriptive_q"] = descriptive_q
        example[f"descriptive_a"] = example[f"descriptive_a{q_number}"]
        return example

    dataset = dataset.map(_process_row, with_indices=True, num_proc=4)
    return dataset


def charxiv_descriptive_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    return doc["descriptive_q"]


def charxiv_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def charxiv_descriptive_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
    question = charxiv_descriptive_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = charxiv_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def charxiv_reasoning_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
    question = charxiv_reasoning_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = charxiv_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def charxiv_reasoning_process_results(doc, results):
    acc_score = 0
    format_score = 0
    for pred in results:
        score_dict = compute_score(data_source="charxiv", solution_str=pred.strip(), ground_truth=doc["reasoning_a"])
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}


def charxiv_descriptive_process_results(doc, results):
    acc_score = 0
    format_score = 0
    for pred in results:
        score_dict = compute_score(data_source="charxiv", solution_str=pred.strip(), ground_truth=doc["descriptive_a"])
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}
