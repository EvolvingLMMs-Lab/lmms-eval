import os

from datasets import Dataset
from openai import OpenAI
from tqdm import tqdm

from lmms_eval.tasks.charxiv.constant import REASONING_RESP_INST
from lmms_eval.tasks.charxiv.descriptive_utils import (
    build_descriptive_grading_queries,
    descriptive_query_helper,
    get_descriptive_result_gpt,
    postprocess_descriptive_grading_queries,
)
from lmms_eval.tasks.charxiv.reasoning_utils import (
    build_reasoning_grading_queries,
    get_number_instruction,
    get_reasoning_result_gpt,
)

# get environment else return dummy values, in a single line
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "YOUR_OPENAI_BASE_URL")
MODEL_VERSION = os.getenv("MODEL_VERSION", "YOUR_MODEL_VERSION")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def charxiv_reasoning_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    inst_category = doc["reasoning_q_source"]
    if inst_category in [1, 2, 3]:
        question = REASONING_RESP_INST[inst_category].format(doc["reasoning_q"])
    # 4: number-in-general -> need to specify the number of decimal places
    elif inst_category == 4:
        question = REASONING_RESP_INST[inst_category].format(doc["reasoning_q"], get_number_instruction(doc["reasoning_a"]))
    return question


def charxiv_descriptive_process_docs(dataset: Dataset) -> Dataset:
    # Four descriptive questions for each reasoning question
    dataset = dataset.repeat(4)

    def _process_row(example, indice):
        q_number = indice % 4 + 1
        descriptive_q = example[f"descriptive_q{q_number}"]
        qid = descriptive_q
        subplot_loc = example["subplot_loc"]
        if subplot_loc is None:
            subplot_row = example["subplot_row"]
            subplot_col = example["subplot_col"]
            subplot_loc = [subplot_row, subplot_col]
        descriptive_q = descriptive_query_helper(descriptive_q, subplot_loc)
        example[f"descriptive_q"] = descriptive_q
        example[f"descriptive_a"] = example[f"descriptive_a{q_number}"]
        return {"qid": qid, **example}

    dataset = dataset.map(_process_row, with_indices=True, num_proc=1)
    return dataset


def charxiv_descriptive_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    return doc["descriptive_q"]


def charxiv_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def charxiv_descriptive_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
    question = charxiv_descriptive_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = charxiv_doc_to_visual(doc)
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    return messages


def charxiv_reasoning_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
    question = charxiv_reasoning_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = charxiv_doc_to_visual(doc)
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    return messages


def charxiv_descriptive_process_results(doc, results):
    figure_id = doc["figure_path"]
    qid = doc["qid"]
    resp_key = f"{figure_id}_{qid}"
    response = results[0].strip()
    answer = doc["descriptive_a"]
    return {"descriptive_acc": {"group": (resp_key, response, answer), "qid": qid}}


def charxiv_descriptive_aggregate_results(results):
    groups = {i: [] for i in range(1, 20)}
    for result in results:
        groups[result["qid"]].append(result["group"])
    queries = build_descriptive_grading_queries(groups)
    combined_queries = []
    for query in tqdm(queries):
        result = get_descriptive_result_gpt(client, query["grading_query"], len(query["resp_keys"]), model=MODEL_VERSION)
        # query contains resp_keys, grading_query, extract_answer and score
        combined_queries.append({**query, **result})
    queries = combined_queries
    # flatten the queries and only keep the necessary fields
    queries = postprocess_descriptive_grading_queries(queries)
    # Return the average score
    scores = [query["score"] for query in queries.values()]
    return sum(scores) / len(scores)


def charxiv_reasoning_process_results(doc, results):
    figure_id = doc["figure_path"]
    inst_category = doc["reasoning_q_source"]
    response = results[0].strip()
    answer = doc["reasoning_a"]
    data = {}
    data["inst_category"] = inst_category
    data["figure_id"] = figure_id
    data["answer"] = answer
    resp_value = {"raw_question": doc["reasoning_q"], "response": response}
    return {"reasoning_acc": {"resp_value": resp_value, "resp_key": figure_id, "data": data}}


def charxiv_reasoning_aggregate_results(results):
    data = {}
    resps = {}
    for i, result in enumerate(results):
        data[i] = result["data"]
        resps[result["resp_key"]] = result["resp_value"]
    queries = build_reasoning_grading_queries(data, resps)
    for figure_id, query in tqdm(queries.items()):
        ext, scr = get_reasoning_result_gpt(client, query["grading_query"])
        queries[figure_id]["extracted_answer"] = ext
        queries[figure_id]["score"] = scr
        queries[figure_id].pop("grading_query")
    # Return the average score
    scores = [query["score"] for query in queries.values()]
    return sum(scores) / len(scores)
