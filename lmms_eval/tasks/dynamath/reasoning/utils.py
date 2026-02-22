from collections import defaultdict

from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)


def worst(metrics, sizes, weight_by_size=False):
    if not weight_by_size:
        sizes = [1] * len(sizes)

    assert len(metrics) == len(sizes)

    return min([metric * size for metric, size in zip(metrics, sizes)])


def dynamath_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    return doc["question"]


def dynamath_doc_to_visual(doc):
    return [doc["decoded_image"].convert("RGB")]


def dynamath_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
    question = dynamath_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = dynamath_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def dynamath_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = dynamath_doc_to_text_cot(doc, None)
    extra_info = {"question": question}
    for pred in results:
        score_dict = compute_score(data_source="dynamath", solution_str=pred.strip(), ground_truth=doc["ground_truth"], extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)
    data_dict = {"acc": score_dict["acc_score"], "question_id": doc["question_id"], "variant_id": doc["id"]}

    return {"dynamath_average": data_dict, "dynamath_worst": data_dict}


def dynamath_aggregate_results(results, metric_name):
    variant_to_results = defaultdict(list)
    for result in results:
        variant_to_results[result["variant_id"]].append(result)

    score = 0.0
    for variant, results in variant_to_results.items():
        if metric_name == "dynamath_average":
            score += sum([result["acc"] for result in results]) / len(results)
        elif metric_name == "dynamath_worst":
            score += min([result["acc"] for result in results])
        else:
            raise ValueError(f"Invalid metric name: {metric_name}")
    return score / len(variant_to_results)


def dynamath_aggregate_results_worst(results):
    return dynamath_aggregate_results(results, "dynamath_worst")


def dynamath_aggregate_results_average(results):
    return dynamath_aggregate_results(results, "dynamath_average")
