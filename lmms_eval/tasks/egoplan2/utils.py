import re

import datasets

egoplan2_features = datasets.Features(
    {
        "sample_id": datasets.Value("string"),
        "domain": datasets.Value("string"),
        "task_goal": datasets.Value("string"),
        "task_start_frame": datasets.Value("int64"),
        "current_observation_frame": datasets.Value("int64"),
        "formatted_question": datasets.Value("string"),
        "choice_a": datasets.Value("string"),
        "choice_b": datasets.Value("string"),
        "choice_c": datasets.Value("string"),
        "choice_d": datasets.Value("string"),
        "ground_truth": datasets.Value("string"),
        "video_file": datasets.Value("string"),
        "keyframes": datasets.Sequence(datasets.Image(decode=True)),
    }
)


def egoplan2_doc_to_visual(doc):
    return doc["keyframes"]


def egoplan2_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["formatted_question"]


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def egoplan2_process_results(doc, results):
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    # Only keep fields needed for aggregation (exclude keyframes to avoid OOM
    # during multi-GPU gather_object which pickles the entire dict).
    data_dict = {
        "sample_id": doc.get("sample_id"),
        "pred_answer": pred_ans,
        "ground_truth": doc["ground_truth"],
    }
    return {"egoplan2_mcq_accuracy": data_dict}


def egoplan2_aggregate_results(results):
    correct_num = 0
    for result in results:
        if result["pred_answer"] == result["ground_truth"]:
            correct_num += 1
    question_num = len(results)
    accuracy = correct_num / question_num
    return accuracy
