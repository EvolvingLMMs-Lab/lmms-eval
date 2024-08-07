import json
import re
from collections import Counter, defaultdict
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def q_bench_doc_to_text(doc, lmms_eval_specific_kwargs):
    candidates = []
    for i in range(4):
        candidate = doc.get(f"option{i}")
        if candidate != "N/A":
            candidates.append(candidate)

    question = doc["question"] + "\n" + "\n".join([". ".join([chr(ord("A") + i), candidate]) for i, candidate in enumerate(candidates)])
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}\n{post_prompt}"


def q_bench_doc_to_visual(doc):
    if "image2" not in doc:
        return [doc["image"].convert("RGB")]
    else:
        return [doc["image1"].convert("RGB"), doc["image2"].convert("RGB")]


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
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
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
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


def evaluate_q_bench(samples):
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_i = sample["parsed_pred"]
        correct = eval_multi_choice(gold_i, pred_i)

        if correct:
            judge_dict[sample["id"]] = "Correct"
            pred_correct += 1
        else:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return {"acc": 0}
    return judge_dict, {"acc": pred_correct / len(samples)}


def eval_multi_choice(gold_i, pred_i):
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def calculate_ins_level_acc(results):
    """Calculate the instruction level accuracy for given Subject results
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L246
    """
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


def q_bench_process_results(doc, results):
    pred = results[0]
    all_choices = []
    index2ans = {}
    for i in range(4):
        option = doc.get(f"option{i}")
        if option == "N/A":
            break
        index2ans[chr(ord("A") + i)] = option
        all_choices.append(chr(ord("A") + i))

    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    id = doc["id"]
    qbench_acc = {"id": id, "question_concern": doc["question_concern"], "question_type": doc["question_type"], "answer": doc["correct_choice"], "parsed_pred": parsed_pred}
    return {
        "qbench_acc": qbench_acc,
        "submission": {
            id: pred,
        },
    }


concern_list = ["Global Distortion", "Global Others", "Local Distortion", "Local Others"]
question_list = ["Yes/No", "How", "What"]


def q_bench_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[concern_list[result["question_concern"]]].append(result)
        subset_to_eval_samples[question_list[result["question_type"]]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_q_bench(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}

    for cat_name, cat_results in evaluation_result.items():
        printable_results[cat_name] = {
            "num": int(cat_results["num_example"]),
            "acc": round(cat_results["acc"], 5),
        }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }
    print(printable_results)
    return printable_results["Overall"]["acc"]


def a_bench_process_results(doc, results):
    pred = results[0]
    all_choices = []
    index2ans = {}
    for i in range(4):
        option = doc.get(f"option{i}")
        if option == "N/A":
            break
        index2ans[chr(ord("A") + i)] = option
        all_choices.append(chr(ord("A") + i))

    parsed_pred = parse_multi_choice_response(pred, all_choices, index2ans)
    id = doc["id"]
    abench_acc = {"id": id, "category": doc["category"], "answer": doc["correct_choice"], "parsed_pred": parsed_pred}
    return {
        "abench_acc": abench_acc,
        "submission": {
            id: pred,
        },
    }


def a_bench_aggregate_results(results):
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["category"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        judge_dict, metric_dict = evaluate_q_bench(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}

    for cat_name, cat_results in evaluation_result.items():
        printable_results[cat_name] = {
            "num": int(cat_results["num_example"]),
            "acc": round(cat_results["acc"], 5),
        }
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }
    print(printable_results)
    return printable_results["Overall"]["acc"]
