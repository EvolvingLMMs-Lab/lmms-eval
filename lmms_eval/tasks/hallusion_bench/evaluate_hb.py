import os
import json
import logging

from lmms_eval.tasks.hallusion_bench.utils import evaluate_by_chatgpt, check_same_by_chatgpt, assign_correctness, get_eval_all, get_eval_fig, get_eval_pair_all

os.makedirs("./submissions", exist_ok=True)
save_json_path = "./submissions/hallusion_output_vd_model.json"
output_entry = "model_prediction"
correctness_entry = "gpt4v_output_gpt_check"

metric = ["aAcc", "fAcc", "qAcc"]

eval_logger = logging.getLogger("lmms-eval")


def hb_doc_to_text(doc):
    return doc["question"]


def hb_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def hb_process_results(doc, result):
    sample = doc
    sample["model_prediction"] = result[0]
    return {k: sample for k in metric}


def hb_aggregation_result(results, metric):
    if not os.path.exists(save_json_path):
        eval_logger.info("Do GPT eval ...")
        results = evaluate_by_chatgpt(results, output_entry=output_entry, correctness_entry=correctness_entry, load_json=True, save_json_path=save_json_path)
        results = check_same_by_chatgpt(results, output_entry=output_entry, load_json=True, save_json_path=save_json_path)
    else:
        eval_logger.info("Load from exisiting files ...")
        results = json.load(open(save_json_path, "r"))

    results = assign_correctness(results, correctness_entry=correctness_entry)
    if metric == "aAcc":
        all_data = get_eval_all(results, model_correctness_entry=correctness_entry)
        return round(100 * all_data["correct"] / all_data["total"], 4)
    elif metric == "fAcc":
        fig_all = get_eval_fig(results)
        return round(100 * fig_all["correct"] / fig_all["total"], 4)
    elif metric == "qAcc":
        all_data = get_eval_pair_all(results, model_correctness_entry=correctness_entry)
        return round(100 * all_data["correct"] / all_data["total"], 4)


def hb_aggregation_result_qAcc(results):
    return hb_aggregation_result(results, "qAcc")


def hb_aggregation_result_fAcc(results):
    return hb_aggregation_result(results, "fAcc")


def hb_aggregation_result_aAcc(results):
    return hb_aggregation_result(results, "aAcc")


def hb_aggregation_result_intern(results, metric):
    scores = []
    for result in results:
        ans = "1" if result["model_prediction"].lower().find("yes") != -1 else "0"
        scores.append(ans == result["gt_answer"])
        result["answer"] = ans

    if metric == "aAcc":
        return sum(scores) / len(scores)
    elif metric == "qAcc":
        qlist = {}
        for r in results:
            key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
            try:
                qlist[key].append(r["answer"] == r["gt_answer"])
            except:
                qlist[key] = [r["answer"] == r["gt_answer"]]
        out = []
        for q, v in qlist.items():
            out.append(min(v))

        return sum(out) / len(out)
    elif metric == "fAcc":
        qlist = {}
        for r in results:
            key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["figure_id"])])
            try:
                qlist[key].append(r["answer"] == r["gt_answer"])
            except:
                qlist[key] = [r["answer"] == r["gt_answer"]]
        out = []
        for q, v in qlist.items():
            out.append(min(v))
        return sum(out) / len(out)


def hb_aggregation_result_qAcc_intern(results):
    eval_logger.info("Calculating qAcc ...")
    return hb_aggregation_result_intern(results, "qAcc")


def hb_aggregation_result_fAcc_intern(results):
    eval_logger.info("Calculating fAcc ...")
    return hb_aggregation_result_intern(results, "fAcc")


def hb_aggregation_result_aAcc_intern(results):
    eval_logger.info("Calculating aAcc ...")
    return hb_aggregation_result_intern(results, "aAcc")
