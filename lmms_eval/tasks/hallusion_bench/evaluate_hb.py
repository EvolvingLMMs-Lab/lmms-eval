import os
import json

from tqdm import tqdm

from lmms_eval.tasks.hallusion_bench.utils import evaluate_by_chatgpt, check_same_by_chatgpt, assign_correctness, get_eval_all, get_eval_fig, get_eval_pair_all

cur_dir = os.path.dirname(os.path.abspath(__file__))
output_entry = "model_prediction"
correctness_entry = "gpt4v_output_gpt_check"

metric = ["aAcc", "fAcc", "qAcc"]

from loguru import logger as eval_logger


def hb_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = model_specific_prompt_kwargs.get("pre_prompt", "")
    post_prompt = model_specific_prompt_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"


def hb_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def hb_process_results(doc, result):
    sample = doc
    # doc.pop("image")
    sample["model_prediction"] = result[0]
    return {k: sample for k in metric}


def hb_aggregation_result(results, metric, args):
    data_vd = []
    data_vs = []
    for data in tqdm(results, desc="Split vd and vs"):
        if data["category"] == "VD":
            data_vd.append(data)
        if data["category"] == "VS":
            data_vs.append(data)
    eval_logger.info("Do gpt eval vd ...")
    path = os.path.join(args.output_path, "gpt_response")
    os.makedirs(path, exist_ok=True)
    save_json_path_vd = f"{path}/hallusion_output_vd_model.json"
    save_json_path_vs = f"{path}/hallusion_output_vs_model.json"
    data_vd = evaluate_by_chatgpt(data_vd, output_entry=output_entry, correctness_entry=correctness_entry, load_json=True, save_json_path=save_json_path_vd)
    # data_vd = check_same_by_chatgpt(data_vd, output_entry=output_entry, load_json=True, save_json_path=save_json_path_vd)
    data_vd = assign_correctness(data_vd, correctness_entry=correctness_entry)
    eval_logger.info("Do gpt eval vs")
    data_vs = evaluate_by_chatgpt(data_vs, output_entry=output_entry, correctness_entry=correctness_entry, load_json=True, save_json_path=save_json_path_vs)
    # data_vs = check_same_by_chatgpt(data_vs, output_entry=output_entry, load_json=True, save_json_path=save_json_path_vs)
    data_vs = assign_correctness(data_vs, correctness_entry=correctness_entry)
    results = data_vs + data_vd

    if metric == "aAcc":
        all_data = get_eval_all(results, model_correctness_entry=correctness_entry)
        return round(100 * all_data["correct"] / all_data["total"], 4)
    elif metric == "fAcc":
        fig_all = get_eval_fig(results)
        return round(100 * fig_all["correct"] / fig_all["total"], 4)
    elif metric == "qAcc":
        all_data = get_eval_pair_all(results, model_correctness_entry=correctness_entry)
        return round(100 * all_data["correct"] / all_data["total"], 4)


def hb_aggregation_result_qAcc(results, args):
    return hb_aggregation_result(results, "qAcc", args)


def hb_aggregation_result_fAcc(results, args):
    return hb_aggregation_result(results, "fAcc", args)


def hb_aggregation_result_aAcc(results, args):
    return hb_aggregation_result(results, "aAcc", args)


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
