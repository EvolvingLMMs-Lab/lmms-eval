import hashlib
import json
import os

from tqdm import tqdm

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.hallusion_bench.utils import (
    assign_correctness,
    evaluate_by_chatgpt,
    get_eval_all,
    get_eval_fig,
    get_eval_pair_all,
)

cur_dir = os.path.dirname(os.path.abspath(__file__))
output_entry = "model_prediction"
correctness_entry = "gpt4v_output_gpt_check"

metric = ["aAcc", "fAcc", "qAcc"]

from loguru import logger as eval_logger  # noqa: E402


def hb_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question']}{post_prompt}"


def hb_doc_to_visual(doc):
    """Convert document to visual input."""
    num_image = int(os.environ.get("NUM_IMAGE", "1"))

    if num_image == 1:
        return [doc["image"].convert("RGB")]
    elif num_image == 2:
        return [doc["image"].convert("RGB"), doc["image"].convert("RGB")]
    else:
        raise ValueError(f"num_image must be 1 or 2, got {num_image}")


def hb_process_results(doc, result):
    sample = {k: v for k, v in doc.items() if k != "image"}
    sample["model_prediction"] = result[0]
    return {k: sample for k in metric}


def _judge_cache_file(split: str, data: list, args) -> str:
    # The GPT judgments are cached on disk, shared by aAcc/qAcc/fAcc. Key the file on the judged content so that
    # a run never reuses the judgments of an earlier run (another model, another --limit) in the same --output_path.
    keys = ("category", "subcategory", "set_id", "figure_id", "question_id", "question", "gt_answer_details", output_entry)
    digest = hashlib.sha256(json.dumps([[sample[key] for key in keys] for sample in data]).encode()).hexdigest()[:16]
    return generate_submission_file(f"hallusion_output_{split}_{digest}.json", args, subpath="gpt_response")


def hb_aggregation_result(results, metric, args):
    data_vd = []
    data_vs = []
    for data in tqdm(results, desc="Split vd and vs"):
        if data["category"] == "VD":
            data_vd.append(data)
        if data["category"] == "VS":
            data_vs.append(data)
    eval_logger.info("Do gpt eval vd ...")
    save_json_path_vd = _judge_cache_file("vd", data_vd, args)
    save_json_path_vs = _judge_cache_file("vs", data_vs, args)
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
            except Exception:
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
            except Exception:
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
