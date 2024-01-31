import csv
import json
from tqdm import tqdm
import numpy as np
import os
import time
import openai
import threading
import requests
import logging

API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")

eval_logger = logging.getLogger("lmms-eval")


def evaluate_by_chatgpt(data, output_entry, correctness_entry, gpt_model="gpt-4", load_json=False, save_json_path="./hallusion_output.json", retries=3):
    if load_json and os.path.exists(save_json_path):
        with open(save_json_path, "r") as f:
            output = json.load(f)
    else:
        output = []
    for sample in tqdm(data[len(output) :]):
        prompt = "Imagine you are an intelligent teacher. Thoroughly read the question, reference answer and the prediction answer to ensure a clear understanding of the information provided. Assess the correctness of the predictions. "
        prompt += 'If the prediction answer does not conflict with the reference answer, please generate “correct”. If the prediction answer conflict with the reference answer, please generate “incorrect”. If the prediction answer is unclear about the answer, please generate "unclear". \n\n Question:'
        prompt += sample["question"]
        prompt += "\nReference answer: "
        prompt += sample["gt_answer_details"]
        prompt += "\nPrediction answer:"
        prompt += sample[output_entry]
        prompt += "\nOutput:"

        # https://github.com/openai/openai-python/issues/322#issuecomment-1767841683
        for attempt in range(retries):
            try:
                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                }

                messages = [{"role": "user", "content": prompt}]

                payload = {
                    "model": gpt_model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 1024,
                }
                response = requests.post(API_URL, headers=headers, json=payload)
                response.raise_for_status()
                response = response.json()
                break
            except Exception as e:
                eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                if attempt < retries - 1:  # If we have retries left, sleep and then continue to next attempt
                    time.sleep(5)
                else:  # If this was the last attempt, log and return empty
                    eval_logger.error(f"All {retries} attempts failed. Last error message: {str(e)}")

        output_text = response["choices"][0]["message"]["content"]

        if "incorrect" in output_text.lower():
            gpt_correctness = "0"

        elif "correct" in output_text.lower():
            gpt_correctness = "1"
        else:
            gpt_correctness = "2"

        sample[correctness_entry] = gpt_correctness
        sample.pop("image")
        output.append(sample)

    with open(save_json_path, "w") as f:
        json.dump(output, f)

    return output


def check_same_by_chatgpt(data, output_entry, gpt_model="gpt-4", load_json=False, save_json_path="./hallusion_output.json", retries=3):
    orig_response = {}

    for r in data:
        if str(r["figure_id"]) == "0":
            key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
            orig_response[key] = r[output_entry]

    for sample in tqdm(data):
        if "same" not in sample.keys():
            key = "_".join([sample["category"], sample["subcategory"], str(sample["set_id"]), str(sample["question_id"])])
            response2 = orig_response[key]

            prompt = "Imagine you are an intelligent teacher. Thoroughly read the two responses to two different questions. Assess the consistency of the information provided within those two responses. "
            prompt += "You do not know the specific questions, but you can asssess the consistency among the two responses by checking for logical conflicts if both responses are correct. "
            prompt += 'If response1 does not conflict with response2, please generate “same”. Otherwise, generate "different". \n\n response1:'
            prompt += sample[output_entry]
            prompt += "\nresponse2: "
            prompt += response2
            prompt += "\nOutput:"

            # https://github.com/openai/openai-python/issues/322#issuecomment-1767841683
            for attempt in range(retries):
                try:
                    headers = {
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    }

                    messages = [{"role": "user", "content": prompt}]

                    payload = {
                        "model": gpt_model,
                        "messages": messages,
                        "temperature": 0.2,
                        "max_tokens": 1024,
                    }
                    response = requests.post(API_URL, headers=headers, json=payload)
                    response.raise_for_status()
                    response = response.json()

                    break
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if attempt < retries - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(5)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All {retries} attempts failed. Last error message: {str(e)}")

            output_text = response["choices"][0]["message"]["content"]

            gpt_same = "0"

            if "same" in output_text.lower():
                gpt_same = "1"

            elif "different" in output_text.lower():
                gpt_same = "0"

            sample["same"] = gpt_same

            with open(save_json_path, "w") as f:
                json.dump(data, f)

    return data


def assign_correctness(data_arr, correctness_entry):
    for r in data_arr:
        assert int(r[correctness_entry]) == 0 or int(r[correctness_entry]) == 1 or int(r[correctness_entry]) == 2
        if r["category"] == "VS" and int(r["figure_id"]) == 0:  # if there is no visual supplement and the model does not know, count it as correct
            r["correct"] = 1 if int(r[correctness_entry]) == 1 or int(r[correctness_entry]) == 2 else 0
        else:
            r["correct"] = 1 if int(r[correctness_entry]) == 1 else 0
    return data_arr


def get_eval_fig(data):  # per figure
    eval_fig_dict = dict()

    for r in data:
        if r["category"] == "VS" and str(r["figure_id"]) == "0":  # no figure
            continue
        name = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["figure_id"])])
        if name in eval_fig_dict:
            c, t = eval_fig_dict[name]
            eval_fig_dict[name] = (c + r["correct"], t + 1)
        else:
            eval_fig_dict[name] = (r["correct"], 1)

    eval_fig_stat = {}
    eval_fig_stat["note"] = "all accuracy per image (consistency test)"
    eval_fig_stat["total"] = len(eval_fig_dict.keys())
    eval_fig_stat["correct"] = 0
    eval_fig_stat["wrong"] = 0
    eval_fig_stat["inconsistent"] = 0
    eval_fig_stat["score"] = 0

    for v in eval_fig_dict.values():
        if v[0] == v[1]:
            eval_fig_stat["correct"] += 1
        elif v[0] == 0:
            eval_fig_stat["wrong"] += 1
        else:
            eval_fig_stat["inconsistent"] += 1
        eval_fig_stat["score"] += v[0] / v[1]

    eval_fig_stat["score"] = eval_fig_stat["score"] / eval_fig_stat["total"]
    return eval_fig_stat


def get_eval_all(data, model_correctness_entry):  # per question
    eval_all_dict = dict()
    eval_all_stat = {}
    eval_all_stat["LH"] = 0
    eval_all_stat["VI"] = 0
    eval_all_stat["Mix"] = 0

    for r in data:
        name = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["figure_id"]), str(r["question_id"])])
        assert name not in eval_all_dict

        eval_all_dict[name] = r["correct"]

        if str(r["category"]) == "VD":  # VD
            if str(r["figure_id"]) == "0":
                if str(r[model_correctness_entry]) == "0" or str(r[model_correctness_entry]) == "2":
                    eval_all_stat["VI"] += 1
            else:
                if str(r[model_correctness_entry]) == "0":
                    eval_all_stat["Mix"] += 1
                elif str(r[model_correctness_entry]) == "2":
                    eval_all_stat["VI"] += 1
        else:  # VS
            if str(r["visual_input"]) == "0":  # no visual
                if str(r[model_correctness_entry]) == "0":
                    eval_all_stat["LH"] += 1
            else:  # original visual or modified visual (isual_input == 1 or 2)
                if str(r[model_correctness_entry]) == "0":
                    eval_all_stat["Mix"] += 1
                elif str(r[model_correctness_entry]) == "2":
                    eval_all_stat["VI"] += 1

    eval_all_stat["note"] = "all accuracy per question"
    eval_all_stat["total"] = len(eval_all_dict.keys())
    eval_all_stat["correct"] = np.count_nonzero(list(eval_all_dict.values()))
    eval_all_stat["wrong"] = eval_all_stat["total"] - eval_all_stat["correct"]

    return eval_all_stat


def get_eval_pair_all(data, model_correctness_entry):  # per question pair
    orig_correctness = dict()
    counter = 0
    lh_counter = 0
    vi_counter = 0
    both_counter = 0

    for r in data:
        if str(r["figure_id"]) == "0":
            key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
            orig_correctness[key] = r[model_correctness_entry]

    get_eval_pair_dict = dict()
    get_analysis_pair_dict = dict()

    for r in data:
        name = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
        if name in get_eval_pair_dict:
            c, t = get_eval_pair_dict[name]
            get_eval_pair_dict[name] = (c + r["correct"], t + 1)
        else:
            get_eval_pair_dict[name] = (r["correct"], 1)
        counter += 1

        # (LH, VI)
        analysis = (0, 0)
        if str(r["figure_id"]) == "0":  # when it's original question
            if str(r["category"]) == "VD":  # VD
                if str(r[model_correctness_entry]) == "0" or str(r[model_correctness_entry]) == "2":
                    analysis = (0, 1)  # VI -- get original image wrong, bad vision
            else:  # VS
                if str(r[model_correctness_entry]) == "0":
                    analysis = (1, 0)  # LH -- wrong answer without visual, making things up
        else:  # when it's not original question
            key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
            orig_c = orig_correctness[key]
            if str(r["category"]) == "VD":  # VD
                if str(orig_c) == "1" and str(r[model_correctness_entry]) == "0":
                    if str(r["same"]) == "1":
                        analysis = (1, 1)  # Mixed -- orig correct but modified wrong, with the same answer as the original question, could be bad vision or language hallucination
                    else:
                        analysis = (0, 1)  # VI -- orig correct but modified wrong, but answer differently, only due to bad vision
                elif str(orig_c) == "1" and str(r[model_correctness_entry]) == "2":
                    analysis = (0, 1)  # VI -- orig correct but modified uncertain, bad vision
                elif str(r[model_correctness_entry]) == "0" or str(r[model_correctness_entry]) == "2":
                    # when orig_c == 0 or 2 and current is wrong
                    analysis = (0, 1)  # VI -- when original is wrong and current is wrong, bad vision
            else:  # VS
                key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
                orig_c = orig_correctness[key]
                if str(orig_c) == "0":  # No visual wrong
                    if str(r[model_correctness_entry]) == "0" and str(r["same"]) == "1":
                        analysis = (1, 0)  # LH -- same answer with and without visual, LH overtake visual
                    elif str(r[model_correctness_entry]) == "0":
                        analysis = (1, 1)  # LH -- different answer with and without visual but both wrong, both language and visual are bad
                    elif str(r[model_correctness_entry]) == "2":
                        analysis = (1, 1)  # Mixed -- no visual wrong, but with visual uncertain, could be either
                elif str(orig_c) == "2":  # No visual uncertain
                    if str(r[model_correctness_entry]) == "0" or str(r[model_correctness_entry]) == "2":
                        analysis = (0, 1)  # VI -- no visual uncertain, with visual still wrong or uncertain, visual capability is bad
                else:  # No visual correct
                    if str(r[model_correctness_entry]) == "2":
                        analysis = (0, 1)  # VI -- no visual correct, with visual uncertain, visual capability is bad
                    elif str(r[model_correctness_entry]) == "0":  # current is wrong
                        if str(r["visual_input"]) == "1":  # common sense visual question
                            analysis = (0, 1)  # VI -- no visual correct, with visual wrong on common sense question, visual capability is bad
                        elif str(r["visual_input"]) == "2":  # counter-common sense visual question
                            if str(r["same"]) == "1":
                                analysis = (1, 0)  # LH -- with visual correct, but modified question wrong with the same answer, not considering visual so the error is attributed to Language
                            else:
                                analysis = (0, 1)  # VI -- with visual correct, but modified question wrong with different answers, visual capability is bad
                        else:
                            assert False, "Data error"

        if analysis[0] > 0 and analysis[1] > 0:
            both_counter += 1
        elif analysis[0] > 0:
            lh_counter += 1
        elif analysis[1] > 0:
            vi_counter += 1

        if name in get_analysis_pair_dict:
            lh, vi = get_analysis_pair_dict[name]
            get_analysis_pair_dict[name] = (lh + analysis[0], vi + analysis[1])
        else:
            get_analysis_pair_dict[name] = analysis

    eval_all_pair_stat = {}
    eval_all_pair_stat["note"] = "all accuracy per question pair"
    eval_all_pair_stat["total"] = len(get_eval_pair_dict.keys())
    eval_all_pair_stat["total_q"] = counter
    eval_all_pair_stat["correct"] = 0
    eval_all_pair_stat["wrong"] = 0
    eval_all_pair_stat["LH"] = 0
    eval_all_pair_stat["VI"] = 0
    eval_all_pair_stat["Mix"] = 0

    eval_all_pair_stat["LH_cg"] = lh_counter
    eval_all_pair_stat["VI_cg"] = vi_counter
    eval_all_pair_stat["Mix_cg"] = both_counter

    # for v in get_eval_pair_dict.values():
    #     if v[0] == v[1]:
    #         eval_all_pair_stat["correct"] += 1
    #     else:
    #         eval_all_pair_stat["wrong"] += 1

    # for v in get_analysis_pair_dict.values():
    #     if v[0] > 0 and v[1] > 0:
    #         eval_all_pair_stat["Mix"] += 1
    #     elif v[0] > 0:
    #         eval_all_pair_stat["LH"] += 1
    #     elif v[1] > 0:
    #         eval_all_pair_stat["VI"] += 1

    for k in get_eval_pair_dict.keys():
        v = get_eval_pair_dict[k]
        a = get_analysis_pair_dict[k]
        if v[0] == v[1]:
            eval_all_pair_stat["correct"] += 1
        else:
            eval_all_pair_stat["wrong"] += 1
        if a[0] > 0 and a[1] > 0:
            eval_all_pair_stat["Mix"] += 1
        elif a[0] > 0:
            eval_all_pair_stat["LH"] += 1
        elif a[1] > 0:
            eval_all_pair_stat["VI"] += 1

    assert eval_all_pair_stat["wrong"] == (eval_all_pair_stat["Mix"] + eval_all_pair_stat["LH"] + eval_all_pair_stat["VI"])

    return eval_all_pair_stat
