import json
import os
from pathlib import Path

import ipdb
import pandas as pd
import yaml
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.mmsearch.constants import *
from lmms_eval.tasks.mmsearch.prompts.prompt import *
from lmms_eval.tasks.mmsearch.prompts.prompt_w_imagesearch import *
from lmms_eval.tasks.mmsearch.retrieve_content.retriever import Content_Retriever
from lmms_eval.tasks.mmsearch.score.f1_score import get_f1_score
from lmms_eval.tasks.mmsearch.score.req_score import get_requery_score
from lmms_eval.tasks.mmsearch.score.result_summary import get_result_summary
from lmms_eval.tasks.mmsearch.utils.image_utils import pil_image_to_bytes
from lmms_eval.tasks.mmsearch.utils.lmms_eval_utils import *
from lmms_eval.tasks.mmsearch.utils.prompt_utils import *
from lmms_eval.tasks.mmsearch.utils.utils import *

with open(Path(__file__).parent / "mmsearch.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

# constants
brief_result_num = 8
fullpage_num = 1
content_retriever = Content_Retriever()


def mmsearch_end2end_doc_to_text(doc, lmms_eval_specific_kwargs=None, previous_output=None, round_idx=None, previous_round_info=None):
    '''
    Returns: 
        visuals (for next round) 
        contexts (for next round)
        terminal_signal
        round_result
        previous_round_info
    '''
    # prepare save dir
    middle_result_dir = lmms_eval_specific_kwargs["middle_resules_dir"] if lmms_eval_specific_kwargs is not None and "middle_resules_dir" in lmms_eval_specific_kwargs else "mmsearch_middile_results"
    result_cache_dir = lmms_eval_specific_kwargs["result_cache_dir"] if lmms_eval_specific_kwargs is not None and "result_cache_dir" in lmms_eval_specific_kwargs else "mmsearch_result_cache_dir"
    os.makedirs(middle_result_dir, exist_ok=True)
    os.makedirs(result_cache_dir, exist_ok=True)
    # prepare query information
    if doc["query_image"] is None:
        query_has_image = False
        prompt_template_dict = text_query_dict
    else:  # query with image
        query_has_image = True
        prompt_template_dict = image_search_text_query_dict
    query = doc["query"]

    # initial round: round_idx is None. This remains the same output format as other benchmark
    eval_logger.info('----------------Round1: Requery----------------')
    if round_idx is None:
        prompt_template = prompt_template_dict["stage1"]
        if not query_has_image:
            text_query = prompt_template.format(question=query)
        else:
            text_query = prompt_template.format(question=DEFAULT_IMAGE_TOKEN + query, image_search_result=DEFAULT_IMAGE_TOKEN)
        return text_query
    # round2: search result + rerank
    if round_idx == 1:
        # if exist, return. This check has to be done here to avoid many
        cache_path = os.path.join(result_cache_dir, f"{doc['sample_id']}.json")
        if os.path.exists(cache_path):
            eval_logger.info(f"{doc['sample_id']} already exists. Load the cache result.")
            round_res = json.load(open(cache_path))["round_res"]
            return None, None, True, round_res, None
        eval_logger.info('----------------Round2: Rerank----------------')
        # prepare
        requery = previous_output[-1]
        stage1_screenshot_dir = os.path.join(middle_result_dir, doc["sample_id"], "stage1")

        # search result
        result_brief = search_text_brief_result(query=requery, max_result_num=brief_result_num, screenshot_dir=stage1_screenshot_dir)  # relative path  # [{'title', 'text','screenshot_path', 'url'}]

        if result_brief is None:  # the search engine returns None to the requery
            round_res = [requery, None, None]
            save_result_to_cache(doc, round_res, dict(), result_cache_dir)
            return None, None, True, round_res, None

        website_information, input_image_list = get_website_information(result_brief)
        input_image_list = [Image.open(f).convert("RGB") for f in input_image_list]

        prompt_template = prompt_template_dict["stage2"]
        if not query_has_image:
            image_files = input_image_list
            text_query = prompt_template.format(brief_result_num=brief_result_num, rerank_num=fullpage_num, question=query, website_information=website_information, incontext_example=get_rerank_incontext_example(fullpage_num))
        else:
            image_files = [doc["query_image"].convert("RGB"), doc["image_search_result"].convert("RGB"), *input_image_list]
            text_query = prompt_template.format(
                brief_result_num=brief_result_num,
                rerank_num=fullpage_num,
                question=DEFAULT_IMAGE_TOKEN + query,
                image_search_result=DEFAULT_IMAGE_TOKEN,
                website_information=website_information,
                incontext_example=get_rerank_incontext_example(fullpage_num),
            )

        image_files[0] = image_files[0].copy()
        return image_files, text_query, False, previous_output, dict(result_brief=result_brief)
    # round3: get full page + summarization
    if round_idx == 2:
        eval_logger.info('----------------Round3: Summarization----------------')
        # prepare
        stage3_screenshot_dir = os.path.join(middle_result_dir, doc["sample_id"], "stage3")
        requery = previous_output[0]
        rerank = previous_output[1]
        result_brief = previous_round_info["result_brief"]

        # postprocess the rerank result
        selected_index, _ = postprocess_rerank(rerank, fullpage_num)
        selected_website = [result_brief[i] for i in selected_index]
        result_full = search_url_full_result(urls=[web["url"] for web in selected_website], screenshot_dir=stage3_screenshot_dir)  # relative path  # [{'content', 'screenshot_fullpage_path'}]

        # add title and snippet
        for full_idx, brief_idx in enumerate(selected_index):
            result_full[full_idx]["title"] = result_brief[brief_idx]["title"]
            result_full[full_idx]["snippet"] = result_brief[brief_idx]["snippet"]

        # conduct content retrieval
        for idx, inst_full in enumerate(result_full):
            if inst_full["content"] is None:  # in case cannot get web content
                inst_full["content"] = ""
            if inst_full["content"].strip() != "":  # some web do not contain language content
                result_full[idx]["content"] = content_retriever.get_retrieved_content(requery, inst_full["content"])

        website_full_information, input_image_list = get_full_website_information(result_full=result_full, image_dir=stage3_screenshot_dir, fullpage_split_dict=FULLPAGE_SPLIT_DICT)

        input_image_list = [Image.open(f).convert("RGB") for f in input_image_list]
        # text_query and input_image_list
        prompt_template = prompt_template_dict["stage3"]
        if not query_has_image:
            image_files = input_image_list
            text_query = prompt_template.format(
                rerank_num=fullpage_num,
                website_information=website_full_information,
                question=query,
            )
        else:
            image_files = [*input_image_list, doc["image_search_result"].convert("RGB"), doc["query_image"].convert("RGB")]
            # assume only 1 image in the query
            text_query = prompt_template.format(rerank_num=fullpage_num, website_information=website_full_information, image_search_result=DEFAULT_IMAGE_TOKEN, question=DEFAULT_IMAGE_TOKEN + query)

        image_files[0] = image_files[0].copy()
        return image_files, text_query, False, previous_output, dict(result_brief=result_brief, website_full_information=website_full_information)
    # the process should terminate
    if round_idx == 3:
        save_result_to_cache(doc, previous_output, previous_round_info, result_cache_dir)
        return None, None, True, previous_output, None


def mmsearch_end2end_doc_to_visual(doc):
    if doc["query_image"] is None:
        return []
    return [doc["query_image"].convert("RGB").copy(), doc["image_search_result"].convert("RGB")]  # .copy is a workround of the type judgement in llava-ov


def mmsearch_rerank_doc_to_visual(doc):
    image_list = []
    # query image
    if doc["query_image"] is not None:
        image_list.extend([doc["query_image"].convert("RGB"), doc["image_search_result"].convert("RGB")])
    # website screenshot
    image_list.extend(doc[f"website{idx}_head_screenshot"].convert("RGB") for idx in range(brief_result_num))  # there are 8 webpages in the dataset

    # a workround to pass the type judgement in llava-ov
    image_list[0] = image_list[0].copy()
    return image_list


def mmsearch_rerank_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if doc["query_image"] is None:
        query_has_image = False
        prompt_template_dict = text_query_dict
    else:
        query_has_image = True
        prompt_template_dict = image_search_text_query_dict

    result_brief = [dict(**doc[f"website{i}_info"], screenshot_path=doc[f"website{i}_head_screenshot"]) for i in range(brief_result_num)]  # [{'title', 'text','screenshot_path', 'd'}]
    query = doc["query"]

    website_information, _ = get_website_information(result_brief)

    # add query image
    prompt_template = prompt_template_dict["stage2"]
    if not query_has_image:
        text_query = prompt_template.format(brief_result_num=brief_result_num, rerank_num=fullpage_num, question=query, website_information=website_information, incontext_example=get_rerank_incontext_example(fullpage_num))
    else:
        text_query = prompt_template.format(
            brief_result_num=brief_result_num,
            rerank_num=fullpage_num,
            question=DEFAULT_IMAGE_TOKEN + query,
            image_search_result=DEFAULT_IMAGE_TOKEN,
            website_information=website_information,
            incontext_example=get_rerank_incontext_example(fullpage_num),
        )
    return text_query


def mmsearch_summarization_doc_to_visual(doc):
    # from https://github.com/CaraJ7/MMSearch/blob/main/eval_summarization.py
    # set up prompt
    if doc["query_image"] is None:
        query_has_image = False
        prompt_template_dict = text_query_dict
    else:
        query_has_image = True
        prompt_template_dict = image_search_text_query_dict

    result_full = [
        dict(
            title=doc["website_title"],
            snippet=doc["website_snippet"],
            content=doc["website_retrieved_content"],
            slimmed_website_fullpage_screenshot=pil_image_to_bytes(doc["website_fullpage_screenshot"]),
        )
    ]  # the screenshot from the dataset has already been slimmed
    _, input_image_list = get_full_website_information(result_full=result_full, fullpage_split_dict=FULLPAGE_SPLIT_DICT)

    # add query image in the input image files
    if not query_has_image:
        image_files = [Image.open(f).convert("RGB") for f in input_image_list]
    else:
        image_files = [*[Image.open(f).convert("RGB") for f in input_image_list], doc["image_search_result"].convert("RGB"), doc["query_image"].convert("RGB")]

    # a workround to pass the type judgement in llava-ov
    image_files[0] = image_files[0].copy()
    return image_files


def mmsearch_summarization_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # from https://github.com/CaraJ7/MMSearch/blob/main/eval_summarization.py
    # set up prompt
    if doc["query_image"] is None:
        query_has_image = False
        prompt_template_dict = text_query_dict
    else:
        query_has_image = True
        prompt_template_dict = image_search_text_query_dict

    result_full = [
        dict(
            title=doc["website_title"],
            snippet=doc["website_snippet"],
            content=doc["website_retrieved_content"],
            slimmed_website_fullpage_screenshot=pil_image_to_bytes(doc["website_fullpage_screenshot"]),
        )
    ]  # the screenshot from the dataset has already been slimmed
    website_full_information, input_image_list = get_full_website_information(result_full=result_full, fullpage_split_dict=FULLPAGE_SPLIT_DICT)
    query = doc["query"]

    # add query image in the input image files
    prompt_template = prompt_template_dict["stage3"]
    if not query_has_image:
        text_query = prompt_template.format(
            rerank_num=fullpage_num,
            website_information=website_full_information,
            question=query,
        )
    else:
        # assume only 1 image in the query
        text_query = prompt_template.format(rerank_num=fullpage_num, website_information=website_full_information, image_search_result=DEFAULT_IMAGE_TOKEN, question=DEFAULT_IMAGE_TOKEN + query)
    return text_query


def mmsearch_end2end_process_results(doc, results):
    round_res = results[0]
    result = {
        "sample_id": doc["sample_id"],
        "query": doc["query"],
        "timestamp": doc["timestamp"],
        "area": doc["area"],
        "subfield": doc["subfield"],
        "gt_answer": doc["gt_answer"],
        "gt_requery": doc["gt_requery"],
        "alternative_gt_answers": doc["alternative_gt_answers"],
        "requery_prediction": round_res[0],
        "answer_prediction": round_res[2],
    }

    return {
        "end2end_f1_score": result,
        "requery_score": result,
    }


def mmsearch_rerank_process_results(doc, results):
    prediction = results[0].strip()

    result = {
        "sample_id": doc["sample_id"],
        "query": doc["query"],
        "timestamp": doc["timestamp"],
        "area": doc["area"],
        "subfield": doc["subfield"],
        "gt_answer": doc["gt_answer"],
        "rerank_prediction": prediction,
        "valid": doc["valid"],
        "not_sure": doc["not_sure"],
        "invalid": doc["invalid"],
    }

    return {
        "rek_score": result,
    }


def mmsearch_summarization_process_results(doc, results):
    prediction = results[0].strip()

    result = {
        "sample_id": doc["sample_id"],
        "query": doc["query"],
        "timestamp": doc["timestamp"],
        "area": doc["area"],
        "subfield": doc["subfield"],
        "gt_answer": doc["gt_answer"],
        "alternative_gt_answers": doc["alternative_gt_answers"],
        "answer_prediction": prediction,
    }

    return {
        "summarization_f1_score": result,
    }


def mmsearch_aggregate_results_f1_score(results, args, *, calculate_gain=False, random_scores=None):
    result_list = []
    for inst in results:
        prediction = inst["answer_prediction"]
        gt_answer = inst["gt_answer"]
        f1_score = get_f1_score(prediction, gt_answer)
        for gt_alternative_answer in inst["alternative_gt_answers"]:
            alternative_f1_score = get_f1_score(prediction, gt_alternative_answer)
            if alternative_f1_score > f1_score:
                f1_score = alternative_f1_score
        inst.update(dict(f1_score=f1_score))
        result_list.append(inst)

    # assert len(result_list) == 300 # assert to be the benchmark length, or the get_result_summary function will not work
    # save results
    path = generate_submission_file(f"{args.tasks}_f1_results.json", args)
    with open(path, "w") as f:
        json.dump(result_list, f, indent=4)
    # save scores
    result_summary = get_result_summary(result_list, result_list, summary_key="f1_score")
    path = generate_submission_file(f"{args.tasks}_f1_score.json", args)
    with open(path, "w") as f:
        json.dump(result_summary, f, indent=4)
    avg_f1_score = result_summary["f1_score"]["total_dict"]["average"]
    return avg_f1_score


def mmsearch_aggregate_results_req_score(results, args, *, calculate_gain=False, random_scores=None):
    result_list = []
    for inst in results:
        requery = inst["requery_prediction"]
        gt_requery = inst["gt_requery"]
        req_score = get_requery_score(requery, gt_requery)
        inst.update(
            dict(
                req_score=req_score["score"],
                req_score_dict=req_score,
            )
        )
        result_list.append(inst)

    assert len(result_list) == 300 # assert to be the benchmark length, or the get_result_summary function will not work
    # save results
    path = generate_submission_file(f"{args.tasks}_requery_results.json", args)
    with open(path, "w") as f:
        json.dump(result_list, f, indent=4)
    # save scores
    result_summary = get_result_summary(result_list, result_list, summary_key="req_score")
    path = generate_submission_file(f"{args.tasks}_requery_score.json", args)
    with open(path, "w") as f:
        json.dump(result_summary, f, indent=4)
    avg_req_score = result_summary["req_score"]["total_dict"]["average"]
    return avg_req_score


def mmsearch_aggregate_results_rek_score(results, args, *, calculate_gain=False, random_scores=None):
    result_list = []
    for inst in results:
        rerank = inst["rerank_prediction"]
        selected_index, valid = postprocess_rerank(rerank, fullpage_num)
        selected_index = selected_index[0]  # only take the first one

        if not valid:
            score = 0
        elif selected_index in inst["valid"]:
            score = 1
        elif selected_index in inst["not_sure"]:
            score = 0.5
        else:
            score = 0

        inst.update(
            dict(
                model_output_valid=valid,
                parsed_answer_rank=selected_index,
                rer_score=score,
            )
        )
        result_list.append(inst)
    assert len(result_list) == 300  # assert to be the benchmark length, or the get_result_summary function will not work

    # save results
    path = generate_submission_file(f"{args.tasks}_rerank_results.json", args)
    with open(path, "w") as f:
        json.dump(result_list, f, indent=4)
    # save score
    result_summary = get_result_summary(result_list, result_list, summary_key="rer_score")
    path = generate_submission_file(f"{args.tasks}_rerank_score.json", args)
    with open(path, "w") as f:
        json.dump(result_summary, f, indent=4)
    avg_rerank_score = result_summary["rer_score"]["total_dict"]["average"]
    return avg_rerank_score
