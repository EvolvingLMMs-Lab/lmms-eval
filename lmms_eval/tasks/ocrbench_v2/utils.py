from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

import re
import ast
from lmms_eval.tasks.ocrbench_v2.vqa_metric import vqa_evaluation, cn_vqa_evaluation, math_expression_evaluation, vqa_evaluation_case_sensitive, counting_evaluation, cn_math_expression_evaluation
from lmms_eval.tasks.ocrbench_v2.IoUscore_metric import vqa_with_position_evaluation, calculate_iou, extract_coordinates
from lmms_eval.tasks.ocrbench_v2.TEDS_metric import TEDS, convert_markdown_table_to_html, convert_str_to_dict, convert_str_to_multi_dict, generate_combinations, dict_to_html, compute_f1_score, doc_parsing_evaluation, wrap_html_table
from lmms_eval.tasks.ocrbench_v2.page_ocr_metric import cal_per_metrics
from lmms_eval.tasks.ocrbench_v2.spotting_metric import extract_bounding_boxes_robust, spotting_evaluation


# Add the following functions to your existing utils.py file
OCRBench_v2_score = {
    "text_recognition_en": [],
    "text_detection_en": [],
    "text_spotting_en": [],
    "relationship_extraction_en": [],
    "element_parsing_en": [],
    "mathematical_calculation_en": [],
    "visual_text_understanding_en": [],
    "knowledge_reasoning_en": [],
    "text_recognition_cn": [],
    "relationship_extraction_cn": [],
    "element_parsing_cn": [],
    "visual_text_understanding_cn": [],
    "knowledge_reasoning_cn": []
}


teds = TEDS(n_jobs=32)


def ocrbench_v2_doc_to_visual(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    return [doc["image"].convert("RGB")]


def ocrbench_v2_doc_to_text(doc):
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["question"].strip()
    return f"{question}"


def is_nan_value(value):
    if value is None:
        return True
    if isinstance(value, str) and value.lower() == 'nan':
        return True
    try:
        import pandas as pd
        if pd.isna(value):
            return True
    except:
        pass
    return False


def get_value_or_zero(value):
    return 0.0 if value is None else value


def ocrbench_v2_process_results(doc, results):
    pred = results[0]
    question = doc["question"]
    gt_ans = doc["answers"]
    data_type = doc["type"]

    score = 0

    if data_type == "APP agent en" or data_type == "ASCII art classification en" or data_type == "math QA en" \
        or data_type == "reasoning VQA en" or data_type == "science QA en" \
        or data_type == "text recognition en" or data_type == "document classification en" \
        or data_type == "cognition VQA en" or data_type == "diagram QA en":
        
        if doc["eval"] == "multiple choice":
            if not isinstance(gt_ans, list):
                gt_ans = [gt_ans]
            assert len(gt_ans) == 1

            if not isinstance(pred, str):
                score = 0
            else:
                predict = ''.join(c for c in pred if c.isalpha())

                if predict == gt_ans[0]:
                    score = 1
                else:
                    score = 0
        elif doc["eval"] == "case sensitive":
            score = vqa_evaluation_case_sensitive(pred, gt_ans)

        else:
            score = vqa_evaluation(pred, gt_ans)
    
    elif data_type == "cognition VQA cn" or data_type == "reasoning VQA cn":
        
        if doc["eval"] == "multiple choice":
            assert len(gt_ans) == 1
            predict = ''.join(c for c in pred if c.isalpha())

            if predict == gt_ans[0]:
                score = 1
            else:
                score = 0
        elif doc["eval"] == "case sensitive":
            score = vqa_evaluation_case_sensitive(pred, gt_ans)
            
        else:
            score = cn_vqa_evaluation(pred, gt_ans)
    
    elif data_type == "handwritten answer extraction cn":
        if "简答" in question:
            ocr_metric = cal_per_metrics(pred, gt_ans[0])
            score = (
                get_value_or_zero(ocr_metric["bleu"]) + 
                get_value_or_zero(ocr_metric["meteor"]) + 
                get_value_or_zero(ocr_metric["f_measure"]) + 
                (1 - get_value_or_zero(ocr_metric["edit_dist"]))
            ) / 4
        else:
            assert len(gt_ans) == 1
            answer = gt_ans[0]
            chars = list(answer)
            if len(answer) > 1:

                answer_list = [
                    "".join(chars),
                    ".".join(chars),
                    ". ".join(chars),
                    ",".join(chars),
                    ", ".join(chars),
                    "、".join(chars),
                    ";".join(chars),
                    "; ".join(chars),
                    " ".join(chars),
                    "和".join(chars)
                ]
                max_score = 0
                for answer in answer_list:
                    if answer in pred:
                        temp_score = 1
                    else:
                        temp_score = 0
                    if temp_score > max_score:
                        max_score = temp_score
                score = max_score

            else:    
                if gt_ans[0] in pred:
                    score = 1
                else:
                    score = 0

    elif data_type == "formula recognition cn":
        if is_nan_value(pred):
            score = 0
        else:
            score = cn_math_expression_evaluation(pred, gt_ans)

    elif data_type == "text counting en":
        score = counting_evaluation(pred, gt_ans, doc["eval"])
    
    elif data_type == "formula recognition en":
        score = math_expression_evaluation(pred, gt_ans)
    
    elif data_type == "table parsing en":
        if type(gt_ans)==list and len(gt_ans) == 1:
            if not isinstance(pred, str):
                score = 0

            elif "html" in question.lower():
                no_find = False
                predict_table = pred.replace('\n','')
                if "<body" in predict_table:
                    predict_table = re.findall('<body.*', predict_table)[0]
                elif "<table" in predict_table:
                    predict_table = re.findall('<table.*', predict_table)[0]
                else:
                    no_find = True

                if no_find:
                    score = 0
                else:
                    pred_table_html = wrap_html_table(predict_table)
                    gold_table_html = wrap_html_table(gt_ans[0])
                    try:
                        score = teds.evaluate(pred_table_html, gold_table_html)
                    except:
                        score = 0

            elif "markdown" in question.lower():
                if not isinstance(pred, str):
                    
                    prediction = str(pred)
                    pred_table_html = convert_markdown_table_to_html(prediction)
                    gt_table_html = convert_markdown_table_to_html(gt_ans[0])
                    score = teds.evaluate(pred_table_html, gt_table_html)

                else:
                    pred_table_html = convert_markdown_table_to_html(pred)
                    gt_table_html = convert_markdown_table_to_html(gt_ans[0])
                    score = teds.evaluate(pred_table_html, gt_table_html)
        else:
            raise ValueError

    elif data_type == "table parsing cn":
        if not isinstance(pred, str):
            score = 0
        else:
            no_find = False
            predict_table = pred.replace('\n','')
            if "<body" in predict_table:
                predict_table = re.findall('<body.*', predict_table)[0]
            elif "<table" in predict_table:
                predict_table = re.findall('<table.*', predict_table)[0]
            else:
                no_find = True

            if no_find:
                score = 0
            else:
                pred_table_html = wrap_html_table(predict_table)
                gold_table_html = wrap_html_table(gt_ans[0])
                try:
                    score = teds.evaluate(pred_table_html, gold_table_html)
                except:
                    score = 0
                    print("error")

    elif data_type == "chart parsing en":
        answer = gt_ans[0]
        if pred:

            pred_chart_dict = convert_str_to_multi_dict(pred)
            if len(pred_chart_dict) == 0:
                score = 0
            else:
                pred_chart_html = dict_to_html(pred_chart_dict)
                if isinstance(answer, str):
                    answer = convert_str_to_multi_dict(pred)
                gt_chart_html = dict_to_html(answer)
                score = teds.evaluate(pred_chart_html, gt_chart_html)
        else:
            score = 0

    elif data_type == "document parsing en":
        assert type(gt_ans)==list and len(gt_ans) == 1
        score = doc_parsing_evaluation(pred, gt_ans[0])
    
    elif data_type == "document parsing cn":
        assert type(gt_ans)==list and len(gt_ans) == 1
        score = doc_parsing_evaluation(pred, gt_ans[0])

    elif data_type == "key information extraction en" or data_type == "key information mapping en":
        assert len(gt_ans) == 1
        answers = generate_combinations(gt_ans[0])
        
        if type(answers)==list and len(answers) == 1:
            if not isinstance(pred, str):
                score = 0
            else:
                pred_kie_dict = convert_str_to_dict(pred)
                score = compute_f1_score(pred_kie_dict, answers[0])
        else:
            max_score = 0
            for answer in answers:
                pred_kie_dict = convert_str_to_dict(pred)
                score = compute_f1_score(pred_kie_dict, answer)
                
                if score > max_score:
                    max_score = score
            score = max_score
    
    elif data_type == "key information extraction cn":
        assert len(gt_ans) == 1
        answers = ast.literal_eval(gt_ans[0])
        answers = {k: v if isinstance(v, list) else [v] for k, v in answers.items()}
        answers = generate_combinations(answers)
        if type(answers)==list and len(answers) == 1:
            if not isinstance(pred, str):
                score = 0
            else:
                pred_kie_dict = convert_str_to_dict(pred)
                score = compute_f1_score(pred_kie_dict, answers[0])
        else:
            max_score = 0
            for answer in answers:
                pred_kie_dict = convert_str_to_dict(pred)
                score = compute_f1_score(pred_kie_dict, answer)
                
                if score > max_score:
                    max_score = score
            score = max_score

    elif data_type == "VQA with position en":  
        if not isinstance(pred, str):
            score = 0
        else:
            pred_dict = convert_str_to_dict(pred)
            score = vqa_with_position_evaluation(pred_dict, doc)

    elif data_type == "text translation cn":
        if len(pred) == 0:
            score = 0
        else:
            ocr_metric = cal_per_metrics(pred, gt_ans[0])
            score = (ocr_metric["bleu"] + ocr_metric["meteor"] + ocr_metric["f_measure"] + (1 - ocr_metric["edit_dist"])) / 4 

    elif data_type == "fine-grained text recognition en":
        if not isinstance(pred, str):
            score = 0
        elif len(pred) == 0:
            score = 0
        else:
            ocr_metric = cal_per_metrics(pred, gt_ans[0])
            score = (
                get_value_or_zero(ocr_metric["bleu"]) + 
                get_value_or_zero(ocr_metric["meteor"]) + 
                get_value_or_zero(ocr_metric["f_measure"]) + 
                (1 - get_value_or_zero(ocr_metric["edit_dist"]))
            ) / 4
    elif data_type == "full-page OCR en":
        if not pred:
            score == 0
        else:
            ocr_metric = cal_per_metrics(pred, gt_ans[0])
            score = (
                get_value_or_zero(ocr_metric["bleu"]) + 
                get_value_or_zero(ocr_metric["meteor"]) + 
                get_value_or_zero(ocr_metric["f_measure"]) + 
                (1 - get_value_or_zero(ocr_metric["edit_dist"]))
            ) / 4

    elif data_type == "full-page OCR cn":
        if not isinstance(pred, str):
            score = 0
        else:
            if len(pred) == 0:
                score = 0
            else:
                ocr_metric = cal_per_metrics(pred, gt_ans[0])
                score = (ocr_metric["bleu"] + ocr_metric["meteor"] + ocr_metric["f_measure"] + (1 - ocr_metric["edit_dist"])) / 4 
    
    elif data_type == "text grounding en":
        if not isinstance(pred, str):
            score = 0
        else:
            predict_bbox = extract_coordinates(pred)
            if not predict_bbox:
                score = 0
            else:
                score = calculate_iou(predict_bbox, gt_ans)

    elif data_type == "text spotting en":
        if not isinstance(pred, str):
            score = 0
        else:
            predict_bbox = extract_bounding_boxes_robust(pred)
            if not predict_bbox:
                score = 0
            else:
                score = spotting_evaluation(predict_bbox, doc)
    
    return {
        "ocrbench_v2_accuracy": {"question_type": data_type, "score": score, "prediction": pred, "ground_truth": gt_ans},
    }


def calculate_average_score(categories):
    return sum(sum(OCRBench_v2_score[cat]) / len(OCRBench_v2_score[cat]) if len(OCRBench_v2_score[cat]) > 0 else 0 for cat in categories) / len(categories)


def ocrbench_v2_aggregate_accuracy(results, args):

    question_type_scores = {}

    for result in results:
        if "ignore" in result.keys() and result["ignore"] == "True":
            continue

        question_type = result["question_type"]
        score = result["score"]

        if question_type not in question_type_scores:
            question_type_scores[question_type] = []
        question_type_scores[question_type].append(score)

        if question_type in ["text recognition en", "fine-grained text recognition en", "full-page OCR en"]:
            OCRBench_v2_score["text_recognition_en"].append(score)

        elif question_type in ["text grounding en", "VQA with position en"]:
            OCRBench_v2_score["text_detection_en"].append(score)

        elif question_type == "text spotting en":
            OCRBench_v2_score["text_spotting_en"].append(score)

        elif question_type in ["key information extraction en", "key information mapping en"]:
            OCRBench_v2_score["relationship_extraction_en"].append(score)

        elif question_type in ["document parsing en", "chart parsing en", "table parsing en", "formula recognition en"]:
            OCRBench_v2_score["element_parsing_en"].append(score)

        elif question_type in ["math QA en", "text counting en"]:
            OCRBench_v2_score["mathematical_calculation_en"].append(score)

        elif question_type in ["document classification en", "cognition VQA en", "diagram QA en"]:
            OCRBench_v2_score["visual_text_understanding_en"].append(score)

        elif question_type in ["reasoning VQA en", "science QA en", "APP agent en", "ASCII art classification en"]:
            OCRBench_v2_score["knowledge_reasoning_en"].append(score)

        elif question_type == "full-page OCR cn":
            OCRBench_v2_score["text_recognition_cn"].append(score)

        elif question_type in ["key information extraction cn", "handwritten answer extraction cn"]:
            OCRBench_v2_score["relationship_extraction_cn"].append(score)

        elif question_type in ["document parsing cn", "table parsing cn", "formula recognition cn"]:
            OCRBench_v2_score["element_parsing_cn"].append(score)

        elif question_type == "cognition VQA cn":
            OCRBench_v2_score["visual_text_understanding_cn"].append(score)

        elif question_type in ["reasoning VQA cn", "text translation cn"]:
            OCRBench_v2_score["knowledge_reasoning_cn"].append(score)

        else:
            print("No such task!")
            raise TypeError

    english_tasks = [
        "text_recognition_en", "text_detection_en", "text_spotting_en",
        "relationship_extraction_en", "element_parsing_en",
        "mathematical_calculation_en", "visual_text_understanding_en",
        "knowledge_reasoning_en"
    ]

    chinese_tasks = [
        "text_recognition_cn", "relationship_extraction_cn",
        "element_parsing_cn", "visual_text_understanding_cn",
        "knowledge_reasoning_cn"
    ]

    OCRBench_v2_English_subset_score = calculate_average_score(english_tasks)
    OCRBench_v2_Chinese_subset_score = calculate_average_score(chinese_tasks)

    Final_score = (OCRBench_v2_English_subset_score + OCRBench_v2_Chinese_subset_score) / 2
    file_name = generate_submission_file("ocrbench_v2_results.txt", args, subpath="results")
    with open(file_name, "w") as f:
        print("######################### OCRBench v2 ##########################", file=f)
        print("################## All Question Types Scores ###################", file=f)
        for q_type, scores in sorted(question_type_scores.items()):
            avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0
            print(f"{q_type} (sample number: {len(scores)}): {avg_score:.2f}", file=f)
        print("######################### English Subsets ######################", file=f)
        for task in english_tasks:
            num_samples = len(OCRBench_v2_score[task])
            avg_score = sum(OCRBench_v2_score[task]) / num_samples if num_samples > 0 else 0
            print(f"{task.replace('_', ' ').title()} (Total {num_samples}): {avg_score:.2f}", file=f)
        print(f"Overall English Score: {OCRBench_v2_English_subset_score:.2f}", file=f)
        print("######################### Chinese Subsets ######################", file=f)
        for task in chinese_tasks:
            num_samples = len(OCRBench_v2_score[task])
            avg_score = sum(OCRBench_v2_score[task]) / num_samples if num_samples > 0 else 0
            print(f"{task.replace('_', ' ').title()} (Total {num_samples}): {avg_score:.2f}", file=f)
        print(f"Overall Chinese Score: {OCRBench_v2_Chinese_subset_score:.2f}", file=f)
        print("######################### Final Score ##########################", file=f)
        print(f"Final Score (Total 10000): {Final_score:.2f}", file=f)
    logger.info(f"OCRBench v2 results saved to {file_name}")
    
    return Final_score # return the final score as accuracy
