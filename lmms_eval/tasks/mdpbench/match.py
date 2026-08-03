"""Vendored matching helpers from the reference MDPBench repository."""

import pdb
import re
import sys
from copy import deepcopy

import Levenshtein
import numpy as np
from bs4 import BeautifulSoup
from scipy.optimize import linear_sum_assignment

from .data_preprocess import (
    clean_string,
    normalized_formula,
    textblock2unicode,
    textblock_with_norm_formula,
)


def get_pred_category_type(pred_idx, pred_items):
    if pred_items[pred_idx].get("fine_category_type"):
        pred_pred_category_type = pred_items[pred_idx]["fine_category_type"]
    else:
        pred_pred_category_type = pred_items[pred_idx]["category_type"]
    return pred_pred_category_type


def compute_edit_distance_matrix_new(gt_lines, matched_lines):
    try:
        distance_matrix = np.zeros((len(gt_lines), len(matched_lines)))
        for i, gt_line in enumerate(gt_lines):
            for j, matched_line in enumerate(matched_lines):
                if len(gt_line) == 0 and len(matched_line) == 0:
                    distance_matrix[i][j] = 0
                else:
                    distance_matrix[i][j] = Levenshtein.distance(gt_line, matched_line) / max(len(matched_line), len(gt_line))
        return distance_matrix
    except ZeroDivisionError:
        raise


## 混合匹配here  0403
def get_gt_pred_lines(gt_mix, pred_dataset_mix, line_type):

    norm_html_lines, gt_lines, pred_lines, norm_gt_lines, norm_pred_lines, gt_cat_list = [], [], [], [], [], []
    if line_type in ["html_table", "latex_table"]:
        for item in gt_mix:
            if item.get("fine_category_type"):
                gt_cat_list.append(item["fine_category_type"])
            else:
                gt_cat_list.append(item["category_type"])
            if item.get("content"):
                gt_lines.append(str(item["content"]))
                norm_html_lines.append(str(item["content"]))
            elif line_type == "text":
                gt_lines.append(str(item["text"]))
            elif line_type == "html_table":
                gt_lines.append(str(item["html"]))
            elif line_type == "formula":
                gt_lines.append(str(item["latex"]))
            elif line_type == "latex_table":
                gt_lines.append(str(item["latex"]))
                norm_html_lines.append(str(item["html"]))

        pred_lines = [str(item["content"]) for item in pred_dataset_mix]
        if line_type == "formula":
            norm_gt_lines = [normalized_formula(_) for _ in gt_lines]
            norm_pred_lines = [normalized_formula(_) for _ in pred_lines]
        elif line_type == "text":
            norm_gt_lines = [clean_string(textblock2unicode(_)) for _ in gt_lines]
            norm_pred_lines = [clean_string(textblock2unicode(_)) for _ in pred_lines]
        else:
            norm_gt_lines = gt_lines
            norm_pred_lines = pred_lines
        if line_type == "latex_table":
            gt_lines = norm_html_lines

    else:
        for item in pred_dataset_mix:
            # text
            if item["category_type"] == "text_all":
                pred_lines.append(str(item["content"]))
                norm_pred_lines.append(clean_string(textblock2unicode(str(item["content"]))))
            # formula
            elif item["category_type"] == "equation_isolated":
                pred_lines.append(str(item["content"]))
                norm_pred_lines.append(normalized_formula(str(item["content"])))
            # table
            else:
                pred_lines.append(str(item["content"]))
                norm_pred_lines.append(str(item["content"]))

        for item in gt_mix:
            if item.get("content"):
                gt_lines.append(str(item["content"]))
                if item["category_type"] == "text_all":
                    norm_gt_lines.append(clean_string(textblock2unicode(str(item["content"]))))
                else:
                    norm_gt_lines.append(item["content"])

                norm_html_lines.append(str(item["content"]))

                if item.get("fine_category_type"):
                    gt_cat_list.append(item["fine_category_type"])
                else:
                    gt_cat_list.append(item["category_type"])
            # text
            elif item["category_type"] in [
                "text_block",
                "title",
                "code_txt",
                "code_txt_caption",
                "reference",
                "equation_caption",
                "figure_caption",
                "figure_footnote",
                "table_caption",
                "table_footnote",
                "code_algorithm",
                "code_algorithm_caption",
                "header",
                "footer",
                "page_footnote",
                "page_number",
            ]:
                gt_lines.append(str(item["text"]))
                norm_gt_lines.append(clean_string(textblock2unicode(str(item["text"]))))

                if item.get("fine_category_type"):
                    gt_cat_list.append(item["fine_category_type"])
                else:
                    gt_cat_list.append(item["category_type"])

            # formula
            elif item["category_type"] == "equation_isolated":
                gt_lines.append(str(item["latex"]))
                norm_gt_lines.append(normalized_formula(str(item["latex"])))

                if item.get("fine_category_type"):
                    gt_cat_list.append(item["fine_category_type"])
                else:
                    gt_cat_list.append(item["category_type"])
            # table
            # elif item['category_type'] == 'table':
            #     gt_lines.append(str(item['html']))
            #     norm_gt_lines.append(str(item['html']))

            #     if item.get('fine_category_type'):
            #         gt_cat_list.append(item['fine_category_type'])
            #     else:
            #         gt_cat_list.append(item['category_type'])

    filtered_lists = [(a, b, c) for a, b, c in zip(gt_lines, norm_gt_lines, gt_cat_list) if a and b]

    # decompress to three lists
    if filtered_lists:
        gt_lines_c, norm_gt_lines_c, gt_cat_list_c = zip(*filtered_lists)

        # convert to lists
        gt_lines_c = list(gt_lines_c)
        norm_gt_lines_c = list(norm_gt_lines_c)
        gt_cat_list_c = list(gt_cat_list_c)
    else:
        gt_lines_c = []
        norm_gt_lines_c = []
        gt_cat_list_c = []

    # pred's empty values
    filtered_lists = [(a, b) for a, b in zip(pred_lines, norm_pred_lines) if a and b]

    # decompress to two lists
    if filtered_lists:
        pred_lines_c, norm_pred_lines_c = zip(*filtered_lists)

        # convert to lists
        pred_lines_c = list(pred_lines_c)
        norm_pred_lines_c = list(norm_pred_lines_c)
    else:
        pred_lines_c = []
        norm_pred_lines_c = []

    return gt_lines_c, norm_gt_lines_c, gt_cat_list_c, pred_lines_c, norm_pred_lines_c, gt_mix, pred_dataset_mix


def match_gt2pred_simple(gt_items, pred_items, line_type, img_name):

    gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines, gt_items, pred_items = get_gt_pred_lines(gt_items, pred_items, line_type)
    match_list = []

    if not norm_gt_lines:  # not matched pred should be concatenate
        pred_idx_list = range(len(norm_pred_lines))
        match_list.append(
            {
                "gt_idx": [""],
                "gt": "",
                "pred_idx": pred_idx_list,
                "pred": "".join(pred_lines[_] for _ in pred_idx_list),
                "gt_position": [""],
                "pred_position": pred_items[pred_idx_list[0]]["position"][0],  # get the first pred's position
                "norm_gt": "",
                "norm_pred": "".join(norm_pred_lines[_] for _ in pred_idx_list),
                "gt_category_type": "",
                "pred_category_type": get_pred_category_type(pred_idx_list[0], pred_items),  # get the first pred's category
                "gt_attribute": [{}],
                "edit": 1,
                "img_id": img_name,
            }
        )
        return match_list, None
    elif not norm_pred_lines:  # not matched gt should be separated
        for gt_idx in range(len(norm_gt_lines)):
            match_list.append(
                {
                    "gt_idx": [gt_idx],
                    "gt": gt_lines[gt_idx],
                    "pred_idx": [""],
                    "pred": "",
                    "gt_position": [gt_items[gt_idx].get("order") if gt_items[gt_idx].get("order") else gt_items[gt_idx].get("position", [""])[0]],
                    "pred_position": "",
                    "norm_gt": norm_gt_lines[gt_idx],
                    "norm_pred": "",
                    "gt_category_type": gt_cat_list[gt_idx],
                    "pred_category_type": "",
                    "gt_attribute": [gt_items[gt_idx].get("attribute", {})],
                    "edit": 1,
                    "img_id": img_name,
                }
            )
        return match_list, None

    cost_matrix = compute_edit_distance_matrix_new(norm_gt_lines, norm_pred_lines)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    for gt_idx in range(len(norm_gt_lines)):
        if gt_idx in row_ind:
            row_i = list(row_ind).index(gt_idx)
            pred_idx = int(col_ind[row_i])
            pred_line = pred_lines[pred_idx]
            norm_pred_line = norm_pred_lines[pred_idx]
            edit = cost_matrix[gt_idx][pred_idx]
        else:
            pred_idx = ""
            pred_line = ""
            norm_pred_line = ""
            edit = 1

        match_list.append(
            {
                "gt_idx": [gt_idx],
                "gt": gt_lines[gt_idx],
                "norm_gt": norm_gt_lines[gt_idx],
                "gt_category_type": gt_cat_list[gt_idx],
                "gt_position": [gt_items[gt_idx].get("order") if gt_items[gt_idx].get("order") else gt_items[gt_idx].get("position", [""])[0]],
                "gt_attribute": [gt_items[gt_idx].get("attribute", {})],
                "pred_idx": [pred_idx],
                "pred": pred_line,
                "norm_pred": norm_pred_line,
                "pred_category_type": get_pred_category_type(pred_idx, pred_items) if pred_idx else "",
                "pred_position": pred_items[pred_idx]["position"][0] if pred_idx else "",
                "edit": edit,
                "img_id": img_name,
            }
        )

    pred_idx_list = [pred_idx for pred_idx in range(len(norm_pred_lines)) if pred_idx not in col_ind]  # get not matched preds
    if pred_idx_list:
        if line_type in ["html_table", "latex_table"]:
            unmatch_table_pred = []
            for i in pred_idx_list:
                original_item = pred_items[i]
                soup = BeautifulSoup(original_item.get("content"), "html.parser")
                text_block = [re.sub(r"\$\\cdot\$", "", item.string).strip() for item in soup.findAll("td") if item.string]
                for concatenate_text in text_block:
                    new_item = deepcopy(original_item)
                    new_item["content"] = concatenate_text
                    new_item["category_type"] = "text_all"
                    unmatch_table_pred.append(new_item)
            return match_list, unmatch_table_pred

        else:
            match_list.append(
                {
                    "gt_idx": [""],
                    "gt": "",
                    "pred_idx": pred_idx_list,
                    "pred": "".join(pred_lines[_] for _ in pred_idx_list),
                    "gt_position": [""],
                    "pred_position": pred_items[pred_idx_list[0]]["position"][0],  # get the first pred's position
                    "norm_gt": "",
                    "norm_pred": "".join(norm_pred_lines[_] for _ in pred_idx_list),
                    "gt_category_type": "",
                    "pred_category_type": get_pred_category_type(pred_idx_list[0], pred_items),  # get the first pred's category
                    "gt_attribute": [{}],
                    "edit": 1,
                    "img_id": img_name,
                }
            )
    return match_list, None


def match_gt2pred_no_split(gt_items, pred_items, line_type, img_name):
    # directly concatenate gt and pred by position
    gt_lines, norm_gt_lines, gt_cat_list, pred_lines, norm_pred_lines = get_gt_pred_lines(gt_items, pred_items)
    gt_line_with_position = []
    for gt_line, norm_gt_line, gt_item in zip(gt_lines, norm_gt_lines, gt_items):
        gt_position = gt_item["order"] if gt_item.get("order") else gt_item.get("position", [""])[0]
        if gt_position:
            gt_line_with_position.append((gt_position, gt_line, norm_gt_line))
    sorted_gt_lines = sorted(gt_line_with_position, key=lambda x: x[0])
    gt = "\n\n".join([_[1] for _ in sorted_gt_lines])
    norm_gt = "\n\n".join([_[2] for _ in sorted_gt_lines])
    pred_line_with_position = [(pred_item["position"], pred_line, pred_norm_line) for pred_line, pred_norm_line, pred_item in zip(pred_lines, norm_pred_lines, pred_items)]
    sorted_pred_lines = sorted(pred_line_with_position, key=lambda x: x[0])
    pred = "\n\n".join([_[1] for _ in sorted_pred_lines])
    norm_pred = "\n\n".join([_[2] for _ in sorted_pred_lines])
    # edit = Levenshtein.distance(norm_gt, norm_pred)/max(len(norm_gt), len(norm_pred))
    if norm_gt or norm_pred:
        return [
            {
                "gt_idx": [0],
                "gt": gt,
                "norm_gt": norm_gt,
                "gt_category_type": "text_merge",
                "gt_position": [""],
                "gt_attribute": [{}],
                "pred_idx": [0],
                "pred": pred,
                "norm_pred": norm_pred,
                "pred_category_type": "text_merge",
                "pred_position": "",
                # 'edit': edit,
                "img_id": img_name,
            }
        ]
    else:
        return []
