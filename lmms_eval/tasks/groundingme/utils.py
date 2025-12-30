import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

eval_logger = logging.getLogger("lmms-eval")

REC_METRICS = [
    "IoU",
    "ACC@0.5",
    "ACC@0.75",
    "ACC@0.9",
    "Center_ACC",
    "MACC",
    "Discriminative_ACC@0.5",
    "Discriminative_ACC@0.75",
    "Discriminative_ACC@0.9",
    "Discriminative_MACC",
    "Spatial_ACC@0.5",
    "Spatial_ACC@0.75",
    "Spatial_ACC@0.9",
    "Spatial_MACC",
    "Limited_ACC@0.5",
    "Limited_ACC@0.75",
    "Limited_ACC@0.9",
    "Limited_MACC",
    "Rejection_ACC",
    "D_Appearance_ACC@0.5",
    "D_Appearance_ACC@0.75",
    "D_Appearance_ACC@0.9",
    "D_Appearance_MACC",
    "D_Component_ACC@0.5",
    "D_Component_ACC@0.75",
    "D_Component_ACC@0.9",
    "D_Component_MACC",
    "D_Text_ACC@0.5",
    "D_Text_ACC@0.75",
    "D_Text_ACC@0.9",
    "D_Text_MACC",
    "D_State_ACC@0.5",
    "D_State_ACC@0.75",
    "D_State_ACC@0.9",
    "D_State_MACC",
    "Relationship_ACC@0.5",
    "Relationship_ACC@0.75",
    "Relationship_ACC@0.9",
    "Relationship_MACC",
    "Counting_ACC@0.5",
    "Counting_ACC@0.75",
    "Counting_ACC@0.9",
    "Counting_MACC",
    "Occlusion_ACC@0.5",
    "Occlusion_ACC@0.75",
    "Occlusion_ACC@0.9",
    "Occlusion_MACC",
    "Small_ACC@0.5",
    "Small_ACC@0.75",
    "Small_ACC@0.9",
    "Small_MACC",
    "R_Appearance_ACC",
    "R_Component_ACC",
    "R_Text_ACC",
    "R_State_ACC",
]


def smart_resize_mimo(height: int, width: int, factor: int = 28, min_pixels: int = 28 * 28 * 8, max_pixels: int = 28 * 28 * 4096):
    """Resize image for MIMO models with factor-divisible dimensions and pixel constraints."""
    if max(height, width) < 10:
        raise ValueError(f"At least one dimension must be larger than 10 pixels, got height:{height}, width:{width}")
    elif min(height, width) < factor:
        if height < width:
            height = factor
            width = int(width * (factor / height))
        else:
            width = factor
            height = int(height * (factor / width))
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def smart_resize_qwen(height: int, width: int, factor: int = 28, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28):
    """Resize image for Qwen models with factor-divisible dimensions and pixel constraints."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than {200}, got {max(height, width) / min(height, width)}")
    h_bar = max(factor, int(round(height / factor) * factor))
    w_bar = max(factor, int(round(width / factor) * factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, int(math.floor(height / beta / factor) * factor))
        w_bar = max(factor, int(math.floor(width / beta / factor) * factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = max(factor, int(math.ceil(height * beta / factor) * factor))
        w_bar = max(factor, int(math.ceil(width * beta / factor) * factor))
    return h_bar, w_bar


def _normalize_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    """Normalize bbox from either [0, 1] or [0, 999] range to pixel coordinates."""
    if all(coord <= 1 for coord in bbox):
        return [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
    else:
        return [bbox[0] / 999 * width, bbox[1] / 999 * height, bbox[2] / 999 * width, bbox[3] / 999 * height]


def convert_bbox_from_mimo(bbox: List[float], width: int, height: int) -> List[float]:
    """Convert bbox coordinates from MIMO resized space to original image space."""
    mimo_width, mimo_height = smart_resize_mimo(height, width)
    return [bbox[0] / mimo_width * width, bbox[1] / mimo_height * height, bbox[2] / mimo_width * width, bbox[3] / mimo_height * height]


def convert_bbox_from_qwen(bbox: List[float], width: int, height: int) -> List[float]:
    """Convert bbox coordinates from Qwen resized space to original image space."""
    qwen_width, qwen_height = smart_resize_qwen(height, width)
    return [bbox[0] / qwen_width * width, bbox[1] / qwen_height * height, bbox[2] / qwen_width * width, bbox[3] / qwen_height * height]


def groundingme_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    """Convert document to visual input for model evaluation."""
    return [doc["image"].convert("RGB")]


PROMPT = "All spatial relationships are defined from the viewer's perspective, where 'front' means closer to the viewer and 'back' means farther from the viewer. Please provide the bounding box coordinate of the object the following statement describes:\n{description}\nEnsure that all details mentioned about the object are accurate. Provide at most one bounding box. If a matching object is found, provide its bounding box as a JSON in the format {{\"bbox_2d\": [x1, y1, x2, y2]}}. If no matching object is found, output {{\"bbox_2d\": null}}."


def groundingme_doc_to_text(doc: Dict[str, Any]) -> str:
    """Convert document to text prompt for model evaluation."""
    assert isinstance(doc["description"], str), "Answer must be a string"
    return PROMPT.format(description=doc["description"])


def parse_bbox(input_str: str) -> List[float]:
    """Extract bounding box from JSON format: {"bbox_2d": [x1, y1, x2, y2]} or {"bbox_2d": null}."""
    try:
        match = re.search(r'\{.*"bbox_2d".*\}', input_str, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            bbox = data["bbox_2d"]

            if bbox is None:
                return [0, 0, 0, 0]

            if isinstance(bbox, list) and len(bbox) == 4:
                return [float(coord) for coord in bbox]
            else:
                raise ValueError(f"Invalid bbox format: {bbox}")
        else:
            raise ValueError(f"No bbox found in input string: {input_str}")

    except (ValueError, AttributeError, IndexError):
        pattern1 = r"(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)"
        matches = re.findall(pattern1, input_str)
        if matches:
            last_match = matches[-1]
            return [float(coord) for coord in last_match]
        else:
            return [0, 0, 0, 0]


def groundingme_process_result(doc, result):
    """Process prediction result and compute evaluation metrics."""
    pred_raw = parse_bbox(result[0] if len(result) > 0 else "")
    bbox = doc["bbox"] if doc["subtask_l1"] != "Rejection" else [0, 0, 0, 0]
    height, width = doc["height"], doc["width"]

    # Try different coordinate format interpretations and select the best one
    pred_candidates = [
        pred_raw,  # Original format
        _normalize_bbox(pred_raw, width, height),  # Normalized or 0-999 range
        convert_bbox_from_mimo(pred_raw, width, height),  # MIMO format
        convert_bbox_from_qwen(pred_raw, width, height),  # Qwen format
    ]

    ious = [compute_iou(bbox, pred) for pred in pred_candidates]
    best_idx = ious.index(max(ious))
    pred_best, iou = pred_candidates[best_idx], ious[best_idx]

    center_acc = compute_center_accuracy(bbox, pred_best)
    thresholds = [round(0.5 + x * 0.05, 2) for x in range(10)]
    accs = [compute_accuracy(iou, t) for t in thresholds]

    data_dict = {
        "subtask_l1": doc["subtask_l1"],
        "subtask_l2": doc["subtask_l2"],
        "description": doc["description"],
        "pred": pred_best,
        "ann_id": doc["id"],
        "bbox": bbox,
        "iou": iou,
        "center_acc": center_acc,
        "acc_5": accs[0],
        "acc_75": accs[5],
        "acc_9": accs[8],
        "macc": sum(accs) / len(accs),
    }

    return {f"groundingme_{metric}": data_dict for metric in REC_METRICS}


def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) of two bounding boxes [x_min, y_min, x_max, y_max]."""
    if box1 == [0, 0, 0, 0]:
        if box2 == [0, 0, 0, 0]:
            return 1
        else:
            return 0

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area

    return iou


def compute_accuracy(iou, threshold=0.5):
    """Check if IoU meets the specified threshold."""
    return iou >= threshold


def compute_center_accuracy(box1, box2):
    """Check if the center point of box2 is within box1."""
    center_x = (box2[0] + box2[2]) / 2
    center_y = (box2[1] + box2[3]) / 2
    return box1[0] <= center_x <= box1[2] and box1[1] <= center_y <= box1[3]


def groundingme_aggregation_result(results, metric):
    """Aggregate evaluation results by specified metric and subtask categories."""

    is_l1_metric, is_l2_metric, l1_name, l2_name = 0, 0, 0, 0
    if "Discriminative" in metric or "Spatial" in metric or "Limited" in metric or "Rejection" in metric:
        is_l1_metric = 1
        l1_name = metric.split("_")[0]
    elif "D_Appearance" in metric or "D_Component" in metric or "D_Text" in metric or "D_State" in metric:
        is_l2_metric = 1
        l1_name = "Discriminative"
        l2_name = metric.split("_")[1]
    elif "R_Appearance" in metric or "R_Component" in metric or "R_Text" in metric or "R_State" in metric:
        is_l2_metric = 1
        l1_name = "Rejection"
        l2_name = metric.split("_")[1]
    elif "Relationship" in metric or "Counting" in metric:
        is_l2_metric = 1
        l1_name = "Spatial"
        l2_name = metric.split("_")[0]
    elif "Occlusion" in metric or "Small" in metric:
        is_l2_metric = 1
        l1_name = "Limited"
        l2_name = metric.split("_")[0]

    metric_type = 0
    if "ACC@0.5" in metric:
        metric_type = "ACC@0.5"
    elif "ACC@0.75" in metric:
        metric_type = "ACC@0.75"
    elif "ACC@0.9" in metric:
        metric_type = "ACC@0.9"
    elif "MACC" in metric:
        metric_type = "MACC"
    else:
        metric_type = metric

    results_dict = {metric: []}
    for result in results:
        if metric_type == "ACC@0.5":
            score = result["acc_5"]
        elif metric_type == "ACC@0.75":
            score = result["acc_75"]
        elif metric_type == "ACC@0.9":
            score = result["acc_9"]
        elif metric_type == "MACC":
            score = result["macc"]
        elif metric_type == "Center_ACC":
            score = result["center_acc"]
        elif metric_type == "IoU":
            score = result["iou"]

        if is_l1_metric == 1:
            if result["subtask_l1"] == l1_name:
                results_dict[metric].append(score)
        elif is_l2_metric == 1:
            if result["subtask_l1"] == l1_name and result["subtask_l2"] == l2_name:
                results_dict[metric].append(score)
        else:
            results_dict[metric].append(score)

    results_dict[metric] = sum(results_dict[metric]) / len(results_dict[metric])
    eval_logger.info(f"Aggregated {metric} score: {results_dict[metric]}")
    return results_dict[metric]


def groundingme_iou(results):
    return groundingme_aggregation_result(results, "IoU")


def groundingme_acc05(results):
    return groundingme_aggregation_result(results, "ACC@0.5")


def groundingme_acc075(results):
    return groundingme_aggregation_result(results, "ACC@0.75")


def groundingme_acc09(results):
    return groundingme_aggregation_result(results, "ACC@0.9")


def groundingme_center_acc(results):
    return groundingme_aggregation_result(results, "Center_ACC")


def groundingme_macc(results):
    return groundingme_aggregation_result(results, "MACC")


def groundingme_discriminative_acc05(results):
    return groundingme_aggregation_result(results, "Discriminative_ACC@0.5")


def groundingme_discriminative_acc075(results):
    return groundingme_aggregation_result(results, "Discriminative_ACC@0.75")


def groundingme_discriminative_acc09(results):
    return groundingme_aggregation_result(results, "Discriminative_ACC@0.9")


def groundingme_discriminative_macc(results):
    return groundingme_aggregation_result(results, "Discriminative_MACC")


def groundingme_spatial_acc05(results):
    return groundingme_aggregation_result(results, "Spatial_ACC@0.5")


def groundingme_spatial_acc075(results):
    return groundingme_aggregation_result(results, "Spatial_ACC@0.75")


def groundingme_spatial_acc09(results):
    return groundingme_aggregation_result(results, "Spatial_ACC@0.9")


def groundingme_spatial_macc(results):
    return groundingme_aggregation_result(results, "Spatial_MACC")


def groundingme_limited_acc05(results):
    return groundingme_aggregation_result(results, "Limited_ACC@0.5")


def groundingme_limited_acc075(results):
    return groundingme_aggregation_result(results, "Limited_ACC@0.75")


def groundingme_limited_acc09(results):
    return groundingme_aggregation_result(results, "Limited_ACC@0.9")


def groundingme_limited_macc(results):
    return groundingme_aggregation_result(results, "Limited_MACC")


def groundingme_rejection_acc(results):
    return groundingme_aggregation_result(results, "Rejection_ACC@0.5")


def groundingme_d_appearance_acc05(results):
    return groundingme_aggregation_result(results, "D_Appearance_ACC@0.5")


def groundingme_d_appearance_acc075(results):
    return groundingme_aggregation_result(results, "D_Appearance_ACC@0.75")


def groundingme_d_appearance_acc09(results):
    return groundingme_aggregation_result(results, "D_Appearance_ACC@0.9")


def groundingme_d_appearance_macc(results):
    return groundingme_aggregation_result(results, "D_Appearance_MACC")


def groundingme_d_component_acc05(results):
    return groundingme_aggregation_result(results, "D_Component_ACC@0.5")


def groundingme_d_component_acc075(results):
    return groundingme_aggregation_result(results, "D_Component_ACC@0.75")


def groundingme_d_component_acc09(results):
    return groundingme_aggregation_result(results, "D_Component_ACC@0.9")


def groundingme_d_component_macc(results):
    return groundingme_aggregation_result(results, "D_Component_MACC")


def groundingme_d_text_acc05(results):
    return groundingme_aggregation_result(results, "D_Text_ACC@0.5")


def groundingme_d_text_acc075(results):
    return groundingme_aggregation_result(results, "D_Text_ACC@0.75")


def groundingme_d_text_acc09(results):
    return groundingme_aggregation_result(results, "D_Text_ACC@0.9")


def groundingme_d_text_macc(results):
    return groundingme_aggregation_result(results, "D_Text_MACC")


def groundingme_d_state_acc05(results):
    return groundingme_aggregation_result(results, "D_State_ACC@0.5")


def groundingme_d_state_acc075(results):
    return groundingme_aggregation_result(results, "D_State_ACC@0.75")


def groundingme_d_state_acc09(results):
    return groundingme_aggregation_result(results, "D_State_ACC@0.9")


def groundingme_d_state_macc(results):
    return groundingme_aggregation_result(results, "D_State_MACC")


def groundingme_relationship_acc05(results):
    return groundingme_aggregation_result(results, "Relationship_ACC@0.5")


def groundingme_relationship_acc075(results):
    return groundingme_aggregation_result(results, "Relationship_ACC@0.75")


def groundingme_relationship_acc09(results):
    return groundingme_aggregation_result(results, "Relationship_ACC@0.9")


def groundingme_relationship_macc(results):
    return groundingme_aggregation_result(results, "Relationship_MACC")


def groundingme_counting_acc05(results):
    return groundingme_aggregation_result(results, "Counting_ACC@0.5")


def groundingme_counting_acc075(results):
    return groundingme_aggregation_result(results, "Counting_ACC@0.75")


def groundingme_counting_acc09(results):
    return groundingme_aggregation_result(results, "Counting_ACC@0.9")


def groundingme_counting_macc(results):
    return groundingme_aggregation_result(results, "Counting_MACC")


def groundingme_occlusion_acc05(results):
    return groundingme_aggregation_result(results, "Occlusion_ACC@0.5")


def groundingme_occlusion_acc075(results):
    return groundingme_aggregation_result(results, "Occlusion_ACC@0.75")


def groundingme_occlusion_acc09(results):
    return groundingme_aggregation_result(results, "Occlusion_ACC@0.9")


def groundingme_occlusion_macc(results):
    return groundingme_aggregation_result(results, "Occlusion_MACC")


def groundingme_small_acc05(results):
    return groundingme_aggregation_result(results, "Small_ACC@0.5")


def groundingme_small_acc075(results):
    return groundingme_aggregation_result(results, "Small_ACC@0.75")


def groundingme_small_acc09(results):
    return groundingme_aggregation_result(results, "Small_ACC@0.9")


def groundingme_small_macc(results):
    return groundingme_aggregation_result(results, "Small_MACC")


def groundingme_r_appearance_acc(results):
    return groundingme_aggregation_result(results, "R_Appearance_ACC@0.5")


def groundingme_r_component_acc(results):
    return groundingme_aggregation_result(results, "R_Component_ACC@0.5")


def groundingme_r_text_acc(results):
    return groundingme_aggregation_result(results, "R_Text_ACC@0.5")


def groundingme_r_state_acc(results):
    return groundingme_aggregation_result(results, "R_State_ACC@0.5")
