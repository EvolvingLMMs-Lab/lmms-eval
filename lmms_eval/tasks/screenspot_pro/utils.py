import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

eval_logger = logging.getLogger("lmms-eval")
_NUM_TOKEN = r"-?\d+(?:\.\d+)?%?"


def _get_image(doc: Dict[str, Any]):
    image = doc.get("image")
    if image is None:
        raise ValueError("ScreenSpot-Pro sample does not contain 'image'")
    return image.convert("RGB")


def screenspot_pro_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    return [_get_image(doc)]


def screenspot_pro_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    instruction = doc.get("instruction", "")
    return "Identify the UI element for the instruction and output exactly one click point as [x, y] in normalized coordinates within [0, 1]. " "Do not output a bounding box.\n" f"Instruction: {instruction}"


def screenspot_pro_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs=None):
    text = screenspot_pro_doc_to_text(doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)
    return [{"role": "user", "content": [{"type": "image", "url": _get_image(doc)}, {"type": "text", "text": text}]}]


def _token_to_float(token: str) -> float:
    token = token.strip()
    if token.endswith("%"):
        return float(token[:-1]) / 100.0
    return float(token)


def _parse_point(prediction: str) -> Optional[Tuple[float, float]]:
    bbox_tag_match = re.search(r"<\s*bbox[^>]*>(.*?)<\s*/\s*bbox\s*>", prediction, flags=re.IGNORECASE | re.DOTALL)
    if bbox_tag_match:
        bbox_tokens = re.findall(_NUM_TOKEN, bbox_tag_match.group(1))
        if len(bbox_tokens) >= 4:
            x1, y1, x2, y2 = [_token_to_float(v) for v in bbox_tokens[:4]]
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    point_match = re.search(r"[\[\(]\s*(-?\d+(?:\.\d+)?%?)\s*(?:,|\s)\s*(-?\d+(?:\.\d+)?%?)\s*[\]\)]", prediction)
    if point_match:
        return _token_to_float(point_match.group(1)), _token_to_float(point_match.group(2))

    xy_match = re.search(r"['\"]?x['\"]?\s*[:=]\s*(-?\d+(?:\.\d+)?%?).*?['\"]?y['\"]?\s*[:=]\s*(-?\d+(?:\.\d+)?%?)", prediction, flags=re.IGNORECASE | re.DOTALL)
    if xy_match:
        return _token_to_float(xy_match.group(1)), _token_to_float(xy_match.group(2))

    numbers = re.findall(_NUM_TOKEN, prediction)
    if len(numbers) >= 4:
        lower_pred = prediction.lower()
        if any(keyword in lower_pred for keyword in ["bbox", "box", "rect", "rectangle"]):
            x1, y1, x2, y2 = [_token_to_float(v) for v in numbers[:4]]
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if len(numbers) >= 2:
        return _token_to_float(numbers[0]), _token_to_float(numbers[1])
    return None


def _normalize_xyxy(bbox_xyxy: List[float], width: int, height: int) -> List[float]:
    if len(bbox_xyxy) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    if bbox_xyxy[2] <= 1 and bbox_xyxy[3] <= 1:
        return bbox_xyxy
    return [bbox_xyxy[0] / width, bbox_xyxy[1] / height, bbox_xyxy[2] / width, bbox_xyxy[3] / height]


def _parse_bbox(value: Any) -> List[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
        except (json.JSONDecodeError, TypeError, ValueError):
            numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
            if len(numbers) >= 4:
                return [float(v) for v in numbers[:4]]
    return [0.0, 0.0, 0.0, 0.0]


def _parse_img_size(value: Any) -> Optional[Tuple[int, int]]:
    if isinstance(value, list) and len(value) == 2:
        w, h = int(value[0]), int(value[1])
        if w > 0 and h > 0:
            return w, h
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list) and len(parsed) == 2:
                w, h = int(parsed[0]), int(parsed[1])
                if w > 0 and h > 0:
                    return w, h
        except (json.JSONDecodeError, TypeError, ValueError):
            numbers = re.findall(r"\d+", value)
            if len(numbers) >= 2:
                w, h = int(numbers[0]), int(numbers[1])
                if w > 0 and h > 0:
                    return w, h
    return None


def _point_in_box(point_xy: Tuple[float, float], box_xyxy: List[float]) -> bool:
    return box_xyxy[0] <= point_xy[0] <= box_xyxy[2] and box_xyxy[1] <= point_xy[1] <= box_xyxy[3]


def screenspot_pro_process_results(doc, result):
    prediction = result[0] if result else ""
    point = _parse_point(prediction)

    width, height = int(doc.get("image_width", 1)), int(doc.get("image_height", 1))
    img_size = _parse_img_size(doc.get("img_size"))
    if img_size is not None:
        width, height = img_size

    gt_bbox = _normalize_xyxy(_parse_bbox(doc["bbox"]), width, height)
    if point is not None and (point[0] > 1 or point[1] > 1):
        point = (point[0] / width, point[1] / height)
    is_correct = bool(point and _point_in_box(point, gt_bbox))
    ui_type = str(doc.get("ui_type", doc.get("data_type", ""))).lower()

    record = {
        "correct": is_correct,
        "ui_type": ui_type,
    }
    return {
        "screenspot_pro_action_acc": record,
        "screenspot_pro_text_acc": record,
        "screenspot_pro_icon_acc": record,
    }


def _aggregate(results: List[Dict[str, Any]], target_type: Optional[str] = None) -> float:
    filtered = [r for r in results if target_type is None or r.get("ui_type") == target_type]
    if not filtered:
        return 0.0
    score = sum(1 for r in filtered if r.get("correct")) / len(filtered)
    eval_logger.info("ScreenSpot-Pro %s accuracy: %.4f", target_type or "overall", score)
    return score


def screenspot_pro_action_acc(results):
    return _aggregate(results, target_type=None)


def screenspot_pro_text_acc(results):
    return _aggregate(results, target_type="text")


def screenspot_pro_icon_acc(results):
    return _aggregate(results, target_type="icon")
