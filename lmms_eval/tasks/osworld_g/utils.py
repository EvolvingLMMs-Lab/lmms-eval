import logging
import re
from typing import Any, Dict, List, Optional, Tuple

eval_logger = logging.getLogger("lmms-eval")
_NUM_TOKEN = r"-?\d+(?:\.\d+)?%?"


def _get_image(doc: Dict[str, Any]):
    image = doc.get("image")
    if image is None:
        raise ValueError("OSWorld-G sample does not contain 'image'")
    return image.convert("RGB")


def osworld_g_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    return [_get_image(doc)]


def osworld_g_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    instruction = doc.get("instruction", "")
    return (
        "Identify the UI target for the instruction and output exactly one click point as [x, y]. "
        "You may use either normalized coordinates in [0, 1] or absolute pixel coordinates. "
        "If the target does not exist, output [-1, -1].\n"
        f"Instruction: {instruction}"
    )


def osworld_g_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs=None):
    text = osworld_g_doc_to_text(doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)
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


def _to_absolute_point(point_xy: Tuple[float, float], width: int, height: int) -> Tuple[float, float]:
    x, y = point_xy
    if 0 <= x <= 1 and 0 <= y <= 1:
        return x * width, y * height
    return x, y


def _point_in_bbox(point_xy: Tuple[float, float], bbox_xywh: List[float]) -> bool:
    x, y = point_xy
    left, top, box_w, box_h = bbox_xywh
    right = left + box_w
    bottom = top + box_h
    return left <= x <= right and top <= y <= bottom


def _point_in_polygon(point_xy: Tuple[float, float], polygon: List[float]) -> bool:
    x, y = point_xy
    if len(polygon) == 4:
        left, top, right, bottom = polygon
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top
        return left <= x <= right and top <= y <= bottom

    if len(polygon) % 2 != 0:
        return False

    n = len(polygon) // 2
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i * 2], polygon[i * 2 + 1]
        xj, yj = polygon[j * 2], polygon[j * 2 + 1]
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def osworld_g_process_results(doc, result):
    prediction = result[0] if result else ""
    parsed = _parse_point(prediction)

    image_size = doc.get("image_size")
    if isinstance(image_size, list) and len(image_size) == 2 and image_size[0] > 0 and image_size[1] > 0:
        width, height = image_size[0], image_size[1]
    else:
        image = _get_image(doc)
        width, height = image.size

    box_type = str(doc.get("box_type", "")).lower()
    is_correct = False
    if parsed is not None:
        point_abs = _to_absolute_point(parsed, width, height)
        if box_type == "bbox":
            is_correct = _point_in_bbox(point_abs, doc.get("box_coordinates", [0, 0, 0, 0]))
        elif box_type == "polygon":
            is_correct = _point_in_polygon(point_abs, doc.get("box_coordinates", []))
        elif box_type == "refusal":
            is_correct = point_abs[0] < 0 and point_abs[1] < 0

    record = {
        "correct": is_correct,
        "box_type": box_type,
    }
    return {
        "osworld_g_acc": record,
        "osworld_g_bbox_acc": record,
        "osworld_g_polygon_acc": record,
        "osworld_g_refusal_acc": record,
    }


def _aggregate(results: List[Dict[str, Any]], target_type: Optional[str] = None) -> float:
    filtered = [r for r in results if target_type is None or r.get("box_type") == target_type]
    if not filtered:
        return 0.0
    score = sum(1 for r in filtered if r.get("correct")) / len(filtered)
    eval_logger.info("OSWorld-G %s accuracy: %.4f", target_type or "overall", score)
    return score


def osworld_g_acc(results):
    return _aggregate(results, target_type=None)


def osworld_g_bbox_acc(results):
    return _aggregate(results, target_type="bbox")


def osworld_g_polygon_acc(results):
    return _aggregate(results, target_type="polygon")


def osworld_g_refusal_acc(results):
    return _aggregate(results, target_type="refusal")
