"""CC-OCR V2 text and object grounding on the original image dimensions."""

import ast
import json
import math
import re


def _iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _coordinates(value):
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        value = value[:4]
        if any(isinstance(x, bool) or not isinstance(x, (float, int, str)) for x in value):
            return None
        try:
            numbers = [float(x) for x in value]
        except ValueError:
            return None
        return numbers if all(math.isfinite(x) for x in numbers) else None
    if isinstance(value, dict):
        values = {str(k).strip().lower(): v for k, v in value.items()}
        for keys in (
            ("x1", "y1", "x2", "y2"),
            ("x1", "y", "x2", "y2"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("left", "top", "right", "bottom"),
            ("x0", "y0", "x1", "y1"),
        ):
            box = _coordinates([values.get(k) for k in keys])
            if box is not None:
                return box
        for w in ("width", "w"):
            for h in ("height", "h"):
                box = _coordinates([values.get(k) for k in ("x", "y", w, h)])
                if box is not None:
                    x, y, width, height = box
                    return [x, y, x + width, y + height]
    return None


def _object_boxes(text):
    try:
        values = json.loads(text)
    except ValueError:
        try:
            values = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(values, list):
        return []
    boxes = []
    for obj in values:
        if not isinstance(obj, dict) or (obj.get("label") is None and obj.get("category_name") is None):
            continue
        box = _coordinates(obj.get("bbox_2d") or obj.get("bbox") or obj.get("box_2d"))
        if box is not None:
            boxes.append(box)
    return boxes


def _to_pixels(boxes, width, height):
    scale = 1 if max(max(box) for box in boxes) <= 1 else 1000
    return [[x * dim / scale for x, dim in zip(box, (width, height, width, height), strict=True)] for box in boxes]


def score(doc, reference, prediction):
    width, height = doc["image_widths"][0], doc["image_heights"][0]
    if doc["sub_task"] == "text_grounding":
        try:
            reference_box = _coordinates(ast.literal_eval(reference.strip()))
        except (ValueError, SyntaxError):
            reference_box = None
        if reference_box is None:
            return None
        match = re.search(r"[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*[\)\]]", prediction)
        if match is None:
            return 0.0
        # The text scorer converts each regex capture with float(). A malformed
        # captured number is a scorer error; only an unmatched pattern earns zero.
        box = [float(number) for number in match.groups()]
        return _iou(_to_pixels([box], width, height)[0], reference_box)

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    reference_boxes = _object_boxes(reference)
    if not reference_boxes:
        return None
    predicted_boxes = _object_boxes(prediction)
    if not predicted_boxes:
        return 0.0
    predicted_boxes = _to_pixels(predicted_boxes, width, height)
    ious = np.array([[_iou(a, b) for b in predicted_boxes] for a in reference_boxes])
    rows, columns = linear_sum_assignment(-ious)
    return float(ious[rows, columns].sum() / len(reference_boxes))
