import json
import math
import os
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from filelock import FileLock
from PIL import Image

from lmms_eval.tasks._task_utils.media_resolver import resolve_media_reference

Document = Mapping[str, Any]
TaskKwargs = Mapping[str, Any]
MetricRecord = dict[str, Any]

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _parse_clues(doc: Document) -> list[Any]:
    clues = doc.get("clue", [])
    if isinstance(clues, str):
        return json.loads(clues)
    return clues


def _default_media_root() -> Path:
    """Return the cache directory populated by lmms-eval's video downloader."""
    hf_home = Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")))
    return hf_home / "cgav_counting"


def _extract_split_archive(root: Path, archive_prefix: str, destination: Path) -> None:
    """Merge and extract one official CG-AV-Counting split ZIP archive."""
    if destination.exists():
        return

    parts = sorted(root.glob(f"{archive_prefix}.zip.part*"))
    if not parts:
        return

    lock = FileLock(str(root / f".{archive_prefix}.extract.lock"), timeout=3600)
    with lock:
        if destination.exists():
            return
        destination.mkdir(parents=True, exist_ok=True)
        archive_path = root / f"{archive_prefix}.zip"
        try:
            with archive_path.open("wb") as archive:
                for part in parts:
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, archive, length=16 * 1024 * 1024)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(destination)
        finally:
            archive_path.unlink(missing_ok=True)


def _ensure_media_extracted(extra_subdirs: Sequence[str]) -> None:
    """Extract the required official media archive when only split ZIP parts exist."""
    root = Path(os.path.expanduser(os.getenv("CGAV_COUNTING_ROOT", ""))) if os.getenv("CGAV_COUNTING_ROOT") else _default_media_root()
    if "ref_videos" in extra_subdirs:
        _extract_split_archive(root, "ref_videos", root / "ref_videos")
    else:
        _extract_split_archive(root, "videos", root / "cg_videos_720p")


def _resolve_video(reference: str, extra_subdirs: Sequence[str] = ()) -> str:
    _ensure_media_extracted(extra_subdirs)
    path = resolve_media_reference(
        reference,
        media_type="video",
        cache_dir="cgav_counting",
        env_vars=("CGAV_COUNTING_ROOT", "CGAV_COUNTING_VIDEO_DIR"),
        extra_subdirs=extra_subdirs,
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"CG-AV-Counting media not found: {path}. Set CGAV_COUNTING_ROOT to the extracted dataset root.")
    return path


def _full_video_path(doc: Document) -> str:
    return _resolve_video(doc["video"], extra_subdirs=("cg_videos_720p", "videos"))


def _reference_filename(doc: Document) -> str:
    start, end = doc["query_interval"]
    return f"{Path(doc['video']).stem}_{float(start):.2f}_{float(end):.2f}.mp4"


def cgav_doc_to_visual_long(doc: Document) -> list[str]:
    """Return the full video used by the long-video counting protocol."""
    return [_full_video_path(doc)]


def cgav_doc_to_visual_ref(doc: Document) -> list[str]:
    """Return the official reference clip for a counting question."""
    return [_resolve_video(_reference_filename(doc), extra_subdirs=("ref_videos",))]


def _clue_timestamps(doc: Document) -> list[float]:
    timestamps = []
    clues = _parse_clues(doc)
    flattened = clues if doc["category"] != "attribute" else [item for cluster in clues for item in cluster]
    for clue in flattened:
        timestamp = float(clue["timestamp"])
        if timestamp not in timestamps:
            timestamps.append(timestamp)
    return timestamps


def _frames_at_timestamps(video_path: str, timestamps: Sequence[float]) -> list[Image.Image]:
    capture = cv2.VideoCapture(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0 or fps <= 0:
        capture.release()
        raise ValueError(f"Cannot decode video: {video_path}")
    frames = []
    for timestamp in timestamps:
        index = min(max(int(timestamp * fps), 0), frame_count - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"Cannot decode frame {index} from video: {video_path}")
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    return frames


def cgav_doc_to_visual_clue(doc: Document) -> list[str] | list[Image.Image]:
    """Return a video for event clues or decoded frames for spatial clues."""
    if doc["category"] == "event":
        return [_full_video_path(doc)]
    return _frames_at_timestamps(_full_video_path(doc), _clue_timestamps(doc))


def cgav_doc_to_text_count(doc: Document, lmms_eval_specific_kwargs: TaskKwargs | None = None) -> str:
    """Format the official number-only counting prompt."""
    kwargs = lmms_eval_specific_kwargs or {}
    prompt = f"Please answer the question '{doc['question']}' with a number. " "Output only the number and nothing else."
    return f"{kwargs.get('pre_prompt', '')}{prompt}{kwargs.get('post_prompt', '')}"


def cgav_doc_to_text_clue(doc: Document, lmms_eval_specific_kwargs: TaskKwargs | None = None) -> str:
    """Format the white-box event, object, or attribute clue prompt."""
    kwargs = lmms_eval_specific_kwargs or {}
    question = doc["question"]
    category = doc["category"]
    if category == "event":
        prompt = (
            f"Watch the video and answer the question '{question}' by giving the start and end timestamps for every event. "
            "Return JSON enclosed in <answer> and </answer> tags, in this form: "
            '<answer>[["start_time", "end_time"], ...]</answer>. Use seconds (for example, "12.34").'
        )
    elif category == "object":
        frame_count = len(_clue_timestamps(doc))
        prompt = (
            f"There are {frame_count} frames, ordered as Frame1 through Frame{frame_count}. "
            f"Answer the question '{question}' by returning the bounding box for every query object in the first frame where it appears. "
            "Do not repeat an object in later frames. Return JSON enclosed in <answer> and </answer> tags, in this form: "
            '<answer>{"Frame1": [[x_min, y_min, x_max, y_max]], "Frame2": []}</answer>. '
            "Use the same 0-100 coordinate scale as the benchmark."
        )
    elif category == "attribute":
        frame_count = len(_clue_timestamps(doc))
        prompt = (
            f"There are {frame_count} frames, ordered as Frame1 through Frame{frame_count}. "
            f"Answer the question '{question}' by clustering objects according to the requested attribute. "
            "Give each cluster a stable label and return each object's box only in its first frame. Return JSON enclosed in "
            '<answer> and </answer> tags, in this form: <answer>{"Frame1": [{"bbox": [x_min, y_min, x_max, y_max], '
            '"label": "Label 1"}], "Frame2": []}</answer>. Use the same 0-100 coordinate scale as the benchmark.'
        )
    else:
        raise ValueError(f"Unsupported CG-AV-Counting category: {category}")
    return f"{kwargs.get('pre_prompt', '')}{prompt}{kwargs.get('post_prompt', '')}"


def _extract_number(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().lower().replace(",", "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match:
        return float(match.group(0))
    tokens = re.findall(r"[a-z]+", text.replace("-", " "))
    values = [_NUMBER_WORDS[token] for token in tokens if token in _NUMBER_WORDS]
    if not values:
        return 0.0
    return float(sum(values))


def _count_result(doc: Document, results: Sequence[str], task_mode: str) -> dict[str, MetricRecord]:
    prediction = results[0] if results else ""
    pred = _extract_number(prediction)
    target = float(doc["answer"])
    error = abs(target - pred)
    capped_error = error if error <= max(2 * target, 100) else abs(2 * target)
    record = {
        "index": doc["index"],
        "task_mode": task_mode,
        "category": doc["category"],
        "type": doc["type"],
        "pred": pred,
        "answer": target,
        "acc": float(error <= 1e-5),
        "oboa": float(error <= 1),
        "mae": capped_error,
        "squared_error": capped_error**2,
    }
    return {metric: record for metric in ("acc", "oboa", "mae", "rmse")}


def cgav_process_results_long(doc: Document, results: Sequence[str]) -> dict[str, MetricRecord]:
    """Score a full-video count with the official long-accuracy metrics."""
    return _count_result(doc, results, "long_acc")


def cgav_process_results_ref(doc: Document, results: Sequence[str]) -> dict[str, MetricRecord]:
    """Score a reference-clip count with the official reference metrics."""
    return _count_result(doc, results, "ref_acc")


def aggregate_acc(results: Sequence[MetricRecord]) -> float:
    """Aggregate exact counting accuracy."""
    return sum(result["acc"] for result in results) / len(results) if results else 0.0


def aggregate_oboa(results: Sequence[MetricRecord]) -> float:
    """Aggregate off-by-one accuracy."""
    return sum(result["oboa"] for result in results) / len(results) if results else 0.0


def aggregate_mae(results: Sequence[MetricRecord]) -> float:
    """Aggregate the official capped mean absolute error."""
    return sum(result["mae"] for result in results) / len(results) if results else 0.0


def aggregate_rmse(results: Sequence[MetricRecord]) -> float:
    """Aggregate root mean squared counting error."""
    return math.sqrt(sum(result["squared_error"] for result in results) / len(results)) if results else 0.0


def _temporal_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0 else 0.0


def _spatial_iou(first: Sequence[float], second: Sequence[float]) -> float:
    first = [min(first[0], first[2]), min(first[1], first[3]), max(first[0], first[2]), max(first[1], first[3])]
    second = [min(second[0], second[2]), min(second[1], second[3]), max(second[0], second[2]), max(second[1], second[3])]
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(0, min(first[3], second[3]) - max(first[1], second[1]))
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def _greedy_spatial_score(gt_instances: Sequence[Sequence[float]], pred_instances: Sequence[Sequence[float]]) -> float:
    unmatched_gt = set(range(len(gt_instances)))
    unmatched_pred = set(range(len(pred_instances)))
    score = 0.0
    while unmatched_gt and unmatched_pred:
        best = max(
            ((_spatial_iou(gt_instances[gt], pred_instances[pred]), gt, pred) for gt in unmatched_gt for pred in unmatched_pred),
            key=lambda item: item[0],
        )
        score += best[0]
        unmatched_gt.remove(best[1])
        unmatched_pred.remove(best[2])
    return score


def _cluster_pair_wcs(gt: Sequence[Any], pred: Sequence[Any], iou_type: str) -> float:
    if iou_type == "temporal":
        localization = sum(max((_temporal_iou(g, p) for p in pred), default=0.0) for g in gt) / len(gt) if gt else 0.0
        count_penalty = 1.0 - abs(len(pred) - len(gt)) / max(len(gt), 1)
        return math.sqrt(localization * max(0.0, count_penalty))

    gt_by_frame = defaultdict(list)
    pred_by_frame = defaultdict(list)
    for frame, box in gt:
        gt_by_frame[frame].append(box)
    for frame, box in pred:
        pred_by_frame[frame].append(box)
    frames = set(gt_by_frame) | set(pred_by_frame)
    score = 0.0
    for frame in frames:
        gt_boxes = gt_by_frame.get(frame, [])
        pred_boxes = pred_by_frame.get(frame, [])
        localization = _greedy_spatial_score(gt_boxes, pred_boxes) / len(gt_boxes) if gt_boxes else 0.0
        count_penalty = 1.0 - abs(len(pred_boxes) - len(gt_boxes)) / max(len(gt_boxes), 1)
        score += math.sqrt(localization * max(0.0, count_penalty))
    return score / max(len(frames), 1)


def _unlabeled_wcs(gt_clusters: Sequence[Sequence[Any]], pred_clusters: Sequence[Sequence[Any]], iou_type: str) -> float:
    if not gt_clusters or not pred_clusters:
        return 0.0
    scores = np.zeros((len(gt_clusters), len(pred_clusters)))
    for gt_index, gt in enumerate(gt_clusters):
        for pred_index, pred in enumerate(pred_clusters):
            scores[gt_index, pred_index] = _cluster_pair_wcs(gt, pred, iou_type)
    return _max_assignment_sum(scores) / len(gt_clusters)


def _max_assignment_sum(scores: Sequence[Sequence[float]] | np.ndarray) -> float:
    """Maximum-weight rectangular assignment using the O(n^3) Hungarian algorithm."""
    matrix = np.asarray(scores, dtype=float)
    if matrix.size == 0:
        return 0.0
    if matrix.shape[0] > matrix.shape[1]:
        matrix = matrix.T

    rows, columns = matrix.shape
    max_score = float(matrix.max())
    costs = max_score - matrix
    u = np.zeros(rows + 1)
    v = np.zeros(columns + 1)
    matching = np.zeros(columns + 1, dtype=int)
    way = np.zeros(columns + 1, dtype=int)

    for row in range(1, rows + 1):
        matching[0] = row
        column0 = 0
        minimum = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[column0] = True
            current_row = matching[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = costs[current_row - 1, column - 1] - u[current_row] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[matching[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matching[column0] == 0:
                break
        while True:
            column1 = way[column0]
            matching[column0] = matching[column1]
            column0 = column1
            if column0 == 0:
                break

    total = 0.0
    for column in range(1, columns + 1):
        if matching[column]:
            total += matrix[matching[column] - 1, column - 1]
    return float(total)


def _extract_json(response: str) -> Any | None:
    match = re.search(r"<answer>(.*?)</answer>", response or "", re.DOTALL | re.IGNORECASE)
    text = match.group(1).strip() if match else (response or "").strip()
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        pass
    for start, char in enumerate(text):
        if char not in "[{":
            continue
        stack = []
        for end in range(start, len(text)):
            current = text[end]
            if current in "[{":
                stack.append(current)
            elif current in "]}":
                if not stack or (stack[-1], current) not in (("[", "]"), ("{", "}")):
                    break
                stack.pop()
                if not stack:
                    try:
                        return json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _time_to_seconds(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip().split()[0]
    if ":" not in value:
        return float(value)
    parts = [float(part) for part in value.split(":")]
    return sum(part * 60**index for index, part in enumerate(reversed(parts)))


def _frame_index(key: Any) -> int | None:
    match = re.search(r"frame\s*(\d+)", str(key), re.IGNORECASE)
    return int(match.group(1)) - 1 if match else None


def _score_clue(doc: Document, response: str) -> tuple[float, float]:
    parsed = _extract_json(response)
    if parsed is None:
        return 0.0, 0.0
    clues = _parse_clues(doc)
    category = doc["category"]
    try:
        if category == "event":
            pred = [[_time_to_seconds(item[0]), _time_to_seconds(item[1])] for item in parsed]
            gt = [[float(item["start"]), float(item["end"])] for item in clues]
            return _unlabeled_wcs([gt], [pred], "temporal"), 1.0

        timestamps = _clue_timestamps(doc)
        if category == "object":
            gt = [(timestamps.index(float(item["timestamp"])), item["bbox"]) for item in clues]
            pred = []
            for key, boxes in parsed.items():
                frame = _frame_index(key)
                if frame is None or not isinstance(boxes, list):
                    continue
                if boxes and all(isinstance(value, (int, float)) for value in boxes) and len(boxes) == 4:
                    boxes = [boxes]
                elif boxes and all(isinstance(point, list) and len(point) == 2 for point in boxes):
                    boxes = [[boxes[index][0], boxes[index][1], boxes[index + 1][0], boxes[index + 1][1]] for index in range(0, len(boxes) - 1, 2)]
                for box in boxes:
                    if isinstance(box, list) and len(box) == 4:
                        pred.append((frame, box))
            return _unlabeled_wcs([gt], [pred], "spatial"), 1.0

        gt_clusters = [[(timestamps.index(float(item["timestamp"])), item["bbox"]) for item in cluster] for cluster in clues]
        pred_clusters = defaultdict(list)
        for key, objects in parsed.items():
            frame = _frame_index(key)
            if frame is None or not isinstance(objects, list):
                continue
            for item in objects:
                if not isinstance(item, dict) or "label" not in item:
                    continue
                box = item.get("bbox", item.get("bbox_2d"))
                if isinstance(box, list) and len(box) == 4:
                    pred_clusters[str(item["label"])].append((frame, box))
        return _unlabeled_wcs(gt_clusters, list(pred_clusters.values()), "spatial"), 1.0
    except (KeyError, TypeError, ValueError, IndexError):
        return 0.0, 0.0


def cgav_process_results_clue(doc: Document, results: Sequence[str]) -> dict[str, MetricRecord]:
    """Score a white-box clue response with WCS and valid-format accuracy."""
    response = results[0] if results else ""
    wcs, ifa = _score_clue(doc, response)
    record = {
        "index": doc["index"],
        "category": doc["category"],
        "type": doc["type"],
        "wcs": float(wcs),
        "ifa": float(ifa),
    }
    return {"wcs": record, "ifa": record}


def aggregate_wcs(results: Sequence[MetricRecord]) -> float:
    """Aggregate weighted clue score."""
    return sum(result["wcs"] for result in results) / len(results) if results else 0.0


def aggregate_ifa(results: Sequence[MetricRecord]) -> float:
    """Aggregate valid-format accuracy for clue responses."""
    return sum(result["ifa"] for result in results) / len(results) if results else 0.0
