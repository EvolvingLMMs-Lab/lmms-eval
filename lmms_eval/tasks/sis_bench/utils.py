import os
from pathlib import Path

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

CACHE_DIR_NAME = "sis_bench"
CHOICES = ("A", "B", "C", "D")

SPATIAL_COGNITION_TASKS = {
    "object_existence",
    "object_attribute",
    "relative_direction",
    "landmark_appearance_order",
    "landmark_recall",
    "positional_relationship",
    "spatial_consistency",
    "spatio-temporal_consistency",
}
SELF_AWARENESS_TASKS = {
    "action_recognition",
    "action_sequence",
    "action_recall",
    "action_prediction",
    "path_planning",
}


def _cache_dir() -> Path:
    hf_home = Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")))
    return hf_home / CACHE_DIR_NAME


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"SIS-Bench video path must be relative: {value!r}")
    return path


def sis_bench_doc_to_visual(doc):
    video_name = _safe_relative_path(doc["video_name"])
    candidates = [_cache_dir() / "video" / video_name]

    # Keep compatibility with the dataset's legacy UAVideo-prefixed field.
    if doc.get("video_path"):
        video_path = _safe_relative_path(doc["video_path"])
        candidates.append(_cache_dir() / video_path)
        if video_path.parts and video_path.parts[0] == "UAVideo":
            candidates.append(_cache_dir() / "video" / Path(*video_path.parts[1:]))

    for video_path in candidates:
        if video_path.is_file():
            return [str(video_path)]

    checked = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"SIS-Bench video not found. Checked:\n{checked}")


def sis_bench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    options = doc["options"]
    option_lines = [f"({choice}) {options[choice]}" for choice in CHOICES]
    prompt = f"{kwargs.get('pre_prompt', '')}{doc['question']}\nOptions:\n" + "\n".join(option_lines)
    post_prompt = kwargs.get("post_prompt", "")
    if post_prompt:
        prompt += f"\n{post_prompt}"
    return prompt


def _dimension_for_task(task_type: str) -> str:
    if task_type in SPATIAL_COGNITION_TASKS:
        return "spatial_cognition"
    if task_type in SELF_AWARENESS_TASKS:
        return "self_awareness"
    raise ValueError(f"Unknown SIS-Bench task type: {task_type!r}")


def sis_bench_process_results(doc, results):
    prediction = results[0] if results else ""
    predicted_answer = extract_mcq_answer(prediction, choices=list(CHOICES))
    answer = str(doc["answer"]).strip().upper()
    record = {
        "question_id": doc["question_id"],
        "task_type": doc["task_type"],
        "dimension": _dimension_for_task(doc["task_type"]),
        "prediction": prediction,
        "predicted_answer": predicted_answer,
        "answer": answer,
        "score": int(predicted_answer == answer),
    }
    dimension_metric = f"sis_bench_{record['dimension']}_accuracy"
    task_metric = f"sis_bench_{doc['task_type'].replace('-', '_')}_accuracy"
    return {
        "sis_bench_overall_accuracy": record,
        dimension_metric: record,
        task_metric: record,
    }


def _aggregate_accuracy(results, dimension=None, task_type=None):
    """Compute question-weighted accuracy, matching the official SIS-Bench scorer."""
    selected = [
        result
        for result in results
        if (dimension is None or result["dimension"] == dimension) and (task_type is None or result["task_type"] == task_type)
    ]
    if not selected:
        return 0.0
    accuracy = 100.0 * sum(result["score"] for result in selected) / len(selected)
    label = task_type or dimension or "overall"
    eval_logger.info(f"SIS-Bench {label} accuracy: {accuracy:.2f}% ({sum(result['score'] for result in selected)}/{len(selected)})")
    return accuracy


def sis_bench_aggregate_overall(results):
    return _aggregate_accuracy(results)


def sis_bench_aggregate_spatial_cognition(results):
    return _aggregate_accuracy(results, "spatial_cognition")


def sis_bench_aggregate_self_awareness(results):
    return _aggregate_accuracy(results, "self_awareness")


def sis_bench_aggregate_object_existence(results):
    return _aggregate_accuracy(results, task_type="object_existence")


def sis_bench_aggregate_object_attribute(results):
    return _aggregate_accuracy(results, task_type="object_attribute")


def sis_bench_aggregate_relative_direction(results):
    return _aggregate_accuracy(results, task_type="relative_direction")


def sis_bench_aggregate_landmark_appearance_order(results):
    return _aggregate_accuracy(results, task_type="landmark_appearance_order")


def sis_bench_aggregate_landmark_recall(results):
    return _aggregate_accuracy(results, task_type="landmark_recall")


def sis_bench_aggregate_positional_relationship(results):
    return _aggregate_accuracy(results, task_type="positional_relationship")


def sis_bench_aggregate_spatial_consistency(results):
    return _aggregate_accuracy(results, task_type="spatial_consistency")


def sis_bench_aggregate_spatio_temporal_consistency(results):
    return _aggregate_accuracy(results, task_type="spatio-temporal_consistency")


def sis_bench_aggregate_action_recognition(results):
    return _aggregate_accuracy(results, task_type="action_recognition")


def sis_bench_aggregate_action_sequence(results):
    return _aggregate_accuracy(results, task_type="action_sequence")


def sis_bench_aggregate_action_recall(results):
    return _aggregate_accuracy(results, task_type="action_recall")


def sis_bench_aggregate_action_prediction(results):
    return _aggregate_accuracy(results, task_type="action_prediction")


def sis_bench_aggregate_path_planning(results):
    return _aggregate_accuracy(results, task_type="path_planning")
