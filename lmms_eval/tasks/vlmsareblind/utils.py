import re
from collections import defaultdict


TASKS = [
    "circled letter",
    "counting grid - blank grids",
    "counting grid - word grids",
    "line plot intersections",
    "nested squares",
    "olympic counting - circles",
    "olympic counting - pentagons",
    "subway connections",
    "touching circles",
]

TASK_MAP = {
    "counting grid - blank grids": "counting grid",
    "counting grid - word grids": "counting grid",
}


def vlmsareblind_doc_to_visual(doc: dict) -> list:
    """Extract image from the document."""
    if "image" in doc:
        return [doc["image"]]
    return []


def vlmsareblind_doc_to_text(doc: dict, lmms_eval_specific_kwargs: dict | None = None) -> str:
    """Format the prompt for the model."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    prompt = doc.get("prompt", "")

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre_prompt}{prompt}{post_prompt}"


def parse_response(response: str, task: str) -> str | None:
    """Parse the response based on the task type.

    Return None if the response does not follow the expected format.
    """
    response = response.strip().lower()
    task = task.lower()
    
    if task == "touching circles":
        if response in ["yes", "no"]:
            return response

    if task in [
        "line plot intersections", 
        "subway connections", 
        "olympic counting - circles", 
        "olympic counting - pentagons", 
        "nested squares"
    ]:
        match = re.search(r"\{?(\d+)\}?", response)
        if match:
            return match.group(1)

    if task == "circled letter":
        match = re.search(r"\{?([a-z])\}?", response)
        if match:
            return match.group(1)

    if task.startswith("counting grid"):
        match = re.search(r"\{(\d+)\}\s*\{(\d+)\}", response) # {3}{4}, {3} {4}, etc.
        if match:
            return f"{match.group(1)},{match.group(2)}"
        match = re.search(r"\((\d+)\s*,\s*(\d+)\)", response) # (3,4), (3, 4), etc.
        if match:
            return f"{match.group(1)},{match.group(2)}"
        match = re.search(r"rows=\{(\d+)\}\scolumns=\{(\d+)\}", response) # rows={3} columns={4}, rows={3} columns={4}, etc.
        if match:
            return f"{match.group(1)},{match.group(2)}"
    
    return None


def vlmsareblind_process_results(doc: dict, results: list[str]) -> dict:
    """Process the model's response and compare with ground truth."""
    result = results[0]
    task: str = doc.get("task", "")
    target: str = doc.get("groundtruth", "")

    normalized_target = target.strip().lower()
    normalized_result = result.strip().lower()
    prediction = parse_response(normalized_result, task)

    if prediction is None:
        is_correct = False
    else:
        is_correct = prediction.strip().lower() == normalized_target

    return {
        "accuracy": float(is_correct),
        "accuracy_by_task": {"task": task, "is_correct": is_correct}
    }


def vlmsareblind_aggregate_by_task(results: list[dict]) -> dict[str, float]:
    task_correct = defaultdict(int)
    task_total = defaultdict(int)

    for result in results:
        task = result["task"].lower()
        task = TASK_MAP.get(task, task)

        is_correct = result["is_correct"]
        task_total[task] += 1
        if is_correct:
            task_correct[task] += 1
        
    task_accuracy = {task: task_correct[task] / task_total[task] for task in task_correct}
    task_accuracy["task_mean"] = sum(task_accuracy.values()) / len(task_accuracy)

    return task_accuracy


def vlmsareblind_sample_100_per_task(dataset: object) -> object:
    """Subsample the dataset to at most 100 examples per task."""
    indices_by_task: dict[str, list[int]] = defaultdict(list)

    for idx, row in enumerate(dataset):
        task = str(row.get("task", ""))
        if not task:
            continue
        if len(indices_by_task[task]) < 100:
            indices_by_task[task].append(idx)

    selected_indices: list[int] = []
    for indices in indices_by_task.values():
        selected_indices.extend(indices)

    return dataset.select(selected_indices)
