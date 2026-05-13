import re

# Mapping from question_type in dataset to metric key
VIEWSPATIAL_QUESTION_TYPES = {
    "Camera perspective - Relative Direction": "camera_perspective_relative_direction",
    "Camera perspective - Object View Orientation": "camera_perspective_object_view_orientation",
    "Person perspective - Relative Direction": "person_perspective_relative_direction",
    "Person perspective - Object View Orientation": "person_perspective_object_view_orientation",
    "Person perspective - Scene Simulation Relative Direction": "person_perspective_scene_simulation_relative_direction",
}


def viewspatial_doc_to_text(doc):
    """Extracts the text prompt from a viewspatial dataset sample.

    Args:
        doc (dict): A viewspatial dataset sample dictionary.

    Returns:
        str: The input prompt text.
    """
    question = doc["question"]
    choices = doc["choices"]

    # prompt format from viewspatial bench paper
    question_text = f"Question: {question}\n"
    choices_text = f"Choices: {choices}\n"
    post_prompt = "Reply only to the corresponding option.\nAnswer:"

    prompt = question_text + choices_text + post_prompt
    return prompt


def viewspatial_doc_to_visual(doc):
    """Extracts and converts visual data from a viewspatial dataset sample.

    This function iterates over the 'images' key in the dataset sample, calling
    the .convert("RGB") method on each item (presumed to be a PIL/Pillow Image).

    Args:
        doc (dict): A viewspatial dataset sample dictionary.

    Returns:
        list: A list of visual elements (e.g., PIL Images) converted to 'RGB' format.
    """
    return [visual.convert("RGB") for visual in doc["images"]]


def extract_option(text):
    match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def viewspatial_process_results(doc, results):
    """Processes the model's output for a single viewspatial document."""
    # extract grounded answer
    grounded_output = doc["answer"]
    grounded_option = extract_option(grounded_output)
    # eval_logger.info(f"Grounded answer: {grounded_output}")

    # extract predicted answer
    pred = results[0]
    pred = pred.split("\n")[-1]
    pred_answer = extract_option(pred)
    # eval_logger.info(f"Predicted answer: {pred_answer}")

    score = 1.0 if pred_answer == grounded_option else 0.0

    question_type = doc.get("question_type", "unknown")

    result = {"overall_accuracy": {"score": score}}

    # Per-subcategory metrics
    for qt_name, metric_key in VIEWSPATIAL_QUESTION_TYPES.items():
        result[f"{metric_key}_accuracy"] = {"score": score, "question_type": question_type, "target_type": qt_name}

    return result


def viewspatial_aggregate_results(results):
    """Aggregates the 'overall_accuracy' results."""
    if not results:
        return 0.0
    total_score = sum(res["score"] for res in results)
    return total_score / len(results)


def _viewspatial_aggregate_by_type(results, target_type: str) -> float:
    """Aggregate accuracy for a specific question type."""
    scores = [r["score"] for r in results if r.get("question_type") == target_type]
    return sum(scores) / len(scores) if scores else 0.0


def viewspatial_aggregate_camera_perspective_relative_direction(results):
    return _viewspatial_aggregate_by_type(results, "Camera perspective - Relative Direction")


def viewspatial_aggregate_camera_perspective_object_view_orientation(results):
    return _viewspatial_aggregate_by_type(results, "Camera perspective - Object View Orientation")


def viewspatial_aggregate_person_perspective_relative_direction(results):
    return _viewspatial_aggregate_by_type(results, "Person perspective - Relative Direction")


def viewspatial_aggregate_person_perspective_object_view_orientation(results):
    return _viewspatial_aggregate_by_type(results, "Person perspective - Object View Orientation")


def viewspatial_aggregate_person_perspective_scene_simulation_relative_direction(results):
    return _viewspatial_aggregate_by_type(results, "Person perspective - Scene Simulation Relative Direction")
