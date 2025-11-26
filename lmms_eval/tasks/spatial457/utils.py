"""
Spatial457 (SpatialViz) Task Utilities - Aligned with EASI
7-level hierarchical spatial reasoning
"""
import json
import re
from typing import Dict, List


def spatial457_doc_to_visual(doc: Dict) -> List[str]:
    """Get visual inputs"""
    image_path = doc.get("image", "")
    if isinstance(image_path, list):
        return image_path
    return [image_path] if image_path else []


def spatial457_doc_to_text(doc: Dict) -> str:
    """Build prompt following EASI Spatial457 (line 141-212)"""
    question = doc.get("question", "")
    category = doc.get("category", "")

    # Task-specific instructions from EASI (line 155-191)
    task_map = {
        "L1_single": (
            "You are an intelligent chatbot designed to answer questions based on an image. "
            "Your task is to analyze the images, identify attributes of the objects, "
            "and then determine the answer to the question.\n"
        ),
        "L2_objects": (
            "You are an intelligent chatbot designed to answer questions based on an image. "
            "Your task is to analyze the images, identify attributes of multiple objects, "
            "and then determine the answer to the question.\n"
        ),
        "L3_2d_spatial": (
            "You are an intelligent chatbot designed to answer questions based on an image. "
            "Your task is to analyze the images, identify attributes of multiple objects "
            "and their spatial relationship from 2D projected camera view, "
            "and then determine the answer to the question.\n"
        ),
        "L4_occ": (
            "You are an intelligent chatbot designed to answer questions based on an image. "
            "Your task is to analyze the images, identify attributes of multiple objects "
            "and their occlusion relationships, and then determine the answer to the question.\n"
        ),
        "L4_pose": (
            "You are an intelligent chatbot designed to answer questions based on an image. "
            "Your task is to analyze the images, identify attributes of multiple objects "
            "and their facing direction in 3D space from the camera view, "
            "and then determine the answer to the question.\n"
        ),
        "L5_6d_spatial": (
            "You are an intelligent chatbot designed to answer questions based on an image. "
            "Your task is to analyze the images, identify attributes of multiple objects "
            "and their spatial relationship from objects' perspective in 3D space, "
            "and then determine the answer to the question.\n"
        ),
        "L5_collision": (
            "You are an intelligent chatbot designed to answer questions based on an image. "
            "Your task is to analyze the images, identify attributes of multiple objects "
            "and their potential collision given the assumption of moving direction in 3D space, "
            "and then determine the answer to the question.\n"
        ),
    }

    instruction_1 = task_map.get(category, "")

    # EASI line 195-206
    instruction_2 = (
        "First, you should identify the related objects refered in the questions, "
        "including their shape, color, size; then add a brief reasoning process about the questions. "
        "Each object in the image has a shape (e.g., 'airliner'), a size (only can be 'small' or 'large'), "
        "a color (e.g. 'blue'). The size of the object is either 'small' or 'large'. "
        "The color of the object is one of the following: 'gray', 'blue', 'purple', 'brown', "
        "'green', 'cyan', 'red', 'yellow'. The direction of the object is one of the following: "
        "'left', 'right', 'front', 'back'.\n\n"
        "Second, give the answer based on the reasoning process. The answer should only be "
        "(1) a phrase chosen from the following options: {}, or (2) an integer [0-10] when asked for "
        "'How many' or 'What is the number of', or (3) 'Yes' or 'No' when asked for 'Is there'. "
        "If you think there are no possible answers or the question is not clear, choose the best answer "
        "that fits the question.\n\n"
    ).format("all_answers")  # Note: EASI uses dataset_utils.all_answers() here

    # EASI line 208-210
    instruction_2 += (
        "Write your response into this json template: "
        "{'Reasoning': '<your reasons>', 'Answer': '<Your answer>'}"
    )

    prompt = instruction_1 + question + "\n" + instruction_2
    return prompt


def spatial457_process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    """Process results following EASI Spatial457 evaluation (line 60-139)"""
    result_text = results[0] if results else ""
    gt_answer = str(doc.get("answer", "")).strip()
    category = doc.get("category", "")

    # Extract answer from JSON response (EASI line 69-84)
    pred = extract_answer_from_response(result_text)

    # Check correctness (EASI line 100)
    correct = is_correct(gt_answer, pred)

    # Return overall and category-specific metrics
    result = {"accuracy": float(correct)}
    result[f"{category}_acc"] = float(correct)

    return result


def extract_answer_from_response(text: str) -> str:
    """Extract answer following EASI line 69-84"""
    # Parse the answer (EASI tries multiple regex patterns)
    pred_try_1 = re.search(r"Answer': '(.*?)'", text)
    pred_try_2 = re.search(r'Answer": "(.*?)"', text)
    pred_try_3 = re.search(r"Answer': (\d)", text)

    if pred_try_1:
        return pred_try_1.group(1)
    elif pred_try_2:
        return pred_try_2.group(1)
    elif pred_try_3:
        return pred_try_3.group(1)
    else:
        # ROBUST mode: use whole response (EASI line 81-82)
        return text.strip()


def is_correct(gt: str, pred: str) -> bool:
    """Check correctness (EASI uses dataset_utils.is_correct)"""
    gt = gt.strip().lower()
    pred = pred.strip().lower()
    return gt == pred
