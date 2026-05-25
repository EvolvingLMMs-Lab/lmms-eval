import hashlib
import os
import re
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

_MASK_CACHE_DIR = os.path.join(tempfile.gettempdir(), "robo_spatial_masks")
os.makedirs(_MASK_CACHE_DIR, exist_ok=True)


def remove_think(content: str) -> str:
    content = re.sub(r"^.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"^.*?</thought>", "", content, flags=re.DOTALL)
    match = re.search(r"<answer>(.*?)</answer>", content, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return content.strip()


def json2pts(text: str, width=640, height=480) -> np.ndarray:
    import json

    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if not match:
        return np.empty((0, 2), dtype=int)

    try:
        data = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return np.empty((0, 2), dtype=int)

    points = []
    for item in data:
        if "point_2d" in item and isinstance(item["point_2d"], list) and len(item["point_2d"]) == 2:
            x_norm, y_norm = item["point_2d"]
            x = int(x_norm / 1000 * width)
            y = int(y_norm / 1000 * height)
            points.append((x, y))
    return np.array(points)


def evaluate_binary_answer(ground_truth, generated_answer):
    gen_answer = generated_answer.strip().lower().split("\n")[-1]
    is_gt_yes = ground_truth.strip().lower() == "yes"
    if is_gt_yes:
        correct = "yes" in gen_answer and "no" not in gen_answer
    else:
        correct = "no" in gen_answer and "yes" not in gen_answer
    return correct


def _save_mask(doc):
    from io import BytesIO

    mask = doc.get("mask")
    if mask is None:
        return {"mask_path": ""}
    # HuggingFace datasets stores image columns as {'bytes': b'...'} dicts
    # when the column is not declared with Image() feature — decode it here.
    if isinstance(mask, dict) and mask.get("bytes"):
        try:
            mask = Image.open(BytesIO(mask["bytes"]))
        except Exception:
            return {"mask_path": ""}
    if isinstance(mask, Image.Image):
        doc_hash = hashlib.md5(doc.get("question", "").encode()).hexdigest()
        mask_path = os.path.join(_MASK_CACHE_DIR, f"{doc_hash}.png")
        if not os.path.exists(mask_path):
            mask.save(mask_path)
        return {"mask_path": mask_path}
    return {"mask_path": ""}


def robo_spatial_process_docs(dataset):
    # load_from_cache_file=False re-saves mask PNGs to /tmp on every run,
    # preventing stale cached mask_path values pointing to deleted temp files.
    return dataset.map(_save_mask, load_from_cache_file=False)


def evaluate_pointing_with_mask(doc, generated_answer):
    mask_path = doc.get("mask_path")
    if not mask_path or not os.path.exists(mask_path):
        return -1.0

    try:
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.uint8)
    except Exception:
        return -1.0

    h, w = mask.shape
    pred_points = json2pts(generated_answer, width=w, height=h)
    if len(pred_points) == 0:
        return 0.0

    for pt in pred_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h and mask[y, x]:
            return 1.0
    return 0.0


def evaluate_pointing_with_mask_first_only(doc, generated_answer):
    # Strict: only the first predicted point counts. Removes the "any point in
    # mask" loophole — the model must commit to a single best point.
    mask_path = doc.get("mask_path")
    if not mask_path or not os.path.exists(mask_path):
        return -1.0

    try:
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.uint8)
    except Exception:
        return -1.0

    h, w = mask.shape
    pred_points = json2pts(generated_answer, width=w, height=h)
    if len(pred_points) == 0:
        return 0.0
    x, y = int(pred_points[0][0]), int(pred_points[0][1])
    if 0 <= x < w and 0 <= y < h and mask[y, x]:
        return 1.0
    return 0.0


def robo_spatial_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return [doc["img"]]


_HIRES_XL_TARGET_PIXELS = 1536 * 28 * 28  # 1,204,224 px ≈ 1536 vision tokens


def _upscale_if_small(img, target_pixels):
    from PIL import Image as _PILImage

    w, h = img.size
    cur = w * h
    if cur >= target_pixels:
        return img
    scale = (target_pixels / cur) ** 0.5
    return img.resize((int(round(w * scale)), int(round(h * scale))), _PILImage.BICUBIC)


def robo_spatial_doc_to_visual_hires_xl_pointing_only(doc, lmms_eval_specific_kwargs=None):
    # Upscale only pointing (context) images — yes/no categories are unaffected
    # by visual detail and over-upscaling hurts them.
    img = doc["img"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    if doc.get("category") == "context":
        return [_upscale_if_small(img, _HIRES_XL_TARGET_PIXELS)]
    return [img]


def get_post_prompt(doc):
    # reference: https://github.com/flageval-baai/FlagEvalMM/blob/main/tasks/robo_spatial_home/robo_spatial_home_all.py
    post_prompt_point = """Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, followed by a list of 2D point coordinates.

    The points should be in JSON format and normalized to the range [0, 1000].

    Do not include additional text after this line.
    """

    post_prompt_yes_no = """Your task is to answer the question above. Respond with a brief explanation if needed, followed by a yes or no answer in the last line of your response.

    Format your final answer strictly as follows on the last line of your response:
    Answer: yes or no

    Do not include additional text after this line.
    """
    if doc["category"] == "context":
        return post_prompt_point
    else:
        return post_prompt_yes_no


def get_post_prompt_one_point_v2(doc):
    # Single-point, terse variant emphasizing "NOT on the object".
    post_prompt_point = """Pinpoint the SINGLE best 2D point in the VACANT SPACE described by the question. The point MUST NOT lie on the reference object. It MUST lie in empty space on the specified side.

    Output exactly one coordinate pair in JSON, normalized to [0, 1000]:
    ```json
    [{"point_2d": [x, y], "label": "vacant space"}]
    ```

    Do not output more than one point. Do not include additional text after the JSON block.
    """
    post_prompt_yes_no = """Your task is to answer the question above. Respond with a brief explanation if needed, followed by a yes or no answer in the last line of your response.

    Format your final answer strictly as follows on the last line of your response:
    Answer: yes or no

    Do not include additional text after this line.
    """
    if doc["category"] == "context":
        return post_prompt_point
    else:
        return post_prompt_yes_no


_QUESTION_SUFFIX_STRIP = [
    "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points.",
    "Answer yes or no.",
]


def _clean_question(doc):
    question = doc.get("question", "")
    for s in _QUESTION_SUFFIX_STRIP:
        question = question.replace(s, "").strip()
    return question


def robo_spatial_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return f"{_clean_question(doc)}\n{get_post_prompt(doc)}"


def robo_spatial_doc_to_text_one_point_v2(doc, lmms_eval_specific_kwargs=None):
    return f"{_clean_question(doc)}\n{get_post_prompt_one_point_v2(doc)}"


def robo_spatial_doc_to_answer(doc):
    return doc.get("answer", "")


def robo_spatial_process_results_generation(doc, result):
    pred = remove_think(result[0])
    ground_truth = robo_spatial_doc_to_answer(doc)
    gt_lower = ground_truth.strip().lower()

    if gt_lower in ["yes", "no"]:
        score = int(evaluate_binary_answer(ground_truth, pred))
    else:
        score = evaluate_pointing_with_mask(doc, pred)

    return {"robo_spatial_score": {"score": score, "category": doc["category"]}, "score": score}


def robo_spatial_process_results_first_point(doc, result):
    # Strict pointing scorer: only the first predicted point is checked.
    pred = remove_think(result[0])
    ground_truth = robo_spatial_doc_to_answer(doc)
    gt_lower = ground_truth.strip().lower()
    if gt_lower in ["yes", "no"]:
        score = int(evaluate_binary_answer(ground_truth, pred))
    else:
        score = evaluate_pointing_with_mask_first_only(doc, pred)
    return {"robo_spatial_score": {"score": score, "category": doc["category"]}, "score": score}


def robo_spatial_aggregate_score(results, args):
    df = pd.DataFrame(results)
    overall_score = df["score"].mean()
    category_score_dict = df.groupby("category")["score"].mean().to_dict()
    return {"overall": overall_score, **category_score_dict}
