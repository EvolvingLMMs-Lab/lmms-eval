import glob
import json
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks.spatialtreebench.metrics import rule_metrics


class TreeNode:
    """
    A node in the SpaTree hierarchy.
    Allows accessing children by name like a dictionary.
    """

    def __init__(self, name, weight=None):
        self.name = name
        self.weight = weight
        self.children = {}
        self.parent = None

    def add_child(self, node):
        """Adds a child node."""
        self.children[node.name] = node
        node.parent = self

    def __getitem__(self, key):
        """Access child by name."""
        return self.children[key]

    def __repr__(self):
        return f"TreeNode(name='{self.name}', weight={self.weight}, children={len(self.children)})"

    def get_node_by_path(self, path):
        """
        Access a node by a path of names.
        Example: root.get_node_by_path(['L1', 'Geometry', 'Size'])
        """
        node = self
        for part in path:
            if node is None:
                return None
            node = node.children.get(part)
        return node


def _dict_to_treenode(d):
    """Recursively converts a dictionary to a TreeNode object."""
    name = d.get("name")
    weight = d.get("weight")
    node = TreeNode(name, weight)
    children_data = d.get("children", [])
    for child_d in children_data:
        child_node = _dict_to_treenode(child_d)
        node.add_child(child_node)
    return node


def load_spatree_hierarchy(file_path):
    """
    Loads the SpaTree hierarchy from a JSON file into a tree of TreeNode objects.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return _dict_to_treenode(data)


spatree_hierarchy = load_spatree_hierarchy(str(Path(__file__).parent / "spatree_hierarchy.json"))

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(os.getenv("SPATREEBENCH_MEDIA_ROOT", hf_home))
hf_datasets_cache_dir = os.path.join(os.path.expanduser(hf_home), "datasets")
with open(Path(__file__).parent / "spatialtreebench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


@lru_cache(maxsize=20000)
def _find_in_hf_datasets_cache(rel_path: str) -> str | None:
    if not rel_path:
        return None
    if not os.path.isdir(hf_datasets_cache_dir):
        return None

    pattern = os.path.join(hf_datasets_cache_dir, "**", "snapshots", "**", rel_path)
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def _ensure_local_media_path(path_or_url: str) -> str:
    if not path_or_url:
        return path_or_url

    # Local path
    if os.path.isabs(path_or_url):
        return path_or_url

    # Check path in the specific cache directory (e.g. ~/.cache/huggingface/SpatialTreeBench/videos/...)
    # This is where 'video: True' unzips files
    candidate_cache = os.path.join(base_cache_dir, cache_name, path_or_url)
    if os.path.exists(candidate_cache):
        return candidate_cache

    candidate = os.path.join(base_cache_dir, path_or_url)
    if os.path.exists(candidate):
        return candidate

    cached = _find_in_hf_datasets_cache(path_or_url)
    if cached and os.path.exists(cached):
        return cached

    raise FileNotFoundError(
        f"Media file not found for '{path_or_url}'. Tried: {candidate} and HF datasets cache under {hf_datasets_cache_dir}. " "Set SPATREEBENCH_MEDIA_ROOT to a folder containing images/ and videos/ if your media is stored elsewhere."
    )


# Define metric functions
METRIC_REGISTRY = {
    "gravityeval": rule_metrics.gravityeval,
    "multichoiceeval": rule_metrics.multichoiceeval,
    "meanrelativeacc": rule_metrics.meanrelativeacc,
    "cogmapeval": rule_metrics.cogmapeval,
    "gpteval": rule_metrics.gpteval,
    "judge": rule_metrics.judgemodel,
    "affmask": rule_metrics.aff_mask_metric,
    "manipulateeval": rule_metrics.manipulateeval,
    "agenticnaveval": rule_metrics.agenticnaveval,
}


def spatialtreebench_doc_to_visual(doc):
    if doc.get("video") and len(doc["video"]) > 0:
        video_ref = doc["video"][0]  # relative path (videos/xxx.mp4) or URL
        video_path = _ensure_local_media_path(video_ref)

        # Use video_info if available (it should be)
        if doc.get("video_info") and len(doc["video_info"]) > 0:
            video_info = doc["video_info"][0]
            source_indices = video_info.get("frame_index", [])
        else:
            source_indices = []

        cap = cv2.VideoCapture(video_path)

        if source_indices:
            num_frames_to_sample = 32
            if len(source_indices) > num_frames_to_sample:
                sampled_indices_of_source = np.linspace(0, len(source_indices) - 1, num_frames_to_sample, dtype=int)
                indices = [source_indices[i] for i in sampled_indices_of_source]
            else:
                indices = source_indices
        else:
            # Fallback to original logic if frame_index is not available
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Sample 16 frames for short videos, 32 for long videos
            num_frames_to_sample = 32 if total_frames > 240 else 16
            indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames
    else:
        # if image, return list of <Image.open(image_path).convert("RGB")>
        images = []
        # New format: doc["image"] is a list of relative paths like ["images/xxx.jpg"]
        image_list = doc.get("image", [])
        for image_rel_path in image_list:
            image_path = _ensure_local_media_path(image_rel_path)
            images.append(Image.open(image_path).convert("RGB"))
        return images


PE_TEMPLATES = {
    "open_default": "{} ",  # gravityeval,
    "cameramotion_open": "{}",  # L2_Underst._MotionUnderstanding_gpteval
    "humananno_open": "{} \n Options : {}\nSelect the option that contains the correct action and the indication of whether the task has been completed. Return only the letter of the correct option (e.g., A, B, C, etc.). Always return in this format: 'answer: X' ",  # L4_Open-worldExploration_Self-Goaling_multichoiceeval
    "mcq_default": "{}\nOptions : {}\nChoose the correct option. Return only the letter of the correct option (e.g., A, B, C, etc.). Always return in this format: 'answer: X'",
    "dopp_judge": "{}",  # L2_Underst._RelationUnderstanding_judge
    "mcq_default2": "{}\n{}",
    "aff": "You will be given a textual description of a point of interest within an image. Your task is to locate this point and return its normalized coordinates.\nThe point to identify is: **{}**\n**Instructions for Coordinate Output:**\n1.  **Format**: The output must be a single JSON array in the format `[x, y]`.\n2.  **Coordinate System**: The origin `(0, 0)` is at the top-left corner of the image.\n3.  **Normalization**:\n- `x` is the horizontal coordinate, normalized by the image's width.\n- `y` is the vertical coordinate, normalized by the image's height.\n4.  **Value Range**: Both `x` and `y` must be float values strictly between 0 and 1 (i.e., `0 < x < 1` and `0 < y < 1`).\n5.  **Strict Output**: Do not provide any text or explanation other than the coordinate array itself.\n**Example:**\n- If the image is 1000px wide and 800px high, and the point is at pixel (500, 200), the output should be `[0.5, 0.25]`.\nNow, based on the image provided and the target description above, output the coordinates.",  # L2_Underst._Affordance_affmask
    "open_l4": "{}\nYou must ensure that the action intensity values represented by `step_nums` are within the range of 0 to 10, inclusive.\nPlease provide the action output formatted as a JSON object and enclosed within a ```json``` code block. Please do your best to move and try. This can help you get more rewards.",  # L4
    "meanrelativeacc_open": "{} Do not response anything other than a single number! If you cannot determine the answer, please guess a value.",
}


def spatialtreebench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    extra_info = json.loads(doc.get("extra_info", "{}"))
    metric_func = extra_info.get("metricfunc")
    level0 = extra_info.get("spatree0")  # e.g., "L1"
    pe_style = extra_info.get("pe_style")  # e.g., "style_1"

    is_mcq = metric_func in ["multichoiceeval", "cogmapeval"]
    question_text = doc.get("question", "")
    if metric_func == "multichoiceeval":
        if level0 == "L4":
            template_key = "humananno_open"
        else:
            template_key = "mcq_default"
    elif metric_func == "judge":
        template_key = "dopp_judge"
    elif metric_func == "affmask":
        template_key = "aff"
    elif metric_func == "manipulateeval" or metric_func == "agenticnaveval":
        template_key = "open_l4"
    elif metric_func == "meanrelativeacc":
        template_key = "meanrelativeacc_open"
    else:
        template_key = None

    if template_key in ["mcq_default", "mcq_default2", "humananno_open"]:
        options = doc.get("option")
        if options:
            # Format options list into a string
            option_str = ""
            for i, option in enumerate(options):
                option_str += f"{option}\n"

            # Select template
            template = PE_TEMPLATES.get(template_key, PE_TEMPLATES["mcq_default2"])

            return template.format(question_text, option_str.strip())

    if template_key in ["mcq_default", "mcq_default2", "humananno_open"]:
        template = PE_TEMPLATES["open_default"]
    else:
        template = PE_TEMPLATES.get(template_key, PE_TEMPLATES["open_default"])
    return template.format(question_text)


def spatialtreebench_doc_to_target(doc):
    return doc.get("answer")


def spatialtreebench_process_results(doc, results):
    """
    Calculate the metric for the document.
    """
    extra_info = json.loads(doc["extra_info"])
    metric_func_name = extra_info.get("metricfunc")
    metric_func = METRIC_REGISTRY.get(metric_func_name)

    if not metric_func:
        eval_logger.warning(f"No metric function found for {metric_func_name}. Returning 0.")
        score = 0
    else:
        prediction = results[0]
        answer = doc.get("answer")
        question_type = doc.get("question_type")
        metric_extra_info = {}

        if doc.get("hint") != "":
            metric_extra_info = json.loads(doc.get("hint", "{}"))
        else:
            metric_extra_info = {}

        if metric_func_name == "multichoiceeval" and question_type != "open":
            metric_extra_info.update({"option": doc.get("option", [])})
        elif metric_func_name == "multichoiceeval" and question_type == "open":
            metric_func = METRIC_REGISTRY.get("meanrelativeacc")

        try:
            result = metric_func(response=prediction, answer=answer, extra_info=metric_extra_info)
            score = result.get("score", 0)
        except Exception as e:
            eval_logger.error(f"Error calculating metric {metric_func_name} for doc {doc.get('session_id', 'N/A')}: {e}")
            score = 0

    # Extract spatree hierarchy tags
    spatree_tags = {k: v for k, v in extra_info.items() if k.startswith("spatree")}

    return {"SpaTreeBench": {"score": score, **spatree_tags}}


def spatialtreebench_aggregate_results(results):
    # Collect scores for each leaf node
    leaf_scores = {}
    for res in results:
        # Find the leaf node in the spatree tags
        spatree_keys = [k for k in res if k.startswith("spatree")]
        if not spatree_keys:
            continue
        leaf_key = max(spatree_keys, key=lambda k: int(k.replace("spatree", "")))
        leaf_name = res.get(leaf_key)

        if leaf_name:
            if leaf_name not in leaf_scores:
                leaf_scores[leaf_name] = []
            leaf_scores[leaf_name].append(res["score"])

    # Calculate average score for each leaf node
    avg_leaf_scores = {name: np.mean(scores) for name, scores in leaf_scores.items()}

    def calculate_node_score(node):
        if not node.children:
            # It's a leaf node
            return avg_leaf_scores.get(node.name.replace(" ", ""), 0)

        weighted_score = 0
        total_weight = 0
        for child in node.children.values():
            child_score = calculate_node_score(child)
            child.score = child_score  # Store score in the hierarchy for printing
            weight = child.weight if child.weight is not None else 1
            weighted_score += child_score * weight
            total_weight += weight

        node_score = weighted_score / total_weight if total_weight > 0 else 0
        return node_score

    # Traverse the hierarchy to calculate all scores
    root_score = calculate_node_score(spatree_hierarchy)
    spatree_hierarchy.score = root_score

    # Print the hierarchical scores
    def print_tree(node, prefix=""):
        score = getattr(node, "score", 0)
        score_str = f"{score:.4f}"
        print(f"{prefix}{node.name} ({score_str})")

        if node.children:
            children_list = list(node.children.values())
            for i, child in enumerate(children_list):
                is_child_last = i == len(children_list) - 1
                connector = "└── " if is_child_last else "├── "
                print_tree(child, prefix=prefix + connector)

    print("\n--- SpaTreeBench Hierarchical Scores ---")
    print_tree(spatree_hierarchy)
    print("----------------------------------------\n")

    # The framework returns the overall score, which is the root score
    return root_score
