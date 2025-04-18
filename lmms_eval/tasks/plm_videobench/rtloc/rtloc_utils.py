import ast
import os
import re

import numpy as np
from lmms_eval.tasks.plm_videobench.eval_utils import *

# Load default config parameters
config = load_defualt_config()

# Load metadta
metadata_map = load_plm_stc_metadata(config)
assert metadata_map, f"metadata_map is not created. Please double check if you have downloaded the metadata and set the correct path in _default_template_yaml."

# Load video paths
video_base_dir = config["plm_stc"]["video_base_dir"]
assert video_base_dir is not None, f"video_base_dir is not set. Please double check if you have downloaded the videos and set the correct path in _default_template_yaml."

# Load the number of video frames
num_video_frames = config["plm_stc"]["num_video_frames"]
assert num_video_frames is not None, f"num_video_frames must not be None."


def plm_rtloc_doc_to_visual(doc):
    video_id = doc["video"]
    video_path = os.path.join(video_base_dir, video_id)
    bbox_dict_map = metadata_map[(video_id, doc["masklet_id"])]["bbox"]
    bbox_dict_map = {int(k): v for k, v in bbox_dict_map.items()}
    video_frames, sample_pos = load_video(video_path, num_video_frames)
    video_frames = draw_bounding_boxes(video_frames, sample_pos, bbox_dict_map)

    return [video_frames]


def plm_rtloc_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    assert lmms_eval_specific_kwargs and "prompt" in lmms_eval_specific_kwargs, "'prompt' must be specified in lmms_eval_specific_kwargs for the 'plm_rtloc' task."

    caption = doc["caption"]
    prompt = lmms_eval_specific_kwargs["prompt"].format(caption=caption, min_frame_idx=0, max_frame_idx=num_video_frames - 1, num_frames=num_video_frames)

    return prompt


def plm_rtloc_doc_to_target(doc):
    total_frames = doc["total_frames"]
    start_frame = doc["start_frame"]
    end_frame = doc["end_frame"]

    rescale_factor = total_frames / num_video_frames
    rescaled_start_frame = int(start_frame / rescale_factor)
    rescaled_end_frame = int(end_frame / rescale_factor)

    return np.array([[rescaled_start_frame, rescaled_end_frame]])


def plm_rtloc_process_results(doc, results):
    uid = doc["uid"]
    pred = results[0]

    try:
        pred_window = re.findall(r"(\[[0-9]+(?:\.[0-9]+)?,\s*[0-9]+(?:\.[0-9]+)?\])", pred)[0]
        pred_segment = np.array([ast.literal_eval(pred_window)])
        parse_error = 0
    except:
        pred_segment = np.array([[doc["end_frame"] + 10, doc["end_frame"] + 20]])
        parse_error = 1

    gt_segment = plm_rtloc_doc_to_target(doc)

    # Compute detection metrics
    detection_precision, detection_recall, iou_matrices, _ = evaluate_detections(pred_segment, gt_segment, iou_thresholds=(0.3, 0.5, 0.7, 0.9))
    mean_precision = sum(detection_precision) / len(detection_precision)
    mean_recall = sum(detection_recall) / len(detection_recall)
    mIOU = iou_matrices[0, 0]

    results_dict = {"uid": uid, "parse_error": parse_error, "mean_precision": mean_precision, "mean_recall": mean_recall, "mIOU": mIOU}

    return {"plm_rtloc_scores": results_dict}


def plm_rtloc_aggregate_results(results):
    printable_results = {key: round(np.mean([result_dict[key] for result_dict in results]).item(), 4) for key in ["mean_precision", "mean_recall", "mIOU"]}
    printable_results["num_instances"] = len(results)

    return printable_results
