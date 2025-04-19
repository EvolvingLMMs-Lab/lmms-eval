import json
import os

import numpy as np
from openai import OpenAI

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

# Initialize LLM Judge for RCap Evaluation
base_url = config["llm_judge"]["base_url"]
api_key = config["llm_judge"]["api_key"]
client = OpenAI(api_key=api_key, base_url=base_url)
llm_judge_name = config["llm_judge"]["model"]


def plm_rdcap_doc_to_visual(doc):
    video_id = doc["video"]
    video_path = os.path.join(video_base_dir, video_id)
    bbox_dict_map = metadata_map[(video_id, doc["masklet_id"])]["bbox"]
    bbox_dict_map = {int(k): v for k, v in bbox_dict_map.items()}
    video_frames, sample_pos = load_video(video_path, num_video_frames)
    video_frames = draw_bounding_boxes(video_frames, sample_pos, bbox_dict_map)

    return [video_frames]


def plm_rdcap_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    assert lmms_eval_specific_kwargs and "prompt" in lmms_eval_specific_kwargs, "'prompt' must be specified in lmms_eval_specific_kwargs for the 'plm_rdcap' task."

    prompt = lmms_eval_specific_kwargs["prompt"].format(start_frame=0, end_frame=num_video_frames - 1, total_frames=num_video_frames)

    return prompt


def plm_rdcap_process_results(doc, results):
    uid = doc["uid"]
    pred_segments, pred_captions = extract_delta_segments(results[0])

    gt_dense_captions = doc["dense_captions"]
    rescale_factor = doc["total_frames"] / num_video_frames
    gt_segments = np.array([[int(entry["start_frame"] / rescale_factor), int(entry["end_frame"] / rescale_factor)] for entry in gt_dense_captions])
    gt_captions = [entry["caption"] for entry in gt_dense_captions]

    if len(pred_segments) == 0:
        # Parsing error or the model did not predict any segments. We penalize the model by assigning a score of zero.
        results_dict = {"uid": uid, "SODA_c": 0.0}
        return {"plm_rdcap_score": results_dict}

    # Pair up every GT caption with every predicted caption and run LLM judge
    iou_thresholds = (0.3, 0.5, 0.7, 0.9)
    scores = []
    for gt_caption in gt_captions:
        for pred_caption in pred_captions:
            llm_prompt = get_caption_judge_prompt(gt_caption, pred_caption)
            completion = call_judge_with_retry(client, model_name=llm_judge_name, prompt=llm_prompt)
            llm_response = completion.choices[0].message.content
            # Parse LLM judge outputs
            try:
                judgement = json.loads(llm_response)
            except:
                judgement = {"score": 0, "explanation": "N/A"}
            score = judgement["score"] / 10
            scores.append(score)
    # Create pairwise score matrix
    score_matrix = np.array(scores).reshape(len(gt_captions), len(pred_captions))
    # Compute SODA metric (Fujita et al., ECCV 2020)
    _, _, iou_matrices, _ = evaluate_detections(pred_segments, gt_segments, iou_thresholds=iou_thresholds)
    SODA_c = sodac_llm_score(iou_matrices, score_matrix, pred_captions, gt_captions, (0.0,))

    results_dict = {"uid": uid, "SODA_c": SODA_c}

    return {"plm_rdcap_score": results_dict}


def plm_rdcap_aggregate_results(results):
    printable_results = {"SODA_c": round(np.mean([result_dict["SODA_c"] for result_dict in results]).item(), 4)}
    printable_results["num_instances"] = len(results)

    return printable_results
