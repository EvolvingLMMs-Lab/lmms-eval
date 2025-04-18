import json
import os

import numpy as np
from lmms_eval.tasks.plm_videobench.eval_utils import *
from openai import OpenAI

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


def plm_rcap_doc_to_visual(doc):
    video_id = doc["video"]
    video_path = os.path.join(video_base_dir, video_id)
    bbox_dict_map = metadata_map[(video_id, doc["masklet_id"])]["bbox"]
    bbox_dict_map = {int(k): v for k, v in bbox_dict_map.items()}
    video_frames, sample_pos = load_video(video_path, num_video_frames)
    video_frames = draw_bounding_boxes(video_frames, sample_pos, bbox_dict_map)

    return [video_frames]


def plm_rcap_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    assert lmms_eval_specific_kwargs and "prompt" in lmms_eval_specific_kwargs, "'prompt' must be specified in lmms_eval_specific_kwargs for the 'plm_rcap' task."

    start_frame, end_frame, total_frames = doc["start_frame"], doc["end_frame"], doc["total_frames"]
    rescale_factor = total_frames / num_video_frames
    rescaled_start_frame = int(start_frame / rescale_factor)
    rescaled_end_frame = int(end_frame / rescale_factor)

    prompt = lmms_eval_specific_kwargs["prompt"].format(start_frame=rescaled_start_frame, end_frame=rescaled_end_frame, total_frames=total_frames)

    return prompt


def plm_rcap_process_results(doc, results):
    uid = doc["uid"]
    pred_caption = results[0]
    gt_caption = doc["caption"]

    llm_prompt = get_caption_judge_prompt(gt_caption, pred_caption)
    completion = call_judge_with_retry(client, model_name=llm_judge_name, prompt=llm_prompt)
    llm_response = completion.choices[0].message.content
    try:
        judgement = json.loads(llm_response)
        success = 1
    except:
        success = 0
        judgement = {"score": 0, "explanation": "N/A"}

    results_dict = {"uid": uid, "success": success, "llm_juge_score": judgement["score"] / 10, "llm_judge_explanation": judgement["explanation"]}

    return {"plm_rcap_llm_judge_score": results_dict}


def plm_rcap_aggregate_results(results):
    printable_results = {"llm_judge_score": round(np.mean([result_dict["llm_juge_score"] for result_dict in results]).item(), 4), "success_rate": round(np.mean([result_dict["success"] for result_dict in results]).item(), 4)}
    printable_results["num_instances"] = len(results)

    return printable_results
