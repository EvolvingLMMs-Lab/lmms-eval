import json
import os
from ast import literal_eval
from collections import defaultdict
from itertools import chain
from pathlib import Path

import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.megabench.image_video_utils import is_video_file, process_text_and_mixed_media, read_image

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "_default_template_yaml", "r", encoding="utf-8") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def _check_media_type(doc, return_media=False):
    media_types = ["global_media", "example_media", "query_media"]
    all_medias = list(chain.from_iterable(literal_eval(doc[media_type]) for media_type in media_types))
    is_video = [is_video_file(file) for file in all_medias]
    if not any(is_video):
        media_type = "image"
    elif all(is_video):
        media_type = "video"
    else:
        media_type = "mixed"
    if return_media:
        return media_type, all_medias
    return media_type


def megabench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    media_type = _check_media_type(doc)
    if media_type in ["image", "video"]:
        prompt_components = [doc["task_description"], doc["example_text"], doc["query_text"]]
        prompt = "\n".join(prompt_components)
    else:
        # mixed video and image input, convert video to image frames,
        # and adjust the image placeholders accordingly.
        cache_dir = os.path.join(base_cache_dir, cache_name)
        prompt, images = process_text_and_mixed_media(doc, lmms_eval_specific_kwargs["max_video_subsample_frame"], cache_dir)
    return prompt


def megabench_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    cache_dir = Path(base_cache_dir) / cache_name
    media_type, all_media = _check_media_type(doc, return_media=True)
    all_media_paths = [str(cache_dir / local_path) for local_path in all_media]
    if media_type == "image":
        medias = [read_image(image_path) for image_path in all_media_paths]
    elif media_type == "video":
        # all videos, only return the list of video paths
        medias = all_media_paths
    else:  # mixed video and image input, convert video to image frames
        cache_dir = os.path.join(base_cache_dir, cache_name)
        _, medias = process_text_and_mixed_media(doc, lmms_eval_specific_kwargs["max_video_subsample_frame"], cache_dir)

    return medias


def megabench_doc_to_target(doc):
    return doc["answer"]


def megabench_process_results(doc, result):
    response = result[0]  # this is model's raw output
    # Follow the response format in original megabench eval results
    data_dict = {
        "task_name": doc["task_name"],
        "global_idx": doc["id"],
        "eval_context": literal_eval(doc["eval_context"]),
        "images": literal_eval(doc["query_media"]),
        "query_text": doc["query_text"],
        "global_images": literal_eval(doc["global_media"]),
        "global_description": doc["task_description"],
        "example_info": {
            "image_paths": literal_eval(doc["example_media"]),
            "example_text": doc["example_text"],
        },
        "correct_answer": literal_eval(doc["answer"]),
        "response": response,
    }

    return {"submission": data_dict}


def megabench_aggregate_results_for_submission(results, args):
    results_by_task = defaultdict(list)
    for result in results:
        results_by_task[result["task_name"]].append(result)
    submission_results = []
    task_level_keys = ["task_name", "global_images", "global_description", "example_info"]
    sample_level_keys = ["response", "correct_answer", "global_idx", "images", "query_text"]
    for per_task_results in results_by_task.values():
        task_result = {key: per_task_results[0][key] for key in task_level_keys}
        all_query_response = []
        for sample in per_task_results:
            sample_response = {key: sample[key] for key in sample_level_keys}
            all_query_response.append(sample_response)
        task_result["query_response"] = all_query_response
        submission_results.append(task_result)

    submission_path = generate_submission_file(f"{args.tasks}_all_query_responses.json", args)
    with open(submission_path, "w", encoding="utf-8") as fd:
        json.dump(submission_results, fd, indent=4)
    eval_logger.info(f"Results saved to {submission_path}.")
