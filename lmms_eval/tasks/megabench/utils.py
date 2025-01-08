import os
import yaml
from pathlib import Path
from itertools import chain
from ast import literal_eval

from loguru import logger as eval_logger
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.megabench.image_video_utils import read_image, is_video_file, process_text_and_mixed_media


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
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
        process_text_and_mixed_media(doc, lmms_eval_specific_kwargs["max_video_subsample_frame"], cache_dir)

    return medias


def megabench_doc_to_target(doc):
    return doc["answer"]


def megabench_process_results(doc, result):
    response = result[0]  # this is model's raw output
    data_dict = {
        "task_name": doc["task_name"],
        "global_idx": doc["global_idx"],
        "response": response,
        "correct_answer": doc["answer"],
        "eval_context": doc["eval_context"],
        "images": doc["query_images"],
    }

    return {"submission": data_dict}


def megabench_aggregate_results_for_submission(results, args):
    import pdb

    pdb.set_trace()
    pass
