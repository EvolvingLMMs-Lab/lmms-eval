import base64
import io
import os

from PIL import Image

from lmms_eval.tasks.worldqa.utils import (
    MultiChoiceRegexFilter,
    worldq_gen_gpt_eval,
    worldqa_aggregate_gen,
    worldqa_aggregate_mc,
    worldqa_aggregate_mc_eval,
    worldqa_aggregate_mc_ppl,
    worldqa_doc_to_answer,
    worldqa_doc_to_answer_mc,
    worldqa_doc_to_answer_mc_ppl,
    worldqa_doc_to_choice,
    worldqa_doc_to_text,
    worldqa_doc_to_visual,
    worldqa_process_results,
    worldqa_process_results_mc,
)


def worldvqa_doc_to_visual(doc):
    if "image" in doc and doc["image"] is not None:
        image = doc["image"]
        if isinstance(image, Image.Image):
            return [image.convert("RGB")]
        if isinstance(image, str):
            if os.path.exists(image):
                return [Image.open(image).convert("RGB")]
            decoded = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
            return [decoded]
        if isinstance(image, dict):
            image_path = image.get("path")
            if image_path and os.path.exists(image_path):
                return [Image.open(image_path).convert("RGB")]
            image_bytes = image.get("bytes")
            if image_bytes is not None:
                return [Image.open(io.BytesIO(image_bytes)).convert("RGB")]

    video = doc.get("video")
    if isinstance(video, str) and video:
        return [video]
    if isinstance(video, dict):
        video_path = video.get("path")
        if video_path:
            return [video_path]

    try:
        return worldqa_doc_to_visual(doc)
    except SystemExit:
        video_idx = doc.get("video_idx")
        if not video_idx:
            return []
        hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
        return [os.path.join(hf_home, "multi-hop-reasoning", "videos", f"{video_idx}.mp4")]


def worldvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if "option" in doc or "video_idx" in doc:
        return worldqa_doc_to_text(doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)

    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question'].strip()}{post_prompt}"


worldvqa_doc_to_answer = worldqa_doc_to_answer
worldvqa_doc_to_answer_mc = worldqa_doc_to_answer_mc
worldvqa_doc_to_answer_mc_ppl = worldqa_doc_to_answer_mc_ppl
worldvqa_doc_to_choice = worldqa_doc_to_choice
worldvqa_process_results = worldqa_process_results
worldvqa_process_results_mc = worldqa_process_results_mc
worldvqa_aggregate_gen = worldqa_aggregate_gen
worldvqa_aggregate_mc = worldqa_aggregate_mc
worldvqa_aggregate_mc_eval = worldqa_aggregate_mc_eval
worldvqa_aggregate_mc_ppl = worldqa_aggregate_mc_ppl
worldvqa_gen_gpt_eval = worldq_gen_gpt_eval

__all__ = [
    "MultiChoiceRegexFilter",
    "worldvqa_doc_to_visual",
    "worldvqa_doc_to_text",
    "worldvqa_doc_to_answer",
    "worldvqa_doc_to_answer_mc",
    "worldvqa_doc_to_answer_mc_ppl",
    "worldvqa_doc_to_choice",
    "worldvqa_process_results",
    "worldvqa_process_results_mc",
    "worldvqa_aggregate_gen",
    "worldvqa_aggregate_mc",
    "worldvqa_aggregate_mc_eval",
    "worldvqa_aggregate_mc_ppl",
    "worldvqa_gen_gpt_eval",
]
