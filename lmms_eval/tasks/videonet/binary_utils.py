from functools import cache
from pathlib import Path

from huggingface_hub import snapshot_download


@cache
def _get_videos_dir() -> Path:
    dataset_root = snapshot_download(repo_id="raivn/VideoNet", repo_type="dataset", allow_patterns=["videos/*.mp4"])
    return Path(dataset_root) / "videos"


def _get_video_path(video_fname) -> str:
    return str(_get_videos_dir() / video_fname)


def videonet_binary_doc_to_visual(doc):
    question = doc["question"]
    video_fnames = [entry["video"] for entry in question if entry["type"] == "video"]
    video_paths = [_get_video_path(video_fname) for video_fname in video_fnames]
    return video_paths


def videonet_binary_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    texts = [entry["text"] for entry in question if entry["type"] == "text"]
    text = "\n".join(texts)
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        text = pre_prompt + text + post_prompt
    return text


def _process_video_entry(entry: dict) -> dict:
    video_fname = entry["video"]
    video_path = _get_video_path(video_fname)
    return {"type": "video", "url": video_path}


def _process_text_entry(entry: dict, lmms_eval_specific_kwargs=None) -> dict:
    text = entry["text"]
    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        text = pre_prompt + text + post_prompt
    return {"type": "text", "text": text}


def _question_to_content(question: list[dict], lmms_eval_specific_kwargs=None) -> list[dict]:
    content = []
    if lmms_eval_specific_kwargs and (pre_prompt := lmms_eval_specific_kwargs.get("pre_prompt")):
        content.append({"type": "text", "text": pre_prompt})
    for entry in question:
        if entry["type"] == "text":
            content.append(entry)
        elif entry["type"] == "video":
            content.append(_process_video_entry(entry))
        else:
            raise Exception("Your copy of the benchmark is corrupted. Please re-download the `benchmarks/` folder from HuggingFace.")
    if lmms_eval_specific_kwargs and (post_prompt := lmms_eval_specific_kwargs.get("post_prompt")):
        content.append({"type": "text", "text": post_prompt})
    return content


def videonet_binary_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    content = _question_to_content(question, lmms_eval_specific_kwargs)
    return [{"role": "user", "content": content}]


def _extract_binary_prediction(text: str) -> str:
    text = text.splitlines()[-1].strip().lower()
    pred = text.replace("*", "").replace("#", "").replace(",", "").replace(".", "").replace(":", "")

    if pred == "yes" or pred == "no":
        return pred
    parts = pred.split(" ")
    if len(parts) > 1:
        first, last = parts[0], parts[-1]
        if first == "yes" or first == "no":
            return first
        elif last == "yes" or last == "no":
            return last
    if "boxed{yes}" in pred:
        return "yes"
    if "boxed{no}" in pred:
        return "no"
    if "the answer is yes" in pred:
        return "yes"
    if "the answer is no" in pred:
        return "no"
    if 'is "yes"' in pred:
        return "yes"
    if "is 'yes'" in pred:
        return "yes"
    if 'is "no"' in pred:
        return "no"
    if "is 'no'" in pred:
        return "no"
    if "does not show" in pred or "does not depict" in pred:
        return "no"
    return pred


def videonet_binary_process_results(doc, results):
    model_output = results[0] if results else ""
    ground_truth = doc["answer"]

    pred = _extract_binary_prediction(model_output)
    correct = 1.0 if pred == ground_truth else 0.0
    return {
        "binary_acc": {
            "question_key": doc["key"],
            "correct": correct,
            "ground_truth": ground_truth,
            "model_prediction": pred,
            "model_output": model_output,
        }
    }


def videonet_binary_aggregate_results(results):
    if not results:
        return 0.0
    num_correct = sum(r["correct"] for r in results)
    total = len(results)
    return num_correct / total
