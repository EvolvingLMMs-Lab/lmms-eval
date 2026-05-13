import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download
from PIL import Image

VPCT_DATASET_REPO = "camelCase12/vpct-1"

_ANSWER_FORMAT_PATTERN = re.compile(r"answer\s*\(\s*([1-3])\s*\)", flags=re.IGNORECASE)
_BUCKET_PATTERN = re.compile(r"bucket\s*([1-3])", flags=re.IGNORECASE)
_STANDALONE_DIGIT_PATTERN = re.compile(r"(?<!\d)([1-3])(?!\d)")


def vpct_process_docs(dataset):
    def _add_sim_id(doc: Dict[str, Any]) -> Dict[str, int]:
        return {"sim_id": int(doc["simId"])}

    return dataset.map(_add_sim_id).sort("sim_id")


def _doc_sim_id(doc: Dict[str, Any]) -> int:
    return int(doc.get("sim_id", doc["simId"]))


@lru_cache(maxsize=128)
def _download_vpct_image(sim_id: int) -> str:
    image_filename = f"sim_{sim_id}_initial.png"
    return hf_hub_download(repo_id=VPCT_DATASET_REPO, repo_type="dataset", filename=image_filename, token=False)


def vpct_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    sim_id = _doc_sim_id(doc)
    image_path = _download_vpct_image(sim_id)
    with Image.open(image_path) as image:
        return [image.convert("RGB")]


def vpct_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")

    question = "You are an expert physics simulator. Looking at this image of a ball-and-bucket simulation, " "predict which bucket (numbered 1, 2, or 3 from left to right) the ball will eventually fall into."
    return f"{pre_prompt}{question}{post_prompt}"


def vpct_doc_to_target(doc: Dict[str, Any]) -> str:
    return str(int(doc["finalBucket"]))


def _extract_bucket(response: str) -> Optional[int]:
    text = str(response).strip()
    for pattern in (_ANSWER_FORMAT_PATTERN, _BUCKET_PATTERN, _STANDALONE_DIGIT_PATTERN):
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def vpct_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    prediction = str(results[0]).strip() if results else ""
    predicted_bucket = _extract_bucket(prediction)
    target_bucket = int(doc["finalBucket"])

    result = {
        "sim_id": _doc_sim_id(doc),
        "prediction": prediction,
        "predicted_bucket": predicted_bucket,
        "target_bucket": target_bucket,
        "is_correct": predicted_bucket == target_bucket,
    }

    return {
        "vpct_accuracy": result,
        "vpct_answered_rate": result,
    }


def vpct_aggregate_accuracy(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(1.0 if result["is_correct"] else 0.0 for result in results) / len(results)


def vpct_aggregate_answered_rate(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(1.0 if result["predicted_bucket"] is not None else 0.0 for result in results) / len(results)
