import json
from argparse import Namespace
from typing import Any

from loguru import logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

EVALUATOR_URL = "https://huggingface.co/spaces/paulgavrikov/visualoverload-submit"


def visualoverload_doc_to_visual(doc: dict[str, Any]) -> list:
    """Return the full-resolution image without resizing or cropping."""
    return [doc["image"].convert("RGB")]


def visualoverload_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict | None = None) -> str:
    """Use the benchmark's official prompt for all question types."""
    return doc["default_prompt"]


def visualoverload_doc_to_messages(doc: dict[str, Any], lmms_eval_specific_kwargs: dict | None = None) -> list[dict]:
    """Provide the same image and prompt to chat-template model backends."""
    content = [{"type": "image", "url": image} for image in visualoverload_doc_to_visual(doc)]
    content.append({"type": "text", "text": visualoverload_doc_to_text(doc)})
    return [{"role": "user", "content": content}]


def visualoverload_doc_to_target(doc: dict[str, Any]) -> str:
    """Answers are private; no local target is available."""
    return ""


def visualoverload_process_results(doc: dict[str, Any], results: list[str]) -> dict:
    """Collect raw responses; official answer extraction happens on the server."""
    return {"submission": {"question_id": doc["question_id"], "response": results[0]}}


def visualoverload_aggregate_results(results: list[dict], args: Namespace | None) -> None:
    """Write an official submission without manufacturing a local accuracy."""
    question_ids = [result["question_id"] for result in results]
    if any(question_id is None for question_id in question_ids) or len(set(question_ids)) != len(question_ids):
        raise ValueError("VisualOverload submissions require unique, non-null question_id values.")
    path = generate_submission_file("visualoverload_submission.json", args)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    logger.info(f"VisualOverload answers are private. Submit {path} to {EVALUATOR_URL}")
