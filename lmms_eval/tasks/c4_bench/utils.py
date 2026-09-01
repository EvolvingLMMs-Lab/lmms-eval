from __future__ import annotations

import json
import re
from functools import cache
from typing import Any

DATASET_REPO_ID = "sci-m-wang/C4-Eval"
EXPLANATION_TASKS = {"E0", "E1"}
PUNCT_RE = re.compile(r"[\s\n\r\t，,。.!！?？:：;；、'\"“”‘’`·]+")
ANSWER_PATTERNS = (
    re.compile(r'["\']?answer["\']?\s*[:：]\s*["“”\']?([\u4e00-\u9fff]{4})', re.IGNORECASE),
    re.compile(r"(?:答案|成语)\s*(?:是|为|[:：])\s*[\"“”']?([\u4e00-\u9fff]{4})"),
)


def normalize_answer(text: str) -> str:
    return PUNCT_RE.sub("", text or "").strip()


def strip_code_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def recover_explicit_answer(text: str) -> str:
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text or "")
        if matches:
            return matches[-1]
    lines = [line.strip().strip('"“”') for line in (text or "").splitlines() if line.strip()]
    if lines and re.fullmatch(r"[\u4e00-\u9fff]{4}", lines[-1]):
        return lines[-1]
    return ""


def parse_task_answer(task: str, output: str) -> tuple[str, bool | None]:
    if task in EXPLANATION_TASKS:
        try:
            parsed = json.loads(strip_code_fence(output))
        except (json.JSONDecodeError, TypeError):
            return recover_explicit_answer(output), False
        if not isinstance(parsed, dict):
            return "", False
        return str(parsed.get("answer", "")).strip(), True

    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    answer = lines[0].strip('"“”') if len(lines) == 1 else recover_explicit_answer(output)
    return answer, None


def _accepted_answers(doc: dict[str, Any]) -> set[str]:
    answers = [doc.get("answer", ""), *(doc.get("answer_aliases") or [])]
    return {normalize_answer(str(answer)) for answer in answers if normalize_answer(str(answer))}


@cache
def _download_image(image_path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=DATASET_REPO_ID, filename=image_path, repo_type="dataset")


def c4_doc_to_visual(doc: dict[str, Any]) -> list[Any]:
    from PIL import Image

    image_path = _download_image(str(doc["image_path"]))
    with Image.open(image_path) as image:
        return [image.convert("RGB")]


def c4_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    return str(doc["question"])


def c4_doc_to_messages(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": c4_doc_to_visual(doc)[0]},
                {"type": "text", "text": c4_doc_to_text(doc, lmms_eval_specific_kwargs)},
            ],
        }
    ]


def _filter_task(dataset, task: str):
    return dataset.filter(lambda doc: doc["task"] == task)


def c4_process_docs_h0(dataset):
    return _filter_task(dataset, "H0")


def c4_process_docs_h1(dataset):
    return _filter_task(dataset, "H1")


def c4_process_docs_h4(dataset):
    return _filter_task(dataset, "H4")


def c4_process_docs_e0(dataset):
    return _filter_task(dataset, "E0")


def c4_process_docs_e1(dataset):
    return _filter_task(dataset, "E1")


def c4_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    output = results[0] if results else ""
    task = str(doc["task"])
    answer, valid_json = parse_task_answer(task, output)
    exact = normalize_answer(answer) in _accepted_answers(doc)

    metrics = {"c4_exact_match": float(exact)}
    if task in EXPLANATION_TASKS:
        metrics["c4_json_valid"] = float(bool(valid_json))
    return metrics
