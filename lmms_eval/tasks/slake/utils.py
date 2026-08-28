import io
import re
import string
import zipfile
from functools import lru_cache
from typing import Any, Dict, Iterable, List

import datasets
import numpy as np
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger
from PIL import Image

SLAKE_REPO_ID = "BoKelvin/SLAKE"
ANSWER_TYPE_OPEN = "OPEN"
ANSWER_TYPE_CLOSED = "CLOSED"
MODALITY_CT = "ct"
MODALITY_MRI = "mri"
MODALITY_XRAY = "x-ray"
CONTENT_ABNORMALITY = "abnormality"
CONTENT_COLOR = "color"
CONTENT_KG = "kg"
CONTENT_MODALITY = "modality"
CONTENT_ORGAN = "organ"
CONTENT_PLANE = "plane"
CONTENT_POSITION = "position"
CONTENT_QUANTITY = "quantity"
CONTENT_SHAPE = "shape"
CONTENT_SIZE = "size"

EN_YES = {"yes", "yeah", "yep", "true"}
EN_NO = {"no", "nope", "false"}
ZH_YES = {"是", "是的", "有", "包含", "可以", "能", "会", "正常"}
ZH_NO = {"否", "不是", "没有", "无", "不包含", "不能", "不会", "不可以", "不正常", "异常"}
ZH_PUNCTUATION = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～、，。；：《》？【】（）"


def slake_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Add stable sample ids without language filtering."""
    return _process_docs(dataset, language=None)


def slake_process_docs_en(dataset: datasets.Dataset) -> datasets.Dataset:
    """Keep English SLAKE samples and add stable sample ids."""
    return _process_docs(dataset, language="en")


def slake_process_docs_zh(dataset: datasets.Dataset) -> datasets.Dataset:
    """Keep Chinese SLAKE samples and add stable sample ids."""
    return _process_docs(dataset, language="zh")


def _process_docs(dataset: datasets.Dataset, language: str | None) -> datasets.Dataset:
    if language is not None:
        dataset = dataset.filter(lambda doc: _normalize_language(doc.get("q_lang")) == language)

    def _add_id(doc: Dict[str, Any], idx: int) -> Dict[str, Any]:
        doc = dict(doc)
        doc["id"] = str(doc.get("qid") or f"slake_{idx}")
        return doc

    return dataset.map(_add_id, with_indices=True)


def slake_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    """Load the SLAKE image referenced by a dataset row."""
    image = doc.get("image")
    if isinstance(image, Image.Image):
        return [image.convert("RGB")]
    return [_open_slake_image(doc["img_name"])]


def slake_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    """Build the text prompt for a SLAKE question."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['question'].strip()}{post_prompt}"


def slake_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> List[Dict[str, Any]]:
    """Build chat-format multimodal messages for chat model backends."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": slake_doc_to_visual(doc)[0]},
                {"type": "text", "text": slake_doc_to_text(doc, lmms_eval_specific_kwargs)},
            ],
        }
    ]


def slake_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict[str, Any]]:
    """Normalize a generated answer and return per-sample SLAKE metric records."""
    prediction = results[0] if results else ""
    target = doc["answer"]
    pred_norm = normalize_answer(prediction)
    target_norm = normalize_answer(target)
    is_correct = pred_norm == target_norm
    record = {
        "id": str(doc.get("qid") or doc.get("id") or ""),
        "prediction": prediction,
        "prediction_normalized": pred_norm,
        "target": target,
        "target_normalized": target_norm,
        "language": _normalize_language(doc.get("q_lang")),
        "answer_type": _normalize_answer_type(doc.get("answer_type")),
        "modality": _normalize_group_value(doc.get("modality")),
        "content_type": _normalize_group_value(doc.get("content_type")),
        "is_correct": is_correct,
    }
    return {metric: record for metric in SLAKE_METRICS}


def normalize_answer(answer: Any) -> str:
    """Normalize English and Chinese short answers for exact-match scoring."""
    text = "" if answer is None else str(answer)
    text = text.strip().lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"（[^）]*）", " ", text)
    punctuation_table = str.maketrans({char: " " for char in string.punctuation + ZH_PUNCTUATION})
    text = text.translate(punctuation_table)
    text = re.sub(r"\s+", " ", text).strip()
    text = _normalize_medical_aliases(text)
    text = _normalize_yes_no(text)
    return text


def slake_aggregate_accuracy(results: List[Dict[str, Any]]) -> float:
    """Aggregate overall SLAKE accuracy and log subgroup breakdowns."""
    _log_breakdown("language", results)
    _log_breakdown("answer_type", results)
    _log_breakdown("modality", results)
    _log_breakdown("content_type", results)
    return _accuracy(results)


def slake_aggregate_open_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "answer_type", ANSWER_TYPE_OPEN.lower())


def slake_aggregate_closed_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "answer_type", ANSWER_TYPE_CLOSED.lower())


def slake_aggregate_ct_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "modality", MODALITY_CT)


def slake_aggregate_mri_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "modality", MODALITY_MRI)


def slake_aggregate_xray_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "modality", MODALITY_XRAY)


def slake_aggregate_abnormality_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "content_type", CONTENT_ABNORMALITY)


def slake_aggregate_color_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "content_type", CONTENT_COLOR)


def slake_aggregate_kg_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "content_type", CONTENT_KG)


def slake_aggregate_modality_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "content_type", CONTENT_MODALITY)


def slake_aggregate_organ_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "content_type", CONTENT_ORGAN)


def slake_aggregate_plane_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "content_type", CONTENT_PLANE)


def slake_aggregate_position_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "content_type", CONTENT_POSITION)


def slake_aggregate_quantity_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "content_type", CONTENT_QUANTITY)


def slake_aggregate_shape_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "content_type", CONTENT_SHAPE)


def slake_aggregate_size_accuracy(results: List[Dict[str, Any]]) -> float:
    return _accuracy_for(results, "content_type", CONTENT_SIZE)


SLAKE_METRICS = (
    "slake_accuracy",
    "slake_open_accuracy",
    "slake_closed_accuracy",
    "slake_ct_accuracy",
    "slake_mri_accuracy",
    "slake_xray_accuracy",
    "slake_abnormality_accuracy",
    "slake_color_accuracy",
    "slake_kg_accuracy",
    "slake_modality_accuracy",
    "slake_organ_accuracy",
    "slake_plane_accuracy",
    "slake_position_accuracy",
    "slake_quantity_accuracy",
    "slake_shape_accuracy",
    "slake_size_accuracy",
)


def _accuracy(results: Iterable[Dict[str, Any]]) -> float:
    results = list(results)
    if len(results) == 0:
        return 0.0
    return float(np.mean([sample["is_correct"] for sample in results]))


def _accuracy_for(results: List[Dict[str, Any]], key: str, value: str) -> float:
    filtered = [sample for sample in results if sample.get(key) == value]
    return _accuracy(filtered)


def _normalize_yes_no(text: str) -> str:
    if text in EN_YES or text in ZH_YES:
        return "yes"
    if text in EN_NO or text in ZH_NO:
        return "no"
    return text


def _normalize_medical_aliases(text: str) -> str:
    if text in {"x ray", "xray"}:
        return "xray"
    return text


def _normalize_language(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_answer_type(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_group_value(value: Any) -> str:
    return str(value or "").strip().lower()


def _log_breakdown(key: str, results: List[Dict[str, Any]]) -> None:
    groups: Dict[str, List[bool]] = {}
    for sample in results:
        groups.setdefault(sample.get(key) or "unknown", []).append(sample["is_correct"])

    eval_logger.info(f"SLAKE {key} breakdown:")
    for name in sorted(groups):
        scores = groups[name]
        eval_logger.info(f"  {name}: {float(np.mean(scores)):.4f} ({sum(scores)}/{len(scores)})")


def _open_slake_image(image_name: str) -> Image.Image:
    image_name = _resolve_zip_member(image_name)
    with zipfile.ZipFile(_slake_image_zip_path()) as archive:
        with archive.open(image_name) as image_file:
            return Image.open(io.BytesIO(image_file.read())).convert("RGB")


def _resolve_zip_member(image_name: str) -> str:
    image_name = str(image_name).lstrip("/")
    with zipfile.ZipFile(_slake_image_zip_path()) as archive:
        if image_name in archive.namelist():
            return image_name
        prefixed_name = f"imgs/{image_name}"
        if prefixed_name in archive.namelist():
            return prefixed_name
    return image_name


@lru_cache(maxsize=1)
def _slake_image_zip_path() -> str:
    root = snapshot_download(
        repo_id=SLAKE_REPO_ID,
        repo_type="dataset",
        revision="main",
        allow_patterns=["imgs.zip"],
    )
    return f"{root}/imgs.zip"
