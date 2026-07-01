"""CC-OCR task utilities for lmms-eval.

Provides:
- Doc-to-visual / doc-to-text / doc-to-target hooks
- process_results: packages one payload used by every metric in every track
- One aggregator per metric declared in a track yaml

The dataset ``wulipc/CC-OCR`` has four configs (multi_scene_ocr, multi_lan_ocr,
doc_parsing, kie), each with a single ``test`` split. Each row carries a
``split`` field naming its original subset (39 in total) and an
``l2-category`` field naming the op group (used by doc_parsing).
"""

import base64
import io
import json
from typing import Any, Dict, Optional

from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

from lmms_eval.tasks.cc_ocr.evaluators import doc_parsing_metric, kie_metric, ocr_metric


def _decode_image(image_obj: Any) -> Optional[Image.Image]:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, bytes):
        return Image.open(io.BytesIO(image_obj)).convert("RGB")
    if isinstance(image_obj, str):
        try:
            raw = base64.b64decode(image_obj)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None
    return None


def cc_ocr_doc_to_visual(doc):
    img = _decode_image(doc.get("image"))
    if img is None:
        eval_logger.warning(f"cc_ocr: failed to decode image for {doc.get('image_name')}")
        return []
    return [img]


def cc_ocr_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["question"]


def cc_ocr_doc_to_target(doc):
    ans = doc.get("answer", "")
    if isinstance(ans, list):
        return ans[0] if ans else ""
    return ans


def _resolve_track(l2_category: str) -> str:
    l2 = str(l2_category or "").strip()
    if l2 in {"doc", "table", "formula", "molecular"}:
        return "doc_parsing"
    if l2 in {"constrained_category", "open_category"}:
        return "kie"
    if l2.endswith("_text"):
        return "multi_scene_ocr"
    return "multi_lan_ocr"


def _resolve_op(track: str, l2_category: str) -> str:
    """For doc_parsing, l2-category is the op label; empty for other tracks."""
    if track == "doc_parsing":
        return str(l2_category or "").strip()
    return ""


_METRIC_KEYS_BY_TRACK = {
    "multi_scene_ocr": ("macro_f1", "micro_f1"),
    "multi_lan_ocr": ("macro_f1", "micro_f1"),
    "doc_parsing": ("doc_score", "table_teds", "formula_score", "molecular_score", "overall_score"),
    "kie": ("f1_score", "acc"),
}


def cc_ocr_process_results(doc, results):
    pred = results[0] if results else ""
    l2_category = doc.get("l2-category", "")
    track = _resolve_track(l2_category)
    payload = {
        "pred": pred,
        "gt": doc.get("answer", ""),
        "image_name": doc.get("image_name", ""),
        "subset": doc.get("split", ""),
        "l2_category": l2_category,
        "op": _resolve_op(track, l2_category),
        "track": track,
    }
    keys = _METRIC_KEYS_BY_TRACK.get(track, ())
    return {key: payload for key in keys}


def _write_submission(track: str, breakdown: Dict, args) -> None:
    """Write per-subset breakdown as JSON alongside the aggregated metric."""
    if args is None:
        return
    try:
        path = generate_submission_file(f"cc_ocr_{track}_breakdown.json", args, subpath="results")
        with open(path, "w") as f:
            json.dump(breakdown, f, ensure_ascii=False, indent=2)
        eval_logger.info(f"cc_ocr[{track}] breakdown written to {path}")
    except Exception as e:
        eval_logger.warning(f"cc_ocr[{track}] failed to write breakdown: {e}")


def cc_ocr_agg_ocr_macro_f1(results, args):
    breakdown = ocr_metric.compute_track_with_breakdown(results)
    _write_submission(results[0]["track"] if results else "ocr", breakdown, args)
    return breakdown["macro_f1"]


def cc_ocr_agg_ocr_micro_f1(results, args):
    return ocr_metric.compute_track(results, key="micro_f1")


def cc_ocr_agg_dp_doc(results, args):
    return doc_parsing_metric.compute_op(results, op="doc")


def cc_ocr_agg_dp_table(results, args):
    return doc_parsing_metric.compute_op(results, op="table")


def cc_ocr_agg_dp_formula(results, args):
    return doc_parsing_metric.compute_op(results, op="formula")


def cc_ocr_agg_dp_molecular(results, args):
    return doc_parsing_metric.compute_op(results, op="molecular")


def cc_ocr_agg_dp_overall(results, args):
    breakdown = doc_parsing_metric.compute_track_with_breakdown(results)
    _write_submission("doc_parsing", breakdown, args)
    return breakdown["overall_score"]


def cc_ocr_agg_kie_f1(results, args):
    breakdown = kie_metric.compute_track_with_breakdown(results)
    _write_submission("kie", breakdown, args)
    return breakdown["f1_score"]


def cc_ocr_agg_kie_acc(results, args):
    return kie_metric.compute_track(results, key="acc")
