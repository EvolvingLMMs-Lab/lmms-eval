"""KIE evaluation: field-level F1 (flatten-based) + tree edit distance (Donut).

Ported from CC-OCR kie_evaluator.py, which itself derives from the Donut paper.
"""

import json
from collections import Counter
from typing import Any, Dict, List, Union

import zss
from nltk import edit_distance
from zss import Node

from lmms_eval.tasks.cc_ocr.evaluators.common import (
    kie_normalize_text,
    normalize_values_of_nested_dict,
    post_process_to_json,
)


def _flatten(data: dict) -> List:
    """Turn nested dict/list into a flat list of (dotted_key, value)."""
    out = []

    def rec(value, key=""):
        if isinstance(value, dict):
            for k, v in value.items():
                rec(v, f"{key}.{k}" if key else k)
        elif isinstance(value, list):
            for item in value:
                rec(item, key)
        else:
            out.append((key, value))

    rec(data)
    return out


def _normalize_dict(data: Any) -> Any:
    """Sort keys, wrap scalars in list — matches Donut normalize_dict semantics."""
    if isinstance(data, dict):
        new_data = {}
        for k in sorted(data.keys(), key=lambda k: (len(k), k)):
            value = _normalize_dict(data[k])
            if value:
                if not isinstance(value, list):
                    value = [value]
                new_data[k] = value
        return new_data
    if isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            return [x for x in (_normalize_dict(item) for item in data) if x]
        return [str(item).strip() for item in data if isinstance(item, (str, int, float)) and str(item).strip()]
    return [str(data).strip()]


def _update_cost(a: Node, b: Node) -> int:
    la, lb = a.label, b.label
    a_leaf, b_leaf = "<leaf>" in la, "<leaf>" in lb
    if a_leaf and b_leaf:
        return edit_distance(la.replace("<leaf>", ""), lb.replace("<leaf>", ""))
    if not a_leaf and b_leaf:
        return 1 + len(lb.replace("<leaf>", ""))
    if a_leaf and not b_leaf:
        return 1 + len(la.replace("<leaf>", ""))
    return int(la != lb)


def _insert_remove_cost(node: Node) -> int:
    label = node.label
    if "<leaf>" in label:
        return len(label.replace("<leaf>", ""))
    return 1


def _dict_to_tree(data: Union[Dict, List], node_name: str = None) -> Node:
    if node_name is None:
        node_name = "<root>"
    node = Node(node_name)
    if isinstance(data, dict):
        for k, v in data.items():
            node.addkid(_dict_to_tree(v, k))
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            for item in data:
                node.addkid(_dict_to_tree(item, "<subtree>"))
        else:
            for item in data:
                node.addkid(Node(f"<leaf>{item}"))
    else:
        raise ValueError(f"unexpected node: {data} ({node_name})")
    return node


def _cal_acc(pred: dict, answer: dict) -> float:
    p_tree = _dict_to_tree(_normalize_dict(pred))
    a_tree = _dict_to_tree(_normalize_dict(answer))
    empty_tree = _dict_to_tree(_normalize_dict({}))
    d = zss.distance(
        p_tree,
        a_tree,
        get_children=zss.Node.get_children,
        insert_cost=_insert_remove_cost,
        remove_cost=_insert_remove_cost,
        update_cost=_update_cost,
        return_operations=False,
    )
    denom = zss.distance(
        empty_tree,
        a_tree,
        get_children=zss.Node.get_children,
        insert_cost=_insert_remove_cost,
        remove_cost=_insert_remove_cost,
        update_cost=_update_cost,
        return_operations=False,
    )
    if denom <= 0:
        return 0.0
    return max(0.0, 1 - d / denom)


def _cal_f1_over_samples(preds: Dict[str, Any], answers: Dict[str, Any]) -> float:
    total_tp = 0
    total_fn_fp = 0
    for fname, ans in answers.items():
        pred = preds.get(fname, {})
        pred_flat = _flatten(_normalize_dict(pred))
        ans_flat = _flatten(_normalize_dict(ans))
        for field in pred_flat:
            if field in ans_flat:
                total_tp += 1
                ans_flat.remove(field)
            else:
                total_fn_fp += 1
        total_fn_fp += len(ans_flat)
    return total_tp / (total_tp + total_fn_fp / 2 + 1e-6)


def _cal_acc_over_samples(preds: Dict[str, Any], answers: Dict[str, Any]) -> float:
    accs = []
    for fname, ans in answers.items():
        pred = preds.get(fname, {})
        accs.append(_cal_acc(pred, ans))
    return sum(accs) / (len(accs) + 1e-6)


def _parse_pred_and_gt(sample: Dict) -> (Dict, Dict):
    pred_raw = sample["pred"]
    pred_json = post_process_to_json(pred_raw, file_name=sample.get("image_name"))
    if pred_json is None:
        pred_json = {}

    gt_raw = sample["gt"]
    if isinstance(gt_raw, str):
        try:
            gt_json = json.loads(gt_raw)
        except Exception:
            gt_json = {}
    elif isinstance(gt_raw, list) and gt_raw and isinstance(gt_raw[0], str):
        try:
            gt_json = json.loads(gt_raw[0])
        except Exception:
            gt_json = {}
    else:
        gt_json = gt_raw or {}

    pred_json = normalize_values_of_nested_dict(pred_json, kie_normalize_text)
    gt_json = normalize_values_of_nested_dict(gt_json, kie_normalize_text)
    return pred_json, gt_json


def compute_track(results: List[Dict], key: str) -> float:
    """Aggregate F1 or accuracy over the whole track (mean of per-subset scores)."""
    if not results:
        return 0.0
    by_subset: Dict[str, List[Dict]] = {}
    for r in results:
        by_subset.setdefault(r["subset"], []).append(r)

    subset_scores = {}
    for subset, samples in by_subset.items():
        preds, gts = {}, {}
        for s in samples:
            p, g = _parse_pred_and_gt(s)
            fname = s["image_name"]
            preds[fname] = p
            gts[fname] = g
        if key == "f1_score":
            subset_scores[subset] = _cal_f1_over_samples(preds, gts)
        elif key == "acc":
            subset_scores[subset] = _cal_acc_over_samples(preds, gts)
        else:
            raise ValueError(f"unknown KIE metric key: {key}")

    return sum(subset_scores.values()) / (len(subset_scores) + 1e-9)


def compute_track_with_breakdown(results: List[Dict]) -> Dict:
    """Return per-subset F1 and accuracy for logging."""
    if not results:
        return {"subsets": {}, "f1_score": 0.0, "acc": 0.0}
    by_subset: Dict[str, List[Dict]] = {}
    for r in results:
        by_subset.setdefault(r["subset"], []).append(r)

    subset_info = {}
    for subset, samples in by_subset.items():
        preds, gts = {}, {}
        for s in samples:
            p, g = _parse_pred_and_gt(s)
            fname = s["image_name"]
            preds[fname] = p
            gts[fname] = g
        subset_info[subset] = {
            "f1_score": _cal_f1_over_samples(preds, gts),
            "acc": _cal_acc_over_samples(preds, gts),
            "num": len(samples),
        }
    f1_avg = sum(v["f1_score"] for v in subset_info.values()) / (len(subset_info) + 1e-9)
    acc_avg = sum(v["acc"] for v in subset_info.values()) / (len(subset_info) + 1e-9)
    return {"subsets": subset_info, "f1_score": f1_avg, "acc": acc_avg}
