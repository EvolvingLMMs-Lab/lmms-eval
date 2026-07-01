"""Document parsing metrics for CC-OCR.

Four operations (identified by ``result["op"]`` derived from HF ``l2-category``):

    doc        -> 1 - edit_distance / max_len  (LaTeX transcription)
    table      -> TEDS on HTML                 (reuses ocrbench_v2 TEDS_metric)
    formula    -> 1 - edit_distance / max_len  (LaTeX formula)
    molecular  -> 1 - edit_distance / max_len  (SMILES)
"""

import re
from typing import Dict, List

import nltk

from lmms_eval.tasks.cc_ocr.evaluators.common import convert_to_halfwidth

# Patterns to strip from doc predictions before scoring
_DOC_PATTERNS = [
    r"\\documentclass\{.*?\}",
    r"\\usepackage\[.*?\]\{.*?\}",
    r"\\usepackage\{.*?\}",
    r"\\geometry\{.*?\}",
    r"\\begin\{document\}",
    r"\\end\{document\}",
    r"\\noindent",
]


def _extract_and_clean_tables(text: str) -> str:
    if "</table>" not in text:
        text = text + "</table>"
    tables = re.findall(r"<table.*?>.*?</table>", text, re.DOTALL)
    cleaned = []
    for t in tables:
        t = re.sub(r"<table.*?>", "<table>", t)
        t = re.sub(r">\s+<", "><", t)
        t = re.sub(
            r">(.*?)<",
            lambda m: ">" + m.group(1).replace("\n", "").replace(" ", "") + "<",
            t,
            flags=re.DOTALL,
        )
        cleaned.append(t.replace("\n", "").strip())
    return "".join(cleaned)


def _eval_doc(pred: str, gt: str) -> float:
    for p in _DOC_PATTERNS:
        pred = re.sub(p, "", pred)
    m = re.search(r"```latex(.+?)```", pred, re.DOTALL)
    if m is not None:
        pred = m.group(1)
    elif "```latex" in pred:
        pred = pred.split("```latex", 1)[1]
    pred = pred.replace(" ", "").replace("\n", "")
    gt = gt.replace(" ", "").replace("\n", "")
    if not pred and not gt:
        return 1.0
    denom = max(len(pred), len(gt))
    if denom == 0:
        return 0.0
    return 1 - nltk.edit_distance(pred, gt) / denom


def _eval_formula(pred: str, gt: str) -> float:
    pred = pred.replace("\n", " ").replace("```latex", "").replace("```", "").replace("\t", " ").replace(" ", "")
    gt = gt.replace(" ", "")
    denom = max(len(pred), len(gt))
    if denom == 0:
        return 0.0
    return 1 - nltk.edit_distance(pred, gt) / denom


def _eval_molecular(pred: str, gt: str) -> float:
    pred = pred.replace("\n", "").replace(" ", "").replace("<smiles>", "").replace("</smiles>", "")
    gt = gt.replace(" ", "")
    denom = max(len(pred), len(gt))
    if denom == 0:
        return 0.0
    return 1 - nltk.edit_distance(pred, gt) / denom


def _eval_table(pred: str, gt: str) -> float:
    from lmms_eval.tasks.ocrbench_v2.TEDS_metric import TEDS

    m = re.search(r"```html(.+?)```", pred, re.DOTALL)
    if m is not None:
        pred = m.group(1)
    elif "```html" in pred:
        pred = pred.split("```html", 1)[1]

    pred = _extract_and_clean_tables(pred)
    pred = convert_to_halfwidth(pred)
    gt = _extract_and_clean_tables(gt)
    gt = convert_to_halfwidth(gt)

    pred_html = f"<html><body>{pred}</body></html>"
    gt_html = f"<html><body>{gt}</body></html>"
    teds = TEDS(structure_only=False, n_jobs=1)
    try:
        return teds.evaluate(pred_html, gt_html)
    except Exception:
        return 0.0


_OP_TO_FUNC = {
    "doc": _eval_doc,
    "table": _eval_table,
    "formula": _eval_formula,
    "molecular": _eval_molecular,
}


def _filter_by_op(results: List[Dict], op: str) -> List[Dict]:
    return [r for r in results if r.get("op") == op]


def _compute_op_score(results: List[Dict], op: str) -> float:
    """Average score across every subset of the given op."""
    subset_samples: Dict[str, List[Dict]] = {}
    for r in _filter_by_op(results, op):
        subset_samples.setdefault(r["subset"], []).append(r)
    if not subset_samples:
        return 0.0
    func = _OP_TO_FUNC[op]
    subset_scores = {}
    for subset, samples in subset_samples.items():
        scores = []
        for s in samples:
            try:
                scores.append(func(str(s["pred"]), str(s["gt"])))
            except Exception:
                scores.append(0.0)
        subset_scores[subset] = sum(scores) / (len(scores) + 1e-9)
    return sum(subset_scores.values()) / (len(subset_scores) + 1e-9)


def compute_op(results: List[Dict], op: str) -> float:
    return _compute_op_score(results, op)


def compute_overall(results: List[Dict]) -> float:
    """Average of the four op scores (ignoring absent ops)."""
    op_scores = []
    for op in ("doc", "table", "formula", "molecular"):
        if any(r.get("op") == op for r in results):
            op_scores.append(_compute_op_score(results, op))
    if not op_scores:
        return 0.0
    return sum(op_scores) / len(op_scores)


def compute_track_with_breakdown(results: List[Dict]) -> Dict:
    """Return per-subset and per-op scores for logging."""
    per_subset: Dict[str, float] = {}
    per_op: Dict[str, float] = {}
    for op in ("doc", "table", "formula", "molecular"):
        subset_samples: Dict[str, List[Dict]] = {}
        for r in _filter_by_op(results, op):
            subset_samples.setdefault(r["subset"], []).append(r)
        if not subset_samples:
            continue
        func = _OP_TO_FUNC[op]
        op_subset_scores = []
        for subset, samples in subset_samples.items():
            scores = []
            for s in samples:
                try:
                    scores.append(func(str(s["pred"]), str(s["gt"])))
                except Exception:
                    scores.append(0.0)
            subset_score = sum(scores) / (len(scores) + 1e-9)
            per_subset[subset] = subset_score
            op_subset_scores.append(subset_score)
        per_op[op] = sum(op_subset_scores) / (len(op_subset_scores) + 1e-9)
    overall = sum(per_op.values()) / (len(per_op) + 1e-9) if per_op else 0.0
    return {"subsets": per_subset, "ops": per_op, "overall_score": overall}
