"""DocAtlas-Bench: multilingual page parsing (https://arxiv.org/abs/2605.12623).

Scoring follows the official script (https://github.com/ahmedheakl/DocAtlas/tree/main/eval),
which uses the OmniDocBench end-to-end pipeline. The parser, the matchers and TEDS are
imported from the MDPBench task, which ships the same OmniDocBench modules.
"""

import json
import threading
from collections import defaultdict

import Levenshtein
from func_timeout import FunctionTimedOut, func_timeout
from loguru import logger as eval_logger

from lmms_eval.tasks.mdpbench.data_preprocess import normalized_table
from lmms_eval.tasks.mdpbench.extract import md_tex_filter
from lmms_eval.tasks.mdpbench.match import match_gt2pred_simple
from lmms_eval.tasks.mdpbench.match_quick import match_gt2pred_quick
from lmms_eval.tasks.mdpbench.teds_metric import TEDS

# Prompt of the official script (the OmniDocBench end-to-end prompt).
PROMPT = r"""You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
- Enclose block formulas with \[ \]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

3. Table Processing:
- Convert tables to HTML format.
- Wrap the entire table with <table> and </table>.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments."""

METRICS = ("docatlas_overall", "docatlas_text_edit", "docatlas_table_teds", "docatlas_reading_order_edit")

# The text matcher falls back to a simpler one after 30 s of wall-clock time, so pages are
# scored one at a time even when lmms-eval post-processes documents in several threads.
_SCORE_LOCK = threading.Lock()


def docatlas_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def docatlas_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    return f"{kwargs.get('pre_prompt', '')}{PROMPT}{kwargs.get('post_prompt', '')}"


def _reading_order_edit(matches):
    matches = [match for match in matches if match["gt_position"] != [""]]
    gt = sorted(position for match in matches for position in match["gt_position"] if position)
    in_pred_order = sorted((match for match in matches if match["pred_position"] != ""), key=lambda match: match["pred_position"])
    pred = [position for match in in_pred_order for position in match["gt_position"]]
    if not gt and not pred:
        return None
    return Levenshtein.distance(gt, pred) / max(len(gt), len(pred))


def _score_page(page, prediction, page_id):
    """Return the text edit distances, the table TEDS scores and the reading-order edit distance of a page."""
    gt = defaultdict(list)
    for item in sorted(page["layout_dets"], key=lambda item: item["order"]):
        gt[item["category_type"]].append(item)
    for block in gt["text_block"]:
        block.setdefault("text", "")  # empty blocks have no "text" key, which the matcher reads

    pred = md_tex_filter(prediction)
    pred_mix = [item for category, items in pred.items() if category not in ("html_table", "latex_table", "md2html_table") for item in items]

    # LaTeX tables are matched when they outnumber the HTML tables, but they score 0.
    table_matches, latex = [], False
    if gt["table"]:
        latex = len(pred["latex_table"]) > len(pred["html_table"])
        table_matches, unmatched = match_gt2pred_simple(gt["table"], pred["latex_table" if latex else "html_table"], "html_table", page_id)
        table_matches = [match for match in table_matches if match["gt_idx"] != [""]]
        pred_mix += unmatched or []  # cells of unmatched tables take part in the text matching

    try:
        text_matches = func_timeout(30, match_gt2pred_quick, args=(gt["text_block"], pred_mix, "text_all", page_id))
    except FunctionTimedOut:
        text_matches, _ = match_gt2pred_simple(gt["text_block"], pred_mix, "text_all", page_id)
    text_matches = [match for match in text_matches if match.get("gt_category_type") == "text_block"]

    text = []
    for match in text_matches:
        gt_text, pred_text = match.get("norm_gt") or match["gt"], match.get("norm_pred") or match["pred"]
        text.append(Levenshtein.distance(pred_text, gt_text) / max(len(pred_text), len(gt_text)))

    tables = []
    for match in table_matches:
        gt_html, pred_html = match["gt"], "" if latex else match["pred"]
        try:
            tables.append(TEDS().evaluate(normalized_table(pred_html) or pred_html, normalized_table(gt_html) or gt_html))
        except Exception as error:
            eval_logger.warning(f"docatlas_bench: TEDS failed on a table of {page_id} ({error}); scored 0.")
            tables.append(0.0)

    return text, tables, _reading_order_edit(text_matches)


def docatlas_process_results(doc, results):
    page = json.loads(doc["annotation"])
    with _SCORE_LOCK:
        text, tables, reading_order = _score_page(page, results[0], doc["id"])
    scores = {"language": page["page_info"]["page_attribute"]["language"], "text": text, "tables": tables, "reading_order": reading_order}
    return {metric: scores for metric in METRICS}


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _by_language(results, key):
    scores = defaultdict(list)
    for result in results:
        scores[result["language"]].extend(result[key])
    return [language_scores for language_scores in scores.values() if language_scores]


def docatlas_aggregate_text_edit(results):
    return _mean(_mean(scores) for scores in _by_language(results, "text"))


def docatlas_aggregate_table_teds(results):
    return _mean(100.0 * _mean(scores) for scores in _by_language(results, "tables"))


def docatlas_aggregate_reading_order_edit(results):
    return _mean(result["reading_order"] for result in results if result["reading_order"] is not None)


def docatlas_aggregate_overall(results):
    """Mean of text accuracy and table TEDS, both macro-averaged over languages."""
    return ((1.0 - docatlas_aggregate_text_edit(results)) * 100.0 + docatlas_aggregate_table_teds(results)) / 2.0
