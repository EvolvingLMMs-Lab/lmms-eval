"""Parsing metrics from Eioss/CC-OCR-V2, including the mixed text/table score."""

import re

from Levenshtein import distance
from lxml import etree, html

from lmms_eval.tasks.cc_ocr.evaluators.common import convert_to_halfwidth
from lmms_eval.tasks.cc_ocr.evaluators.ocr_metric import _compute_pr_f1, _text_normalize_and_tokenize
from lmms_eval.tasks.ocrbench_v2.TEDS_metric import TEDS

_LATEX_PREAMBLE = (
    r"\\documentclass\{.*?\}",
    r"\\usepackage\[.*?\]\{.*?\}",
    r"\\usepackage\{.*?\}",
    r"\\geometry\{.*?\}",
    r"\\begin\{document\}",
    r"\\end\{document\}",
    r"\\noindent",
)
_LATEX_META_TAIL = re.compile(
    r"(?is)\n\s*(?:#{1,6}\s*(?:📝\s*)?(?:Notes(?:\s+on\s+Transcription)?\s*:?[^\n]*)"
    r"|📌\s*Notes[^\n]*)(?:\n.*)*\Z"
)
_LATEX_LEADING_FLUFF = re.compile(
    r"(?is)^(?:Here (?:is|’s|'s|are) (?:the )?(?:LaTeX|latex) [^\n]+(?:\n|$)"
    r"|Note that [^\n]+(?:\n|$))+"
)
_HTML_META_TAIL = re.compile(
    r"(?is)\n\s*(?:#{1,6}\s*(?:📝\s*)?Notes[^\n]*|📌\s*Notes[^\n]*"
    r"|✅\s*\*{0,2}Note\*{0,2}\s*:|>{0,1}\s*\*{0,2}Note\*{0,2}\s*:)(?:\n.*)*\Z"
)


def _unfence(text, language):
    marker = "```" + language
    match = re.search(re.escape(marker) + r"(.+?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if marker in text:
        return text.split(marker, 1)[1].split("```", 1)[0].strip()
    return text.strip()


def _edit_similarity(pred, gt):
    # Compiled unit-cost Levenshtein distance for full-page LaTeX strings.
    return 1.0 - distance(pred, gt) / max(len(pred), len(gt))


def _latex(prediction, reference):
    for pattern in _LATEX_PREAMBLE:
        prediction = re.sub(pattern, "", prediction)
    text = _LATEX_META_TAIL.sub("", _unfence(prediction, "latex"))
    start = re.search(r"(?m)^\\(?:documentclass\b|begin\s*\{)", text)
    if start:
        text = text[start.start() :]
    else:
        text = _LATEX_LEADING_FLUFF.sub("", text)
    normalized = text.strip().replace(" ", "").replace("\n", "")
    return _edit_similarity(normalized, reference.replace(" ", "").replace("\n", ""))


def _html_prediction(prediction):
    text = _HTML_META_TAIL.sub("", _unfence(prediction, "html"))
    start = re.search(r"(?i)<\s*(?:!DOCTYPE\b|html\b|body\b|table\b|div\b|h[1-6]\b|p\b|main\b|section\b)", text)
    return text[start.start() :].lstrip() if start else text.lstrip()


def _tables(text):
    def compact_content(match):
        return ">" + match[1].replace("\n", "").replace(" ", "") + "<"

    if "</table>" not in text:
        text += "</table>"
    tables = []
    for table in re.findall(r"<table.*?>.*?</table>", text, re.DOTALL):
        table = re.sub(r"<table.*?>", "<table>", table)
        table = re.sub(r">\s+<", "><", table)
        table = re.sub(r">(.*?)<", compact_content, table, flags=re.DOTALL)
        tables.append(table.replace("\n", "").strip())
    return convert_to_halfwidth("".join(tables))


def _wrapped(table):
    return f"<html><body>{table}</body></html>"


def _teds(prediction, reference):
    return TEDS(structure_only=False, n_jobs=1).evaluate(_wrapped(prediction), _wrapped(reference))


def _board_tokens(text):
    text = re.sub(r"<table\b[^>]*>.*?</table>", "", text, flags=re.IGNORECASE | re.DOTALL)
    if text.strip():
        try:
            parser = html.HTMLParser(encoding="utf-8", remove_comments=True)
            text = html.fromstring(f"<div>{text}</div>", parser=parser).text_content().strip()
        except (ValueError, etree.Error):
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.I | re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[/／|｜]+", " ", convert_to_halfwidth(text))
    return _text_normalize_and_tokenize(re.sub(r"\s+", " ", text).strip(), False, False, False)


def _board_tables(prediction, reference):
    parser = html.HTMLParser(encoding="utf-8", remove_comments=True)

    def parse(text):
        root = html.fromstring(_wrapped(_tables(text)), parser=parser)
        return [etree.tostring(el, encoding="unicode", method="html") for el in root.xpath("body/table")]

    predicted, expected = parse(prediction), parse(reference)
    n = max(len(predicted), len(expected))
    return sum(_teds(p, g) for p, g in zip(predicted, expected, strict=False)) / n if n else 1.0


def score(scenario, reference, prediction):
    prediction = prediction.strip()
    scenario = scenario.lower()
    if "formula" in scenario:
        pred = prediction.replace("\n", " ").replace("```latex", "").replace("```", "")
        pred = pred.replace("\t", " ").replace(" ", "")
        return _edit_similarity(pred, reference.replace(" ", ""))
    if "molecular" in scenario:
        pred = prediction.replace("\n", "").replace(" ", "").replace("<smiles>", "").replace("</smiles>", "")
        return _edit_similarity(pred, reference.replace(" ", ""))
    if "custom" in scenario:
        pred = _html_prediction(prediction)
        expected, predicted = _board_tokens(reference), _board_tokens(pred)
        text_f1 = _compute_pr_f1(expected, predicted)[2] if expected or predicted else 1.0
        return (1.0 - 0.9) * text_f1 + 0.9 * _board_tables(pred, reference)
    if "table" in scenario:
        return _teds(_tables(_html_prediction(prediction)), _tables(reference))
    return _latex(prediction, reference)
