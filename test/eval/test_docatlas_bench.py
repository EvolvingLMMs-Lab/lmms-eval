import json

import pytest

for _module in ("bs4", "func_timeout", "lxml", "pylatexenc", "scipy"):
    pytest.importorskip(_module)

from lmms_eval.tasks.docatlas_bench import utils  # noqa: E402

TABLE = "<table><tr><td>Year</td><td>Pages</td></tr><tr><td>2025</td><td>5575</td></tr></table>"
FIRST = "DocAtlas covers eighty two languages with model free annotations."
SECOND = "Pages are converted to Markdown and scored against the ground truth."


def _score(prediction):
    attribute = {"text_language": "text_english"}
    layout_dets = [
        {"category_type": "section_header", "order": 0, "anno_id": 0, "text": "Benchmark", "attribute": attribute},
        {"category_type": "text_block", "order": 1, "anno_id": 1, "text": FIRST, "attribute": attribute},
        {"category_type": "text_block", "order": 2, "anno_id": 2, "text": SECOND, "attribute": attribute},
        {"category_type": "text_block", "order": 3, "anno_id": 3, "attribute": attribute},  # empty block, as in the dataset
        {"category_type": "table", "order": 4, "anno_id": 4, "html": TABLE, "attribute": {}},
    ]
    page = {"page_info": {"page_attribute": {"language": "english"}}, "layout_dets": layout_dets}
    doc = {"id": "page_0001", "annotation": json.dumps(page)}
    return utils.docatlas_process_results(doc, [prediction])["docatlas_overall"]


def test_exact_prediction():
    scores = _score(f"# Benchmark\n\n{FIRST}\n\n{SECOND}\n\n{TABLE}\n")

    assert scores == {"language": "english", "text": [0.0, 0.0], "tables": [pytest.approx(1.0)], "reading_order": 0.0}
    assert utils.docatlas_aggregate_overall([scores]) == pytest.approx(100.0)


def test_empty_prediction():
    scores = _score("")

    assert scores == {"language": "english", "text": [1.0, 1.0], "tables": [0.0], "reading_order": 1.0}
    assert utils.docatlas_aggregate_overall([scores]) == 0.0


def test_pipe_tables_are_scored_and_latex_tables_are_not():
    pipe_table = "| Year | Pages |\n|---|---|\n| 2025 | 5575 |\n"
    latex_table = "\\begin{tabular}{ll}\nYear & Pages \\\\\n2025 & 5575 \\\\\n\\end{tabular}\n"

    assert _score(f"{FIRST}\n\n{SECOND}\n\n{pipe_table}")["tables"][0] > 0.5
    assert _score(f"{FIRST}\n\n{SECOND}\n\n{latex_table}")["tables"] == [0.0]


def test_swapped_paragraphs_change_only_the_reading_order():
    scores = _score(f"{SECOND}\n\n{FIRST}\n")

    assert scores["text"] == [0.0, 0.0]
    assert scores["reading_order"] > 0.0


def test_scores_are_macro_averaged_over_languages():
    english = {"language": "english", "text": [0.0], "tables": [1.0], "reading_order": 0.0}
    french = {"language": "french", "text": [1.0], "tables": [0.0], "reading_order": 1.0}
    results = [english, english, english, french]

    assert utils.docatlas_aggregate_text_edit(results) == pytest.approx(0.5)
    assert utils.docatlas_aggregate_table_teds(results) == pytest.approx(50.0)
    assert utils.docatlas_aggregate_reading_order_edit(results) == pytest.approx(0.25)
    assert utils.docatlas_aggregate_overall(results) == pytest.approx(50.0)
