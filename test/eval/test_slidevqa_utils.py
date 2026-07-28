import math

import pytest
from PIL import Image

from lmms_eval.tasks.slidevqa import utils


def _image(color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    return Image.new("RGB", (10, 8), color)


def _doc(**overrides):
    doc = {f"page_{index}": None for index in range(1, 21)}
    doc.update({"deck_name": "demo", "question": "What is shown?", "answer": "chart"})
    doc.update(overrides)
    return doc


def test_doc_to_visual_packs_twenty_pages_into_five_grids():
    doc = _doc(**{f"page_{index}": _image() for index in range(1, 21)})

    grids = utils.slidevqa_doc_to_visual(doc)

    assert len(grids) == 5
    assert all(grid.size == (20, 16) for grid in grids)


def test_doc_to_visual_skips_empty_pages_and_errors_on_empty_deck():
    doc = _doc(page_1=_image((255, 0, 0)), page_3=_image((0, 255, 0)))

    grids = utils.slidevqa_doc_to_visual(doc)

    assert len(grids) == 2
    assert grids[0].getpixel((0, 0)) == (255, 0, 0)
    assert grids[1].getpixel((0, 0)) == (0, 255, 0)

    with pytest.raises(ValueError, match="has no images"):
        utils.slidevqa_doc_to_visual(_doc())


def test_doc_to_text_and_messages():
    doc = _doc(**{f"page_{index}": _image() for index in range(1, 21)})
    kwargs = {"pre_prompt": "", "post_prompt": "\nAnswer using only the short answer, without explanation."}

    assert utils.slidevqa_doc_to_text(doc, kwargs) == "What is shown?\nAnswer using only the short answer, without explanation."

    messages = utils.slidevqa_doc_to_messages(doc, kwargs)
    content = messages[0]["content"]
    assert messages[0]["role"] == "user"
    assert [item["type"] for item in content] == ["image"] * 5 + ["text"]
    assert content[-1]["text"].endswith("without explanation.")


def test_normalize_answer_handles_missing_and_newlines():
    assert utils.normalize_answer(None) == "not answerable"
    assert utils.normalize_answer(float("nan")) == "not answerable"
    assert utils.normalize_answer("New\nYork") == "newyork"


def test_scoring_matches_short_answer_metrics():
    assert utils.anls_score("newyork", "newyork") == 1.0
    assert utils.anls_score("abc", "xyz") == 0.0
    assert math.isclose(utils.word_f1("total revenue", "revenue"), 2 / 3)

    scores = utils.slidevqa_process_results({"answer": "New\nYork"}, ["newyork"])
    assert scores == {"anls": 1.0, "em": 1.0, "f1": 1.0}
