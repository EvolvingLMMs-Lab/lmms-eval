import pytest
from datasets import Dataset
from PIL import Image

from lmms_eval.tasks.countqa import utils


def image_row(questions, answers, is_focused=False):
    return {
        "image": Image.new("RGB", (8, 8), "red"),
        "questions": questions,
        "answers": answers,
        "objects": ["chair", " table"],
        "categories": ["Furniture & Home Decor"],
        "is_focused": is_focused,
        "full_config": "{}",
    }


def test_process_docs_expands_one_document_per_question():
    source = Dataset.from_list([image_row(["How many chairs?", "How many tables?"], ["4", "1"]), image_row(["How many cups?"], ["12"], True)])
    docs = utils.countqa_process_docs(source)
    assert len(docs) == 3
    assert [d["question"] for d in docs] == ["How many chairs?", "How many tables?", "How many cups?"]
    assert [d["answer"] for d in docs] == ["4", "1", "12"]
    assert [(d["image_index"], d["qa_index"]) for d in docs] == [(0, 0), (0, 1), (1, 0)]
    assert [d["is_focused"] for d in docs] == [False, False, True]
    assert docs[0]["objects"] == ["chair", "table"]
    assert utils.countqa_doc_to_visual(docs[2])[0].size == (8, 8)


def test_doc_to_text_appends_post_prompt():
    doc = {"question": "How many chairs are there?"}
    assert utils.countqa_doc_to_text(doc, {"pre_prompt": "", "post_prompt": "\nAnswer with a single number."}) == "How many chairs are there?\nAnswer with a single number."


@pytest.mark.parametrize(
    "response, expected",
    [
        ("12", 12),
        ("There are 7 chairs.", 7),
        ("1,200", 1200),
        ("seven", 7),
        ("<think>I see 3, then 4 more, so 7.</think>\n7", 7),
        ("<think>Counting 1, 2, 3.</think> There are five.", 5),
        ("about 2.5", None),
        ("I cannot tell.", None),
        ("", None),
    ],
)
def test_parse_count(response, expected):
    assert utils.parse_count(response) == expected


def test_process_results_and_aggregations():
    docs = [
        {"answer": "10", "image_index": 0, "qa_index": 0, "is_focused": True},
        {"answer": "10", "image_index": 1, "qa_index": 0, "is_focused": False},
        {"answer": "3", "image_index": 2, "qa_index": 0, "is_focused": False},
    ]
    records = [utils.countqa_process_results(doc, [pred])["countqa_acc"] for doc, pred in zip(docs, ["10", "11", "no idea"])]
    assert utils.countqa_aggregate_acc(records) == pytest.approx(1 / 3)
    assert utils.countqa_aggregate_off_by_one_acc(records) == pytest.approx(2 / 3)
    assert utils.countqa_aggregate_focused_acc(records) == 1.0
    assert utils.countqa_aggregate_unfocused_acc(records) == 0.0
