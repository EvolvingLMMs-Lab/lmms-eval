from PIL import Image

from lmms_eval.tasks.erqa import utils


def test_erqa_process_results_extracts_final_choice():
    doc = {"answer": "B", "question_id": "ERQA_1", "question_type": "Action Reasoning"}

    result = utils.erqa_process_results(doc, ["I considered A, but the answer is B."])

    assert result["erqa_acc"]["is_correct"] is True
    assert result["erqa_acc"]["id"] == "ERQA_1"


def test_erqa_doc_to_messages_interleaves_images_before_text():
    image = Image.new("RGB", (2, 2), color="white")
    doc = {"images": [image], "question": "Choices: A. yes B. no"}

    messages = utils.erqa_doc_to_messages(doc)

    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "image"
    assert messages[0]["content"][1] == {"type": "text", "text": doc["question"]}


def test_erqa_aggregate_results_returns_accuracy():
    results = [
        {"sub_task": "Action Reasoning", "is_correct": True},
        {"sub_task": "Action Reasoning", "is_correct": False},
    ]

    assert utils.erqa_aggregate_results(results) == 0.5
