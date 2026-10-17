"""Tests for the MMMU scoring and interleaved-image pipelines.

Covers the contract between ``mmmu_process_results``, ``parse_open_response`` and
``eval_open``: open-ended predictions must stay a list of answer candidates so that
``eval_open`` iterates over candidates instead of over single characters.

No dataset download, no model inference, no API keys required.
"""

from collections.abc import Callable
from typing import Any

import pytest
from PIL import Image

from lmms_eval.tasks.mmmu import utils
from lmms_eval.tasks.mmmu.reasoning import utils as reasoning_utils


def _open_doc(answer, doc_id="validation_Math_1"):
    return {
        "id": doc_id,
        "question_type": "open",
        "question": "What is the value?",
        "options": "[]",
        "answer": answer,
    }


def _multi_choice_doc(answer, doc_id="validation_Math_2"):
    return {
        "id": doc_id,
        "question_type": "multiple-choice",
        "question": "Which one is correct?",
        "options": "['first', 'second', 'third', 'fourth']",
        "answer": answer,
    }


# ===========================================================================
# Open-ended predictions keep their candidate list
# ===========================================================================


def test_open_ended_parsed_pred_stays_a_candidate_list():
    doc = _open_doc("0.5")

    result = utils.mmmu_process_results(doc, ["Therefore the answer is 0.5"])

    parsed_preds = result["mmmu_acc"]["parsed_pred"]
    assert len(parsed_preds) == 1, "one entry per generation is expected"
    assert isinstance(parsed_preds[0], list), "candidates must not be collapsed into a string"
    assert parsed_preds[0] == utils.parse_open_response("Therefore the answer is 0.5")


def test_open_ended_numeric_answer_is_scored_correct():
    doc = _open_doc("0.5")

    result = utils.mmmu_process_results(doc, ["Therefore the answer is 0.5"])
    judge_dict, metrics = utils.evaluate_mmmu([result["mmmu_acc"]])

    assert judge_dict[doc["id"]] == "Correct"
    assert metrics["acc"] == 1.0


def test_open_ended_string_answer_is_scored_correct():
    doc = _open_doc("hexagon")

    result = utils.mmmu_process_results(doc, ["The answer is hexagon."])
    judge_dict, metrics = utils.evaluate_mmmu([result["mmmu_acc"]])

    assert judge_dict[doc["id"]] == "Correct"
    assert metrics["acc"] == 1.0


def test_open_ended_wrong_answer_is_scored_wrong():
    doc = _open_doc("0.5")

    result = utils.mmmu_process_results(doc, ["Therefore the answer is 42"])
    judge_dict, metrics = utils.evaluate_mmmu([result["mmmu_acc"]])

    assert judge_dict[doc["id"]] == "Wrong"
    assert metrics["acc"] == 0.0


def test_open_ended_scores_correct_when_any_generation_matches():
    doc = _open_doc("0.5")

    result = utils.mmmu_process_results(doc, ["The answer is 42", "The answer is 0.5"])
    judge_dict, metrics = utils.evaluate_mmmu([result["mmmu_acc"]])

    assert len(result["mmmu_acc"]["parsed_pred"]) == 2
    assert judge_dict[doc["id"]] == "Correct"
    assert metrics["acc"] == 1.0


# ===========================================================================
# Submission files still carry a single scalar answer
# ===========================================================================


def test_open_ended_submission_is_a_single_string():
    doc = _open_doc("0.5")

    result = utils.mmmu_process_results(doc, ["Therefore the answer is 0.5"])

    submission = result["submission"]
    assert set(submission) == {doc["id"]}
    assert submission[doc["id"]] == "0.5"


def test_submission_uses_the_first_generation_only():
    doc = _open_doc("0.5")

    result = utils.mmmu_process_results(doc, ["The answer is 42", "The answer is 0.5"])

    # parse_open_response normalizes numbers to float, so "42" becomes 42.0
    assert result["submission"][doc["id"]] == "42.0"


def test_submission_answer_handles_empty_candidates():
    assert utils.to_submission_answer([]) == ""


def test_submission_answer_passes_through_multi_choice_letter():
    assert utils.to_submission_answer("B") == "B"


# ===========================================================================
# Multiple-choice behaviour is unchanged
# ===========================================================================


def test_multi_choice_parsed_pred_is_an_option_letter():
    doc = _multi_choice_doc("B")

    result = utils.mmmu_process_results(doc, ["B"])

    assert result["mmmu_acc"]["parsed_pred"] == ["B"]
    assert result["submission"][doc["id"]] == "B"


def test_multi_choice_answer_is_scored_correct():
    doc = _multi_choice_doc("B")

    result = utils.mmmu_process_results(doc, ["B"])
    judge_dict, metrics = utils.evaluate_mmmu([result["mmmu_acc"]])

    assert judge_dict[doc["id"]] == "Correct"
    assert metrics["acc"] == 1.0


@pytest.fixture
def image_doc(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("NUM_IMAGE", "1")
    monkeypatch.setitem(utils.config["metadata"], "interleaved_format", False)
    monkeypatch.setitem(reasoning_utils.config["metadata"], "interleaved_format", False)
    doc = _multi_choice_doc("A")
    doc["image_1"] = Image.new("RGB", (2, 2), "red")
    doc["image_2"] = Image.new("RGB", (2, 2), "blue")
    return doc


_MESSAGE_BUILDERS = [
    utils.mmmu_doc_to_messages,
    reasoning_utils.mmmu_doc_to_messages,
    reasoning_utils.mmmu_reward_doc_to_messages,
]
_PROMPT_KWARGS = {"prompt_type": "format", "multiple_choice_prompt": "", "open_ended_prompt": ""}


@pytest.mark.parametrize("builder", _MESSAGE_BUILDERS)
@pytest.mark.parametrize(
    ("question", "options", "expected_colors"),
    [
        ("Compare <image 2> with <image 1>.", "['first', 'second']", ["blue", "red"]),
        (
            "A graph <image 1> and a spanning tree <image 2> of <image 1> are given, Give the Fundamental Cut sets of <image 1> with respect of <image 2>",
            "['first', 'second']",
            ["red", "blue", "red", "red", "blue"],
        ),
        ("Compare <image 2> with the options.", "['<image 1>', '<image 2>']", ["blue", "red", "blue"]),
        ("Describe <image 1>.", "['first', 'second']", ["red"]),
    ],
)
def test_interleaved_images_follow_every_reference(builder: Callable, question: str, options: str, expected_colors: list[str], image_doc: dict[str, Any]) -> None:
    image_doc.update(question=question, options=options)

    content = builder(image_doc, _PROMPT_KWARGS)[-1]["content"]

    colors = [part["url"].getpixel((0, 0)) for part in content if part["type"] == "image"]
    assert colors == [Image.new("RGB", (1, 1), color).getpixel((0, 0)) for color in expected_colors]


@pytest.mark.parametrize("builder", _MESSAGE_BUILDERS)
def test_repeated_image_references_keep_intermediate_question_text(builder: Callable, image_doc: dict[str, Any]) -> None:
    image_doc["question"] = "A graph <image 1> and a spanning tree <image 2> of <image 1> are given, Give the Fundamental Cut sets of <image 1> with respect of <image 2>"

    content = builder(image_doc, _PROMPT_KWARGS)[-1]["content"]

    assert [part["text"] for part in content if part["type"] == "text"] == [
        "A graph",
        "and a spanning tree",
        "of",
        "are given, Give the Fundamental Cut sets of",
        "with respect of",
        "A. first\nB. second\nC. third\nD. fourth",
    ]


@pytest.mark.parametrize("module", [utils, reasoning_utils])
def test_numbered_visuals_remain_unique_and_sorted(module: Any, image_doc: dict[str, Any]) -> None:
    image_doc["question"] = "Compare <image 2> with <image 1> and <image 2>."

    visuals = module.mmmu_doc_to_visual(image_doc)

    assert [image.getpixel((0, 0)) for image in visuals] == [(255, 0, 0), (0, 0, 255)]


def test_qwen3_numbered_messages_keep_unique_images(image_doc: dict[str, Any]) -> None:
    image_doc["question"] = "Compare <image 2> with <image 1> and <image 2>."

    content = utils.mmmu_doc_to_messages(image_doc, {"format": "qwen3_vl"})[-1]["content"]

    assert [part["url"].getpixel((0, 0)) for part in content if part["type"] == "image"] == [(255, 0, 0), (0, 0, 255)]
    assert image_doc["question"] in content[-1]["text"]


def test_num_image_two_still_duplicates_a_single_image(image_doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUM_IMAGE", "2")
    image_doc["question"] = "Describe <image 1>."

    content = utils.mmmu_doc_to_messages(image_doc, _PROMPT_KWARGS)[-1]["content"]

    assert [part["url"].getpixel((0, 0)) for part in content if part["type"] == "image"] == [(255, 0, 0), (255, 0, 0)]
    assert content[-1]["text"] == ".\nA. first\nB. second\nC. third\nD. fourth"
