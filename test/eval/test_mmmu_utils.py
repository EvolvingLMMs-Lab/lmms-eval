"""Tests for the MMMU scoring pipeline.

Covers the contract between ``mmmu_process_results``, ``parse_open_response`` and
``eval_open``: open-ended predictions must stay a list of answer candidates so that
``eval_open`` iterates over candidates instead of over single characters.

No dataset download, no model inference, no API keys required.
"""

from lmms_eval.tasks.mmmu import utils


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
