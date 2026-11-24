from lmms_eval.tasks.ok_vqa.utils import ok_vqa_process_results


def test_submission_uses_numeric_question_id() -> None:
    result = ok_vqa_process_results({"question_id": "2971475"}, ["Racing."])

    assert result["submission"] == {"question_id": 2971475, "answer": "racing"}
