import json
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.visualoverload import utils


@pytest.mark.parametrize("question_type,prompt", [("choice", "Choose A or B."), ("counting", "How many animals?"), ("ocr", "Read the ninth word.")])
def test_official_prompt_and_full_resolution(question_type: str, prompt: str) -> None:
    doc = {"image": Image.new("RGB", (3840, 2160)), "default_prompt": prompt, "question_type": question_type}
    assert utils.visualoverload_doc_to_text(doc, {"post_prompt": "unwanted extra instruction"}) == prompt
    assert utils.visualoverload_doc_to_visual(doc)[0].size == (3840, 2160)
    messages = utils.visualoverload_doc_to_messages(doc)
    assert messages[0]["content"][0]["url"].size == (3840, 2160)
    assert messages[0]["content"][1] == {"type": "text", "text": prompt}
    assert utils.visualoverload_doc_to_target(doc) == ""


def test_task_is_discoverable() -> None:
    assert "visualoverload" in TaskManager().all_tasks


def test_submission_export_has_no_local_score(tmp_path: Path) -> None:
    expected = [
        {"question_id": "001abc", "response": "B"},
        {"question_id": "count", "response": "There are 17 animals."},
        {"question_id": "ocr", "response": "café"},
        {"question_id": "empty", "response": ""},
    ]
    results = [utils.visualoverload_process_results({"question_id": item["question_id"]}, [item["response"]])["submission"] for item in expected]
    assert utils.visualoverload_aggregate_results(results, Namespace(output_path=str(tmp_path))) is None
    path = tmp_path / "submissions" / "visualoverload_submission.json"
    assert json.loads(path.read_text()) == expected


@pytest.mark.parametrize("ids", [["same", "same"], ["valid", None]])
def test_submission_rejects_invalid_ids(tmp_path: Path, ids: list) -> None:
    records = [{"question_id": question_id, "response": "A"} for question_id in ids]
    with pytest.raises(ValueError, match="question_id"):
        utils.visualoverload_aggregate_results(records, Namespace(output_path=str(tmp_path)))
