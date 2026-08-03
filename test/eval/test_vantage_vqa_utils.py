import json
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset

from lmms_eval.tasks.vantage_vqa import utils


def test_vantage_vqa_process_docs_adds_index_and_video():
    dataset = Dataset.from_list([{"q_uid": "sample.json", "options": [], "question": "Q"}])

    processed = utils.vantage_vqa_process_docs(dataset)

    assert processed[0]["index"] == 0
    assert processed[0]["video"] == "sample.mp4"


def test_vantage_vqa_doc_to_text_formats_labeled_and_unlabeled_options():
    doc = {
        "question": "What happened?",
        "options": ["A: One", "Two", "C. Three", "D) Four"],
    }

    prompt = utils.vantage_vqa_doc_to_text(doc, {"post_prompt": "\nAnswer only."})

    assert "Question: What happened?" in prompt
    assert "A. One" in prompt
    assert "B. Two" in prompt
    assert "C. Three" in prompt
    assert "D. Four" in prompt
    assert prompt.endswith("\nAnswer only.")


def test_vantage_vqa_doc_to_messages_uses_downloaded_video(monkeypatch):
    monkeypatch.setattr(utils, "_download_video", lambda video: f"/tmp/{video}")
    doc = {"q_uid": "clip.json", "question": "Q?", "options": ["one", "two", "three", "four"]}

    messages = utils.vantage_vqa_doc_to_messages(doc)

    assert messages[0]["content"][0] == {"type": "video", "url": "/tmp/clip.mp4"}
    assert messages[0]["content"][1]["type"] == "text"


def test_vantage_vqa_process_results_emits_canonical_submission_record():
    doc = {"q_uid": "clip.json", "video": "clip.mp4", "index": 7}

    result = utils.vantage_vqa_process_results(doc, ["The answer is C."])

    assert result["submission"] == {
        "id": "clip__q_000007",
        "task": "video_qa",
        "conversations": [{"from": "assistant", "value": "The answer is C."}],
        "metadata": {"model": "", "extra": {}},
    }


def test_vantage_vqa_aggregate_submissions_writes_jsonl(tmp_path: Path):
    args = SimpleNamespace(output_path=str(tmp_path), model="dummy")
    rows = [
        {
            "id": "clip__q_000007",
            "task": "video_qa",
            "conversations": [{"from": "assistant", "value": "C"}],
            "metadata": {"model": "", "extra": {}},
        },
    ]

    utils.vantage_vqa_aggregate_submissions(rows, args)

    output = tmp_path / "submissions" / "vantage_vqa_submission.jsonl"
    assert json.loads(output.read_text(encoding="utf-8")) == {**rows[0], "metadata": {"model": "dummy", "extra": {}}}
