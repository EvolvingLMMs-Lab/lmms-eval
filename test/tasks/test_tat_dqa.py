"""Offline behavior contracts for the pinned TAT-DQA VLM adaptation."""

import io
import json
import shutil
import zipfile
from pathlib import Path

import datasets
import pytest
from PIL import Image

from lmms_eval.api.reasoning import strip_reasoning_tags
from lmms_eval.api.task import ConfigurableMessagesTask, ConfigurableTask
from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.tat_dqa import utils
from lmms_eval.utils import load_yaml_config

TASK_DIR = Path(__file__).resolve().parents[2] / "lmms_eval" / "tasks" / "tat_dqa"
DOC_ID = "a" * 32


def _row(answer, answer_type="arithmetic", scale=""):
    return {"answer_json": json.dumps(answer), "answer_type": answer_type, "scale": scale}


def _prediction(answer, scale=""):
    return json.dumps({"answer": answer, "scale": scale})


@pytest.fixture
def raw_release(tmp_path):
    if shutil.which("pdftoppm") is None:
        pytest.fail("TAT-DQA media tests require Poppler's pdftoppm")
    archive = tmp_path / "pages.zip"
    pdf_bytes = io.BytesIO()
    pages = [Image.new("RGB", (90, 70), color) for color in ("red", "green", "blue")]
    pages[0].save(pdf_bytes, "PDF", save_all=True, append_images=pages[1:], resolution=72)
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(f"test/{DOC_ID}.json", json.dumps({"pages": [{}, {}, {}]}))
        zipped.writestr(f"test/{DOC_ID}.pdf", pdf_bytes.getvalue())
        # Misleading thumbnails must not be used as the visual input.
        thumb = io.BytesIO()
        Image.new("RGB", (224, 224), "black").save(thumb, "PNG")
        for number in range(1, 4):
            zipped.writestr(f"test/{DOC_ID}_{number}.png", thumb.getvalue())
    questions = [
        {"uid": "q1", "question": "What was the amount?", "answer": 0, "answer_type": "arithmetic", "scale": "million"},
        {
            "uid": "q2",
            "question": "Which names?",
            "answer": ["ALPHA", "BETA"],
            "answer_type": "multi-span",
            "scale": "",
        },
    ]
    records = [{"doc": {"uid": DOC_ID, "page": 2}, "questions": questions}]
    return records, archive


def test_expansion_page_order_and_offline_reload(raw_release, tmp_path):
    records, archive = raw_release
    dataset = utils.build_dataset(records, archive)
    assert dataset["question_id"] == ["q1", "q2"]
    assert json.loads(dataset[0]["answer_json"]) == 0
    assert json.loads(dataset[1]["answer_json"]) == ["ALPHA", "BETA"]
    pages = utils.tat_dqa_doc_to_visual(dataset[0])
    assert [page.getpixel((0, 0)) for page in pages] == [(254, 0, 0), (0, 128, 1), (0, 0, 254)]
    assert [page.size for page in pages] == [(180, 140)] * 3
    assert len(pages) == 3  # doc.page=2 denotes the table page, not the page count.
    saved = tmp_path / "prepared"
    datasets.DatasetDict({"test": dataset}).save_to_disk(saved)
    archive.unlink()
    restored = datasets.load_from_disk(saved)["test"]
    assert utils.tat_dqa_process_docs(restored) is restored
    assert len(utils.tat_dqa_doc_to_visual(restored[1])) == 3
    assert utils.tat_dqa_doc_to_target(restored[0]) == '{"answer": 0, "scale": "million"}'


def test_process_docs_uses_document_text_and_pinned_media(raw_release, monkeypatch):
    records, archive = raw_release
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(archive)

    monkeypatch.setattr(utils, "hf_hub_download", download)
    raw = datasets.Dataset.from_dict({"text": [json.dumps(records)]})
    result = utils.tat_dqa_process_docs(raw)
    assert len(result) == 2
    assert calls == [
        {
            "repo_id": "next-tat/TAT-DQA",
            "filename": "tatdqa_docs_test.zip",
            "repo_type": "dataset",
            "revision": "bc46208b22f0bfe47152c98fd0938d238ce868c2",
        },
    ]
    with pytest.raises(ValueError, match="one raw JSON document"):
        utils.tat_dqa_process_docs(datasets.Dataset.from_dict({"text": ["[", "]"]}))


def test_data_errors_are_not_silently_filtered(raw_release):
    records, archive = raw_release
    records[0]["questions"].append(records[0]["questions"][0])
    with pytest.raises(ValueError, match="duplicate question"):
        utils.build_dataset(records, archive)
    records[0]["questions"].pop()
    records[0]["doc"]["uid"] = "../outside"
    with pytest.raises(ValueError, match="invalid document"):
        utils.build_dataset(records, archive)


def test_missing_media_is_a_data_failure(raw_release, tmp_path):
    records, _ = raw_release
    archive = tmp_path / "missing.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(f"test/{DOC_ID}.json", json.dumps({"pages": [{}]}))
    with pytest.raises(ValueError, match="missing document metadata or PDF"):
        utils.build_dataset(records, archive)


def test_prompt_and_chat_share_full_ordered_pages_without_gold(raw_release):
    records, archive = raw_release
    row = utils.build_dataset(records, archive)[1]
    text = utils.tat_dqa_doc_to_text(row)
    messages = utils.tat_dqa_doc_to_messages(row)
    assert [message["role"] for message in messages] == ["user"]
    content = messages[0]["content"]
    assert [part["type"] for part in content] == ["image", "image", "image", "text"]
    assert content[-1]["text"] == text
    assert text.endswith("Question: Which names?")
    assert all(secret not in text for secret in ("ALPHA", "BETA", DOC_ID, "multi-span", "<image>"))
    assert [part["url"].getpixel((0, 0)) for part in content[:-1]] == [(254, 0, 0), (0, 128, 1), (0, 0, 254)]


@pytest.mark.parametrize(
    "gold,answer_type,gold_scale,prediction,pred_scale,expected",
    [
        (0, "arithmetic", "", 0, "", (1.0, 1.0, 1.0)),
        (12.5, "arithmetic", "percent", 12.5, "percent", (1.0, 1.0, 1.0)),
        (12.5, "arithmetic", "percent", "12.5%", "", (1.0, 1.0, 0.0)),
        (12.5, "arithmetic", "percent", 0.125, "percent", (0.0, 0.0, 1.0)),
        (12.5, "arithmetic", "percent", 12.6, "percent", (0.0, 0.0, 1.0)),
        (1.234, "arithmetic", "million", 1.23, "million", (1.0, 1.0, 1.0)),
        (100, "arithmetic", "", "(100)", "", (0.0, 0.0, 1.0)),
        (2, "count", "", 3, "", (0.0, 0.0, 1.0)),
        (["net revenue"], "span", "", "the net revenue", "", (1.0, 1.0, 1.0)),
        (["net revenue"], "span", "", "revenue", "", (0.0, 0.67, 1.0)),
        (["alpha", "beta"], "multi-span", "", ["beta", "alpha"], "", (1.0, 1.0, 1.0)),
        (["alpha", "beta"], "multi-span", "", ["alpha"], "", (0.0, 0.67, 1.0)),
        (["alpha"], "span", "", "", "", (0.0, 0.0, 1.0)),
    ],
)
def test_scores_match_author_reference_cases(gold, answer_type, gold_scale, prediction, pred_scale, expected):
    """Expected literals follow Doc2SoarGraph@71a0271, including zero and rounding."""
    result = utils.tat_dqa_process_results(_row(gold, answer_type, gold_scale), [_prediction(prediction, pred_scale)])
    assert (result["tat_dqa_em"], result["tat_dqa_f1"], result["tat_dqa_scale_accuracy"]) == expected
    assert result["tat_dqa_parse_success"] == 1.0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "answer_type,gold_count,prediction_count,common_count,expected_f1",
    [
        ("span", 5, 75, 3, 0.08),
        ("span", 5, 76, 3, 0.07),
        ("span", 5, 74, 3, 0.08),
        ("multi-span", 2, 78, 1, 0.02),
        ("multi-span", 2, 79, 1, 0.02),
        ("multi-span", 2, 77, 1, 0.03),
    ],
)
def test_partial_f1_matches_reference_rounding(answer_type, gold_count, prediction_count, common_count, expected_f1):
    """Doc2SoarGraph rounds partial F1 through NumPy, including both tie directions."""
    shared = [f"shared{i}" for i in range(common_count)]
    gold = shared + [f"gold{i}" for i in range(gold_count - common_count)]
    prediction = shared + [f"prediction{i}" for i in range(prediction_count - common_count)]
    if answer_type == "span":
        gold = [" ".join(gold)]
        prediction = " ".join(prediction)
    result = utils.tat_dqa_process_results(_row(gold, answer_type), [_prediction(prediction)])
    assert result == {
        "tat_dqa_em": 0.0,
        "tat_dqa_f1": expected_f1,
        "tat_dqa_scale_accuracy": 1.0,
        "tat_dqa_parse_success": 1.0,
    }
    assert all(type(value) is float for value in result.values())
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "response",
    [
        "",
        "I cannot answer",
        "12.5",
        '{"answer": 12.5}',
        '{"answer": true, "scale": ""}',
        '{"answer": NaN, "scale": ""}',
        '{"answer": 1e309, "scale": ""}',
        '{"answer": {}, "scale": ""}',
        '{"answer": 12.5, "scale": "trillion"}',
        '{"answer": 12.5, "scale": "", "explanation": "extra"}',
        "__import__('os').system('false')",
        '{"answer": ' + "9" * 400 + ', "scale": ""}',
        "[" * 2000,
    ],
)
def test_invalid_response_is_counted_as_zero(response):
    result = utils.tat_dqa_process_results(_row(12.5), [response])
    assert result == {
        "tat_dqa_em": 0.0,
        "tat_dqa_f1": 0.0,
        "tat_dqa_scale_accuracy": 0.0,
        "tat_dqa_parse_success": 0.0,
    }


def test_fenced_response_and_framework_reasoning_boundary():
    response = '```json\n{"answer": 0, "scale": ""}\n```'
    assert utils.tat_dqa_process_results(_row(0), [response])["tat_dqa_em"] == 1.0
    response = '<think>unscored analysis</think>\n{"answer": 0, "scale": ""}'
    final = strip_reasoning_tags(response, [["<think>", "</think>"]])
    assert utils.tat_dqa_process_results(_row(0), [final])["tat_dqa_em"] == 1.0
    assert utils.tat_dqa_process_results(_row(0), ["<think>unfinished"])["tat_dqa_em"] == 0.0


def test_real_task_loading_preserves_simple_chat_and_denominator(raw_release, tmp_path):
    records, archive = raw_release
    prepared = tmp_path / "data"
    datasets.DatasetDict({"test": utils.build_dataset(records, archive)}).save_to_disk(prepared)
    config = load_yaml_config(str(TASK_DIR / "tat_dqa.yaml"))
    config.update(dataset_path=str(prepared), dataset_kwargs={"load_from_disk": True})
    manager = TaskManager("ERROR")
    assert "tat_dqa" in manager.all_tasks
    simple = ConfigurableTask(config=dict(config))
    chat = ConfigurableMessagesTask(config=dict(config))
    for task in (simple, chat):
        row = task.dataset_no_image["test"][0]
        assert "images" not in row
        rows = [task.process_results(row, [_prediction(0, "million")]), task.process_results(row, ["invalid"])]
        assert task.aggregation()["tat_dqa_em"]([item["tat_dqa_em"] for item in rows]) == 0.5
        assert task.aggregation()["tat_dqa_f1"]([item["tat_dqa_f1"] for item in rows]) == 0.5
        assert task.aggregation()["tat_dqa_parse_success"]([item["tat_dqa_parse_success"] for item in rows]) == 0.5
    message_text = chat.doc_to_messages(chat.dataset["test"][0])[-1]["content"][-1]["text"]
    assert message_text == simple.doc_to_text(simple.dataset_no_image["test"][0])


def test_reference_overflow_fallback_is_preserved():
    # Doc2SoarGraph turns failed number formatting into an empty answer string.
    # Article-only gold also normalizes to empty; preserve this reference quirk.
    response = _prediction("9" * 400)
    result = utils.tat_dqa_process_results(_row(["the"], "span"), [response])
    assert result["tat_dqa_em"] == result["tat_dqa_f1"] == 1.0


def test_malformed_gold_is_rejected_at_loading(raw_release):
    records, archive = raw_release
    records[0]["questions"][0]["answer"] = {"invalid": 1}
    with pytest.raises(ValueError, match="numerical gold"):
        utils.build_dataset(records, archive)


def test_pdf_page_count_must_match_metadata(raw_release, tmp_path):
    records, archive = raw_release
    with zipfile.ZipFile(archive) as source:
        pdf = source.read(f"test/{DOC_ID}.pdf")
    broken = tmp_path / "wrong_page_count.zip"
    with zipfile.ZipFile(broken, "w") as target:
        target.writestr(f"test/{DOC_ID}.json", json.dumps({"pages": [{}]}))
        target.writestr(f"test/{DOC_ID}.pdf", pdf)
    with pytest.raises(ValueError, match="page count"):
        utils.build_dataset(records, broken)
