"""Hermetic CPU unit tests for the DocVQA 2026 benchmark task."""

import io
import math
from pathlib import Path

import datasets
import pytest
from PIL import Image

from lmms_eval.api.task import ConfigurableMessagesTask
from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.docvqa_2026 import utils
from lmms_eval.utils import load_yaml_config

TASK_DIR = Path(__file__).resolve().parents[2] / "lmms_eval" / "tasks" / "docvqa_2026"
YAML_PATH = TASK_DIR / "docvqa_2026.yaml"


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (12, 8), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_yaml_config_and_task_discovery() -> None:
    cfg = load_yaml_config(str(YAML_PATH))
    assert cfg["task"] == "docvqa_2026"
    assert cfg["dataset_path"] == "VLR-CVC/DocVQA-2026"
    assert cfg["dataset_kwargs"]["revision"] == "2dba69d9cbc402416a8a7f2dacbbbc2887f6ff88"
    assert cfg["dataset_kwargs"]["data_files"] == {"val": "val.parquet"}
    assert cfg["test_split"] == "val"
    assert cfg["output_type"] == "generate_until"
    assert cfg["process_docs"].__name__ == "docvqa_2026_process_docs"
    assert cfg["doc_to_visual"].__name__ == "docvqa_2026_doc_to_visual"
    assert cfg["doc_to_text"].__name__ == "docvqa_2026_doc_to_text"
    assert cfg["process_results"].__name__ == "docvqa_2026_process_results"

    metric_names = [m["metric"] for m in cfg["metric_list"]]
    assert metric_names == list(utils.ALL_METRICS)

    tm = TaskManager("ERROR")
    assert "docvqa_2026" in tm.all_tasks


def test_process_docs_expands_documents_and_resolves_visuals() -> None:
    raw_ds = (
        datasets.Dataset.from_dict(
            {
                "doc_id": ["maps_2", "comics_2"],
                "doc_category": ["maps", "comics"],
                "preview": [
                    {"bytes": _png_bytes((1, 2, 3)), "path": "prev0.png"},
                    {"bytes": _png_bytes((4, 5, 6)), "path": "prev1.png"},
                ],
                "document": [
                    [{"bytes": _png_bytes((10, 20, 30)), "path": "p0.png"}],
                    [
                        {"bytes": _png_bytes((40, 50, 60)), "path": "p1.png"},
                        {"bytes": _png_bytes((70, 80, 90)), "path": "p2.png"},
                    ],
                ],
                "questions": [
                    {
                        "question_id": ["1", "2"],
                        "question": ["First map question?", "Second map question?"],
                    },
                    {
                        "question_id": ["1"],
                        "question": ["Comic panel question?"],
                    },
                ],
                "answers": [
                    {
                        "question_id": ["1", "2"],
                        "answer": ["50 km", "2024-01-01"],
                    },
                    {
                        "question_id": ["1"],
                        "answer": ["olive green"],
                    },
                ],
            },
        )
        .cast_column("preview", datasets.Image())
        .cast_column("document", datasets.Sequence(datasets.Image()))
    )

    expanded = utils.docvqa_2026_process_docs(raw_ds)
    assert len(expanded) == 3
    assert "preview" not in expanded.column_names
    assert [row["doc_id"] for row in expanded] == ["maps_2", "maps_2", "comics_2"]
    assert [row["question_id"] for row in expanded] == ["1", "2", "1"]
    assert [row["answer"] for row in expanded] == ["50 km", "2024-01-01", "olive green"]

    pages_0 = utils.docvqa_2026_doc_to_visual(expanded[0])
    pages_1 = utils.docvqa_2026_doc_to_visual(expanded[1])
    pages_2 = utils.docvqa_2026_doc_to_visual(expanded[2])
    assert len(pages_0) == 1 and pages_0[0].mode == "RGB"
    assert len(pages_1) == 1 and pages_1[0].getpixel((0, 0)) == (10, 20, 30)
    assert len(pages_2) == 2 and pages_2[1].getpixel((0, 0)) == (70, 80, 90)

    # Verify YAML !function loading and reloading a modified document with the same doc_id
    cfg = load_yaml_config(str(YAML_PATH))
    yaml_expanded = cfg["process_docs"](raw_ds)
    yaml_pages = cfg["doc_to_visual"](yaml_expanded[2])
    assert len(yaml_pages) == 2 and yaml_pages[0].getpixel((0, 0)) == (40, 50, 60)

    updated_ds = datasets.Dataset.from_dict(
        {
            "doc_id": ["comics_2"],
            "doc_category": ["comics"],
            "document": [
                [{"bytes": _png_bytes((200, 100, 50)), "path": "p_updated.png"}],
            ],
            "questions": [
                {"question_id": ["1"], "question": ["Updated comic panel question?"]},
            ],
            "answers": [
                {"question_id": ["1"], "answer": ["bright orange"]},
            ],
        },
    ).cast_column("document", datasets.Sequence(datasets.Image()))
    updated_expanded = utils.docvqa_2026_process_docs(updated_ds)
    updated_pages = utils.docvqa_2026_doc_to_visual(updated_expanded[0])
    assert len(updated_pages) == 1 and updated_pages[0].getpixel((0, 0)) == (200, 100, 50)

    # Also verify cache invalidation if _DOC_VISUAL_SOURCE is reassigned for the same doc_id
    utils._DOC_VISUAL_SOURCE["comics_2"] = (raw_ds, 1)
    rebound_pages = utils.docvqa_2026_doc_to_visual(updated_expanded[0])
    assert len(rebound_pages) == 2 and rebound_pages[0].getpixel((0, 0)) == (40, 50, 60)


def test_doc_to_text_and_doc_to_messages_parity() -> None:
    page1 = Image.new("RGBA", (10, 10), color=(100, 150, 200, 255))
    page2 = Image.new("L", (10, 10), color=128)
    doc = {
        "doc_id": "slide_1",
        "doc_category": "slide",
        "document": [page1, page2],
        "question_id": "1",
        "question": "What is the total revenue in 2024?",
        "answer": "10 USD",
    }

    prompt = utils.docvqa_2026_doc_to_text(
        doc,
        {"pre_prompt": "[PRE] ", "post_prompt": " [POST]"},
    )
    assert prompt.startswith(
        "[PRE] ACT AS an expert Document Visual Question Answering (DocVQA) system.",
    )
    assert "FINAL ANSWER: [Your exact formatted answer]" in prompt
    assert prompt.endswith("\n\nQuestion: What is the total revenue in 2024? [POST]")

    visuals = utils.docvqa_2026_doc_to_visual(doc)
    assert len(visuals) == 2
    assert all(isinstance(v, Image.Image) and v.mode == "RGB" for v in visuals)

    task = ConfigurableMessagesTask.__new__(ConfigurableMessagesTask)
    task._config = type(
        "Cfg",
        (),
        {
            "doc_to_messages": None,
            "doc_to_visual": utils.docvqa_2026_doc_to_visual,
            "doc_to_text": utils.docvqa_2026_doc_to_text,
        },
    )()
    task.doc_to_visual = utils.docvqa_2026_doc_to_visual
    task.doc_to_text = lambda d: utils.docvqa_2026_doc_to_text(d)
    messages = task.doc_to_messages(doc)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert [item["type"] for item in content] == ["image", "image", "text"]
    assert content[2]["text"] == utils.docvqa_2026_doc_to_text(doc)


@pytest.mark.parametrize(
    ("raw_prediction", "ground_truth", "expected_ok", "expected_extracted"),
    [
        # Missing mandatory FINAL ANSWER marker -> False
        ("50 kg", "50 kg", False, "50 kg"),
        # Reasoning trace + strict numeric & unit match
        ("Step 1: Convert 50.0 kg.\nFINAL ANSWER: 50 kg", "50.0 kg", True, "50 kg"),
        # Numeric match but unit mismatch -> False
        ("FINAL ANSWER: 50 m", "50 kg", False, "50 m"),
        # Numeric ground truth blocks ANLS fallback even when string similarity >= 0.9
        ("FINAL ANSWER: 1000000001", "1000000000", False, "1000000001"),
        # Overflow strings must not match each other as float('inf') == float('inf')
        (f"FINAL ANSWER: {'9' * 400}", "8" * 400, False, "9" * 400),
        # Calendar date normalization across formats (including digit-leading dates)
        ("FINAL ANSWER: Jan 1st 2024", "2024-01-01", True, "Jan 1st 2024"),
        ("FINAL ANSWER: 15 Jan 2024", "2024-01-15", True, "15 Jan 2024"),
        ("FINAL ANSWER: 01/15/2024", "2024-01-15", True, "01/15/2024"),
        # Version strings X.Y.Z must match literally, not as dates
        ("FINAL ANSWER: 12.0.0", "12.0.0", True, "12.0.0"),
        ("FINAL ANSWER: 12.0.0", "12.0.1", False, "12.0.0"),
        ("FINAL ANSWER: 12.0.0", "13.0.0", False, "12.0.0"),
        # Multi-candidate ground truth list parsed via ast.literal_eval
        (
            "FINAL ANSWER: Dark Green!",
            "['olive green', 'green', 'dark green']",
            True,
            "Dark Green!",
        ),
        # Safe handling of non-literal / adversarial ground truth strings
        ("FINAL ANSWER: safe", "__import__('os').system('echo bad')", False, "safe"),
        # Relaxed text match: strips articles and punctuation, ANLS >= 0.9
        ("FINAL ANSWER: The Eiffel Tower.", "Eiffel Tower", True, "The Eiffel Tower."),
        ("FINAL ANSWER: Subfamily", "Subfamilly", True, "Subfamily"),
        ("FINAL ANSWER: Completely different", "Eiffel Tower", False, "Completely different"),
        # <think> block containing a draft FINAL ANSWER is stripped before scoring
        ("<think>FINAL ANSWER: wrong</think>\nFINAL ANSWER: Unknown", "Unknown", True, "Unknown"),
        # Empty FINAL ANSWER payload must never match
        ("FINAL ANSWER:", "Unknown", False, ""),
        ("FINAL ANSWER:   ", "!!!", False, ""),
    ],
)
def test_evaluate_docvqa_prediction(
    raw_prediction: str,
    ground_truth: str,
    expected_ok: bool,
    expected_extracted: str,
) -> None:
    is_correct, extracted = utils.evaluate_docvqa_prediction(raw_prediction, ground_truth)
    assert is_correct is expected_ok
    assert extracted == expected_extracted


def test_process_results_and_category_aggregations() -> None:
    samples = [
        (
            {"doc_id": "m1", "question_id": "1", "doc_category": "maps", "answer": "50 km"},
            ["FINAL ANSWER: 50.0 km"],
        ),
        (
            {"doc_id": "m1", "question_id": "2", "doc_category": "maps", "answer": "North"},
            ["FINAL ANSWER: South"],
        ),
        (
            {
                "doc_id": "c1",
                "question_id": "1",
                "doc_category": "comics",
                "answer": "['green', 'dark green']",
            },
            ["FINAL ANSWER: the dark green!"],
        ),
    ]
    per_metric_records: dict[str, list[object]] = {m: [] for m in utils.ALL_METRICS}
    for doc, res in samples:
        out = utils.docvqa_2026_process_results(doc, res)
        assert set(out.keys()) == set(utils.ALL_METRICS)
        for m, rec in out.items():
            per_metric_records[m].append(rec)

    assert per_metric_records["docvqa_2026_maps"] == [1.0, 0.0, None]
    assert per_metric_records["docvqa_2026_comics"] == [None, None, 1.0]
    assert per_metric_records["docvqa_2026_slide"] == [None, None, None]

    acc_records = per_metric_records["docvqa_2026_acc"]
    assert utils.docvqa_2026_aggregate_acc(acc_records) == pytest.approx(2.0 / 3.0)  # type: ignore[arg-type]
    assert utils.docvqa_2026_aggregate_maps(per_metric_records["docvqa_2026_maps"]) == pytest.approx(0.5)
    assert utils.docvqa_2026_aggregate_comics(per_metric_records["docvqa_2026_comics"]) == pytest.approx(1.0)
    assert math.isnan(utils.docvqa_2026_aggregate_slide(per_metric_records["docvqa_2026_slide"]))
    assert math.isnan(utils.docvqa_2026_aggregate_acc([]))


@pytest.mark.parametrize(
    "bad_gt",
    [None, "", "   ", "null", "NULL", [], [""], "[]", "['valid', '']"],
)
def test_missing_or_empty_ground_truth_rejected(bad_gt: object) -> None:
    with pytest.raises(ValueError, match="DocVQA 2026"):
        utils.evaluate_docvqa_prediction("FINAL ANSWER: anything", bad_gt)


def test_process_docs_rejects_missing_answer_annotation() -> None:
    incomplete_ds = datasets.Dataset.from_dict(
        {
            "doc_id": ["maps_99"],
            "doc_category": ["maps"],
            "questions": [{"question_id": ["1", "2"], "question": ["Q1?", "Q2?"]}],
            "answers": [{"question_id": ["1"], "answer": ["50 km"]}],
        },
    )
    with pytest.raises(ValueError, match="question_id='2'"):
        utils.docvqa_2026_process_docs(incomplete_ds)
