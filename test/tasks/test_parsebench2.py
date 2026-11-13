"""Offline contracts with the actual optional scorer; public replay is separate."""

import json
import math
from pathlib import Path

import pytest
from PIL import Image

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.parsebench2 import utils
from lmms_eval.utils import load_yaml_config

YAML = Path(__file__).resolve().parents[2] / "lmms_eval/tasks/parsebench2/parsebench2.yaml"


@pytest.fixture
def scorer() -> None:
    pytest.importorskip("parse_bench")
    utils.require_scorer()


@pytest.fixture
def document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    root = tmp_path / "text"
    root.mkdir()
    image = Image.new("RGB", (100, 80), "white")
    image.save(root / "example.pdf", save_all=True, append_images=[image])
    (root / "example.test.json").write_text(json.dumps({"expected_markdown": "# Heading\n\nA **bold** example."}))
    monkeypatch.setattr(utils, "fetch", lambda name: str(tmp_path / name))
    utils.load_case.cache_clear()
    yield {"example_id": "text/example", "subset": "text", "document": "text/example.pdf"}
    utils.load_case.cache_clear()


def test_discovery_does_not_import_optional_scorer() -> None:
    import subprocess
    import sys

    # A fresh interpreter guards against accidental optional imports at discovery.
    subprocess.run([sys.executable, "-c", "import sys; from lmms_eval.tasks.parsebench2 import utils; assert 'parse_bench' not in sys.modules"], check=True)
    assert "parsebench2" in TaskManager("ERROR").all_tasks


def test_whitelisted_inputs_and_interleaving(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for subset in utils.SUBSETS:
        rows = [{"id": f"{subset}/p{i}", "document": f"{subset}/p{i}.pdf", "expected_markdown": "SECRET", "test_rules": ["SECRET"]} for i in range(2)]
        (tmp_path / f"{subset}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr(utils, "fetch", lambda name: str(tmp_path / name))
    docs = utils.load_inputs()
    assert [d["subset"] for d in docs[:13]] == list(utils.SUBSETS)
    assert len(docs) == 26
    assert all(set(d) == {"example_id", "subset", "document"} for d in docs)
    assert "SECRET" not in json.dumps(docs)
    assert utils.doc_to_text({"expected_markdown": "SECRET"}) == utils.doc_to_text({})
    with pytest.raises(ValueError, match="disagree"):
        utils.input_record({"id": "text/p1", "document": "text/p2.pdf"}, "text")
    with pytest.raises(ValueError):
        utils.input_record({"id": "../p1", "document": "../p1.pdf"}, "text")


@pytest.mark.parametrize("protocol", ["simple", "chat"])
def test_real_task_loading_requests_filters_and_sample_logs(scorer: None, document: dict, monkeypatch: pytest.MonkeyPatch, protocol: str) -> None:
    from lmms_eval.evaluator import _process_document_results

    monkeypatch.setattr(utils, "load_inputs", lambda: [document])
    manager = TaskManager("ERROR", model_name="dummy")
    task = manager.load_config(load_yaml_config(str(YAML)), task_type=protocol)["parsebench2"]
    assert task.test_docs()[0] == document
    assert task.test_docs_no_media()[0] == document
    request = task.construct_requests(0, utils.PROMPT, metadata={"task": "parsebench2", "doc_id": 0, "repeats": 1, "split": "test"})
    assert request.args[3:] == (0, "parsebench2", "test")
    if protocol == "simple":
        assert len(request.args[2](document)) == 2
    else:
        message = request.args[1](document)[0]
        assert [p["type"] for p in message["content"]] == ["image", "image", "text"]
        assert message["content"][-1]["text"] == utils.PROMPT
    response = "  # Heading\n\nA **bold** example.\n "
    request.doc = document
    request.resps = [response]
    task._instances = [request]
    task.apply_filters()
    assert request.filtered_resps["none"] == response
    result = _process_document_results((0, document, [request]), task=task, filter_key="none", reasoning_tags=None, log_samples=True)
    assert result.logged_sample["doc"] == document
    assert result.logged_sample["filtered_resps"] == [response]
    assert "SECRET" not in json.dumps(result.logged_sample)
    assert result.metrics["parsebench2_response_failure_rate"] == 0
    assert math.isfinite(task.aggregation()["parsebench2_text"]([result.metrics["parsebench2_text"]]))


def test_rendering_all_pages(document: dict) -> None:
    pytest.importorskip("pypdfium2")
    pages = utils.doc_to_visual(document)
    assert len(pages) == 2
    assert all(image.mode == "RGB" and image.size == (200, 160) for image in pages)


@pytest.mark.parametrize("response", [None, "", "{broken", '{"markdown":null}', '{"markdown":"x","layout_pages":"bad"}'])
def test_missing_and_malformed_responses_stay_in_denominator(scorer: None, document: dict, response: str | None) -> None:
    good = utils.process_results(document, ["# Heading\n\nA **bold** example."])["parsebench2_text"]
    bad = utils.process_results(document, [response])
    assert bad["parsebench2_response_failure_rate"] == 1
    assert utils.aggregate_text([bad["parsebench2_text"]]) == 0
    other = json.loads(json.dumps(bad["parsebench2_text"]))
    other["evaluation"]["example_id"] = "text/other"
    other["evaluation"]["test_id"] = "text/other"
    assert utils.aggregate_text([good, other]) == pytest.approx(utils.aggregate_text([good]) / 2)


def test_full_record_roundtrip_and_identity(scorer: None, document: dict) -> None:
    record = utils._native_result(document, {"markdown": "x", "layout_pages": []}).model_dump(mode="json")
    record["raw_output"] = {"pages": [{"items": [{"md": "provider data"}]}], "other": [1, 2]}
    restored, layout, status = utils.decode_response(document, json.dumps(record))
    assert restored.model_dump(mode="json") == record
    assert layout and status == "official_record"
    record["request"]["example_id"] = "text/wrong"
    with pytest.raises(ValueError, match="IDs disagree"):
        utils.decode_response(document, json.dumps(record))


def test_layout_normalization_and_missing_dimension(scorer: None, document: dict, tmp_path: Path) -> None:
    # Actual official adapter + layout scorer, fractional xywh in the prompt's IR.
    root = tmp_path / "layout"
    root.mkdir()
    source = Path(utils.fetch(document["document"]))
    (root / "example.pdf").write_bytes(source.read_bytes())
    rule = {"type": "layout", "bbox": [0.1, 0.1, 0.5, 0.2], "canonical_class": "Text", "content": {"type": "text", "text": "Hello"}, "page": 1}
    (root / "example.test.json").write_text(json.dumps({"test_rules": [rule]}))
    doc = {"example_id": "layout/example", "subset": "layout", "document": "layout/example.pdf"}
    prediction = {"markdown": "Hello", "layout_pages": [{"page_number": 1, "width": 1, "height": 1, "items": [{"type": "text", "md": "Hello", "bbox": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.2}}]}]}
    item = utils.process_results(doc, [json.dumps(prediction)])["parsebench2_layout"]
    assert utils.aggregate_layout([item]) == pytest.approx(1)
    empty = utils.process_results(doc, ['{"markdown":"","layout_pages":[]}'])["parsebench2_layout"]
    assert utils.aggregate_layout([empty]) == 0
    plain = utils.process_results(doc, ["Hello"])["parsebench2_layout"]
    assert math.isnan(utils.aggregate_layout([plain]))
    assert math.isnan(utils.aggregate_overall([item]))  # absent other subsets
    assert math.isnan(utils.aggregate_layout([item, {**plain, "evaluation": {**plain["evaluation"], "example_id": "layout/other"}}]))


def test_duplicate_and_repeat_rejection(scorer: None, document: dict) -> None:
    item = utils.process_results(document, ["x"])["parsebench2_text"]
    with pytest.raises(ValueError, match="Duplicate"):
        utils.combined_report([item, item])
    with pytest.raises(ValueError, match="one filtered response"):
        utils.process_results(document, ["x", "x"])


def test_scorer_error_is_fatal(scorer: None, document: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from parse_bench.evaluation.metrics.parse import next_metrics

    def broken(*args: object) -> None:
        raise RuntimeError("scorer environment broken")

    monkeypatch.setattr(next_metrics, "compute_next_metrics", broken)
    with pytest.raises(RuntimeError, match="environment broken"):
        utils.process_results(document, ["x"])


def test_hub_symlink_identity(scorer: None, document: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path(utils.fetch(document["document"]))
    sidecar = Path(utils.fetch("text/example.test.json"))
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()
    blob = blob_dir / "hash-with-no-extension"
    source.rename(blob)
    source.symlink_to(blob)
    case = utils.load_case(document["document"])
    assert case.test_id == document["example_id"]
    assert case.group == "text"
    assert case.file_path == blob.resolve()
    assert case.expected_markdown is not None
    assert sidecar.exists()


@pytest.mark.parametrize(
    "payload",
    [
        {"markdown": "x", "layout_pages": [{"page_number": 1, "width": 100, "height": 100}]},
        {"markdown": "x", "layout_pages": [{"page_number": 1, "width": 1, "height": 1, "items": [{"bbox": {"x": 0.9, "y": 0, "w": 0.5, "h": 1}}]}]},
    ],
)
def test_native_coordinates_are_explicit(scorer: None, document: dict, payload: dict) -> None:
    _, supported, status = utils.decode_response(document, json.dumps(payload))
    assert not supported and status == "invalid"
