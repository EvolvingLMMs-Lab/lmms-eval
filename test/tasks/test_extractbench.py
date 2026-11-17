"""Offline behavioral checks for the full-document ExtractBench adaptation."""

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import datasets
import pytest
from PIL import Image

from lmms_eval.tasks.extractbench import utils

TASK_DIR = Path(__file__).resolve().parents[2] / "lmms_eval" / "tasks" / "extractbench"


@pytest.fixture
def official():
    if sys.version_info < (3, 12) or importlib.util.find_spec("extract_bench") is None:
        pytest.skip("Install the extractbench extra on Python >=3.12 for official scorer integration")
    utils.check_dependency()


def row(expected=None, schema=None, rules=None):
    return {
        "id": "short/example",
        "category": "short",
        "pdf": "docs/short/example.pdf",
        "data_schema": json.dumps(schema or {"type": "object", "properties": {"name": {"type": "string"}}}),
        "expected_output": json.dumps({"name": "Alice"} if expected is None else expected),
        "field_rules": json.dumps({} if rules is None else rules),
        "repeated_structure": "{}",
        "tags": [],
    }


def test_task_discovery_and_group_weighting():
    from lmms_eval.tasks import TaskManager
    from lmms_eval.utils import load_yaml_config

    manager = TaskManager("ERROR", include_path=str(TASK_DIR), include_defaults=False)
    assert {"extractbench", "extractbench_short", "extractbench_medium", "extractbench_long"} <= set(manager.all_tasks)
    group = load_yaml_config(str(TASK_DIR / "extractbench.yaml"))
    assert all(metric["weight_by_size"] for metric in group["aggregate_metric_list"])
    for split in ("short", "medium", "long"):
        config = load_yaml_config(str(TASK_DIR / f"extractbench_{split}.yaml"))
        assert config["test_split"] == split
        assert config["dataset_kwargs"]["revision"] == utils.DATASET_REVISION
        assert config["reasoning_tags"] is False


@pytest.mark.parametrize(
    "pdf",
    ["/tmp/a.pdf", "docs/../a.pdf", "docs/short/../../a.pdf", "other/a.pdf", "docs/a.png", "docs/a\\b.pdf"],
)
def test_unsafe_media_paths_fail_before_download(pdf, monkeypatch):
    monkeypatch.setattr(utils, "hf_hub_download", lambda *args, **kwargs: pytest.fail("must not download"))
    with pytest.raises(ValueError, match="unsafe PDF"):
        utils.extractbench_doc_to_visual({**row(), "pdf": pdf})


@pytest.mark.parametrize("package", ["llama-extract-bench", "parse-bench"])
@pytest.mark.parametrize("installed", [None, "0.0.0"])
def test_missing_or_wrong_scorer_version_fails(monkeypatch, package, installed):
    def version(name):
        if name == package:
            if installed is None:
                raise utils.importlib.metadata.PackageNotFoundError(name)
            return installed
        return {"llama-extract-bench": utils.SCORER_VERSION, "parse-bench": utils.SCHEMA_VERSION}[name]

    monkeypatch.setattr(utils.importlib.metadata, "version", version)
    utils.check_dependency.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="pinned scorer"):
            utils.check_dependency()
    finally:
        utils.check_dependency.cache_clear()


def test_target_preserves_types():
    expected = {"zero": 0, "absent": None, "flag": False, "rows": ["a", "b"]}
    assert json.loads(utils.extractbench_doc_to_target(row(expected))) == expected


@pytest.mark.usefixtures("official")
@pytest.mark.parametrize("list_rules", [False, True])
def test_case_mapping_matches_official_sidecar_loader(tmp_path, list_rules):
    from extract_bench.test_cases.loader import load_test_case

    body = {"field_path": "ignored_body_path", "expected_value": "Alice", "evidence": [{"value": "Alice"}]}
    rules = [{**body, "type": "extract_field", "field_path": "name"}] if list_rules else {"name": body}
    doc = row(rules=rules)
    doc["repeated_structure"] = json.dumps({"rows": {"identity_key": "id"}})
    original = copy.deepcopy(doc)
    path = tmp_path / "short" / "example.pdf"
    path.parent.mkdir()
    path.touch()
    sidecar = {
        "data_schema": json.loads(doc["data_schema"]),
        "expected_output": json.loads(doc["expected_output"]),
        "_eval_row_identity": json.loads(doc["repeated_structure"]),
        "test_rules" if list_rules else "_field_rules": rules,
    }
    path.with_suffix(".test.json").write_text(json.dumps(sidecar))
    reference = load_test_case(path)
    actual = utils.case_from_doc(doc)
    assert actual.model_dump(exclude={"file_path"}) == reference.model_dump(exclude={"file_path"})
    assert doc == original
    assert actual.get_extract_field_rules()[0].field_path == "name"


@pytest.mark.usefixtures("official")
@pytest.mark.parametrize("rules", [{"name": {"type": "table"}}, {"name": 42}, 42])
def test_bad_field_rules_raise(rules):
    with pytest.raises(ValueError):
        utils.case_from_doc(row(rules=rules))


@pytest.mark.usefixtures("official")
def test_validation_preserves_rows_and_rejects_duplicates_or_bad_gold():
    rows = datasets.Dataset.from_list([row(), {**row(), "id": "short/second"}])
    assert utils.extractbench_process_docs(rows) is rows
    with pytest.raises(ValueError, match="Duplicate"):
        utils.extractbench_process_docs(datasets.Dataset.from_list([row(), row()]))
    with pytest.raises(ValueError, match="no expected output"):
        utils.extractbench_process_docs([{**row(), "expected_output": "null"}])
    with pytest.raises(ValueError, match="split/ID"):
        utils.extractbench_process_docs([{**row(), "category": "long"}])


@pytest.mark.usefixtures("official")
def test_prompt_matches_official_and_excludes_gold(monkeypatch):
    from extract_bench.inference.providers.extract.direct_model_utils import (
        DEFAULT_SYSTEM_PROMPT,
        DEFAULT_USER_INSTRUCTION,
    )
    from extract_bench.inference.providers.extract.vllm_extract import VLLMExtractProvider

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "repeated_structure": {"rows": {"type": "array", "items": {"type": "string"}}},
    }
    doc = row(expected={"name": "GOLD_SECRET"}, schema=schema, rules={"name": {"expected_value": "RULE_SECRET"}})
    doc["repeated_structure"] = '{"hidden_identity": {"identity_key": "EVAL_SECRET"}}'
    pages = [Image.new("RGB", (3, 2), color) for color in ("red", "blue")]
    monkeypatch.setattr(utils, "extractbench_doc_to_visual", lambda doc: pages)
    # These pure methods do not need a model client or a provider constructor.
    provider = object.__new__(VLLMExtractProvider)
    provider._additional_properties_false = True
    provider._user_instruction = DEFAULT_USER_INSTRUCTION
    reference = provider._build_user_content([], provider._prepare_schema(schema))[-1]["text"]
    messages = utils.extractbench_doc_to_messages(doc)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]
    assert [part["url"] for part in messages[1]["content"][:-1]] == pages
    assert messages[1]["content"][-1] == {"type": "text", "text": reference}
    text = utils.extractbench_doc_to_text(doc)
    assert text == DEFAULT_SYSTEM_PROMPT + "\n\n" + reference
    assert all(secret not in text for secret in ("GOLD_SECRET", "RULE_SECRET", "EVAL_SECRET"))


def test_real_pdf_renders_all_pages_at_150_dpi(tmp_path, monkeypatch):
    pytest.importorskip("pdf2image")
    if shutil.which("pdftoppm") is None or shutil.which("pdfinfo") is None:
        pytest.skip("Poppler is needed for the PDF integration check")
    pdf = tmp_path / "example.pdf"
    pages = [Image.new("RGB", (72, 36), color) for color in ("red", "green", "blue")]
    pages[0].save(pdf, "PDF", save_all=True, append_images=pages[1:], resolution=72)
    calls = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(pdf)

    monkeypatch.setattr(utils, "hf_hub_download", download)
    images = utils.extractbench_doc_to_visual(row())
    assert [image.size for image in images] == [(150, 75)] * 3
    assert [image.getpixel((0, 0)) for image in images] == [(254, 0, 0), (0, 128, 1), (0, 0, 254)]
    assert calls == [
        ((utils.DATASET_REPO, "docs/short/example.pdf"), {"repo_type": "dataset", "revision": utils.DATASET_REVISION}),
    ]


@pytest.mark.usefixtures("official")
@pytest.mark.parametrize(
    "response,parsed,salvaged",
    [
        ('{"name":"Alice"}', True, False),
        ('<think>private</think>{"name":"Alice"}', True, False),
        ('```json\n{"name":"Alice"}\n```', True, True),
        ('{"name":"Alice","unfinished":', True, True),
        ("", False, False),
        ("I cannot answer", False, False),
        ("null", True, False),
        ("<think>unfinished", False, False),
    ],
)
def test_official_parser_branches(response, parsed, salvaged):
    data, actual_parsed, actual_salvaged = utils.parse_prediction(response)
    assert (actual_parsed, actual_salvaged) == (parsed, salvaged)
    if parsed and response != "null":
        assert data["name"] == "Alice"


@pytest.mark.usefixtures("official")
def test_wrong_empty_and_refusal_outputs_stay_in_denominator():
    outputs = ['{"name":"Alice"}', '{"name":"Bob"}', "", "I cannot answer", "null", "123"]
    results = [utils.extractbench_process_results(row(), [response]) for response in outputs]
    assert [result["extractbench_value_f1"] for result in results] == [1, 0, 0, 0, 0, 0]
    assert sum(result["extractbench_value_f1"] for result in results) / len(results) == 1 / 6
    json.dumps(results, allow_nan=False)
    with pytest.raises(TypeError, match="transport-failure"):
        utils.parse_prediction(None)


@pytest.mark.usefixtures("official")
def test_reordered_missing_and_duplicate_nested_records():
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "detail": {"type": "object", "properties": {"amount": {"type": "number"}}},
                    },
                },
            }
        },
    }
    records = [{"name": "A", "detail": {"amount": 1}}, {"name": "B", "detail": {"amount": 2}}]
    doc = row({"rows": records}, schema)

    def score(values):
        return utils.extractbench_process_results(doc, [json.dumps({"rows": values})])["extractbench_value_f1"]

    assert score(list(reversed(records))) == 1.0
    assert 0 < score(records[:1]) < 1
    assert score(records + records[:1]) < 1


@pytest.mark.usefixtures("official")
def test_alternative_accepted_value_uses_unified_scorer():
    doc = row(rules={"name": {"expected_value": "Alice", "evidence": [{"value": "Alicia"}]}})
    result = utils.extractbench_process_results(doc, ['{"name":"Alicia"}'])
    assert result["extractbench_value_f1"] == result["extractbench_perfect"] == 1.0


@pytest.mark.usefixtures("official")
def test_official_list_and_eob_normalization():
    singleton = utils.extractbench_process_results(row(), ['[{"name":"Alice"}]'])
    assert singleton["extractbench_value_f1"] == 1.0
    expected = {"payment_details": [{"claims": [{"claim_id": "C1", "amount": 2}]}]}
    prediction = {"payment_details": {}, "claims": [{"claim_id": "C1", "amount": 2}]}
    result = utils.extractbench_process_results(row(expected, {"type": "object"}), [json.dumps(prediction)])
    assert result["extractbench_value_f1"] == 1.0


@pytest.mark.usefixtures("official")
def test_salvage_retains_partial_credit():
    expected = {"name": "Alice", "city": "Paris"}
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "city": {"type": "string"}}}
    result = utils.extractbench_process_results(row(expected, schema), ['{"name":"Alice","city":'])
    assert 0 < result["extractbench_value_f1"] < 1
    assert result["extractbench_perfect"] == 0.0
    assert result["extractbench_salvaged"] == result["extractbench_parse_success"] == 1.0


@pytest.mark.usefixtures("official")
def test_scorer_failures_are_not_model_scores(monkeypatch):
    def broken_scorer(*args):
        raise RuntimeError("scorer failed")

    monkeypatch.setattr(utils, "_evaluator", lambda: SimpleNamespace(compute_metrics=broken_scorer))
    with pytest.raises(RuntimeError, match="scorer failed"):
        utils.extractbench_process_results(row(), ['{"name":"Alice"}'])


@pytest.mark.usefixtures("official")
@pytest.mark.parametrize("response", ["[1]", "[null]", '[{"name":"Alice"}, 1]', "[[1]]"])
def test_unscoreable_json_arrays_receive_zero(response):
    result = utils.extractbench_process_results(row(), [response])
    assert result["extractbench_value_f1"] == result["extractbench_perfect"] == 0.0
    assert result["extractbench_parse_success"] == 1.0
    assert utils.extractbench_process_results(row(), ['{"name":"Alice"}'])["extractbench_value_f1"] == 1.0


@pytest.mark.usefixtures("official")
@pytest.mark.parametrize("fenced", [False, True])
def test_excessive_integer_is_a_model_parse_failure(fenced):
    limit = sys.get_int_max_str_digits()
    if limit == 0:
        pytest.skip("Python integer conversion limit is disabled")
    response = '{"number":' + "1" * (limit + 1) + "}"
    if fenced:
        response = "```json\n" + response + "\n```"
    result = utils.extractbench_process_results(row(), [response])
    assert result["extractbench_value_f1"] == result["extractbench_parse_success"] == 0.0


@pytest.mark.usefixtures("official")
def test_excessive_nesting_is_a_model_parse_failure():
    response = "[" * 11000 + "0" + "]" * 11000
    result = utils.extractbench_process_results(row(), [response])
    assert result["extractbench_value_f1"] == result["extractbench_parse_success"] == 0.0


@pytest.mark.usefixtures("official")
@pytest.mark.parametrize("budget_exceeded", [False, True])
def test_openai_failure_is_not_a_prediction(monkeypatch, budget_exceeded):
    from lmms_eval.models.chat import openai as backend

    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("synthetic endpoint unavailable")

    model = SimpleNamespace(
        rank=1,
        num_concurrent=1,
        adaptive_config=SimpleNamespace(max_concurrency=1),
        prefix_aware_queue=False,
        adaptive_concurrency=False,
        max_retries=1,
        strict_generation=True,
        timeout=1800,
        return_response_json=False,
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail))),
        task_dict={"extractbench_short": {"short": [{}]}},
        video_fps=None,
        max_frames_num=1,
        pass_video_url=False,
        audio_format="audio_url",
        audio_encoding="base64",
        model_version="synthetic-fixture",
        token_limit_field="max_tokens",
        image_detail=None,
        enable_thinking_kwarg=None,
    )
    request = SimpleNamespace(
        args=(
            "",
            lambda doc: [{"role": "user", "content": [{"type": "text", "text": "synthetic"}]}],
            {"max_new_tokens": 10},
            0,
            "extractbench_short",
            "short",
        )
    )
    monkeypatch.setattr(backend, "is_budget_exceeded", lambda: budget_exceeded)
    sentinel = backend.OpenAICompatible.generate_until(model, [request])[0].text
    assert len(calls) == (0 if budget_exceeded else 1)
    with pytest.raises(RuntimeError, match="backend failure sentinel"):
        utils.extractbench_process_results(row(), [sentinel])
    # A real JSON string field containing that text is still model output.
    result = utils.extractbench_process_results(row(), [json.dumps({"name": sentinel})])
    assert result["extractbench_value_f1"] == 0.0
    assert result["extractbench_parse_success"] == 1.0


@pytest.mark.usefixtures("official")
def test_framework_simple_and_chat_scoring(monkeypatch):
    from lmms_eval.api.task import ConfigurableMessagesTask, ConfigurableTask
    from lmms_eval.utils import load_yaml_config

    dataset = datasets.DatasetDict({"short": datasets.Dataset.from_list([row()])})
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(utils, "extractbench_doc_to_visual", lambda doc: [Image.new("RGB", (2, 2))])
    config = load_yaml_config(str(TASK_DIR / "extractbench_short.yaml"))
    for task_type in (ConfigurableTask, ConfigurableMessagesTask):
        task = task_type(config=dict(config))
        doc = task.dataset_no_image["short"][0]
        values = [task.process_results(doc, [response]) for response in ('{"name":"Alice"}', "")]
        assert task.aggregation()["extractbench_value_f1"]([value["extractbench_value_f1"] for value in values]) == 0.5
        assert task.aggregation()["extractbench_perfect"]([value["extractbench_perfect"] for value in values]) == 0.5
