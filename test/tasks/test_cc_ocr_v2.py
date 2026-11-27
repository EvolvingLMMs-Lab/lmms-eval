import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
from datasets import Image as DatasetImage
from datasets import Sequence, load_from_disk
from PIL import Image

from lmms_eval.protocol import ChatMessages
from lmms_eval.tasks.cc_ocr_v2 import prepare_dataset as prep
from lmms_eval.tasks.cc_ocr_v2 import utils
from lmms_eval.tasks.cc_ocr_v2.scoring import aggregate_results, aggregate_track, scenario_score, score_sample


def test_ordered_pages_then_verbatim_question():
    pages = [Image.new("RGB", (5, 7), "red"), Image.new("RGB", (3, 4), "blue")]
    doc = {"images": pages, "question": "Which number appears on page two?\n", "answer": "private target"}
    [message] = utils.doc_to_messages(doc)
    assert message["role"] == "user"
    assert message["content"] == [
        {"type": "image", "url": pages[0]},
        {"type": "image", "url": pages[1]},
        {"type": "text", "text": doc["question"]},
    ]
    assert utils.doc_to_visual(doc) == pages
    assert utils.doc_to_text(doc) == doc["question"]
    wire = ChatMessages(messages=[message]).to_openai_messages()
    assert [part["type"] for part in wire[0]["content"]] == ["image_url", "image_url", "text"]
    assert wire[0]["content"][-1]["text"] == doc["question"]


@pytest.mark.parametrize("response", ["", " \n"])
def test_empty_response_contributes_to_recognition_and_overall(response):
    doc = {
        "track": "recognition",
        "sub_task": "multilingual_recognition",
        "scenario": "latin",
        "source_key": "recognition/multilingual_recognition/latin/1",
        "answer": "hello",
    }
    metrics = utils.process_results(doc, [response])
    assert set(metrics) == {"cc_ocr_v2_average", "cc_ocr_v2_recognition"}
    assert metrics["cc_ocr_v2_average"] == metrics["cc_ocr_v2_recognition"]
    assert metrics["cc_ocr_v2_recognition"]["row_score"] == 0.0
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize(
    "failure",
    [
        "[LMMS_EVAL_REQUEST_FAILED after 3 retries] connection timed out",
        "[LMMS_EVAL_BUDGET_EXCEEDED]",
    ],
)
def test_failed_request_cannot_be_scored_as_a_model_answer(failure):
    doc = {"source_key": "qa/forms/fields/7", "track": "qa", "sub_task": "forms", "scenario": "fields", "answer": "7"}
    with pytest.raises(RuntimeError, match=doc["source_key"]):
        utils.process_results(doc, [failure])


@pytest.fixture()
def snapshot(tmp_path):
    root = tmp_path / "snapshot"
    for track, sub_task in [("extraction", "transactions"), ("grounding", "objects")]:
        task = root / track / sub_task
        for kind in ("question", "answer", "images"):
            (task / kind / "shared").mkdir(parents=True)
        (task / "question/shared/doc.txt").write_text("  Read this.\n", encoding="utf-8")

    single = root / "grounding/objects"
    (single / "answer/shared/doc.txt").write_text("42\n", encoding="utf-8")
    Image.new("RGB", (7, 8), "blue").save(single / "images/shared/doc.JPG")

    multi = root / "extraction/transactions"
    (multi / "answer/shared/doc.json").write_bytes('{"值": "1"}\r\n'.encode())
    folder = multi / "images/shared/doc"
    folder.mkdir()
    Image.new("RGBA", (10, 9), (151, 22, 3, 127)).save(folder / "page_10.png")
    Image.new("L", (2, 3), 64).save(folder / "page_2.png")
    return root


def test_pinned_preparation_round_trip(snapshot, tmp_path, monkeypatch):
    def fake_download(**kwargs):
        assert (kwargs["repo_id"], kwargs["repo_type"], kwargs["revision"]) == (prep.REPO_ID, "dataset", prep.REVISION)
        return str(snapshot)

    # Substitute only the synthetic release inventory and download transport.
    files = sorted(p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file())
    fixture_sha256 = hashlib.sha256(("\n".join(files) + "\n").encode()).hexdigest()
    monkeypatch.setattr(prep, "snapshot_download", fake_download)
    monkeypatch.setattr(prep, "RELEASE_PATHS_SHA256", fixture_sha256)
    output_dir = tmp_path / "prepared"
    prep.prepare_dataset(output_dir)
    data = load_from_disk(str(output_dir))
    assert list(data) == ["test"]
    docs = data["test"]
    assert list(docs["source_key"]) == ["extraction/transactions/shared/doc", "grounding/objects/shared/doc"]
    row, single = docs[0], docs[1]
    assert (row["track"], row["sub_task"], row["scenario"]) == ("extraction", "transactions", "shared")
    assert (row["question"], row["answer"], single["answer"]) == ("  Read this.\n", '{"值": "1"}\r\n', "42\n")
    assert row["image_widths"] == [2, 10]
    assert row["image_heights"] == [3, 9]
    assert [(p.format, p.mode, p.size) for p in row["images"]] == [("JPEG", "RGB", (2, 3)), ("JPEG", "RGB", (10, 9))]
    assert single["images"][0].size == (7, 8)

    raw_images = docs.cast_column("images", Sequence(DatasetImage(decode=False)))[0]["images"]
    for page, encoded in zip(["page_2.png", "page_10.png"], raw_images, strict=True):
        with (
            Image.open(snapshot / "extraction/transactions/images/shared/doc" / page) as original,
            BytesIO() as expected,
        ):
            original.convert("RGB").save(expected, format="JPEG", quality=95, optimize=True)
            assert encoded["bytes"] == expected.getvalue()


def test_rejects_incomplete_but_consistently_paired_snapshot(snapshot):
    with pytest.raises(ValueError, match="file inventory"):
        prep.validate_release(snapshot, prep.collect_samples(snapshot))


def test_reports_unmatched_image_key(snapshot):
    orphan = snapshot / "grounding/objects/images/shared/no-question.jpg"
    Image.new("RGB", (1, 1)).save(orphan)
    with pytest.raises(ValueError, match="Unmatched question/images key: grounding/objects/shared/no-question"):
        prep.collect_samples(snapshot)


def test_failed_write_keeps_destination_unpublished(snapshot, tmp_path, monkeypatch):
    def fail_after_first_write(self, target):
        Path(target).mkdir()
        (Path(target) / "partial").touch()
        raise OSError("interrupted")

    monkeypatch.setattr(prep.DatasetDict, "save_to_disk", fail_after_first_write)
    output_dir = tmp_path / "result"
    with pytest.raises(OSError, match="interrupted"):
        prep.save_dataset(prep.collect_samples(snapshot), output_dir)
    assert not output_dir.exists()
    assert list(tmp_path.glob(".result-*")) == []


def test_existing_destination_preserved_before_download(tmp_path):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    sentinel = output_dir / "keep"
    sentinel.write_text("existing")
    with pytest.raises(FileExistsError):
        prep.prepare_dataset(output_dir)
    assert sentinel.read_text() == "existing"


def _row(track, answer, prediction, *, sub_task="", scenario="scene", key="id"):
    doc = {
        "source_key": key,
        "track": track,
        "sub_task": sub_task,
        "scenario": scenario,
        "answer": answer,
        "image_widths": [200],
        "image_heights": [100],
    }
    # Distributed workers must be able to return results without images or Python-specific types.
    return json.loads(json.dumps(score_sample(doc, prediction), allow_nan=False))


def test_micro_f1_is_computed_inside_each_scenario():
    recognition = [_row("recognition", "A", "a"), _row("recognition", "b " * 9, "", key="2")]
    extraction = [
        _row("extraction", '{"f":"x"}', '{"f":"x"}'),
        _row("extraction", json.dumps({"f": ["y"] * 9}), "invalid", key="2"),
    ]
    assert scenario_score(recognition) == pytest.approx(2 / 11)
    assert scenario_score(extraction) == 0.1818
    # A second scenario gets equal weight, independent of how many rows it contains.
    assert aggregate_track([*recognition, _row("recognition", "z", "z", scenario="other")], "recognition") == 0.590909


def test_recognition_uses_chinese_character_and_scene_specific_word_rules():
    assert _row("recognition", "汉汉文", "```json\n汉 文\n```")["row_score"] == pytest.approx(0.8)
    assert _row("recognition", "a-b", "ab", scenario="multi_scene_ocr_street")["row_score"] == pytest.approx(1)
    assert _row("recognition", "a-b", "ab", scenario="multilingual")["row_score"] == 0


def test_kie_parses_fences_and_retains_value_multiplicity():
    gold = json.dumps({"total": "￥１１–２℃。", "lines": [{"sku": "A"}, {"sku": "A"}]})
    pred = '```json\n{"total":"¥112°C","lines":[{"sku":"A"}]}\n```'
    row = _row("extraction", gold, pred)
    assert (row["matched"], row["predicted"], row["expected"]) == (2, 2, 3)
    assert row["row_score"] == pytest.approx(4 / 5)
    assert scenario_score([_row("extraction", gold, "not JSON")]) is None
    assert scenario_score([_row("extraction", gold, "{}")]) == 0.0


def test_qa_scalar_and_list_branches():
    assert _row("qa", "甲乙丙丁", "甲乙丙戊")["row_score"] == 0.75
    assert _row("qa", '["甲乙丙丁"]', "甲乙丙戊")["row_score"] == 0
    assert _row("qa", "甲,乙,丙,丁", "甲,乙,丙,丁")["row_score"] == 0
    assert _row("qa", '["甲,乙,丙,丁"]', "甲,乙,丙,丁")["row_score"] == 1
    assert _row("qa", "[2, 3]", "there are 3")["row_score"] == 1
    assert _row("qa", "abcdef", "abcdeg")["row_score"] == 0


def test_grounding_uses_original_size_and_assignment_over_ground_truth_boxes():
    gt = '[{"label":"a","bbox":[0,0,100,50]},{"label":"b","bbox":[100,50,200,100]}]'
    pred = '[{"label":"ignored","bbox":{"left":500,"top":500,"right":1000,"bottom":1000}}]'
    objects = _row("grounding", gt, pred, sub_task="object_grounding")
    assert objects["row_score"] == 0.5
    exact = _row("grounding", "(0,0,100,50)", "(0,0,0.5,0.5)", sub_task="text_grounding")
    miss = _row("grounding", "(0,0,100,50)", "no box", sub_task="text_grounding", scenario="other")
    assert exact["row_score"] == 1
    assert aggregate_track([exact, miss, objects], "grounding") == 0.5
    assert aggregate_track([exact], "grounding") == 0.5


def test_parsing_uses_original_reference_whitespace_and_markup():
    pytest.importorskip("lxml")
    formula = _row("parsing", "xy\n", "```latex\nx y\n```", scenario="handwritten_formula")
    assert formula["row_score"] == pytest.approx(2 / 3)
    assert _row("parsing", "CF", "<smiles>C F</smiles>", scenario="molecular")["row_score"] == 1
    assert _row("parsing", r"\alpha", "```latex\n\\alpha\n### Notes: ignored\n```", scenario="doc")["row_score"] == 1
    table = "<table><tr><td>Ｘ</td></tr></table>"
    assert _row("parsing", table, "```html\n" + table + "\n```", scenario="table")["row_score"] == 1
    assert _row("parsing", "<p>aa</p>" + table, "<p>bb</p>" + table, scenario="info_board_custom")["row_score"] == 0.9


def test_leaderboard_requires_five_tracks_and_applies_cli_rounding():
    rows = [
        _row("recognition", "a", "a"),
        _row("qa", "x", "wrong"),
        _row("parsing", "3", "", scenario="formula"),
        _row("grounding", "(0,0,20,10)", "(0,0,100,100)", sub_task="text_grounding"),
        _row("extraction", '{"a":[1,2]}', '{"a":1}'),
    ]
    assert aggregate_results(rows[:-1]) is None
    assert aggregate_results(rows) == pytest.approx((1 + 0.5 + 0.6667) / 5)


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_raw_predictions_match_the_official_text_file(tmp_path, newline):
    from lmms_eval.tasks.cc_ocr_v2.utils import process_results

    path = tmp_path / "prediction.txt"
    raw = f"one{newline}two"
    path.write_bytes(raw.encode("utf-8"))
    for track, scenario, answer in (("qa", "qa", "one two"), ("parsing", "formula", "onetwo")):
        doc = {"source_key": "sample", "track": track, "sub_task": scenario, "scenario": scenario, "answer": answer}
        responses = [raw]
        live = process_results(doc, responses)["cc_ocr_v2_average"]
        replay = score_sample(doc, path.read_text(encoding="utf-8"))
        assert live == replay
        assert live["row_score"] == 1
        assert responses == [raw]


@pytest.mark.parametrize(
    ("track", "sub_task", "scenario", "reference", "prediction"),
    [
        (
            "parsing",
            "complex_table_parsing",
            "table",
            "<table><tr><td>a</td></tr></table>",
            "<table><tr><td colspan=x>a</td></tr></table>",
        ),
        ("grounding", "text_grounding", "scene", "(0,0,20,10)", "(0,0,1..0,1)"),
    ],
)
def test_fatal_official_scorer_errors_propagate(track, sub_task, scenario, reference, prediction):
    with pytest.raises(ValueError):
        _row(track, reference, prediction, sub_task=sub_task, scenario=scenario)


def test_invalid_grounding_reference_and_missing_box_keep_their_official_dispositions():
    invalid_reference = _row("grounding", "bad reference", "(0,0,1,1)", sub_task="text_grounding")
    missing_box = _row("grounding", "(0,0,20,10)", "no box", sub_task="text_grounding")
    assert invalid_reference["row_score"] is None
    assert missing_box["row_score"] == 0
    assert scenario_score([invalid_reference, missing_box]) == 0
