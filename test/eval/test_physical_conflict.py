import json
import os
from pathlib import Path

import pytest

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.physical_conflict import utils


def _presence_qa(sample_id: str, *, answer: str = "B") -> dict:
    return {
        "qa_id": f"{sample_id}__conflict_presence",
        "question_type": "conflict_presence",
        "question": "Does this video contain a physical conflict?",
        "answer_type": "single_choice",
        "options": [{"id": "A", "text": "No"}, {"id": "B", "text": "Yes"}],
        "answer": answer,
        "evidence": [{"type": "canonical_label", "value": "physical_conflict"}],
        "generator_version": "task_c_atomic_v1",
    }


def _quarter_qa(sample_id: str) -> dict:
    return {
        "qa_id": f"{sample_id}__conflict_quarter_coverage",
        "question_type": "conflict_quarter_coverage",
        "question": "Which quarters of the video contain a physical conflict?",
        "answer_type": "multiple_choice",
        "options": [
            {"id": "A", "text": "Q1"},
            {"id": "B", "text": "Q2"},
            {"id": "C", "text": "Q3"},
            {"id": "D", "text": "Q4"},
        ],
        "answer": ["A", "C"],
        "evidence": [{"type": "event", "event_id": "event_000001"}],
        "generator_version": "task_c_atomic_v1",
    }


def _numeric_qa(sample_id: str) -> dict:
    return {
        "qa_id": f"{sample_id}__first_conflict_duration",
        "question_type": "first_conflict_duration",
        "question": "How long does the first physical conflict last?",
        "answer_type": "single_choice",
        "options": [
            {"id": "A", "text": "2.00"},
            {"id": "B", "text": "5.00"},
            {"id": "C", "text": "8.00"},
            {"id": "D", "text": "11.00"},
        ],
        "answer": "B",
        "evidence": [{"type": "event", "event_id": "event_000001"}],
        "generator_version": "task_c_atomic_v1",
        "target_value": 5.0,
        "unit": "seconds",
        "tolerance": 0.5,
    }


def _sample(sample_id: str = "sample_001", *, source: str = "ubi_fights", qa_pairs: list[dict] | None = None) -> dict:
    return {
        "id": sample_id,
        "video_path": f"0001__{source}__fixture__video.mp4",
        "video_time_sec": 12.0,
        "annotation": {"events": []},
        "qa_pairs": qa_pairs if qa_pairs is not None else [_presence_qa(sample_id)],
    }


def _install_targets(targets: dict[str, utils._Target]) -> None:
    utils._runtime.targets = targets


def test_physical_conflict_task_is_registered():
    assert "physical_conflict" in TaskManager("ERROR").all_tasks


def test_atomic_projection_uses_exact_safe_allowlist():
    rows, targets, media_index, sidecar = utils._project_main_samples([_sample()], enforce_release_contract=False)

    assert len(rows) == len(targets) == len(sidecar) == 1
    assert set(rows[0]) == set(utils.MODEL_INPUT_FIELDS)
    assert not (utils.FORBIDDEN_MODEL_INPUT_FIELDS & rows[0].keys())
    assert "ubi_fights" not in json.dumps(rows[0])
    assert media_index == {"sample_001": "0001__ubi_fights__fixture__video.mp4"}


def test_atomic_projection_rejects_old_grouped_qa_shape():
    sample = _sample()
    sample["qa_pairs"] = [
        {
            "qa_id": "legacy",
            "questions": ["Does this video contain conflict?"],
            "answer": ["yes"],
        }
    ]

    with pytest.raises(ValueError, match="missing"):
        utils._project_main_samples([sample], enforce_release_contract=False)


def test_sidecar_must_be_exact_closed_derivative():
    _, _, _, expected = utils._project_main_samples([_sample()], enforce_release_contract=False)
    utils._validate_sidecar(expected, expected)

    changed = [dict(expected[0], answer="A")]
    with pytest.raises(ValueError, match="exact derivative"):
        utils._validate_sidecar(changed, expected)


def test_split_projection_filters_by_opaque_sample_id():
    samples = [_sample("sample_001"), _sample("sample_002", source="rwf2000")]
    rows, _, media_index, _ = utils._project_main_samples(samples, enforce_release_contract=False)
    split_ids = utils._split_sample_ids([{"id": "sample_002"}], valid_sample_ids=set(media_index), enforce_release_contract=False)

    projected = utils._project_test_rows(rows, split_ids, enforce_release_contract=False)

    assert [row["sample_id"] for row in projected] == ["sample_002"]


def test_prompt_contains_options_but_not_target_or_raw_path():
    rows, _, _, _ = utils._project_main_samples([_sample()], enforce_release_contract=False)

    prompt = utils.physical_conflict_doc_to_text(rows[0])

    assert "A. No" in prompt
    assert "B. Yes" in prompt
    assert "0001__ubi_fights" not in prompt
    assert "physical_conflict" not in prompt


def test_single_choice_prediction_is_strict_option_id():
    sample_id = "sample_001"
    rows, targets, _, _ = utils._project_main_samples([_sample(sample_id)], enforce_release_contract=False)
    _install_targets(targets)
    doc = rows[0]

    assert utils.physical_conflict_normalize_prediction(doc, "B") == "B"
    assert utils.physical_conflict_normalize_prediction(doc, "B.") is None
    assert utils.physical_conflict_normalize_prediction(doc, "The answer is B") is None


def test_multiple_choice_prediction_requires_json_array():
    sample_id = "sample_001"
    sample = _sample(sample_id, source="ntu_cctv_fights", qa_pairs=[_quarter_qa(sample_id), _presence_qa(sample_id)])
    rows, targets, _, _ = utils._project_main_samples([sample], enforce_release_contract=False)
    _install_targets(targets)
    doc = rows[0]

    assert utils.physical_conflict_normalize_prediction(doc, '["C","A"]') == ["A", "C"]
    assert utils.physical_conflict_normalize_prediction(doc, "AC") is None
    assert utils.physical_conflict_normalize_prediction(doc, '["A","A"]') is None


def test_multiple_choice_metrics_match_exact_set_semantics():
    sample_id = "sample_001"
    sample = _sample(sample_id, source="ntu_cctv_fights", qa_pairs=[_quarter_qa(sample_id), _presence_qa(sample_id)])
    rows, targets, _, _ = utils._project_main_samples([sample], enforce_release_contract=False)
    _install_targets(targets)

    result = utils.physical_conflict_process_results(rows[0], ['["A","B"]'])

    assert result["Overall_Exact_Accuracy"] == 0.0
    assert result["MultipleChoice_Exact_Set_Accuracy"] == 0.0
    assert result["MultipleChoice_Macro_Precision"] == 0.5
    assert result["MultipleChoice_Macro_Recall"] == 0.5


def test_numeric_metrics_use_selected_option_and_hidden_target():
    sample_id = "sample_001"
    sample = _sample(sample_id, source="ntu_cctv_fights", qa_pairs=[_numeric_qa(sample_id), _presence_qa(sample_id)])
    rows, targets, _, _ = utils._project_main_samples([sample], enforce_release_contract=False)
    _install_targets(targets)

    correct = utils.physical_conflict_process_results(rows[0], ["B"])
    wrong = utils.physical_conflict_process_results(rows[0], ["A"])

    assert correct["Numeric_Option_MAE"] == 0.0
    assert correct["Numeric_Accuracy_at_0_5s"] == 1.0
    assert wrong["Numeric_Option_MAE"] == 3.0
    assert wrong["Numeric_Accuracy_at_0_5s"] == 0.0


def test_opaque_media_alias_hides_source_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    video = tmp_path / "0001__ubi_fights__fight__source.mp4"
    video.write_bytes(b"fixture")
    alias_root = tmp_path / "opaque"
    monkeypatch.setenv("PHYSICAL_CONFLICT_OPAQUE_MEDIA_ROOT", str(alias_root))
    utils._runtime.media_index = {"sample_001": str(video)}

    resolved = Path(utils.physical_conflict_doc_to_visual({"sample_id": "sample_001"})[0])

    assert resolved.name == "sample_001.mp4"
    assert resolved.is_symlink()
    assert resolved.resolve() == video.resolve()


@pytest.mark.skipif(not os.environ.get("GOODVISION_REPO_ROOT"), reason="GOODVISION_REPO_ROOT is not configured")
def test_current_goodvision_release_projects_to_334_test_questions(monkeypatch: pytest.MonkeyPatch):
    root = Path(os.environ["GOODVISION_REPO_ROOT"])
    monkeypatch.syspath_prepend(str(root / "src"))
    from numera_vision.task_c_evaluation import (
        load_model_inputs,
        load_trusted_media_index,
    )

    main_path = root / "data/samples/v1/clean_samples_1000_human_reviewed_with_qa.jsonl"
    sidecar_path = root / "data/qa/v1/task_c_qa_evaluation_items_v1.jsonl"
    split_path = root / "data/samples/v1/splits/test.jsonl"

    assert utils._validate_hash(main_path, utils.DEFAULT_MAIN_SHA256, artifact_name="Task C main")
    assert utils._validate_hash(sidecar_path, utils.DEFAULT_SIDECAR_SHA256, artifact_name="Task C sidecar")
    assert utils._validate_hash(split_path, utils.DEFAULT_TEST_SPLIT_SHA256, artifact_name="Task C test split")
    samples = utils._read_jsonl(main_path, artifact_name="Task C main")
    rows, _, media_index, expected_sidecar = utils._project_main_samples(samples, enforce_release_contract=True)
    sidecar = utils._read_jsonl(sidecar_path, artifact_name="Task C sidecar")
    utils._validate_sidecar(sidecar, expected_sidecar)
    split = utils._read_jsonl(split_path, artifact_name="Task C test split")
    split_ids = utils._split_sample_ids(split, valid_sample_ids=set(media_index), enforce_release_contract=True)
    projected = utils._project_test_rows(rows, split_ids, enforce_release_contract=True)

    assert rows == load_model_inputs(main_path)
    assert media_index == load_trusted_media_index(main_path)
    assert len(projected) == 334
    assert len({row["sample_id"] for row in projected}) == 149
    assert set(projected[0]) == set(utils.MODEL_INPUT_FIELDS)
