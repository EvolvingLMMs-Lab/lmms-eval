import math

import pytest

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.physical_conflict import utils


def _sample_with_one_qa() -> dict:
    return {
        "id": "sample_001",
        "video_path": "sample.mp4",
        "video_time_sec": 20.0,
        "annotation": {"events": []},
        "qa_pairs": [
            {
                "qa_id": "qa_001",
                "questions": ["Does this video contain any physical conflict?"],
                "answer": ["yes"],
            }
        ],
    }


def _matching_sidecar(**overrides) -> dict:
    item = {
        "item_id": "qa_001:0",
        "sample_id": "sample_001",
        "qa_id": "qa_001",
        "question_index": 0,
        "question": "Does this video contain any physical conflict?",
        "canonical_answer": "yes",
        "question_type": "physical_conflict_existence",
        "answer_type": "binary",
    }
    item.update(overrides)
    return item


def test_physical_conflict_task_is_registered():
    assert "physical_conflict" in TaskManager("ERROR").all_tasks


def test_full_task_rejects_binary_only_category_coverage():
    rows = [{"question_category": "physical_conflict_existence"}]

    with pytest.raises(ValueError, match="Missing required question categories"):
        utils._validate_required_categories(rows, utils._REQUIRED_QUESTION_CATEGORIES)


def test_full_task_accepts_all_six_question_categories():
    rows = [{"question_category": category} for category in utils._REQUIRED_QUESTION_CATEGORIES]

    counts = utils._validate_required_categories(rows, utils._REQUIRED_QUESTION_CATEGORIES)

    assert set(counts) == utils._REQUIRED_QUESTION_CATEGORIES


def test_lineage_validation_rejects_stale_task_b_base():
    samples = [_sample_with_one_qa()]
    expected_hash = utils._canonical_base_sha256(samples)
    assert (
        utils._validate_lineage(
            samples,
            lineage="official",
            expected_sha256=expected_hash,
            expected_sample_count=1,
        )
        == expected_hash
    )

    samples[0]["annotation"] = {"events": [{"event_id": "unexpected"}]}
    with pytest.raises(ValueError, match="not aligned"):
        utils._validate_lineage(
            samples,
            lineage="official",
            expected_sha256=expected_hash,
            expected_sample_count=1,
        )


def test_sidecar_rejects_duplicate_identity():
    sidecar_item = _matching_sidecar()

    with pytest.raises(ValueError, match="Duplicate sidecar item_id"):
        utils._sidecar_indexes([sidecar_item, dict(sidecar_item)])


def test_strict_join_rejects_missing_sidecar_item():
    with pytest.raises(ValueError, match="Missing sidecar item"):
        utils._atomic_rows(
            [_sample_with_one_qa()],
            split_records=[],
            sidecar_records=[],
            default_tolerance=0.5,
            require_sidecar=True,
        )


@pytest.mark.parametrize(
    ("sidecar_overrides", "message"),
    [
        ({"question": "A different question"}, "question mismatch"),
        ({"canonical_answer": "no"}, "answer mismatch"),
    ],
)
def test_strict_join_rejects_main_sidecar_content_mismatch(sidecar_overrides, message):
    with pytest.raises(ValueError, match=message):
        utils._atomic_rows(
            [_sample_with_one_qa()],
            split_records=[],
            sidecar_records=[_matching_sidecar(**sidecar_overrides)],
            default_tolerance=0.5,
            require_sidecar=True,
        )


def test_strict_join_rejects_unmatched_sidecar_item():
    extra = _matching_sidecar(
        item_id="qa_extra:0",
        qa_id="qa_extra",
        question="Does another video contain conflict?",
    )

    with pytest.raises(ValueError, match="Unmatched sidecar item"):
        utils._atomic_rows(
            [_sample_with_one_qa()],
            split_records=[],
            sidecar_records=[_matching_sidecar(), extra],
            default_tolerance=0.5,
            require_sidecar=True,
        )


def test_full_sidecar_entries_outside_selected_split_are_valid():
    second_sample = {
        "id": "sample_002",
        "video_path": "second.mp4",
        "qa_pairs": [
            {
                "qa_id": "qa_002",
                "questions": ["Does this video contain any physical conflict?"],
                "answer": ["no"],
            }
        ],
    }
    second_sidecar = _matching_sidecar(
        item_id="qa_002:0",
        sample_id="sample_002",
        qa_id="qa_002",
        canonical_answer="no",
    )

    rows = utils._atomic_rows(
        [_sample_with_one_qa(), second_sample],
        split_records=[{"id": "sample_001"}],
        sidecar_records=[_matching_sidecar(), second_sidecar],
        default_tolerance=0.5,
        require_sidecar=True,
    )

    assert [row["sample_id"] for row in rows] == ["sample_001"]


def test_conflict_quarter_presence_is_multiselect_and_parses_conjunction():
    row = utils._normalize_atomic_row(
        sample={"id": "sample_001", "video_path": "sample.mp4"},
        question="Which quarters contain physical conflict?",
        target="A and C",
        qa_id="qa_quarters",
        question_index=0,
        metadata={
            "question_type": "conflict_quarter_presence",
            "choices": ["Q1", "Q2", "Q3", "Q4"],
        },
        default_tolerance=0.5,
    )

    assert row["answer_type"] == "multi_select"
    assert row["answer_text"] == "AC"
    assert row["question_category"] == "conflict_quarters"


def test_invalid_numeric_prediction_has_finite_penalty_and_parse_error():
    doc = {
        "answer_type": "numeric",
        "answer_text": "12.5",
        "target_value": 12.5,
        "tolerance": 0.5,
        "video_time_sec": 20.0,
    }

    result = utils.physical_conflict_process_results(doc, ["I cannot determine it"])

    assert result["Numeric_Accuracy_at_0_5s"] == 0.0
    assert result["Numeric_Parse_Error_Rate"] == 1.0
    assert result["Numeric_MAE"] == 20.0
    assert math.isfinite(result["Numeric_MAE"])


def test_valid_numeric_prediction_has_no_parse_error():
    doc = {
        "answer_type": "numeric",
        "answer_text": "12.5",
        "target_value": 12.5,
        "tolerance": 0.5,
        "video_time_sec": 20.0,
    }

    result = utils.physical_conflict_process_results(doc, ["12.8 seconds"])

    assert result["Numeric_Accuracy_at_0_5s"] == 1.0
    assert result["Numeric_Parse_Error_Rate"] == 0.0
    assert result["Numeric_MAE"] == pytest.approx(0.3)
