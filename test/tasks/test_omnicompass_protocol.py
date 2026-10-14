"""Small protocol regressions without network access or model weights."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from lmms_eval.tasks.omnicompass import utils
from lmms_eval.tasks.omnicompass._core.bridge import evaluate, prediction
from lmms_eval.tasks.omnicompass._core.run_identity import model_identity


def documents(task: str = "QA", track: str = "VL") -> list[dict]:
    """Build one paired question with fixed ground truth."""
    gold = {
        "QA": {"answer": "C"},
        "TG": {"start_time": 1.0, "end_time": 3.0},
        "SG": {"start_bbox": [0.1, 0.1, 0.5, 0.5], "end_bbox": [0.2, 0.2, 0.6, 0.6]},
    }
    question = {
        "question_id": f"{track}_test_{task}",
        "video_id": "oc_test",
        "track": track,
        "task": task,
        "given_interval": [1.0, 3.0],
        "ground_truth": gold[task],
    }
    return [
        {
            "question_id": question["question_id"],
            "video": "oc_test",
            "track": track,
            "task": task,
            "setting": setting,
            "question_json": json.dumps(question),
            "protocol_id": "test",
        }
        for setting in ("text", "text_image")
    ]


def response(content: str, *, status: int = 200, finish_reason: str = "stop") -> str:
    """Encode the backend's response envelope."""
    body = {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]} if status == 200 else {"error": "temporarily unavailable"}
    return json.dumps({"api_response": {"status_code": status, "body": body}})


@pytest.mark.parametrize("track", ["VL", "AV"])
@pytest.mark.parametrize(
    ("task", "answer"),
    [
        ("QA", "C"),
        ("TG", "[1, 3]"),
        ("SG", '{"start_time":1,"end_time":3,"start_bbox":[0.1,0.1,0.5,0.5],"end_bbox":[0.2,0.2,0.6,0.6]}'),
    ],
)
def test_perfect_pair_scores_one(track: str, task: str, answer: str) -> None:
    docs = documents(task, track)
    rows = [{"doc": doc, "output": response(answer)} for doc in docs]
    assert utils.aggregate_text(rows) == pytest.approx(1.0)
    assert utils.aggregate_text_image(rows) == pytest.approx(1.0)
    assert utils.aggregate_cs(rows) == pytest.approx(1.0)
    assert utils.aggregate_cs(list(reversed(rows))) == pytest.approx(1.0)


def test_cs_requires_both_qa_answers_correct() -> None:
    docs = documents()
    rows = [{"doc": docs[0], "output": "C"}, {"doc": docs[1], "output": "A"}]
    assert utils.aggregate_text(rows) == 1.0
    assert utils.aggregate_text_image(rows) == 0.0
    assert utils.aggregate_cs(rows) == 0.0


@pytest.mark.parametrize("answer", ["", "unparseable", "<think>unfinished"])
def test_completed_invalid_answer_remains_zero(answer: str) -> None:
    docs = documents()
    result = evaluate(docs, [response(answer, finish_reason="length"), response("C")])
    track = result["tracks"]["VL"]
    assert track["complete"]
    assert len(track["scoring_ids"]) == 2
    assert track["paired_cs"]["tasks"]["QA"]["value"] == 0.0


def test_transport_failure_does_not_emit_successful_scores() -> None:
    with pytest.raises(ValueError, match="transport|incomplete|Incomplete"):
        evaluate(documents(), [response("", status=503), response("C")])


def test_missing_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="Missing"):
        evaluate(documents()[:1], ["C"])


def test_final_refusal_excluded_and_metadata_retained() -> None:
    docs = documents()
    result = evaluate(docs, [response("", finish_reason="content_filter"), response("C")])
    track = result["tracks"]["VL"]
    assert track["complete"]
    assert len(track["policy_excluded_ids"]) == 1
    assert track["paired_cs"]["tasks"]["QA"]["scored_pairs"] == 0
    row = prediction(docs[0], response("<think>reasoning</think>C", finish_reason="length"))
    assert row["raw_output"] == "C"
    assert row["response_meta"]["finish_reason"] == "length"


def test_av_requires_explicit_audio_support() -> None:
    with pytest.raises(ValueError, match="separate-audio"):
        utils.doc_to_messages(documents(track="AV")[0])


def test_model_hash_works_without_python311_file_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"test model")
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    assert model_identity(path)["local_artifacts_sha256"] == {path.name: hashlib.sha256(b"test model").hexdigest()}


@pytest.mark.parametrize("track", ["vl", "av"])
def test_overall_is_unweighted_three_task_mean(track: str) -> None:
    root = Path(utils.__file__).parent
    group = yaml.safe_load((root / f"omnicompass_{track}.yaml").read_text())
    assert len(group["task"]) == 3
    assert {entry["metric"] for entry in group["aggregate_metric_list"]} == {"text", "text_image", "cs"}
    assert all(entry["aggregation"] == "mean" and not entry["weight_by_size"] for entry in group["aggregate_metric_list"])
