from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from datasets import Dataset

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.vrbench import utils as vrbench_utils
from lmms_eval.verifiers.base import VerifyResult

VIDEO_SUMMARY = "The detective returns to the harbour at night and confronts the smuggler."


def _qa(question: str, answer: str, reasoning_type: str) -> dict[str, Any]:
    return {
        "question": question,
        "options": {"A": "First", "B": "Second", "C": "Third", "D": "Fourth"},
        "answer": answer,
        "original_question": question,
        "original_answer": "Fourth" if answer == "D" else "First",
        "reasoning_process": {"1": "The detective boards the boat.", "2": "The smuggler runs away."},
        "reasoning_type": reasoning_type,
    }


def _record() -> dict[str, Any]:
    return {
        "video_id": "demo-video",
        "video_path": "VRBench/videos/v001/demo-video.mp4",
        "video_summary": VIDEO_SUMMARY,
        "video_read_type": "video",
        "mcq": {
            "qa1": _qa("Why did the detective return?", "D", "Implicit Inference"),
            "qa2": _qa("What happens next?", "A", "Event Prediction"),
            "qa10": _qa("Summarise the chase.", "B", "Event Summarization"),
        },
    }


def _doc(reasoning_type: str = "Implicit Inference", answer: str = "D") -> dict[str, Any]:
    return {
        "question_id": "demo-video_qa1",
        "qa_key": "qa1",
        "video_id": "demo-video",
        "video_path": "VRBench/videos/v001/demo-video.mp4",
        "video_read_type": "video",
        "video_summary": VIDEO_SUMMARY,
        "question": "Why did the detective return?",
        "options": {"A": "First", "B": "Second", "C": "Third", "D": "Fourth"},
        "answer": answer,
        "original_answer": "Fourth",
        "reasoning_process": "<Step 1> The detective boards the boat.\n<Step 2> The smuggler runs away.",
        "reasoning_type": reasoning_type,
    }


def _write_test_video(path: str, frame_count: int = 4, fps: int = 2) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (8, 8))
    assert writer.isOpened(), "OpenCV cannot create the temporary MP4 test fixture"
    for value in range(frame_count):
        writer.write(np.full((8, 8, 3), value * 50, dtype=np.uint8))
    writer.release()


class _RecordingPipeline:
    """Stand-in for the judge pipeline that records its call and never uses the network."""

    def __init__(self, route: str, calls: list[dict[str, Any]], judge_reply: str) -> None:
        self.route = route
        self.calls = calls
        self.judge_reply = judge_reply

    def __call__(self, question: str, prediction: str, ground_truth: str, **kwargs: Any) -> VerifyResult:
        self.calls.append({"route": self.route, "question": question, "prediction": prediction, "ground_truth": ground_truth, "kwargs": kwargs})
        return vrbench_utils._parse_rate_response(self.judge_reply)


@pytest.fixture
def judge_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the judge pipeline factory so no judge API is ever contacted."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        vrbench_utils,
        "_get_pipeline",
        lambda route: _RecordingPipeline(route, calls, "<rate>7</rate>.\n<reason>The steps match the annotation.</reason>"),
    )
    return calls


def test_vrbench_tasks_are_registered() -> None:
    task_manager = TaskManager("ERROR")
    expected = {"vrbench_mcq", "vrbench_process", "vrbench"}
    assert not expected.difference(task_manager.all_tasks)


def test_vrbench_process_docs_flattens_one_document_per_question() -> None:
    docs = vrbench_utils.vrbench_process_docs(Dataset.from_list([_record()]))

    assert len(docs) == 3
    assert [doc["question_id"] for doc in docs] == ["demo-video_qa1", "demo-video_qa2", "demo-video_qa10"]
    assert [doc["qa_key"] for doc in docs] == ["qa1", "qa2", "qa10"]
    assert {doc["video_id"] for doc in docs} == {"demo-video"}
    assert {doc["video_summary"] for doc in docs} == {VIDEO_SUMMARY}
    assert docs[0]["reasoning_type"] == "Implicit Inference"
    assert docs[0]["reasoning_process"] == "<Step 1> The detective boards the boat.\n<Step 2> The smuggler runs away."


def test_vrbench_doc_to_visual_resolves_configured_media_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "VRBench" / "videos" / "v001" / "demo-video.mp4"
    video.parent.mkdir(parents=True)
    _write_test_video(video)
    monkeypatch.setenv("VRBENCH_VIDEO_DIR", str(tmp_path))

    assert vrbench_utils.vrbench_doc_to_visual(_doc()) == [str(video)]


def test_vrbench_doc_to_visual_reports_the_missing_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VRBENCH_VIDEO_DIR", str(tmp_path / "empty-root"))

    with pytest.raises(FileNotFoundError, match="VRBENCH_VIDEO_DIR"):
        vrbench_utils.vrbench_doc_to_visual(_doc())


def test_vrbench_prompt_lists_every_option_and_keeps_the_step_format() -> None:
    prompt = vrbench_utils.vrbench_doc_to_text(_doc(), {"pre_prompt": "Video QA:\n", "post_prompt": "\nLetter only.", "include_video_summary": True})

    assert prompt.startswith("Video QA:\n")
    assert "Question: Why did the detective return?" in prompt
    assert "A: First" in prompt
    assert "D: Fourth" in prompt
    assert "<Answer> [Option letter]" in prompt
    assert prompt.endswith("\nLetter only.")


def test_vrbench_video_summary_is_removable_from_the_model_prompt() -> None:
    """``include_video_summary: false`` must strip the narrative summary completely.

    The summary retells the plot, so a model that reads it can answer without
    watching the video.  The official protocol keeps it (see the README), so the
    port only guarantees that the ablation switch removes every trace of it.
    """
    ablated = vrbench_utils.vrbench_doc_to_text(_doc(), {"include_video_summary": False})

    assert VIDEO_SUMMARY not in ablated
    assert "detective returns to the harbour" not in ablated
    assert "Question: Why did the detective return?" in ablated


def test_vrbench_extract_mcq_answer_follows_the_official_cascade() -> None:
    assert vrbench_utils.extract_mcq_answer("Reasoning ... \\boxed{B}") == "B"
    assert vrbench_utils.extract_mcq_answer("<Step 1> x\n<Answer> C") == "C"
    assert vrbench_utils.extract_mcq_answer("<Answer> a") == "A"
    assert vrbench_utils.extract_mcq_answer("A. first option\nD. last option") == "D"
    assert vrbench_utils.extract_mcq_answer("Some talk.\nAnswer: A") == "A"
    assert vrbench_utils.extract_mcq_answer("The best option is B)") == "B"
    assert vrbench_utils.extract_mcq_answer("I pick C and stop") == "C"
    assert vrbench_utils.extract_mcq_answer("") is None
    assert vrbench_utils.extract_mcq_answer("no letters here") is None


def test_vrbench_extract_mcq_answer_keeps_the_last_letter_of_a_long_chain() -> None:
    response = "<Step 1> Option A looks plausible.\n<Step 2> Option B is wrong.\n<Answer> D"

    assert vrbench_utils.extract_mcq_answer(response) == "D"


def test_vrbench_mcq_process_results_and_aggregate_accuracy() -> None:
    correct = vrbench_utils.vrbench_mcq_process_results(_doc(answer="D"), ["<Answer> D"])
    wrong = vrbench_utils.vrbench_mcq_process_results(_doc(answer="B"), ["<Answer> C"])
    correct_record = correct["vrbench_score"]
    wrong_record = wrong["vrbench_score"]

    assert correct_record["question_id"] == "demo-video_qa1"
    assert correct_record["parsed_prediction"] == "D"
    assert correct_record["score"] == 1.0
    assert wrong_record["score"] == 0.0
    assert "vrbench_mcq_implicit_inference_accuracy" in correct
    assert vrbench_utils.vrbench_mcq_aggregate_results([correct_record, wrong_record]) == 50.0
    assert vrbench_utils.vrbench_mcq_aggregate_implicit_inference([correct_record, wrong_record]) == 50.0
    assert vrbench_utils.vrbench_mcq_aggregate_logical_linkage([correct_record, wrong_record]) == 0.0


def test_vrbench_parse_rate_response_reads_the_rate_tag() -> None:
    result = vrbench_utils._parse_rate_response("<rate>7</rate>.\n<reason>Mostly correct.</reason>")

    assert result.metadata["rate"] == 7.0
    assert result.metadata["rate_parsed"] is True
    assert result.score == pytest.approx(0.7)
    assert result.is_correct is True


def test_vrbench_parse_rate_response_clamps_and_flags_unparsable_replies() -> None:
    clamped = vrbench_utils._parse_rate_response("<rate>13</rate>")
    missing = vrbench_utils._parse_rate_response("The reasoning is good but I forgot the tag.")

    assert clamped.metadata["rate"] == 10.0
    assert missing.metadata["rate"] == 0.0
    assert missing.metadata["rate_parsed"] is False
    assert missing.is_correct is False


@pytest.mark.parametrize("reasoning_type", sorted(vrbench_utils.UNIQUE_ANSWER_TYPES))
def test_vrbench_unique_types_use_the_unique_judge_without_the_summary(reasoning_type: str, judge_calls: list[dict[str, Any]]) -> None:
    metrics = vrbench_utils.vrbench_process_process_results(_doc(reasoning_type), ["<Step 1> ...\n<Answer> D"])
    record = metrics["vrbench_score"]

    assert len(judge_calls) == 1
    assert judge_calls[0]["route"] == "unique"
    assert judge_calls[0]["kwargs"] == {"procedure": _doc()["reasoning_process"]}
    assert "video_summary" not in judge_calls[0]["kwargs"]
    assert record["judged"] is True
    assert record["score"] == 7.0


@pytest.mark.parametrize("reasoning_type", sorted(vrbench_utils.NON_UNIQUE_ANSWER_TYPES))
def test_vrbench_non_unique_types_use_the_non_unique_judge_with_the_summary(reasoning_type: str, judge_calls: list[dict[str, Any]]) -> None:
    metrics = vrbench_utils.vrbench_process_process_results(_doc(reasoning_type), ["<Step 1> ...\n<Answer> A"])
    record = metrics["vrbench_score"]

    assert len(judge_calls) == 1
    assert judge_calls[0]["route"] == "non_unique"
    assert judge_calls[0]["kwargs"]["video_summary"] == VIDEO_SUMMARY
    assert judge_calls[0]["kwargs"]["procedure"] == _doc()["reasoning_process"]
    assert record["judged"] is True
    assert record["score"] == 7.0


def test_vrbench_non_unique_summary_is_truncated_for_the_judge(judge_calls: list[dict[str, Any]]) -> None:
    doc = dict(_doc("Event Prediction"))
    doc["video_summary"] = "x" * (vrbench_utils.VIDEO_SUMMARY_CHAR_LIMIT + 50)

    vrbench_utils.vrbench_process_process_results(doc, ["<Answer> A"])

    assert len(judge_calls[0]["kwargs"]["video_summary"]) == vrbench_utils.VIDEO_SUMMARY_CHAR_LIMIT


@pytest.mark.parametrize("reasoning_type", ["Event Summarization", "Counting Porblems", "Brand New Type"])
def test_vrbench_unrouted_types_are_never_judged(reasoning_type: str, judge_calls: list[dict[str, Any]]) -> None:
    metrics = vrbench_utils.vrbench_process_process_results(_doc(reasoning_type), ["<Step 1> ...\n<Answer> B"])

    assert judge_calls == []
    assert list(metrics) == ["vrbench_score"]
    assert metrics["vrbench_score"]["judged"] is False
    assert metrics["vrbench_score"]["score"] == 0.0


def test_vrbench_judge_prompts_place_the_summary_only_on_the_non_unique_route() -> None:
    unique_prompt = vrbench_utils._unique_judge_prompt("Q?", "prediction", "gt", procedure="<Step 1> a", video_summary=VIDEO_SUMMARY)
    non_unique_prompt = vrbench_utils._non_unique_judge_prompt("Q?", "prediction", "gt", procedure="<Step 1> a", video_summary=VIDEO_SUMMARY)

    assert VIDEO_SUMMARY not in unique_prompt
    assert "# Video Summary" not in unique_prompt
    assert VIDEO_SUMMARY in non_unique_prompt
    assert "<rate>the score (0-10)</rate>" in unique_prompt
    assert "<rate>the score (0-10)</rate>" in non_unique_prompt


def test_vrbench_process_aggregate_scales_judged_rates_to_percent(judge_calls: list[dict[str, Any]]) -> None:
    judged = vrbench_utils.vrbench_process_process_results(_doc("Logical Linkage"), ["<Answer> D"])["vrbench_score"]
    unjudged = vrbench_utils.vrbench_process_process_results(_doc("Event Summarization"), ["<Answer> D"])["vrbench_score"]

    assert vrbench_utils.vrbench_process_aggregate_results([judged, unjudged]) == 70.0
    assert vrbench_utils.vrbench_process_aggregate_logical_linkage([judged, unjudged]) == 70.0
    assert vrbench_utils.vrbench_process_aggregate_event_attribution([judged, unjudged]) == 0.0


def test_vrbench_format_reasoning_process_renders_numbered_steps() -> None:
    rendered = vrbench_utils.format_reasoning_process({"2": "Second step", "1": "First step", "10": "Tenth step"})

    assert rendered == "<Step 1> First step\n<Step 2> Second step\n<Step 10> Tenth step"
    assert vrbench_utils.format_reasoning_process("already a string") == "already a string"
    assert vrbench_utils.format_reasoning_process(None) == ""
