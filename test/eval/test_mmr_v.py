import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.mmr_v import utils as mmr_v_utils


def _letters(option_count: int) -> list[str]:
    return [chr(ord("A") + index) for index in range(option_count)]


def _doc(
    option_count: int = 8,
    answer: str = "(C)",
    ability_type: str = "Metaphor Understanding",
    video_type: str = "Animation",
    question_idx: int = 0,
    video: str = "demo video.mp4",
) -> dict[str, Any]:
    return {
        "video": video,
        "videoType": video_type,
        "question": "What does the object being chased by the people refer to?",
        "options": [f"({letter}) Option {letter}" for letter in _letters(option_count)],
        "correctAnswer": answer,
        "abilityType_L2": ability_type,
        "abilityType_L3": "Ontological Metaphor",
        "question_idx": question_idx,
    }


def _write_test_video(path: Path, frame_count: int = 4, fps: int = 2) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (8, 8))
    assert writer.isOpened(), "OpenCV cannot create the temporary MP4 test fixture"
    for value in range(frame_count):
        writer.write(np.full((8, 8, 3), value * 50, dtype=np.uint8))
    writer.release()


def _isolate_media_env(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("MMR_V_VIDEO_DIR", str(root))
    monkeypatch.delenv("MMR_V_ROOT", raising=False)
    monkeypatch.delenv("LMMS_EVAL_MEDIA_ROOT", raising=False)


def _declared_metrics() -> set[str]:
    template = Path(mmr_v_utils.__file__).parent / "_default_template_yaml"
    return set(re.findall(r"^\s*- metric:\s*(\S+)\s*$", template.read_text(encoding="utf-8"), flags=re.MULTILINE))


def test_mmr_v_tasks_are_registered() -> None:
    task_manager = TaskManager("ERROR")
    expected = {"mmr_v", "mmr_v_cot", "mmr_v_all"}
    assert not expected.difference(task_manager.all_tasks)


@pytest.mark.parametrize("option_count,letter", [(7, "G"), (11, "K"), (13, "M")])
def test_mmr_v_extracts_every_offered_letter(option_count: int, letter: str) -> None:
    choices = _letters(option_count)

    assert mmr_v_utils.extract_mmr_v_answer(f"[[{letter}]]", choices) == letter
    assert mmr_v_utils.extract_mmr_v_answer(f"Option A is wrong. Option B is a distractor. [[{letter}]]", choices) == letter
    assert mmr_v_utils.extract_mmr_v_answer(f"<think>Option (A) looks close.</think>\n[[{letter}]]", choices) == letter
    assert mmr_v_utils.extract_mmr_v_answer(f"\\boxed{{{letter}}}", choices) == letter


def test_mmr_v_rejects_a_letter_the_question_does_not_offer() -> None:
    assert mmr_v_utils.extract_mmr_v_answer("[[M]]", _letters(7)) == ""


def test_mmr_v_reads_the_letters_back_from_options_that_skip_one() -> None:
    doc = _doc()
    doc["options"] = ["(A) First", "(B) Second", "", "(D) Fourth"]

    offered = mmr_v_utils.mmr_v_process_results(dict(doc, correctAnswer="(D)"), ["The answer is [[D]]."])["mmr_v_overall_accuracy"]
    skipped = mmr_v_utils.mmr_v_process_results(dict(doc, correctAnswer="(D)"), ["[[C]]"])["mmr_v_overall_accuracy"]

    assert offered["pred_answer"] == "D"
    assert offered["score"] == 1.0
    assert skipped["pred_answer"] == ""
    assert skipped["score"] == 0.0


def test_mmr_v_doc_to_text_keeps_the_dataset_option_letters() -> None:
    prompt = mmr_v_utils.mmr_v_doc_to_text(_doc(13), {"pre_prompt": "Video QA:\n", "post_prompt": "\nAnswer with the option letter only."})

    assert prompt.startswith("Video QA:\nPlease select the best answer")
    assert "Options:\n(A) Option A\n(B) Option B\n" in prompt
    assert "(M) Option M" in prompt
    assert "A. (A)" not in prompt
    assert prompt.count("(A)") == 1
    assert prompt.endswith("\nAnswer with the option letter only.")


def test_mmr_v_doc_to_text_defaults_to_an_empty_prompt_wrapper() -> None:
    prompt = mmr_v_utils.mmr_v_doc_to_text(_doc())

    assert prompt.startswith("Please select the best answer")
    assert prompt.endswith("(H) Option H")


def test_mmr_v_doc_to_visual_resolves_configured_media_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "videos" / "demo video.mp4"
    video.parent.mkdir()
    _write_test_video(video)
    _isolate_media_env(monkeypatch, tmp_path)

    assert mmr_v_utils.mmr_v_doc_to_visual(_doc()) == [str(video)]


def test_mmr_v_doc_to_visual_reports_a_missing_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate_media_env(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError, match="MMR-V video not found"):
        mmr_v_utils.mmr_v_doc_to_visual(_doc(video="absent video.mp4"))


def test_mmr_v_parenthesised_gold_matches_a_bare_prediction() -> None:
    bare = mmr_v_utils.mmr_v_process_results(_doc(answer="(C)"), ["C"])["mmr_v_overall_accuracy"]
    verbose = mmr_v_utils.mmr_v_process_results(_doc(answer="(C)"), ["The best answer is (C)."])["mmr_v_overall_accuracy"]
    wrong = mmr_v_utils.mmr_v_process_results(_doc(answer="(C)"), ["[[D]]"])["mmr_v_overall_accuracy"]

    assert bare["answer"] == "C"
    assert bare["pred_answer"] == "C"
    assert bare["score"] == 1.0
    assert verbose["score"] == 1.0
    assert wrong["pred_answer"] == "D"
    assert wrong["score"] == 0.0


def test_mmr_v_process_results_emits_the_declared_metric_keys() -> None:
    declared = _declared_metrics()

    for ability_type in mmr_v_utils.ABILITY_TYPES:
        for video_type in mmr_v_utils.VIDEO_TYPES:
            doc = _doc(ability_type=ability_type, video_type=video_type)
            keys = set(mmr_v_utils.mmr_v_process_results(doc, ["[[C]]"]))
            assert len(keys) == 3
            assert "mmr_v_overall_accuracy" in keys
            assert keys.issubset(declared)


def test_mmr_v_process_results_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="abilityType_L2"):
        mmr_v_utils.mmr_v_process_results(_doc(ability_type="Mind Reading"), ["C"])
    with pytest.raises(ValueError, match="videoType"):
        mmr_v_utils.mmr_v_process_results(_doc(video_type="Documentary"), ["C"])


def test_mmr_v_aggregates_per_ability_type_and_per_video_type() -> None:
    records = [
        mmr_v_utils.mmr_v_process_results(_doc(answer="(C)", ability_type="Theme Understanding", video_type="Animation", question_idx=1), ["[[C]]"])["mmr_v_overall_accuracy"],
        mmr_v_utils.mmr_v_process_results(_doc(answer="(C)", ability_type="Theme Understanding", video_type="movie", question_idx=2), ["[[D]]"])["mmr_v_overall_accuracy"],
        mmr_v_utils.mmr_v_process_results(_doc(answer="(B)", ability_type="Causal Reasoning", video_type="Animation", question_idx=3), ["[[B]]"])["mmr_v_overall_accuracy"],
    ]

    assert mmr_v_utils.mmr_v_aggregate_overall(records) == pytest.approx(200.0 / 3.0)
    assert mmr_v_utils.mmr_v_aggregate_theme_understanding(records) == 50.0
    assert mmr_v_utils.mmr_v_aggregate_causal_reasoning(records) == 100.0
    assert mmr_v_utils.mmr_v_aggregate_metaphor_understanding(records) == 0.0
    assert mmr_v_utils.mmr_v_aggregate_animation_video(records) == 100.0
    assert mmr_v_utils.mmr_v_aggregate_movie_video(records) == 0.0
    assert mmr_v_utils.mmr_v_aggregate_philosophy_video(records) == 0.0
