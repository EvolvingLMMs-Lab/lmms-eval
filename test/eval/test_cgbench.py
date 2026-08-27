import cv2
import numpy as np

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.cgbench import utils as cgbench_utils


def _doc(answer="A"):
    return {
        "qid": 1,
        "video_uid": "demo-video",
        "question": "Which option is correct?",
        "choices": [f"Option {index}" for index in range(14)],
        "right_answer": answer,
        "domain": "Knowledge",
        "sub_category": "Entity Perception",
        "duration": 600,
    }


def _write_test_video(path, frame_count=4, fps=2):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (8, 8))
    assert writer.isOpened(), "OpenCV cannot create the temporary MP4 test fixture"
    for value in range(frame_count):
        writer.write(np.full((8, 8, 3), value * 50, dtype=np.uint8))
    writer.release()


def test_cgbench_tasks_are_registered():
    task_manager = TaskManager("ERROR")
    expected = {"cgbench", "cgbench_subtitles", "cgbench_all"}
    assert not expected.difference(task_manager.all_tasks)


def test_cgbench_supports_all_fourteen_choice_letters():
    assert cgbench_utils.extract_characters_regex("The correct answer is (N).") == "N"


def test_cgbench_doc_to_visual_resolves_configured_media_root(monkeypatch, tmp_path):
    video = tmp_path / "cg_videos_720p" / "demo-video.mp4"
    video.parent.mkdir()
    video.touch()
    monkeypatch.setenv("CGBENCH_VIDEO_DIR", str(tmp_path))

    assert cgbench_utils.cgbench_doc_to_visual(_doc()) == [str(video)]


def test_cgbench_prompts_include_all_options_and_subtitle_fallback():
    doc = _doc()
    prompt = cgbench_utils.cgbench_doc_to_text(doc, {"pre_prompt": "Video QA:\n", "post_prompt": "\nLetter only."})
    subtitle_prompt = cgbench_utils.cgbench_doc_to_text_subtitle(doc, {"post_prompt": "\nLetter only."})

    assert prompt.startswith("Video QA:\nSelect the best answer")
    assert "A. Option 0" in prompt
    assert "N. Option 13" in prompt
    assert prompt.endswith("\nLetter only.")
    assert "No subtitles available." in subtitle_prompt
    assert "N. Option 13" in subtitle_prompt


def test_cgbench_subtitles_are_aligned_to_decoded_video_frames(monkeypatch, tmp_path):
    video_dir = tmp_path / "cg_videos_720p"
    subtitle_dir = tmp_path / "cg_subtitles"
    video_dir.mkdir()
    subtitle_dir.mkdir()
    _write_test_video(video_dir / "demo-video.mp4")
    (subtitle_dir / "demo-video.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nFirst caption\n\n2\n00:00:01,000 --> 00:00:02,000\nSecond caption\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CGBENCH_ROOT", str(tmp_path))

    prompt = cgbench_utils.cgbench_doc_to_text_subtitle(_doc(), {"frame_num": 1})

    assert "First caption" in prompt
    assert "Second caption" not in prompt


def test_cgbench_process_results_and_aggregate_accuracy():
    correct = cgbench_utils.cgbench_process_results(_doc("A"), ["The answer is A."])["cgbench_accuracy"]
    wrong = cgbench_utils.cgbench_process_results(_doc("B"), ["A"])["cgbench_accuracy"]

    assert correct["question_id"] == 1
    assert correct["pred_answer"] == "A"
    assert correct["score"] == 1.0
    assert wrong["score"] == 0.0
    assert cgbench_utils.cgbench_aggregate_results([correct, wrong]) == 50.0
