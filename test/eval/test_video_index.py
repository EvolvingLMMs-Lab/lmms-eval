import random

import cv2
import numpy as np
import pytest
from datasets import Dataset

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.video_index import utils as video_index_utils

OPTIONS = ["a red cup", "a knife", "the phone"]


def _doc(item_id="Demo_1", group="perception", **extra):
    doc = {
        "item_id": item_id,
        "benchmark": "Demo",
        "capability_group": group,
        "question": "What does the person pick up?",
        "options": OPTIONS,
        "answer": "C",
        "answer_idx": 2,
        "video": "videos/demo.mp4",
        "video_id": "demo",
        "duration_s": 20.0,
    }
    doc.update(extra)
    return doc


def _write_test_video(path, frame_count=40, fps=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert writer.isOpened(), "OpenCV cannot create the temporary MP4 test fixture"
    for value in range(frame_count):
        writer.write(np.full((48, 64, 3), (value * 5) % 255, dtype=np.uint8))
    writer.release()


def test_video_index_tasks_are_registered():
    task_manager = TaskManager("ERROR")
    expected = {"video_index", "video_index_1fps", "video_index_64frame", "video_index_32frame", "video_index_8frame", "video_index_blind"}
    assert not expected.difference(task_manager.all_tasks)


def test_video_index_reply_parsing():
    extract = video_index_utils.extract_letter
    assert extract("B", OPTIONS) == "B"
    assert extract("(c) the phone", OPTIONS) == "C"
    assert extract("The answer is A. Wait, the answer is C", OPTIONS) == "C"
    assert extract("A person picks up the phone", OPTIONS) == "C"
    assert extract("I cannot tell", OPTIONS) is None
    assert extract("", OPTIONS) is None


def test_video_index_prompt_is_the_paper_prompt():
    text = video_index_utils.video_index_doc_to_text(_doc(), {"intro": video_index_utils.INTRO_BLIND})
    assert text == (
        "You are given NO frames from the video. Answer the question from the text alone.\n\n"
        "Question: What does the person pick up?\nA. a red cup\nB. a knife\nC. the phone\n"
        "Reply with ONLY the option letter (or the exact short answer if no options)."
    )


def test_video_index_blind_documents_follow_the_fixed_permutations():
    docs = video_index_utils.video_index_blind_process_docs(Dataset.from_list([_doc("Demo_1"), _doc("Demo_2")]))
    assert len(docs) == 8
    rng = random.Random("42|Demo_1")
    assert [doc["perm"] for doc in docs][:4] == [rng.sample(range(3), 3) for _ in range(4)]
    for doc in docs:
        shown = [OPTIONS[index] for index in doc["perm"]]
        target = video_index_utils.video_index_doc_to_target(doc)
        assert shown["ABC".index(target)] == "the phone"
        assert f"{target}. the phone" in video_index_utils.video_index_doc_to_text(doc, {"intro": video_index_utils.INTRO_BLIND})


def test_video_index_frame_rule():
    indices = video_index_utils.one_fps_indices
    assert indices(40, 2.0, 20.0) == list(range(0, 40, 2))
    assert indices(10, 0.5, 20.0) == list(range(10))
    capped = indices(2000, 2.0, 1000.0, cap=512)
    assert len(capped) == 512 and capped[0] == 0 and capped[-1] == 1998


def test_video_index_frames_come_from_the_local_copy(monkeypatch, tmp_path):
    pytest.importorskip("av")
    _write_test_video(tmp_path / "videos" / "demo.mp4")
    monkeypatch.setenv("VIDEO_INDEX_DIR", str(tmp_path))
    kwargs = {"intro": video_index_utils.INTRO_FRAMES, "fps": 1.0, "max_frames": 8, "short_side": 32}

    assert video_index_utils.video_index_doc_to_visual(_doc()) == [str(tmp_path / "videos" / "demo.mp4")]
    frames = video_index_utils.video_index_doc_to_visual_frames(_doc(), kwargs)
    assert len(frames) == 8 and min(frames[0].size) == 32
    assert video_index_utils.video_index_doc_to_text_frames(_doc(), kwargs).startswith("You are given 8 frame(s) sampled from a video.")


def test_video_index_blind_replies_are_averaged_per_item():
    records = []
    for index, group in enumerate(["perception", "temporal"]):
        docs = video_index_utils.video_index_blind_process_docs(Dataset.from_list([_doc(f"Demo_{index}", group)]))
        for doc in docs:
            correct = index == 0 or doc["perm_idx"] == 0
            reply = video_index_utils.video_index_doc_to_target(doc) if correct else "I cannot tell"
            result = video_index_utils.video_index_process_results(doc, [reply])
            assert set(result) == set(video_index_utils.METRICS)
            records.append(result["video_index_acc"])

    assert video_index_utils.video_index_aggregate_overall(records) == pytest.approx(62.5)
    assert video_index_utils.video_index_aggregate_perception(records) == pytest.approx(100.0)
    assert video_index_utils.video_index_aggregate_temporal(records) == pytest.approx(25.0)
    assert np.isnan(video_index_utils.video_index_aggregate_spatial(records))
