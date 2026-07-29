import json
import zipfile

import cv2
import numpy as np

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.cgav_counting import utils as cgav_utils


def _write_test_video(path, frame_count=3, fps=2):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (8, 8))
    assert writer.isOpened(), "OpenCV cannot create the temporary MP4 test fixture"
    for value in range(frame_count):
        writer.write(np.full((8, 8, 3), value * 50, dtype=np.uint8))
    writer.release()


def test_cgav_counting_tasks_are_registered():
    task_manager = TaskManager("ERROR")
    expected = {"cgav_counting", "cgav_counting_long", "cgav_counting_ref", "cgav_counting_clue"}
    assert not expected.difference(task_manager.all_tasks)


def test_cgav_counting_metrics():
    doc = {"index": 0, "answer": "18", "category": "object", "type": "A2V"}
    result = cgav_utils.cgav_process_results_long(doc, ["The answer is 18."])

    assert result["acc"]["acc"] == 1.0
    assert result["oboa"]["oboa"] == 1.0
    assert result["mae"]["mae"] == 0.0
    assert result["rmse"]["squared_error"] == 0.0


def test_cgav_counting_caps_outlier_error_like_official_evaluator():
    doc = {"index": 0, "answer": "18", "category": "object", "type": "A2V"}
    result = cgav_utils.cgav_process_results_long(doc, ["999"])

    assert result["mae"]["mae"] == 36.0
    assert result["rmse"]["squared_error"] == 1296.0


def test_cgav_counting_aggregations_match_official_definitions():
    first = cgav_utils.cgav_process_results_long({"index": 0, "answer": "10", "category": "object", "type": "V"}, ["10"])
    second = cgav_utils.cgav_process_results_long({"index": 1, "answer": "10", "category": "object", "type": "V"}, ["12"])

    assert cgav_utils.aggregate_acc([first["acc"], second["acc"]]) == 0.5
    assert cgav_utils.aggregate_oboa([first["oboa"], second["oboa"]]) == 0.5
    assert cgav_utils.aggregate_mae([first["mae"], second["mae"]]) == 1.0
    assert cgav_utils.aggregate_rmse([first["rmse"], second["rmse"]]) == 2**0.5


def test_cgav_reference_filename_uses_official_rounding():
    doc = {"video": "abc.mp4", "query_interval": [1, 2.345]}
    assert cgav_utils._reference_filename(doc) == "abc_1.00_2.35.mp4"


def test_cgav_clue_frame_extraction_decodes_real_video(tmp_path):
    video = tmp_path / "video.mp4"
    _write_test_video(video)

    frames = cgav_utils._frames_at_timestamps(str(video), [0.0, 0.9])

    assert len(frames) == 2
    assert all(frame.size == (8, 8) for frame in frames)
    assert np.asarray(frames[1]).mean() > np.asarray(frames[0]).mean()


def test_cgav_extracts_official_style_split_archives(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    _write_test_video(source)
    archive = tmp_path / "videos.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(source, "demo-video.mp4")
    payload = archive.read_bytes()
    midpoint = len(payload) // 2
    (tmp_path / "videos.zip.part000").write_bytes(payload[:midpoint])
    (tmp_path / "videos.zip.part001").write_bytes(payload[midpoint:])
    archive.unlink()
    monkeypatch.setenv("CGAV_COUNTING_ROOT", str(tmp_path))

    cgav_utils._ensure_media_extracted(("cg_videos_720p", "videos"))

    assert (tmp_path / "cg_videos_720p" / "demo-video.mp4").is_file()
    assert not (tmp_path / "videos.zip").exists()


def test_cgav_event_clue_perfect_wcs():
    doc = {
        "index": 1,
        "category": "event",
        "type": "A",
        "clue": json.dumps([{"start": 1, "end": 3}]),
    }
    result = cgav_utils.cgav_process_results_clue(doc, ['<answer>[["1.00", "3.00"]]</answer>'])

    assert result["wcs"]["wcs"] == 1.0
    assert result["ifa"]["ifa"] == 1.0


def test_cgav_object_and_attribute_clue_perfect_wcs():
    object_doc = {
        "index": 2,
        "category": "object",
        "type": "V",
        "clue": json.dumps([{"timestamp": 2, "bbox": [0, 0, 10, 10]}]),
    }
    object_result = cgav_utils.cgav_process_results_clue(object_doc, ['<answer>{"Frame1": [[0,0,10,10]]}</answer>'])
    assert object_result["wcs"]["wcs"] == 1.0

    attribute_doc = {
        "index": 3,
        "category": "attribute",
        "type": "V",
        "clue": json.dumps([[{"timestamp": 2, "bbox": [0, 0, 10, 10]}]]),
    }
    prediction = '<answer>{"Frame 1": [{"bbox": [0,0,10,10], "label": "x"}]}</answer>'
    attribute_result = cgav_utils.cgav_process_results_clue(attribute_doc, [prediction])
    assert attribute_result["wcs"]["wcs"] == 1.0


def test_cgav_malformed_clue_response_has_zero_wcs_and_ifa():
    doc = {
        "index": 4,
        "category": "event",
        "type": "A",
        "clue": json.dumps([{"start": 1, "end": 3}]),
    }
    result = cgav_utils.cgav_process_results_clue(doc, ["not valid JSON"])

    assert result["wcs"]["wcs"] == 0.0
    assert result["ifa"]["ifa"] == 0.0
