"""Offline contracts: task discovery, schemas, scoring, and safe video resolution."""

import hashlib
import json
import math
from pathlib import Path

import datasets
import pytest

from lmms_eval import utils as harness_utils
from lmms_eval.api.task import ConfigurableTask, TaskConfig
from lmms_eval.models import MODEL_REGISTRY_V2, get_model
from lmms_eval.models.model_utils.grt.profile import command, load_profiles
from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.dive_bench import utils

TASK_DIR = Path(utils.__file__).parent
TASKS = {
    "dive_bench_educational_high_fps": 634,
    "dive_bench_high_motion_high_fps": 3243,
    "dive_bench_high_motion_high_fps_preview1000": 1000,
    "densevideo": 634,
    "densevideo_highmotion": 1000,
}


def fixture_dataset(highmotion):
    count = 3243 if highmotion else 634
    return datasets.Dataset.from_dict(
        {
            "video": ["0"] * count,
            "qid": ["0"] * count,
            "video_path": [f"egodex/action/{index}.mp4" if highmotion else f"DenseVideo-LPM/videos/{index}.mp4" for index in range(count)],
            "question": ["Track the hand." if highmotion else "What subtitles appear in the entire video?"] * count,
            "answer": ['["top", "middle", "bottom"]' if highmotion else "Hello world!"] * count,
            "frame_count": [3] * count,
        }
    )


def fixture_fingerprint(dataset):
    rows = [[str(row["video_path"]), str(row["qid"]), str(row["question"]), str(row["answer"]), int(row["frame_count"])] for row in dataset]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def test_discovery_and_full_config_schema():
    manager = TaskManager(include_defaults=False, include_path=str(TASK_DIR))
    assert set(manager.all_tasks) == set(TASKS)
    for name, info in manager.task_index.items():
        config = harness_utils.load_yaml_config(info["yaml_path"])
        TaskConfig(**config)
        assert config["task"] == name
        metrics = {entry["metric"] for entry in config["metric_list"]}
        assert metrics == ({"cer", "wer", "token_f1", "exact_match"} if TASKS[name] == 634 else {"grid_acc", "grid_ade", "grid_fde", "grid_transition_acc", "token_f1"})
        assert not any("gpt" in metric or "mos" in metric for metric in metrics)


@pytest.mark.parametrize("name,expected", TASKS.items())
def test_native_task_initialization_without_hub(monkeypatch, name, expected):
    """Exercise actual upstream ConfigurableTask with synthetic local documents."""
    highmotion = expected != 634
    source = fixture_dataset(highmotion)
    if highmotion:
        monkeypatch.setattr(utils, "HIGHMOTION_ORDERED_CONTENT_SHA256", fixture_fingerprint(source))
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return datasets.DatasetDict({"test": source})

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    manager = TaskManager(include_defaults=False, include_path=str(TASK_DIR))
    config = harness_utils.load_yaml_config(manager.task_index[name]["yaml_path"])
    if highmotion:
        # YAML !function loads its own module object; patch that test fixture's
        # globals rather than weakening or bypassing the production validator.
        monkeypatch.setitem(config["process_docs"].__globals__, "HIGHMOTION_ORDERED_CONTENT_SHA256", fixture_fingerprint(source))
    task = ConfigurableTask(config=config, model_name="grt_llava_hf")
    assert len(task.eval_docs) == expected
    assert calls
    selected_files = calls[0][1]["data_files"]
    assert selected_files == {"test": "Egodex_traj.parquet" if highmotion else "LPM_videos.parquet"}
    doc = task.eval_docs[0]
    target = task.doc_to_target(doc)
    metrics = task.process_results(doc, [target])
    assert metrics["token_f1"] == 1.0
    assert metrics["grid_acc" if highmotion else "exact_match"] == 1.0
    assert task.doc_to_text(doc)


def test_exact_objective_metrics_ignore_fast_metric_environment(monkeypatch):
    monkeypatch.setenv("DENSEVIDEO_FAST_TEXT_METRICS", "1")
    assert utils.educational_process_results({"answer": "  Hello WORLD!\n"}, ["hello world!"]) == {"cer": 0.0, "wer": 0.0, "token_f1": 1.0, "exact_match": 1.0}
    assert utils._compute_token_f1("a a b", "a b b") == pytest.approx(2 / 3)
    assert utils._compute_token_f1("", "") == 1.0
    assert utils._compute_token_f1("x", "") == 0.0
    assert utils._compute_cer("abcd", "a") == 3.0
    assert utils._compute_wer("a b c d", "a") == 3.0
    assert utils._compute_exact_match("word!", "word") == 0.0


def test_missing_and_extra_trajectory_predictions():
    doc = {"answer": '["top", "middle", "bottom"]', "question": "Track the hand."}
    missing = utils.highmotion_process_results(doc, [])
    assert missing == {"grid_acc": 0.0, "grid_ade": math.sqrt(2), "grid_fde": math.sqrt(2), "grid_transition_acc": 0.0, "token_f1": 0.0}
    extra = utils.highmotion_process_results(doc, ["top,middle,bottom,left"])
    assert extra["grid_acc"] == 1.0
    assert extra["grid_fde"] == 0.0
    assert extra["token_f1"] < 1.0


def test_prefix_is_fixed_and_repeated_qids_do_not_collapse_rows(monkeypatch):
    monkeypatch.setenv("DENSEVIDEO_HIGHMOTION_MAX_EXAMPLES", "0")
    full = fixture_dataset(True)
    monkeypatch.setattr(utils, "HIGHMOTION_ORDERED_CONTENT_SHA256", fixture_fingerprint(full))
    assert len(utils.highmotion_preview_1000(full)) == 1000
    assert len(utils.validate_highmotion(full)) == 3243
    with pytest.raises(ValueError, match="3243"):
        utils.highmotion_preview_1000(full.select(range(5)))
    with pytest.raises(ValueError, match="634"):
        utils.validate_educational(fixture_dataset(False).select(range(317)))
    with pytest.raises(ValueError, match="content/order"):
        utils.validate_highmotion(full.select(list(reversed(range(3243)))))


def test_video_paths_preserve_highmotion_action_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("DIVE_BENCH_DATA_ROOT", str(tmp_path))
    wrong = tmp_path / "videos" / "0.mp4"
    wrong.parent.mkdir()
    wrong.touch()
    doc = {"video_path": "egodex/action_b/0.mp4"}
    with pytest.raises(FileNotFoundError, match="preserve"):
        utils.doc_to_visual(doc)
    right = tmp_path / doc["video_path"]
    right.parent.mkdir(parents=True)
    right.touch()
    assert utils.doc_to_visual(doc) == [str(right)]
    assert utils.doc_to_visual({"video_path": "DenseVideo-LPM/videos/0.mp4"}) == [str(wrong)]
    with pytest.raises(ValueError, match="traversal"):
        utils.doc_to_visual({"video_path": "../0.mp4"})


def test_model_registration_is_additive():
    expected = {"grt_llava_hf": "GRTLlavaHf", "grt_qwen2_5_vl": "GRTQwen2_5VL", "grt_qwen2_5_vl_floor": "GRTQwen2_5VLFloor"}
    for name, class_name in expected.items():
        assert get_model(name).__name__ == class_name
        assert MODEL_REGISTRY_V2.resolve(name).class_path.startswith("lmms_eval.models.simple.grt_")
    for stock in ("llava_hf", "qwen2_5_vl"):
        assert ".grt_" not in MODEL_REGISTRY_V2.resolve(stock).class_path


def test_profiles_preserve_matched_controls_and_caps(tmp_path):
    profiles = load_profiles()["profiles"]
    for name, cap in (("route31", 128), ("qwen3", 128), ("qwen7", 48)):
        profile = profiles[name]
        assert profile["max_new_tokens"] == cap
        assert set(profile["roles"]) == {"base", "all", "candidate"}
        for role in profile["roles"]:
            cmd = command(name, role, tmp_path / "not-created")
            assert cmd[cmd.index("--seed") + 1] == "0,1234,1234,1234"
            assert "revision=" in cmd[cmd.index("--model_args") + 1]
            assert "--log_samples" in cmd
    assert not (tmp_path / "not-created").exists()


def test_unvalidated_transformers_rejected_before_weights(monkeypatch):
    from lmms_eval.models.model_utils.grt import runtime

    monkeypatch.setattr(runtime, "version", lambda _: "5.0.0")
    for name in ("grt_llava_hf", "grt_qwen2_5_vl", "grt_qwen2_5_vl_floor"):
        with pytest.raises(RuntimeError, match="4.57.6"):
            get_model(name)()
