import inspect
from types import SimpleNamespace

import pytest

from lmms_eval.models.simple.diffusers_wm_base import DiffusersWMBase
from lmms_eval.models.simple.wan2_2_t2v import Wan2_2_T2V
from lmms_eval.tasks.vbench.build_dataset import build_configs, deterministic_seed, merge_prompts
from lmms_eval.tasks.vbench.compare_wan22_results import OFFICIAL, aggregate
from lmms_eval.tasks.vbench.reproduce_wan22_temporal_flickering import expected_video_paths


def test_vbench_builder_expands_official_sample_counts():
    records = {
        "vbench": [
            {"prompt_en": "still", "dimension": ["temporal_flickering"]},
            {"prompt_en": "moving", "dimension": ["subject_consistency", "motion_smoothness"]},
        ],
        "vbench2": [
            {"prompt_en": "varied", "dimension": ["Diversity"]},
            {"prompt_en": "camera", "dimension": ["Camera_Motion"]},
        ],
    }

    configs = build_configs(records)

    assert len(configs["vbench"]) == 30
    assert len(configs["vbench_temporal_flickering"]) == 25
    assert len(configs["vbench_subject_consistency"]) == 5
    assert len(configs["vbench_motion_smoothness"]) == 5
    assert len(configs["vbench2"]) == 23
    assert len(configs["vbench2_diversity"]) == 20
    assert len(configs["vbench2_camera_motion"]) == 3


def test_vbench_builder_merges_duplicate_prompts_and_keeps_metadata():
    records = [
        {"prompt_en": "same", "dimension": ["subject_consistency"]},
        {"prompt_en": "same", "dimension": ["overall_consistency"], "auxiliary_info": {"key": "value"}},
    ]

    merged = merge_prompts(records, "vbench")

    assert len(merged) == 1
    assert merged[0]["dimensions"] == ["subject_consistency", "overall_consistency"]
    assert merged[0]["source_record_ids"] == [0, 1]
    assert merged[0]["auxiliary"][0]["value"] == {"key": "value"}


def test_vbench_seed_is_stable_and_sample_specific():
    assert deterministic_seed("vbench", "prompt", 0) == deterministic_seed("vbench", "prompt", 0)
    assert deterministic_seed("vbench", "prompt", 0) != deterministic_seed("vbench", "prompt", 1)


def test_vbench_request_metadata_controls_seed_and_v1_filename(tmp_path):
    model = object.__new__(DiffusersWMBase)
    model.output_dir = str(tmp_path)
    model.task_dict = {
        "vbench_subject_consistency": {
            "test": [
                {
                    "suite": "vbench",
                    "sample_index": 3,
                    "seed": 1234,
                    "official_dimensions": ["subject_consistency"],
                }
            ]
        }
    }
    request = SimpleNamespace(args=("A prompt", {"temperature": 0}, None, 0, "vbench_subject_consistency", "test"))

    extras = model._request_extras(request, request.args[1])
    output = model._output_path("A prompt", 0, "vbench_subject_consistency", "signature", extras)

    assert extras["_lmms_eval_seed"] == 1234
    assert output == tmp_path / "vbench" / "A prompt-3.mp4"


def test_request_metadata_is_optional_only_for_non_vbench_tasks():
    model = object.__new__(DiffusersWMBase)
    ordinary = SimpleNamespace(args=("prompt", {"temperature": 0}, None, 0, "ordinary_task", "test"))
    vbench = SimpleNamespace(args=("prompt", {}, None, 0, "vbench", "test"))

    assert model._request_extras(ordinary, ordinary.args[1]) == {"temperature": 0}
    with pytest.raises(RuntimeError, match="required VBench row metadata"):
        model._request_extras(vbench, vbench.args[1])


def test_vbench2_filename_uses_dimension_and_truncated_prompt(tmp_path):
    model = object.__new__(DiffusersWMBase)
    model.output_dir = str(tmp_path)
    prompt = "x" * 200
    extras = {
        "_lmms_eval_vbench": {
            "suite": "vbench2",
            "sample_index": 2,
            "official_dimensions": ["Camera_Motion"],
        }
    }

    output = model._output_path(prompt, 0, "vbench2_camera_motion", "signature", extras)

    assert output == tmp_path / "vbench2" / "Camera_Motion" / f"{'x' * 180}-2.mp4"


def test_combined_vbench2_task_keeps_single_dimension_prompts_in_root(tmp_path):
    model = object.__new__(DiffusersWMBase)
    model.output_dir = str(tmp_path)
    extras = {
        "_lmms_eval_vbench": {
            "suite": "vbench2",
            "sample_index": 1,
            "official_dimensions": ["Camera_Motion"],
        }
    }

    output = model._output_path("A prompt", 0, "vbench2", "signature", extras)

    assert output == tmp_path / "vbench2" / "A prompt-1.mp4"


def test_generation_uses_row_seed_and_atomically_publishes_video(tmp_path):
    model = object.__new__(DiffusersWMBase)
    model.seed = 42
    model.plan = SimpleNamespace(device_str=lambda: "cpu")
    model._ensure_loaded = lambda: None
    model._generation_signature = lambda prompt, visuals, extras: "signature"
    output_path = tmp_path / "video.mp4"
    model._output_path = lambda prompt, doc_id, task, signature, extras: output_path
    observed = {}

    def invoke(prompt, visuals, generator, **extras):
        observed["seed"] = generator.initial_seed()
        return object()

    model._invoke_pipeline = invoke
    model._export = lambda output, path: path.write_bytes(b"video")

    result = model._generate_one("prompt", [], 0, "vbench", {"_lmms_eval_seed": 1234})

    assert result == str(output_path)
    assert observed["seed"] == 1234
    assert output_path.read_bytes() == b"video"
    assert not list(tmp_path.glob("*.partial.mp4"))


def test_pipeline_load_kwargs_pin_optional_model_revision():
    model = object.__new__(DiffusersWMBase)
    model.revision = "fixed-revision"

    assert model._pipeline_load_kwargs("dtype") == {"torch_dtype": "dtype", "revision": "fixed-revision"}


def test_wan22_defaults_match_the_native_720p_recipe():
    defaults = {name: parameter.default for name, parameter in inspect.signature(Wan2_2_T2V.__init__).parameters.items()}

    assert defaults["height"] == 720
    assert defaults["width"] == 1280
    assert defaults["revision"] == "5be7df9619b54f4e2667b2755bc6a756675b5cd7"
    assert defaults["num_frames"] == 81
    assert defaults["num_inference_steps"] == 40
    assert defaults["guidance_scale"] == 4.0
    assert defaults["guidance_scale_2"] == 3.0
    assert defaults["flow_shift"] == 12.0
    assert defaults["fps"] == 16
    assert defaults["negative_prompt"]


def test_official_wan22_dimension_scores_reproduce_published_aggregates():
    for baseline in OFFICIAL.values():
        raw_scores = {dimension: score / 100 for dimension, score in baseline["dimensions"].items()}
        measured = aggregate(raw_scores)

        assert measured["total"] == pytest.approx(baseline["total"], abs=0.01)
        assert measured["quality"] == pytest.approx(baseline["quality"], abs=0.01)
        assert measured["semantic"] == pytest.approx(baseline["semantic"], abs=0.01)


def test_temporal_reproduction_requires_complete_official_subset(tmp_path):
    records = [{"prompt_en": f"prompt {index}", "dimension": ["temporal_flickering"]} for index in range(75)]

    paths = expected_video_paths(tmp_path, records)

    assert len(paths) == 375
    assert paths[0] == tmp_path / "prompt 0-0.mp4"
    assert paths[-1] == tmp_path / "prompt 74-4.mp4"
    with pytest.raises(ValueError, match="75 unique"):
        expected_video_paths(tmp_path, records[:-1])
