import inspect
from types import SimpleNamespace

import pytest

from lmms_eval.models.simple.diffusers_wm_base import DiffusersWMBase
from lmms_eval.models.simple.wan2_2_t2v import Wan2_2_T2V
from lmms_eval.tasks.vbench.build_dataset import (
    build_configs,
    deterministic_seed,
    merge_prompts,
)


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


def test_generation_uses_row_seed(tmp_path):
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


def test_wan22_keeps_backward_compatible_defaults():
    defaults = {name: parameter.default for name, parameter in inspect.signature(Wan2_2_T2V.__init__).parameters.items()}

    assert defaults["height"] == 480
    assert defaults["width"] == 832
    assert defaults["num_frames"] == 81
    assert defaults["num_inference_steps"] == 50
    assert defaults["guidance_scale"] == 5.0
    assert defaults["guidance_scale_2"] is None
    assert defaults["flow_shift"] is None
    assert defaults["fps"] == 16
    assert defaults["negative_prompt"] is None
