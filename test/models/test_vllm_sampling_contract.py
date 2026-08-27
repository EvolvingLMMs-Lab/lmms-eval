"""Behavioral contract for task-level vLLM sampling parameters."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from lmms_eval.models.chat.vllm import VLLM


def _model():
    model = VLLM.__new__(VLLM)
    model.task_dict = {"task": {"test": [{"question": "What is shown?"}]}}
    model.max_new_tokens = 16
    model.max_pixels = 1605632
    model.min_image_pixels = 28
    model.max_frame_num = 32
    model.fps = None
    model.nframes = 32
    model.is_qwen3_vl = False
    return model


def _request(generation_kwargs):
    def doc_to_messages(doc):
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": doc["question"]}],
            }
        ]

    return SimpleNamespace(
        arguments=(
            "context",
            doc_to_messages,
            generation_kwargs,
            0,
            "task",
            "test",
        )
    )


def test_vllm_forwards_task_sampling_parameters_without_mutating_input():
    model = _model()
    generation_kwargs = {
        "max_new_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.8,
        "seed": 0,
        "top_k": 20,
        "min_p": 0.1,
        "repetition_penalty": 1.1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    original_generation_kwargs = deepcopy(generation_kwargs)

    _, sampling_params = model.make_one_request(_request(generation_kwargs))

    assert sampling_params == {
        "max_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.8,
        "seed": 0,
        "top_k": 20,
        "min_p": 0.1,
        "repetition_penalty": 1.1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    assert generation_kwargs == original_generation_kwargs


def test_vllm_preserves_explicit_single_choice():
    model = _model()
    generation_kwargs = {
        "max_new_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
    }

    _, sampling_params = model.make_one_request(_request(generation_kwargs))

    assert sampling_params["n"] == 1


@pytest.mark.parametrize("n", [0, 2, -1, 1.0, True, "1"])
def test_vllm_rejects_unsupported_choice_counts(n):
    model = _model()
    generation_kwargs = {
        "max_new_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": n,
    }

    with pytest.raises(ValueError, match="n.*integer 1"):
        model.make_one_request(_request(generation_kwargs))
