"""Hermetic unit tests for whisper_vllm LibriSpeech EOT-recovery helpers.

No Whisper checkpoint, no LibriSpeech download, no GPU. `vllm` is stubbed so
the CPU contract suite can import `whisper_vllm.py`.
"""

from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest


def _install_vllm_stub() -> None:
    class _SamplingParams:
        def __init__(self, **kwargs):
            self.temperature = kwargs.get("temperature", 0)
            self.top_p = kwargs.get("top_p", 0)
            self.max_tokens = kwargs.get("max_tokens", 256)
            self.min_tokens = kwargs.get("min_tokens", 0)

        def clone(self):
            return copy.copy(self)

    vllm = types.ModuleType("vllm")
    vllm.LLM = object
    vllm.SamplingParams = _SamplingParams
    sys.modules["vllm"] = vllm

    if "tqdm" not in sys.modules:
        tqdm_mod = types.ModuleType("tqdm")

        def _tqdm(iterable=None, **_kwargs):
            return iterable if iterable is not None else []

        tqdm_mod.tqdm = _tqdm
        sys.modules["tqdm"] = tqdm_mod

    if "lmms_eval.api.instance" not in sys.modules:
        inst = types.ModuleType("lmms_eval.api.instance")
        inst.Instance = object
        sys.modules["lmms_eval.api.instance"] = inst
    if "lmms_eval.api.model" not in sys.modules:
        model = types.ModuleType("lmms_eval.api.model")

        class lmms:
            def __init__(self):
                self._rank = 0
                self._world_size = 1
                self.task_dict = {}

        model.lmms = lmms
        sys.modules["lmms_eval.api.model"] = model
    if "lmms_eval.api.registry" not in sys.modules:
        registry = types.ModuleType("lmms_eval.api.registry")
        registry.register_model = lambda _name: lambda cls: cls
        sys.modules["lmms_eval.api.registry"] = registry


def _load_whisper_vllm():
    _install_vllm_stub()
    path = pathlib.Path(__file__).resolve().parents[2] / "lmms_eval/models/simple/whisper_vllm.py"
    spec = importlib.util.spec_from_file_location("whisper_vllm_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


w = _load_whisper_vllm()


@pytest.mark.parametrize(
    "task, expected",
    [
        ("librispeech", True),
        ("librispeech_dev_clean", True),
        ("librispeech_dev_other", True),
        ("librispeech_test_clean", True),
        ("librispeech_test_other", True),
        ("  librispeech_test_clean  ", True),
        ("openslr_librispeech", False),
        ("open_asr_librispeech_test", False),
        ("open_asr_librispeech_dev", False),
        ("fleurs", False),
        ("fleurs_en_us", False),
        ("common_voice", False),
        ("", False),
    ],
)
def test_is_librispeech_task_soft_gate(task, expected):
    assert w._is_librispeech_task(task) is expected


@pytest.mark.parametrize(
    "head, tail, expected",
    [
        ("the cat sat", "sat on the mat", "the cat sat on the mat"),
        ("hello world", "foo bar", "hello world foo bar"),
        ("one two three", "two three four", "one two three four"),
        ("", "tail only", "tail only"),
        ("head only", "", "head only"),
        ("repeat repeat", "repeat repeat more", "repeat repeat more"),
    ],
)
def test_seam_dedup(head, tail, expected):
    assert w._seam_dedup(head, tail) == expected


def test_keep_tail_empty_rejected():
    assert w._keep_tail("", 10.0) is False
    assert w._keep_tail("   ", 10.0) is False


def test_keep_tail_short_clip_keeps_sparse_text():
    # Density gate only applies at >= _MIN_DENSITY_TAIL_S (4 s).
    assert w._keep_tail("one", 3.0) is True


def test_keep_tail_long_sparse_rejected():
    assert w._keep_tail("one", 10.0) is False


def test_keep_tail_long_dense_kept():
    text = "one two three four five six"
    assert w._keep_tail(text, 10.0) is True


def test_too_short_below_min_duration_only_empty():
    assert w._too_short("", 1.0) is True
    assert w._too_short("hello world", 1.0) is False


def test_too_short_empty_always():
    assert w._too_short("", 10.0) is True


def test_too_short_below_expected_rate():
    # floor = 0.5 * duration * 2.0 w/s = duration words. 9 words in 10 s is short.
    assert w._too_short(" ".join(["w"] * 9), 10.0) is True
    assert w._too_short(" ".join(["w"] * 10), 10.0) is False


def test_keep_retry_requires_longer_and_dense():
    old = "short hyp"
    longer = "this recovered transcript has many more words than before"
    assert w._keep_retry(old, old, 10.0) is False
    assert w._keep_retry(old, longer, 10.0) is True
    # 3 words / 20 s = 0.15 w/s < 0.5 keep floor
    assert w._keep_retry("a b", "a b c", 20.0) is False


def test_generate_with_fallback_omits_gzip_kwargs_when_disabled():
    captured = []

    class _LLM:
        def generate(self, prompts, sampling_params, **kwargs):
            captured.append(kwargs)
            return ["ok"]

    w._generate_with_fallback(_LLM(), ["p"], object(), whisper_mod=types.SimpleNamespace(), enable_fallback=False)
    assert captured == [{"use_tqdm": False}]


def test_generate_with_fallback_passes_gzip_kwargs_for_librispeech():
    captured = []
    whisper_mod = types.SimpleNamespace(
        TEMPERATURES=(0.0, 1.0),
        COMPRESSION_RATIO_THRESHOLD=1.35,
        LOGPROB_THRESHOLD=-1.0,
    )

    class _LLM:
        def generate(self, prompts, sampling_params, **kwargs):
            captured.append(kwargs)
            return ["ok"]

    w._generate_with_fallback(_LLM(), ["p"], object(), whisper_mod=whisper_mod, enable_fallback=True)
    assert captured[0]["temperature"] == (0.0, 1.0)
    assert captured[0]["compression_ratio_threshold"] == 1.35
    assert captured[0]["logprob_threshold"] == -1.0


def test_generate_with_fallback_typeerror_drops_gzip_kwargs():
    calls = []

    class _LLM:
        def generate(self, prompts, sampling_params, **kwargs):
            calls.append(kwargs)
            if "compression_ratio_threshold" in kwargs:
                raise TypeError("stock LLM.generate has no gzip kwargs")
            return ["ok"]

    whisper_mod = types.SimpleNamespace(
        TEMPERATURES=(0.0,),
        COMPRESSION_RATIO_THRESHOLD=1.35,
        LOGPROB_THRESHOLD=-1.0,
    )
    out = w._generate_with_fallback(_LLM(), ["p"], object(), whisper_mod=whisper_mod, enable_fallback=True)
    assert out == ["ok"]
    assert "compression_ratio_threshold" in calls[0]
    assert calls[1] == {"use_tqdm": False}


class _FakeSP:
    def __init__(self, max_tokens=256):
        self.max_tokens = max_tokens
        self.min_tokens = 0

    def clone(self):
        return copy.copy(self)


def _whisper_for_retry():
    inst = w.WhisperVllm.__new__(w.WhisperVllm)
    inst._whisper_mod = object()
    inst.calls = []

    def _generate(prompts, sampling_params, *, enable_fallback=False):
        inst.calls.append({"n": len(prompts), "min_tokens": sampling_params.min_tokens, "fallback": enable_fallback})
        recovered = "this recovered transcript has many more words than the original hyp"
        return [SimpleNamespace(outputs=[SimpleNamespace(text=recovered)]) for _ in prompts]

    inst._generate = _generate
    return inst


def test_retry_short_hyps_skips_when_timestamps_present():
    inst = _whisper_for_retry()
    hyps = inst._retry_short_hyps(
        transcriptions=["tiny"],
        batched_prompts=[{}],
        durations=[10.0],
        covered=[3.5],
        recover=[True],
        sampling_params=_FakeSP(),
    )
    assert hyps == ["tiny"]
    assert inst.calls == []


def test_retry_short_hyps_skips_non_librispeech():
    inst = _whisper_for_retry()
    hyps = inst._retry_short_hyps(
        transcriptions=[""],
        batched_prompts=[{}],
        durations=[10.0],
        covered=[None],
        recover=[False],
        sampling_params=_FakeSP(),
    )
    assert hyps == [""]
    assert inst.calls == []


def test_retry_short_hyps_runs_when_no_timestamps_and_too_short():
    inst = _whisper_for_retry()
    hyps = inst._retry_short_hyps(
        transcriptions=["hi"],
        batched_prompts=[{}],
        durations=[10.0],
        covered=[None],
        recover=[True],
        sampling_params=_FakeSP(),
    )
    assert inst.calls and inst.calls[0]["fallback"] is True
    assert len(hyps[0].split()) > 1
