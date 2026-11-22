from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest
import torch

from lmms_eval.api.model import lmms
from lmms_eval.models import get_model
from lmms_eval.models.simple.qwen3_asr import Qwen3ASR, _waveform


def test_audio_types_and_resampling() -> None:
    wav = np.stack([np.zeros(2400), np.ones(2400)]).astype("float32")
    decoded = NS(get_all_samples=lambda: NS(data=torch.from_numpy(wav), sample_rate=24000))
    modern = _waveform([decoded], 16000)
    legacy = _waveform([{"array": wav, "sampling_rate": 24000}], 16000)
    assert modern.shape == (1600,)
    np.testing.assert_allclose(modern, legacy)
    np.testing.assert_allclose(modern[50:-50], 0.5, atol=0.005)


@pytest.mark.parametrize("media", [[], [{}, {}], [{"array": [], "sampling_rate": 16000}], [{"array": [0.0], "sampling_rate": 0}]])
def test_invalid_media(media: list[dict]) -> None:
    with pytest.raises(ValueError):
        _waveform(media, 16000)


def test_batched_transcription_preserves_order_and_options() -> None:
    wrapper = Qwen3ASR.__new__(Qwen3ASR)
    lmms.__init__(wrapper)
    wrapper.device, wrapper.batch_size, wrapper.language = "cpu", 2, "en"
    samples = [np.array([i], dtype="float32") for i in [7, 9, 3]]
    wrapper.task_dict = {"asr": {"test": samples}}
    calls = []

    class Batch(dict):
        def to(self, *_: Any) -> dict:
            return self

    def prepare(audio: list[np.ndarray], **kwargs: Any) -> dict:
        assert kwargs["language"] == "en"
        ids = torch.tensor([[5, 6] for _ in audio])
        return Batch(input_ids=ids, markers=torch.tensor([int(w[0]) for w in audio]))

    def generate(input_ids: torch.Tensor, markers: torch.Tensor, **options: Any) -> torch.Tensor:
        calls.append(options)
        return torch.cat([input_ids, markers[:, None]], dim=1)

    def decode(generated: torch.Tensor, return_format: str) -> list[str]:
        assert return_format == "transcription_only" and generated.shape[1] == 1
        return [f"clip {row.item()} STOP more" for row in generated]

    wrapper.processor = NS(feature_extractor=NS(sampling_rate=16000), apply_transcription_request=prepare, decode=decode)
    wrapper.model = NS(generate=generate, dtype=torch.float32)
    first = {"temperature": 0, "until": " STOP"}
    next_options = {"max_new_tokens": 8}
    requests = [NS(args=("transcribe", config, lambda w: [{"array": w, "sampling_rate": 16000}], i, "asr", "test")) for i, config in enumerate([first, first, next_options])]
    assert get_model("qwen3_asr") is Qwen3ASR
    assert wrapper.generate_until(requests) == ["clip 7", "clip 9", "clip 3 STOP more"]
    assert first == {"temperature": 0, "until": " STOP"}
    assert calls == [{"max_new_tokens": 256, "do_sample": False}, {"max_new_tokens": 8}]
