"""Short-form ASR with native Transformers Qwen3-ASR checkpoints."""

from itertools import groupby, islice
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from transformers import AutoProcessor

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.audio_processing import downsample_audio

AutoModelForMultimodalLM, _ = optional_import("transformers", "AutoModelForMultimodalLM")


def _waveform(media: list[Any], target_rate: int) -> np.ndarray:
    if len(media) != 1:
        raise ValueError("Qwen3-ASR expects one recording per example")
    (audio,) = media
    if hasattr(audio, "get_all_samples"):
        decoded = audio.get_all_samples()
        samples, rate = decoded.data.numpy(), decoded.sample_rate
    else:
        samples, rate = audio["array"], audio["sampling_rate"]
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:  # Hugging Face datasets and TorchCodec use channels first.
        samples = samples.mean(axis=0)
    if samples.ndim != 1 or samples.size == 0 or rate is None or rate <= 0:
        raise ValueError("Expected nonempty audio with a positive sampling rate")
    return downsample_audio(samples, rate, target_rate) if rate != target_rate else samples


@register_model("qwen3_asr")
class Qwen3ASR(lmms):
    """Evaluate native Qwen3-ASR models on single-recording transcription tasks."""

    def __init__(self, pretrained: str = "Qwen/Qwen3-ASR-0.6B-hf", revision: str | None = None, language: str | None = None, device: str | None = None, batch_size: int | str = 1) -> None:
        super().__init__()
        if AutoModelForMultimodalLM is None:
            raise ImportError("qwen3_asr requires Transformers with native Qwen3-ASR support (tested with 5.24.0).")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.language = language
        self.accelerator = Accelerator(cpu=device == "cpu")
        self._rank = self.accelerator.process_index
        self._world_size = self.accelerator.num_processes
        self.device = self.accelerator.device if self.world_size > 1 or device is None else torch.device(device)
        self.processor = AutoProcessor.from_pretrained(pretrained, revision=revision)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForMultimodalLM.from_pretrained(pretrained, revision=revision, dtype="auto").to(self.device).eval()

    @torch.inference_mode()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Transcribe one recording per request, preserving request order and stop strings."""
        responses = []
        for options, group in groupby(requests, key=lambda request: request.args[1]):
            generation_kwargs = dict(options)
            until = generation_kwargs.pop("until", []) or []
            if isinstance(until, str):
                until = [until]
            generation_kwargs.setdefault("max_new_tokens", 256)
            if generation_kwargs.get("temperature") == 0:
                generation_kwargs.pop("temperature")
                generation_kwargs["do_sample"] = False
            while batch := list(islice(group, self.batch_size)):
                waveforms = []
                for request in batch:
                    _, _, to_audio, doc_id, task, split = request.args
                    media = to_audio(self.task_dict[task][split][doc_id])
                    waveforms.append(_waveform(media, self.processor.feature_extractor.sampling_rate))
                inputs = self.processor.apply_transcription_request(audio=waveforms, language=self.language, return_tensors="pt", processor_kwargs={"padding": True})
                inputs = inputs.to(self.device, self.model.dtype)
                output_ids = self.model.generate(**inputs, **generation_kwargs)
                generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
                texts = self.processor.decode(generated_ids, return_format="transcription_only")
                for request, text in zip(batch, texts, strict=True):
                    for stop in until:
                        if stop:
                            text = text.split(stop, 1)[0]
                    responses.append(text)
                    self.cache_hook.add_partial("generate_until", (request.args[0], options), text)
        return responses

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """This backend evaluates generation tasks."""
        raise NotImplementedError("Qwen3-ASR supports transcription tasks only")

    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        """This backend accepts single-turn transcription requests."""
        raise NotImplementedError("Qwen3-ASR supports single-turn transcription only")
