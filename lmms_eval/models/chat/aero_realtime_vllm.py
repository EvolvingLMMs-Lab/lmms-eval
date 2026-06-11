"""AeroRealtime + vLLM-Omni chat wrapper for lmms-eval.

Differences vs. ``aero_realtime_chat`` (HuggingFace):
- Backed by ``vllm_omni.AsyncOmni`` (vLLM continuous batching).
- Streams the realtime ``[video, audio_chunk, video, audio_chunk, ..., text]``
  prompt into the engine using the official ``buffer_realtime_omni`` helper
  from ``vllm_omni.model_executor.models.aero_realtime.aero_realtime``.
- Each input chunk produces exactly one decode step. Tokens generated while
  audio/video is still arriving are emitted into the ``audio_pad`` slots of
  the *next* chunk as teacher-forced text-stream ids (this is what
  ``buffer_realtime_omni`` does internally). Because lmms-eval only cares
  about the final text answer, we DO NOT feed sampled tokens back into the
  input stream during the prefill phase — those slots become ``<|rt_pad|>``
  automatically.
- Once the real audio runs out, we keep feeding silence chunks (one decode
  step each) until ``max_new_tokens`` text tokens have been collected or EOS
  is emitted.
- Multiple requests in a batch are run concurrently with ``asyncio.gather``;
  the vLLM-Omni engine schedules them via continuous batching.

CLI registration: ``aero_realtime_vllm_chat``.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

try:
    from qwen_vl_utils import fetch_video
except Exception:  # pragma: no cover
    fetch_video = None

try:
    from vllm.engine.protocol import StreamingInput
    from vllm.renderers.inputs.preprocess import parse_model_prompt
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.tokenizers import cached_tokenizer_from_config

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.inputs.data import OmniTokensPrompt
    from vllm_omni.model_executor.models.aero_realtime.aero_realtime import (
        AeroRealtimeForConditionalGeneration,
        AeroRealtimeStreamState,
    )
except Exception as e:  # pragma: no cover
    eval_logger.error(f"Failed to import vllm-omni / AeroRealtime: {e}")
    AsyncOmni = None
    AeroRealtimeForConditionalGeneration = None
    AeroRealtimeStreamState = None
    OmniTokensPrompt = None
    SamplingParams = None
    RequestOutputKind = None
    cached_tokenizer_from_config = None
    StreamingInput = None
    parse_model_prompt = None


class _StopGen(Exception):
    pass


SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Data loading helpers (mirrors offline_realtime_debug.py)
# ---------------------------------------------------------------------------


@dataclass
class RealtimeSample:
    audio: np.ndarray            # mono float32 @ 16 kHz
    video: np.ndarray            # (T, H, W, 3) uint8
    video_metadata: object
    sample_fps: float


def _load_audio_or_silence(video_path: str, num_frames: int, sample_fps: float) -> np.ndarray:
    """Load the full audio track as mono float32 @ ``SAMPLE_RATE``.

    Uses PyAV which does container-level seek and decodes only what we ask
    for. ~1.7x faster than librosa.load on long mp4s, and degrades to
    silence (not an exception) when the file has no audio stream.
    """
    try:
        import av

        chunks: list[np.ndarray] = []
        src_sr: Optional[int] = None
        with av.open(video_path) as container:
            if not container.streams.audio:
                raise RuntimeError("no audio stream")
            stream = container.streams.audio[0]
            src_sr = int(stream.rate)
            for frame in container.decode(stream):
                arr = frame.to_ndarray()
                if arr.ndim == 2:  # planar (channels, samples) -> mono
                    arr = arr.mean(axis=0)
                chunks.append(arr)
        if not chunks:
            raise RuntimeError("decoded zero audio frames")
        audio = np.concatenate(chunks)
        if audio.dtype.kind == "i":
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32)
        if src_sr is not None and src_sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=src_sr, target_sr=SAMPLE_RATE)
        return audio.astype(np.float32, copy=False)
    except Exception as exc:
        duration_s = num_frames / max(sample_fps, 1e-6)
        eval_logger.warning(f"[aero_rt_vllm] no audio track ({exc}); using {duration_s:.2f}s silence")
        return np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)


def _load_video_sample(video_path: str, *, max_frames: int) -> RealtimeSample:
    video_inputs, sample_fps = fetch_video(
        {
            "type": "video",
            "video": f"file://{video_path}" if not video_path.startswith(("file://", "http")) else video_path,
            "fps": 1,
            "min_frames": 1,
            "max_frames": max_frames,
            "min_pixels": 28800,
            "max_pixels": 300 * 300,
        },
        return_video_sample_fps=True,
        return_video_metadata=True,
    )
    video, metadata = video_inputs
    if hasattr(video, "numpy"):
        video = video.numpy()
    audio = _load_audio_or_silence(video_path, video.shape[0], sample_fps)
    return RealtimeSample(audio=audio, video=video, video_metadata=metadata, sample_fps=sample_fps)


# ---------------------------------------------------------------------------
# Chunk generator
# ---------------------------------------------------------------------------


def _meta(metadata, key: str, default=None):
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def _slice_video_metadata(metadata, frame_idx: int) -> dict[str, object]:
    frames_indices = list(_meta(metadata, "frames_indices", []))
    frame_index = frames_indices[frame_idx] if frame_idx < len(frames_indices) else frame_idx
    fps = _meta(metadata, "fps", 1.0) or 1.0
    return {
        "fps": fps,
        "duration": 1.0 / float(fps),
        "total_num_frames": 1,
        "frames_indices": [frame_index],
        "video_backend": _meta(metadata, "video_backend", "decord"),
    }


def _frame_time(metadata, frame_idx: int, sample_fps: float) -> float:
    frames_indices = list(_meta(metadata, "frames_indices", []))
    if frame_idx < len(frames_indices):
        fps = _meta(metadata, "fps", None) or sample_fps or 1.0
        return float(frames_indices[frame_idx]) / float(fps)
    return float(frame_idx) / float(sample_fps or 1.0)


def _iter_realtime_chunks(
    sample: RealtimeSample,
    *,
    audio_chunk_ms: float,
    ask_text: Optional[str],
    ask_second: Optional[float],
    extra_silence_chunks: int,
):
    """Yield realtime chunks. After real audio is exhausted, yields
    ``extra_silence_chunks`` zero-filled audio chunks for the decode tail."""
    num_frames = int(sample.video.shape[0])
    samples_per_chunk = max(1, int(round(SAMPLE_RATE * audio_chunk_ms / 1000.0)))
    num_audio_chunks = int(np.ceil(len(sample.audio) / samples_per_chunk))
    frame_times = [_frame_time(sample.video_metadata, i, sample.sample_fps) for i in range(num_frames)]

    text_emitted = False
    next_frame = 0
    audio = sample.audio

    for idx in range(num_audio_chunks):
        start = idx * samples_per_chunk
        end = min(start + samples_per_chunk, len(audio))
        audio_chunk = audio[start:end].astype(np.float32, copy=False)
        if audio_chunk.size == 0:
            break
        pad = samples_per_chunk - audio_chunk.shape[0]
        if pad > 0:
            audio_chunk = np.pad(audio_chunk, (0, pad)).astype(np.float32, copy=False)

        t0, t1 = start / SAMPLE_RATE, end / SAMPLE_RATE
        chunk: dict[str, object] = {"audio": audio_chunk, "timestamp": t0}

        if next_frame < num_frames and frame_times[next_frame] < t1:
            video_chunk = sample.video[next_frame : next_frame + 1]
            chunk["video"] = (video_chunk, _slice_video_metadata(sample.video_metadata, next_frame))
            chunk["mm_processor_kwargs"] = {"fps": sample.sample_fps, "do_sample_frames": False}
            next_frame += 1

        if ask_text and not text_emitted and ask_second is not None and t0 <= ask_second < t1:
            chunk["text"] = ask_text
            text_emitted = True

        yield chunk

    # If the question never got injected (e.g. ask_second past the audio end),
    # attach it to a trailing silence chunk so the model still sees it.
    if ask_text and not text_emitted:
        silence = np.zeros(samples_per_chunk, dtype=np.float32)
        t0 = num_audio_chunks * (samples_per_chunk / SAMPLE_RATE)
        yield {"audio": silence, "timestamp": t0, "text": ask_text}
        text_emitted = True

    # Decode-phase silence padding.
    silence = np.zeros(samples_per_chunk, dtype=np.float32)
    for j in range(extra_silence_chunks):
        t0 = (num_audio_chunks + j + (0 if text_emitted else 1)) * (samples_per_chunk / SAMPLE_RATE)
        yield {"audio": silence, "timestamp": t0}


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


@register_model("aero_realtime_vllm_chat")
class AeroRealtimeVLLM(lmms):
    is_simple = False

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
    }

    def __init__(
        self,
        pretrained: str,
        deploy_config: str = "vllm_omni/deploy/aero_realtime.yaml",
        batch_size: Union[int, str] = 4,
        audio_chunk_ms: float = 80.0,
        video_max_frames: int = 512,
        mode: str = "streaming",
        max_model_len: Optional[int] = None,
        tensor_parallel_size: Optional[int] = None,
        stage_0_devices: Optional[str] = None,
        gpu_memory_utilization: float = 0.9,
        skip_mm_profiling: bool = False,
        limit_mm_per_prompt: Optional[str] = None,
        enforce_eager: Optional[bool] = None,
        stage_overrides: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Accept (and ignore) unknown harness kwargs gracefully — be lenient
        # vs. ``aero_realtime_chat`` which is strict, to match how other vLLM
        # wrappers behave.
        if AsyncOmni is None or AeroRealtimeForConditionalGeneration is None:
            raise ImportError(
                "vllm-omni is required for aero_realtime_vllm_chat. "
                "Install vllm-omni and ensure it is importable."
            )
        if fetch_video is None:
            raise ImportError("qwen_vl_utils is required (`pip install qwen-vl-utils`)")
        if librosa is None:
            raise ImportError("librosa is required (`pip install librosa`)")

        self.pretrained = pretrained
        self.audio_chunk_ms = float(audio_chunk_ms)
        self.video_max_frames = int(video_max_frames)
        if mode not in ("streaming", "prefill_decode"):
            raise ValueError(f"mode must be 'streaming' or 'prefill_decode', got {mode!r}")
        self.mode = mode
        self.batch_size_per_gpu = int(batch_size)

        # ---- Construct AsyncOmni ------------------------------------------------
        omni_kwargs: dict[str, object] = {
            "model": pretrained,
            "deploy_config": deploy_config,
            "log_stats": False,
            "gpu_memory_utilization": gpu_memory_utilization,
            "skip_mm_profiling": skip_mm_profiling,
        }
        if max_model_len is not None:
            omni_kwargs["max_model_len"] = max_model_len
        if tensor_parallel_size is not None:
            omni_kwargs["tensor_parallel_size"] = tensor_parallel_size
            omni_kwargs["stage_0_devices"] = stage_0_devices or ",".join(
                str(i) for i in range(tensor_parallel_size)
            )
        elif stage_0_devices is not None:
            omni_kwargs["stage_0_devices"] = stage_0_devices
        if limit_mm_per_prompt:
            # Accept either a dict-like string "audio=4096,video=4096" or json.
            mm_limits: dict[str, int] = {}
            if "=" in limit_mm_per_prompt and "{" not in limit_mm_per_prompt:
                for pair in limit_mm_per_prompt.split(","):
                    if not pair.strip():
                        continue
                    k, v = pair.split("=")
                    mm_limits[k.strip()] = int(v)
            else:
                import json as _json
                mm_limits = {k: int(v) for k, v in _json.loads(limit_mm_per_prompt).items()}
            omni_kwargs["limit_mm_per_prompt"] = mm_limits

        # Per-stage engine overrides (e.g. flip enforce_eager off for cudagraph).
        # Accept either a JSON string or a single-stage-0 shortcut.
        overrides: dict[str, dict] = {}
        if stage_overrides:
            import json as _json
            overrides = _json.loads(stage_overrides)
        if enforce_eager is not None:
            overrides.setdefault("0", {})["enforce_eager"] = bool(enforce_eager)
        if overrides:
            omni_kwargs["stage_overrides"] = overrides

        eval_logger.info(f"[aero_rt_vllm] Building AsyncOmni with: {omni_kwargs}")
        self.omni = AsyncOmni(**omni_kwargs)
        self.model_config = self.omni.model_config
        self.renderer = self.omni.renderer
        if self.model_config is None or self.renderer is None:
            raise RuntimeError("AsyncOmni did not expose model_config/renderer")

        self._tokenizer = cached_tokenizer_from_config(self.model_config)
        self.rt_pad_id = self._tokenizer.convert_tokens_to_ids("<|rt_pad|>")
        self.audio_pad_id = self._tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.video_pad_id = self._tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.im_end_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
        # Tokens to drop from collected output before decoding.
        self._skip_token_ids = {
            self.rt_pad_id,
            self.audio_pad_id,
            self.video_pad_id,
            self._tokenizer.convert_tokens_to_ids("<|audio_start|>"),
            self._tokenizer.convert_tokens_to_ids("<|audio_end|>"),
            self._tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            self._tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        }
        self._stop_token_ids = {self.im_end_id}
        if self._tokenizer.eos_token_id is not None:
            self._stop_token_ids.add(self._tokenizer.eos_token_id)

        # Single-process for now (vLLM does its own TP/PP internally).
        self._rank = 0
        self._world_size = 1
        self._max_length = max_model_len or 8192

    # --------------------------- lmms required properties -----------------------

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return "cuda"

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def config(self):
        return self.model_config

    def loglikelihood(self, requests):
        raise NotImplementedError("Loglikelihood is not implemented for AeroRealtimeVLLM")

    # --------------------------- request -> sample ------------------------------

    def _extract_video_and_text(self, chat: ChatMessages) -> Tuple[str, str]:
        """Pull (first_video_path, concatenated_user_text) out of ChatMessages."""
        video_path: Optional[str] = None
        text_parts: list[str] = []
        for message in chat.messages:
            for content in message.content:
                if content.type == "video" and video_path is None:
                    video_path = content.url
                elif content.type == "text" and message.role == "user":
                    text_parts.append(content.text)
        if video_path is None:
            raise ValueError("aero_realtime_vllm_chat requires a video in the request")
        ask_text = " ".join(t for t in text_parts if t).strip()
        if ask_text and not ask_text.endswith(" "):
            ask_text = ask_text + " "
        return video_path, ask_text

    # --------------------------- streaming pipeline -----------------------------

    async def _render_streaming_input(self, prompt):
        """Convert a chunk-prompt dict / OmniTokensPrompt to a StreamingInput."""
        parsed = parse_model_prompt(self.model_config, prompt)
        (engine_input,) = await self.renderer.render_cmpl_async([parsed])
        extra = prompt.get("additional_information") if isinstance(prompt, dict) else None
        if extra is not None and isinstance(engine_input, dict):
            engine_input["additional_information"] = extra
        return StreamingInput(prompt=engine_input)

    @staticmethod
    def _split_chunks(chunks):
        """(prefill, tail). Prefill includes everything up to and including
        the text-injection chunk (so the question's audio_start/audio_pad
        live in the fused prefill, matching training); tail is only the
        decode-phase silence chunks."""
        all_chunks = list(chunks)
        text_idx = next(
            (i for i, c in enumerate(all_chunks) if c.get("text")), len(all_chunks)
        )
        return all_chunks[: text_idx + 1], all_chunks[text_idx + 1 :]

    def _fuse_prompts(self, prompts):
        """Concat per-chunk prompts into one OmniTokensPrompt.

        Runs of consecutive pure-audio chunks (no video/text in the delta,
        no video item in mm_data) are merged: their audio arrays are
        concatenated into one item and the contiguous ``<|audio_pad|>``
        tokens in the prompt collapse to a single placeholder. vllm's
        chunked audio path then expands that placeholder to N audio tokens
        (one per 80ms sub-chunk) via ``audio_output_lengths``. This
        reduces the multi-modal item count from ~33k to ~65 for a 1h
        video, making vllm's O(items × prompt_len) placeholder matching
        tractable.

        Returns None if empty.
        """
        prompts = list(prompts)
        if not prompts:
            return None

        audio_pad_id = self.audio_pad_id

        pids: list[int] = []
        ts_ids: list[int] = []
        audios: list = []
        videos: list = []
        mm_kwargs: dict[str, object] = {}

        # Running buffer of pure-audio chunks waiting to be flushed.
        pending_audio: list = []
        pending_ts_ids: list[int] = []

        def flush_audio_buffer():
            if not pending_audio:
                return
            merged = (
                pending_audio[0]
                if len(pending_audio) == 1
                else np.concatenate(pending_audio).astype(np.float32, copy=False)
            )
            audios.append(merged)
            pids.append(audio_pad_id)
            ts_ids.extend(pending_ts_ids)
            pending_audio.clear()
            pending_ts_ids.clear()

        for p in prompts:
            prompt_ids = p["prompt_token_ids"]
            mm = p.get("multi_modal_data") or {}
            extra = p.get("additional_information") or {}
            chunk_ts = (extra.get("ids") or {}).get("text_stream_ids") or []
            audio_item = mm.get("audio")
            video_item = mm.get("video")

            for k, v in (p.get("mm_processor_kwargs") or {}).items():
                mm_kwargs.setdefault(k, v)

            # A "pure audio" chunk: only audio_pad token(s) in the delta,
            # no video item, no extra prompt scaffolding.
            non_audio_pad_ids = [tid for tid in prompt_ids if tid != audio_pad_id]
            is_pure_audio = (
                audio_item is not None
                and video_item is None
                and not non_audio_pad_ids
            )

            if is_pure_audio:
                pending_audio.append(np.asarray(audio_item))
                pending_ts_ids.extend(chunk_ts)
                continue

            # Boundary: flush, then emit this chunk verbatim.
            flush_audio_buffer()
            pids.extend(prompt_ids)
            ts_ids.extend(chunk_ts)
            if audio_item is not None:
                audios.append(audio_item)
            if video_item is not None:
                videos.append(video_item)

        flush_audio_buffer()

        mm_data: dict[str, object] = {}
        if audios:
            mm_data["audio"] = audios
        if videos:
            mm_data["video"] = videos
        fused = {
            "prompt_token_ids": pids,
            "multi_modal_data": mm_data,
            "additional_information": {
                "ids": {"text_stream_ids": ts_ids},
                "meta": {"aero_realtime": True},
            },
        }
        if mm_kwargs:
            fused["mm_processor_kwargs"] = mm_kwargs
        return OmniTokensPrompt(**fused)

    async def _generate_one(
        self,
        chat: ChatMessages,
        gen_kwargs: dict,
    ) -> Tuple[str, int]:
        """Drive AeroRealtime through ``self.mode``:

        - ``streaming``: every chunk goes through as a separate StreamingInput
          (matches the demo's offline_realtime_debug.py).
        - ``prefill_decode``: chunks before the question are fused into one
          big OmniTokensPrompt and submitted as a single prefill (text_stream
          on every audio_pad is rt_pad — model is forced silent in this
          window). The question chunk and silence tail are then streamed
          chunk-by-chunk with token feedback.
        """
        video_path, ask_text = self._extract_video_and_text(chat)
        sample = await asyncio.to_thread(
            _load_video_sample, video_path, max_frames=self.video_max_frames
        )

        merged = {**self.DEFAULT_GEN_KWARGS, **(gen_kwargs or {})}
        max_new_tokens = int(merged.get("max_new_tokens", 128))
        temperature = float(merged.get("temperature", 0.0))
        top_p = float(merged.get("top_p", 1.0) or 1.0)
        top_k = int(merged.get("top_k", -1) or -1)

        audio_duration = float(sample.audio.shape[0]) / SAMPLE_RATE
        ask_second = max(0.0, audio_duration - (self.audio_chunk_ms / 1000.0) * 0.5)
        chunk_kwargs = dict(
            audio_chunk_ms=self.audio_chunk_ms,
            ask_text=ask_text or None,
            ask_second=ask_second if ask_text else None,
            extra_silence_chunks=max_new_tokens,
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=1,
            output_kind=RequestOutputKind.DELTA,
            skip_clone=True,
        )

        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()

        async def streaming_gen_pure():
            async def chunk_iter():
                for c in _iter_realtime_chunks(sample, **chunk_kwargs):
                    yield c

            async for p in AeroRealtimeForConditionalGeneration.buffer_realtime_omni(
                chunk_iter(), input_stream, self.model_config
            ):
                yield await self._render_streaming_input(p)

        async def streaming_gen_prefill_decode():
            prefill_chunks, tail_chunks = self._split_chunks(
                _iter_realtime_chunks(sample, **chunk_kwargs)
            )
            shared_state = AeroRealtimeStreamState()

            # Phase 1 — fuse pre-question chunks into one prefill (every
            # audio_pad becomes rt_pad; model is silent across the video).
            async def pre_iter():
                for c in prefill_chunks:
                    yield c

            prefill_q: asyncio.Queue[list[int]] = asyncio.Queue()
            prefill_prompts = []
            async for p in AeroRealtimeForConditionalGeneration.buffer_realtime_omni(
                pre_iter(), prefill_q, self.model_config, state=shared_state
            ):
                prefill_q.put_nowait([])
                prefill_prompts.append(p)
            fused = self._fuse_prompts(prefill_prompts)
            if fused is not None:
                yield await self._render_streaming_input(fused)

            # Phase 2 — question chunk + silence tail with token feedback.
            async def tail_iter():
                for c in tail_chunks:
                    yield c

            async for p in AeroRealtimeForConditionalGeneration.buffer_realtime_omni(
                tail_iter(), input_stream, self.model_config, state=shared_state
            ):
                yield await self._render_streaming_input(p)

        streaming_gen = (
            streaming_gen_prefill_decode
            if self.mode == "prefill_decode"
            else streaming_gen_pure
        )

        request_id = f"aero-rt-vllm-{uuid.uuid4()}"
        collected: list[int] = []
        try:
            async for output in self.omni.generate(
                streaming_gen(),
                request_id=request_id,
                sampling_params_list=[sampling_params],
            ):
                if not output.outputs:
                    continue
                tids = list(output.outputs[0].token_ids)
                if not tids:
                    continue
                input_stream.put_nowait(tids)
                for tid in tids:
                    if tid in self._stop_token_ids:
                        raise _StopGen()
                    if tid in self._skip_token_ids:
                        continue
                    collected.append(tid)
                    if len(collected) >= max_new_tokens:
                        raise _StopGen()
        except _StopGen:
            pass
        finally:
            try:
                await self.omni.abort(request_id)
            except Exception:
                pass

        text = self._tokenizer.decode(collected, skip_special_tokens=True)
        eval_logger.info(f"[aero_rt_vllm/{request_id[-8:]}] done text={text!r}")
        return text, len(collected)

    # --------------------------- public lmms API --------------------------------

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res: list[Optional[GenerationResult]] = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="AeroRT-vLLM Responding")
        bs = self.batch_size_per_gpu
        totals = {"elapsed": 0.0, "tokens": 0}

        # All batches share one event loop so that AsyncOmni's per-call
        # background tasks (final_output_handler, ZMQ output reader) stay
        # alive across batches. Re-creating a loop per batch leaves those
        # tasks cancelled while AsyncOmni still holds dead references and
        # skips re-spawning them, hanging every batch after the first.
        async def _run_all() -> None:
            for start in range(0, len(requests), bs):
                batch = requests[start : start + bs]
                chats_and_gens = []
                for req in batch:
                    _, doc_to_messages, gen_kwargs, doc_id, task, split = req.arguments
                    raw = doc_to_messages(self.task_dict[task][split][doc_id])
                    chat = ChatMessages(messages=raw)
                    chats_and_gens.append((chat, dict(gen_kwargs or {})))

                t0 = time.time()
                batch_results = await asyncio.gather(
                    *[self._generate_one(c, g) for c, g in chats_and_gens]
                )
                totals["elapsed"] += time.time() - t0

                for i, (text, ntok) in enumerate(batch_results):
                    gen_kwargs = chats_and_gens[i][1]
                    ctx = batch[i].arguments[0]
                    res[start + i] = GenerationResult(text=text, token_counts=TokenCounts(output_tokens=ntok))
                    totals["tokens"] += ntok
                    self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), text)
                    eval_logger.debug(f"[aero_rt_vllm] response={text!r}")
                    pbar.update(1)

        asyncio.run(_run_all())
        total_elapsed = totals["elapsed"]
        total_tokens = totals["tokens"]

        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed,
            avg_speed=(total_tokens / total_elapsed) if total_elapsed > 0 else 0,
            additional_metrics={"rank": self.rank},
        )
        pbar.close()
        return res  # type: ignore[return-value]

    def generate_until_multi_round(self, requests: List[Instance]) -> List[List[str]]:
        """Per-round full re-prefill (KV not reused across rounds).

        Mirrors how ``ovo_forward_doc_to_text`` is consumed by chat models in
        ``configurable_messages_task``: round 0 calls ``doc_to_messages(doc)``,
        subsequent rounds call ``doc_to_messages(doc, round_idx, previous_output,
        previous_round_info)`` and end when terminal_sign==True or payload is
        not a length-4 tuple.

        Multiple samples are dispatched concurrently; within a single sample
        rounds are serial (they depend on previous_output).
        """
        results: list[Optional[List[str]]] = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="AeroRT-vLLM MR Responding")

        async def _run_sample(req: Instance) -> List[str]:
            _, doc_to_messages, gen_kwargs, doc_id, task, split = req.arguments
            doc = self.task_dict[task][split][doc_id]
            gen_kwargs = dict(gen_kwargs or {})

            round_outputs: List[str] = []
            previous_round_info = None
            round_idx = 0
            while True:
                if round_idx == 0:
                    raw = doc_to_messages(doc)
                else:
                    payload = doc_to_messages(
                        doc,
                        round_idx=round_idx,
                        previous_output=list(round_outputs),
                        previous_round_info=previous_round_info,
                    )
                    if not (isinstance(payload, tuple) and len(payload) == 4):
                        break
                    raw, terminal, _prev, previous_round_info = payload
                    if terminal:
                        break

                chat = ChatMessages(messages=raw)
                text, _ = await self._generate_one(chat, gen_kwargs)
                round_outputs.append(text)
                round_idx += 1
            return round_outputs

        bs = self.batch_size_per_gpu

        async def _run_all() -> None:
            for start in range(0, len(requests), bs):
                batch = requests[start : start + bs]
                batch_results = await asyncio.gather(*[_run_sample(r) for r in batch])
                for i, ro in enumerate(batch_results):
                    results[start + i] = ro
                    pbar.update(1)

        asyncio.run(_run_all())

        pbar.close()
        return results  # type: ignore[return-value]

    # --------------------------- teardown ---------------------------------------

    def __del__(self):
        pass
