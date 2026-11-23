"""minicpm_o_45_duplex — MiniCPM-o-4.5 full-duplex streaming evaluation wrapper.

MiniCPM-o 4.5 is (to our knowledge) the first released open-weight model with
a *full-duplex* omni-modal live streaming mode: while audio/video input chunks
keep arriving, the model concurrently emits text+speech and reports
``is_listen`` / ``end_of_turn`` decisions per chunk. This wrapper ports that
official "Duplex Omni Mode" into lmms-eval so duplex benchmarks
(Full-Duplex-Bench, Omni-DuplexEval, ...) can run offline: input streams are
replayed chunk-by-chunk through ``streaming_prefill`` + ``streaming_generate``
exactly as in the model card example, at full speed instead of wall-clock
speed (the model's internal clock advances per chunk, so behaviour only
depends on the chunk sequence, not on wall time).

Official recipe being wrapped:

    model = AutoModel.from_pretrained("openbmb/MiniCPM-o-4_5", trust_remote_code=True, ...)
    model = model.as_duplex()
    model.prepare(prefix_system_prompt="Streaming Omni Conversation.",
                  ref_audio=ref_audio, prompt_wav_path=ref_audio_path)
    for chunk_idx in range(len(audio_segments)):
        model.streaming_prefill(audio_waveform=audio_chunk, frame_list=frame_list, max_slice_nums=1)
        result = model.streaming_generate(prompt_wav_path=ref_audio_path,
                                          max_new_speak_tokens_per_chunk=20,
                                          decode_mode="sampling")
        # result: {"is_listen", "text", "end_of_turn", "current_time", "audio_waveform"}

Request contract (per doc):
    doc_to_visual returns an ordered list of media files. Audio files play the
    role of the *user's speech* and are prefill-streamed first; video files are
    *live streams* and follow afterwards in the same session. When the task
    sets ``duplex_multi_session`` in generation_kwargs, every media item is
    instead run as an independent session (used by Full-Duplex-Bench, which
    compares a clean and an overlapped conversation in one sample).

The wrapper returns a JSON string::

    {"sessions": [{"chunks": [{"chunk_idx", "text", "is_listen",
                               "end_of_turn", "current_time"}]}]}

so a task's ``process_results`` can score streaming behaviour (response
timing, interruption handling, proactive speech, ...) without any
model-specific knowledge.

The reference voice defaults to ``assets/HT_ref_audio.wav`` from the model
repo (deterministic across runs); override with ``ref_audio=``.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.minicpm_o_45 import (
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    _load_minicpmo_utils,
    _patch_transformers_tp_plan_bug,
)

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    eval_logger.warning("Failed to import transformers; MiniCPM-o-4.5 requires transformers==4.51.0")

SPEECH_SAMPLE_RATE = 16000
DEFAULT_REF_AUDIO_REPO_FILE = "assets/HT_ref_audio.wav"


@register_model("minicpm_o_45_duplex")
class MiniCPM_O_45Duplex(lmms):
    """MiniCPM-o-4.5 full-duplex offline evaluation."""

    def __init__(
        self,
        pretrained: str = "openbmb/MiniCPM-o-4_5",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: str = "sdpa",
        dtype: Optional[torch.dtype] = torch.bfloat16,
        init_vision: bool = True,
        init_audio: bool = True,
        init_tts: bool = True,
        duplex_system_prompt: str = "Streaming Omni Conversation.",
        ref_audio: Optional[str] = None,
        max_new_speak_tokens_per_chunk: int = 20,
        max_slice_nums: int = 1,
        stack_frames: int = 1,
        use_ffmpeg: bool = True,
        # The official demo decodes with "sampling"; greedy keeps offline
        # evaluation reproducible and can be overridden via model_args.
        decode_mode: str = "greedy",
        duplex_save_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        _patch_transformers_tp_plan_bug()

        self.system_prompt = duplex_system_prompt
        self.max_new_speak_tokens_per_chunk = max_new_speak_tokens_per_chunk
        self.max_slice_nums = max_slice_nums
        self.stack_frames = stack_frames
        self.use_ffmpeg = use_ffmpeg
        self.decode_mode = decode_mode
        self.duplex_save_dir = duplex_save_dir

        # Rank/device come from the launcher environment only: an early
        # `Accelerator()` would initialize torch.distributed before
        # `from_pretrained`, and transformers 4.51's `caching_allocator_warmup`
        # then iterates `model._tp_plan`, which is None for the custom
        # MiniCPMO class. Evaluation shards requests across ranks; no DDP
        # wrapping is needed.
        self._rank = int(os.environ.get("RANK", 0))
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if self._world_size > 1:
            self._device = torch.device(f"cuda:{local_rank}")
            self.device_map = f"cuda:{local_rank}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        self._model = AutoModel.from_pretrained(
            pretrained,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
            device_map=self.device_map,
            init_vision=init_vision,
            init_audio=init_audio,
            init_tts=init_tts,
        )
        if self.device_map == "auto":
            self._model = self._model.eval()
        else:
            self._model = self._model.to(self._device).eval()

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self._config = self._model.config

        # Convert the half-duplex checkpoint into its duplex twin exactly once.
        if not hasattr(self._model, "streaming_prefill"):
            raise RuntimeError(
                f"{pretrained!r} does not expose the MiniCPM-o-4.5 duplex API (missing streaming_prefill); "
                "the minicpm_o_45_duplex wrapper requires openbmb/MiniCPM-o-4_5 or an as_duplex()-capable checkpoint."
            )
        if hasattr(self._model, "as_duplex"):
            self._model = self._model.as_duplex()

        self.ref_audio_path = self._resolve_ref_audio(pretrained, ref_audio)
        self.batch_size_per_gpu = int(batch_size)

    @staticmethod
    def _resolve_ref_audio(pretrained: str, ref_audio: Optional[str]) -> str:
        if ref_audio:
            return ref_audio
        # Local checkpoint directories carry the reference clip in assets/.
        local_candidate = os.path.join(pretrained, DEFAULT_REF_AUDIO_REPO_FILE)
        if os.path.exists(local_candidate):
            return local_candidate
        try:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(repo_id=pretrained, filename=DEFAULT_REF_AUDIO_REPO_FILE)
        except Exception as e:
            raise RuntimeError(
                "MiniCPM-o-4.5 duplex mode needs a reference voice clip. Pass ref_audio=<path> "
                f"or make sure `{DEFAULT_REF_AUDIO_REPO_FILE}` from the {pretrained!r} repo is reachable "
                f"(hf_hub_download failed: {e})."
            ) from e

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for MiniCPM_O_45Duplex")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for MiniCPM_O_45Duplex")

    # ------------------------------------------------------------------
    # Duplex session drivers
    # ------------------------------------------------------------------

    def _load_ref_audio(self) -> np.ndarray:
        cached = getattr(self, "_ref_audio_cache", None)
        if cached is not None:
            return cached
        import librosa

        ref, _ = librosa.load(self.ref_audio_path, sr=SPEECH_SAMPLE_RATE, mono=True)
        self._ref_audio_cache = np.asarray(ref, dtype=np.float32)
        return self._ref_audio_cache

    @staticmethod
    def _split_audio_chunks(audio: np.ndarray) -> List[np.ndarray]:
        """Split a waveform into the official 1-second duplex chunks."""
        audio = np.asarray(audio, dtype=np.float32)
        chunk_size = SPEECH_SAMPLE_RATE
        num_chunks = int(np.ceil(len(audio) / chunk_size))
        chunks = []
        for i in range(num_chunks):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            chunks.append(chunk)
        return chunks

    def _run_stream(self, stream: Dict[str, Any], ref_audio: np.ndarray, gen_kwargs: dict) -> List[Dict[str, Any]]:
        """Prefill+generate over one media stream inside the current session."""
        records: List[Dict[str, Any]] = []

        if stream["kind"] == "video":
            get_video_frame_audio_segments = _load_minicpmo_utils()
            video_frames, audio_segments, stacked_frames = get_video_frame_audio_segments(
                stream["path"], stack_frames=self.stack_frames, use_ffmpeg=self.use_ffmpeg, adjust_audio_length=True
            )
            num_chunks = len(video_frames) if video_frames is not None else 0
            for chunk_idx in range(num_chunks):
                audio_chunk = audio_segments[chunk_idx] if audio_segments is not None and chunk_idx < len(audio_segments) else None
                if audio_chunk is not None and not getattr(audio_chunk, "size", 1):
                    audio_chunk = None
                frame_list = []
                if video_frames[chunk_idx] is not None:
                    frame_list.append(video_frames[chunk_idx])
                if stacked_frames is not None and chunk_idx < len(stacked_frames) and stacked_frames[chunk_idx] is not None:
                    frame_list.append(stacked_frames[chunk_idx])
                record = self._step(audio_chunk, frame_list)
                record["chunk_idx"] = chunk_idx
                records.append(record)
        else:
            for chunk_idx, audio_chunk in enumerate(self._split_audio_chunks(stream["audio"])):
                record = self._step(audio_chunk, [])
                record["chunk_idx"] = chunk_idx
                records.append(record)
        return records

    def _step(self, audio_chunk: Optional[np.ndarray], frame_list: List) -> Dict[str, Any]:
        """One streaming prefill + generate step (one duplex chunk)."""
        self.model.streaming_prefill(
            audio_waveform=audio_chunk,
            frame_list=frame_list,
            max_slice_nums=self.max_slice_nums,
        )
        result = self.model.streaming_generate(
            prompt_wav_path=self.ref_audio_path,
            max_new_speak_tokens_per_chunk=int(self.max_new_speak_tokens_per_chunk),
            decode_mode=self.decode_mode,
        )
        record = {
            "text": result.get("text", "") or "",
            "is_listen": bool(result.get("is_listen", False)),
            "end_of_turn": bool(result.get("end_of_turn", False)),
            "current_time": result.get("current_time"),
        }
        waveform = result.get("audio_waveform")
        if waveform is not None:
            if hasattr(waveform, "cpu"):
                waveform = waveform.detach().cpu().numpy()
            record["audio_len"] = int(np.asarray(waveform).size)
        return record

    def _run_session(self, streams: List[Dict[str, Any]], gen_kwargs: dict) -> List[Dict[str, Any]]:
        ref_audio = self._load_ref_audio()
        self.model.prepare(
            prefix_system_prompt=self.system_prompt,
            ref_audio=ref_audio,
            prompt_wav_path=self.ref_audio_path,
        )
        records: List[Dict[str, Any]] = []
        preamble = 0.0
        for stream in streams:
            if stream["kind"] == "audio" and not records:
                # user speech prefilled before the live stream shifts the
                # session clock; report it so tasks can map chunk times back
                # onto the media timeline
                preamble += len(stream["audio"]) / SPEECH_SAMPLE_RATE
            records.extend(self._run_stream(stream, ref_audio, gen_kwargs))
        self._last_preamble = preamble
        return records

    # ------------------------------------------------------------------
    # lmms-eval entry point
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_media(visual) -> List[Dict[str, Any]]:
        """Turn one doc_to_visual entry into ordered duplex streams."""
        items = visual if isinstance(visual, (list, tuple)) else [visual]
        streams = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, str) and item.lower().endswith(VIDEO_EXTENSIONS):
                streams.append({"kind": "video", "path": item})
            elif isinstance(item, str) and item.lower().endswith(AUDIO_EXTENSIONS):
                import librosa

                audio, _ = librosa.load(item, sr=SPEECH_SAMPLE_RATE, mono=True)
                streams.append({"kind": "audio", "audio": np.asarray(audio, dtype=np.float32)})
            elif isinstance(item, dict) and "array" in item:
                from lmms_eval.models.simple.minicpm_o_45 import MiniCPM_O_45

                audio = MiniCPM_O_45._resample_audio(np.asarray(item["array"], dtype=np.float32), int(item.get("sampling_rate", SPEECH_SAMPLE_RATE)))
                streams.append({"kind": "audio", "audio": audio})
            elif isinstance(item, dict) and (item.get("path") or item.get("bytes")):
                # parquet media column: {"bytes": ..., "path": ...}
                path_hint = str(item.get("path") or "")
                data = item.get("bytes") or b""
                if path_hint.lower().endswith(VIDEO_EXTENSIONS) or (not path_hint and data[:4] == b"\x00\x00\x00 "):
                    import tempfile

                    suffix = os.path.splitext(path_hint)[1] or ".mp4"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                        handle.write(bytes(data))
                        streams.append({"kind": "video", "path": handle.name})
                else:
                    eval_logger.warning("minicpm_o_45_duplex: non-video parquet media dict without decode support")
            else:
                eval_logger.warning(f"minicpm_o_45_duplex: unsupported visual item {type(item)}")
        return streams

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return x[0], x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding (duplex)")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            # Keep each doc's media list intact: a doc may carry several media
            # files (e.g. Full-Duplex-Bench's clean + overlapped conversations).
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]

            gen_kwargs = all_gen_kwargs[0]

            for i, context in enumerate(contexts):
                visual = visuals[i] if i < len(visuals) else None
                streams = self._classify_media(visual)
                multi_session = bool(gen_kwargs.get("duplex_multi_session", False))
                try:
                    if multi_session:
                        sessions = [self._run_session([stream], gen_kwargs) for stream in streams]
                        preamble = 0.0
                    elif streams:
                        # user speech first, then the live video streams
                        sessions = [self._run_session(streams, gen_kwargs)]
                        preamble = getattr(self, "_last_preamble", 0.0)
                    else:
                        sessions = [[]]
                        preamble = 0.0
                    answer = json.dumps({"sessions": sessions, "preamble": preamble}, ensure_ascii=False)
                except Exception as e:
                    eval_logger.error(f"Error in duplex generation: {e}")
                    answer = json.dumps({"sessions": [[]], "preamble": 0.0, "error": str(e)}, ensure_ascii=False)

                if self.duplex_save_dir:
                    self._save_artifacts(self.duplex_save_dir, task, split, doc_id[i], answer)
                res.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def _save_artifacts(self, save_dir: str, task: str, split: str, doc_id: Any, answer: str) -> None:
        try:
            out_dir = os.path.join(save_dir, str(task))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{split}_{doc_id}.json")
            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(answer)
        except Exception as e:
            eval_logger.warning(f"Failed to save duplex artifacts for doc {doc_id}: {e}")

