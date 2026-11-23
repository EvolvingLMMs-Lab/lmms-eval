"""MiniCPM-o-4.5 model for lmms-eval (half-duplex offline inference).

MiniCPM-o 4.5 is OpenBMB's 9B end-to-end omni-modal model (SigLip2 +
Whisper-medium + CosyVoice2 + Qwen3-8B). This wrapper targets the released
``openbmb/MiniCPM-o-4_5`` checkpoint and drives the *half-duplex* (simplex)
``model.chat`` path from the official model card:

    res = model.chat(
        msgs=msgs,
        max_new_tokens=...,
        do_sample=...,
        omni_mode=True,        # required whenever the input carries audio
        enable_thinking=False, # instruct mode
    )

https://huggingface.co/openbmb/MiniCPM-o-4_5
https://github.com/OpenBMB/MiniCPM-o

IMPORTANT: the released remote code is pinned to ``transformers==4.51.0``;
other versions may fail to import or misbehave. The wrapper warns when a
different version is installed.

Environment Setup:
    conda create -n minicpm_o_45 python=3.10 -y
    conda activate minicpm_o_45
    pip install "transformers==4.51.0" accelerate "torch>=2.3.0,<=2.8.0" "torchaudio<=2.8.0"
    pip install "minicpmo-utils>=1.0.5,<=1.0.6"   # get_video_frame_audio_segments
    pip install librosa soundfile decord Pillow numpy

Example Usage:
    python -m lmms_eval --model minicpm_o_45 \\
        --model_args pretrained=openbmb/MiniCPM-o-4_5 \\
        --tasks videomme \\
        --batch_size 1
"""

import os
from io import BytesIO
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    eval_logger.warning("Failed to import transformers; MiniCPM-o-4.5 requires transformers==4.51.0")

RECOMMENDED_TRANSFORMERS_VERSION = "4.51.0"

# Media roles follow the official chat-inference recipe: a video with audio is
# split into interleaved per-second [frame, audio] segments via
# minicpmo.utils.get_video_frame_audio_segments (stack_frames=1).
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".ts")


def _load_minicpmo_utils():
    """Return ``get_video_frame_audio_segments`` from the minicpmo-utils package."""
    try:
        from minicpmo.utils import get_video_frame_audio_segments
    except ImportError as e:
        raise ImportError(
            "MiniCPM-o-4.5 evaluation requires the `minicpmo-utils` package for omni "
            "video/audio segmentation. Install it with "
            '`pip install "minicpmo-utils>=1.0.5,<=1.0.6"`. See the model card of '
            "openbmb/MiniCPM-o-4_5 for the pinned versions."
        ) from e
    return get_video_frame_audio_segments


def _patch_transformers_tp_plan_bug() -> None:
    """Guard transformers 4.51's caching_allocator_warmup for remote-code models.

    When torch.distributed was initialized before ``from_pretrained`` (the
    lmms-eval CLI creates an ``Accelerator`` first under ``accelerate launch``),
    transformers 4.51 iterates ``model._tp_plan``, which is ``None`` for
    trust_remote_code classes like MiniCPMO, and crashes with
    ``TypeError: 'NoneType' object is not iterable``. Upstream guards this from
    transformers 4.52 on; mirror that fix so the pinned 4.51 release works.
    """
    try:
        import functools

        import transformers.modeling_utils as modeling_utils

        warmup = getattr(modeling_utils, "caching_allocator_warmup", None)
        if warmup is None or getattr(warmup, "_minicpmo_tp_plan_patched", False):
            return

        @functools.wraps(warmup)
        def _safe_warmup(model, device_map, factor=1.0, *args, **kwargs):
            if getattr(model, "_tp_plan", None) is None:
                try:
                    model._tp_plan = []
                except Exception:
                    pass
            return warmup(model, device_map, factor=factor, *args, **kwargs)

        _safe_warmup._minicpmo_tp_plan_patched = True
        modeling_utils.caching_allocator_warmup = _safe_warmup
    except Exception:
        pass


@register_model("minicpm_o_45")
class MiniCPM_O_45(lmms):
    """MiniCPM-o-4.5 half-duplex (simplex) offline evaluation.

    Runs single-turn ``model.chat`` calls over offline media: images, audio
    files, and videos. When audio is present (audio files or videos with an
    audio track, and ``init_audio=True``), the call is made with
    ``omni_mode=True`` per the official "Half-Duplex Omni Mode" recipe.
    """

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
        init_tts: bool = False,
        enable_thinking: bool = False,
        max_slice_nums: Optional[int] = None,
        stack_frames: int = 1,
        use_ffmpeg: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self._warn_transformers_version()
        _patch_transformers_tp_plan_bug()

        self.system_prompt = system_prompt
        self.init_vision = init_vision
        self.init_audio = init_audio
        self.init_tts = init_tts
        self.enable_thinking = enable_thinking
        self.max_slice_nums = max_slice_nums
        self.stack_frames = stack_frames
        self.use_ffmpeg = use_ffmpeg

        # Rank/device come from the launcher environment only. Creating an
        # `Accelerator()` here instead would initialize torch.distributed
        # before `from_pretrained`, and transformers 4.51's
        # `caching_allocator_warmup` then iterates `model._tp_plan`, which is
        # None for the custom MiniCPMO class. Data-parallel evaluation works
        # through request sharding; no DDP wrapping is needed.
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

        # TTS is only needed to *generate* speech; offline text benchmarks can
        # skip the extra weights unless explicitly requested.
        if init_tts and hasattr(self._model, "init_tts"):
            self._model.init_tts()
            if hasattr(self._model, "tts"):
                self._model.tts.float()

        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)

    @staticmethod
    def _warn_transformers_version() -> None:
        try:
            import transformers
            from packaging.version import Version
        except ImportError:
            return
        if Version(transformers.__version__) != Version(RECOMMENDED_TRANSFORMERS_VERSION):
            eval_logger.warning(f"MiniCPM-o-4.5 remote code is pinned to transformers=={RECOMMENDED_TRANSFORMERS_VERSION} (installed: {transformers.__version__}); import errors or wrong results are possible.")

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
        raise NotImplementedError("Loglikelihood is not implemented for MiniCPM_O_45")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for MiniCPM_O_45")

    # ------------------------------------------------------------------
    # Media helpers (shared with the chat / duplex wrappers)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_video(path) -> bool:
        return isinstance(path, str) and path.lower().endswith(VIDEO_EXTENSIONS)

    @staticmethod
    def _is_audio(path) -> bool:
        return isinstance(path, str) and path.lower().endswith(AUDIO_EXTENSIONS)

    def _load_audio(self, path: str) -> np.ndarray:
        """Load an audio file as 16 kHz mono float32 (official input format)."""
        import librosa

        wav, _ = librosa.load(path, sr=16000, mono=True)
        return np.asarray(wav, dtype=np.float32)

    def _audio_from_visual(self, visual):
        """Extract an audio array from datasets' Audio feature decodes."""
        if isinstance(visual, dict) and "array" in visual:
            audio = np.asarray(visual["array"], dtype=np.float32)
            sr = int(visual.get("sampling_rate", 16000))
            return self._resample_audio(audio, sr)
        if type(visual).__name__ == "AudioDecoder":
            decoded = visual.get_all_samples()
            audio = decoded.samples
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            sr = decoded.sample_rate if hasattr(decoded, "sample_rate") else 16000
            return self._resample_audio(np.asarray(audio, dtype=np.float32), int(sr))
        raise ValueError(f"Unsupported audio visual: {type(visual)}")

    @staticmethod
    def _resample_audio(audio: np.ndarray, current_sample_rate: int) -> np.ndarray:
        if not isinstance(audio, np.ndarray):
            return audio
        if audio.ndim == 2:
            axis = 0 if audio.shape[0] <= audio.shape[1] else 1
            audio = np.mean(audio, axis=axis)
        audio = audio.astype(np.float32)
        if current_sample_rate != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=current_sample_rate, target_sr=16000)
            audio = audio.astype(np.float32)
        return audio

    def _omni_video_content(self, video_path: str) -> List:
        """Interleave per-second [frame, audio] segments for one video.

        Mirrors the official chat-inference example: with ``init_audio`` the
        video is expanded into ``[frame_i, audio_i, (stacked_i)]*`` via
        ``get_video_frame_audio_segments`` so the model receives the audio
        track in omni mode; without audio only the frames are kept.
        """
        get_video_frame_audio_segments = _load_minicpmo_utils()
        video_frames, audio_segments, stacked_frames = get_video_frame_audio_segments(video_path, stack_frames=self.stack_frames, use_ffmpeg=self.use_ffmpeg, adjust_audio_length=True)
        content: List = []
        num_frames = len(video_frames) if video_frames is not None else 0
        num_audio = len(audio_segments) if audio_segments is not None else 0
        has_audio_track = any(seg is not None and getattr(seg, "size", 0) for seg in (audio_segments or []))
        if self.init_audio and not has_audio_track:
            # silent clip: interleave one second of zeros per frame so omni
            # mode still receives a strictly paired [frame, audio] stream
            for frame in list(video_frames or []):
                content.append(frame)
                content.append(np.zeros(16000, dtype=np.float32))
            return content
        # The extractor can produce an off-by-one between frames and audio
        # segments (e.g. 120 frames vs 119 one-second audio chunks); the omni
        # processor requires strictly paired [frame, audio] chunks, so pair by
        # the shorter side and drop the trailing unpaired items.
        if num_frames and num_audio:
            num_chunks = min(num_frames, num_audio)
        else:
            num_chunks = max(num_frames, num_audio)
        for i in range(num_chunks):
            if i < num_frames:
                content.append(video_frames[i])
            if self.init_audio and i < num_audio:
                segment = audio_segments[i]
                if segment is not None and getattr(segment, "size", 1):
                    content.append(np.asarray(segment, dtype=np.float32))
            if stacked_frames is not None and i < len(stacked_frames) and stacked_frames[i] is not None:
                content.append(stacked_frames[i])
        return content

    @staticmethod
    def _interleave_silence(content: List, num_chunks: int) -> List:
        frames = [item for item in content if isinstance(item, Image.Image)]
        merged: List = []
        for frame in frames[:num_chunks]:
            merged.append(frame)
            merged.append(np.zeros(16000, dtype=np.float32))
        return merged

    def _media_bytes_to_video(self, data, path_hint) -> List:
        """Materialize parquet video bytes and build omni content for them."""
        import tempfile

        data = bytes(data or b"")
        suffix = os.path.splitext(str(path_hint or ""))[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(data)
            temp_path = handle.name
        try:
            if self.init_audio:
                return self._omni_video_content(temp_path)
            return self._frames_only_content(temp_path)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _frames_only_content(self, video_path: str) -> List[Image.Image]:
        get_video_frame_audio_segments = _load_minicpmo_utils()
        video_frames, _, _ = get_video_frame_audio_segments(video_path, stack_frames=self.stack_frames, use_ffmpeg=self.use_ffmpeg, use_audio=False)
        return list(video_frames or [])

    def build_media_content(self, visual) -> Tuple[List, bool]:
        """Turn a doc_to_visual entry into (content list, has_audio)."""
        content: List = []
        has_audio = False

        if visual is None:
            return content, has_audio

        items = visual if isinstance(visual, (list, tuple)) else [visual]
        for item in items:
            if item is None:
                continue
            if isinstance(item, Image.Image):
                content.append(item.convert("RGB"))
            elif self._is_video(item):
                if self.init_audio:
                    content.extend(self._omni_video_content(item))
                    has_audio = True
                else:
                    content.extend(self._frames_only_content(item))
            elif self._is_audio(item):
                if self.init_audio:
                    content.append(self._load_audio(item))
                    has_audio = True
            elif isinstance(item, dict) and "array" in item:
                # datasets' decoded Audio feature
                if self.init_audio:
                    content.append(self._audio_from_visual(item))
                    has_audio = True
            elif type(item).__name__ == "AudioDecoder":
                if self.init_audio:
                    content.append(self._audio_from_visual(item))
                    has_audio = True
            elif isinstance(item, dict) and ("bytes" in item or item.get("path")):
                # parquet media columns: {"bytes": ..., "path": ...} —
                # dispatch by the referenced file extension when available,
                # otherwise treat as an image payload.
                path_hint = str(item.get("path") or "")
                data = item.get("bytes") or b""
                if self._is_video(path_hint):
                    content.extend(self._media_bytes_to_video(data, item.get("path")))
                elif data:
                    content.append(Image.open(BytesIO(bytes(data))).convert("RGB"))
            elif isinstance(item, str):
                content.append(Image.open(item).convert("RGB"))
            else:
                eval_logger.warning(f"MiniCPM_O_45: unsupported visual item {type(item)}")

        return content, has_audio

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _call_chat(self, content: List, context: str, gen_kwargs: dict, has_audio: bool) -> str:
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": [*content, context]})

        max_new_tokens = int(gen_kwargs.get("max_new_tokens", 512))
        temperature = float(gen_kwargs.get("temperature", 0) or 0)
        do_sample = temperature > 0
        if "do_sample" in gen_kwargs:
            do_sample = bool(gen_kwargs["do_sample"])

        chat_kwargs = {
            "msgs": msgs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "enable_thinking": self.enable_thinking,
        }
        if do_sample:
            chat_kwargs["temperature"] = temperature
            if gen_kwargs.get("top_p") is not None:
                chat_kwargs["top_p"] = float(gen_kwargs["top_p"])
        # The official omni chat recipe pins max_slice_nums=1 whenever the
        # input carries audio/video chunks (image slicing desynchronizes the
        # interleaved placeholder stream); pure-image prompts keep the model
        # default unless overridden.
        slice_nums = self.max_slice_nums
        if slice_nums is None and has_audio:
            slice_nums = 1
        if slice_nums is not None:
            chat_kwargs["max_slice_nums"] = slice_nums
        if has_audio and self.init_audio:
            # Required for omni inference whenever the input carries audio.
            chat_kwargs["omni_mode"] = True

        response = self.model.chat(**chat_kwargs)
        if isinstance(response, tuple):
            answer = response[0] if response else ""
        elif isinstance(response, dict):
            answer = response.get("text", response.get("response", str(response)))
        else:
            answer = str(response) if response is not None else ""
        return answer

    @staticmethod
    def _truncate_until(answer: str, until: List[str]) -> str:
        stop = min((answer.find(item) for item in until if item and item in answer), default=-1)
        return answer[:stop] if stop >= 0 else answer

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            # Keep each doc's media list intact (multi-media docs are legal for
            # e.g. omni_duplexeval's [question_audio, video] ordering).
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]

            gen_kwargs = all_gen_kwargs[0]
            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            for i, context in enumerate(contexts):
                visual = visuals[i] if i < len(visuals) else None
                try:
                    content, has_audio = self.build_media_content(visual)
                    answer = self._call_chat(content, context, gen_kwargs, has_audio)
                    answer = self._truncate_until(answer, until)
                except Exception as e:
                    eval_logger.error(f"Error in generating: {e}")
                    answer = ""

                res.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
