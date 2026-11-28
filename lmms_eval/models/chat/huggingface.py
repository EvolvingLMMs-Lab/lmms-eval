import math
import time
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.processor_kwargs import audio_processor_kwargs, media_processor_kwargs
from lmms_eval.protocol import ChatMessages

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


def _processor_video(part: dict[str, Any], files: ExitStack) -> Any:
    path = part["video"]
    start, end = part.get("video_start"), part.get("video_end")
    if start is None and end is None:
        return path
    if any(not math.isfinite(t) for t in (start, end) if t is not None) or (start is not None and end is not None and start > end):
        raise ValueError(f"Invalid video interval [{start}, {end}]")

    import av

    # Processors differ in how they sample files, and some cannot mix paths and arrays.
    # Keep the normal file-input path (including fps metadata) even in a mixed batch.
    # The clip also carries the audio that the selected frames present: a frame-only
    # clip would hand bounded requests a different medium than the declared one.
    clip = Path(files.enter_context(TemporaryDirectory())) / "clip.mkv"
    with av.open(path) as source, av.open(str(clip), "w") as dest:
        if not source.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = source.streams.video[0]
        fps = stream.average_rate or stream.guessed_rate
        origin = stream.start_time
        if origin is None:
            # No declared start: the first frame defines the origin of the media axis.
            with av.open(path) as probe:
                first = next(probe.decode(video=0), None)
            if first is None or first.pts is None:
                raise ValueError(f"Missing video timestamp in {path}")
            origin = first.pts
        elif start is not None and start > 0:
            source.seek(origin + int(start / stream.time_base), stream=stream)
        # Audio is cut to the requested [start, end) on the same media axis the video
        # selection uses (its zero is the video origin), never past the declared end:
        # reading the last frame's display duration would pull audio from outside the
        # window into the evaluation, and re-anchoring each stream to its own first
        # sample would drop the file's real A/V offset (e.g. late audio tracks).
        zero_seconds = float(origin * stream.time_base)
        cuts = []
        for index in range(len(source.streams.audio)):
            cut = _clip_audio(path, index, start, end, zero_seconds)
            if cut is not None:
                cuts.append(cut)
        output = dest.add_stream("ffv1", rate=fps)
        output.width, output.height = stream.width, stream.height
        output.pix_fmt = "bgr0"
        output.codec_context.time_base = stream.time_base
        # Every output stream must exist before the first packet is muxed: the
        # container header is written on that packet, and a stream added later
        # corrupts the trailer (a native matroska assertion in practice). The audio
        # is therefore cut first, so video packets go to disk as they are encoded
        # instead of being held in memory for the whole window.
        outputs = []
        for rate, layout, clip_pts_start, buf in cuts:
            out = dest.add_stream("pcm_s16le", rate=rate)
            out.layout = layout
            outputs.append((out, rate, layout, clip_pts_start, buf))
        first_pts = None
        for frame in source.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Missing video timestamp in {path}")
            seconds = float((frame.pts - origin) * stream.time_base)
            if end is not None and seconds > end:
                break
            if start is not None and seconds < start:
                continue
            if first_pts is None:
                first_pts = frame.pts
            # Use the requested clip origin for both streams. The first retained
            # frame can legitimately occur after the first retained audio sample.
            frame.pts -= origin + round((start or 0.0) / stream.time_base)
            dest.mux(output.encode(frame))
        if first_pts is None:
            raise ValueError(f"No video frames in interval [{start}, {end}] for {path}")
        dest.mux(output.encode())
        for out, rate, layout, clip_pts_start, buf in outputs:
            channels = len(layout.channels)
            total = buf.shape[1] // channels
            for offset in range(0, total, rate):
                piece = buf[:, offset * channels : min(offset + rate, total) * channels]
                frame = av.AudioFrame.from_ndarray(piece, format="s16", layout=layout)
                frame.sample_rate = rate
                frame.pts = clip_pts_start + offset
                frame.time_base = Fraction(1, rate)
                dest.mux(out.encode(frame))
            dest.mux(out.encode())
    return str(clip)


def _clip_audio(path: str, index: int, start: Optional[float], end: Optional[float], zero_seconds: float) -> Optional[tuple[int, Any, int, Any]]:
    """Return audio stream ``index`` trimmed to ``[start, end)`` on the common media axis.

    ``zero_seconds`` is the media-time zero of the video selection, so audio and video
    are measured against one clock and a late audio track keeps its real A/V offset:
    the returned clip pts start is that offset in the stream's own sample clock (0 when
    the audio reaches the window start, positive when it only arrives later). Samples
    are cut exactly at the stream's rate; a stream with nothing inside the window yields
    ``None`` instead of an empty track.
    """
    import av
    import numpy as np

    with av.open(path) as source:
        astream = source.streams.audio[index]
        rate = int(astream.rate)
        channels = len(astream.layout.channels)
        resampler = av.AudioResampler(format="s16", layout=astream.layout, rate=astream.rate)
        # Audio decoding is cheap enough to read from the stream head and filter by
        # timestamp, which stays correct even where a seek would land mid-packet.
        first_sample = None if start is None else math.ceil(start * rate - 1e-9)
        last_sample = None if end is None else math.ceil(end * rate - 1e-9)
        # Container timestamps can be coarser than a sample.
        tolerance = max(1, math.ceil(astream.time_base * rate))
        kept: List[Any] = []
        cursor = None
        for frame in source.decode(astream):
            if frame.pts is None:
                raise ValueError(f"Missing audio timestamp in {path}")
            media_sample = round((frame.pts * astream.time_base - zero_seconds) * rate)
            if media_sample + frame.samples <= (0 if first_sample is None else first_sample):
                continue
            if last_sample is not None and media_sample >= last_sample:
                break
            # Frames within the timestamp rounding stay contiguous. A frame after a genuine
            # gap keeps its own position (the gap stays silent below) instead of being
            # pulled forward; an overlap has no faithful place on the timeline.
            jump = 0 if cursor is None else media_sample - cursor
            if jump < -tolerance:
                raise ValueError(f"Overlapping audio timestamps in {path}")
            if cursor is None or jump > tolerance:
                cursor = media_sample
            for piece in resampler.resample(frame):
                arr = piece.to_ndarray()
                kept.append((cursor, arr))
                cursor += arr.shape[1] // channels
        for piece in resampler.resample(None):
            if cursor is None:
                break
            arr = piece.to_ndarray()
            kept.append((cursor, arr))
            cursor += arr.shape[1] // channels
        if not kept:
            return None
        base = kept[0][0]
        buf = np.zeros((1, (kept[-1][0] - base) * channels + kept[-1][1].shape[1]), dtype=kept[0][1].dtype)
        for position, arr in kept:
            buf[:, (position - base) * channels : (position - base) * channels + arr.shape[1]] = arr
        lo = max(0, (0 if first_sample is None else first_sample - base) * channels)
        hi = buf.shape[1] if last_sample is None else min(buf.shape[1], max(0, (last_sample - base) * channels))
        if hi <= lo:
            return None
        clip_pts_start = base + lo // channels - (0 if first_sample is None else first_sample)
        return rate, astream.layout, clip_pts_start, buf[:, lo:hi]


def resolve_auto_model_class(config):
    """Return the Auto class that loads a generation-capable model for ``config``.

    The auto-model mappings in transformers are keyed by config class, not by ``model_type`` string, and
    ``trust_remote_code`` models declare their classes in ``config.auto_map`` instead.
    """
    auto_map = getattr(config, "auto_map", None) or {}
    # Image-text-to-text first: some multimodal configs (e.g. mllama) also map to a text-only causal LM class
    for auto_cls in (AutoModelForImageTextToText, AutoModelForCausalLM):
        if type(config) in auto_cls._model_mapping or auto_cls.__name__ in auto_map:
            return auto_cls
    return AutoModel


@register_model("huggingface")
class Huggingface(lmms):
    is_simple = False
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = None,
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self.trust_remote_code = trust_remote_code
        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        model_cls = resolve_auto_model_class(config)

        self._model = model_cls.from_pretrained(pretrained, trust_remote_code=trust_remote_code, **model_kwargs).eval()
        self.max_num_frames = max_num_frames

        raw_prompt = reasoning_prompt or system_prompt
        if raw_prompt:
            self.system_prompt = self._resolve_system_prompt(raw_prompt.replace("\\n", "\n"))
        else:
            self.system_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):  # noqa: A002 - Preserve the existing keyword argument name.
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[2], x[2]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [doc_to_messages[0](self.task_dict[task][split][ids]) for ids, task, split in zip(doc_id, task, split)]
            if self.system_prompt:
                chat_messages = [self._apply_system_prompt(messages, self.system_prompt) for messages in chat_messages]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            gen_kwargs = all_gen_kwargs[0]

            # Apply chat template
            batched_messages = [chat_message.to_hf_messages() for chat_message in chat_messages]
            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            images = []
            audios = []
            for messages in chat_messages:
                image, _, audio = messages.extract_media()
                images.extend(image)
                audios.extend(audio)
            # Decoder-only generation needs left padding when batching.
            padding_side = "left" if self.batch_size > 1 else "right"
            with ExitStack() as files:
                videos = [_processor_video(part, files) for conversation in batched_messages for message in conversation for part in message["content"] if part["type"] == "video"]
                kwargs = {**media_processor_kwargs(images, videos), **audio_processor_kwargs(self.processor, audios)}
                inputs = self.processor(text=texts, padding=True, padding_side=padding_side, return_tensors="pt", **kwargs)

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 4096,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Calculate timing metrics for batch
            total_elapsed_time += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for i, (ans, context) in enumerate(zip(answers, texts)):
                res.append(GenerationResult(text=ans, token_counts=TokenCounts(output_tokens=len(generated_ids_trimmed[i]))))
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Response: {ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)
        # Calculate average speed
        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        # Log metrics
        metric_dict = {
            "total_gen_tokens": total_tokens,
            "total_elapsed_time": total_elapsed_time,
            "avg_speed": avg_speed,
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
