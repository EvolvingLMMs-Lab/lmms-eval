"""LLaVA-OneVision2 inference wrapper (trust_remote_code).

Registered as ``llava_onevision2_chat``. Targets the released checkpoint
`lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
<https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct>`_,
whose bundled remote code (``modeling_llava_onevision2.py``,
``processing_llava_onevision2.py``, ``video_processing_llava_onevision2.py``)
implements ``patch_positions``, RoPE block layout, and the video
preprocessing pipeline (frame sampling, smart_resize, timestamp
precision, decord + torchvision BICUBIC resize, per-frame text
expansion) used during training.

The wrapper:
  - Pre-fetches frames via ``qwen_vl_utils.fetch_video`` (kwargs: fps /
    min_pixels / max_pixels / max_frames).
  - Builds a per-frame chat content list of ``<t seconds>`` text +
    ``image`` PIL pairs.
  - Feeds the resulting PIL list via ``images=...`` (NOT ``videos=...``)
    so the model takes the same image-processor branch used at training
    time.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

fetch_video, _ = optional_import("qwen_vl_utils", "fetch_video")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@register_model("llava_onevision2_chat")
class Llava_OneVision2(lmms):
    """Trust-remote-code wrapper for LLaVA-OneVision2.

    See https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct.
    """

    is_simple = False

    def __init__(
        self,
        pretrained: str = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
        device: str = "cuda",
        device_map: str = "",
        batch_size: int = 1,
        use_cache: bool = True,
        attn_implementation: str = "flash_attention_2",
        # Vision sampling controls (forwarded to processor on each call)
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        fixed_num_frames: int | None = None,
        target_fps: float | None = None,
        # Generation controls
        max_new_tokens: int = 128,
        system_prompt: str = "You are a helpful assistant.",
        torch_dtype: str = "bfloat16",
        # Video frame sampling (used when pre-fetching frames via
        # qwen_vl_utils.fetch_video to mirror V2 chat path).
        fps: float = 1.0,
        messages_format: str = "timestamp",
        timestamp_decimals: int = 1,
        # Codec sub-mode (in-processor preprocessing via cv-preinfer).
        # When ``use_codec=True``, video items are forwarded to the bundled
        # ``processor(video_backend='codec', ...)`` entry shipped with the
        # checkpoint; the wrapper itself contains no codec-specific logic.
        # ``max_pixels`` doubles as the codec canvas pixel budget so users
        # configure a single knob.
        use_codec: bool | str = False,
        codec_target_canvas: int | str | None = None,
        codec_group_size: int | str | None = None,
        codec_images_per_group: int | str | None = None,
        codec_patch: int | str | None = None,
        codec_min_group_frames: int | str | None = None,
        codec_max_group_frames: int | str | None = None,
        codec_spatial_mask_mode: str | None = None,
        codec_cache_dir: str | None = None,
        **kwargs,
    ):
        super().__init__()

        # --- Distributed setup (mirror the existing chat wrapper) --------
        accelerator = Accelerator()
        if accelerator.num_processes > 1 and not device_map:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # --- Load model + processor via trust_remote_code -----------------
        self.pretrained = pretrained
        eval_logger.info(f"[llava_onevision2_chat] Loading from: {pretrained}")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype_t = dtype_map.get(torch_dtype, torch.bfloat16)

        cfg = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        load_kwargs = dict(
            config=cfg,
            torch_dtype=torch_dtype_t,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        # If device_map is a multi-GPU spec (e.g. "auto", "balanced", or a dict),
        # let accelerate shard the model across GPUs and skip the explicit
        # `.to(device)` afterwards (which would undo the sharding / OOM).
        use_sharded = isinstance(self.device_map, dict) or (isinstance(self.device_map, str) and self.device_map in {"auto", "balanced", "balanced_low_0", "sequential"})
        if use_sharded:
            load_kwargs["device_map"] = self.device_map
        self.model = AutoModelForImageTextToText.from_pretrained(
            pretrained,
            **load_kwargs,
        )
        if not use_sharded:
            self.model.to(self._device)
        self.model.eval()
        # Pick the device where inputs should be sent (= where the embedding
        # layer ended up after sharding).
        if use_sharded and hasattr(self.model, "hf_device_map"):
            dmap = self.model.hf_device_map
            embed_key = next(
                (k for k in dmap if "embed_tokens" in k or k.endswith(".embed")),
                None,
            )
            input_dev = dmap.get(embed_key, None) if embed_key else None
            if input_dev is None:
                # Fallback: use first device in the map.
                input_dev = next(iter(dmap.values()))
            self._input_device = torch.device(f"cuda:{input_dev}" if isinstance(input_dev, int) else input_dev)
        else:
            self._input_device = self._device

        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        # Override pixel budget on the bundled VideoProcessor + ImageProcessor so
        # CLI args min_pixels/max_pixels actually take effect. The processor's
        # __call__ does NOT accept these per-call, so we mutate the underlying
        # processor instances once at init.
        try:
            vp = self.processor.video_processor
            if vp is not None:
                vp.min_pixels = int(min_pixels)
                vp.max_pixels = int(max_pixels)
        except AttributeError:
            pass
        try:
            ip = self.processor.image_processor
            if ip is not None:
                ip.min_pixels = int(min_pixels)
                ip.max_pixels = int(max_pixels)
                # Qwen2VLImageProcessor also reads `size`{shortest_edge,longest_edge}.
                if hasattr(ip, "size") and isinstance(ip.size, dict):
                    ip.size = dict(ip.size)
                    ip.size["shortest_edge"] = int(min_pixels)
                    ip.size["longest_edge"] = int(max_pixels)
        except AttributeError:
            pass

        # The bundled trust_remote modules in the checkpoint directory
        # provide the canonical preprocessing pipeline; see
        # ``video_processing_llava_onevision2.py`` and
        # ``processing_llava_onevision2.py`` shipped with the model.

        # Tokenizer convenience handle.
        try:
            self.tokenizer = self.processor.tokenizer
        except AttributeError:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)

        # --- Save sampling / generation knobs -----------------------------
        self.batch_size = int(batch_size)
        self.use_cache = bool(use_cache)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.max_num_frames = int(max_num_frames)
        self.fixed_num_frames = int(fixed_num_frames) if fixed_num_frames else None
        self.target_fps = float(target_fps) if target_fps else None
        self.max_new_tokens = int(max_new_tokens)
        self.system_prompt = system_prompt
        self.fps = float(fps)
        self.messages_format = str(messages_format)
        self.timestamp_decimals = int(timestamp_decimals)

        # --- Codec sub-mode (in-processor) -------------------------------
        # The wrapper holds no codec logic. We only collect user overrides
        # for ``codec_config`` and the optional cache dir, then forward both
        # to the bundled ``processor(video_backend='codec', ...)`` at call
        # time. Defaults come from preprocessor_config.json (model repo).
        self.use_codec = _as_bool(use_codec)
        codec_overrides: dict = {}
        for k, v in (
            ("target_canvas", codec_target_canvas),
            ("group_size", codec_group_size),
            ("images_per_group", codec_images_per_group),
            ("patch", codec_patch),
            ("min_group_frames", codec_min_group_frames),
            ("max_group_frames", codec_max_group_frames),
            ("spatial_mask_mode", codec_spatial_mask_mode),
        ):
            if v is not None and v != "":
                codec_overrides[k] = int(v) if k != "spatial_mask_mode" else str(v)
        if codec_cache_dir or os.getenv("ONLINE_CODEC_CACHE_DIR"):
            codec_overrides["cache_root"] = Path(codec_cache_dir or os.getenv("ONLINE_CODEC_CACHE_DIR"))
        self.codec_overrides = codec_overrides
        if self.use_codec:
            eval_logger.info(f"[codec] pixel budget unified: max_pixels={max_pixels}; " f"codec_overrides={codec_overrides}")
        # Per-build_messages scratch for codec video URLs.
        self._current_codec_video_urls: list = []

        # --- Distributed bookkeeping --------------------------------------
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in (
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ), "Unsupported distributed type for this wrapper."
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    # ------------------------------------------------------------------ utils

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def device(self):
        return self._device

    # ---------------------------------------------------- message building

    # ----------------------------- standard video timestamp path --------

    def _process_video_with_timestamp(self, video_url: str):
        """Decode a video and build a per-frame ``[timestamp, image, ...]`` content list.

        Frames are fetched via ``qwen_vl_utils.fetch_video`` and converted to
        PIL images; the caller passes the resulting list via ``images=...`` to
        the processor (the image-processor branch the model was trained on).

        When ``use_codec=True``, this short-circuits and emits a single
        ``{"type": "video", "video": url}`` content item; the bundled
        ``processor(video_backend='codec', ...)`` entry does the codec
        preprocessing + text rewrite downstream.
        """
        if self.use_codec:
            self._current_codec_video_urls.append(video_url)
            return [{"type": "video", "video": video_url}], []

        assert fetch_video is not None, "qwen_vl_utils is required. Please install it via `pip install qwen-vl-utils`"

        video_request = {
            "type": "video",
            "video": video_url,
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels,
        }
        if self.fixed_num_frames:
            total_frames = self._get_video_total_frames(video_url)
            video_request["nframes"] = min(self.fixed_num_frames, total_frames)
        else:
            video_request["fps"] = self.fps
            video_request["max_frames"] = self.max_num_frames
        video_input, video_metadata = fetch_video(
            video_request,
            return_video_metadata=True,
        )
        frames = video_input  # tensor [T, 3, H, W] uint8
        indices = video_metadata["frames_indices"]
        fps = video_metadata.get("fps", 30.0)
        if not isinstance(indices, list):
            indices = indices.tolist()
        pil_frames = [Image.fromarray(f.permute(1, 2, 0).numpy().astype(np.uint8)) for f in frames]

        merge_size = 2
        if len(indices) % merge_size != 0:
            pad = merge_size - len(indices) % merge_size
            indices.extend(indices[-1] for _ in range(pad))
            pil_frames = list(pil_frames) + [pil_frames[-1]] * pad
        timestamps = [idx / fps for idx in indices]

        video_content: list[dict] = []
        pil_images: list = []
        for img, t in zip(pil_frames, timestamps):
            ts_text = f"<{t:.{self.timestamp_decimals}f} seconds>"
            video_content.append({"type": "text", "text": ts_text})
            video_content.append({"type": "image", "image": img})
            pil_images.append(img)
        return video_content, pil_images

    def _get_video_total_frames(self, video_url: str) -> int:
        cap = cv2.VideoCapture(video_url)
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        return max(1, total_frames)

    def _build_messages(self, chat_message: ChatMessages):
        """Convert lmms-eval ChatMessages -> HF chat-template list.

        Returns ``(messages, pil_images, codec_video_urls)``. ``codec_video_urls``
        is populated only when ``use_codec=True`` and the message contained a
        video item; otherwise empty.
        """
        self._current_codec_video_urls = []
        out_msgs = []
        all_pil_images: list = []
        if self.system_prompt:
            out_msgs.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        for message in chat_message.messages:
            content: list[dict] = []
            for c in message.content:
                if c.type == "text":
                    content.append({"type": "text", "text": c.text})
                elif c.type == "image":
                    content.append({"type": "image", "image": c.url})
                    all_pil_images.append(c.url)
                elif c.type == "video":
                    if "timestamp" in self.messages_format:
                        video_content, pil_images = self._process_video_with_timestamp(c.url)
                        content.extend(video_content)
                        all_pil_images.extend(pil_images)
                    else:
                        # Fall back to bundled video pipeline (non-timestamp)
                        content.append({"type": "video", "video": c.url})
            out_msgs.append({"role": message.role, "content": content})
        return out_msgs, all_pil_images, list(self._current_codec_video_urls)

    # -------------------------------------------------------- main loop

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []

        def _collate(x):
            return x[2], x[2]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        n_iter = (len(requests) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(total=n_iter, disable=(self.rank != 0), desc="Model Responding")

        e2e_latency = 0.0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, gen_kwargs_raw, doc_ids, tasks, splits = zip(*chunk)
            task = tasks[0]
            split = splits[0]

            # Build chat messages for each doc in chunk.
            hf_messages_batch = []
            pil_images_batch: List[list] = []
            codec_urls_batch: List[list] = []
            for did in doc_ids:
                raw = doc_to_messages[0](self.task_dict[task][split][did])
                cm = ChatMessages(**{"messages": raw})
                msgs, pil_imgs, codec_urls = self._build_messages(cm)
                hf_messages_batch.append(msgs)
                pil_images_batch.append(pil_imgs)
                codec_urls_batch.append(codec_urls)

            texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in hf_messages_batch]

            # Per-frame PIL images go via images=, NOT videos=, matching the
            # image-processor branch used during training.
            images_arg = None
            first_pil = pil_images_batch[0] if pil_images_batch else []
            first_codec_urls = codec_urls_batch[0] if codec_urls_batch else []
            if first_pil:
                images_arg = first_pil

            if self.use_codec and first_codec_urls:
                if len(texts) != 1:
                    raise ValueError("codec path currently expects batch_size=1")
                # Single entry point: defer all codec preprocessing to the
                # bundled ``processor(video_backend='codec')`` shipped with
                # the checkpoint (trust_remote_code).
                proc_kwargs = dict(
                    text=[texts[0]],
                    videos=[first_codec_urls[0]],
                    video_backend="codec",
                    max_pixels=self.max_pixels,
                    return_tensors="pt",
                    padding=True,
                )
                if self.codec_overrides:
                    proc_kwargs["codec_config"] = self.codec_overrides
                inputs = self.processor(**proc_kwargs)
            else:
                inputs = self.processor(
                    text=texts,
                    images=images_arg,
                    videos=None,
                    return_tensors="pt",
                    padding=True,
                )

            # Move to device. Under multi-GPU sharding, send inputs to the
            # device where the embedding layer lives.
            inputs = {k: (v.to(self._input_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

            # Build generation kwargs.
            user_gk = dict(gen_kwargs_raw[0] or {})
            max_new = int(user_gk.get("max_new_tokens", self.max_new_tokens))
            do_sample = bool(user_gk.get("temperature", 0) and user_gk["temperature"] > 0)
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

            gen_args = dict(inputs)
            gen_args.update(
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_id,
                max_new_tokens=max_new,
                num_beams=int(user_gk.get("num_beams", 1)),
                do_sample=do_sample,
                use_cache=self.use_cache,
            )
            if do_sample:
                gen_args["temperature"] = float(user_gk["temperature"])
                if user_gk.get("top_p"):
                    gen_args["top_p"] = float(user_gk["top_p"])

            start = time.time()
            with torch.inference_mode():
                cont = self.model.generate(**gen_args)
            e2e_latency += time.time() - start
            cont = cont[:, inputs["input_ids"].shape[-1] :]
            total_tokens += int(cont.shape[-1])

            text_outs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            for txt in text_outs:
                res.append(txt)
                self.cache_hook.add_partial("generate_until", (texts[0], user_gk), txt)
            pbar.update(1)

        res = re_ords.get_original(res)

        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=e2e_latency,
            avg_speed=total_tokens / e2e_latency if e2e_latency > 0 else 0,
            additional_metrics={"rank": self.rank},
        )
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError

    def loglikelihood(self, requests):
        raise NotImplementedError
