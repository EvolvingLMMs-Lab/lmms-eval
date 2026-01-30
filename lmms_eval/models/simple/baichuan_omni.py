"""Baichuan-Omni-1.5 model for lmms-eval.

Baichuan-Omni-1.5 is an end-to-end trained omni-modal large model that supports
comprehensive input modalities (text, image, video, audio) and dual output
modalities (text and audio).

https://huggingface.co/baichuan-inc/Baichuan-Omni-1d5
https://github.com/baichuan-inc/Baichuan-Omni-1.5

Required Dependencies:
    pip install easydict fire ujson cairosvg imagesize decord deepspeed flash_attn

Example Usage:
    python -m lmms_eval --model baichuan_omni \\
        --model_args pretrained=baichuan-inc/Baichuan-Omni-1d5 \\
        --tasks omni_bench \\
        --batch_size 1
"""

import os
import tempfile
import traceback
import uuid
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torchaudio
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Patch for torchaudio compatibility (list_audio_backends removed in newer versions)
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile", "sox"]

try:
    from moviepy import VideoFileClip
except ImportError:
    VideoFileClip = None

try:
    import ujson
except ImportError:
    import json as ujson

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Role prefixes for Baichuan-Omni
ROLE_PREFIX = {
    "system": "<B_SYS>",
    "user": "<C_Q>",
    "assistant": "<C_A>",
}


@register_model("baichuan_omni")
class BaichuanOmni(lmms):
    """
    Baichuan-Omni-1.5 model for multi-modal evaluation.

    https://huggingface.co/baichuan-inc/Baichuan-Omni-1d5
    """

    def __init__(
        self,
        pretrained: str = "baichuan-inc/Baichuan-Omni-1d5",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        max_num_frames: int = 32,
        system_prompt: str = "You are a helpful assistant.",
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.max_num_frames = max_num_frames
        self.system_prompt = system_prompt

        # Create cache directory for media files
        if cache_dir is None:
            self.cache_dir = tempfile.mkdtemp(prefix="baichuan_omni_")
        else:
            self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "image"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "audio"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "video"), exist_ok=True)

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        # Load model with trust_remote_code for custom classes
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            trust_remote_code=True,
        )

        # Bind processor to model with relative_path for media handling
        self._model.bind_processor(
            self._tokenizer,
            training=False,
            relative_path=self.cache_dir,
        )

        # Get special tokens
        self.image_start_token = self._tokenizer.convert_ids_to_tokens(self._model.config.video_config.image_start_token_id)
        self.image_end_token = self._tokenizer.convert_ids_to_tokens(self._model.config.video_config.image_end_token_id)
        self.video_start_token = self._tokenizer.convert_ids_to_tokens(self._model.config.video_config.video_start_token_id)
        self.video_end_token = self._tokenizer.convert_ids_to_tokens(self._model.config.video_config.video_end_token_id)
        self.audio_start_token = self._tokenizer.convert_ids_to_tokens(self._model.config.audio_config.audio_start_token_id)
        self.audio_end_token = self._tokenizer.convert_ids_to_tokens(self._model.config.audio_config.audio_end_token_id)

        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type. Only DDP and FSDP are supported."
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
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
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
        raise NotImplementedError("Loglikelihood is not implemented for BaichuanOmni")

    def flatten(self, input_list):
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list

    def resample_audio(self, audio: np.ndarray, current_sample_rate: int) -> np.ndarray:
        """Resample audio to 16kHz and convert to mono if needed."""
        if not isinstance(audio, np.ndarray):
            return audio

        # Convert stereo to mono
        if audio.ndim == 2:
            axis = 0 if audio.shape[0] <= audio.shape[1] else 1
            audio = np.mean(audio, axis=axis)
        elif audio.ndim > 2:
            audio = audio.mean(axis=tuple(range(audio.ndim - 1)))

        audio = audio.astype(np.float32)

        # Resample to 16kHz if needed
        if current_sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=current_sample_rate, target_sr=16000)
            audio = audio.astype(np.float32)

        return audio

    def _check_if_video_has_audio(self, video_path):
        if VideoFileClip is None:
            return False
        try:
            clip = VideoFileClip(video_path)
            has_audio = clip.audio is not None
            clip.close()
            return has_audio
        except Exception:
            return False

    def _save_image(self, image: Image.Image) -> str:
        """Save image to cache directory and return path."""
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(self.cache_dir, "image", filename)
        image.save(filepath)
        return filepath

    def _save_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Save audio to cache directory and return path."""
        filename = f"{uuid.uuid4().hex}.wav"
        filepath = os.path.join(self.cache_dir, "audio", filename)
        # Convert to tensor for torchaudio
        if audio.ndim == 1:
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        else:
            audio_tensor = torch.from_numpy(audio).float()
        torchaudio.save(filepath, audio_tensor, sample_rate)
        return filepath

    def _build_message_content(self, context: str, visual) -> str:
        """Build message content with special tokens for media."""
        content_parts = []

        # Process visual inputs
        if visual is not None:
            if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                # Video file path
                content_parts.append(f"{self.video_start_token}" + ujson.dumps({"local": visual}, ensure_ascii=False) + f"{self.video_end_token}")
            elif isinstance(visual, Image.Image):
                # Single image
                img_path = self._save_image(visual)
                content_parts.append(f"{self.image_start_token}" + ujson.dumps({"local": img_path}, ensure_ascii=False) + f"{self.image_end_token}")
            elif isinstance(visual, (list, tuple)):
                for v in visual:
                    if isinstance(v, Image.Image):
                        img_path = self._save_image(v)
                        content_parts.append(f"{self.image_start_token}" + ujson.dumps({"local": img_path}, ensure_ascii=False) + f"{self.image_end_token}")
                    elif isinstance(v, dict) and "array" in v:
                        # Audio dict
                        audio = self.resample_audio(v["array"], v["sampling_rate"])
                        audio_path = self._save_audio(audio)
                        content_parts.append(f"{self.audio_start_token}" + ujson.dumps({"path": audio_path}, ensure_ascii=False) + f"{self.audio_end_token}")
                    elif isinstance(v, str) and v.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                        content_parts.append(f"{self.video_start_token}" + ujson.dumps({"local": v}, ensure_ascii=False) + f"{self.video_end_token}")
            elif isinstance(visual, dict) and "array" in visual:
                # Audio dict
                audio = self.resample_audio(visual["array"], visual["sampling_rate"])
                audio_path = self._save_audio(audio)
                content_parts.append(f"{self.audio_start_token}" + ujson.dumps({"path": audio_path}, ensure_ascii=False) + f"{self.audio_end_token}")

        # Add text
        content_parts.append(context)

        return "".join(content_parts)

    def _format_prompt(self, user_content: str) -> str:
        """Format the full prompt with role prefixes."""
        # System message + User message + Assistant prefix
        prompt = f"{ROLE_PREFIX['system']}{self.system_prompt}" f"{ROLE_PREFIX['user']}{user_content}" f"{ROLE_PREFIX['assistant']}"
        return prompt

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
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Handle until parameter
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            for i, context in enumerate(contexts):
                visual = visuals[i] if i < len(visuals) else None

                try:
                    # Build message content with media tags
                    user_content = self._build_message_content(context, visual)

                    # Format full prompt
                    prompt = self._format_prompt(user_content)

                    # Process with model's processor
                    inputs = self.model.processor([prompt])

                    # Prepare model inputs
                    input_ids = inputs.input_ids.cuda()
                    attention_mask = inputs.attention_mask.cuda() if inputs.attention_mask is not None else None

                    model_inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "tokenizer": self.tokenizer,
                    }

                    # Handle audios
                    if inputs.audios is not None:
                        model_inputs["audios"] = inputs.audios.cuda()
                    if inputs.encoder_length is not None:
                        model_inputs["encoder_length"] = inputs.encoder_length.cuda()
                    if inputs.bridge_length is not None:
                        model_inputs["bridge_length"] = inputs.bridge_length.cuda()

                    # Handle images
                    if inputs.images is not None:
                        model_inputs["images"] = [torch.tensor(img, dtype=torch.float32).cuda() for img in inputs.images]
                        if inputs.patch_nums is not None:
                            model_inputs["patch_nums"] = inputs.patch_nums
                        if inputs.images_grid is not None:
                            model_inputs["images_grid"] = inputs.images_grid

                    # Handle videos
                    if inputs.videos is not None:
                        model_inputs["videos"] = [torch.tensor(vid, dtype=torch.float32).cuda() for vid in inputs.videos]
                        if inputs.videos_patch_nums is not None:
                            model_inputs["videos_patch_nums"] = inputs.videos_patch_nums
                        if inputs.videos_grid is not None:
                            model_inputs["videos_grid"] = inputs.videos_grid

                    max_new_tokens = gen_kwargs.get("max_new_tokens", 1024)
                    temperature = gen_kwargs.get("temperature", 0.0)
                    do_sample = temperature > 0

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **model_inputs,
                            max_new_tokens=max_new_tokens,
                            stop_strings=["<|endoftext|>"],
                            temperature=temperature if do_sample else None,
                            do_sample=do_sample,
                            use_cache=self.use_cache,
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    # Decode output
                    if isinstance(outputs, tuple):
                        output_ids = outputs[0]
                    else:
                        output_ids = outputs

                    # Get only generated tokens
                    input_len = input_ids.shape[1]
                    generated_ids = output_ids[0, input_len:]
                    answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                except Exception as e:
                    eval_logger.error(f"Error in generating: {e}")
                    eval_logger.error(traceback.format_exc())
                    answer = ""

                res.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented")
