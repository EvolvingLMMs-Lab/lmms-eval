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

from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from moviepy import VideoFileClip
except ImportError:
    VideoFileClip = None

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.audio_processing import split_audio


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
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.max_num_frames = max_num_frames
        self.system_prompt = system_prompt

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

        # Bind processor to model
        self.processor = self._model.bind_processor(
            self._tokenizer,
            training=False,
        )

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
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
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

    def _build_prompt_text(
        self,
        context: str,
        visual,
        image_tag: str = "<image_start_baichuan>{}<image_end_baichuan>",
        audio_tag: str = "<audio_start_baichuan>{}<audio_end_baichuan>",
        video_tag: str = "<video_start_baichuan>{}<video_end_baichuan>",
    ) -> str:
        """Build the prompt text with appropriate modality tags."""
        prefix_parts = []

        if visual is not None:
            if isinstance(visual, str) and visual.endswith(
                (".mp4", ".avi", ".mov", ".mkv", ".webm")
            ):
                prefix_parts.append(video_tag.format('{"path": "' + visual + '"}'))
            elif isinstance(visual, Image.Image):
                prefix_parts.append(image_tag.format('{"local": "image"}'))
            elif isinstance(visual, (list, tuple)):
                for v in visual:
                    if isinstance(v, Image.Image):
                        prefix_parts.append(image_tag.format('{"local": "image"}'))
                    elif isinstance(v, dict):
                        prefix_parts.append(audio_tag.format('{"path": "audio"}'))
                    elif isinstance(v, str) and v.endswith(
                        (".mp4", ".avi", ".mov", ".mkv", ".webm")
                    ):
                        prefix_parts.append(video_tag.format('{"path": "' + v + '"}'))
            elif isinstance(visual, dict):
                prefix_parts.append(audio_tag.format('{"path": "audio"}'))

        prefix = "\n".join(prefix_parts)
        if prefix:
            return f"{prefix}\n{context}"
        return context

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type "
                        f"Union[str,list] but got {type(until)}"
                    )

            for i, context in enumerate(contexts):
                visual = visuals[i] if i < len(visuals) else None

                try:
                    # Build prompt with modality tags
                    prompt_text = self._build_prompt_text(context, visual)

                    # Prepare inputs using processor
                    inputs = self.processor(prompt_text)

                    # Convert to tensors and move to device
                    input_ids = torch.tensor(inputs.input_ids).unsqueeze(0)
                    if self.device_map == "auto":
                        input_ids = input_ids.to("cuda")
                    else:
                        input_ids = input_ids.to(self.model.device)

                    # Prepare additional inputs
                    model_inputs = {
                        "input_ids": input_ids,
                        "audios": (
                            torch.tensor(inputs.audios).to(self.model.device)
                            if inputs.audios is not None
                            else None
                        ),
                        "encoder_length": (
                            torch.tensor(inputs.encoder_length).to(self.model.device)
                            if inputs.encoder_length is not None
                            else None
                        ),
                        "bridge_length": (
                            torch.tensor(inputs.bridge_length).to(self.model.device)
                            if inputs.bridge_length is not None
                            else None
                        ),
                        "images": (
                            [
                                torch.tensor(img).to(self.model.device)
                                for img in inputs.images
                            ]
                            if inputs.images is not None
                            else None
                        ),
                        "patch_nums": inputs.patch_nums,
                        "images_grid": inputs.images_grid,
                        "videos": (
                            [
                                torch.tensor(vid).to(self.model.device)
                                for vid in inputs.videos
                            ]
                            if inputs.videos is not None
                            else None
                        ),
                        "videos_patch_nums": inputs.videos_patch_nums,
                        "videos_grid": inputs.videos_grid,
                    }

                    # Set generation parameters
                    max_new_tokens = gen_kwargs.get("max_new_tokens", 512)
                    temperature = gen_kwargs.get("temperature", 0)
                    do_sample = temperature > 0

                    # Generate
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **model_inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature if temperature > 0 else None,
                            do_sample=do_sample,
                            use_cache=self.use_cache,
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    # Decode output
                    generated_ids = outputs[0][input_ids.shape[-1] :]
                    answer = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )

                except Exception as e:
                    eval_logger.error(f"Error in generating: {e}")
                    answer = ""

                res.append(answer)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), answer
                )
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented")
