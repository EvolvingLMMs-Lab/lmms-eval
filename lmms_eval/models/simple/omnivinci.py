"""OmniVinci model for lmms-eval.

OmniVinci is an omni-modal LLM for joint understanding of vision, audio, and language.

https://huggingface.co/nvidia/omnivinci
https://github.com/NVlabs/OmniVinci

IMPORTANT: This model requires a specific environment setup. OmniVinci is built on
the VILA codebase and requires dependencies from the NVILA environment.

Environment Setup:
    # Clone the model and set up the environment
    huggingface-cli download nvidia/omnivinci --local-dir ./omnivinci --local-dir-use-symlinks False
    cd ./omnivinci
    bash ./environment_setup.sh omnivinci

    # Or install dependencies manually:
    pip install git+https://github.com/bfshi/scaling_on_scales  # s2wrapper
    pip install beartype kaldiio soundfile

    # System dependencies (via conda):
    conda install -c conda-forge libsndfile

Note: This model uses custom code from HuggingFace (trust_remote_code=True) which
may require specific transformers versions compatible with the model code.

Example Usage:
    python -m lmms_eval --model omnivinci \\
        --model_args pretrained=nvidia/omnivinci \\
        --tasks omnibench \\
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

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.audio_processing import split_audio

try:
    from transformers import AutoConfig, AutoModel, AutoProcessor
except ImportError:
    eval_logger.warning("Failed to import transformers; Please install transformers")

try:
    from moviepy import VideoFileClip
except ImportError:
    VideoFileClip = None


@register_model("omnivinci")
class OmniVinci(lmms):
    """
    OmniVinci model for multi-modal evaluation.

    https://huggingface.co/nvidia/omnivinci
    """

    def __init__(
        self,
        pretrained: str = "nvidia/omnivinci",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        num_video_frames: int = 128,
        load_audio_in_video: bool = True,
        audio_length: str = "max_3600",
        system_prompt: str = "You are a helpful assistant.",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.num_video_frames = num_video_frames
        self.load_audio_in_video = load_audio_in_video
        self.audio_length = audio_length
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

        # Load config and update settings
        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)

        # Load model
        self._model = AutoModel.from_pretrained(
            pretrained,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self.device_map,
        ).eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            trust_remote_code=True,
        )

        # Configure audio/video settings
        self._model.config.load_audio_in_video = self.load_audio_in_video
        if hasattr(self.processor, "config"):
            self.processor.config.load_audio_in_video = self.load_audio_in_video

        # Get generation config
        if hasattr(self._model, "default_generation_config"):
            self.generation_config = self._model.default_generation_config
        else:
            self.generation_config = None

        self._tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else None
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
        if self.tokenizer is not None:
            return self.tokenizer.eos_token_id
        return None

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
        raise NotImplementedError("Loglikelihood is not implemented for OmniVinci")

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

    def _decode_audio(self, audio_obj) -> dict:
        """Decode audio object to standard dict format."""
        if isinstance(audio_obj, dict) and "array" in audio_obj and "sampling_rate" in audio_obj:
            return audio_obj

        type_name = type(audio_obj).__name__
        if type_name == "AudioDecoder":
            try:
                if hasattr(audio_obj, "get_all_samples"):
                    decoded = audio_obj.get_all_samples()
                    audio_array = decoded.samples if hasattr(decoded, "samples") else decoded
                    if hasattr(audio_array, "cpu"):
                        audio_array = audio_array.cpu().numpy()
                    sr = decoded.sample_rate if hasattr(decoded, "sample_rate") else 16000
                    return {"array": audio_array, "sampling_rate": sr}
            except Exception as e:
                eval_logger.warning(f"Failed to decode AudioDecoder: {e}")

        raise ValueError(f"Unknown audio type: {type(audio_obj)}")

    def _check_if_video_has_audio(self, video_path):
        if VideoFileClip is None:
            return self.load_audio_in_video
        try:
            clip = VideoFileClip(video_path)
            has_audio = clip.audio is not None
            clip.close()
            return has_audio
        except Exception:
            return self.load_audio_in_video

    def _build_message(self, context: str, visual) -> list:
        """Build message in the format expected by OmniVinci."""
        message = [{"role": "system", "content": self.system_prompt}]

        user_content = []

        if visual is not None:
            if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                # Video file
                user_content.append({"type": "video", "video": visual})
            elif isinstance(visual, Image.Image):
                # Single image
                user_content.append({"type": "image", "image": visual})
            elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                # Multiple images
                for v in visual:
                    user_content.append({"type": "image", "image": v})
            elif isinstance(visual, dict) or type(visual).__name__ == "AudioDecoder":
                # Audio
                audio_dict = self._decode_audio(visual)
                audio = self.resample_audio(audio_dict["array"], audio_dict["sampling_rate"])
                audio_splits = split_audio(audio, 4800000)
                for audio_chunk in audio_splits:
                    user_content.append({"type": "audio", "audio": audio_chunk})
            elif isinstance(visual, (list, tuple)):
                # Mixed content
                for v in visual:
                    if isinstance(v, Image.Image):
                        user_content.append({"type": "image", "image": v})
                    elif isinstance(v, dict) or type(v).__name__ == "AudioDecoder":
                        audio_dict = self._decode_audio(v)
                        audio = self.resample_audio(audio_dict["array"], audio_dict["sampling_rate"])
                        audio_splits = split_audio(audio, 4800000)
                        for audio_chunk in audio_splits:
                            user_content.append({"type": "audio", "audio": audio_chunk})
                    elif isinstance(v, str) and v.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                        user_content.append({"type": "video", "video": v})

        user_content.append({"type": "text", "text": context})
        message.append({"role": "user", "content": user_content})

        return message

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            if self.tokenizer is not None:
                toks = self.tokenizer.encode(x[0])
                return -len(toks), x[0]
            return 0, x[0]

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

            until = []
            if self.eot_token_id is not None:
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
                    # Determine if we're using audio
                    use_audio = False
                    if visual is not None:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                            use_audio = self._check_if_video_has_audio(visual)
                        elif isinstance(visual, dict) or type(visual).__name__ == "AudioDecoder":
                            use_audio = True
                        elif isinstance(visual, (list, tuple)):
                            use_audio = any(
                                isinstance(v, dict) or type(v).__name__ == "AudioDecoder"
                                for v in visual
                            )

                    # Build message
                    message = self._build_message(context, visual)

                    # Process inputs
                    if hasattr(self.processor, "apply_chat_template"):
                        text = self.processor.apply_chat_template(
                            message, add_generation_prompt=True, tokenize=False
                        )
                        inputs = self.processor(
                            text=text,
                            return_tensors="pt",
                            padding=True,
                        )
                    else:
                        # Fallback for processors without chat template
                        inputs = self.processor(
                            message,
                            return_tensors="pt",
                        )

                    # Move inputs to device
                    if self.device_map == "auto":
                        inputs = inputs.to("cuda").to(self.model.dtype)
                    else:
                        inputs = inputs.to(self.model.device).to(self.model.dtype)

                    # Set generation parameters
                    max_new_tokens = gen_kwargs.get("max_new_tokens", 1024)
                    temperature = gen_kwargs.get("temperature", 0)
                    do_sample = temperature > 0

                    # Generate
                    gen_params = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": do_sample,
                        "use_cache": self.use_cache,
                    }

                    if temperature > 0:
                        gen_params["temperature"] = temperature
                        gen_params["top_p"] = gen_kwargs.get("top_p", None)

                    if self.eot_token_id is not None:
                        gen_params["eos_token_id"] = self.eot_token_id
                        gen_params["pad_token_id"] = self.tokenizer.pad_token_id

                    if self.generation_config is not None:
                        self.generation_config.update(**gen_params)
                        outputs = self.model.generate(**inputs, generation_config=self.generation_config)
                    else:
                        outputs = self.model.generate(**inputs, **gen_params)

                    # Decode output
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    if hasattr(inputs, "input_ids"):
                        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
                    else:
                        generated_ids = outputs[0]

                    if self.tokenizer is not None:
                        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    elif hasattr(self.processor, "decode"):
                        answer = self.processor.decode(generated_ids, skip_special_tokens=True)
                    else:
                        answer = str(generated_ids)

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
