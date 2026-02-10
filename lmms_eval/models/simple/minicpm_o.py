"""MiniCPM-o-2.6 model for lmms-eval.

MiniCPM-o 2.6 is a GPT-4o level MLLM for vision, speech, and multimodal live streaming.

https://huggingface.co/openbmb/MiniCPM-o-2_6
https://github.com/OpenBMB/MiniCPM-V

IMPORTANT: This model requires transformers==4.44.2 specifically. Other versions may
cause import errors due to changes in the transformers API.

Environment Setup:
    # Create a dedicated environment (recommended)
    conda create -n minicpm_o python=3.10 -y
    conda activate minicpm_o

    # Install specific transformers version
    pip install transformers==4.44.2

    # Install other dependencies
    pip install torch torchvision torchaudio
    pip install librosa soundfile decord moviepy
    pip install Pillow numpy accelerate

    # For best performance with audio
    pip install vector-quantize-pytorch vocos

Example Usage:
    python -m lmms_eval --model minicpm_o \\
        --model_args pretrained=openbmb/MiniCPM-o-2_6 \\
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
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    eval_logger.warning("Failed to import transformers; Please install transformers==4.44.2")

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None

try:
    from moviepy import VideoFileClip
except ImportError:
    VideoFileClip = None


MAX_NUM_FRAMES = 64


def encode_video(video_path: str, max_frames: int = MAX_NUM_FRAMES) -> List[Image.Image]:
    """Extract frames from video file."""
    if VideoReader is None:
        raise ImportError("decord is required for video processing. Install with: pip install decord")

    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_frames:
        frame_idx = uniform_sample(frame_idx, max_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    return frames


@register_model("minicpm_o")
class MiniCPM_O(lmms):
    """
    MiniCPM-o-2.6 model for multi-modal evaluation.

    https://huggingface.co/openbmb/MiniCPM-o-2_6
    """

    def __init__(
        self,
        pretrained: str = "openbmb/MiniCPM-o-2_6",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: str = "sdpa",
        init_vision: bool = True,
        init_audio: bool = True,
        init_tts: bool = False,
        max_num_frames: int = MAX_NUM_FRAMES,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.max_num_frames = max_num_frames
        self.system_prompt = system_prompt
        self.init_vision = init_vision
        self.init_audio = init_audio
        self.init_tts = init_tts

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

        # Load model with omni initialization
        self._model = AutoModel.from_pretrained(
            pretrained,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            init_vision=init_vision,
            init_audio=init_audio,
            init_tts=init_tts,
        )

        if self.device_map == "auto":
            self._model = self._model.eval()
        else:
            self._model = self._model.to(self._device).eval()

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            trust_remote_code=True,
        )

        # Initialize TTS if needed
        if init_tts and hasattr(self._model, "init_tts"):
            self._model.init_tts()
            if hasattr(self._model, "tts"):
                self._model.tts.float()

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
        raise NotImplementedError("Loglikelihood is not implemented for MiniCPM_O")

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
            return True
        try:
            clip = VideoFileClip(video_path)
            has_audio = clip.audio is not None
            clip.close()
            return has_audio
        except Exception:
            return True

    def _build_message_content(self, context: str, visual) -> list:
        """Build message content list for model.chat()."""
        content = []

        if visual is not None:
            if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                # Video file - extract frames
                try:
                    frames = encode_video(visual, self.max_num_frames)
                    content.extend(frames)
                except Exception as e:
                    eval_logger.warning(f"Failed to encode video: {e}")

            elif isinstance(visual, Image.Image):
                # Single image
                content.append(visual)

            elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                # Multiple images
                content.extend(visual)

            elif isinstance(visual, dict) or type(visual).__name__ == "AudioDecoder":
                # Audio
                audio_dict = self._decode_audio(visual)
                audio = self.resample_audio(audio_dict["array"], audio_dict["sampling_rate"])
                content.append(audio)

            elif isinstance(visual, (list, tuple)):
                # Mixed content
                for v in visual:
                    if isinstance(v, Image.Image):
                        content.append(v)
                    elif isinstance(v, dict) or type(v).__name__ == "AudioDecoder":
                        audio_dict = self._decode_audio(v)
                        audio = self.resample_audio(audio_dict["array"], audio_dict["sampling_rate"])
                        content.append(audio)
                    elif isinstance(v, str) and v.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                        try:
                            frames = encode_video(v, self.max_num_frames)
                            content.extend(frames)
                        except Exception as e:
                            eval_logger.warning(f"Failed to encode video: {e}")

        content.append(context)
        return content

    def _has_audio_content(self, visual) -> bool:
        """Check if visual contains audio content."""
        if visual is None:
            return False
        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            return self._check_if_video_has_audio(visual)
        if isinstance(visual, dict) or type(visual).__name__ == "AudioDecoder":
            return True
        if isinstance(visual, (list, tuple)):
            return any(isinstance(v, dict) or type(v).__name__ == "AudioDecoder" for v in visual)
        return False

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

            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type " f"Union[str,list] but got {type(until)}")

            for i, context in enumerate(contexts):
                visual = visuals[i] if i < len(visuals) else None

                try:
                    # Check if we have audio content
                    has_audio = self._has_audio_content(visual)

                    # Build message content
                    content = self._build_message_content(context, visual)
                    msgs = [{"role": "user", "content": content}]

                    # Set generation parameters
                    max_new_tokens = gen_kwargs.get("max_new_tokens", 512)
                    temperature = gen_kwargs.get("temperature", 0)
                    do_sample = temperature > 0

                    # Call model.chat()
                    chat_kwargs = {
                        "msgs": msgs,
                        "tokenizer": self.tokenizer,
                        "sampling": do_sample,
                        "max_new_tokens": max_new_tokens,
                    }

                    if temperature > 0:
                        chat_kwargs["temperature"] = temperature
                    if gen_kwargs.get("top_p"):
                        chat_kwargs["top_p"] = gen_kwargs["top_p"]
                    if gen_kwargs.get("num_beams"):
                        chat_kwargs["num_beams"] = gen_kwargs["num_beams"]

                    # Add omni-specific params if audio is present
                    if has_audio and self.init_audio:
                        chat_kwargs["omni_input"] = True

                    response = self.model.chat(**chat_kwargs)

                    # Handle different response formats
                    if isinstance(response, tuple):
                        answer = response[0] if response else ""
                    elif isinstance(response, dict):
                        answer = response.get("text", response.get("response", str(response)))
                    else:
                        answer = str(response) if response else ""

                except Exception as e:
                    eval_logger.error(f"Error in generating: {e}")
                    answer = ""

                res.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented")
