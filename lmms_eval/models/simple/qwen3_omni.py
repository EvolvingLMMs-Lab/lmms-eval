from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from moviepy import VideoFileClip
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.audio_processing import split_audio

try:
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
except ImportError:
    eval_logger.warning("Failed to import Qwen3OmniMoe classes; Please install transformers from source: pip install git+https://github.com/huggingface/transformers")

try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    eval_logger.warning("Failed to import qwen_omni_utils; Please install it via `pip install qwen-omni-utils`")


@register_model("qwen3_omni")
class Qwen3_Omni(lmms):
    """
    Qwen3-Omni-30B-A3B-Instruct
    https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = "flash_attention_2",
        max_num_frames: int = 128,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: str = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

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

        model_kwargs = {
            "torch_dtype": "auto",
            "device_map": self.device_map,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(pretrained)
        self.max_num_frames = max_num_frames
        self._tokenizer = self.processor.tokenizer

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.system_prompt = system_prompt

        if hasattr(self._model, "disable_talker"):
            self._model.disable_talker()

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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3_Omni")

    def flatten(self, input):
        new_list = []
        for i in input:
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
        """Decode an AudioDecoder object or audio dict to a standard dict format."""
        if isinstance(audio_obj, dict) and "array" in audio_obj and "sampling_rate" in audio_obj:
            return audio_obj

        type_name = type(audio_obj).__name__
        if type_name != "AudioDecoder":
            raise ValueError(f"Unknown audio type: {type(audio_obj)}")

        try:
            return self._decode_audio_decoder(audio_obj)
        except Exception as e:
            raise ValueError(f"Failed to decode AudioDecoder object: {e}") from e

    def _decode_audio_decoder(self, audio_obj) -> dict:
        """Extract audio array and sampling rate from AudioDecoder object."""
        if hasattr(audio_obj, "get_all_samples"):
            decoded_audio = audio_obj.get_all_samples()
            audio_array = self._extract_audio_array(decoded_audio)
            sampling_rate = self._extract_sampling_rate(decoded_audio, audio_obj)
            return {"array": audio_array, "sampling_rate": sampling_rate}

        if hasattr(audio_obj, "decode"):
            decoded_audio = audio_obj.decode()
            if isinstance(decoded_audio, dict):
                return decoded_audio
            if hasattr(decoded_audio, "array") and hasattr(decoded_audio, "sampling_rate"):
                return {
                    "array": decoded_audio.array,
                    "sampling_rate": decoded_audio.sampling_rate,
                }

        if hasattr(audio_obj, "__call__"):
            decoded_audio = audio_obj()
            if isinstance(decoded_audio, dict):
                return decoded_audio
            if hasattr(decoded_audio, "array") and hasattr(decoded_audio, "sampling_rate"):
                return {
                    "array": decoded_audio.array,
                    "sampling_rate": decoded_audio.sampling_rate,
                }

        if hasattr(audio_obj, "array") and hasattr(audio_obj, "sampling_rate"):
            return {"array": audio_obj.array, "sampling_rate": audio_obj.sampling_rate}

        raise ValueError("Could not decode AudioDecoder object")

    def _extract_audio_array(self, decoded_audio):
        """Extract audio array from decoded audio object."""
        if hasattr(decoded_audio, "samples"):
            audio_array = decoded_audio.samples
        elif hasattr(decoded_audio, "array"):
            audio_array = decoded_audio.array
        elif hasattr(decoded_audio, "data"):
            audio_array = decoded_audio.data
        else:
            audio_array = decoded_audio

        # Convert tensor to numpy if needed
        if hasattr(audio_array, "cpu") and hasattr(audio_array, "numpy"):
            return audio_array.cpu().numpy()
        if hasattr(audio_array, "detach"):
            return audio_array.detach().cpu().numpy()
        if type(audio_array).__name__ == "Tensor":
            try:
                return audio_array.cpu().numpy()
            except Exception:
                return np.array(audio_array)
        return audio_array

    def _extract_sampling_rate(self, decoded_audio, audio_obj) -> int:
        """Extract sampling rate from decoded audio or audio object."""
        if hasattr(decoded_audio, "sample_rate"):
            return decoded_audio.sample_rate
        if hasattr(decoded_audio, "sampling_rate"):
            return decoded_audio.sampling_rate
        if hasattr(audio_obj, "metadata") and audio_obj.metadata:
            if hasattr(audio_obj.metadata, "sample_rate"):
                return audio_obj.metadata.sample_rate
            if isinstance(audio_obj.metadata, dict) and "sample_rate" in audio_obj.metadata:
                return audio_obj.metadata["sample_rate"]
        if hasattr(audio_obj, "_desired_sample_rate") and audio_obj._desired_sample_rate:
            return audio_obj._desired_sample_rate
        return 16000  # default

    def _check_if_video_has_audio(self, video_path):
        clip = VideoFileClip(video_path)
        return clip.audio is not None

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        current_use_audio = False

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

            # Preserve grouping for mixed modalities (e.g., [audio, image])
            should_flatten = True
            if visuals and isinstance(visuals[0], (list, tuple)) and len(visuals[0]) > 1:
                first_visual = visuals[0]
                has_audio = any(isinstance(v, dict) or type(v).__name__ == "AudioDecoder" for v in first_visual)
                has_image = any(isinstance(v, Image.Image) for v in first_visual)
                has_video = any(isinstance(v, str) and v.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")) for v in first_visual)
                if sum([has_audio, has_image, has_video]) > 1:
                    should_flatten = False

            if should_flatten:
                visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            message = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            ]
            for i, context in enumerate(contexts):
                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                        current_use_audio = self._check_if_video_has_audio(visual)
                        message.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "video", "video": visual},
                                    {"type": "text", "text": context},
                                ],
                            }
                        )

                    elif isinstance(visual, Image.Image):
                        message.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": visual},
                                    {"type": "text", "text": context},
                                ],
                            }
                        )

                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                        single_message = {"role": "user", "content": []}
                        for v in visual:
                            single_message["content"].append({"type": "image", "image": v})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)

                    elif isinstance(visual, dict) or type(visual).__name__ == "AudioDecoder":
                        current_use_audio = True
                        audio_dict = self._decode_audio(visual)
                        audio = self.resample_audio(audio_dict["array"], audio_dict["sampling_rate"])
                        audio_splits = split_audio(audio, 4800000)
                        single_message = {"role": "user", "content": []}
                        for j in range(len(audio_splits)):
                            single_message["content"].append({"type": "audio", "audio": audio_splits[j]})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)

                    elif isinstance(visual, (list, tuple)) and len(visual) > 0 and all(isinstance(v, dict) or type(v).__name__ == "AudioDecoder" for v in visual):
                        current_use_audio = True
                        for j, v in enumerate(visual):
                            audio_dict = self._decode_audio(v)
                            audio = self.resample_audio(audio_dict["array"], audio_dict["sampling_rate"])
                            audio_splits = split_audio(audio, 4800000)
                            single_message = {"role": "user", "content": []}
                            for k in range(len(audio_splits)):
                                single_message["content"].append({"type": "audio", "audio": audio_splits[k]})
                            single_message["content"].append({"type": "text", "text": context})
                            message.append(single_message)

                    elif isinstance(visual, (list, tuple)) and len(visual) > 0:
                        single_message = {"role": "user", "content": []}
                        for v in visual:
                            if isinstance(v, Image.Image):
                                single_message["content"].append({"type": "image", "image": v})
                            elif isinstance(v, dict) or type(v).__name__ == "AudioDecoder":
                                current_use_audio = True
                                audio_dict = self._decode_audio(v)
                                audio = self.resample_audio(audio_dict["array"], audio_dict["sampling_rate"])
                                audio_splits = split_audio(audio, 4800000)
                                for audio_chunk in audio_splits:
                                    single_message["content"].append({"type": "audio", "audio": audio_chunk})
                            elif isinstance(v, str) and v.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                                current_use_audio = self._check_if_video_has_audio(v)
                                single_message["content"].append({"type": "video", "video": v})
                        single_message["content"].append({"type": "text", "text": context})
                        message.append(single_message)

                    else:
                        raise ValueError(f"Unknown visual type: {type(visual)}")
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})

            text = self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(message, use_audio_in_video=current_use_audio)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=current_use_audio,
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda").to(self.model.dtype)
            else:
                inputs = inputs.to(self.model.device).to(self.model.dtype)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            try:
                cont = self.model.generate(
                    **inputs,
                    return_audio=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"] if gen_kwargs["temperature"] > 0 else None,
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    use_audio_in_video=current_use_audio,
                    thinker_do_sample=False,
                )
                if isinstance(cont, tuple):
                    cont = cont[0]
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                answer = ""
                res.append(answer)
                pbar.update(1)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                continue

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
