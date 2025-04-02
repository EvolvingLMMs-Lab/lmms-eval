import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

import audioread
import av
import decord
import librosa
import numpy as np
import soundfile as sf
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

try:
    from qwen_omni_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_omni_utils; Please install it via `pip install qwen-omni-utils[decord]`")


@register_model("qwen2_5_omni")
class Qwen2_5_Omni(lmms):
    """
    Qwen2.5-Omni-7B
    "https://huggingface.co/Qwen/Qwen2.5-Omni-7B"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-Omni-7B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        max_num_frames: int = 768,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
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

        if use_flash_attention_2:
            self._model = Qwen2_5OmniModel.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen2_5OmniModel.from_pretrained(pretrained, torch_dtype="auto", device_map="auto").eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        self.max_num_frames = max_num_frames
        self._tokenizer = self.processor.tokenizer

        self._config = self.model.config
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

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def resample_audio(self, audio: np.ndarray, current_sample_rate: int):
        """
        Resample the audio to the target sample rate.
        """
        if current_sample_rate != 16000:  # The sample rate for Qwen2.5-Omni is 16kHz
            if isinstance(audio, np.ndarray):
                audio = librosa.resample(audio, orig_sr=current_sample_rate, target_sr=16000).astype(np.float32)
        return audio

    def _check_if_video_has_audio(self, video_path):
        container = av.open(video_path)
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            return False
        return True

    def lmms_eval_process_audio_info(self, conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
        """
        Lmms_eval function to process audio information from conversations.
        This function is adapted from the original Qwen2.5-Omni code.
        Original code can be found here:
        https://github.com/QwenLM/Qwen2.5-Omni/blob/main/qwen-omni-utils/src/qwen_omni_utils/v2_5/audio_process.py#L15
        """
        audios = []
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            for message in conversation:
                if not isinstance(message["content"], list):
                    continue
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        if "audio" in ele:
                            path = ele["audio"]
                            if isinstance(path, np.ndarray):
                                if path.ndim > 1:
                                    raise ValueError("Support only mono audio")
                                audios.append(path)
                            elif isinstance(path, str):
                                if path.startswith("http://") or path.startswith("https://"):
                                    audios.append(librosa.load(audioread.ffdec.FFmpegAudioFile(path), sr=16000)[0])
                                elif path.startswith("file://"):
                                    audios.append(librosa.load(path[len("file://") :], sr=16000)[0])
                                else:
                                    audios.append(librosa.load(path, sr=16000)[0])
                            else:
                                raise ValueError("Unsupported type for audio: {}".format(type(path)))
                        else:
                            raise ValueError("Unknown audio {}".format(ele))
                    if use_audio_in_video and ele["type"] == "video":
                        if "video" in ele:
                            path = ele["video"]
                            assert self._check_if_video_has_audio(path), "Video must has audio track when use_audio_in_video=True"
                            if path.startswith("http://") or path.startswith("https://"):
                                audios.append(librosa.load(audioread.ffdec.FFmpegAudioFile(path), sr=16000)[0])
                            elif path.startswith("file://"):
                                audios.append(librosa.load(path[len("file://") :], sr=16000)[0])
                            else:
                                audios.append(librosa.load(path, sr=16000)[0])
                        else:
                            raise ValueError("Unknown video {}".format(ele))
        if len(audios) == 0:
            audios = None
        return audios

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        current_use_video = False  # Flag to check whether we are using video or not

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            # if isinstance(contexts, tuple):
            #     contexts = list(contexts)

            # for i in range(len(contexts)):
            #     for j in range(32):
            #         if f"<image {j}>" in contexts[i]:
            #             contexts[i] = contexts[i].replace(f"<image {j}>", "<image>")
            #         if f"\\<image {j}\\>" in contexts[i]:
            #             contexts[i] = contexts[i].replace(f"\\<image {j}\\>", "<image>")
            # if "<image>" in contexts[i]:
            #     contexts[i] = contexts[i].replace("<image>", "")
            # print(contexts[i])

            # for i in range(len(contexts)):
            #     if "<image>" in contexts[i]:
            #         contexts[i] = contexts[i].replace("<image>", "")
            # print(contexts)
            # print(visuals)
            audio_paths = []  # This will be deprecated in future when Qwen2.5 Omni supports loading numpy array audio directly
            message = [{"role": "system", "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
            for i, context in enumerate(contexts):
                if len(visuals) > 0:
                    visual = visuals[i] if i < len(visuals) else None
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                        current_use_video = True
                        if self.use_custom_video_loader:
                            visual = read_video_pyav_base64(visual, num_frm=self.max_num_frames, fps=self.fps, img_format="JPEG", max_image_size=self.max_image_size)
                            image_contents = list(map(lambda x: f"data:image/jpeg;base64,{x}", visual))
                            message.append({"role": "user", "content": [{"type": "video", "video": image_contents}, {"type": "text", "text": context}]})
                        else:  # Model video loader
                            message.append({"role": "user", "content": [{"type": "video", "video": visual}, {"type": "text", "text": context}]})

                    elif isinstance(visual, Image.Image):  # Single image
                        message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})

                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
                        for v in visual:
                            message.append({"role": "user", "content": [{"type": "image", "image": v}, {"type": "text", "text": context}]})

                    # elif isinstance(visual, dict):  # Single audio
                    #     audio = self.resample_audio(visual["array"], visual["sampling_rate"])
                    #     # Writing temp audio - this will be deprecated in future when Qwen2.5 Omni supports numpy array
                    #     sf.write(f"./audio_{self.rank}.wav", audio, 16000)
                    #     audio_path = f"./audio_{self.rank}.wav"
                    #     audio_paths.append(audio_path)
                    #     message.append({"role": "user", "content": [{"type": "audio", "audio": audio_path}, {"type": "text", "text": context}]})
                    # elif isinstance(visual, (list, tuple)) and all(isinstance(v, np.ndarray) for v in visual):  # Multiple audios
                    #     for i, v in enumerate(visual):
                    #         audio = self.resample_audio(v["array"], v["sampling_rate"])
                    #         # Writing temp audio - this will be deprecated in future when Qwen2.5 Omni supports numpy array
                    #         sf.write(f"./audio_{self.rank}_{i}.wav", audio, 16000)
                    #         audio_path = f"./audio_{self.rank}_{i}.wav"
                    #         audio_paths.append(audio_path)
                    #         message.append({"role": "user", "content": [{"type": "audio", "audio": audio_path}, {"type": "text", "text": context}]})

                    # Fixed code for audio messages
                    elif isinstance(visual, dict):  # Single audio
                        audio = self.resample_audio(visual["array"], visual["sampling_rate"])
                        message.append({"role": "user", "content": [{"type": "audio", "audio": audio}, {"type": "text", "text": context}]})
                    elif isinstance(visual, (list, tuple)) and all(isinstance(v, dict) for v in visual):  # Multiple audios
                        for i, v in enumerate(visual):
                            audio = self.resample_audio(v["array"], v["sampling_rate"])
                            message.append({"role": "user", "content": [{"type": "audio", "audio": audio}, {"type": "text", "text": context}]})

                    else:
                        raise ValueError(f"Unknown visual type: {type(visual)}")

            text = self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            # audios, images, videos = process_mm_info(message, use_audio_in_video=current_use_video)
            audios = self.lmms_eval_process_audio_info(message, use_audio_in_video=current_use_video)
            images, videos = process_vision_info(message, return_video_kwargs=False)

            inputs = self.processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)

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
                cont, _ = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    use_audio_in_video=current_use_video,
                )
                # the second return in this function is for audio, I assume we don't need it
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                answer = ""
                res.append(answer)
                pbar.update(1)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                continue

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                answers[i] = ans
            content = []
            for ans, context in zip(answers, contexts):
                res.append(ans)
                content.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            # answer = self.processor.batch_decode(cont, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
            #     eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{answer}\n")
            # parts = answer[0].split("\nassistant")
            # if len(parts) > 1:
            #     # This will give the answer after the first occurrence of "\nassistant"
            #     text_after = parts[1].strip()
            #     print("Model answer: ", text_after)

            # res.append(text_after)
            # self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_after)
            # pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
