import os
import warnings
from typing import List, Optional, Tuple, Union

import librosa
import moviepy as mp
import numpy as np
import PIL
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.audio_processing import downsample_audio
from synvo_engine.models.kino import KinoForConditionalGeneration
from synvo_engine.models.kino.processing_kino import KinoProcessor

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_AUDIO_TOKEN = "<|AUDIO|>"


@register_model("kino")
class Kino(lmms):
    """
    Llava Model for Hugging Face Transformers: https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llava

    Adapted from the InstructBLIP model in lmms_eval/models/instructblip.py

    Example usage:

    accelerate launch --num_processes=8 --main_process_port 12345 -m lmms_eval \
        --model llava_hf \
        --model_args pretrained=llava-hf/llava-1.5-7b-hf \
        --tasks seedbench \
        --batch_size 1 \
        --output_path ./logs/ \
        --log_samples
    """

    def __init__(
        self,
        pretrained: str = "Evo-LMM/kino-7b-init",
        revision: str = "main",
        device: str = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        device_map: str = "",
        chat_template: Optional[str] = None,
        use_cache: bool = True,
        max_frames_num: Optional[int] = 32,
        pretrained_mlp_projector: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        video_max_pixels: Optional[int] = 360 * 420,
        fps: Optional[int] = 1,
        use_video_audio: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        self.max_frames_num = max_frames_num
        self._model = KinoForConditionalGeneration.from_pretrained(pretrained, revision=revision, torch_dtype=dtype, device_map=self.device_map, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation)
        if pretrained_mlp_projector:
            mm_projector_weights = torch.load(pretrained_mlp_projector, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self._model.multi_modal_projector.load_state_dict(get_w(mm_projector_weights, "multi_modal_projector"), strict=False)
            eval_logger.info(f"Loaded multi_modal_projector weights from {pretrained_mlp_projector}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self._model.audio_modal_projector.load_state_dict(get_w(mm_projector_weights, "audio_modal_projector"), strict=False)
            eval_logger.info(f"Loaded audio_modal_projector weights from {pretrained_mlp_projector}. Incompatible keys: {incompatible_keys}")

        self.pretrained = pretrained
        if self.model.config.vision_aspect_ratio == "navit":
            self._processor = KinoProcessor.from_pretrained("Evo-LMM/kino-maas-7B_v12_18000_init", revision=revision, trust_remote_code=trust_remote_code)
            if max_pixels:
                self._processor.image_processor.max_pixels = max_pixels
                self._processor.video_processor.max_pixels = max_pixels
            if min_pixels:
                self._processor.image_processor.min_pixels = min_pixels
                self._processor.video_processor.min_pixels = min_pixels
        else:
            self._processor = KinoProcessor.from_pretrained("Evo-LMM/kino-7b-init", revision=revision, trust_remote_code=trust_remote_code)
        # Pad from left for batched generation: https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llava#usage-tips
        self._processor.tokenizer.padding_side = "left"
        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.chat_template = chat_template
        self.use_cache = use_cache
        self.use_video_audio = use_video_audio
        self.fps = fps
        self.video_max_pixels = video_max_pixels
        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with pipeline parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1
        self.accelerator = accelerator

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
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    def extract_audio(self, videos_file_path):
        my_clip = mp.VideoFileClip(videos_file_path)
        return my_clip.audio

    def split_audio(self, audio_arrays):
        CHUNK_LIM = 480000
        SAMPLE_RATE = 16000
        audio_splits = []
        # Split the loaded audio to 30s chunks and extend the messages content
        for i in range(
            0,
            len(audio_arrays),
            CHUNK_LIM,
        ):
            audio_splits.append(audio_arrays[i : i + CHUNK_LIM])
        return audio_splits

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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TODO: Implement loglikelihood for Kino")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            messages = [{"role": "user", "content": []}]
            audios = []
            for visual in visuals:
                if isinstance(visual, str):
                    messages[0]["content"].append({"type": "video", "video": visual, "max_pixels": self.video_max_pixels, "fps": self.fps})
                    if self.use_video_audio:
                        video_audio = self.extract_audio(visual)
                        temp_audio_path = f"temp_video_audio_{self._rank}.wav"
                        video_audio.write_audiofile(temp_audio_path)
                        video_audio = librosa.load(temp_audio_path, sr=self._processor.audio_processor.sampling_rate)[0]
                        splited_video_audio = self.split_audio(video_audio)
                        audios.extend(splited_video_audio)
                        for _ in range(len(splited_video_audio)):
                            messages[0]["content"].append({"type": "audio", "audio_url": "<placeholder>"})
                        os.remove(temp_audio_path)
                elif isinstance(visual, PIL.Image.Image):
                    height = visual.size[0]
                    width = visual.size[1]
                    if width < 28 and height < 28:
                        visual = visual.resize((28, 28))
                    elif height < 28:
                        visual = visual.resize((28, width))
                    elif width < 28:
                        visual = visual.resize((height, 28))
                    # images.append(visual)
                    messages[0]["content"].append({"type": "image", "image": visual})
                elif isinstance(visual, dict) and "array" in visual:
                    splited_video_audio = self.split_audio(downsample_audio(visual["array"], visual["sampling_rate"], self._processor.audio_processor.sampling_rate))
                    audios.extend(splited_video_audio)
                    messages[0]["content"].append({"type": "audio", "audio_url": "<placeholder>"})
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            # Okay be I am assuming bs always == 1
            context = contexts[0]
            messages[0]["content"].append({"type": "text", "text": context})
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Prompt for doc ID {doc_id[0]}:\n\n{text}\n")

            if len(audios) == 0:
                audios = None

            inputs = self._processor(images=image_inputs, videos=video_inputs, audios=audios, text=text, sampling_rate=self._processor.audio_processor.sampling_rate, return_tensors="pt", **video_kwargs).to(self._device, self.model.dtype)
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            try:
                cont = self.model.generate(
                    **inputs,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    pad_token_id=self.eot_token_id,
                    eos_token_id=self.eot_token_id,
                )
                cont = cont[:, inputs["input_ids"].shape[-1] :]
            except Exception as e:
                eval_logger.error(f"Error {e} in generating")
                text_outputs = ""
                res.append(text_outputs)
                pbar.update(1)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
                continue
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                eval_logger.debug(f"Generated text for doc ID {doc_id[0]}:\n\n{text_outputs}\n")

            res.append(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAHF")
