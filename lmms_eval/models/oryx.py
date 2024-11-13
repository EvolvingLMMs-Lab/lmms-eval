import copy
import logging
import math
from datetime import timedelta
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav

eval_logger = logging.getLogger("lmms-eval")
import os
import sys

try:
    from oryx.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from oryx.conversation import SeparatorStyle, conv_templates
    from oryx.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_anyres_highres_image_genli,
        process_anyres_video_genli,
        tokenizer_image_token,
    )
    from oryx.model.builder import load_pretrained_model
except ImportError:
    eval_logger.debug("Oryx is not installed. Please install Oryx to use this model.")

try:
    from oryx.model.language_model.oryx_qwen import OryxQwenConfig

    AutoConfig.register("oryx_qwen", OryxQwenConfig)
except:
    eval_logger.debug("")


@register_model("oryx")
class Oryx(lmms):
    def __init__(
        self,
        pretrained: str = "",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation=(
            "sdpa" if torch.__version__ >= "2.1.2" else "eager"
        ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map="",
        conv_template="qwen_1_5",
        use_cache=True,
        truncate_context=False,
        max_frames_num: int = 32,
        mm_resampler_type: str = "spatial_pool",
        overwrite: bool = True,
        video_decode_backend: str = "decord",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained)
        self.video_decode_backend = video_decode_backend
        # self._config = AutoConfig.from_pretrained(self.pretrained)
        self.overwrite = overwrite
        self.mm_resampler_type = mm_resampler_type
        self.max_frames_num = int(max_frames_num)
        if self.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_resampler_type"] = self.mm_resampler_type
            overwrite_config["patchify_video_feature"] = False
            overwrite_config["attn_implementation"] = attn_implementation

            cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, self.model_name, device_map=self.device_map, overwrite_config=overwrite_config)
        else:
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                pretrained,
                None,
                self.model_name,
                device_map=self.device_map,
            )

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        if accelerator.num_processes > 1:
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
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
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
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

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

    def load_video(self, video_path, max_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())

        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        modality = "video"

        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames, modality  # (frames, height, width, channels)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            # video
            if type(visuals[0][0]) == str:
                for visual in visuals:
                    video = self.load_video(visual, self.max_frames_num)
                    video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].bfloat16().to(self.device)
                    videos.append(video)
                task_type = "video"
            # image
            else:
                for visual in visuals:
                    image_tensor_, image_highres_tensor_ = process_anyres_highres_image_genli(visual, self._image_processor)
                    image_tensor.append(image_tensor_)
                    image_highres_tensor.append(image_highres_tensor_)
                if all(x.shape == image_tensor[0].shape for x in image_tensor):
                    image_tensor = torch.stack(image_tensor, dim=0)
                if all(x.shape == image_highres_tensor[0].shape for x in image_highres_tensor):
                    image_highres_tensor = torch.stack(image_highres_tensor, dim=0)
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)
                if type(image_highres_tensor) is list:
                    image_highres_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_highres_tensor]
                else:
                    image_highres_tensor = image_highres_tensor.to(dtype=torch.bfloat16, device=self.device)

                image_sizes = [visuals[idx].size for idx in range(len(visuals))]
                task_type = "image"

            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], continuation)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100

            with torch.inference_mode():
                if task_type == "video":
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        modalities=["video"],
                        images=videos,
                        images_highres=videos,
                    )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        modalities=["image"] * len(image_sizes),
                        images=image_tensor,
                        images_highres=image_highres_tensor,
                        image_sizes=image_sizes,
                    )

            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            modalities = []
            try:
                if task == "mvbench_episodic_reasoning":
                    sampled_frm = min(len(visuals), self.max_frames_num)
                    indices = np.linspace(0, len(visuals) - 1, sampled_frm, dtype=int)
                    frames = [visuals[i] for i in indices]
                    video = np.stack([np.array(x) for x in frames])
                    modality = "video"
                    frames = []
                    for frame in video:
                        self._image_processor.do_resize = False
                        self._image_processor.do_center_crop = False
                        frames.append(process_anyres_video_genli(Image.fromarray(frame).convert("RGB"), self._image_processor))
                    video = torch.stack(frames, dim=0).bfloat16().to(self.device)
                    videos.append(video)
                    modalities.append(modality)
                else:
                    if type(visuals[0][0]) == str:
                        for visual in visuals:
                            if self.video_decode_backend == "decord":
                                video, modality = self.load_video(visual, self.max_frames_num)
                            elif self.video_decode_backend == "pyav":
                                video, modality = read_video_pyav(visual, num_frm=self.max_frames_num)
                            # video = self.load_video(visual, self.max_frames_num)
                            frames = []
                            for frame in video:
                                self._image_processor.do_resize = False
                                self._image_processor.do_center_crop = False
                                frames.append(process_anyres_video_genli(Image.fromarray(frame).convert("RGB"), self._image_processor))
                            video = torch.stack(frames, dim=0).bfloat16().to(self.device)
                            videos.append(video)
                            modalities.append(modality)
                        task_type = "video"
                    else:
                        self._image_processor.do_resize = False
                        self._image_processor.do_center_crop = False
                        image_tensor, image_highres_tensor = [], []
                        for visual in visuals:
                            image_tensor_, image_highres_tensor_ = process_anyres_highres_image_genli(visual, self._image_processor)
                            image_tensor.append(image_tensor_)
                            image_highres_tensor.append(image_highres_tensor_)
                        if all(x.shape == image_tensor[0].shape for x in image_tensor):
                            image_tensor = torch.stack(image_tensor, dim=0)
                        if all(x.shape == image_highres_tensor[0].shape for x in image_highres_tensor):
                            image_highres_tensor = torch.stack(image_highres_tensor, dim=0)
                        if type(image_tensor) is list:
                            image_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_tensor]
                        else:
                            image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)
                        if type(image_highres_tensor) is list:
                            image_highres_tensor = [_image.to(dtype=torch.bfloat16, device=self.device) for _image in image_highres_tensor]
                        else:
                            image_highres_tensor = image_highres_tensor.to(dtype=torch.bfloat16, device=self.device)
                        task_type = "image"

            except Exception as e:
                eval_logger.info(f"{e}")
                eval_logger.info(f"Video {visuals} can not load, check the source")
                video_path = "\n".join(visuals)
                res.append(f"Video {video_path} can not load, check the source")
                pbar.update(1)
                continue

            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_template].copy()

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            attention_masks = input_ids.ne(pad_token_ids).long().to(self.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            cur_prompt = contexts
            if task_type == "image":
                gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.2
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            try:
                with torch.inference_mode():
                    if task_type == "video":
                        output_ids = self.model.generate(
                            inputs=input_ids,
                            images=videos,
                            images_highres=videos,
                            attention_mask=attention_masks,
                            modalities=modalities,
                            use_cache=self.use_cache,
                            stopping_criteria=[stopping_criteria],
                            do_sample=True if gen_kwargs["temperature"] > 0 else False,
                            temperature=gen_kwargs["temperature"],
                            top_p=gen_kwargs["top_p"],
                            num_beams=gen_kwargs["num_beams"],
                            max_new_tokens=gen_kwargs["max_new_tokens"],
                        )
                    else:
                        output_ids = self.model.generate(
                            input_ids,
                            attention_mask=attention_masks,
                            pad_token_id=pad_token_ids,
                            modalities=["image"] * len(gen_kwargs["image_sizes"]),
                            images=image_tensor,
                            images_highres=image_highres_tensor,
                            image_sizes=gen_kwargs["image_sizes"],
                            do_sample=True if gen_kwargs["temperature"] > 0 else False,
                            temperature=gen_kwargs["temperature"],
                            top_p=gen_kwargs["top_p"],
                            num_beams=gen_kwargs["num_beams"],
                            max_new_tokens=gen_kwargs["max_new_tokens"],
                            use_cache=self.use_cache,
                        )
                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                # print(outputs)
                res.append(outputs)
                pbar.update(1)
            except Exception as e:
                eval_logger.info(f"{e}")
                eval_logger.info(f"Video {visuals} generate failed, check the source")
                video_path = "\n".join(visuals)
                res.append(f"Video {video_path} generate failed, check the source")
                pbar.update(1)
                continue
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
