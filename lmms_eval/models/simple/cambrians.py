import glob
import math
import os
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState

try:
    from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from cambrian.conversation import conv_templates
    from cambrian.model.builder import load_pretrained_model
    from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, expand2square
except ImportError:
    eval_logger.error("Cambrian is not installed. pip install git+https://github.com/cambrian-mllm/cambrian-s.git")

from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

def is_video_file(file_path: str) -> bool:
    if isinstance(file_path, Image.Image):
        return False
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions

def is_image_file(file_path: str) -> bool:
    if isinstance(file_path, Image.Image):
        return True
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions

from decord import VideoReader, cpu

def process_video_with_decord(video_file, model_cfg, num_threads=-1):

    if num_threads < 1:
        vr = VideoReader(video_file, ctx=cpu(0))
    else:
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=num_threads)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / model_cfg.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    frame_time = [i/avg_fps for i in frame_idx]

    if model_cfg.video_max_frames > 0:
        if len(frame_idx) > model_cfg.video_max_frames or model_cfg.video_force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, model_cfg.video_max_frames, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]

    video = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = num_frames = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample

def process_videos(videos, image_processor, model_cfg, num_threads=-1):

    processor_aux_list = image_processor

    new_videos_aux_list = []
    video_sizes = []

    for video in videos:
        video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video, model_cfg, num_threads=num_threads)
        video_sizes.append((video.shape[2], video.shape[1], video.shape[0])) # W, H, T
        video = [Image.fromarray(video[_], mode="RGB") for _ in range(video.shape[0])] # covert to PIL.Image.Image

        video_aux_list = []
        for processor_aux in processor_aux_list:
            video_aux = video
            video_aux = [expand2square(image, tuple(int(x*255) for x in processor_aux.image_mean)) for image in video_aux]
            video_aux_list.append(processor_aux.preprocess(video_aux, return_tensors='pt')['pixel_values'])

        new_videos_aux_list.append(video_aux_list)

    new_videos_aux_list = [list(batch_video_aux) for batch_video_aux in zip(*new_videos_aux_list)]
    new_videos_aux_list = [torch.stack(video_aux) for video_aux in new_videos_aux_list]

    return new_videos_aux_list, video_sizes, (video_time, frame_time, num_frames_to_sample)

@register_model("cambrians")
class CambrianS(lmms):

    def __init__(
        self,
        pretrained: str = "",
        torch_dtype: Optional[Union[str, torch.dtype]] = "float16",
        batch_size: Optional[Union[int, str]] = 1,
        device_map="cuda:0",
        conv_template="qwen_2",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        #############################
        video_max_frames: int = 32,
        video_fps: int = 1,
        video_force_sample: bool = False,
        add_time_instruction: bool = False,
        #############################
        miv_token_len: int = 196,
        si_token_len: int = 729,
        image_aspect_ratio: str = "anyres",
        anyres_max_subimages: int = 9,
        #############################
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and (device_map == "auto" or device_map == "balanced_low_0"):
            raise NotImplementedError("device_map == auto is not supported yet.")
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained)

        self.torch_dtype = torch_dtype
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, self.model_name, device_map=self.device_map)

        self._model.config.video_max_frames = video_max_frames
        self._model.config.video_fps = video_fps
        self._model.config.video_force_sample = video_force_sample
        self._model.config.add_time_instruction = add_time_instruction
        self._model.config.miv_token_len = miv_token_len
        self._model.config.si_token_len = si_token_len
        self._model.config.image_aspect_ratio = image_aspect_ratio
        self._model.config.anyres_max_subimages = anyres_max_subimages

        eval_logger.info(f"video_max_frames: {video_max_frames}")
        eval_logger.info(f"video_fps: {video_fps}")
        eval_logger.info(f"video_force_sample: {video_force_sample}")
        eval_logger.info(f"add_time_instruction: {add_time_instruction}")
        eval_logger.info(f"miv_token_len: {miv_token_len}")
        eval_logger.info(f"si_token_len: {si_token_len}")
        eval_logger.info(f"image_aspect_ratio: {image_aspect_ratio}")
        eval_logger.info(f"anyres_max_subimages: {anyres_max_subimages}")

        self._config = self._model.config

        self.model.eval()

        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
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
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
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
        raise NotImplementedError

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, requests, task_dict, tokenizer, image_processor, model_config, conv_template):
                self.requests = requests
                self.task_dict = task_dict
                self.tokenizer = tokenizer
                self.image_processor = image_processor
                self.model_config = model_config
                self.conv_template = conv_template

            def __len__(self):
                return len(self.requests)

            def __getitem__(self, idx):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = self.requests[idx].args
                visuals = doc_to_visual(self.task_dict[task][split][doc_id])

                if visuals is not None:
                    qs = contexts
                    try:
                        if len(visuals) == 1:
                            if is_image_file(visuals[0]):
                                visual_tensors, visual_sizes = process_images(visuals, self.image_processor, self.model_config)
                            elif is_video_file(visuals[0]):
                                num_threads = 1 if 'Ego4D' in visuals[0] or "video_mmmu" in visuals[0] else -1
                                visual_tensors, visual_sizes, (_, _, _) = process_videos(visuals, self.image_processor, self.model_config, num_threads=num_threads)
                            else:
                                raise NotImplementedError
                        elif len(visuals) > 1:
                            if all(is_image_file(_) for _ in visuals):
                                visual_tensors, visual_sizes = process_images(visuals, self.image_processor, self.model_config, use_pad=True)
                            elif sum(is_video_file(_) for _ in visuals) > 1:
                                raise NotImplementedError("Multiple videos are not supported yet.")
                            else:
                                visual_tensors = []
                                visual_sizes = []
                                for visual in visuals:
                                    if is_video_file(visual):
                                        num_threads = 1 if 'Ego4D' in visual else -1
                                        visual_tensor, visual_size, (_, _, _) = process_videos([visual], self.image_processor, self.model_config, num_threads=num_threads)
                                        visual_tensors.append(visual_tensor[0])
                                        visual_sizes.append(visual_size[0])
                                    elif is_image_file(visual):
                                        if isinstance(visual, str):
                                            visual = Image.open(visual).convert("RGB")
                                        visual_tensor, visual_size = process_images([visual], self.image_processor, self.model_config, use_pad=True)
                                        visual_tensors.append(visual_tensor[0])
                                        visual_sizes.append(visual_size[0])
                                    else:
                                        raise NotImplementedError
                        else:
                            raise NotImplementedError
                    except Exception as e:
                        raise e

                    if isinstance(qs, (list, tuple)):
                        real_qs = ""
                        for _, sub_qs in enumerate(qs):
                            if sub_qs:
                                real_qs += sub_qs
                            else:
                                if qs[_ + 1] != "" and qs[_ + 1][0] == "\n":
                                    real_qs += DEFAULT_IMAGE_TOKEN
                                else:
                                    real_qs += DEFAULT_IMAGE_TOKEN + "\n"
                        qs = real_qs
                    elif isinstance(qs, str):
                        if self.model_config.mm_use_im_start_end:
                            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                        else:
                            assert len(visual_tensors) == 1, "This should not happen."
                            qs = DEFAULT_IMAGE_TOKEN * len(visual_tensors) + "\n" + qs
                    else:
                        raise NotImplementedError

                else:
                    visual_tensors = None
                    visual_sizes = None
                    qs = contexts

                conv = conv_templates[self.conv_template].copy()

                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
                return input_ids, visual_tensors, visual_sizes, prompt, gen_kwargs

        dataset = Dataset(requests, self.task_dict, self.tokenizer, self._image_processor, self._config, self.conv_template)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=4, pin_memory=True)

        for _, (input_ids, visual_tensors, visual_sizes, cur_prompt, gen_kwargs) in enumerate(dataloader):

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 16
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            with torch.inference_mode():
                input_ids = input_ids.cuda()

                visual_tensors = [_.half().cuda() for _ in visual_tensors]

                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=visual_tensors,
                    image_sizes=visual_sizes,
                    use_cache=self.use_cache,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            eval_logger.debug(f"Question: {cur_prompt}")
            eval_logger.debug(f"Answer: {outputs}")
            res.append(outputs)
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError
