import copy
import json
import logging
import math
import os
import os.path as osp
import queue
import re
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import av
import einops
import numpy as np
import PIL
import torch
import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from huggingface_hub import snapshot_download
from moviepy.video.io.VideoFileClip import VideoFileClip
from packaging import version
from PIL import Image
from scipy.spatial.distance import cosine
from skimage import transform
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True

# Import LLaVA modules
try:
    from MovieChat.common.registry import registry
except ImportError as e:
    eval_logger.debug(
        f"MovieChat is not installed. First, install MovieChat by 'https://github.com/rese1f/MovieChat.git' and 'cd MovieChat'. Change the torch version with `python -m pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118`"
    )


# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


@register_model("moviechat")
class MovieChat(lmms):
    """
    MovieChat Model
    """

    def __init__(
        self,
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        pretrained_llama_model: str = "Enxin/MovieChat-vicuna",
        pretrained_llama_proj_model: str = "Enxin/MovieChat-proj",
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        short_memory_length: Optional[int] = 18,
        long_memory_length: Optional[int] = 256,
        sliding_window_length: Optional[int] = 8,
        merge_frame_length: Optional[int] = 2,
        tmp_folder: Optional[str] = "tmp/",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
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

        llama_model = snapshot_download(repo_id=pretrained_llama_model) if not osp.isdir(pretrained_llama_model) else pretrained_llama_model
        llama_proj_pth = snapshot_download(repo_id=pretrained_llama_proj_model) if not osp.isdir(pretrained_llama_proj_model) else pretrained_llama_proj_model
        llama_proj = osp.join(llama_proj_pth, "finetune-vicuna7b-v2.pth")
        model_config = {
            "arch": "moviechat",
            "model_type": "pretrain_vicuna",
            "freeze_vit": True,
            "freeze_qformer": True,
            "max_txt_len": 256,
            "end_sym": "###",
            "low_resource": False,
            "frozen_llama_proj": False,
            "llama_model": llama_model,
            "llama_proj_model": llama_proj,
        }

        model_cls = registry.get_model_class(model_config["arch"])
        self._model = model_cls.from_config(model_config).to(self.device_map)

        vis_processor_cfg = {
            "name": "alpro_video_eval",
            "n_frms": 8,
        }
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]  # Resize to 224x224  # Convert PIL Image to Tensor with shape [C, H, W]  # Normalize
        )
        self._image_processor = registry.get_processor_class(vis_processor_cfg["name"]).from_config(vis_processor_cfg)

        self.model.short_memory_length = short_memory_length
        self.model.long_memory_length = long_memory_length
        self.merge_frame_length = merge_frame_length
        self.sliding_window_length = sliding_window_length
        self.num_clips = (self.model.long_memory_length // self.merge_frame_length) * ((self.model.short_memory_length - self.merge_frame_length) // self.sliding_window_length)
        self.tmp_folder = tmp_folder

        self._tokenizer = self.model.llama_tokenizer
        stop_words_ids = [torch.tensor([835]).to(self.device), torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "MovieChat currently does not support batched generation."

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
            self._world_size = 1

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
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        raise NotImplementedError("MovieChat only supports generation.")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_context_emb(self, input_text, img_list):
        prompt_1 = "You are able to understand the visual content that the user provides.Follow the instructions carefully and explain your answers in details.###Human: <Video><ImageHere></Video>"
        prompt_2 = input_text
        prompt_3 = "###Assistant:"

        prompt = prompt_1 + " " + prompt_2 + prompt_3

        prompt_segs = prompt.split("<ImageHere>")
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def answer(self, img_list, input_text, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9, repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        embs = self.get_context_emb(input_text, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print("Warning: The number of tokens in current conversation exceeds the max length. " "The model will not see the contexts outside the range.")
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )

        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token  at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split("###")[0]  # remove the stop sign '###'
        output_text = output_text.split("Assistant:")[-1].strip()
        return output_text, output_token.cpu().numpy()

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
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            text_outputs = []

            for visual, context in zip(batched_visuals, batched_contexts):
                if type(visual[0]) == PIL.Image.Image and "task_type" not in metadata and "sample_frames" not in metadata:  # For image task
                    raise NotImplementedError("MovieChat only supports video inputs.")

                elif "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:
                    raise NotImplementedError("MovieChat only supports video inputs.")

                elif type(visual[0]) == str:  # For video task
                    image_tensor = []
                    self.model.short_memory_buffer = []
                    self.model.long_memory_buffer = []
                    img_list = []
                    # try:
                    os.makedirs(self.tmp_folder, exist_ok=True)

                    video = VideoFileClip(visual[0])
                    clip_duration = video.duration / self.num_clips

                    cur_frame = 0
                    for i in range(self.num_clips):
                        preprocess_frames = []
                        start_time = i * clip_duration
                        end_time = start_time + clip_duration
                        # uniformly sample self.sliding_window_length frames from the video from start_time to end_time
                        frames = list(video.subclip(start_time, end_time).iter_frames(fps=self.sliding_window_length / clip_duration))[: self.sliding_window_length]
                        for frame in frames:
                            frame = Image.fromarray(frame)
                            frame_tensor = self.transform(frame)
                            frame_tensor = frame_tensor.permute(2, 0, 1)
                            frame_tensor = frame_tensor.unsqueeze(0)
                            frame_tensor = self._image_processor.transform(frame_tensor)
                            frame_tensor = frame_tensor.squeeze(-1).permute(1, 2, 0)
                            preprocess_frames.append(frame_tensor)

                        frames_tensor = torch.stack(preprocess_frames, dim=0)

                        image_embeds = self.model.ln_vision(self.model.visual_encoder(frames_tensor.half().to(self.device)))
                        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
                        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                        query_output = self.model.Qformer.bert(
                            query_embeds=query_tokens,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            return_dict=True,
                        )
                        encoded_window = query_output.last_hidden_state

                        for frame in encoded_window:
                            if cur_frame < (self.model.short_memory_length - self.merge_frame_length):
                                if len(self.model.short_memory_buffer) == self.model.short_memory_length:
                                    self.model.short_memory_buffer.pop(0)
                                self.model.short_memory_buffer.append(frame)
                            cur_frame += 1

                        if cur_frame == (self.model.short_memory_length - self.merge_frame_length):
                            cur_frame = 0

                            # merge short_memory_frames
                            similar_list = []
                            for frame_i in range(len(self.model.short_memory_buffer) - 1):
                                scores = self.model.short_memory_buffer[frame_i] @ self.model.short_memory_buffer[frame_i + 1].transpose(-1, -2)
                                frame_silimar = torch.mean(scores)
                                similar_list.append(frame_silimar)

                            while len(self.model.short_memory_buffer) > self.merge_frame_length:
                                max_value = max(similar_list)
                                max_index = similar_list.index(max_value)
                                new_frame_feature = (self.model.short_memory_buffer[max_index].cpu() + self.model.short_memory_buffer[max_index + 1].cpu()) / 2
                                self.model.short_memory_buffer[max_index] = new_frame_feature.cuda()
                                del self.model.short_memory_buffer[max_index + 1]
                                similar_list = []
                                for frame_i in range(len(self.model.short_memory_buffer) - 1):
                                    scores = self.model.short_memory_buffer[frame_i] @ self.model.short_memory_buffer[frame_i + 1].transpose(-1, -2)
                                    frame_silimar = torch.mean(scores)
                                    similar_list.append(frame_silimar)

                            for frame in self.model.short_memory_buffer:
                                self.model.long_memory_buffer.append(frame)

                    cur_image = self.model.encode_image(preprocess_frames[-1].unsqueeze(0).unsqueeze(2).half(), self.device)
                    video_emb, _ = self.model.encode_long_video(cur_image, device=self.device, middle_video=False)
                    img_list.append(video_emb)
                    llm_message = self.answer(img_list=img_list, input_text=context, num_beams=1, temperature=1.0, max_new_tokens=300, max_length=2000)[0]
                    text_outputs.append(llm_message)

                    # except Exception as e:
                    #     eval_logger.error(f"Error {e} in loading video")
                    #     image_tensor = None

            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            print(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
