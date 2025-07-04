import json
import warnings
from typing import List, Optional, Tuple, Union

import PIL
from accelerate import Accelerator, DistributedType
from sglang import Engine
from tqdm import tqdm
from transformers import AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import load_video_decord
from lmms_eval.protocol import ChatMessages

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger


@register_model("sglang_runtime")
class Sglang(lmms):
    is_simple = False

    def __init__(
        self,
        model_version: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: int = 1,
        max_frame_num: int = 32,
        threads: int = 16,  # Threads to use for decoding visuals
        trust_remote_code: Optional[bool] = True,
        chat_template: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.max_frame_num = max_frame_num
        self.threads = threads
        self.chat_template = chat_template

        # Convert any string arguments that start with { and end with } to dictionaries
        for key, value in kwargs.items():
            if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                try:
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    eval_logger.warning(f"Failed to parse JSON-like string for argument '{key}': {value}")

        # Set up vllm client
        self.client = Engine(model_path=model_version, tp_size=tensor_parallel_size, mem_fraction_static=gpu_memory_utilization, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_version)

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)

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
        return self.client

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

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
        assert False, "TODO, not implemented"

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        for batch_requests in batched_requests:
            batched_messages = []
            image_data = []
            for idx in range(len(batch_requests)):
                ctx, doc_to_messages, gen_kwargs, doc_id, task, split = batch_requests[idx].arguments
                chat_messages = doc_to_messages(self.task_dict[task][split][doc_id])
                chat_messages: ChatMessages = ChatMessages(**{"messages": chat_messages})
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if gen_kwargs["max_new_tokens"] > 4096:
                    gen_kwargs["max_new_tokens"] = 4096
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = 0.95

                params = {
                    "temperature": gen_kwargs["temperature"],
                    "max_new_tokens": gen_kwargs["max_new_tokens"],
                    "top_p": gen_kwargs["top_p"],
                }
                video_kwargs = {"enforce_image": True, "num_frames": self.max_frame_num}
                messages = chat_messages.to_hf_messages(video_kwargs)

                images, videos, audio = chat_messages.extract_media()
                video_data = []
                for video in videos:
                    video_data.extend(load_video_decord(video, max_frames_num=self.max_frame_num))
                if len(images) > 0:
                    image_data.append(images)
                if len(video_data) > 0:
                    image_data.append(video_data)

                batched_messages.append(messages)

            texts = self.processor.apply_chat_template(
                batched_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            outputs = self.client.generate(texts, params, image_data=image_data)

            response_text = [o["text"] for o in outputs]

            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAHF")
