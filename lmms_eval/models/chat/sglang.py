import json
import time
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from accelerate import Accelerator, DistributedType
from PIL import Image
from sglang import Engine
from tqdm import tqdm
from transformers import AutoProcessor

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.load_video import load_video_decord
from lmms_eval.protocol import ChatMessages

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger
from qwen_vl_utils import process_vision_info


@register_model("sglang_runtime")
class Sglang(lmms):
    is_simple = False

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: int = 1,
        nframes: int = 32,
        max_frame_num: int = 768,
        fps: Optional[int] = None,
        max_pixels: int = 1605632,
        min_pixels: int = 28 * 28,
        threads: int = 16,  # Threads to use for decoding visuals
        trust_remote_code: Optional[bool] = True,
        chat_template: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self._model = model
        self.nframes = nframes
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

        # Set up sglang client
        self.client = Engine(model_path=model, tp_size=tensor_parallel_size, mem_fraction_static=gpu_memory_utilization, trust_remote_code=trust_remote_code, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=trust_remote_code)

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
        self.fps = fps
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

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

    @property
    def image_token_id(self):
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            image_token = getattr(self.processor, "image_token", None)
            if image_token is None:
                raise ValueError("Image token not found in processor")
            image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        return image_token_id

    @property
    def video_token_id(self):
        video_token_id = getattr(self.processor, "video_token_id", None)
        if video_token_id is None:
            video_token = getattr(self.processor, "video_token", None)
            if video_token is None:
                raise ValueError("Video token not found in processor")
            video_token_id = self.tokenizer.convert_tokens_to_ids(video_token)
        return video_token_id

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        total_tokens = 0
        e2e_latency = 0
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
                video_kwargs = {"enforce_image": True, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels, "max_frames": self.max_frame_num}
                if self.fps is not None:
                    video_kwargs["fps"] = self.fps
                else:
                    video_kwargs["nframes"] = self.nframes
                messages = chat_messages.to_hf_messages(video_kwargs)

                batched_messages.append(messages)
            image_inputs, video_inputs, video_kwargs = process_vision_info(batched_messages, return_video_kwargs=True, return_video_metadata=True)
            texts = self.processor.apply_chat_template(
                batched_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
            else:
                video_metadatas = None
            assert image_inputs is None or video_inputs is None, "Only one of image or video inputs should be provided"
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, video_metadata=video_metadatas, **video_kwargs, padding=True, return_tensors="pt")
            # If video inputs is not None, we need to replace the image token ids with the video token ids before generating
            # so that the visual tokens are being scattered correctly.
            if video_inputs is not None:
                input_ids = inputs.pop("input_ids")
                input_ids[input_ids == self.video_token_id] = self.image_token_id
                input_ids = input_ids.tolist()
                image_inputs = []
                for video_input in video_inputs:
                    images = [Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8)) for frame in video_input]
                    image_inputs.append(images)
            else:
                input_ids = inputs.pop("input_ids").tolist()

            start_time = time.time()
            outputs = self.client.generate(input_ids=input_ids, sampling_params=params, image_data=image_inputs)
            end_time = time.time()

            response_text = [o["text"] for o in outputs]

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time

            for output_idx, output in enumerate(outputs):
                # Get token count from output
                if "meta_info" in output and "completion_tokens" in output["meta_info"]:
                    output_tokens = output["meta_info"]["completion_tokens"]
                else:
                    output_tokens = len(output["text"].split())

                total_tokens += output_tokens

            if len(outputs) >= 1:
                avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0

            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAHF")
