import asyncio
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from multiprocessing import cpu_count
from typing import List, Optional, Tuple, Union

import numpy as np
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

NUM_SECONDS_TO_SLEEP = int(os.getenv("NUM_SECONDS_TO_SLEEP", "5"))
WORKERS = int(os.getenv("WORKERS", "32"))

try:
    from vllm import LLM, SamplingParams
except ImportError:
    vllm = None


@register_model("vllm")
class VLLM(lmms):
    """
    VLLM model wrapper for large multimodal models evaluation.

    This class provides a wrapper around the VLLM library to run inference on
    vision-language models. It supports both image and video inputs with automatic
    encoding and batched processing.

    Supported models: https://docs.vllm.ai/en/latest/models/supported_models.html

    Supported media formats:
        - Images: .jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp
        - Videos: .mp4, .avi, .mov, .flv, .wmv

    Chat template:
        The chat template is used to format the conversation for the model. It can be
        provided as a file path or as a template string directly.
        - Chat template intro: https://huggingface.co/docs/transformers/en/chat_templating
        - VLLM chat method: https://docs.vllm.ai/en/stable/models/generative_models.html#llmchat

    Args:
        model_version (str): HuggingFace model identifier or path to the model.
            Default: "Qwen/Qwen2.5-VL-3B-Instruct"
        tensor_parallel_size (int): Number of GPUs to use for tensor parallelism.
            Default: 1
        gpu_memory_utilization (float): Fraction of GPU memory to use for model weights.
            Should be between 0.0 and 1.0. Default: 0.8
        batch_size (int): Number of requests to process in parallel per GPU.
            Default: 1
        max_frame_num (int): Maximum number of frames to extract from videos.
            Frames are sampled uniformly across the video duration. Default: 32
        threads (int): Number of threads to use for parallel visual encoding.
            Default: 16
        trust_remote_code (bool, optional): Whether to trust remote code when loading
            the model. Default: True
        chat_template (str, optional): Path to chat template file or template string.
            If None, uses the model's default template. Default: None
        **kwargs: Additional arguments passed to the VLLM LLM constructor.
            - NOTE: model specific arguments can be passed here without the need to add more arguments to this class (see example below)
            - String arguments that look like JSON dictionaries will be automatically parsed.


    Python Example 1: (example of passing model specific arguments)
    # ---------------------
    import subprocess
    cmd = [
            "python3",
            "-m",
            "lmms_eval",
            "--model",
            "vllm",
            "--model_args",
            "model_version=meta-llama/Llama-4-Scout-17B-16E-Instruct,"
            "tensor_parallel_size=4,"
            "dtype=bfloat16,"
            "max_model_len=10240,"
            "gpu_memory_utilization=0.9,"
            'override_generation_config={"attn_temperature_tuning": true},' # example of passing model specific arguments, JSON string will be parsed automatically
            "enforce_eager=True,"
            "kv_cache_dtype=fp8",
            "--tasks",
            task, # change this to your task
            "--batch_size",
            "1",
            "--limit",
            "10",
            "--log_samples",
            "--output_path",
            "logs",
        ]
    cmd_result = subprocess.run(cmd, check=False)
    # ---------------------


    Python Example 2: (example of using chat template file)
    # ---------------------
    chat_template_file = "template_deepseek_vl2.jinja"
    subprocess.run(
        f"wget https://raw.githubusercontent.com/vllm-project/vllm/main/examples/template_deepseek_vl2.jinja -O {chat_template_file}",
        check=True,
        shell=True,
    )
    cmd = [
        "python3",
        "-m",
        "lmms_eval",
        "--model",
        "vllm",
        "--model_args",
        "model_version=deepseek-ai/deepseek-vl2,"
        'hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},' # example of passing model specific arguments, JSON string will be parsed automatically
        f"chat_template={chat_template_file}," # chat template file path
        "tensor_parallel_size=2,"
        "dtype=bfloat16",
        "--tasks",
        task,
        "--batch_size",
        "1",
        "--limit",
        "1000",
        "--log_samples",
        "--output_path",
        "logs",
    ]
    cmd_result = subprocess.run(cmd, check=False)
    # ---------------------


    # NOTE: No need to pass the chat template file if it is already defined in the model tokenizer.
    # The chat method automatically applies the model's chat template to format the prompt
    # - vllm chat method: https://docs.vllm.ai/en/stable/models/generative_models.html#llmchat

    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: int = 1,
        max_frame_num: int = 32,
        trust_remote_code: Optional[bool] = True,
        chat_template: Optional[str] = None,
        min_image_pixels: int = 28,  # minimum image dimension, required for Qwen 2/2.5-VL models
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model = model
        self.max_frame_num = max_frame_num
        self.chat_template = chat_template
        self.min_image_pixels = min_image_pixels
        self.data_parallel_size = data_parallel_size
        # Qwen 2/2.5-VL models enforce minimum image dimensions
        self._enforce_image_resize = self._is_qwen_vl_model(model)

        # Load chat template during initialization
        self.chat_template = None
        if chat_template is not None:
            # Check if it looks like a file path (contains path separators or has common template extensions)
            if os.path.sep in chat_template or chat_template.endswith((".jinja", ".jinja2", ".j2")):
                # It appears to be a file path, so it must exist
                if not os.path.isfile(chat_template):
                    raise FileNotFoundError(f"Chat template file not found: {chat_template}")
                with open(chat_template, "r") as f:
                    self.chat_template = f.read()
            else:
                # Treat as a template string
                self.chat_template = chat_template

        # Convert any string arguments that start with { and end with } to dictionaries
        for key, value in kwargs.items():
            if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                try:
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    eval_logger.warning(f"Failed to parse JSON-like string for argument '{key}': {value}")

        # Set up vllm client
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

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
        # TODO: Support tensor parallelism in the future for flexible vllm parallel
        if data_parallel_size > 1:
            assert tensor_parallel_size == 1, "Data parallelism is not supported with tensor parallelism. For current vllm version"
        self.client = LLM(
            model=self.model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            disable_log_stats=False,
            distributed_executor_backend="external_launcher",
            seed=1,
            **kwargs,
        )

        self.device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)

    def _is_qwen_vl_model(self, model_version: str) -> bool:
        qwen_vl_patterns = ["qwen2-vl", "qwen2.5-vl"]
        return any(pattern in model_version.lower() for pattern in qwen_vl_patterns)

    def _maybe_resize_image(self, img: Image.Image) -> Image.Image:
        # edge‐case validation
        if self.min_image_pixels <= 0:
            return img
        if min(img.size) <= 0:
            raise ValueError(f"Invalid image dimensions: {img.size}")

        if not self._enforce_image_resize or min(img.size) >= self.min_image_pixels:
            return img

        scale = self.min_image_pixels / min(img.size)  # maintain original aspect ratio
        new_size = tuple(int(dim * scale) for dim in img.size)
        return img.resize(new_size, Image.BICUBIC)

    # Function to encode the image
    def encode_image(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        img = self._maybe_resize_image(img)
        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frame_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            img = self._maybe_resize_image(img)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            if isinstance(i, (list, tuple)):
                new_list.extend(i)
            else:
                new_list.append(i)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        for batch_requests in batched_requests:
            batched_messages = []
            for idx in range(len(batch_requests)):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[idx].arguments
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
                    "max_tokens": gen_kwargs["max_new_tokens"],
                    "top_p": gen_kwargs["top_p"],
                }
                sampling_params = SamplingParams(**params)

                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                if None in visuals:
                    visuals = []
                    imgs = []
                else:
                    visuals = self.flatten(visuals)
                    imgs = []  # multiple images or frames for video
                    all_tasks = []
                    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                        for visual in visuals:
                            if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                                all_tasks.append(executor.submit(self.encode_video, visual))
                            elif isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                                all_tasks.append(executor.submit(self.encode_image, visual))
                            elif isinstance(visual, Image.Image):
                                all_tasks.append(executor.submit(self.encode_image, visual))

                        for task in all_tasks:
                            imgs.append(task.result())

                messages = [{"role": "user", "content": []}]
                # When there is no image token in the context, append the image to the text
                messages[0]["content"].append({"type": "text", "text": contexts})
                for img in self.flatten(imgs):
                    messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

                batched_messages.append(messages)

            sampling_params = SamplingParams(**params)

            # NOTE:
            # The chat method automatically applies the model's chat template to format the prompt
            # - vllm chat method: https://docs.vllm.ai/en/stable/models/generative_models.html#llmchat
            # The logic here is similar to the vllm implementation as shown here (https://docs.vllm.ai/en/stable/models/generative_models.html#llmchat)
            # - vllm implementation: https://github.com/vllm-project/vllm/blob/d97841078b6e0dde8da36d5a2b8e8857a2c37944/vllm/entrypoints/chat_utils.py#L829
            if self.chat_template is not None:
                response = self.client.chat(sampling_params=sampling_params, messages=batched_messages, chat_template=self.chat_template)
            else:
                response = self.client.chat(sampling_params=sampling_params, messages=batched_messages)
            response_text = [o.outputs[0].text for o in response]

            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
