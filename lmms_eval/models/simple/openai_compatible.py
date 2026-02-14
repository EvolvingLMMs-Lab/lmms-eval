import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import List, Tuple, Union
from urllib.parse import unquote

import numpy as np
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.concurrency_control import (
    AdaptiveConcurrencyConfig,
    decide_next_concurrency,
    is_rate_limit_error,
    parse_bool,
)

VideoReader, _ = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")

from dotenv import load_dotenv
from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI
from PIL import Image

try:
    from openai import DefaultHttpxClient
except ImportError:
    DefaultHttpxClient = None

load_dotenv(verbose=True)


@register_model("openai_compatible")
class OpenAICompatible(lmms):
    def __init__(
        self,
        model_version: str = "grok-2-latest",
        base_url: str = None,
        api_key: str = None,
        timeout: int = 10,
        max_retries: int = 5,
        max_size_in_mb: int = 20,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        azure_openai: bool = False,
        max_frames_num: int = 10,
        httpx_trust_env: bool = True,
        batch_size: int = 64,
        num_concurrent: int = 32,
        adaptive_concurrency: bool = False,
        adaptive_min_concurrency: int = 1,
        adaptive_max_concurrency: int = 128,
        adaptive_target_latency_s: float = 15.0,
        adaptive_increase_step: float = 0.1,
        adaptive_decrease_factor: float = 0.7,
        adaptive_failure_threshold: float = 0.05,
        **kwargs,
    ) -> None:
        """
        :param httpx_trust_env: bool
            httpx.Client used by openai-python has trust_env set to True by default. A
            False value of this param constructs a httpx.Client with trust_env set to
            False.  Such a httpx.Client ignores environment variables (HTTP_PROXY,
            HTTPS_PROXY, ALL_PROXY) and macOS proxy server settings.
        """
        super().__init__()
        self.model_version = model_version
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_size_in_mb = max_size_in_mb  # some models have a limit on the size of the image
        self.continual_mode = continual_mode
        self.max_frames_num = max_frames_num
        self.num_concurrent = max(1, int(num_concurrent))
        self.adaptive_concurrency = parse_bool(adaptive_concurrency)
        self.adaptive_config = AdaptiveConcurrencyConfig.from_raw(
            min_concurrency=adaptive_min_concurrency,
            max_concurrency=adaptive_max_concurrency,
            target_latency_s=adaptive_target_latency_s,
            increase_step=adaptive_increase_step,
            decrease_factor=adaptive_decrease_factor,
            failure_threshold=adaptive_failure_threshold,
        )
        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")

            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        # In China mainland, people usually use a VPN client to access international web
        # sites such as Google. Such a client usually configures macOS proxy server
        # settings. openai-python uses a httpx.Client with trust_env set to True. Such a
        # httpx.Client uses macOS proxy server settings. Adding httpx_trust_env option
        # allows httpx to ignore proxy server settings set by VPN clients.
        if not httpx_trust_env and DefaultHttpxClient is None:
            eval_logger.warning("DefaultHttpxClient is unavailable in current openai package; " "falling back to default HTTP client with trust_env=True.")
            http_client = None
        else:
            if not httpx_trust_env and DefaultHttpxClient is not None:
                http_client = DefaultHttpxClient(trust_env=httpx_trust_env)
            else:
                http_client = None

        # Use provided parameters or fall back to environment variables
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_API_BASE")

        # Fix URL encoding issue - decode if it's URL encoded
        if base_url and "%" in base_url:
            base_url = unquote(base_url)

        # Remove trailing slash if present
        if base_url and base_url.endswith("/"):
            base_url = base_url.rstrip("/")

        self.client = (
            OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
            if not azure_openai
            else AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                http_client=http_client,
            )
        )

        accelerator = Accelerator()
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
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
    def batch_size(self):
        return self.batch_size_per_gpu

    def tok_encode(self, string: str):
        return list(string.encode("utf-8"))

    def tok_decode(self, tokens):
        return ""

    @property
    def eot_token_id(self):
        return 0

    @property
    def rank(self):
        return self._rank

    # Function to encode the image
    def encode_image(self, image: Union[Image.Image, str]):
        max_size = self.max_size_in_mb * 1024 * 1024  # 20MB in bytes
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        # If image is too large, resize it while maintaining aspect ratio
        while len(byte_data) > max_size and img.size[0] > 100 and img.size[1] > 100:
            new_size = (int(img.size[0] * 0.75), int(img.size[1] * 0.75))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        from lmms_eval import utils

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        ordered_requests = []
        for single_request in re_ords.get_batched(n=1, batch_fn=None):
            ordered_requests.extend(single_request)

        if not ordered_requests:
            return []

        pbar = tqdm(
            total=len(ordered_requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )
        reordered_responses: List[str] = []
        current_concurrency = min(
            self.num_concurrent,
            self.adaptive_config.max_concurrency,
        )
        cursor = 0

        while cursor < len(ordered_requests):
            if self.adaptive_concurrency:
                window_size = max(1, current_concurrency)
            else:
                window_size = max(1, max(self.batch_size, self.num_concurrent))

            chunk = ordered_requests[cursor : cursor + window_size]

            batch_payloads = []
            batch_doc_uuids = []
            batch_responses: List[Union[str, None]] = []

            for idx, request_args in enumerate(chunk):
                (
                    context,
                    gen_kwargs,
                    doc_to_visual_fn,
                    doc_id_single,
                    task_name,
                    split_name,
                ) = request_args
                doc_uuid = f"{task_name}___{split_name}___{doc_id_single}"
                batch_doc_uuids.append(doc_uuid)

                if self.continual_mode and self.cache_mode == "resume":
                    cached_response = self.response_cache.get(doc_uuid)
                    if cached_response:
                        batch_responses.append(cached_response)
                        continue

                visuals = [doc_to_visual_fn(self.task_dict[task_name][split_name][doc_id_single])]
                if None in visuals:
                    imgs = []
                else:
                    visuals = self.flatten(visuals)
                    imgs = []
                    for visual in visuals:
                        if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                            frames = self.encode_video(visual, self.max_frames_num)
                            imgs.extend(frames)
                        elif isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                            imgs.append(self.encode_image(visual))
                        elif isinstance(visual, Image.Image):
                            imgs.append(self.encode_image(visual))

                request_gen_kwargs = dict(gen_kwargs)
                max_new_tokens = min(request_gen_kwargs.get("max_new_tokens", 1024), 4096)
                temperature = request_gen_kwargs.get("temperature", 0)

                payload = {
                    "model": self.model_version,
                    "messages": [{"role": "user", "content": []}],
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                }
                payload["messages"][0]["content"].append({"type": "text", "text": context})
                for img in imgs:
                    payload["messages"][0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                    )

                if "o1" in self.model_version or "o3" in self.model_version:
                    payload.pop("temperature")
                    payload["reasoning_effort"] = "medium"
                    payload["response_format"] = {"type": "text"}
                    payload.pop("max_tokens")
                    payload["max_completion_tokens"] = max_new_tokens

                batch_payloads.append((idx, payload))
                batch_responses.append(None)

            def process_single_request(local_index: int, payload: dict):
                started_at = time.time()
                rate_limited = False

                for attempt in range(self.max_retries):
                    try:
                        response = self.client.chat.completions.create(**payload)
                        response_text = response.choices[0].message.content
                        latency = time.time() - started_at
                        return response_text, local_index, True, rate_limited, latency
                    except Exception as exc:
                        error_msg = str(exc)
                        rate_limited = rate_limited or is_rate_limit_error(error_msg)
                        eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")
                        if attempt == self.max_retries - 1:
                            eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                        else:
                            time.sleep(self.timeout)

                latency = time.time() - started_at
                return "", local_index, False, rate_limited, latency

            failed_requests = 0
            rate_limited_requests = 0
            request_latencies: List[float] = []

            tasks_to_run = [(local_index, payload) for local_index, payload in batch_payloads if batch_responses[local_index] is None]

            if tasks_to_run:
                max_workers = min(len(tasks_to_run), current_concurrency)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_single_request, local_index, payload): local_index for local_index, payload in tasks_to_run}

                    for future in as_completed(futures):
                        (
                            response_text,
                            local_index,
                            success,
                            rate_limited,
                            latency,
                        ) = future.result()
                        batch_responses[local_index] = response_text
                        if not success:
                            failed_requests += 1
                        if rate_limited:
                            rate_limited_requests += 1
                        request_latencies.append(latency)

            completed_batch_responses = [response if response is not None else "" for response in batch_responses]

            if self.continual_mode:
                for doc_uuid, response_text in zip(
                    batch_doc_uuids,
                    completed_batch_responses,
                ):
                    self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

            reordered_responses.extend(completed_batch_responses)
            if self.adaptive_concurrency and tasks_to_run:
                decision = decide_next_concurrency(
                    current_concurrency=current_concurrency,
                    total_requests=len(tasks_to_run),
                    failed_requests=failed_requests,
                    rate_limited_requests=rate_limited_requests,
                    latencies=request_latencies,
                    config=self.adaptive_config,
                )
                if decision.next_concurrency != decision.current_concurrency:
                    eval_logger.info(
                        "Adaptive concurrency update: "
                        f"{decision.current_concurrency} -> "
                        f"{decision.next_concurrency} "
                        f"(fail_rate={decision.failure_rate:.3f}, "
                        f"rate_limit_rate={decision.rate_limit_rate:.3f}, "
                        f"p95_latency={decision.p95_latency_s:.3f}s)"
                    )
                current_concurrency = decision.next_concurrency

            cursor += len(chunk)
            pbar.update(len(chunk))

        pbar.close()
        return re_ords.get_original(reordered_responses)

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for OpenAI compatible models")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TODO: Implement loglikelihood for OpenAI compatible models")
