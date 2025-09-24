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

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from dotenv import load_dotenv
from loguru import logger as eval_logger
from openai import AzureOpenAI, DefaultHttpxClient, OpenAI
from PIL import Image

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
        http_client = DefaultHttpxClient(trust_env=httpx_trust_env) if not httpx_trust_env else None

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
            else AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"), api_version=os.getenv("AZURE_OPENAI_API_VERSION"), http_client=http_client)
        )

        accelerator = Accelerator()
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
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
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        from lmms_eval import utils

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]
            task = task[0]
            split = split[0]

            batch_payloads = []
            batch_doc_uuids = []
            batch_responses = []

            for i, (context, doc_id_single) in enumerate(zip(contexts, doc_id)):
                doc_uuid = f"{task}___{split}___{doc_id_single}"
                batch_doc_uuids.append(doc_uuid)

                if self.continual_mode is True and self.cache_mode == "resume":
                    if doc_uuid in self.response_cache:
                        response_text = self.response_cache[doc_uuid]
                        if response_text:
                            batch_responses.append(response_text)
                            continue

                visuals = [doc_to_visual[i](self.task_dict[task][split][doc_id_single])]
                if None in visuals:
                    visuals = []
                    imgs = []
                else:
                    visuals = self.flatten(visuals)
                    imgs = []
                    for visual in visuals:
                        if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                            frames = self.encode_video(visual, self.max_frames_num)
                            imgs.extend(frames)
                        elif isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                            img = self.encode_image(visual)
                            imgs.append(img)
                        elif isinstance(visual, Image.Image):
                            img = self.encode_image(visual)
                            imgs.append(img)

                payload = {"messages": []}
                payload["model"] = self.model_version

                payload["messages"].append({"role": "user", "content": []})
                payload["messages"][0]["content"].append({"type": "text", "text": context})
                for img in imgs:
                    payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if gen_kwargs["max_new_tokens"] > 4096:
                    gen_kwargs["max_new_tokens"] = 4096
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                payload["max_tokens"] = gen_kwargs["max_new_tokens"]
                payload["temperature"] = gen_kwargs["temperature"]

                if "o1" in self.model_version or "o3" in self.model_version:
                    del payload["temperature"]
                    payload["reasoning_effort"] = "medium"
                    payload["response_format"] = {"type": "text"}
                    payload.pop("max_tokens")
                    payload["max_completion_tokens"] = gen_kwargs["max_new_tokens"]

                batch_payloads.append(payload)
                batch_responses.append(None)

            def process_single_request(payload, i):
                if batch_responses[i] is not None:
                    return batch_responses[i], i

                for attempt in range(self.max_retries):
                    try:
                        response = self.client.chat.completions.create(**payload)
                        response_text = response.choices[0].message.content
                        return response_text, i

                    except Exception as e:
                        error_msg = str(e)
                        eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")

                        if attempt == self.max_retries - 1:
                            eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                            return "", i
                        else:
                            time.sleep(self.timeout)

                return "", i

            tasks_to_run = [(payload, i) for i, payload in enumerate(batch_payloads) if batch_responses[i] is None]

            if tasks_to_run:
                max_workers = min(len(tasks_to_run), 32)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = {executor.submit(process_single_request, payload, i): i for payload, i in tasks_to_run}

                    for future in as_completed(future_to_index):
                        response_text, i = future.result()
                        batch_responses[i] = response_text

            if self.continual_mode is True:
                for doc_uuid, response_text in zip(batch_doc_uuids, batch_responses):
                    if response_text is not None:
                        self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

            res.extend([r for r in batch_responses if r is not None])
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for OpenAI compatible models")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TODO: Implement loglikelihood for OpenAI compatible models")
