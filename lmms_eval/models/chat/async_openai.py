import asyncio
import base64
import json
import os
import tempfile
import time
import uuid
from io import BytesIO
from multiprocessing import cpu_count
from typing import List, Optional, Tuple, Union

import numpy as np
import requests as url_requests
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from dotenv import find_dotenv, load_dotenv
from loguru import logger as eval_logger
from openai import AsyncOpenAI
from PIL import Image

from lmms_eval.api.model import lmms
from lmms_eval.mcp import MCPClient
from lmms_eval.models.simple.openai_compatible import (
    OpenAICompatible as OpenAICompatibleSimple,
)
from lmms_eval.protocol import ChatMessages

load_dotenv(verbose=True)


@register_model("async_openai_compatible_chat")
class AsyncOpenAIChat(lmms):
    is_simple = False

    def __init__(
        self,
        model_version: str = "grok-2-latest",
        base_url: str = None,
        api_key: str = None,
        timeout: int = 600,
        max_retries: int = 5,
        max_size_in_mb: int = 20,
        mcp_server_path: str = None,
        num_cpus: int = None,
        work_dir: str = None,
        fps: Optional[int] = None,
        nframes: Optional[int] = 64,
        max_pixels: Optional[int] = 151200,
        min_pixels: Optional[int] = 28 * 28,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_size_in_mb = max_size_in_mb  # some models have a limit on the size of the image
        if num_cpus is None:
            self.num_cpus = cpu_count() // 2
        else:
            self.num_cpus = num_cpus
        self.work_dir = work_dir if work_dir is not None else tempfile.mkdtemp()
        self.fps = fps
        self.nframes = nframes
        self.base_url = base_url if base_url is not None else os.getenv("OPENAI_API_BASE")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        if mcp_server_path is not None:
            self.mcp_client = MCPClient(mcp_server_path)
            os.makedirs(self.work_dir, exist_ok=True)
        else:
            self.mcp_client = None

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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "TODO, not implemented"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAHF")

    async def maybe_forward_with_tool(self, request: Instance, idx: int):
        """
        Forward the request to the OpenAI API, using tools if available.
        This method is designed to handle chat messages and tool calls in an asynchronous manner.
        It retrieves the chat messages from the request, prepares the payload, and sends it to the OpenAI API.
        If the response indicates that tool calls are needed, it will call the tools using the MCP client and continue the conversation until a final response is received.
        :param request: The request instance containing the chat messages and other parameters.
        :param idx: The index of the request in the batch. (Use to restore the original order of responses)
        """
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
        chat_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages: ChatMessages = ChatMessages(**{"messages": chat_messages})
        video_kwargs = {"max_pixels": self.max_pixels, "min_pixels": self.min_pixels}
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        else:
            video_kwargs["nframes"] = self.nframes
        messages = chat_messages.to_openai_messages(video_kwargs)
        images, videos, audios = chat_messages.extract_media()
        if self.mcp_client is not None:
            for image_idx, image in enumerate(images):
                image_path = os.path.join(self.work_dir, f"{uuid.uuid4()}.jpg")
                image.save(image_path)
                messages[-1]["content"].append({"type": "text", "text": f"\nImage {image_idx} has image path: {image_path}"})
            for video_idx, video in enumerate(videos):
                messages[-1]["content"].append({"type": "text", "text": f"\nVideo {video_idx} has video path: {video}"})

        payload = {"messages": messages}
        payload["model"] = self.model_version
        all_response = ""

        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = False
        # payload["max_completion_tokens"] = gen_kwargs["max_new_tokens"]
        payload["max_tokens"] = gen_kwargs["max_new_tokens"]
        payload["temperature"] = gen_kwargs["temperature"]

        if self.mcp_client is not None:
            # get the function list from the MCP server
            functions = await self.mcp_client.get_function_list()
            payload["tools"] = functions
            payload["tool_choice"] = "auto"  # or "auto" for automatic tool selection

        response = await self.client.chat.completions.create(**payload)
        last_response = response.choices[0].message.content
        all_response += last_response

        while response.choices[0].finish_reason == "tool_calls":
            messages.append({"role": "assistant", "content": last_response})
            messages.append({"role": "assistant", "tool_calls": response.choices[0].message.tool_calls})
            message = response.choices[0].message
            tool_messages = []
            if message.tool_calls:
                eval_logger.debug("Calling tool with MCP client")
                for call in message.tool_calls:
                    eval_logger.debug(f"Calling {call.function.name}...")
                    result = await self.mcp_client.run_tool(call.function.name, eval(call.function.arguments))
                    all_response += f"<tool_call>{call.function.name} {call.function.arguments}</tool_call></tool_response>"
                    tool_messages.append({"role": "tool", "name": call.function.name, "content": []})
                    for content in result.content:
                        tool_message = self.mcp_client.convert_result_to_openai_format(content)
                        for content in tool_message:
                            if content["type"] == "image_url":
                                all_response += "<image_url>"
                            elif content["type"] == "text":
                                all_response += content["text"]
                        tool_messages[-1]["content"].extend(tool_message)
                    all_response += "</tool_response>"

            response = await self.client.chat.completions.create(
                model=self.model_version,
                messages=messages + tool_messages,
                max_tokens=gen_kwargs["max_new_tokens"],
                temperature=gen_kwargs["temperature"],
                tools=functions,
                tool_choice="auto",
            )
            last_response = response.choices[0].message.content
            all_response += last_response
        self.add_request_response_to_cache(request, all_response)
        return all_response, idx

    def generate_until(self, requests) -> List[str]:
        self.load_cache()
        results, requests = self.get_response_from_cache(requests)

        async def run():
            res = []
            pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
            sem = asyncio.Semaphore(self.num_cpus)

            async def _process(req, idx):
                async with sem:
                    return await self.maybe_forward_with_tool(req, idx)

            tasks = [asyncio.create_task(_process(req, idx)) for idx, req in enumerate(requests)]
            for task in asyncio.as_completed(tasks):
                content, idx = await task
                res.append((content, idx))
                pbar.update(1)

            pbar.close()
            return res

        eval_results = asyncio.run(run())
        eval_results.sort(key=lambda x: x[1])  # Sort by index to restore original
        results = results + [content for content, _ in eval_results]
        return results
