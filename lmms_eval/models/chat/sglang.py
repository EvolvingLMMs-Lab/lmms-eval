import asyncio
import base64
import io
import json
import os
import shutil
import tempfile
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from typing import List, Optional, Tuple, Union

import numpy as np
from accelerate import Accelerator, DistributedType
from mcp.types import AudioContent, ImageContent, TextContent
from PIL import Image
from sglang import Engine
from tqdm import tqdm
from transformers import AutoProcessor

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.mcp import MCPClient
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.load_video import load_video_decord
from lmms_eval.protocol import ChatMessages

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger
from qwen_vl_utils import process_vision_info

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser

try:
    from sglang.srt.entrypoints.openai.protocol import Tool
except ImportError:
    from sglang.srt.openai_api.protocol import Tool


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
        mcp_server_path: str = None,
        max_turn: int = 5,
        work_dir: str = None,
        # No need to json decode this argument, it will be decoded by the server_args.py
        json_model_override_args: Optional[str] = None,
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
        self.max_turn = max_turn
        self.work_dir = work_dir if work_dir is not None else tempfile.mkdtemp()
        # Convert any string arguments that start with { and end with } to dictionaries
        for key, value in kwargs.items():
            if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                try:
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    eval_logger.warning(f"Failed to parse JSON-like string for argument '{key}': {value}")
        if json_model_override_args is not None:
            kwargs["json_model_override_args"] = json_model_override_args
        if mcp_server_path is not None:
            self.mcp_client = MCPClient(mcp_server_path)
        else:
            self.mcp_client = None
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.tools, self.tool_call_parser_type, self.sgl_tools, self.function_call_parser = self._init_tools_sglang()
        # Set up sglang client
        self.client = Engine(model_path=model, tp_size=tensor_parallel_size, mem_fraction_static=gpu_memory_utilization, trust_remote_code=trust_remote_code, **kwargs)
        if chat_template is not None:
            with open(chat_template, "r") as f:
                chat_template = f.read()
                self.processor.chat_template = chat_template

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

    def get_mcp_tools(self):
        """
        Get the list of available MCP tools.
        :return: List of available tools in OpenAI-compatible format.
        """
        if self.mcp_client is None:
            return None

        try:
            tools = self.mcp_client.get_function_list_sync()
            return tools
        except Exception as e:
            eval_logger.error(f"Failed to retrieve MCP tools: {str(e)}")
            return None

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "TODO, not implemented"

    def _prepare_video_kwargs(self):
        video_kwargs = {"max_pixels": self.max_pixels, "min_pixels": self.min_pixels, "max_frames": self.max_frame_num}
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        else:
            video_kwargs["nframes"] = self.nframes
        return video_kwargs

    def _prepare_single_message(self, batch_request):
        """
        Helper method to prepare a single message for batching.
        This can be parallelized using ThreadPoolExecutor.
        """
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = batch_request.arguments
        chat_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages: ChatMessages = ChatMessages(**{"messages": chat_messages})

        # Set default generation parameters
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if gen_kwargs["max_new_tokens"] > 4096:
            gen_kwargs["max_new_tokens"] = 4096
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = 0.95

        # Prepare video kwargs
        video_kwargs = self._prepare_video_kwargs()

        # Convert to HF messages and extract media
        messages = chat_messages.to_hf_messages(video_kwargs)
        images, videos, audios = chat_messages.extract_media()

        # Handle media files if tools are available
        if self.tools is not None:
            for image_idx, image in enumerate(images):
                image_path = os.path.join(self.work_dir, f"{uuid.uuid4()}.jpg")
                image.save(image_path)
                messages[-1]["content"].append({"type": "text", "text": f"\nImage {image_idx} has image path: {image_path}"})
            for video_idx, video in enumerate(videos):
                messages[-1]["content"].append({"type": "text", "text": f"\nVideo {video_idx} has video path: {video}"})

        return messages, images

    def _extract_gen_params(self, gen_kwargs):
        """Extract generation parameters with defaults."""
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if gen_kwargs["max_new_tokens"] > 4096:
            gen_kwargs["max_new_tokens"] = 4096
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = 0.95

        return {
            "temperature": gen_kwargs["temperature"],
            "max_new_tokens": gen_kwargs["max_new_tokens"],
            "top_p": gen_kwargs["top_p"],
        }

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
            # Prepare messages in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(batch_requests), self.threads)) as executor:
                batched_messages_and_images = list(executor.map(self._prepare_single_message, batch_requests))

            # Unpack messages and images from parallel results
            batched_messages = [msg for msg, _ in batched_messages_and_images]
            image_data = [imgs for _, imgs in batched_messages_and_images]

            # Extract generation parameters from first request (should be same for batch)
            ctx, doc_to_messages, gen_kwargs, doc_id, task, split = batch_requests[0].arguments
            params = self._extract_gen_params(gen_kwargs)

            image_inputs, video_inputs, video_kwargs = process_vision_info(batched_messages, return_video_kwargs=True, return_video_metadata=True)
            texts = self.processor.apply_chat_template(
                batched_messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=self.tools,
            )
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
                for req, meta in zip(batch_requests, video_metadatas):
                    req.video_metadata = meta
            else:
                video_metadatas = None
                for req in batch_requests:
                    req.video_metadata = None
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
            if self.mcp_client is None:
                outputs = self.batch_level_generate(input_ids=input_ids, sampling_params=params, image_data=image_inputs)
            else:
                outputs = self.req_level_generate(input_ids=input_ids, image_data=image_inputs, sampling_params=params, batched_messages=batched_messages)
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

        if self.mcp_client is not None:
            shutil.rmtree(self.work_dir)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAHF")

    def get_tool_call_parser_type(
        self,
        processing_class,
    ) -> str:
        items = FunctionCallParser.ToolCallParserEnum.items()
        if "gpt-oss" in getattr(processing_class, "name_or_path", "").lower():
            eval_logger.debug(f"gpt-oss model detected from name_or_path: {processing_class.name_or_path}")
            eval_logger.debug("Using 'gpt-oss' tool call parser.")
            return "gpt-oss"
        for parser_type, parser_cls in items:
            parser = parser_cls()
            try:
                # This is when processing_class is a tokenizer
                tokenizer_vocab = processing_class.get_vocab()
            except AttributeError:
                try:
                    # This is when processing_class is a processor
                    tokenizer_vocab = processing_class.tokenizer.get_vocab()
                except AttributeError as e:
                    raise ValueError(f"Cannot get vocab from processing_class {processing_class}") from e

            if parser.bot_token.strip() in tokenizer_vocab and (parser.eot_token == "" or parser.eot_token.strip() in tokenizer_vocab):
                return parser_type
        else:
            raise ValueError(f"No tool call parser found for processing_class {processing_class}")

    def _init_tools_sglang(self):
        if self.mcp_client is None:
            return [], None, [], None

        tools = self.get_mcp_tools()
        tool_call_parser_type = self.get_tool_call_parser_type(self.processor)
        sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tools]
        function_call_parser = FunctionCallParser(
            sgl_tools,
            tool_call_parser_type,
        )
        # Make the detector to ignore new line token
        function_call_parser.detector.bot_token = function_call_parser.detector.bot_token.strip()
        function_call_parser.detector.eot_token = function_call_parser.detector.eot_token.strip()

        return (
            tools,
            tool_call_parser_type,
            sgl_tools,
            function_call_parser,
        )

    async def async_a_request(self, input_id, image, sampling_params, messages):
        if not isinstance(image, list):
            image = [image]
        keep_rolling = True
        turn_count = 0
        while keep_rolling:
            output = await self.client.async_generate(input_ids=input_id, image_data=image, sampling_params=sampling_params)
            content = output["text"]
            content_id = self.processor.tokenizer.encode(content)

            finish_reason = output["meta_info"]["finish_reason"]["type"]
            if self.function_call_parser.has_tool_call(content):
                finish_reason = "tool_calls"
            if finish_reason == "stop" or finish_reason == "length":
                messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
                return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=self.tools, skip_special_tokens=True)
            elif finish_reason == "tool_calls":
                try:
                    normed_content, tool_calls = self.function_call_parser.parse_non_stream(content)
                except JSONDecodeError:
                    normed_content = content
                    tool_calls = []
                except AttributeError:
                    normed_content = content
                    tool_calls = []

                tool_messages = []
                new_image_data = []
                for tool_call in tool_calls:
                    try:
                        arguments = json.loads(tool_call.parameters)
                    except JSONDecodeError:
                        arguments = {}
                    results = await self.mcp_client.run_tool(tool_call.name, arguments)
                    content_list = []
                    for result in results.content:
                        if isinstance(result, ImageContent):
                            new_image = Image.open(io.BytesIO(base64.b64decode(result.data)))
                            new_image_data.append(new_image)
                            content_list.append({"type": "image"})
                        elif isinstance(result, TextContent):
                            content_list.append({"type": "text", "text": result.text})
                        else:
                            raise ValueError(f"Unsupported result type: {type(result)}. Only ImageContent, TextContent are supported.")
                    tool_messages.append({"role": "tool", "name": tool_call.name, "content": content_list})
                original_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                # Get the text for the tool calling part without system prompt
                tool_calling_text = self.processor.apply_chat_template(messages + tool_messages, tokenize=False, add_generation_prompt=True)
                tool_calling_text = tool_calling_text.split(original_text)[1]
                if len(new_image_data) == 0:
                    new_image_data = None
                inputs = self.processor(text=tool_calling_text, images=new_image_data, return_tensors="pt")
                tool_input_ids = inputs.pop("input_ids").flatten().tolist()

                # Append this round's result
                messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
                messages.extend(tool_messages)
                if new_image_data is not None:
                    image.extend(new_image_data)
                input_id = input_id + content_id + tool_input_ids
            else:
                # Finish reason is neither "stop", "length", nor "tool_calls"
                messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
                return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=self.tools, skip_special_tokens=True)
            turn_count += 1
            if turn_count >= self.max_turn:
                keep_rolling = False

        # Return the final message if max turns reached
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=self.tools, skip_special_tokens=True)

    def req_level_generate(self, input_ids, image_data, sampling_params, batched_messages):
        """
        Generate at request level with tool calling support.
        Returns output in the same format as batch_level_generate for consistency.
        Note: Metrics (token counts, latency) are approximate for this mode.
        """
        loop = asyncio.get_event_loop()
        output_list = []
        text_list = loop.run_until_complete(asyncio.gather(*[self.async_a_request(input_id, image, sampling_params, messages) for input_id, image, messages in zip(input_ids, image_data, batched_messages)]))
        output_list = [{"text": text} for text in text_list]
        return output_list

    def batch_level_generate(self, input_ids, image_data, sampling_params):
        """
        Generate at batch level without tool calling support.
        Returns list of outputs with format: [{"text": "...", "meta_info": {...}}, ...]
        """
        return self.client.generate(input_ids=input_ids, image_data=image_data, sampling_params=sampling_params)
