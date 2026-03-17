"""SGLang model wrapper for lmms-eval.

Supports image and video multimodal evaluation via SGLang's Engine API.
Handles TP via SGLang's internal parallelism (not torch.distributed).
"""

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
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from accelerate import Accelerator, DistributedType
from PIL import Image
from sglang import Engine
from transformers import AutoConfig, AutoProcessor

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.progress import make_progress
from lmms_eval.protocol import ChatMessages

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

# ---------------------------------------------------------------------------
# Optional imports with version-compatibility fallbacks
# ---------------------------------------------------------------------------

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser

try:
    from sglang.srt.entrypoints.openai.protocol import Tool
except ImportError:
    from sglang.srt.openai_api.protocol import Tool

if TYPE_CHECKING:
    from lmms_eval.mcp.client import MCPClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mcp_client(server_path: str) -> "MCPClient":
    try:
        from lmms_eval.mcp.client import MCPClient
    except ImportError as exc:
        raise ImportError("MCP support requires the optional 'mcp' dependency. " "Install with: pip install 'lmms_eval[mcp]'") from exc
    return MCPClient(server_path)


def _is_mcp_image_content(result: Any) -> bool:
    return result.__class__.__name__ == "ImageContent" and hasattr(result, "data")


def _is_mcp_text_content(result: Any) -> bool:
    return result.__class__.__name__ == "TextContent" and hasattr(result, "text")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


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
        threads: int = 16,
        trust_remote_code: Optional[bool] = True,
        chat_template: Optional[str] = None,
        mcp_server_path: str = None,
        max_turn: int = 5,
        work_dir: str = None,
        json_model_override_args: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._model = model
        self._config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.nframes = nframes
        self.max_frame_num = max_frame_num
        self.threads = threads
        self.chat_template = chat_template
        self.max_turn = max_turn
        self.work_dir = work_dir if work_dir is not None else tempfile.mkdtemp()

        # Parse JSON-like string arguments
        for key, value in kwargs.items():
            if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                try:
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    eval_logger.warning(f"Failed to parse JSON string for '{key}': {value}")
        if json_model_override_args is not None:
            kwargs["json_model_override_args"] = json_model_override_args

        # MCP tools
        self.mcp_client = _build_mcp_client(mcp_server_path) if mcp_server_path else None

        # Processor and tools
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.tools, self.tool_call_parser_type, self.sgl_tools, self.function_call_parser = self._init_tools_sglang()

        # SGLang Engine
        self.client = Engine(
            model_path=model,
            tp_size=tensor_parallel_size,
            mem_fraction_static=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        # Custom chat template
        if chat_template is not None:
            with open(chat_template, "r") as f:
                self.processor.chat_template = f.read()

        # Distributed setup
        accelerator = Accelerator()
        self.accelerator = accelerator
        self._rank = accelerator.process_index
        self._world_size = accelerator.num_processes
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type. Only DDP, FSDP, and DeepSpeed are supported."
            if accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")

        self.device = accelerator.device
        self.batch_size_per_gpu = int(batch_size)
        self.fps = fps
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

    # -- Properties ----------------------------------------------------------

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def model(self):
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

    @property
    def image_token_id(self):
        tid = getattr(self.processor, "image_token_id", None)
        if tid is None:
            token = getattr(self.processor, "image_token", None)
            if token is None:
                raise ValueError("Image token not found in processor")
            tid = self.tokenizer.convert_tokens_to_ids(token)
        return tid

    @property
    def video_token_id(self):
        tid = getattr(self.processor, "video_token_id", None)
        if tid is None:
            token = getattr(self.processor, "video_token", None)
            if token is None:
                raise ValueError("Video token not found in processor")
            tid = self.tokenizer.convert_tokens_to_ids(token)
        return tid

    # -- Tokenization --------------------------------------------------------

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    # -- Request types -------------------------------------------------------

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "loglikelihood not implemented for SGLang"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not implemented")

    # -- Message preparation -------------------------------------------------

    def _prepare_video_kwargs(self):
        kwargs = {
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels,
            "max_frames": self.max_frame_num,
        }
        if self.fps is not None:
            kwargs["fps"] = self.fps
        else:
            kwargs["nframes"] = self.nframes
        return kwargs

    def _prepare_single_message(self, batch_request):
        """Prepare a single request's messages and media for batching."""
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = batch_request.arguments
        chat_messages = ChatMessages(**{"messages": doc_to_messages(self.task_dict[task][split][doc_id])})

        video_kwargs = self._prepare_video_kwargs()
        messages = chat_messages.to_hf_messages(video_kwargs)
        images, videos, audios = chat_messages.extract_media()

        # Append media file paths for MCP tool calling
        if self.tools:
            for i, image in enumerate(images):
                path = os.path.join(self.work_dir, f"{uuid.uuid4()}.jpg")
                image.save(path)
                messages[-1]["content"].append({"type": "text", "text": f"\nImage {i} has image path: {path}"})
            for i, video in enumerate(videos):
                messages[-1]["content"].append({"type": "text", "text": f"\nVideo {i} has video path: {video}"})

        return messages, images, videos

    @staticmethod
    def _extract_gen_params(gen_kwargs):
        """Extract sampling parameters from gen_kwargs with defaults."""
        return {
            "temperature": gen_kwargs.get("temperature", 0),
            "max_new_tokens": gen_kwargs.get("max_new_tokens", 1024),
            "top_p": gen_kwargs.get("top_p", 0.95),
        }

    # -- Main generation -----------------------------------------------------

    def generate_until(self, requests) -> List[GenerationResult]:
        res = []
        pbar = make_progress(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        total_tokens = 0
        total_elapsed_time = 0

        for batch_requests in batched_requests:
            # Prepare messages in parallel
            with ThreadPoolExecutor(max_workers=min(len(batch_requests), self.threads)) as executor:
                prepared = list(executor.map(self._prepare_single_message, batch_requests))

            batched_messages = [msg for msg, _, _ in prepared]
            # Per-request image/video lists — SGLang expects image_data[i]
            # to correspond to texts[i] (one entry per prompt in the batch).
            image_data = [imgs for _, imgs, _ in prepared]
            video_data = [vids for _, _, vids in prepared]
            has_video = any(len(vids) > 0 for vids in video_data)

            # Extract generation parameters
            ctx, doc_to_messages, gen_kwargs, doc_id, task, split = batch_requests[0].arguments
            params = self._extract_gen_params(gen_kwargs)

            # Apply chat template
            texts = self.processor.apply_chat_template(
                batched_messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=self.tools or None,
            )

            # Generate — SGLang Engine handles its own image/video processing
            # internally, so we pass raw PIL images or video paths directly.
            start_time = time.time()
            if has_video:
                outputs = self._generate_video(texts, video_data, params, batched_messages)
            else:
                outputs = self._generate_image(texts, image_data, params, batched_messages)
            elapsed = time.time() - start_time
            total_elapsed_time += elapsed

            # Collect results
            response_text = [o["text"] for o in outputs]
            response_output_tokens = []
            for output in outputs:
                tok = output.get("meta_info", {}).get("completion_tokens", len(output["text"].split()))
                response_output_tokens.append(tok)
                total_tokens += tok

            assert len(response_text) == len(batch_requests)
            res.extend(GenerationResult(text=text, token_counts=TokenCounts(output_tokens=tok)) for text, tok in zip(response_text, response_output_tokens))
            pbar.update(len(batch_requests))

        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        log_metrics(total_gen_tokens=total_tokens, total_elapsed_time=total_elapsed_time, avg_speed=avg_speed)

        if self.mcp_client is not None:
            shutil.rmtree(self.work_dir)

        pbar.close()
        return res

    def _generate_image(self, texts, image_data, params, batched_messages):
        """Generate for image-only inputs.

        Args:
            texts: List of prompt strings (one per request).
            image_data: Per-request image lists — image_data[i] is a list of
                PIL images for texts[i]. SGLang Engine handles its own image
                processing (tokenization, resizing, etc.).
        """
        if self.mcp_client is None:
            return self.client.generate(prompt=texts, image_data=image_data or None, sampling_params=params)

        # MCP path: pre-tokenize for tool-call round-trips
        flat_images = [img for imgs in image_data for img in imgs] or None
        inputs = self.processor(text=texts, images=flat_images, do_resize=False, padding=True, return_tensors="pt")
        input_ids = inputs.pop("input_ids").tolist()
        return self.req_level_generate(input_ids=input_ids, image_data=flat_images, sampling_params=params, batched_messages=batched_messages)

    def _generate_video(self, texts, video_data, params, batched_messages):
        """Generate for video inputs.

        Passes video paths directly to SGLang Engine which handles
        video reading and frame sampling natively.
        """
        if self.mcp_client is None:
            return self.client.generate(prompt=texts, video_data=video_data, sampling_params=params)

        # MCP + video path not yet supported
        eval_logger.warning("MCP + video generation not supported, falling back to text-only")
        return self.client.generate(prompt=texts, video_data=video_data, sampling_params=params)

    # -- Tool calling (MCP) --------------------------------------------------

    def get_mcp_tools(self):
        if self.mcp_client is None:
            return None
        try:
            return self.mcp_client.get_function_list_sync()
        except Exception as e:
            eval_logger.error(f"Failed to retrieve MCP tools: {e}")
            return None

    def _init_tools_sglang(self):
        if self.mcp_client is None:
            return [], None, [], None

        tools = self.get_mcp_tools()
        parser_type = self.get_tool_call_parser_type(self.processor)
        sgl_tools = [Tool.model_validate(schema) for schema in tools]
        parser = FunctionCallParser(sgl_tools, parser_type)
        parser.detector.bot_token = parser.detector.bot_token.strip()
        parser.detector.eot_token = parser.detector.eot_token.strip()
        return tools, parser_type, sgl_tools, parser

    def get_tool_call_parser_type(self, processing_class) -> str:
        items = FunctionCallParser.ToolCallParserEnum.items()
        if "gpt-oss" in getattr(processing_class, "name_or_path", "").lower():
            return "gpt-oss"
        for parser_type, parser_cls in items:
            parser = parser_cls()
            try:
                vocab = processing_class.get_vocab()
            except AttributeError:
                try:
                    vocab = processing_class.tokenizer.get_vocab()
                except AttributeError as e:
                    raise ValueError(f"Cannot get vocab from {processing_class}") from e
            if parser.bot_token.strip() in vocab and (parser.eot_token == "" or parser.eot_token.strip() in vocab):
                return parser_type
        raise ValueError(f"No tool call parser found for {processing_class}")

    async def async_a_request(self, input_id, image, sampling_params, messages):
        if not isinstance(image, list):
            image = [image]
        turn_count = 0
        while True:
            output = await self.client.async_generate(input_ids=input_id, image_data=image, sampling_params=sampling_params)
            content = output["text"]
            content_id = self.processor.tokenizer.encode(content)

            finish_reason = output["meta_info"]["finish_reason"]["type"]
            if self.function_call_parser.has_tool_call(content):
                finish_reason = "tool_calls"

            if finish_reason in ("stop", "length"):
                messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
                return self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    tools=self.tools,
                    skip_special_tokens=True,
                )

            if finish_reason == "tool_calls":
                try:
                    normed_content, tool_calls = self.function_call_parser.parse_non_stream(content)
                except (JSONDecodeError, AttributeError):
                    tool_calls = []

                tool_messages = []
                new_image_data = []
                for tc in tool_calls:
                    try:
                        arguments = json.loads(tc.parameters)
                    except JSONDecodeError:
                        arguments = {}
                    results = await self.mcp_client.run_tool(tc.name, arguments)
                    content_list = []
                    for r in results.content:
                        if _is_mcp_image_content(r):
                            img = Image.open(io.BytesIO(base64.b64decode(r.data)))
                            new_image_data.append(img)
                            content_list.append({"type": "image"})
                        elif _is_mcp_text_content(r):
                            content_list.append({"type": "text", "text": r.text})
                        else:
                            raise ValueError(f"Unsupported MCP result type: {type(r)}")
                    tool_messages.append({"role": "tool", "name": tc.name, "content": content_list})

                original_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                tool_text = self.processor.apply_chat_template(messages + tool_messages, tokenize=False, add_generation_prompt=True)
                tool_text = tool_text.split(original_text)[1]

                inputs = self.processor(text=tool_text, images=new_image_data or None, return_tensors="pt")
                tool_input_ids = inputs.pop("input_ids").flatten().tolist()

                messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
                messages.extend(tool_messages)
                if new_image_data:
                    image.extend(new_image_data)
                input_id = input_id + content_id + tool_input_ids
            else:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
                return self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    tools=self.tools,
                    skip_special_tokens=True,
                )

            turn_count += 1
            if turn_count >= self.max_turn:
                break

        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=self.tools,
            skip_special_tokens=True,
        )

    def req_level_generate(self, input_ids, image_data, sampling_params, batched_messages):
        """Per-request generation with tool calling support."""
        loop = asyncio.get_event_loop()
        text_list = loop.run_until_complete(asyncio.gather(*[self.async_a_request(iid, img, sampling_params, msgs) for iid, img, msgs in zip(input_ids, image_data, batched_messages)]))
        return [{"text": text} for text in text_list]

    def batch_level_generate(self, input_ids, image_data, sampling_params):
        """Batch generation without tool calling."""
        return self.client.generate(input_ids=input_ids, image_data=image_data, sampling_params=sampling_params)
