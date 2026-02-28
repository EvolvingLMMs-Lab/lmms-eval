import asyncio
import os
import shutil
import tempfile
import time
import uuid
from multiprocessing import cpu_count
from typing import List, Optional, Tuple

from accelerate import Accelerator, DistributedType
from dotenv import load_dotenv
from loguru import logger as eval_logger
from openai import AsyncOpenAI
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.concurrency_control import (
    AdaptiveConcurrencyConfig,
    decide_next_concurrency,
    extract_text_prefix_from_chat_messages,
    is_rate_limit_error,
    make_prefix_hash,
    parse_bool,
)
from lmms_eval.models.model_utils.usage_metrics import (
    get_running_totals,
    is_budget_exceeded,
    log_usage,
)
from lmms_eval.protocol import ChatMessages

VideoReader, _ = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")

load_dotenv(verbose=True)


@register_model("async_openai")
class AsyncOpenAIChat(lmms):
    is_simple = False

    def __init__(
        self,
        model_version: str = "grok-2-latest",
        base_url: str = None,
        api_key: str = None,
        timeout: int = 600,
        retry_backoff_s: Optional[float] = None,
        max_retries: int = 5,
        max_size_in_mb: int = 20,
        mcp_server_path: str = None,
        num_cpus: int = None,
        work_dir: str = None,
        fps: Optional[int] = None,
        nframes: Optional[int] = 64,
        max_frames: Optional[int] = 768,
        max_pixels: Optional[int] = 151200,
        min_pixels: Optional[int] = 28 * 28,
        is_qwen3_vl: bool = False,
        adaptive_concurrency: bool = False,
        adaptive_min_concurrency: int = 1,
        adaptive_max_concurrency: int = 128,
        adaptive_target_latency_s: float = 15.0,
        adaptive_increase_step: float = 0.1,
        adaptive_decrease_factor: float = 0.7,
        adaptive_failure_threshold: float = 0.05,
        prefix_aware_queue: bool = True,
        prefix_hash_chars: int = 256,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.timeout = timeout
        self.retry_backoff_s = max(0.0, float(1.0 if retry_backoff_s is None else retry_backoff_s))
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
        self.max_frames = max_frames
        self.is_qwen3_vl = is_qwen3_vl
        self.adaptive_concurrency = parse_bool(adaptive_concurrency)
        self.adaptive_config = AdaptiveConcurrencyConfig.from_raw(
            min_concurrency=adaptive_min_concurrency,
            max_concurrency=adaptive_max_concurrency,
            target_latency_s=adaptive_target_latency_s,
            increase_step=adaptive_increase_step,
            decrease_factor=adaptive_decrease_factor,
            failure_threshold=adaptive_failure_threshold,
        )
        self.prefix_aware_queue = parse_bool(prefix_aware_queue)
        self.prefix_hash_chars = max(32, int(prefix_hash_chars))
        if system_prompt is not None:
            self.system_prompt = self._resolve_system_prompt(system_prompt)
        else:
            self.system_prompt = None
        if mcp_server_path is not None:
            from lmms_eval.mcp import MCPClient

            self.mcp_client = MCPClient(mcp_server_path)
            os.makedirs(self.work_dir, exist_ok=True)
        else:
            self.mcp_client = None

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
        if self.max_frames is not None:
            video_kwargs["max_frames"] = self.max_frames
        if self.is_qwen3_vl:
            messages = chat_messages.to_qwen3_vl_openai_messages(video_kwargs)
        else:
            messages = chat_messages.to_openai_messages(video_kwargs)
        messages = self._apply_system_prompt(messages, self.system_prompt) if self.system_prompt else messages
        images, videos, audios = chat_messages.extract_media()
        if self.mcp_client is not None:
            for image_idx, image in enumerate(images):
                image_path = os.path.join(self.work_dir, f"{uuid.uuid4()}.jpg")
                image.save(image_path)
                messages[-1]["content"].append(
                    {
                        "type": "text",
                        "text": f"\nImage {image_idx} has image path: {image_path}",
                    }
                )
            for video_idx, video in enumerate(videos):
                messages[-1]["content"].append(
                    {
                        "type": "text",
                        "text": f"\nVideo {video_idx} has video path: {video}",
                    }
                )

        payload = {"messages": messages}
        payload["model"] = self.model_version
        all_response = ""
        total_input_tokens = 0
        total_output_tokens = 0
        total_reasoning_tokens = 0

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
        # Extract usage metrics
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
            if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                reasoning_tokens = getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_reasoning_tokens += reasoning_tokens
        log_usage(
            model_name=self.model_version,
            task_name=task if "task" in locals() else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            source="model",
        )
        # Sometimes asyncio return None, skip this case
        try:
            all_response += last_response
        except Exception as e:
            all_response += f"Error: {str(e)}"

        while response.choices[0].finish_reason == "tool_calls":
            messages.append({"role": "assistant", "content": last_response})
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": response.choices[0].message.tool_calls,
                }
            )
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
            # Extract usage metrics
            input_tokens = 0
            output_tokens = 0
            reasoning_tokens = 0
            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                    reasoning_tokens = getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0) or 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_reasoning_tokens += reasoning_tokens
            log_usage(
                model_name=self.model_version,
                task_name=task if "task" in locals() else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                source="model",
            )
            last_response = response.choices[0].message.content
            try:
                all_response += last_response
            except Exception as e:
                all_response += str(e)
        return all_response, idx, TokenCounts(input_tokens=total_input_tokens, output_tokens=total_output_tokens, reasoning_tokens=total_reasoning_tokens)

    def generate_until(self, requests) -> List[GenerationResult]:
        results = []

        async def run():
            res: List[Tuple[GenerationResult, int]] = []
            pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
            current_concurrency = (
                min(
                    max(1, self.num_cpus),
                    self.adaptive_config.max_concurrency,
                )
                if self.adaptive_concurrency
                else max(1, self.num_cpus)
            )
            dispatch_order = list(range(len(requests)))
            if self.prefix_aware_queue:
                prefix_hashes = {}
                for idx in dispatch_order:
                    request_obj = requests[idx]
                    prefix_text = request_obj.args[0] if isinstance(request_obj.args[0], str) else ""
                    if not prefix_text:
                        _, doc_to_messages, _, doc_id, task, split = request_obj.args
                        chat_messages_raw = doc_to_messages(self.task_dict[task][split][doc_id])
                        prefix_text = extract_text_prefix_from_chat_messages(chat_messages_raw, self.prefix_hash_chars)
                    prefix_hashes[idx] = make_prefix_hash(prefix_text, self.prefix_hash_chars)
                dispatch_order.sort(key=lambda idx: (prefix_hashes[idx], idx))
            cursor = 0

            async def _process(req, idx):
                if is_budget_exceeded():
                    return "[LMMS_EVAL_BUDGET_EXCEEDED]", idx, TokenCounts(), True, False, 0.0
                started_at = time.time()
                rate_limited = False
                last_error_msg = "unknown error"
                for attempt in range(self.max_retries):
                    try:
                        content, original_idx, token_counts = await self.maybe_forward_with_tool(req, idx)
                        elapsed = time.time() - started_at
                        return content, original_idx, token_counts, True, rate_limited, elapsed
                    except Exception as exc:
                        error_msg = str(exc)
                        last_error_msg = error_msg
                        rate_limited = rate_limited or is_rate_limit_error(error_msg)
                        eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed for request {idx} with error: {error_msg}")
                        if attempt == self.max_retries - 1:
                            eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                        else:
                            await asyncio.sleep(self.retry_backoff_s)

                elapsed = time.time() - started_at
                error_preview = last_error_msg.replace("\n", " ")[:200]
                failure_content = f"[LMMS_EVAL_REQUEST_FAILED after {self.max_retries} retries] {error_preview}"
                return failure_content, idx, TokenCounts(), False, rate_limited, elapsed

            failed_requests = 0
            rate_limited_requests = 0
            request_latencies: List[float] = []
            completed_since_adapt = 0
            in_flight: dict[asyncio.Task, int] = {}

            def maybe_update_concurrency(force: bool = False) -> None:
                nonlocal current_concurrency
                nonlocal failed_requests
                nonlocal rate_limited_requests
                nonlocal request_latencies
                nonlocal completed_since_adapt

                if not self.adaptive_concurrency:
                    return

                sample_threshold = max(4, current_concurrency)
                if not force and completed_since_adapt < sample_threshold:
                    return
                if completed_since_adapt <= 0:
                    return

                decision = decide_next_concurrency(
                    current_concurrency=current_concurrency,
                    total_requests=completed_since_adapt,
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
                failed_requests = 0
                rate_limited_requests = 0
                request_latencies = []
                completed_since_adapt = 0

            while cursor < len(dispatch_order) or in_flight:
                while cursor < len(dispatch_order) and len(in_flight) < max(1, current_concurrency):
                    request_index = dispatch_order[cursor]
                    task = asyncio.create_task(_process(requests[request_index], request_index))
                    in_flight[task] = request_index
                    cursor += 1

                if not in_flight:
                    break

                done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    in_flight.pop(task, None)
                    (
                        content,
                        request_idx,
                        token_counts,
                        success,
                        rate_limited,
                        elapsed,
                    ) = task.result()
                    res.append((GenerationResult(text=content, token_counts=token_counts), request_idx))
                    if not success:
                        failed_requests += 1
                    if rate_limited:
                        rate_limited_requests += 1
                    request_latencies.append(elapsed)
                    completed_since_adapt += 1
                    totals = get_running_totals()
                    pbar.set_postfix({"tokens": f"{totals['total_tokens']:,}"}, refresh=False)
                    pbar.update(1)
                    maybe_update_concurrency(force=False)

            maybe_update_concurrency(force=True)

            pbar.close()
            return res

        eval_results = asyncio.run(run())
        eval_results.sort(key=lambda x: x[1])  # Sort by index to restore original
        results = results + [content for content, _ in eval_results]
        if self.mcp_client is not None:
            shutil.rmtree(self.work_dir)
        return results
