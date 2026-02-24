import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import List, Union

from dotenv import load_dotenv
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.concurrency_control import (
    decide_next_concurrency,
    extract_text_prefix_from_chat_messages,
    is_rate_limit_error,
    make_prefix_hash,
)
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.usage_metrics import (
    get_running_totals,
    is_budget_exceeded,
    log_usage,
)
from lmms_eval.models.simple.openai import OpenAICompatible as OpenAICompatibleSimple
from lmms_eval.protocol import ChatMessages

VideoReader, _ = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")

load_dotenv(verbose=True)


@register_model("openai")
class OpenAICompatible(OpenAICompatibleSimple):
    is_simple = False

    def generate_until(self, requests) -> List[GenerationResult]:
        if not requests:
            return []

        reordered_requests = list(requests)
        pbar = tqdm(
            total=len(reordered_requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )

        responses: List[Union[GenerationResult, None]] = [None] * len(reordered_requests)
        total_latency = 0.0
        total_tokens = 0
        current_concurrency = min(
            self.num_concurrent,
            self.adaptive_config.max_concurrency,
        )
        dispatch_order = list(range(len(reordered_requests)))
        if self.prefix_aware_queue:
            prefix_hashes = {}
            for idx in dispatch_order:
                req = reordered_requests[idx]
                prefix_text = req.args[0] if isinstance(req.args[0], str) else ""
                if not prefix_text:
                    _, doc_to_messages, _, doc_id, task, split = req.args
                    chat_messages_raw = doc_to_messages(self.task_dict[task][split][doc_id])
                    prefix_text = extract_text_prefix_from_chat_messages(chat_messages_raw, self.prefix_hash_chars)
                prefix_hashes[idx] = make_prefix_hash(prefix_text, self.prefix_hash_chars)
            dispatch_order.sort(key=lambda idx: (prefix_hashes[idx], idx))
        cursor = 0
        failed_requests = 0
        rate_limited_requests = 0
        latencies: List[float] = []
        completed_since_adapt = 0
        in_flight = {}
        max_workers = max(
            1,
            self.adaptive_config.max_concurrency if self.adaptive_concurrency else current_concurrency,
        )

        def process_single_request(local_index: int, payload: dict | None):
            if payload is None:
                return "", local_index, False, False, 0.0, 0, 0, 0
            started_at = time.time()
            rate_limited = False
            last_error_msg = "unknown error"
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(**payload)
                    elapsed = time.time() - started_at
                    response_text = response.choices[0].message.content
                    input_tokens = 0
                    output_tokens = 0
                    reasoning_tokens = 0
                    if hasattr(response, "usage") and response.usage:
                        input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                        if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                            reasoning_tokens = getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0) or 0
                        completion_tokens = output_tokens
                    else:
                        completion_tokens = len(response_text.split())
                        output_tokens = completion_tokens
                    log_usage(
                        model_name=self.model_version,
                        task_name=None,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                        source="model",
                    )
                    return (
                        response_text,
                        local_index,
                        True,
                        rate_limited,
                        elapsed,
                        completion_tokens,
                        input_tokens,
                        reasoning_tokens,
                    )
                except Exception as exc:
                    error_msg = str(exc)
                    last_error_msg = error_msg
                    rate_limited = rate_limited or is_rate_limit_error(error_msg)
                    eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")
                    if attempt == self.max_retries - 1:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                    else:
                        time.sleep(self.retry_backoff_s)

            elapsed = time.time() - started_at
            error_preview = last_error_msg.replace("\n", " ")[:200]
            failure_content = f"[LMMS_EVAL_REQUEST_FAILED after {self.max_retries} retries] {error_preview}"
            return failure_content, local_index, False, rate_limited, elapsed, 0, 0, 0

        def maybe_update_concurrency(force: bool = False) -> None:
            nonlocal current_concurrency
            nonlocal failed_requests
            nonlocal rate_limited_requests
            nonlocal latencies
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
                latencies=latencies,
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
            latencies = []
            completed_since_adapt = 0

        def build_payload_for_index(global_index: int) -> dict:
            req = reordered_requests[global_index]
            _, doc_to_messages, gen_kwargs, doc_id, task, split = req.args

            chat_messages_raw = doc_to_messages(self.task_dict[task][split][doc_id])
            chat_messages: ChatMessages = ChatMessages(**{"messages": chat_messages_raw})
            request_gen_kwargs = dict(gen_kwargs)
            max_new_tokens = min(request_gen_kwargs.get("max_new_tokens", 1024), 4096)
            temperature = request_gen_kwargs.get("temperature", 0)

            if self.video_fps is not None and self.video_fps > 0:
                video_kwargs = {"fps": self.video_fps}
            else:
                video_kwargs = {"nframes": self.max_frames_num}

            payload = {
                "messages": chat_messages.to_openai_messages(video_kwargs=video_kwargs),
                "model": self.model_version,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }

            if "o1" in self.model_version or "o3" in self.model_version or "o4" in self.model_version or "gpt-5" in self.model_version:
                payload.pop("temperature")
                payload.pop("max_tokens")
                payload["response_format"] = {"type": "text"}
                payload["max_completion_tokens"] = 5000

            return payload

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while cursor < len(dispatch_order) or in_flight:
                while cursor < len(dispatch_order) and len(in_flight) < max(1, current_concurrency):
                    request_index = dispatch_order[cursor]
                    payload = build_payload_for_index(request_index)
                    if payload is None:
                        responses[request_index] = GenerationResult(text="", token_counts=TokenCounts())
                        pbar.update(1)
                        cursor += 1
                        continue

                    if is_budget_exceeded():
                        responses[request_index] = GenerationResult(text="[LMMS_EVAL_BUDGET_EXCEEDED]", token_counts=TokenCounts())
                        pbar.update(1)
                        cursor += 1
                        continue

                    assert payload is not None
                    future = executor.submit(process_single_request, request_index, payload)
                    in_flight[future] = request_index
                    cursor += 1

                if not in_flight:
                    break

                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    (
                        response_text,
                        local_index,
                        success,
                        rate_limited,
                        elapsed,
                        completion_tokens,
                        input_tokens,
                        reasoning_tokens,
                    ) = future.result()
                    in_flight.pop(future, None)
                    responses[local_index] = GenerationResult(
                        text=response_text,
                        token_counts=TokenCounts(
                            input_tokens=input_tokens,
                            output_tokens=completion_tokens,
                            reasoning_tokens=reasoning_tokens,
                        ),
                    )
                    total_latency += elapsed
                    total_tokens += completion_tokens
                    latencies.append(elapsed)
                    if not success:
                        failed_requests += 1
                    if rate_limited:
                        rate_limited_requests += 1
                    completed_since_adapt += 1
                    totals = get_running_totals()
                    pbar.set_postfix({"tokens": f"{totals['total_tokens']:,}"}, refresh=False)
                    pbar.update(1)
                    maybe_update_concurrency(force=False)

        maybe_update_concurrency(force=True)

        avg_speed = total_tokens / total_latency if total_latency > 0 else 0
        log_metrics(
            total_elapsed_time=total_latency,
            total_gen_tokens=total_tokens,
            avg_speed=avg_speed,
        )

        pbar.close()
        return [response if response is not None else GenerationResult(text="", token_counts=TokenCounts()) for response in responses]
