import functools
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import List, Optional, Union

from dotenv import load_dotenv
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.concurrency_control import (
    decide_next_concurrency,
    extract_text_prefix_from_chat_messages,
    is_rate_limit_error,
    make_prefix_hash,
)
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.load_video import _probe_video_metadata
from lmms_eval.models.model_utils.usage_metrics import (
    get_running_totals,
    is_budget_exceeded,
    log_usage,
)
from lmms_eval.models.simple.openai import OpenAICompatible as OpenAICompatibleSimple
from lmms_eval.models.simple.openai import _get_max_new_tokens
from lmms_eval.protocol import ChatMessages

load_dotenv(verbose=True)


def _validate_single_choice_n(gen_kwargs: dict) -> None:
    """Reject choice counts that this backend cannot return faithfully."""
    n = gen_kwargs.get("n")
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n != 1):
        raise ValueError("generation parameter n must be the integer 1 because this backend consumes exactly one response choice")


def _optional_positive_int(name: str, value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


# Per process, keyed by URL: the same clip can appear in several requests (several
# questions per video, retries), and a probe may decode the whole file when the
# container header has no frame count.
@functools.lru_cache(maxsize=None)
def _local_video_duration_s(url: str) -> float:
    """Duration of a local video addressed by a file:// URL (as built by ChatMessages)."""
    if not url.startswith("file://"):
        raise ValueError(f"video_target_frames needs the clip duration, which is only read from local videos; got {url[:80]!r}. Use a local path or file:// URL, or unset video_target_frames.")
    try:
        total_frames, frame_rate = _probe_video_metadata(url[len("file://") :], count_frames=True)
    except Exception as exc:
        raise ValueError(f"Cannot read the duration of {url!r} for video_target_frames: {exc}") from exc
    if total_frames <= 0 or not frame_rate:
        raise ValueError(f"Cannot read the duration of {url!r} for video_target_frames")
    return total_frames / frame_rate


@register_model("openai")
class OpenAICompatible(OpenAICompatibleSimple):
    is_simple = False

    def __init__(
        self,
        *args,
        pass_video_url: bool = False,
        enable_thinking_kwarg: object = None,
        video_target_frames: Optional[int] = None,
        video_max_frames: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pass_video_url = bool(pass_video_url)
        self.enable_thinking_kwarg = enable_thinking_kwarg
        self.video_target_frames = _optional_positive_int("video_target_frames", video_target_frames)
        self.video_max_frames = _optional_positive_int("video_max_frames", video_max_frames)
        if (self.video_target_frames is not None or self.video_max_frames is not None) and not self.pass_video_url:
            raise ValueError("video_target_frames and video_max_frames are sent to the server in media_io_kwargs and require pass_video_url=True")

    def _video_media_io_kwargs(self, messages: list) -> dict:
        """Server-side sampling options for the videos in one request.

        Some servers sample by frame count and others by rate (e.g. vLLM's
        MiniMax-M3 and Qwen3-VL loaders ignore num_frames), so a target frame
        count is sent as both num_frames and fps = frames / clip duration.
        """
        video = {"num_frames": int(self.max_frames_num)}
        if self.video_target_frames is not None:
            urls = {part["video_url"]["url"] for message in messages for part in message["content"] if part.get("type") == "video_url"}
            if len(urls) > 1:
                raise ValueError("video_target_frames sets one fps per request, so it supports one video per request")
            if urls:
                duration_s = _local_video_duration_s(urls.pop())
                # Loaders take floor(duration * fps) or int(...) frames, and
                # duration * (N / duration) can round to just below N, which
                # loses a frame (a frame pair on Qwen2-VL). The nudge is far
                # below one frame, so it never raises that count above N.
                fps = self.video_target_frames / duration_s * (1 + 1e-6)
                video = {"num_frames": self.video_target_frames, "fps": fps}
        if self.video_max_frames is not None:
            video["max_frames"] = self.video_max_frames
        return video

    def generate_until(self, requests) -> List[GenerationResult]:
        if not requests:
            return []

        reordered_requests = list(requests)
        for request in reordered_requests:
            _validate_single_choice_n(request.args[2])
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
            max_new_tokens = _get_max_new_tokens(request_gen_kwargs)
            temperature = request_gen_kwargs.get("temperature", 0)

            if self.video_fps is not None and self.video_fps > 0:
                video_kwargs = {"fps": self.video_fps}
            else:
                video_kwargs = {"nframes": self.max_frames_num}

            payload = {
                "messages": chat_messages.to_openai_messages(video_kwargs=video_kwargs, pass_video_url=self.pass_video_url),
                "model": self.model_version,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }
            for parameter in ("top_p", "seed", "presence_penalty", "frequency_penalty", "n"):
                value = request_gen_kwargs.get(parameter)
                if value is not None:
                    payload[parameter] = value
            extra_body = {}
            if self.pass_video_url:
                # Fail fast: sending a default frame count instead would make
                # this request's results silently incomparable with the rest.
                try:
                    video_media_io_kwargs = self._video_media_io_kwargs(payload["messages"])
                except ValueError as exc:
                    raise ValueError(f"{task}/{split} doc {doc_id}: {exc}") from exc
                extra_body["media_io_kwargs"] = {"video": video_media_io_kwargs}
            if self.enable_thinking_kwarg is not None:
                ek = self.enable_thinking_kwarg
                ek_bool = ek.lower() == "true" if isinstance(ek, str) else bool(ek)
                extra_body["chat_template_kwargs"] = {"enable_thinking": ek_bool}
            if extra_body:
                payload["extra_body"] = extra_body

            if "o1" in self.model_version or "o3" in self.model_version or "o4" in self.model_version or "gpt-5" in self.model_version:
                payload.pop("temperature")
                payload.pop("max_tokens")
                payload["response_format"] = {"type": "text"}
                payload["max_completion_tokens"] = max_new_tokens

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
