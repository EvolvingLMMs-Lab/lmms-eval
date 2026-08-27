import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import List, Union

from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.concurrency_control import (
    decide_next_concurrency,
    is_rate_limit_error,
    make_prefix_hash,
)
from lmms_eval.models.model_utils.usage_metrics import (
    get_running_totals,
    is_budget_exceeded,
    log_usage,
)
from lmms_eval.models.simple.gemini import Gemini as GeminiSimple
from lmms_eval.models.simple.gemini import (
    _audio_mime_type,
    _extract_safety_tag,
    _image_to_bytes,
)
from lmms_eval.protocol import ChatMessages

try:
    from google.genai import types
except ImportError:
    types = None


def _chat_messages_to_gemini_contents(chat_messages: ChatMessages, upload_fn, system_parts: list) -> list:
    """Convert ChatMessages protocol to Gemini-native types.Content list.

    System messages are extracted into ``system_parts`` (modified in-place)
    so they can be passed via GenerateContentConfig.system_instruction.

    Args:
        chat_messages: Structured chat messages from the protocol layer.
        upload_fn: Callable to upload a video and return a file reference.
        system_parts: Mutable list; system message text is appended here.
    """
    contents = []
    for message in chat_messages.messages:
        parts = []
        for content in message.content:
            if content.type == "text":
                parts.append(types.Part.from_text(text=content.text))
            elif content.type == "image":
                img = content.url
                if isinstance(img, Image.Image):
                    parts.append(types.Part.from_bytes(data=_image_to_bytes(img), mime_type="image/png"))
                elif isinstance(img, str):
                    parts.append(types.Part.from_bytes(data=_image_to_bytes(Image.open(img).convert("RGB")), mime_type="image/png"))
                elif isinstance(img, dict):
                    if "bytes" in img and img["bytes"] is not None:
                        parts.append(types.Part.from_bytes(data=img["bytes"], mime_type="image/png"))
                    elif "path" in img and img["path"] is not None:
                        parts.append(types.Part.from_bytes(data=_image_to_bytes(Image.open(img["path"]).convert("RGB")), mime_type="image/png"))
            elif content.type == "video":
                video_url = content.url
                if isinstance(video_url, dict):
                    video_url = video_url.get("path") or video_url.get("url")
                if isinstance(video_url, str):
                    file_ref = upload_fn(video_url)
                    parts.append(types.Part.from_uri(file_uri=file_ref.uri, mime_type=file_ref.mime_type))
            elif content.type == "audio":
                audio_url = content.url
                if isinstance(audio_url, str):
                    with open(audio_url, "rb") as f:
                        parts.append(types.Part.from_bytes(data=f.read(), mime_type=_audio_mime_type(audio_url)))
                elif isinstance(audio_url, dict) and "array" in audio_url:
                    import io

                    import soundfile as sf

                    buf = io.BytesIO()
                    sf.write(buf, audio_url["array"], audio_url["sampling_rate"], format="WAV")
                    buf.seek(0)
                    parts.append(types.Part.from_bytes(data=buf.read(), mime_type="audio/wav"))

        if message.role == "system":
            # Gemini has no system role in contents; collect for system_instruction
            for p in parts:
                if hasattr(p, "text"):
                    system_parts.append(p.text)
                else:
                    system_parts.append(p)
            continue

        role = "model" if message.role == "assistant" else "user"
        if parts:
            contents.append(types.Content(role=role, parts=parts))

    return contents


@register_model("gemini")
class Gemini(GeminiSimple):
    is_simple = False

    def generate_until(self, requests) -> List[GenerationResult]:
        if not requests:
            return []

        reordered_requests = list(requests)
        pbar = tqdm(total=len(reordered_requests), disable=(self.rank != 0), desc="Model Responding")

        responses: List[Union[GenerationResult, None]] = [None] * len(reordered_requests)
        total_latency = 0.0
        total_tokens = 0
        current_concurrency = min(self.num_concurrent, self.adaptive_config.max_concurrency)

        dispatch_order = list(range(len(reordered_requests)))
        if self.prefix_aware_queue:
            prefix_hashes = {}
            for idx in dispatch_order:
                req = reordered_requests[idx]
                prefix_text = req.args[0] if isinstance(req.args[0], str) else ""
                if not prefix_text:
                    _, doc_to_messages, _, doc_id, task, split = req.args
                    chat_raw = doc_to_messages(self.task_dict[task][split][doc_id])
                    # Extract first text content for prefix hash
                    for msg in chat_raw if isinstance(chat_raw, list) else []:
                        for c in msg.get("content", []) if isinstance(msg, dict) else []:
                            if isinstance(c, dict) and c.get("type") == "text":
                                prefix_text = c.get("text", "")
                                break
                        if prefix_text:
                            break
                prefix_hashes[idx] = make_prefix_hash(prefix_text, self.prefix_hash_chars)
            dispatch_order.sort(key=lambda idx: (prefix_hashes[idx], idx))

        cursor = 0
        failed_requests = 0
        rate_limited_requests = 0
        latencies: List[float] = []
        completed_since_adapt = 0
        in_flight = {}
        max_workers = max(1, self.adaptive_config.max_concurrency if self.adaptive_concurrency else current_concurrency)

        def process_single_request(local_index: int, contents: list, config, task_name: str):
            if contents is None:
                return "", local_index, False, False, 0.0, 0, 0, 0

            started_at = time.time()
            rate_limited = False
            last_error_msg = "unknown error"
            token_counts_result = (0, 0, 0)

            for attempt in range(self.max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_version,
                        contents=contents,
                        config=config,
                    )

                    elapsed = time.time() - started_at
                    input_tokens = 0
                    output_tokens = 0
                    reasoning_tokens = 0

                    meta = getattr(response, "usage_metadata", None)
                    if meta:
                        input_tokens = getattr(meta, "prompt_token_count", 0) or 0
                        output_tokens = getattr(meta, "candidates_token_count", 0) or 0
                        reasoning_tokens = getattr(meta, "thoughts_token_count", 0) or 0

                    log_usage(
                        model_name=self.model_version,
                        task_name=task_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                        source="model",
                    )

                    try:
                        response_text = response.text
                    except (ValueError, AttributeError):
                        response_text = None

                    # google-genai SDK returns None for blocked/truncated responses.
                    # Try to salvage partial thinking text before falling back to a tag.
                    if response_text is None:
                        try:
                            for candidate in response.candidates:
                                for part in getattr(candidate.content, "parts", []):
                                    text = getattr(part, "text", None)
                                    if text:
                                        response_text = text
                                        break
                                if response_text:
                                    break
                        except Exception:
                            pass

                    if response_text is None:
                        response_text = _extract_safety_tag(response)

                    return response_text, local_index, True, rate_limited, elapsed, output_tokens, input_tokens, reasoning_tokens

                except Exception as exc:
                    error_msg = str(exc)
                    last_error_msg = error_msg

                    if "finish_reason" in error_msg and ("SAFETY" in error_msg or "is 2" in error_msg):
                        elapsed = time.time() - started_at
                        tag = f"[SAFETY_BLOCKED:{error_msg[:100]}]"
                        return tag, local_index, True, False, elapsed, 0, 0, 0

                    rate_limited = rate_limited or is_rate_limit_error(error_msg)
                    eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed: {error_msg}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_backoff_s * (attempt + 1))
                    else:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")

            elapsed = time.time() - started_at
            error_preview = last_error_msg.replace("\n", " ")[:200]
            return f"[LMMS_EVAL_REQUEST_FAILED after {self.max_retries} retries] {error_preview}", local_index, False, rate_limited, elapsed, 0, 0, 0

        def maybe_update_concurrency(force: bool = False):
            nonlocal current_concurrency, failed_requests, rate_limited_requests, latencies, completed_since_adapt

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
                eval_logger.info(f"Adaptive concurrency: {decision.current_concurrency} -> {decision.next_concurrency} " f"(fail={decision.failure_rate:.3f}, rl={decision.rate_limit_rate:.3f}, p95={decision.p95_latency_s:.3f}s)")
            current_concurrency = decision.next_concurrency
            failed_requests = 0
            rate_limited_requests = 0
            latencies = []
            completed_since_adapt = 0

        def build_payload_for_index(global_index: int):
            req = reordered_requests[global_index]
            _, doc_to_messages, gen_kwargs, doc_id, task, split = req.args

            chat_messages_raw = doc_to_messages(self.task_dict[task][split][doc_id])
            chat_messages = ChatMessages(**{"messages": chat_messages_raw})

            request_gen_kwargs = dict(gen_kwargs)

            system_parts = []
            contents = _chat_messages_to_gemini_contents(chat_messages, self._upload_video, system_parts)

            config = self._build_generation_config(request_gen_kwargs)
            if system_parts:
                # Inject system instruction
                system_text = "\n".join(p if isinstance(p, str) else "" for p in system_parts)
                config = types.GenerateContentConfig(
                    max_output_tokens=config.max_output_tokens,
                    temperature=config.temperature,
                    safety_settings=config.safety_settings,
                    thinking_config=config.thinking_config if hasattr(config, "thinking_config") else None,
                    system_instruction=system_text,
                )

            return contents, config, task

        # ---- Dispatch loop ----

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while cursor < len(dispatch_order) or in_flight:
                while cursor < len(dispatch_order) and len(in_flight) < max(1, current_concurrency):
                    request_index = dispatch_order[cursor]

                    if is_budget_exceeded():
                        responses[request_index] = GenerationResult(text="[LMMS_EVAL_BUDGET_EXCEEDED]", token_counts=TokenCounts())
                        pbar.update(1)
                        cursor += 1
                        continue

                    contents, config, task_name = build_payload_for_index(request_index)
                    future = executor.submit(process_single_request, request_index, contents, config, task_name)
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
        pbar.close()

        self._cleanup_files()

        return [r if r is not None else GenerationResult(text="", token_counts=TokenCounts()) for r in responses]
